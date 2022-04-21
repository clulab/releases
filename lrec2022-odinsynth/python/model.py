import config
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from queryparser import QueryParser
from torch import nn
from transformers import BertModel, BertConfig, AdamW
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import BertTokenizerFast
from loss import margin_loss_two_way
from typing import Dict, List
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import calc_reciprocal_ranks, highlighted_indices_tokenization_space, highligh_word_start_tokenization_space, highlighted_start_continuation_indices_tokenization_space

"""
The base model which provides:
    - component initialization (BertTokenizer*, BertModel, Dropout, Linear, Sigmoid)
    - augmentation to the components with special tokens (tree special tokens)
    - base methods used by pytorch-lightning

Each class that inherit from this should define (or have an already defined method) that creates
the batch from a List[Dict] (The dict comes from using Huggingface datasets with defaults)

Example:
        :param tokenizer      -> used to tokenize the input (BertTokenizer, usually)
        :param symbols        -> used by collate_fn to encode the special symbols (tree, sentence_in, etc)
        :param symbol_tensors -> tensor version of symbols. Used to avoid recreating the same tensor
        :param parser         -> used by collate_fn; Should be able to parse strings (for example, □ -> HoleQuery)
        :param batch -> A list of Dict. Each dict contains "text" (due to Huggingface datasets). The value associated with
                        the key "text" is a list with 6 elements:
                            - index 0 (the sentence)        (as a single string; will be splitted with .split(' ') before 
                                                            being fed to the tokenizer)
                            - index 1 (start_highlight)     (index in the sentence; where the sequence of interest starts)
                            - index 2 (end_highlight)       (index in the sentence; where the sequence of interest ends)
                            - index 3 (current_rule)        (string, the current state of the pattern)
                            - index 4 (next_potential_rule) (string, a potential (valid) continuation from current_rule)
                            - index 5 (is_it_correct)       (int, 1 if it's correct (next_potential_rule is the correct
                                                            continuation for current_rule), 0 otherwise)
    def collate_fn(tokenizer, symbols, symbol_tensors, parser: QueryParser, batch: List[Dict]):
        pass

"""
class BaseModel(pl.LightningModule):
    def __init__(self, hparams={}):
        super().__init__()
        self.hparams = hparams
        model_name = self.hparams.get('model_name', 'google/bert_uncased_L-8_H-512_A-8')
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

        if self.hparams.get('add_pos_tags', False):
            self.tokenizer.add_special_tokens({'additional_special_tokens': config.TREE_SPECIAL_TOKENS + config.POS_TAGS})
        else:
            self.tokenizer.add_special_tokens({'additional_special_tokens': config.TREE_SPECIAL_TOKENS})

        self.model = BertModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer)) 

        # If you don't wish to differentiate between the first tree and the second tree, or you use only one tree
        # then set 'use_default_special_tokens' to False and overrwite in the child class the embedding types (if needed)
        if self.hparams.get('use_default_special_tokens', True):
            # Make token_type_embeddings matrix compatible with the number of token types we want to use.
            # NOTE Investigate alternative ways of achieving this? 
            # for tree1, tree2, sentence_out, sentence_in
            # keep the BERT pretrained token embedding for token_out and token_in, but append freshly initialized for tree1 and tree2
            self.model.embeddings.token_type_embeddings = torch.nn.modules.sparse.Embedding(4, self.hparams.get('projection_size', self.model.config.hidden_size))
            torch.nn.init.uniform_(self.model.embeddings.token_type_embeddings.weight, -0.01, 0.01)

        # Append instead of reinitialzing everythingss
        # append_weights = torch.empty(2, 512)
        # torch.nn.init.uniform_(append_weights, -0.01, 0.01)
        # weight = torch.nn.Parameter(torch.cat([append_weights, self.model.embeddings.token_type_embeddings.weight.detach()], dim=0))
        # self.model.embeddings.token_type_embeddings.weight = weight
        
        self.dropout    = nn.Dropout(config.HIDDEN_DROPOUT_PROB)

        # Depending on the model (Pointwise, Poitwise threeway, etc), the projection size can be 1 * hidden_size, 2 * hidden_size etc
        self.projection = nn.Linear(self.hparams.get('projection_size', self.model.config.hidden_size), 1)
        self.sigmoid    = nn.Sigmoid()
        # self.cel        = nn.BCELoss()
        self.cel        = nn.BCEWithLogitsLoss()

        
        if self.hparams.get('use_default_special_tokens', True):
            self.tree1_symbol = 0
            self.tree2_symbol = 1
            self.sentence_out = 2
            self.sentence_in  = 3
            
            # Store the tensors to not recreate them repeteadly
            self.cls_token_tensor    = torch.tensor([self.tokenizer.cls_token_id]).unsqueeze(dim=0)
            self.sep_token_tensor    = torch.tensor([self.tokenizer.sep_token_id]).unsqueeze(dim=0)
            self.tree1_symbol_tensor = torch.tensor([[self.tree1_symbol]])
            self.tree2_symbol_tensor = torch.tensor([[self.tree2_symbol]])
            self.sentence_out_tensor = torch.tensor([[self.sentence_out]])
            self.symbols = {
                'tree1_symbol': self.tree1_symbol,
                'tree2_symbol': self.tree2_symbol,
                'sentence_out': self.sentence_out,
                'sentence_in' : self.sentence_in
            }
            self.symbol_tensors = {
                'cls_token_tensor': self.cls_token_tensor,
                'sep_token_tensor': self.sep_token_tensor,
                'tree1_symbol_tensor': self.tree1_symbol_tensor,
                'tree2_symbol_tensor': self.tree2_symbol_tensor,
                'sentence_out_tensor': self.sentence_out_tensor
            }

        """
            training_regime = 1 -> training on unrolled data
            training_regime = 2 -> training on rolled data
        """
        self.training_regime    = 1

        """
            Scheduler data
        """
        self.num_training_steps = self.hparams['num_training_steps']
        self.use_scheduler      = self.hparams['use_scheduler']
        self.lr_scheduler_name  = ''
        
        self.logging_suffix     = ''

        self.name_to_lr_scheduler = {
            'transformers.get_linear_schedule_with_warmup': get_linear_schedule_with_warmup,
            'transformers.get_cosine_schedule_with_warmup': get_cosine_schedule_with_warmup,
            'torch.optim.lr_scheduler.CyclicLR': torch.optim.lr_scheduler.CyclicLR,
        }

        # Dynamically decide how to differentiate between the highlighted portion
        self.name_to_highlight_function = {
            'highlighted_indices_tokenization_space': highlighted_indices_tokenization_space,
            'highligh_word_start_tokenization_space': highligh_word_start_tokenization_space,
            'highlighted_start_continuation_indices_tokenization_space': highlighted_start_continuation_indices_tokenization_space,
        }

        self.save_hyperparameters()

    def encode(self, batch):
        pass

    """
    Training regime 1

    How we are training: We are predicting if a transition is the correct transition and using cross-entropy loss
    (current_pattern, next_potential_pattern, sentence) -> binary classification
    """
    def training_step_regime1(self, batch, batch_idx):
        logits = self.forward(batch, return_logits=True).squeeze(1)
        gold   = batch['is_it_correct'].to(self.device)
        
        loss = self.cel(logits, gold)
        self.log(f'train_loss', loss, on_step=True, on_epoch=True)

        return loss    

    """
    Training regime 2

    How we are training: We have access to all the possible transition from the current_pattern, and
    we maximize the score of the correct one using a margin loss
    We also want the current_pattern to have a higher score than the wrong ones (maximizing the scores
    along the correct path)
    (current_pattern, next_potential_pattern, sentence) -> score
    """        
    def training_step_regime2(self, batch, batch_idx):
        logits = self.forward(batch, return_logits=True).squeeze(1)
        # gold   = batch['is_it_correct'].to(self.device)
        if 'number_of_transitions' not in batch or 'number_of_sentences' not in batch:
            raise ValueError(f'The current training regime is {self.training_regime}, but the batch does not contain the required keys')
        
        number_of_transitions  = batch['number_of_transitions']
        number_of_sentences = batch['number_of_sentences']
        
        # Split using the number of patterns
        logits_split = logits.split(number_of_transitions)
        scores_per_step =  []
        # Average. Use the fact that every possible transition from a given rule has
        # the same number of sentences (has to)
        for idx, pattern in enumerate(logits_split):
            scores_per_step.append(pattern.reshape(-1, number_of_sentences[idx][0]).mean(1))

        loss = margin_loss_two_way(scores_per_step, config.MARGIN)
        self.log(f'train_loss', loss, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        if self.training_regime == 1:
            return self.training_step_regime1(batch, batch_idx)
        elif self.training_regime == 2:
            return self.training_step_regime2(batch, batch_idx)
        else:
            raise ValueError(f'Training regime should be in {1, 2}. Currently, it is {self.training_regime}')

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch, return_logits=True).squeeze(1)
        scores = self.sigmoid(logits)
        gold   = batch['is_it_correct'].to(self.device)
        pred   = (scores>=0.5).float()#.squeeze(1)
        loss = self.cel(logits, gold)

        if self.training_regime == 1:
            return {'val_loss': loss, 'pred': pred, 'gold': gold}
        elif self.training_regime == 2:
            number_of_transitions = batch['number_of_transitions']
            number_of_sentences = batch['number_of_sentences']
            logits_split = logits.split(number_of_transitions)

            scores_per_step =  []
            # Average. Use the fact that every possible transition from a given rule has
            # the same number of sentences (has to)
            for idx, pattern in enumerate(logits_split):
                scores_per_step.append(pattern.reshape(-1, number_of_sentences[idx][0]).mean(1))

            return {'val_loss': loss, 'pred': pred, 'gold': gold, 'partial_rr': calc_reciprocal_ranks(scores_per_step)}
        else:
            raise ValueError(f'Training regime should be in (1, 2). Currently, it is {self.training_regime}')
    
    def validation_epoch_end(self, outputs: List):
        pred = torch.cat([o['pred'] for o in outputs], axis=0)
        gold = torch.cat([o['gold'] for o in outputs], axis=0)

        f1 = f1_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy())
        p  = precision_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy())
        r  = recall_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy())

        self.log(f'f1', f1_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy()), prog_bar=True)
        self.log(f'p',  precision_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy()), prog_bar=True)
        self.log(f'r',  recall_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy()), prog_bar=True)
        if self.training_regime == 1:
            return {'f1': f1, 'p': p, 'r': r}
        else:
            mrr = torch.cat([o['partial_rr'] for o in outputs]).mean().item()
            self.log('mrr', mrr, prog_bar=True)
            return {'f1': f1, 'p': p, 'r': r, 'mrr': mrr}

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch, return_logits=True).squeeze(1)
        scores = self.sigmoid(logits)
        gold   = batch['is_it_correct'].to(self.device)
        pred   = (scores>=0.5).float()#.squeeze(1)
        loss = self.cel(logits, gold)

        if self.training_regime == 1:
            return {'val_loss': loss, 'pred': pred, 'gold': gold}
        elif self.training_regime == 2:
            number_of_transitions  = batch['number_of_transitions']
            number_of_sentences = batch['number_of_sentences']
            logits_split = logits.split(number_of_transitions)

            scores_per_step =  []
            # Average. Use the fact that every possible transition from a given rule has
            # the same number of sentences (has to)
            for idx, pattern in enumerate(logits_split):
                scores_per_step.append(pattern.reshape(-1, number_of_sentences[idx][0]).mean(1))

            return {'val_loss': loss, 'pred': pred, 'gold': gold, 'partial_rr': calc_reciprocal_ranks(scores_per_step)}
        else:
            raise ValueError(f'Training regime should be in {1, 2}. Currently, it is {self.training_regime}')

    def test_epoch_end(self, outputs):
        pred = torch.cat([o['pred'] for o in outputs], axis=0)
        gold = torch.cat([o['gold'] for o in outputs], axis=0)

        f1 = f1_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy())
        p  = precision_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy())
        r  = recall_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy())

        self.log(f'f1', f1_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy()), prog_bar=True)
        self.log(f'p',  precision_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy()), prog_bar=True)
        self.log(f'r',  recall_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy()), prog_bar=True)

        if self.training_regime == 1:
            return {'f1': f1, 'p': p, 'r': r}
        else:
            mrr = torch.cat([o['partial_rr'] for o in outputs]).mean().item()
            self.log('mrr', mrr, prog_bar=True)
            return {'f1': f1, 'p': p, 'r': r, 'mrr': mrr}

    def configure_optimizers(self):
        named_params = list(self.named_parameters())
        params = [
            {
                'params': [p for n,p in named_params if not any(nd in n for nd in config.NO_DECAY)],
                'weight_decay': self.hparams['weight_decay'],
            },
            {
                'params': [p for n,p in named_params if any(nd in n for nd in config.NO_DECAY)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = AdamW(params, lr=self.hparams['learning_rate'])
        
        if 'use_scheduler' in self.hparams and self.hparams['use_scheduler']:     
            return (
                [optimizer], 
                [{
                    # 'scheduler': get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams['num_training_steps']//16, num_training_steps=self.hparams['num_training_steps'],),
                    # 'scheduler': get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams['num_training_steps']//2, num_training_steps=self.hparams['num_training_steps'],),
                    # 'scheduler': torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.hparams['learning_rate']/5, max_lr = self.hparams['learning_rate'], mode='triangular2', cycle_momentum=False, step_size_up=self.num_training_steps//2),
                    'scheduler': self.name_to_lr_scheduler[self.hparams['scheduler_name']](optimizer, base_lr=self.hparams['learning_rate']/5, max_lr = self.hparams['learning_rate'], mode='triangular2', cycle_momentum=False, step_size_up=self.num_training_steps//2),
                    'interval': 'step',
                    'frequency': 1,
                    'strict': True,
                    'reduce_on_plateau': False, 
                }]
            )
        else:
            return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model-name', type=str, help="Path to the BERT model to use")

        parser.add_argument('--add-pos-tags', type=bool, required=False, help='Whether to add the part-of-speech tags as special tokens or not.')
        
        parser.add_argument('--use-scheduler', type=bool, required=False, help="Whether to use a learning rate scheduler or not")
        parser.add_argument('--scheduler-name', type=str, help='Which scheduler to use (e.g. "get_cosine_schedule_with_warmup", "get_linear_schedule_with_warmup", or "torch.optim.lr_scheduler.CyclicLR")')

        parser.add_argument('-nts', '--num-training-steps', type=int, required=False, 
            help='For schedulers. This is an alternative way of setting this parameter and it is recommended only when you exactly know the value. Different datasets have different length. The preffered way is to use a scaling factor to scale up or down the (len(train)/batch_size) value')
        parser.add_argument('--num-warmup-steps', type=int, required=False, 
            help='For schedulers. This is an alternative way of setting this parameter and it is recommended only when you exactly know the value. Different datasets have different length. The preffered way is to use a scaling factor to scale up or down the (len(train)/batch_size) value. Also, not all have this parameter')
            
        parser.add_argument('--learning-rate', type=float, required=False,
            help="What learning rate to use.")            
        parser.add_argument('--base-learning-rate', type=float, required=False,
            help="The base learning rate. Used only with schedulers. For example, CyclicLR needs to know a base_lr (this parameter), and a max_lr (set using --learning-rate)")            
        parser.add_argument('--weight-decay', type=float, required=False,
            help="What value to use for the weight decay.")

        return parser

    """
        Also see the base class documentation for an explanation of the rest of the parameters. The parameters 
        that differ are:
        
        :param batch      - a list of lists of lists of dict. The final list of dict is the same as in collate_fn
                            function, with the documentation at the top of this class. The other two lists are needed
                            because we differentiate between:
                                - a given rule
                                - a given next_rule (such that we can average)
                            
                            batch[0]       -> all the possible transitions, together with all the sentences 
                                              for a particular current rule (e.g., current_rule = □,
                                              next_possible_transitions are [□?, □*, □+, □ □, [□]] (5 in total)
                                              and there are two sentences (sentence_1 and sentence_2)
                                              So batch[0] will be (conceptually):
                                              [
                                                  [{'text': ['sentence_1', '□?']}, {'text': ['sentence_2', '□?']}],
                                                  [{'text': ['sentence_1', '□*']}, {'text': ['sentence_2', '□*']}],
                                                  [{'text': ['sentence_1', '□+']}, {'text': ['sentence_2', '□+']}],
                                                  [{'text': ['sentence_1', '□ □']}, {'text': ['sentence_2', '□ □']}],
                                                  [{'text': ['sentence_1', '[□]']}, {'text': ['sentence_2', '[□]']}],
                                              ]
                            batch[0][0]    -> a particular transition, same for each sentence in this list
                            batch[0][0][0] -> a particular transition with a particular sentence
                                              Therefore, an average of the scores obtain for 
                                              batch[0][0] gives the average score for that 
                                              transition to be the correct transition
        
        :param collate_fn - the collate_fn function specific to each model (e.g. PointwiseBM.collate_fn)

        NOTE that the way this method is called is a bit convoluted
    """
    @staticmethod
    def collate_fn_rolled(tokenizer, symbols, symbol_tensors, parser: QueryParser, batch: List[List[List[Dict]]], collate_fn):
        flat = [z for x in batch for y in x for z in y]
        # number_of_pattern[0] tells how many elements from flat are corresponding to the same pattern
        number_of_transitions   = [sum([len(y) for y in x]) for x in batch]
        number_of_sentences  = [[len(y) for y in x] for x in batch]
        data = collate_fn(tokenizer, symbols, symbol_tensors, parser, flat)
        data['number_of_transitions'] = number_of_transitions
        data['number_of_sentences'] = number_of_sentences
        return data


"""
Receives as input:
    (current_state, potential_next_state, sentence)
    Should return a high score if the potential_next_state is indeed correct
    and a low score otherwise using a FFN on the concatenation
    It gets the score by 
"""
class PointwiseBM(BaseModel):
    def __init__(self, hparams={}):
        super().__init__(hparams)

    def encode(self, batch):
        input_ids       = batch['input_ids'].to(self.device)
        attention_masks = batch['attention_masks'].to(self.device)
        token_type_ids  = batch['token_type_ids'].to(self.device)

        enc = self.model(input_ids = input_ids, attention_mask = attention_masks, token_type_ids = token_type_ids)[1]

        return enc

    def forward(self, batch, return_logits=False, return_encoding=False):
        input_ids       = batch['input_ids'].to(self.device)
        attention_masks = batch['attention_masks'].to(self.device)
        token_type_ids  = batch['token_type_ids'].to(self.device)

        encoded   = self.model(input_ids = input_ids, attention_mask = attention_masks, token_type_ids = token_type_ids)[1]
        projected = self.projection(self.dropout(encoded))

        output = self.sigmoid(projected)

        # Return logits is useful when computing the loss
        if return_logits and return_encoding:
            return (projected, encoded)
        elif return_logits:
            return projected
        elif return_encoding:
            return encoded
        else:
            return output

    """
        See BaseModel for a description of how this should behave
    """
    @staticmethod
    def collate_fn(tokenizer, symbols, symbol_tensors, parser: QueryParser, batch: List[Dict]):
        tree1_symbol = symbols['tree1_symbol']
        tree2_symbol = symbols['tree2_symbol']
        sentence_out = symbols['sentence_out']
        sentence_in  = symbols['sentence_in']
        
        sep_token_tensor = symbol_tensors['sep_token_tensor']
        cls_token_tensor = symbol_tensors['cls_token_tensor']
        tree1_symbol_tensor = symbol_tensors['tree1_symbol_tensor']
        tree2_symbol_tensor = symbol_tensors['tree2_symbol_tensor']
        sentence_out_tensor = symbol_tensors['sentence_out_tensor']

        tokens = []
        is_it_correct = []

        for bdict in batch:
            b = bdict['text']
            sentence            = b[0]
            start_highlight     = int(b[1])
            end_highlight       = int(b[2])
            current_rule        = parser.parse(b[3])
            next_potential_rule = parser.parse(b[4])
            is_it_correct.append(int(b[5]))


            crt = tokenizer(current_rule.get_tokens(), truncation=False, padding='do_not_pad', return_tensors='pt', is_split_into_words=True, add_special_tokens=False)  # current_rule_tree
            crt['token_type_ids'][:]  = tree1_symbol
            nprt = tokenizer(next_potential_rule.get_tokens(), truncation=False, padding='do_not_pad', return_tensors='pt', is_split_into_words=True, add_special_tokens=False) # next_potential_rule_tree
            nprt['token_type_ids'][:] = tree2_symbol

            # Note that this ignores the '\n' characters
            sentence_tokenized = tokenizer(sentence.split(' '), truncation=False, padding='do_not_pad', return_tensors='pt', is_split_into_words=True, add_special_tokens=False, return_offsets_mapping=True)

            # Set everything to out
            sentence_tokenized['token_type_ids'][0][:]     = sentence_out

            # Negative examples should not have a spec, but if they do
            # they will have start_highlight bigger than
            # end_highlight, meaning that there is nothing highlighted
            if start_highlight <= end_highlight and end_highlight >= 0:
                # Override the tokens that are part of the highlight
                index = highlighted_indices_tokenization_space(list(range(start_highlight, end_highlight)), sentence_tokenized['offset_mapping'])
                sentence_tokenized['token_type_ids'][0][index] = sentence_in
            
            input_ids = torch.cat([cls_token_tensor, crt['input_ids'], sep_token_tensor, nprt['input_ids'], sep_token_tensor, sentence_tokenized['input_ids'], sep_token_tensor], dim=1)
            tokens.append({
                'input_ids': input_ids,
                'attention_masks': torch.ones_like(input_ids),
                'token_type_ids': torch.cat([tree1_symbol_tensor, crt['token_type_ids'], tree1_symbol_tensor, nprt['token_type_ids'], tree2_symbol_tensor, sentence_tokenized['token_type_ids'], sentence_out_tensor], dim=1),
            })


        # Start padding the tokens
        max_input_length = min(max([x['input_ids'].shape[1] for x in tokens]), 512)
        for t1 in tokens:
            current_length  = t1['input_ids'].shape[1]
            if current_length < max_input_length:
                pad = torch.tensor([tokenizer.pad_token_id] * (max_input_length - current_length)).unsqueeze(dim=0)
                for key in t1.keys():
                    t1[key] = torch.cat([t1[key], pad], dim=1)

        return {
            'input_ids':       torch.cat([x['input_ids'][:, :512]       for x in tokens], dim=0),
            'attention_masks': torch.cat([x['attention_masks'][:, :512] for x in tokens], dim=0),
            'token_type_ids':  torch.cat([x['token_type_ids'][:, :512]  for x in tokens], dim=0),
            'is_it_correct':   torch.tensor(is_it_correct).float()
            }


"""
Receives as input:
    (current_state, potential_next_state, sentence)
    Should return a high score if the potential_next_state is indeed correct
    And a low score otherwise
    It uses a FFN to map (current_state, sentence) and (potential_next_state, sentence)
    to two vectors which are then subtracted (like in Learning to Rank using Gradient Descent)
"""
class PointwiseWithSubtractionBM(BaseModel):
    def __init__(self, hparams={}):
        super().__init__(hparams)

    def encode(self, batch):
        input_ids_1       = batch['batch_1']['input_ids'].to(self.device)
        attention_masks_1 = batch['batch_1']['attention_masks'].to(self.device)
        token_type_ids_1  = batch['batch_1']['token_type_ids'].to(self.device)
        input_ids_2       = batch['batch_2']['input_ids'].to(self.device)
        attention_masks_2 = batch['batch_2']['attention_masks'].to(self.device)
        token_type_ids_2  = batch['batch_2']['token_type_ids'].to(self.device)

        enc_1 = self.dropout(self.model(input_ids = input_ids_1, attention_mask = attention_masks_1, token_type_ids = token_type_ids_1)[1])
        enc_2 = self.dropout(self.model(input_ids = input_ids_2, attention_mask = attention_masks_2, token_type_ids = token_type_ids_2)[1])

        return enc_1 - enc_2

    def forward(self, batch, return_logits=False):
        input_ids_1       = batch['batch_1']['input_ids'].to(self.device)
        attention_masks_1 = batch['batch_1']['attention_masks'].to(self.device)
        token_type_ids_1  = batch['batch_1']['token_type_ids'].to(self.device)
        input_ids_2       = batch['batch_2']['input_ids'].to(self.device)
        attention_masks_2 = batch['batch_2']['attention_masks'].to(self.device)
        token_type_ids_2  = batch['batch_2']['token_type_ids'].to(self.device)
        # print(input_ids_1.sum())


        enc_1 = self.dropout(self.model(input_ids = input_ids_1, attention_mask = attention_masks_1, token_type_ids = token_type_ids_1)[1])
        enc_2 = self.dropout(self.model(input_ids = input_ids_2, attention_mask = attention_masks_2, token_type_ids = token_type_ids_2)[1])
        enc = enc_1 - enc_2
        enc = self.projection(enc)

        output = self.sigmoid(enc)

        # Return logits is useful when computing the loss
        if return_logits:
            return enc
        else:
            return output

    """
        See BaseModel for a description of how this should behave
    """
    @staticmethod
    def collate_fn(tokenizer, symbols, symbol_tensors, parser: QueryParser, batch: List[Dict]):
        tree1_symbol = symbols['tree1_symbol']
        tree2_symbol = symbols['tree2_symbol']
        sentence_out = symbols['sentence_out']
        sentence_in  = symbols['sentence_in']
        
        sep_token_tensor = symbol_tensors['sep_token_tensor']
        cls_token_tensor = symbol_tensors['cls_token_tensor']
        tree1_symbol_tensor = symbol_tensors['tree1_symbol_tensor']
        tree2_symbol_tensor = symbol_tensors['tree2_symbol_tensor']
        sentence_out_tensor = symbol_tensors['sentence_out_tensor']

        tokens_1 = []
        tokens_2 = []
        is_it_correct = []

        for bdict in batch:
            b = bdict['text']
            sentence            = b[0]
            start_highlight     = int(b[1])
            end_highlight       = int(b[2])
            current_rule        = parser.parse(b[3])
            next_potential_rule = parser.parse(b[4])
            is_it_correct.append(int(b[5]))

            crt = tokenizer(current_rule.get_tokens(), truncation=False, padding='do_not_pad', return_tensors='pt', is_split_into_words=True, add_special_tokens=False)
            crt['token_type_ids'][:]  = tree1_symbol
            nprt = tokenizer(next_potential_rule.get_tokens(), truncation=False, padding='do_not_pad', return_tensors='pt', is_split_into_words=True, add_special_tokens=False)
            nprt['token_type_ids'][:] = tree2_symbol

            sentence_tokenized = tokenizer(sentence.split(' '), truncation=False, padding='do_not_pad', return_tensors='pt', is_split_into_words=True, add_special_tokens=False, return_offsets_mapping=True)

            # Set everything to out
            sentence_tokenized['token_type_ids'][0][:]     = sentence_out

            # Negative examples should not have a spec, but if they do
            # they will have start_highlight bigger than
            # end_highlight, meaning that there is nothing highlighted
            if start_highlight <= end_highlight and end_highlight >= 0:
                # Override the tokens that are part of the highlight
                index = highlighted_indices_tokenization_space(list(range(start_highlight, end_highlight)), sentence_tokenized['offset_mapping'])
                sentence_tokenized['token_type_ids'][0][index] = sentence_in         

            
            input_ids_1 = torch.cat([cls_token_tensor, crt['input_ids'], sep_token_tensor, sentence_tokenized['input_ids'], sep_token_tensor], dim=1)
            input_ids_2 = torch.cat([cls_token_tensor, nprt['input_ids'], sep_token_tensor, sentence_tokenized['input_ids'], sep_token_tensor], dim=1)
            tokens_1.append({
                'input_ids': input_ids_1,
                'attention_masks': torch.ones_like(input_ids_1),
                'token_type_ids': torch.cat([tree1_symbol_tensor, crt['token_type_ids'], tree1_symbol_tensor, sentence_tokenized['token_type_ids'], sentence_out_tensor], dim=1),
            })
            tokens_2.append({
                'input_ids': input_ids_2,
                'attention_masks': torch.ones_like(input_ids_2),
                'token_type_ids': torch.cat([tree2_symbol_tensor, nprt['token_type_ids'], tree2_symbol_tensor, sentence_tokenized['token_type_ids'], sentence_out_tensor], dim=1),
            })


        # Start padding the tokens
        max_input_length_1 = min(max([x['input_ids'].shape[1] for x in tokens_1]), 512)
        for t1 in tokens_1:
            current_length  = t1['input_ids'].shape[1]
            if current_length < max_input_length_1:
                pad = torch.tensor([tokenizer.pad_token_id] * (max_input_length_1 - current_length)).unsqueeze(dim=0)
                for key in t1.keys():
                    t1[key] = torch.cat([t1[key], pad], dim=1)

        max_input_length_2 = min(max([x['input_ids'].shape[1] for x in tokens_2]), 512)
        for t2 in tokens_2:
            current_length  = t2['input_ids'].shape[1]
            if current_length < max_input_length_2:
                pad = torch.tensor([tokenizer.pad_token_id] * (max_input_length_2 - current_length)).unsqueeze(dim=0)
                for key in t2.keys():
                    t2[key] = torch.cat([t2[key], pad], dim=1)

        return {
                'batch_1': {
                    'input_ids':       torch.cat([x['input_ids'][:, :512]       for x in tokens_1], dim=0),
                    'attention_masks': torch.cat([x['attention_masks'][:, :512] for x in tokens_1], dim=0),
                    'token_type_ids':  torch.cat([x['token_type_ids'][:, :512]  for x in tokens_1], dim=0),
                },
                'batch_2': {
                    'input_ids':       torch.cat([x['input_ids'][:, :512]       for x in tokens_2], dim=0),
                    'attention_masks': torch.cat([x['attention_masks'][:, :512] for x in tokens_2], dim=0),
                    'token_type_ids':  torch.cat([x['token_type_ids'][:, :512]  for x in tokens_2], dim=0),
                },
                'is_it_correct': torch.tensor(is_it_correct).float()
            }

