from model import PointwiseBM
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
from collections import namedtuple
from dataclasses import asdict, dataclass, make_dataclass

# EncoderBatchType = make_dataclass('EncoderBatchType', [('sentence', str), ('sentence', str), ('sentence', str), ])
@dataclass
class EncoderBatchType:
    sentence           : str
    start_highlight    : int
    end_highlight      : int
    current_rule       : str
    next_potential_rule: str

# PredictorBatchType = make_dataclass('PredictorBatchType', [('is_it_correct', int)], bases=(EncoderBatchType,))
@dataclass
class PredictorBatchType(EncoderBatchType):
    is_it_correct: int

EncodeAggregateBatch = List[EncoderBatchType]

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
class BaseEncoderModel(nn.Module):
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
            Scheduler data
        """
        self.num_training_steps = self.hparams['num_training_steps']
        self.use_scheduler      = self.hparams['use_scheduler']
        self.lr_scheduler_name  = ''
        
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


    def forward(self, batch, device):
        input_ids       = batch['input_ids'].to(device)
        attention_masks = batch['attention_masks'].to(device)
        token_type_ids  = batch['token_type_ids'].to(device)

        encoded   = self.model(input_ids = input_ids, attention_mask = attention_masks, token_type_ids = token_type_ids)[1]

        return encoded


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
        # is_it_correct = []
        
        for bdict in batch:
            sentence            = bdict.sentence
            start_highlight     = bdict.start_highlight
            end_highlight       = bdict.end_highlight
            current_rule        = parser.parse(bdict.current_rule)
            next_potential_rule = parser.parse(bdict.next_potential_rule)


            crt = tokenizer(current_rule.get_tokens(), truncation=False, padding='do_not_pad', return_tensors='pt', is_split_into_words=True, add_special_tokens=False)  # current_rule_tree
            crt['token_type_ids'][:]  = tree1_symbol
            nprt = tokenizer(next_potential_rule.get_tokens(), truncation=False, padding='do_not_pad', return_tensors='pt', is_split_into_words=True, add_special_tokens=False) # next_potential_rule_tree
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
            'token_type_ids':  torch.cat([x['token_type_ids'][:, :512]  for x in tokens], dim=0)
            }

    @staticmethod
    def load_from_pointwisebm(path):
        loaded_model = PointwiseBM.load_from_checkpoint(path)
        return_model = BaseEncoderModel(loaded_model.hparams)
        
        return_model.encoder.model.load_state_dict(loaded_model.model.state_dict())

        return return_model


"""
Receives as input:
    (current_state, potential_next_state, sentence)
    Should return a high score if the potential_next_state is indeed correct
    and a low score otherwise using a FFN on the concatenation
    It gets the score by applying the underlying encoder model, then
    using a linear layer to predict the score
"""
class PointwiseHeadPredictor(pl.LightningModule):
    def __init__(self, hparams={}):
        super().__init__()
        self.encoder    = BaseEncoderModel(hparams)
        self.dropout    = nn.Dropout(config.HIDDEN_DROPOUT_PROB)

        # Depending on the model (Pointwise, Poitwise threeway, etc), the projection size can be 1 * hidden_size, 2 * hidden_size etc
        self.projection = nn.Linear(self.hparams.get('projection_size', self.encoder.model.config.hidden_size), 1)
        self.sigmoid    = nn.Sigmoid()
        # self.cel        = nn.BCELoss()
        self.cel        = nn.BCEWithLogitsLoss()

    def forward(self, batch, return_logits=False, return_encoding=False):
        encoded   = self.encoder(batch, device=self.device)
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


    def training_step(self, batch, batch_idx):
        logits = self.forward(batch, return_logits=True).squeeze(1)
        gold   = batch['is_it_correct'].to(self.device)
        
        loss = self.cel(logits, gold)
        self.log(f'train_loss', loss, on_step=True, on_epoch=True)

        return loss

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

    """
        See BaseModel for a description of how this should behave
    """
    @staticmethod
    def collate_fn(tokenizer, symbols, symbol_tensors, parser: QueryParser, batch: List[PredictorBatchType], collate_fn):
        collated_batch = collate_fn(tokenizer, symbols, symbol_tensors, parser, batch)
        is_it_correct  = [bdict.is_it_correct for bdict in batch] # Note that we are traversing the batch twice.

        collated_batch['is_it_correct'] = torch.tensor(is_it_correct).float()

        return collated_batch

    @staticmethod
    def load_from_pointwisebm(path):
        loaded_model = PointwiseBM.load_from_checkpoint(path)
        return_model = PointwiseHeadPredictor(loaded_model.hparams)
        
        return_model.encoder.model.load_state_dict(loaded_model.model.state_dict())
        return_model.projection.load_state_dict(loaded_model.projection.state_dict())

        return return_model

class BaseAggregatorModel(nn.Module):
    def __init__(self, hparams = {}):
        super().__init__()
        self.hparams = hparams
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.hparams.get('aggregator_input_size', 512), 
            nhead   = self.hparams.get('aggregator_number_of_attention_heads', 4),
            **self.hparams.get('aggregator_additional_parameters', {})
        )
        self.aggregator = nn.TransformerEncoder(encoder_layer, self.hparams.get('aggregator_layers', 2))

        self.cls_vector = nn.Parameter(torch.zeros(1, self.hparams.get('aggregator_input_size', 512)))
        nn.init.xavier_normal_(self.cls_vector)

        self.hidden_size = self.hparams.get('aggregator_input_size', 512)

    """

        Aggregate everything by first appending a [CLS] token,
        then applying a transformer method, then returning the
        [CLS] embedding for each element in the batch

        The current implementation batches everything
        For an un-batched implementation, that is, one where we
        apply the transformer over each element of the list, see below:

        ```
            batch_with_cls = [torch.cat([self.cls_vector, x]) for x in batch]
            # Unsqueeze to obtain a (1, number_of_sentences + 1, hidden_size) shaped tensor (number_of_sentences + 1 is because we appended [CLS])
            # Transpose because in the current pytorch version we cannot specify that we have a batch_first tensor, so we need to feed a tensor of
            # shape (number_of_sentences + 1, 1, hidden_size)
            # Select the embedding for the [CLS] token (which will have shape (1, hidden_size))
            aggregated     = [self.aggregator(x.unsqueeze(0).transpose(0, 1))[0] for x in batch_with_cls]
            return torch.cat(aggregated, dim=0)
        ```
        Note that the results have to coincide (up to some rounding error)

    """
    def forward(self, batch):
        batch_with_cls = [torch.cat([self.cls_vector, x]) for x in batch]

        # Now batching
        max_len = max([x.shape[0] for x in batch_with_cls])

        # Pad the batch (which is a list of tensors) and concatenate it
        # Will obtain a tensor of shape:
        # (max_len, batch_size, hidden_size)
        batch_with_cls_padded = [torch.cat([x, torch.zeros(max_len-x.shape[0], x.shape[1])], dim=0) for x in batch_with_cls]
        batch_with_cls_padded = torch.cat([x.unsqueeze(1) for x in batch_with_cls_padded], dim=1) # Concatenate on dim 1, the batching dimension

        # Prepare the mask
        # The mask will tell which entry in the tensor of shape:(max_len, batch_size, hidden_size) is a pad
        # It has to be of shape:
        # (batch_size, max_len)
        # This is because this is the shape the torch.nn.MultiheadAttention which is used in the TransformerEncoderLayer
        # expects
        mask = torch.ones(len(batch_with_cls), max_len)
        for i, x in enumerate(batch_with_cls):
            mask[i][:x.shape[0]] = 0
        mask = mask.bool() # This is required. Otherwise the result will be different than when applied over each element in the batch individually

        result = self.aggregator(batch_with_cls_padded, src_key_padding_mask=mask)
        output = result.transpose(0, 1)[:, 0]

        return output



"""
    Compared to the BaseModel, this class will have a different notion of "batch" (different looking elements)
    The BaseModel works over pairs like:
            (sentence, start_highlight, end_highlight, current_rule, next_potential_rule, is_it_correct)
    which means that we have no notion of multiple sentences
    Here, we do not score individual sentences, but encode them and score at the end
    As such, a batch contains:
            List of (sentence, start_highlight, end_highlight, current_rule, next_potential_rule, is_it_correct)
"""
class BaseEncodeAggregateModel(pl.LightningModule):

    import collections
    EncoderBatchType = collections.namedtuple('EncoderBatchType', ['sentence', 'start_highlight', 'end_highlight', 'current_rule', 'next_potential_rule', 'is_it_correct']) 
    EncodeAggregateBatch = List[EncoderBatchType]
    
    def __init__(self, encoder: BaseEncoderModel, aggregator: BaseAggregatorModel, hparams = {}):
        super().__init__()
        self.hparams = hparams
        self.encoder = encoder.to(self.device)
        self.aggregator = aggregator.to(self.device)

        self.dropout    = nn.Dropout(config.HIDDEN_DROPOUT_PROB)
        self.projection = nn.Linear(self.aggregator.hidden_size, 1)

        self.sigmoid    = nn.Sigmoid()

        self.cel        = nn.BCEWithLogitsLoss()

    """
    The batch and how it should look like is explained in forward
    """
    def encode(self, batch_for_encoder):
        return self.encoder(batch_for_encoder, device=self.device)
        
    """
    The batch and how it should look like is explained in forward
    """
    def aggregate(self, batch_for_aggregator):
        return self.aggregator(batch_for_aggregator)

    def forward(self, batch, return_logits=False):
        # Batch will be a dictionary, at leaset with the following keys:
        #       'input_ids'           - tensor of shape (total_number_of_sentences, encoder_input_shape)
        #       'attention_masks'     - tensor of shape (total_number_of_sentences, encoder_input_shape)
        #       'token_type_ids'      - tensor of shape (total_number_of_sentences, encoder_input_shape)
        #       'is_it_correct'       - tensor of shape (total_number_of_sentences, encoder_input_shape)
        #       'number_of_sentences' - list with the number of sentences of each problem_specification
        # Note that the total number of sentences is sum(number_of_sentences)
        encoded = self.encoder(batch, device=self.device)

        # List of tensors
        # [
        #   tensor1, of shape (number_of_sentences_of_problem_specification_1, encoder_encoding_size)
        #   tensor2, of shape (number_of_sentences_of_problem_specification_2, encoder_encoding_size)
        #   ...
        # ]
        encoded_per_problem_specification    = list(encoded.split(batch['number_of_sentences']))

        # tensor of shape (number_of_problem_specifications, aggregator_encoding_size)
        aggregated_per_problem_specification = self.aggregator(encoded_per_problem_specification)

        # tensor of shape (number_of_problem_specifications, 1)
        # In other words, the value in the tensor tells how good is
        # the (current_node, next_potential_node) type of transition, considering
        # the given problem specification
        result = self.projection(self.dropout(aggregated_per_problem_specification))
        
        if return_logits:
            return result
        else:
            return self.sigmoid(result)

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch, return_logits=True).squeeze(1) # Squeeze call because our projection layer has output size 1
        gold = batch['is_it_correct'].to(self.device)

        loss = self.cel(logits, gold)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch, return_logits=True).squeeze(1) # Squeeze call because our projection layer has output size 1
        gold = batch['is_it_correct'].to(self.device)

        scores = self.sigmoid(logits)

        pred   = (scores>=0.5).float()#.squeeze(1)

        loss = self.cel(logits, gold)

        return {'val_loss': loss, 'pred': pred, 'gold': gold}

    def validation_epoch_end(self, outputs: List):
        pred = torch.cat([o['pred'] for o in outputs], axis=0)
        gold = torch.cat([o['gold'] for o in outputs], axis=0)
        
        f1 = f1_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy())
        p  = precision_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy())
        r  = recall_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy())

        self.log(f'f1{self.logging_suffix}', f1_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy()), prog_bar=True)
        self.log(f'p{self.logging_suffix}',  precision_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy()), prog_bar=True)
        self.log(f'r{self.logging_suffix}',  recall_score(gold.detach().cpu().numpy(), pred.detach().cpu().numpy()), prog_bar=True)

        return {'f1': f1, 'p': p, 'r': r}

    """
    With this strategy, the batch is not a list of depth 1, but a list of depth 2
    This is necessary because we do not want to discard the information about the
    sentences
    Apply a collate_fn, typically that of the encoder, to the flatten batch
    """
    @staticmethod
    def collate_fn(tokenizer, symbols, symbol_tensors, parser: QueryParser, batch: List[EncodeAggregateBatch], collate_fn):
        # Flatten everything
        flat = [y for x in batch for y in x]

        # Collate the flatten batch with standard collate_fn (typically, that of the encoder)
        # The result will be something on which the encoder can be called
        data = collate_fn(tokenizer, symbols, symbol_tensors, parser, flat)

        # Calculate the number of sentences 
        number_of_sentences         = [len(x) for x in batch]
        data['number_of_sentences'] = number_of_sentences
        
        return data
        



if __name__ == '__main__':
    hparams = {
    'model_name': 'google/bert_uncased_L-2_H-128_A-2',
    'aggregator_input_size': 128,
    'aggregator_number_of_attention_heads': 2,
    'use_scheduler': True,
    'num_training_steps': 10000,
    }
    from utils import init_random
    init_random(1)
    bem = BaseEncoderModel(hparams)
    bam = BaseAggregatorModel(hparams)
    beam = BaseEncodeAggregateModel(bem, bam, hparams).eval()

    b11 = EncoderBatchType(
        sentence            = 'This is a test one',
        start_highlight     = 1,
        end_highlight       = 3,
        current_rule        = '□',
        next_potential_rule = '□ □',
    )
    b12 = EncoderBatchType(
        sentence            = 'This is a test two',
        start_highlight     = 1,
        end_highlight       = 4,
        current_rule        = '□',
        next_potential_rule = '□ □',
    )
    b21 = EncoderBatchType(
        sentence            = 'This is a second batch test one',
        start_highlight     = 1,
        end_highlight       = 4,
        current_rule        = '□',
        next_potential_rule = '□?',
    )
    b22 = EncoderBatchType(
        sentence            = 'This is a second batch test two',
        start_highlight     = 1,
        end_highlight       = 4,
        current_rule        = '□',
        next_potential_rule = '□?',
    )
    b23 = EncoderBatchType(
        sentence            = 'This is a second batch test three',
        start_highlight     = 1,
        end_highlight       = 4,
        current_rule        = '□',
        next_potential_rule = '□?',
    )

    eab1 = [b11, b12]
    eab2 = [b21, b22, b23]

    beam_batch = beam.collate_fn(beam.encoder.tokenizer, beam.encoder.symbols, beam.encoder.symbol_tensors, QueryParser(), [eab1, eab2], beam.encoder.collate_fn)
    # print(beam.encode(beam_batch))
    print(beam(beam_batch))
