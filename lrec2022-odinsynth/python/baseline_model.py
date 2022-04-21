from typing import Dict, List
from datasets import load_dataset, load_metric
from torch import tensor
from torch.utils.data.dataloader import DataLoader
from transformers import EncoderDecoderModel, BertTokenizer, AdamW, BertConfig, EncoderDecoderConfig, BertModel
from argparse import ArgumentParser
from transformers import BertTokenizerFast
from utils import highligh_word_start_tokenization_space, highlighted_indices_tokenization_space, highlighted_start_continuation_indices_tokenization_space
import json
import torch
import transformers
import config
import pytorch_lightning as pl
import tqdm
import numpy as np

"""
You can:
    - train using one sentence per datapoint using train()
    - train using all the sentences in the spec using train_multiple_sentences()
    - generate rules using test_generate()
    - handle TACRED clusters (generated with tacred_data_prep.py) by using preprocess_tacred_cluster to transform it what the model expects to receive


"""

# {"text: {"sentences":[], "start": [], "end": [], "correct_rule": ""}}
def build_training_data_from_stepsspecs_file(filename: str='/data/nlp/corpora/odinsynth/data/rules100k_unrolled/train_names', output_filename: str='/data/nlp/corpora/odinsynth/data/rules100k_seq2seq/train.tsv'):
    with open(filename) as fin:
        lines = fin.readlines()
    
    lines = [[x.strip() for x in l.split('\t')] for l in lines]
    steps = [l[0] for l in lines]
    specs = [l[1] for l in lines]
    print(len(steps))

    with open(output_filename, 'w+') as fout:
        for step_f, spec_f in tqdm.tqdm(list(zip(steps, specs))):
            with open(step_f) as fin:
                step_jsonlines = fin.readlines()
            with open(spec_f) as fin:
                spec_jsonlines = fin.readlines()
            spec_json    = json.loads(spec_jsonlines[0])
            doc_json     = json.loads(spec_jsonlines[1])
            
            spec_json['specs'].sort(key=lambda x: x['sentId'])
            
            correct_rule = json.loads(step_jsonlines[-1])['next_correct_rule']
            start = [x['start'] for x in spec_json['specs']]
            end   = [x['end'] for x in spec_json['specs']]
            sentences = [list(filter(lambda x: x['name'] == 'word', s['fields']))[0]['tokens'] for s in doc_json['sentences']]

            data_line = {
                "sentences": sentences,
                "start": start,
                "end": end,
                "correct_rule": correct_rule,
            }
            _ = json.dump(data_line, fout)
            _ = fout.write('\n')

# build_training_data_from_stepsspecs_file(filename='/data/nlp/corpora/odinsynth/data/rules100k_unrolled/train_names', output_filename='/data/nlp/corpora/odinsynth/data/rules100k_seq2seq/train.tsv')
# build_training_data_from_stepsspecs_file(filename='/data/nlp/corpora/odinsynth/data/rules100k_unrolled/test_names', output_filename='/data/nlp/corpora/odinsynth/data/rules100k_seq2seq/test.tsv')

def collate_fn(tknz, batch: List[Dict]):
    add_special_tokens = True
    input_tokens = []
    output_tokens = []
    for bdict in batch:
        b = bdict['text']
        sentences = b['sentences']
        all_sentences = [y for x in sentences for y in x[:1]]
        print(all_sentences)
        exit()
        lens = [0] + [len(x) for x in sentences[:-1]]
        lens = np.cumsum(lens).tolist()
        starts = [lens[x] + b['start'][x] for x in range(len(b['start']))]
        ends = [lens[x] + b['end'][x] for x in range(len(b['end']))]

        correct_rule = b['correct_rule']

        sentence_tokenized = tknz(all_sentences, padding='max_length', truncation=True, max_length=512, return_tensors='pt', is_split_into_words=True, add_special_tokens=add_special_tokens, return_offsets_mapping=True)
        if add_special_tokens:
            starts = [s+1 for s in starts]
            ends   = [s+1 for s in ends]

        words_to_highlight = []
        for s, e in zip(starts, ends):
            if e < 512:
                words_to_highlight += range(s, e)
        index = highlighted_indices_tokenization_space(words_to_highlight, sentence_tokenized['offset_mapping'])
        sentence_tokenized['token_type_ids'][0][index] = 1


        # shapes are (1, length), where length is the length of each one. Mmight be different between items in the batch. We pad below, before returning
        input_tokens.append({
            'input_ids': sentence_tokenized['input_ids'],
            'attention_mask': sentence_tokenized['attention_mask'],
            'token_type_ids': sentence_tokenized['token_type_ids'],
        })
        output_tokenized = tknz(correct_rule, padding='max_length', truncation=True, max_length=256, return_tensors='pt', is_split_into_words=False, add_special_tokens=True, return_offsets_mapping=False)
        labels = torch.clone(output_tokenized['input_ids'])

        output_tokens.append({
            **output_tokenized,
            'labels': labels
        })
    

    # # Start padding the tokens
    # # encoder's input
    # max_input_length = min(max([x['input_ids'].shape[1] for x in input_tokens]), 512)
    # for t1 in input_tokens:
    #     current_length  = t1['input_ids'].shape[1]
    #     if current_length < max_input_length:
    #         pad = torch.tensor([tokenizer.pad_token_id] * (max_input_length - current_length)).unsqueeze(dim=0)
    #         for key in t1.keys():
    #             t1[key] = torch.cat([t1[key], pad], dim=1)
    # 
    # # decoder's input
    # max_input_length = min(max([x['input_ids'].shape[1] for x in output_tokens]), 512)
    # for t1 in output_tokens:
    #     current_length  = t1['input_ids'].shape[1]
    #     if current_length < max_input_length:
    #         pad = torch.tensor([tokenizer.pad_token_id] * (max_input_length - current_length)).unsqueeze(dim=0)
    #         for key in t1.keys():
    #             t1[key] = torch.cat([t1[key], pad], dim=1)

    batch = {
        'input_ids':              torch.cat([x['input_ids'][:, :512]      for x in input_tokens],  dim=0).to(torch.device('cuda:0')),
        'attention_mask':         torch.cat([x['attention_mask'][:, :512] for x in input_tokens],  dim=0).to(torch.device('cuda:0')),
        'token_type_ids':         torch.cat([x['token_type_ids'][:, :512] for x in input_tokens],  dim=0).to(torch.device('cuda:0')),
        'decoder_input_ids':      torch.cat([x['input_ids'][:, :512]      for x in output_tokens], dim=0).to(torch.device('cuda:0')),
        'decoder_attention_mask': torch.cat([x['attention_mask'][:, :512] for x in output_tokens], dim=0).to(torch.device('cuda:0')),
        'labels':                 torch.cat([x['labels'][:, :512]         for x in output_tokens], dim=0).to(torch.device('cuda:0')),
    }
    
    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
    # We have to make sure that the PAD token is ignored
    batch["labels"][batch["labels"] == tknz.pad_token_id] = -100

    return batch

def unroll(x):
    result = []
    for sen, sta, end in zip(x['sentences'], x['start'], x['end']):
        result.append({
            "sentence": sen,
            "start": sta,
            "end": end,
            "rule": x['correct_rule']
        })
    return result
    

# Only one sentence
def preprocess_data_unrolled(tokenizer, batch: Dict):
    loaded_batch = [json.loads(x) for x in batch['text']]

    output = []
    
    for b in loaded_batch:
        bt      = tokenizer(b['sentence'], padding="max_length", truncation=True, max_length=512, return_tensors='pt', is_split_into_words=True, add_special_tokens=True, return_offsets_mapping=True)
        outputs = tokenizer(b['rule'], padding="max_length", truncation=True, max_length=192, add_special_tokens=True, return_offsets_mapping=True)
        idx = highlighted_indices_tokenization_space(range(b['start']+1,b['end']+1), bt['offset_mapping'])
        bt['token_type_ids'][0][idx]=1
        bt["labels"] = torch.tensor(outputs['input_ids']).unsqueeze(0)
        bt['decoder_attention_mask'] = torch.tensor(outputs['attention_mask']).unsqueeze(0)

        output.append({
            "input_ids":              bt['input_ids'],
            "token_type_ids":         bt['token_type_ids'],
            "attention_mask":         bt['attention_mask'],
            "offset_mapping":         bt['offset_mapping'],
            "labels":                 bt['labels'],
            "decoder_attention_mask": bt['decoder_attention_mask'],
        })

    output = {
        "input_ids":              torch.cat([x['input_ids'] for x in output], dim=0).tolist(),
        "token_type_ids":         torch.cat([x['token_type_ids'] for x in output], dim=0).tolist(),
        "attention_mask":         torch.cat([x['attention_mask'] for x in output], dim=0).tolist(),
        "decoder_input_ids":      torch.cat([x['labels'] for x in output], dim=0).tolist(),
        "decoder_attention_mask": torch.cat([x['decoder_attention_mask'] for x in output], dim=0).tolist(),
        "labels":                 torch.cat([x['labels'] for x in output], dim=0).tolist(),
    }   

    output["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in output["labels"]]
    return output



# Multiple sentences corresponding to each rule. The sentences are appended one after the other
def preprocess_data_multiple_sentences(tokenizer, batch: Dict):
    loaded_batch = [json.loads(x) for x in batch['text']]

    output = []

    add_special_tokens = True
    
    for b in loaded_batch:
        sentences = []

        # During generation we cannot control for token_type_ids. Therefore, we have to add a dummy 
        for s, e, sent in zip(b['start'], b['end'], b['sentences']):
            sentences.append(sent[:s] + ['[HIGHLIGHT-START]'] + sent[s:e] + ['[HIGHLIGHT-END]'] + sent[e:] + ["SEP"])
            
        all_sentences = [y for x in sentences for y in x]

        correct_rule = b['correct_rule']

        sentence_tokenized = tokenizer(all_sentences, padding='max_length', truncation=True, max_length=512, return_tensors='pt', is_split_into_words=True, add_special_tokens=add_special_tokens, return_offsets_mapping=True)
        outputs = tokenizer(correct_rule, padding="max_length", truncation=True, max_length=192, add_special_tokens=True, return_offsets_mapping=True)

        sentence_tokenized["labels"] = torch.tensor(outputs['input_ids']).unsqueeze(0)
        sentence_tokenized['decoder_attention_mask'] = torch.tensor(outputs['attention_mask']).unsqueeze(0)

        output.append({
            "input_ids":              sentence_tokenized['input_ids'],
            "token_type_ids":         sentence_tokenized['token_type_ids'],
            "attention_mask":         sentence_tokenized['attention_mask'],
            "offset_mapping":         sentence_tokenized['offset_mapping'],
            "labels":                 sentence_tokenized['labels'],
            "decoder_attention_mask": sentence_tokenized['decoder_attention_mask'],
        })

    output = {
        "input_ids":              torch.cat([x['input_ids'] for x in output], dim=0).tolist(),
        "token_type_ids":         torch.cat([x['token_type_ids'] for x in output], dim=0).tolist(),
        "attention_mask":         torch.cat([x['attention_mask'] for x in output], dim=0).tolist(),
        "decoder_input_ids":      torch.cat([x['labels'] for x in output], dim=0).tolist(),
        "decoder_attention_mask": torch.cat([x['decoder_attention_mask'] for x in output], dim=0).tolist(),
        "labels":                 torch.cat([x['labels'] for x in output], dim=0).tolist(),
    }   
    
    output["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in output["labels"]]

    return output


def preprocess_tacred_cluster(tokenizer, path: str):
    add_special_tokens = True
    import pandas as pd
    from ast import literal_eval
    data = pd.read_csv(path, sep='\t', quoting=3, index_col=0, converters={'highlighted': literal_eval, 'tokens': literal_eval})
    sentences = data['tokens'].tolist()
    relation  = data['relation']
    is_reversed = data['reversed'].tolist()[0]
    if is_reversed == 0:
        first_type  = data['subj_type'].tolist()
        second_type = data['obj_type'].tolist()
        start       = data['subj_end'].tolist()
        end         = data['obj_start'].tolist()
        first_type_start  = data['subj_start'].tolist()
        first_type_end    = data['subj_end'].tolist()
        second_type_start = data['obj_start'].tolist()
        second_type_end   = data['obj_end'].tolist()
    else:
        first_type  = data['obj_type'].tolist()
        second_type = data['subj_type'].tolist()
        start       = data['obj_end'].tolist()
        end         = data['subj_start'].tolist()
        first_type_start  = data['obj_start'].tolist()
        first_type_end    = data['obj_end'].tolist()
        second_type_start = data['subj_start'].tolist()
        second_type_end   = data['subj_end'].tolist()

    sentences_processed = []
    # highlight_start     = []
    # highlight_end       = []
    highlighted = []
    for fss, fse, sts, ste, ft, st, s in zip(first_type_start, first_type_end, second_type_start, second_type_end, first_type, second_type, sentences):
        result = s[:(fss+1)] + [ft, '[START-HIGHLIGHT]'] + s[(fse+1):sts] + ['[END-HIGHLIGHT]', st] + s[(ste+1):]
        sentences_processed.append(result)        
        highlighted.append(['[START-HIGHLIGHT]'] + s[(fse+1):sts] + ['[END-HIGHLIGHT]'])
        # highlight_start.append(len(s[:(fss+1)]))
        # highlight_end.append(len(s[:(fss+1)]) + 1 + len(s[(fse+1):sts]) + 1)
    
    ##### Finished the preparation for calling the tokenizer

    # Process
    # print(sentences_processed)
    # exit()
    all_sentences = [y for x in sentences_processed for y in x]

    lens = [0] + [len(x) for x in sentences_processed[:-1]]
    lens = np.cumsum(lens).tolist()
    sentence_tokenized = tokenizer(all_sentences, padding='max_length', truncation=True, max_length=512, return_tensors='pt', is_split_into_words=True, add_special_tokens=add_special_tokens, return_offsets_mapping=True)

    pattern = 0
    skip = 0
    if max([len(l) for l in highlighted]) - 2 == 0 and min([len(l) for l in highlighted]) - 2 == 0:
        # pattern = '[' + ' | '.join([f'word={x}' for x in first_type]) + ']' + ' ' + '[' + ' | '.join([f'word={x}' for x in second_type]) + ']'
        skip = 1
    elif max([len(l) for l in highlighted]) - 2 == 1 and min([len(l) for l in highlighted]) - 2 == 1:
        # h = [x[1] for x in highlighted]
        skip = 1
    
    
    output = {
        "text":                 all_sentences,
        "input_ids":            sentence_tokenized['input_ids'],
        "token_type_ids":       sentence_tokenized['token_type_ids'],
        "attention_mask":       sentence_tokenized['attention_mask'],
        "highlighted":          highlighted,
        "spec_size":            len(sentences),
        "max_highlight_length": max([len(l) for l in highlighted]) - 2,
        "min_highlight_length": min([len(l) for l in highlighted]) - 2,
        "cluster_path":         path,
        "direction":            is_reversed,
        "relation":             relation.tolist()[0],
        "skip":                 skip,
        'first_type':           first_type,
        'second_type':          second_type,
    }   
    return output



def train():

    config_encoder = BertConfig()
    config_decoder = BertConfig()
    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    tokenizer = BertTokenizerFast.from_pretrained("google/bert_uncased_L-8_H-512_A-8")
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token


    model = EncoderDecoderModel.from_encoder_decoder_pretrained("google/bert_uncased_L-8_H-512_A-8", "google/bert_uncased_L-8_H-512_A-8")#.to(torch.device('cuda:0')) # initialize Bert2Bert
    
    dataset = load_dataset(
        'text', 
        data_files = {
            'train': '/data/nlp/corpora/odinsynth/data/rules100k_seq2seq/train_unrolled.tsv',
            'test' : '/data/nlp/corpora/odinsynth/data/rules100k_seq2seq/test_unrolled.tsv'
            }, 
        cache_dir = '/data/nlp/corpora/huggingface-datasets-cache/'
    )

    train = dataset['train'].filter(lambda x: json.loads(x['text'])['end'] < 512).map(lambda x: preprocess_data_unrolled(tokenizer, x), batched=True, batch_size=8)
    test  = dataset['test'].filter(lambda x: json.loads(x['text'])['end'] < 512).map(lambda x: preprocess_data_unrolled(tokenizer, x), batched=True, batch_size=8)
    
    train.remove_columns_('text')
    test.remove_columns_('text')
    train.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )
    test.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
    training_args = Seq2SeqTrainingArguments(
        output_dir="/home/rvacareanu/projects/odinsynth/python/logs/seq2seq/model",
        learning_rate=3e-5,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        warmup_ratio=0.1,
        save_total_limit=10,
        num_train_epochs=10,
        fp16=True, 
        logging_strategy='epoch',
        dataloader_num_workers=0,
        metric_for_best_model='bleu_score',
        greater_is_better=True,
        load_best_model_at_end=True,
        logging_dir='/home/rvacareanu/projects/odinsynth/python/logs/seq2seq/logs/'
    )

    # print(training_args)
    # exit()
    # instantiate trainer

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.max_length = 512
    model.config.min_length = 1
    model.config.no_repeat_ngram_size = 15
    model.early_stopping = True
    model.length_penalty = 3.0
    model.num_beams = 7

    import datasets
    bleu = datasets.load_metric("sacrebleu")
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
    
        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        label_str = [[x] for x in label_str]

        bleu_output = bleu.compute(predictions=pred_str, references=label_str)
        return {
            "eval_bleu_score": round(bleu_output['score'], 4),
            "bleu_precision1": round(bleu_output['precisions'][0], 4),
            "bleu_precision2": round(bleu_output['precisions'][1], 4),
            "bleu_precision3": round(bleu_output['precisions'][2], 4),
            "bleu_precision4": round(bleu_output['precisions'][3], 4),
        }


    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        compute_metrics=compute_metrics,
        # data_collator=data_collator,
    )
    trainer.train()
    import os
    os.mkdir('/home/rvacareanu/projects/odinsynth/python/logs/seq2seq/best')
    trainer.save_model("/home/rvacareanu/projects/odinsynth/python/logs/seq2seq")


def train_multiple_sentences():
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    tokenizer.add_special_tokens({'additional_special_tokens': ["[lemma=", "[word=", "[tag=", "] ", "[START-HIGHLIGHT]", "[END-HIGHLIGHT]"]})

    config_encoder = BertConfig.from_pretrained(model_name)
    config_decoder = BertConfig.from_pretrained(model_name)
    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    # enc_conf = BertConfig()
    
    # model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name, pad_token_id=tokenizer.eos_token_id)#.to(torch.device('cuda:0')) # initialize Bert2Bert

    model = EncoderDecoderModel(config=config)
    model.encoder.resize_token_embeddings(len(tokenizer)) 
    model.decoder.resize_token_embeddings(len(tokenizer)) 


    dataset = load_dataset(
        'text', 
        data_files = {
            'train': '/data/nlp/corpora/odinsynth/data/rules100k_seq2seq/train.tsv',
            'test' : '/data/nlp/corpora/odinsynth/data/rules100k_seq2seq/test.tsv'
            }, 
        cache_dir = '/data/nlp/corpora/huggingface-datasets-cache/'
    )
    train = dataset['train'].map(lambda x: preprocess_data_multiple_sentences(tokenizer, x), batched=True, batch_size=8)
    test  = dataset['test'].map(lambda x: preprocess_data_multiple_sentences(tokenizer, x), batched=True, batch_size=8)
    
    train.remove_columns_('text')
    test.remove_columns_('text')
    train.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )
    test.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
    training_args = Seq2SeqTrainingArguments(
        output_dir="/home/rvacareanu/projects/odinsynth/python/logs/seq2seq3/bert-base-uncased/model",
        learning_rate=3e-5,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=32,
        per_device_eval_batch_size=32,
        predict_with_generate=True,
        warmup_ratio=0.1,
        save_total_limit=10,
        num_train_epochs=10,
        fp16=True, 
        logging_strategy='epoch',
        dataloader_num_workers=0,
        metric_for_best_model='bleu_score',
        greater_is_better=True,
        load_best_model_at_end=True,
        logging_dir='/home/rvacareanu/projects/odinsynth/python/logs/seq2seq3/bert-base-uncased/logs/'
    )

    # instantiate trainer
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.config.max_length = 512
    model.config.min_length = 1
    model.config.no_repeat_ngram_size = 50
    model.early_stopping = True
    model.length_penalty = 1.0
    model.num_beams = 7

    import datasets
    bleu = datasets.load_metric("sacrebleu")
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
    
        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        label_str = [[x] for x in label_str]

        bleu_output = bleu.compute(predictions=pred_str, references=label_str)
        return {
            "eval_bleu_score": round(bleu_output['score'], 4),
            "bleu_precision1": round(bleu_output['precisions'][0], 4),
            "bleu_precision2": round(bleu_output['precisions'][1], 4),
            "bleu_precision3": round(bleu_output['precisions'][2], 4),
            "bleu_precision4": round(bleu_output['precisions'][3], 4),
        }


    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        compute_metrics=compute_metrics,
        # data_collator=data_collator,
    )
    trainer.train()
    import os
    # if not os.path.exists('/home/rvacareanu/projects/odinsynth/python/logs/seq2seq/best'):
        # os.mkdir('/home/rvacareanu/projects/odinsynth/python/logs/seq2seq/best')
    trainer.save_model("/home/rvacareanu/projects/odinsynth/python/logs/seq2seq2/bert-base-uncased/best")



def test_generate():
    # model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased').to(torch.device('cuda:0'))
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    from_where = "/home/rvacareanu/projects/odinsynth/python/logs/seq2seq2/bert-base-uncased/best"
    import glob
    clusters = glob.glob('/data/nlp/corpora/odinsynth/data/TACRED/odinsynth_tacred101/*/*')
    
    tokenizer = BertTokenizerFast.from_pretrained(from_where)
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    
    config = EncoderDecoderConfig.from_pretrained(from_where)
    model = EncoderDecoderModel.from_pretrained(from_where, config=config).to(torch.device('cuda:0'))
    # set special tokens
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    # model.config.vocab_size = model.config.decoder.vocab_size
    # model.config.max_length = 512
    # model.config.min_length = 128
    # model.config.no_repeat_ngram_size = 10
    # model.config.early_stopping = True
    # model.config.length_penalty = 2.0
    # model.config.num_beams = 4
    model.eval()

    fout = open('/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/seq2seq_bert-base-uncased/all_solutions_partial.tsv', 'w+')
    fout.write('spec_size\tmax_highlight_length\tmin_highlight_length\tcluster_path\tpattern\tdirection\trelation\tscorerVersion\tsolution_status')
    fout.write('\n')
    for cluster_path in tqdm.tqdm(clusters):
        cluster = preprocess_tacred_cluster(tokenizer, cluster_path)
        # print(cluster)
        # exit()
        if cluster['skip'] == 0:
            generated = model.generate(cluster['input_ids'].to(torch.device('cuda:0')), decoder_start_token_id=model.config.decoder_start_token_id, num_beams=5, early_stopping=True)#, eos_token_id=model.config.decoder.sep_token_id)
            rule = tokenizer.decode(generated[0]).replace(' ] ', ']').replace('[PAD] ', '').replace('[SEP]', '').replace('[CLS]', '')
            rule = '[' + ' | '.join([f'word={x.lower()}' for x in set(cluster['first_type'])]) + ']' + ' ' + rule.strip() + ' ' + '[' + ' | '.join([f'word={x.lower()}' for x in set(cluster['second_type'])]) + ']'
            line = str(cluster['spec_size']) + '\t' + str(cluster['max_highlight_length']) + '\t' + str(cluster['min_highlight_length']) + '\t' + cluster['cluster_path'] + '\t' + rule + '\t' + str(cluster['direction']) + '\t' + cluster['relation'] + '\t' + "seq2seq_bertbase" + "\t" + "1"
            fout.write(line)
            fout.write('\n')
    fout.close()

train_multiple_sentences()
# test_generate()
# exit()
# train()