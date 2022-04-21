import argparse
from vocabulary import Vocabulary
from bert_tokenizer_custom import BertTokenizerFast

import os
from tokenizers.processors import BertProcessing
import config

from utils import init_random
from transformers import BertForMaskedLM
from datasets.load import load_dataset
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import pipeline

from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BertConfig,
)


def main():
    parser = argparse.ArgumentParser(description='Train a Transformer-based language model from scratch on a given dataset')
    parser.add_argument('--train', type=str, required=True, help='Path to train. Can be folder or file')
    parser.add_argument('--eval', type=str, required=False, default=None, help='Path to eval. Can be folder or file. [default=None]')
    parser.add_argument('--output-dir', type=str, required=True, help='The output directory to where to save the resulting model. Will save both the tokenizer and the Transformer-based model')
    parser.add_argument('--random-seed', type=int, default=1, help='The random seed to used for initialization. [default=1]')
    parser.add_argument('--config-file', type=str, default=None, help='Path to a json config file used to ')
    parser.add_argument('--epochs', type=int, default=3, help='How many epochs to train for. [default=3]')
    parser.add_argument('--warmup-steps', type=int, default=500, help='The number of warmup steps for learning rate scheduler. [default=500]')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay. [default=0.01]')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size. Check the recommended batch size for the model you are training. [default=64]')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Number of updates steps to accumulate before performing a backward/update pass. Set it accordingly to your memory constraints [default=1]')
    parser.add_argument('--cache-dir', type=str, required=False, default='~/.cache/huggingface/datasets/', help='Number of updates steps to accumulate before performing a backward/update pass. Set it accordingly to your memory constraints [default=1]')
    parser.add_argument('--subword-level', action='store_true', help='If set, will use subword tokenization (BBPE)')
    parser.add_argument('--truncation', type=int, default=300, help='Truncation size. [default=300]')
    parser.add_argument('--subword-merges-file', type=str, default=None, required=False, help='Path to the merges file for the subword tokenizer. Required if subword-level flag is set')
    parser.add_argument('--vocab', type=str, default=None, required=False, help='Path to the vocab file. Used by both word-level and subword-level tokenziers')
    args = parser.parse_args()

    init_random(args.random_seed)



    # Prepare the train and eval files
    # If folder, list all *.txt files
    # Create the dictionary used by the datasets library
    data_files = {}

    if os.path.isdir(args.train):
        train_files = [str(x) for x in list(Path(args.train).glob('*.txt'))]
    else:
        train_files = [args.train]

    data_files['train'] = train_files # [str(x) for x in list(Path('/data/nlp/corpora/odinsynth/umbc/train/').glob('*.txt'))] + [str(x) for x in list(Path('/data/nlp/corpora/odinsynth/umbc/test/').glob('*.txt'))]

    length = len(data_files['train'])
    print(f'train with {length} files')

    do_eval = True if args.eval is not None else False

    if do_eval:
        if os.path.isdir(args.eval):
            eval_files = [str(x) for x in list(Path(args.eval).glob('*.txt'))]
        else:
            eval_files = [args.eval]
        data_files['test'] = eval_files


    print('prepare the training arguments ...')
    training_args = TrainingArguments(
        output_dir                  = args.output_dir,                  # output directory
        num_train_epochs            = args.epochs,                      # total # of training epochs
        per_device_train_batch_size = args.batch_size,                  # batch size per device during training
        per_device_eval_batch_size  = args.batch_size,                  # batch size for evaluation
        do_eval                     = do_eval,                          # evaluate on held-out dataset or not
        warmup_steps                = args.warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay                = args.weight_decay,                # strength of weight decay
        logging_dir                 = './logs',                         # directory for storing logs
        save_steps                  = 1000,                             # how often to save the checkpoints
        gradient_accumulation_steps = args.gradient_accumulation_steps, # gradient accumulation steps
        fp16                        = True,                             # whether to use 16-bit (mixed) precision training instead of 32-bit training.
        load_best_model_at_end      = True,
        # dataloader_num_workers      = 16                                # number of processes to use for data loading
    )


    if args.subword_level:
        print('build the tokenizer (subword) ...')
        tokenizer = ByteLevelBPETokenizer(
                args.vocab, 
                args.subword_merges_file
            )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
        )
        tokenizer.enable_truncation(max_length=args.truncation)
    else:
        print('build the tokenizer (word level) ...')
        from tokenizers.implementations import BaseTokenizer
        vocab = Vocabulary.load(args.vocab).t2i
        tokenizer = Tokenizer(WordLevel(vocab, unk_token='[UNK]'), )
        tokenizer.pre_tokenizer = WhitespaceSplit(' ')
        tokenizer = BaseTokenizer(tokenizer)
        tokenizer.enable_truncation(max_length=args.truncation)
        tokenizer.post_processor = BertProcessing(
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ) 
        
    tokenizer = BertTokenizerFast(tokenizer)
    dc = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    print('build the config ...')
    bert_config = BertConfig(
                vocab_size                   = len(tokenizer),
                hidden_size                  = config.HIDDEN_SIZE,
                num_hidden_layers            = config.NUM_HIDDEN_LAYERS,
                num_attention_heads          = config.NUM_ATTENTION_HEADS,
                intermediate_size            = config.INTERMEDIATE_SIZE,
                hidden_act                   = config.HIDDEN_ACT,
                hidden_dropout_prob          = config.HIDDEN_DROPOUT_PROB,
                attention_probs_dropout_prob = config.ATTENTION_PROBS_DROPOUT_PROB,
                max_position_embeddings      = config.MAX_POSITION_EMBEDDINGS,
                type_vocab_size              = config.TYPE_VOCAB_SIZE,
                initializer_range            = config.INITIALIZER_RANGE,
                layer_norm_eps               = config.LAYER_NORM_EPS,
                pad_token_id                 = tokenizer.pad_token_id,
                gradient_checkpointing       = config.GRADIENT_CHECKPOINTING,
    )

    bert = BertForMaskedLM(bert_config)


    print('load dataset ...')
    dataset = load_dataset('text', data_files = data_files, cache_dir = args.cache_dir)

    def encode(example):
        return tokenizer(example['text'], truncation=True, padding='longest', max_length=args.truncation)

    
    print('process dataset ...')
    dataset = dataset.map(encode, batched=True, num_proc=40).shuffle()

    print('build trainer ...')
    trainer = Trainer(
        model                = bert,
        args                 = training_args,
        data_collator        = dc,
        train_dataset        = dataset['train'],
        eval_dataset         = dataset['test'] if 'test' in dataset.keys() else None,
        # tokenizer            = tokenizer,
        prediction_loss_only = True,
    )
        
    print('start training ...')
    trainer.train()
    print('finished training ...')
    print("run example ...")
    fill_mask = pipeline(
        'fill-mask',
        model=bert,
        tokenizer=tokenizer,
        device=0,
    )
    fe = pipeline(
        'feature-extraction',
        model=bert,
        tokenizer=tokenizer,
        device=0,
    )
    print('\tPredict the mask for: "word-This lemma-this tag-[MASK]"')
    print(fill_mask('word-This lemma-this tag-[MASK]'))
    print('\tExtract the representation of each token for: "word-This lemma-this tag-DT"')
    import torch
    print(torch.tensor(fe(['word-This lemma-this tag-DT'])))
    print(torch.tensor(fe(['word-This lemma-this tag-DT'])).shape)

# python pretrain.py --train /data/nlp/corpora/odinsynth/umbc_corpus_odinsynth_tokenized_subset3 --output-dir /home/rvacareanu/projects/odinsynth/python/results_all/word_level_small_3_300 --random-seed 1 --cache-dir /data/nlp/corpora/huggingface-datasets-cache/ --truncation 300 --batch-size 8 --gradient-accumulation-steps 32 --vocab '/data/nlp/corpora/odinsynth/umbc/vocab_50_filtered.tsv'
# python pretrain.py --train /data/nlp/corpora/odinsynth/umbc_corpus_odinsynth_tokenized_subset3 --output-dir /home/rvacareanu/projects/odinsynth/python/results_all/word_level_small_3_300 --random-seed 1 --cache-dir /data/nlp/corpora/huggingface-datasets-cache/ --truncation 300 --batch-size 8 --gradient-accumulation-steps 32 --subword-level --vocab '/home/rvacareanu/projects/odinsynth/python/results/tokenizer_100k_bbpe-vocab.json' --subword-merges-file '/home/rvacareanu/projects/odinsynth/python/results/tokenizer_100k_bbpe-merges.txt'
# 
if __name__ == '__main__':
    main()
