from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer


import argparse
import os
import config
import random

from utils import init_random
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer


def main():
    parser = argparse.ArgumentParser(description='Train a Transformer-based language model from scratch on a given dataset')
    parser.add_argument('--train', type=str, required=True, help='Path to train. Can be folder or file')
    parser.add_argument('--output-dir', type=str, required=True, help='The output directory to where to save the resulting model. Will save both the tokenizer and the Transformer-based model')
    parser.add_argument('--random-seed', type=int, default=1, help='The random seed to used for initialization. [default=1]')
    parser.add_argument('--config-file', type=str, default=None, help='Path to a json config file used to ')
    args = parser.parse_args()
    print('init random ...')
    init_random(args.random_seed)
    # Prepare the train and eval files
    # If folder, list all *.txt files
    print('load files ...')
    if os.path.isdir(args.train):
        train_files = [str(x) for x in list(Path(args.train).glob('*.txt'))]
    else:
        train_files = [args.train]
    
    print('shuffle ...')
    random.shuffle(train_files)


    tokenizer = ByteLevelBPETokenizer()
    print('train ...')
    tokenizer.train(files=train_files, vocab_size=32_000, min_frequency=2, special_tokens=[
        config.UNK_TOKEN,
        config.PAD_TOKEN,
        config.CLS_TOKEN,
        config.SEP_TOKEN,
        config.MASK_TOKEN,
    ])
    print('save ...')
    tokenizer.save_model(args.output_dir, 'tokenizer')
    print('example for "word-This lemma-this tag-DT"...')
    print(tokenizer.encode('word-This lemma-this tag-DT').ids)
    print(tokenizer.encode('word-This lemma-this tag-DT').tokens)

if __name__ == '__main__':
    main()
