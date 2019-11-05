#!/usr/bin/env bash
mkdir -p data/rte/fnc/test/
mkdir -p data/rte/fever/train/
mkdir -p data/rte/fever/dev/
mkdir -p data/rte/fever/test/
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_test_lex_4labels.jsonl -O data/rte/fever/test/fever_test_lex_fourlabels.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_train_lex_4labels.jsonl  -O data/rte/fever/train/fever_train_lex_4labels.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fn_test_split_fourlabels.jsonl -O data/rte/fnc/test/fn_test_split_fourlabels.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_dev_lex_4labels.jsonl  -O data/rte/fever/dev/fever_dev_split_fourlabels.jsonl

