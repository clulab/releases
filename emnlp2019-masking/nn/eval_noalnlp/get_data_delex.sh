#!/usr/bin/env bash
mkdir -p data/rte/fnc/test/
mkdir -p model_storage/
wget https://storage.googleapis.com/fact_verification_mithun_files/fnc_test_delex_oaner_4labels.jsonl -O data/rte/fnc/test/fn_test_split_fourlabels.jsonl
