#!/usr/bin/env bash
mkdir -p model_storage/
wget https://storage.googleapis.com/fact_verification_mithun_files/best_model_fever_lex_82.20.pth  -O model_storage/best_model.pth
wget https://storage.googleapis.com/fact_verification_mithun_files/vectorizer_fever_lex.json  -O model_storage/vectorizer.json
