#!/usr/bin/env bash
mkdir -p model_storage/
wget https://storage.googleapis.com/fact_verification_mithun_files/best_model_trained_on_delex_fever_84PercentDevAccuracy.pth -O model_storage/best_model.pth
wget https://storage.googleapis.com/fact_verification_mithun_files/vectorizer_delex_lr0.0005_136epochs.json -O model_storage/vectorizer.json
