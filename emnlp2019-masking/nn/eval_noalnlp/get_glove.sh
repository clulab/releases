#!/usr/bin/env bash
mkdir -p data
mkdir -p data/glove
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip -d data/glove
