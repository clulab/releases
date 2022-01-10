# Automatic Correction of Syntactic Dependency Annotation Differences

## Highlights

This repo contains the code necessary to convert and combine training data files to retrain the PaT and Stanza parsers.


## Data

Our base corpus is the GUM corpus.

Our augment corpus is the WSJ portion of the Penn Treebank, converted into UD format by [CLU Processors](https://github.com/clulab/processors).


## Vectors

For the Converted-GloVe condition, we use the pretrained GloVe word vectors glove.840B.300d.zip, available [here](https://nlp.stanford.edu/projects/glove/).

For the Converted-BERT condition, we generate contextualized word embeddings using our training data. For this, we first run `get_text_from_files.py` to get just the texts from our conllu style files. We then use [BERT wordvecs](https://github.com/spyysalo/bert-wordvecs) with a pretrained multilingual BERT model to generate the contextualized embeddings. We do this for each seed for each partition.

## Run

Before retraining our parsers, we need to (1) identify the mismatches between our two data sets, (2) convert the data, and (3) combine the converted data with our original training data.

First prepare the random samples from the original training and dev files by running `prepare_training_data.py`.

Next, create the converted training data files by running `get_corpus_comparison_lists.py`. Note that you will need to adjust the file according to the partition size and seed you wish to create the file for (unlike the other Python scripts, which already account for this).

After creating the converted files for the augment corpus, you can combine them with the original, unconverted training partitions of the base corpus by running `combine_train_files.py`.


## Retraining the Parsers

For this project we retrain the Stanza and PaT parsers using default settings. Consult the documentation for [Stanza](https://stanfordnlp.github.io/stanza/training.html) and [PaT](https://github.com/clulab/releases/tree/master/lrec2020-pat) for more details on how to train and test with these parsers.
