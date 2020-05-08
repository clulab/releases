# Exploring Interpretability in Event Extraction: Multitask Learning of a Neural Event Classifier and an Explanation Decoder: 

This is the accompanying code and data for our ACL SRW 2020 [paper](http://clulab.cs.arizona.edu/papers/aclsrw2020-edin.pdf) 

You can find the BioNLP 2013 GE Task data and evaluation tool [here](http://bionlp.dbcls.jp/projects/bionlp-st-ge-2013/wiki)

## Requirements

- Python 3 (tested using version 3.6.9)
- [DyNet](https://dynet.readthedocs.io/en/latest/) 
- [NLTK](https://www.nltk.org/)
- [NumPy](https://numpy.org/)


## Usage

Get data and evaluation tool from the website of [BioNLP 2013 GE Task]http://bionlp.dbcls.jp/projects/bionlp-st-ge-2013/wiki) 

Make preprocessed silver data:

```bash
python prep_corpus.py [silver_data_json path_processed_dir event_type]
```

Preprocess data and write in a pickle file:

```bash
python prep_data.py [BioNLP dir-datasets path-embeddings path-output]
```

Review hyperparameter settings in `hparams.conf`.

Train model on training and dev data in pickle file, write a model and a log file in `dir-model`:

```bash
python train.py [trainning_data_dir] [development_data_dir] 
```

Generate the predicted .a2p files:

```bash
python brat.py [trainning_data_dir] [development_data_dir] 
```

You can rename all a2p files to a2 and upload them to the [online evaluation](http://bionlp-st.dbcls.jp/GE/2013/eval-test/)

## Silver Data

You may also need a pre-trained embedding file and replace the file claimed in train.py with it.

```
We put our rule-based system extracted sliver data here.

ph_events.json contains all the phosphorylation events we extracted, lo_events.json contains all the localization events we extracted and ge_events.json contains all gene expression events we extracted.

They are lists of JSON objects follow this format:
{
    "sentence": The sentence contains the event,
    "rule": The rule name of which find this event,
    "trigger": [
      trigger text,
      [
        character offset for the start of the event trigger,
        character offset for the start of the event trigger
      ]
    ],
    "entity": [
      entity text,
      [
        character offset for the start of the event entity,
        character offset for the start of the event entity
      ]
    ]
}

We have put the rules we used here in rules.yml
```

Citation:
```

```
