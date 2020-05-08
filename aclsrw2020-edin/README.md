# Exploring Interpretability in Event Extraction: Multitask Learning of a Neural Event Classifier and an Explanation Decoder: 

This is the accompanying code and data for our ACL SRW 2020 [paper](http://clulab.cs.arizona.edu/papers/aclsrw2020-edin.pdf) 

You can find the BioNLP 2013 GE Task data and evaluation tool [here](http://bionlp.dbcls.jp/projects/bionlp-st-ge-2013/wiki)

## Requirements

- Python 3 
- [DyNet](https://dynet.readthedocs.io/en/latest/) 
- [NLTK](https://www.nltk.org/)
- [NumPy](https://numpy.org/)


## Usage

Get the data and evaluation tool from the shared task website: [BioNLP 2013 GE Task](http://bionlp.dbcls.jp/projects/bionlp-st-ge-2013/wiki) 

Preprocessed silver data:

```bash
python prep_corpus.py [silver_data_json path_silver_processed_dir event_type]
```

Preprocess data and write in a pickle file:

```bash
python prep_data.py BioNLP-ST-2013_GE_train_data_rev3/ [path_silver_processed_dir] BioNLP-ST-2013_GE_devel_data_rev3/ BioNLP-ST-2013_GE_test_data_rev1/ [event_type path_processed_dir]
```

Train model on training and dev data in pickle file, write a model:

```bash
python train.py [path_processed_dir embedding_file model_dir]
```

Generate the predicted .a2p files:

```bash
python brat.py [model_dir path_processed_dir] 
```

You can rename all a2p files to a2 and upload them to the BioNLP [online evaluation](http://bionlp-st.dbcls.jp/GE/2013/eval-test/).

## Silver Data

The silver data extracted by our rule-based system are also available in this repository.
In particular: `ph_events.json` contains all the phosphorylation events we extracted, `lo_events.json` contains all the localization events we extracted, and, lastly, `ge_events.json` contains all gene expression events we extracted.

All these files follow the same JSON format:

```
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
```

The rules used to extract these statements are available in `rules.yml`.

Citation:
```
@inproceedings{zheng-tang-2019-edin,
    title = "Exploring Interpretability in Event Extraction: Multitask Learning of a Neural Event Classifier and an Explanation Decoder",
    author = "Tang, Zheng and Hahn-Powell, Gustave and Surdeanu, Mihai",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop",
    month = jul,
    year = "2020",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "http://clulab.cs.arizona.edu/papers/aclsrw2020-edin.pdf"
}
```
