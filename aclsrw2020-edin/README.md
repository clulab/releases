# Exploring Interpretability in Event Extraction: Multitask Learning of a Neural Event Classifier and an Explanation Decoder: 

This is the accompanying code for our ACL SRW 2020 [paper]() 

## Train:

`python train.py [trainning_data_dir] [development_data_dir] `

## Test:

`python test.py [model] [trainning_data_dir] [testing_data_dir]`

## Data
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
