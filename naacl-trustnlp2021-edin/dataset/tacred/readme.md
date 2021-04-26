# Interpretability Rules: Jointly Bootstrapping a Neural Relation Extractor with an Explanation Decoder: 


## Mapping files

mappings_train.txt is the file which maps the Stanford's Tokensregex and Semregex rules to the TACRED training partition.

mappings_dev.txt is the file which maps the Stanford's Tokensregex and Semregex rules to the TACRED dev partition.

mappings_test.txt is the file which maps the Stanford's Tokensregex and Semregex rules to the TACRED test partition.


mappings_train_ssl.txt is the datapoints with matched with a rules, we also generate negative examples for it. We also put the train_ssl.json here, it is the same format as the TACRED dataset but with only the datapoints matched to the mappings_train_ssl.txt

Each line is corresponding to the datapoints within the same location in the dataset, if there is no rule matched, we labeled that line 'no_rule' else we keep a list of matched rules with the query key in that line:

```

...
no_rule
no_rule
no_rule
no_rule
[('per:title', 't_4592'), ('per:title', 't_4613'), ('per:title', 's_288')]
no_rule
no_rule
...

```

## Rules file
rules.json is the dictionary file of all rules, you can find the rule with the unique query key.