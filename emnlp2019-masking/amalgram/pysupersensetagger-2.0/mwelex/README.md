English Multiword Expression Lexicons
=====================================

Compiled by Nathan Schneider, Emily Danchik, Chris Dyer, and Noah A. Smith.

 - version 1.0 (2014-04-19): 9 lexicons, SAID extraction script, Yelp word clusters

The lexical resources can be downloaded at 

  http://www.ark.cs.cmu.edu/LexSem/

This dataset is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) license (see LICENSE).


Contents
--------

This is a collection of type-level lexical resources that were used 
to help identify multiword expressions in (Schneider et al., *TACL* 2014).
The resources are as follows:

## Multiword Lexicons

  - cedt_mwes.json: Multiword lemmas, named entities, and CPHR and DPHR phrases 
    from the English side of the Prague Czech-English Dependency Treebank (Čmejrek et al., 2005; 
    http://ufal.mff.cuni.cz/pcedt2.0/; http://catalog.ldc.upenn.edu/LDC2012T08)

  - enwikt.json: Multiword entries from English Wiktionary (http://en.wiktionary.org; 
    data from https://toolserver.org/~enwikt/definitions/enwikt-defs-20130814-en.tsv.gz)

  - LVCs.json: List of light verb constructions provided by Claire Bonial

  - oyz_idioms.json: Multiword entries from Oyz's compilation of dictionary entries for 
    frequent English verbs (http://home.postech.ac.kr/~oyz/doc/idiom.html)

  - phrases_dot_net.json: Multiword entries on the Phrases.net website

  - semcor_mwes.json: Multiword entries in SemCor (Miller et al., 1993; accessed with NLTK)

  - vpc.json: Verb-particle constructions in the dataset of (Baldwin, 2008; 
    http://www.csse.unimelb.edu.au/research/lt/resources/vpc/vpc.tgz)

  - wikimwe.json: Entries from WikiMwe (Hartmann et al., 2011; 
    http://www.ukp.tu-darmstadt.de/data/lexical-resources/wikimwe/)

  - wordnet_mwes.json: Multiword lemma entries in English WordNet (Fellbaum, 1998; 
    http://wordnet.princeton.edu/; accessed with NLTK)

## SAID Extraction Script

  - said2json.py: The SAID idioms database (Kuiper et al., 2003) is 
    [distributed by LDC](http://catalog.ldc.upenn.edu/LDC2003T10). This script will access 
    a local installation of SAID to extract a JSON file of lexicon entries. 
    Run it by passing the path to the 'data' directory of the SAID installation:

        $ python2.7 said2json.py /path/to/said/data > said.json

## Word Clusters

  - yelpac-c1000-m25.gz: These were obtained by running Brown clustering (implementation 
    by Liang, 2005; https://github.com/percyliang/brown-cluster) on the Yelp Academic Dataset 
    (https://www.yelp.com/academic_dataset), which has 21 million words of online reviews.
    It is a hard hierarchical clustering into 1000 clusters of words appearing at least 25 times.


JSON Format
-----------

Each of the lexicon JSON files contains one entry per line. Two examples from cedt_mwes.json:

```
{"count": 1, "lemmas": ["ibm", "australia", "ltd."], "datasource": "Prague CEDT 2.0", "label": "NE"}
{"count": 3, "lemmas": ["have", "hand"], "datasource": "Prague CEDT 2.0", "label": "DPHR"}
```

The fields in each entry are:

  1. `lemmas` or `words`: words or lemmas comprising the expression (some resources 
     provide lemmas; others provide fully inflected words). LVCs.json instead has 
     `verblemma` and `noun`.

  2. `poses`: parts of speech, if available

  3. `datasource`: name of the lexicon

  4. `label`: category of expression (resource-specific; may simply be `"MWE"`)

  5. `count`: frequency in the source data, if available

There are also some fields specific to a single resource (`pmi` for wikimwe.json, 
`context` for vpc.json, `contexts` for LVCs.json, `files` for semcor_mwes.json).


Further information
-------------------

These resources were used to train a system that identifies multiword expressions in context; 
this is described in

 -  Nathan Schneider, Emily Danchik, Chris Dyer, and Noah A. Smith (2014). 
    Discriminative lexical semantic segmentation with gaps: running the MWE gamut. 
    _Transactions of the Association for Computational Linguistics._

Contact [Nathan Schneider](http://nathan.cl) with questions.
