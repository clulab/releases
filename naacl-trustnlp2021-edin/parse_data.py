from data.loader import *

with open(filename) as infile:
    data = json.load(infile)

def preprocess(data):
    """ Preprocess the data and convert to ids. """
    
    with open('/mappings_train.txt') as f:
        mappings = f.readlines()
    with open('dataset/tacred/rules.json') as f:
        rules = json.load(f)
    for c, d in enumerate(data):
        tokens = list(d['token'])
        tokens = [t.lower() for t in tokens]
        rule_mask = [0 for t in range(tokens)]
        # anonymize tokens
        ss, se = d['subj_start'], d['subj_end']
        os, oe = d['obj_start'], d['obj_end']
        tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
        tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)
        masks[ss:se+1] = [1] * (se-ss+1)
        masks[os:oe+1] = [1] * (oe-os+1)
        rule = []
        if 't_' in mappings[c] or 's_' in mappings[c]:
            rule = helper.word_tokenize(rules[eval(mappings[c])[0][1]])
            
