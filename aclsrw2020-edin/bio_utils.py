from nltk import sent_tokenize, pos_tag
import re
import numpy as np
np.random.seed(1)
import math
import json

import os
import glob
import argparse
import random

from collections import defaultdict

simple_events = ["Gene_expression", "Transcription", "Protein_catabolism", "Localization", "Binding", "Protein_modification",
 "Phosphorylation", "Ubiquitination", "Acetylation", "Deacetylation"]

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"EOS":0,"UNK":1,"THEME":2}
        self.index2word = {0: "EOS", 1:"UNK", 2:"THEME"}
        self.n_words = 3  # Count SOS and EOS
        self.labels = ["NoRel"]
        self.label2id = {"NoRel":0}

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

def load_embeddings(file, lang):
    emb_matrix = None
    emb_dict = dict()
    for line in open(file):
        if not len(line.split()) == 2:
            if "\t" in line:
                delimiter = "\t"
            else:
                delimiter = " "
            line_split = line.rstrip().split(delimiter)
            # extract word and vector
            word = line_split[0]
            vector = np.array([float(i) for i in line_split[1:]])
            embedding_size = vector.shape[0]
            emb_dict[word] = vector
    print (lang.n_words)
    for i in range(3, lang.n_words):
        base = math.sqrt(6/embedding_size)
        word = lang.index2word[i]
        try:
            vector = emb_dict[word]
        except KeyError:
            vector = np.random.uniform(-base,base,embedding_size)
        if np.any(emb_matrix):
            emb_matrix = np.vstack((emb_matrix, vector))
        else:
            emb_matrix = np.random.uniform(-base,base,(4, embedding_size))
            emb_matrix[3] = vector
    return emb_matrix

def sanitizeWord(w):
    if w.startswith("$T"):
        return w
    if w == ("xTHEMEx"):
        return "xTHEMEx"
    if w == ("xTRIGGERx"):
        return "xTRIGGERx"
    w = w.lower()
    if is_number(w):
        return "xnumx"
    w = re.sub("[^a-z_]+","",w)
    if w:
        return w

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def word_tokenize(text):
    return re.findall(r"[\w]+|[^\w\s,]",text)

def get_token_spans(text):
    """
    returns (words, start_offsets, end_offsets)
    for each sentence in the provided text
    """
    offset = 0
    for s in sent_tokenize(text):
        offset = text.find(s, offset)
        yield sentence_tokens(s, offset)

def sentence_tokens(sentence, offset):
    """this is meant to be used by get_token_spans() only"""
    pos = 0
    starts = []
    ends = []
    words = word_tokenize(sentence)
    for w in words:
        pos = sentence.find(w, pos)
        starts.append(pos + offset)
        pos += len(w)
        ends.append(pos + offset)
    return words, starts, ends

def get_trigger(s, e, entities, phosphorylations):
    for tlbl, trigger, entity, rule in phosphorylations:
        if entities[trigger][2] >= s and entities[trigger][1] <= e:
            yield tlbl, trigger, entity, rule

def check_entity(e, x):
    for p in x:
        if e == p[-2]:
            return False
    return True

def get_entity(s, e, entities, x):
    for entity in entities:
        if entities[entity][2] > s and entities[entity][1] < e and entities[entity][0] == "Protein" and check_entity(entity,x):
            yield entity

def token_span(entity, starts):
    res = []
    offset = entity[1]
    for w in word_tokenize(entity[-1]):
        if entity[-1][offset-entity[1]] == " ":
            offset += 1
        try:
            res.append(starts.index(offset))
        except:
            for i, s in enumerate(starts):
                if i<len(starts)-1 and s<offset and starts[i+1]>offset:
                    res.append(i)
                    break
        offset += len(w)
    return res

def get_id(proteins, i, starts, entities):
    for p in proteins:
        if token_span(entities[p], starts) and i == token_span(entities[p], starts)[0]:
            return p
    return None

def replace_protein(words, entities, starts, ends, proteins):
    res = []
    ps = []
    ss = []
    es = []
    for p in proteins:
        ps += token_span(entities[p], starts)
    for i, w in enumerate(words):
        if i not in ps:
            res.append(w)
            ss.append(starts[i])
            es.append(ends[i])
        p = get_id(proteins, i, starts, entities)
        if p:
            res.append("$"+p)
            ss.append(starts[i])
            es.append(ends[i])
    return res, ss, es

def check_events(line):
    event_set = {"Gene_expression", "Localization", "Phosphorylation"}
    for event in event_set:
        if event in line:
            return True
    return False

def prepare_data(dirname, input_lang=Lang("1"), pos_lang=Lang("2"), 
    char_lang=Lang("3"), rule_lang=Lang("4"), train=list(), valids=None, n_sample=0):
    raw_train = dict()
    if valids:
        vj = json.load(open(valids))
    else:
        vj = dict()
    with open("rules/rules.json") as f:
        rules = json.load(f)
    maxl = 0
    for fname in glob.glob(os.path.join(dirname, '*.a1')):
        root = os.path.splitext(fname)[0]
        name = os.path.basename(root)
        txt = root + '.txt'
        a1 = root + '.a1'
        a2 = root + '.a2'
        entities = dict()
        with open(a1) as f:
            for line in f:
                line = line.strip()
                if line.startswith('T'):
                    [id, data, text] = line.split('\t') 
                    [label, start, end] = data.split(' ')
                    entities[id] = (label, int(start), int(end), text)
        phosphorylations = []
        with open(a2) as f:
            for line in f:
                line = line.strip()
                if line.startswith('T'):
                    [id, data, text] = line.split('\t') 
                    [label, start, end] = data.split(' ')
                    entities[id] = (label, int(start), int(end), text)
                if line.startswith('E') and check_events(line):
                    [id, data] = line.split('\t')
                    temp = data.split(' ')
                    [tlbl, trigger] = temp[0].split(':')
                    [elbl, entity] = temp[1].split(':')
                    if valids and root+"/"+id in vj:
                        rule = (vj[root+"/"+id])
                    else:
                        rule = "null"
                    if ((tlbl, trigger, entity, rule)) not in phosphorylations:
                        phosphorylations.append((tlbl, trigger, entity, rule))
                    if tlbl not in input_lang.labels:
                        input_lang.label2id[tlbl] = len(input_lang.labels)
                        input_lang.labels.append(tlbl)
        with open(txt) as f:
            sentences = list()
            text = f.read()
            sentnece_count = 0
            for words, starts, ends in get_token_spans(text):
                sentnece_count += 1
                if len(words) > maxl:
                    maxl = len(words)
                s = int(starts[0])
                e = int(ends[-1])
                x = list(get_trigger(s, e, entities, phosphorylations))
                y = list(get_entity(s, e, entities, x))
                for entity in y:
                    words, starts, ends = replace_protein(words, entities, starts, ends, [entity])
                for res in x:
                    tlbl, trigger, entity, rule = res
                    if rule not in rules:
                        rule = []
                    else:
                        rule = rules[rule]
                    words, starts, ends = replace_protein(words, entities, starts, ends, [trigger, entity])
                temp = []
                temp_s = []
                temp_e = []
                for i,w in enumerate(words):
                    sw = sanitizeWord(w)
                    if sw:
                        temp.append(sw)
                        temp_s.append(starts[i])
                        temp_e.append(ends[i])
                words = temp
                starts = temp_s
                ends = temp_e
                triggers = [t[1] for t in x]
                for res in x:
                    tlbl, trigger, entity, rule = res
                    if rule in rules:
                        rule = rules[rule]
                        rule_lang.addSentence(rule)
                    else:
                        rule = []
                    try:
                        temp_w = words[:]
                        trigger_pos = (words.index("$"+trigger))
                        temp_w[trigger_pos] = sanitizeWord(entities[trigger][-1])
                        e_pos = words.index("$"+entity)
                        temp_w[e_pos] = "xTHEMEx"
                        for i, w in enumerate(temp_w):
                            if "$" in w: 
                                if w[1:] not in triggers:
                                    temp_w[i] = "xOTHERx"
                                else:
                                    if sanitizeWord(entities[w[1:]][-1]):
                                        temp_w[i] = sanitizeWord(entities[w[1:]][-1])
                                    else:
                                        temp_w[i] = entities[w[1:]][-1]
                        # temp_w = ["xOTHERx" if "$" in w and w[1:] not in triggers else w for w in temp_w]
                        pos = [i-e_pos for i in range(len(words))]
                        pos_lang.addSentence(pos)
                        input_lang.addSentence(temp_w)
                        if txt+entity not in raw_train:
                            raw_train[txt+entity] = (temp_w, entity, e_pos, [trigger_pos], [tlbl], pos, [rule])
                        else:
                            if (raw_train[txt+entity][3][0] == -1):
                                print (raw_train[txt+entity])
                                print ('/////')
                                exit()
                            raw_train[txt+entity][3].append(trigger_pos)
                            raw_train[txt+entity][4].append(tlbl)
                            raw_train[txt+entity][-1].append(rule)
                    except Exception as ex:
                        continue
                for entity in y:
                    try:
                        temp_w = words[:]
                        e_pos = words.index("$"+entity)
                        temp_w[e_pos] = "xTHEMEx"
                        for i, w in enumerate(temp_w):
                            if "$" in w: 
                                if w[1:] not in triggers:
                                    temp_w[i] = "xOTHERx"
                                else:
                                    if sanitizeWord(entities[w[1:]][-1]):
                                        temp_w[i] = sanitizeWord(entities[w[1:]][-1])
                                    else:
                                        temp_w[i] = entities[w[1:]][-1]
                        # temp_w = ["xOTHERx" if "$" in w else w for w in temp_w]
                        pos = [i-e_pos for i in range(len(words))]
                        pos_lang.addSentence(pos)
                        input_lang.addSentence(temp_w)
                        raw_train[txt+entity] = (temp_w, entity, e_pos, [-1], ["NoRel"], pos, [[]])
                    except:
                        continue
                        # print (words)
                        # print (new_words)
                        # print ([(p,entities[p]) for p in triggers+proteins])
    for i, w in input_lang.index2word.items():
        char_lang.addSentence(w)
    if n_sample:
        sample = random.sample(list(raw_train.values()), n_sample)
        return input_lang, pos_lang, char_lang, rule_lang, train+sample
    return input_lang, pos_lang, char_lang, rule_lang, train+list(raw_train.values())

def prepare_test_data(dirname, input_lang=Lang("1"), pos_lang=Lang("2"), 
    char_lang=Lang("3"), rule_lang=Lang("4")):
    raw_test = list()
    maxl = 0
    for fname in glob.glob(os.path.join(dirname, '*.a1')):
        root = os.path.splitext(fname)[0]
        name = os.path.basename(root)
        txt = root + '.txt'
        a1 = root + '.a1'
        open(root+".a2p", "w").close()
        entities = dict()
        lasteid = None
        with open(a1) as f:
            for line in f:
                line = line.strip()
                if line.startswith('T'):
                    [id, data, text] = line.split('\t') 
                    [label, start, end] = data.split(' ')
                    entities[id] = (label, int(start), int(end), text)
                    lasteid = id
        with open(txt) as f:
            sentences = list()
            text = f.read()
            sentnece_count = 0
            for words, starts, ends in get_token_spans(text):
                sentnece_count += 1
                if len(words) > maxl:
                    maxl = len(words)
                s = int(starts[0])
                e = int(ends[-1])
                y = list(get_entity(s, e, entities, []))
                for entity in y:
                    words, starts, ends = replace_protein(words, entities, starts, ends, [entity])
                temp = []
                temp_s = []
                temp_e = []
                for i,w in enumerate(words):
                    sw = sanitizeWord(w)
                    if sw:
                        temp.append(sw)
                        temp_s.append(starts[i])
                        temp_e.append(ends[i])
                words = temp
                starts = temp_s
                ends = temp_e
                for entity in y:
                    try:
                        temp_w = words[:]
                        e_pos = words.index("$"+entity)
                        temp_w[e_pos] = "xTHEMEx"
                        temp_w = ["xOTHERx" if "$" in w else w for w in temp_w]
                        pos = [i-e_pos for i in range(len(words))]
                        pos_lang.addSentence(pos)
                        input_lang.addSentence(temp_w)
                        raw_test.append((temp_w, entity, e_pos, pos, starts, ends, lasteid, root))
                    except:
                        continue
                        # print (words)
                        # print (new_words)
                        # print ([(p,entities[p]) for p in triggers+proteins])
    for i, w in input_lang.index2word.items():
        char_lang.addSentence(w)
    return input_lang, pos_lang, char_lang, rule_lang, raw_test

def parse_json_data():
    triggers = dict()
    valids = dict()
    with open("lo_events.json") as f:
        pubmed = json.load(f)
        i = 0
        for sentence in pubmed:
            i += 1
            with open("pubmed_loc/%d.txt"%i, "w") as txt:
                txt.write(sentence)
            j = 1
            k = len(pubmed[sentence].keys())+1
            l = 1
            for eid in pubmed[sentence]:
                entity = pubmed[sentence][eid]["entity"]
                with open("pubmed_loc/%d.a1"%i, "a") as a1:
                    a1.write("T%d\tProtein %d %d\t%s\n"%(j, entity[1][0], entity[1][1], entity[0]))
                for event in pubmed[sentence][eid]["events"]:
                    trigger = event["trigger"]
                    rule = event["rule"]
                    valids["pubmed_loc/%d/E%d"%(i, l)] = rule 
                    with open("pubmed_loc/%d.a2"%i, "a") as a2:
                        if "%s%d%d"%(trigger[0], trigger[1][0], trigger[1][1]) not in triggers:
                            triggers["%s%d%d"%(trigger[0], trigger[1][0], trigger[1][1])] = k
                            k += 1
                        a2.write("T%d\tLocalization %d %d\t%s\n"%(triggers["%s%d%d"%(trigger[0], trigger[1][0], trigger[1][1])], trigger[1][0], trigger[1][1], trigger[0]))
                        a2.write("E%d\tLocalization:T%d Theme:T%d\n"%(l, triggers["%s%d%d"%(trigger[0], trigger[1][0], trigger[1][1])], j))
                    l += 1
                j += 1
    print (json.dumps(valids))


# if __name__ == '__main__':
#     # parse_json_data()

#     input_lang = Lang("input")
#     pl1 = Lang("position")
#     char = Lang("char")
#     rule_lang = Lang("rule")
#     raw_train = list()
#     input_lang, pl1, char, rule_lang, raw_train = prepare_data("BioNLP-ST-2013_GE_train_data_rev3", input_lang, pl1, char, rule_lang, raw_train)
#     input_lang, pl1, char, rule_lang, raw_train = prepare_data("pubmed_loc", input_lang, pl1, char, rule_lang, raw_train, "valids_loc.json")
#     input_lang, pl1, char, rule_lang, raw_train = prepare_data("pubmed_ge", input_lang, pl1, char, rule_lang, raw_train, "valids_ge.json")
#     input_lang, pl1, char, rule_lang, raw_train = prepare_data("pubmed2", input_lang, pl1, char, rule_lang, raw_train, "valids2.json")
#     # # input_lang, pl1, char, rule_lang, raw_train2 = prepare_data("BioNLP-ST-2013_GE_devel_data_rev3")
#     # trainning_set = []
#     # i = j = 0
#     print (input_lang.label2id)
#     for datapoint in raw_train:
#         print ([input_lang.label2id[l] for l in datapoint[4]])
    #     if datapoint[3][0] != -1:
    #         i += len(datapoint[3])
    #         trainning_set.append(([input_lang.word2index[w] for w in datapoint[0]]+[1],
    #             datapoint[1],#entity
    #             datapoint[2],#entity position
    #             datapoint[3],#trigger position
    #             [input_lang.label2id[l] for l in datapoint[4]],#trigger label
    #             [pl1.word2index[p] for p in datapoint[5]]+[0],#positions
    #             [[char.word2index[c] for c in w] for w in datapoint[0]+["EOS"]],
    #             [[rule_lang.word2index[p] for p in rule + ["EOS"]] for rule in datapoint[6]]))
    #     else:
    #         j += 1
    #         trainning_set.append(([input_lang.word2index[w] for w in datapoint[0]]+[1],
    #             datapoint[1],
    #             datapoint[2],
    #             datapoint[3], [0], 
    #             [pl1.word2index[p] for p in datapoint[5]]+[0],
    #             [[char.word2index[c] for c in w] for w in datapoint[0]+["EOS"]],
    #             [rule_lang.word2index["EOS"]]))
    #         print (datapoint)
    # print(i,j)
        # if (t[0] != raw_train2[i][0] and t[1]==raw_train2[i][1]):
#             print (t[0])
#             print (raw_train2[i][0])
#             print (t[-1])
#             print (raw_train2[i][-1])
#             print (t[1], raw_train2[i][1], raw_train2[i][3])
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('datadir')
#     args = parser.parse_args()
#     input_lang, pos_lang, char_lang, train = prepare_data(args.datadir, "valids.json")
#     print (len(train))
#     input_lang, pos_lang, char_lang, rule_lang, train = parse_json_data(input_lang, pos_lang, char_lang, train)
#     # print (load_embeddings("embeddings_november_2016.txt", input_lang))
#     offset_dict = dict()
    # for t in train:
    #     if t[3] != -1:
    #         print (t)
    #         print (t[-1])
    #         print (len(t[0]), len(t[-1]))
    #         print (t[3])
    #         print (t[0][t[3]], t[4])
    #     if t[3] != -1:
    #         print (t[2], t[3], t[-1][t[3]])
    #         try:
    #             offset_dict[t[-1][t[3]]] += 1
    #         except KeyError:
    #             offset_dict[t[-1][t[3]]] = 1
    # with open("histogram.tsv", "w") as f:
    #     for i in range(min(offset_dict.keys()), max(offset_dict.keys())+1):
    #         l = offset_dict[i] if i in offset_dict else 0
    #         f.write("%d\t%d\n"%(i,l)) 









