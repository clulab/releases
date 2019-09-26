#!/usr/bin/env python

import numpy as np
from collections import defaultdict
import re
import io
import json, sys
from tqdm import tqdm
import mmap

class Datautils:




    def get_num_lines(file_path):

        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines

    ## read the data from the file with the entity_ids provided by entity_vocab and context_ids provided by context_vocab
    ## data format:
    #################
    ## [label]\t[Entity Mention]\t[[context mention1]\t[context mention2],\t...[]]
    ## NOTE: The label to be removed later from the dataset and the routine suitably adjusted. Inserted here for debugging

    @classmethod
    def read_data(cls, filename, entity_vocab, context_vocab):
        labels = []
        entities = []
        contexts = []

        with open(filename) as f:
            word_counts = dict()
            for line in tqdm(f, total= get_num_lines(filename)):
                vals = line.strip().split('\t')
                labels.append(vals[0].strip())
                if vals[1] not in word_counts:
                    word_counts[vals[1]] = 1
                else:
                    word_counts[vals[1]] += 1
                word_id = entity_vocab.get_id(vals[1])
                if word_id is not None:
                    entities.append(word_id)
                contexts.append([context_vocab.get_id(c) for c in vals[2:] if context_vocab.get_id(c) is not None])

            num_count_words = 0
            for word in word_counts:
                if word_counts[word] >= 6:
                    num_count_words+=1
            print('num count words:',num_count_words)

        # return np.array(entities), np.array([np.array(c) for c in contexts]), np.array(labels)
        #askajay why return as lists. why not a list of objects.
        # am assuming you will iterate through zipped lists of all these 3, but isn't that risky?
        return entities, contexts, labels

    @classmethod
    def read_rte_data(cls, filename,args):
        tr_len=args.truncate_words_length
        all_labels = []
        all_claims = []
        all_evidences = []

        with open(filename) as f:
            for index,line in enumerate(tqdm(f, total=cls.get_num_lines(filename))):
                multiple_ev = False
                x = json.loads(line)
                claim = x["claim"]
                evidences = x["sents"]
                label = x["label"]
                label = label.upper()
                evidences_this_list=[]
                evidences_this_str = ""
                if (len(evidences) > 1):
                    #some claims have more than one evidences. Join them all together.
                    multiple_ev=True
                    for e in evidences:
                        evidences_this_list.append(e)
                    evidences_this_str=" ".join(evidences_this_list)
                else:
                    evidences_this_str = "".join(evidences)


                ## truncate at n words. irrespective of claim or evidence truncate it at n...
                # Else it was overloading memory due to the packing/padding of all sentences into the longest size..
                # which was like 180k words or something

                claim_split=claim.split(" ")
                if(len(claim_split) > tr_len):
                    claim_tr=claim_split[:1000]
                    claim = " ".join(claim_tr)

                evidences_split = evidences_this_str.split(" ")
                if (len(evidences_split) > tr_len):
                    evidences_tr = evidences_split[:1000]
                    evidences_this_str=" ".join(evidences_tr)

                all_claims.append(claim)
                all_evidences.append(evidences_this_str)
                all_labels.append(label)

        return all_claims, all_evidences, all_labels


       # if the input data is NER neutered, replace PERSON-c1 with PERSONC1. This is vestigial. My code does fine, but sandeep said his code splits the tokens based on dashes.
    # so doing this to avoid that.

    def replace_if_PERSON_C1_format(sent, args):
        sent_replaced = ""
        regex = re.compile('([A-Z]+)(-)([ce])([0-99])')
        if (args.type_of_data == "ner_replaced" and regex.search(sent)):
            sent_replaced = regex.sub(r'\1\3\4', sent)
        else:
            sent_replaced = sent
        return sent_replaced

    @classmethod
    def read_data_where_evidences_are_strings(cls, filename, args):
        tr_len=args.truncate_words_length
        all_labels = []
        all_claims = []
        all_evidences = []

        with open(filename) as f:
            for index,line in enumerate(tqdm(f, total=cls.get_num_lines(filename))):
                multiple_ev = False
                x = json.loads(line)
                claim = x["claim"]
                evidences_this_str = x["evidence"]
                label = x["label"]
                label=label.upper()

                #if reading NER neutered data replace PERSON-C1 with PERSONC1 etc- this is to avoid splitting based on - during tokenization
                claim = cls.replace_if_PERSON_C1_format(claim, args)
                evidences_this_str = cls.replace_if_PERSON_C1_format(evidences_this_str, args)

                ## truncate at n words. irrespective of claim or evidence truncate it at n...
                # Else it was overloading memory due to the packing/padding of all sentences into the longest size..
                # which was like 180k words or something
                claim_split=claim.split(" ")
                if(len(claim_split) > tr_len):
                    claim_tr=claim_split[:1000]
                    claim = " ".join(claim_tr)

                evidences_split = evidences_this_str.split(" ")
                if (len(evidences_split) > tr_len):
                    evidences_tr = evidences_split[:1000]
                    evidences_this_str=" ".join(evidences_tr)

                all_claims.append(claim)
                all_evidences.append(evidences_this_str)
                all_labels.append(label)

        return all_claims, all_evidences, all_labels

    @classmethod
    def read_re_data(cls, filename, type, max_entity_len, max_inbetween_len, train_labels):
        labels = []
        entities1 = []
        entities2 = []
        chunks_inbetween = []
        word_counts = dict()
        oov_label = []
        # sanitized_file = '/'.join(filename.split('/')[:-1]) + '/../' + type + '_sanitized_file.txt'
        # with io.open(sanitized_file, 'w', encoding='utf8') as ff:
        with open(filename) as f:
            for line_id, line in enumerate(f):
                vals = line.strip().split('\t')

                label = vals[4]
                if type is 'test' and label not in train_labels:
                    oov_label.append(line_id)
                    continue

                sentence_str = ' ' + vals[5].strip()
                sentence_str = sentence_str.replace('###END###', '')

                entity1 = vals[2].strip()
                entity2 = vals[3].strip()

                entity1_pattern = entity1.replace('(', "\(")
                entity1_pattern = entity1_pattern.replace(')', "\)")
                entity1_pattern = entity1_pattern.replace('[', "\[")
                entity1_pattern = entity1_pattern.replace(']', "\]")
                entity1_pattern = entity1_pattern.replace('{', "\{")
                entity1_pattern = entity1_pattern.replace('}', "\}")
                entity1_pattern = entity1_pattern.replace('"', '\\"')

                entity2_pattern = entity2.replace('(', "\(")
                entity2_pattern = entity2_pattern.replace(')', "\)")
                entity2_pattern = entity2_pattern.replace('[', "\[")
                entity2_pattern = entity2_pattern.replace(']', "\]")
                entity2_pattern = entity2_pattern.replace('{', "\{")
                entity2_pattern = entity2_pattern.replace('}', "\}")
                entity2_pattern = entity2_pattern.replace('"', '\\"')

                entity1_idxs = [m.start() for m in re.finditer(' ' + entity1_pattern + ' ', sentence_str)]
                entity2_idxs = [m.start() for m in re.finditer(' ' + entity2_pattern + ' ', sentence_str)]

                if len(entity1_idxs) == 0:
                    entity1_idxs = [m.start() for m in re.finditer(entity1_pattern, sentence_str)]

                if len(entity2_idxs) == 0:
                    entity2_idxs = [m.start() for m in re.finditer(entity2_pattern, sentence_str)]

                # this happens when not all words of entity are all connected by '_' in sentence
                # e.g.: m.03h64	m.01ky9c	hong_kong	hong_kong_international_airport	/location/location/contains	turbo jet ferries depart from the hong_kong macao ferry terminal , sheung wan , the hong_kong china ferry terminal in kowloon and cross boundary passenger ferry terminal at hong_kong international airport . ###END###
                if len(entity1_idxs) == 0:
                    sentence_str_tab = sentence_str.replace(' ', "_")
                    entity1_idxs = [m.start() for m in re.finditer('_' + entity1_pattern + '_', sentence_str_tab)]
                if len(entity2_idxs) == 0:
                    sentence_str_tab = sentence_str.replace(' ', "_")
                    entity2_idxs = [m.start() for m in re.finditer('_' + entity2_pattern + '_', sentence_str_tab)]

                if len(entity1_idxs) > 0 and len(entity2_idxs) > 0:
                    # initial the shortest distance between two entities as some big num, such as 2000
                    d_abs = 2000   #todo: replace with constant max of system

                    entity1_idx = entity1_idxs[0]  # entity can appear more than once in sentence
                    entity2_idx = entity2_idxs[0]
                    for idx1 in entity1_idxs:
                        for idx2 in entity2_idxs:
                            if abs(idx1-idx2) < d_abs and idx1 != idx2:
                                d_abs = abs(idx1-idx2)
                                entity1_idx = idx1
                                entity2_idx = idx2

                    if entity1_idx < entity2_idx:
                        sentence_str_1 = sentence_str[:entity1_idx] + ' @entity ' + sentence_str[entity1_idx+len(entity1)+2:entity2_idx]
                        sentence_str_2 = ' @entity ' + sentence_str[entity2_idx+len(entity2)+2:]
                        sentence_str = sentence_str_1 + ' ' + sentence_str_2

                    elif entity1_idx > entity2_idx:
                        sentence_str_1 = sentence_str[:entity2_idx] + ' @entity ' + sentence_str[entity2_idx + len(entity2)+2:entity1_idx]
                        sentence_str_2 = ' @entity ' + sentence_str[entity1_idx+len(entity1)+2:]
                        sentence_str = sentence_str_1 + ' ' + sentence_str_2

                    sentence_str = sentence_str.lower()
                    sentence_str = sentence_str.replace('-lrb-', " ( ")
                    sentence_str = sentence_str.replace('-rrb-', " ) ")
                    sentence_str = sentence_str.replace('-lsb-', " [ ")
                    sentence_str = sentence_str.replace('-rsb-', " ] ")

                    sentence_str = ' '.join(sentence_str.split())
                    sentence_words = re.split(r'(\\n| |#|%|\'|\"|,|:|-|_|;|!|=|\(|\)|\$|\?|\*|\+|\]|\[|\{|\}|\\|\||\<|\>|\^|\`|\~)',sentence_str)
                    inbetween_str = sentence_str.partition("@entity")[2].partition("@entity")[0]
                    inbetween_words = re.split(r'(\\n| |#|%|\'|\"|,|:|-|_|;|!|=|\(|\)|\$|\?|\*|\+|\]|\[|\{|\}|\\|\||\<|\>|\^|\`|\~)',inbetween_str)

                    entity1 = entity1.lower()
                    entity1 = entity1.replace('-lrb-', " ( ")
                    entity1 = entity1.replace('-rrb-', " ) ")
                    entity1 = entity1.replace('-lsb-', " [ ")
                    entity1 = entity1.replace('-rsb-', " ] ")
                    entity1 = ' '.join(entity1.split())

                    entity2 = entity2.lower()
                    entity2 = entity2.replace('-lrb-', " ( ")
                    entity2 = entity2.replace('-rrb-', " ) ")
                    entity2 = entity2.replace('-lsb-', " [ ")
                    entity2 = entity2.replace('-rsb-', " ] ")
                    entity2 = ' '.join(entity2.split())
                    # entities1_words = entity1.strip().split('_')
                    # entities2_words = entity2.strip().split('_')
                    entities1_words = re.split(r'(\\n| |#|%|\'|\"|,|:|-|_|;|!|=|\(|\)|\$|\?|\*|\+|\]|\[|\{|\}|\\|\||\<|\>|\^|\`|\~)',entity1)
                    entities2_words = re.split(r'(\\n| |#|%|\'|\"|,|:|-|_|;|!|=|\(|\)|\$|\?|\*|\+|\]|\[|\{|\}|\\|\||\<|\>|\^|\`|\~)',entity2)

                    i = 0
                    while i < len(sentence_words):
                        word = sentence_words[i]

                        if len(word) == 0 or word is ' ':
                            sentence_words.remove(word)
                            i -= 1
                        elif word[0] is not '@' and '@' in word:
                            sentence_words[i] = '@email'
                        elif word.startswith("http") or word.startswith("www") or ".com" in word or ".org" in word:
                            sentence_words[i] = '@web'
                        elif len(word) > 1 and word[-1] is '.':
                            sentence_words[i] = word[:-1]
                        elif any(char.isdigit() for char in word):
                            sentence_words[i] = 'xnumx'
                        i += 1

                    i = 0
                    while i < len(inbetween_words):
                        word = inbetween_words[i]

                        if len(word) == 0 or word is ' ':
                            inbetween_words.remove(word)
                            i -= 1
                        elif word[0] is not '@' and '@' in word:
                            inbetween_words[i] = '@email'
                        elif word.startswith("http") or word.startswith("www") or ".com" in word or ".org" in word:
                            inbetween_words[i] = '@web'
                        elif len(word) > 1 and word[-1] is '.':
                            inbetween_words[i] = word[:-1]
                        elif any(char.isdigit() for char in word):
                            inbetween_words[i] = 'xnumx'
                        i += 1

                    i = 0
                    while i < len(entities1_words):
                        word = entities1_words[i]

                        if len(word) == 0 or word is ' ' or word is '_':
                            entities1_words.remove(word)
                            i -= 1
                        elif word[0] is not '@' and '@' in word:
                            entities1_words[i] = '@email'
                        elif word.startswith("http") or word.startswith("www") or ".com" in word or ".org" in word:
                            entities1_words[i] = '@web'
                        elif len(word) > 1 and word[-1] is '.':
                            entities1_words[i] = word[:-1]
                        elif any(char.isdigit() for char in word):
                            entities1_words[i] = 'xnumx'
                        i += 1

                    i = 0
                    while i < len(entities2_words):
                        word = entities2_words[i]

                        if len(word) == 0 or word is ' ' or word is '_':
                            entities2_words.remove(word)
                            i -= 1
                        elif word[0] is not '@' and '@' in word:
                            entities2_words[i] = '@email'
                        elif word.startswith("http") or word.startswith("www") or ".com" in word or ".org" in word:
                            entities2_words[i] = '@web'
                        elif len(word) > 1 and word[-1] is '.':
                            entities2_words[i] = word[:-1]
                        elif any(char.isdigit() for char in word):
                            entities2_words[i] = 'xnumx'
                        i += 1

                    if len(inbetween_words) <= 2*max_inbetween_len or type is not 'train':   # throw away sentences with too many inbetween words

                        labels.append(label)

                        if len(entities1_words) > max_entity_len:
                            entities1_words = entities1_words[:max_entity_len]
                        if len(entities2_words) > max_entity_len:
                            entities2_words = entities2_words[:max_entity_len]

                        # ff.write(vals[0].strip())
                        # ff.write('\t')
                        # ff.write(vals[1].strip())
                        # ff.write('\t')

                        for word in inbetween_words:
                            if word not in word_counts:
                                word_counts[word] = 1
                            else:
                                word_counts[word] += 1

                        for word in entities1_words:
                            # ff.write(word + ' ')
                            if word not in word_counts:
                                word_counts[word] = 1
                            else:
                                word_counts[word] += 1

                        # ff.write('\t')

                        for word in entities2_words:
                            # ff.write(word + ' ')
                            if word not in word_counts:
                                word_counts[word] = 1
                            else:
                                word_counts[word] += 1
                        #
                        # ff.write('\t')
                        # ff.write(vals[4].strip())
                        # ff.write('\t')
                        # for word in sentence_words:
                        #     ff.write(word + ' ')
                        # ff.write('\n')

                        entities1.append(entities1_words)
                        entities2.append(entities2_words)
                        chunks_inbetween.append(inbetween_words)

                else:
                    assert False, line

        if type is 'test' and len(oov_label) > 0:
            print('Number of test datapoints thrown away because of its label did not seen in train:' + str(len(oov_label)))

        return entities1, entities2, labels, chunks_inbetween, word_counts, oov_label



    ## Takes as input an array of entity mentions(ids) along with their contexts(ids) and converts them to individual pairs of entity and context
    ## Entity_Mention_1  -- context_mention_1, context_mention_2, ...
    ## ==>
    ## Entity_Mention_1 context_mention_1 0 ## Note the last number is the mention id, needed later to associate entity mention with all its contexts
    ## Entity_Mention_1 context_mention_2 1
    ## ....

    @classmethod
    def read_re_data_syntax(cls, filename, type, max_entity_len, max_syntax_len, train_labels, labels_set):
        labels = []
        entities1 = []
        entities2 = []
        chunks_inbetween = []
        word_counts = dict()
        oov_label = []

        with open(filename) as f:
            for line_id, line in enumerate(f):
                vals = line.strip().split('\t')

                label = vals[4]
                if len(labels_set) > 0 and label not in labels_set:
                    label = 'NA'

                if type is 'test' and label not in train_labels:
                    oov_label.append(line_id)
                    continue

                if len(vals) > 5 :
                    syntax_str = vals[5].strip()
                    syntax_str = syntax_str.replace('###END###', '')

                    entity1 = vals[2].strip()
                    entity2 = vals[3].strip()

                    syntax_str = syntax_str.lower()
                    syntax_str = syntax_str.replace('-lrb-', " ( ")
                    syntax_str = syntax_str.replace('-rrb-', " ) ")
                    syntax_str = syntax_str.replace('-lsb-', " [ ")
                    syntax_str = syntax_str.replace('-rsb-', " ] ")

                    syntax_str = ' '.join(syntax_str.split())

                else:
                    syntax_str = ''

                syntax_tokens = re.split(r'(\\n| |#|%|\'|\"|,|:|-|_|;|!|=|\(|\)|\$|\?|\*|\+|\]|\[|\{|\}|\\|\||\^|\`|\~)', syntax_str)

                entity1 = entity1.lower()
                entity1 = entity1.replace('-lrb-', " ( ")
                entity1 = entity1.replace('-rrb-', " ) ")
                entity1 = entity1.replace('-lsb-', " [ ")
                entity1 = entity1.replace('-rsb-', " ] ")
                entity1 = ' '.join(entity1.split())

                entity2 = entity2.lower()
                entity2 = entity2.replace('-lrb-', " ( ")
                entity2 = entity2.replace('-rrb-', " ) ")
                entity2 = entity2.replace('-lsb-', " [ ")
                entity2 = entity2.replace('-rsb-', " ] ")
                entity2 = ' '.join(entity2.split())
                # entities1_words = entity1.strip().split('_')
                # entities2_words = entity2.strip().split('_')
                entities1_words = re.split(r'(\\n| |#|%|\'|\"|,|:|-|_|;|!|=|\(|\)|\$|\?|\*|\+|\]|\[|\{|\}|\\|\||\<|\>|\^|\`|\~)', entity1)
                entities2_words = re.split(r'(\\n| |#|%|\'|\"|,|:|-|_|;|!|=|\(|\)|\$|\?|\*|\+|\]|\[|\{|\}|\\|\||\<|\>|\^|\`|\~)', entity2)


                i = 0
                while i < len(syntax_tokens):
                    word = syntax_tokens[i]

                    if len(word) == 0 or word is ' ':
                        syntax_tokens.remove(word)
                        i -= 1
                    elif word[0] is not '@' and '@' in word:
                        syntax_tokens[i] = '@email'
                    elif word.startswith("http") or word.startswith("www") or ".com" in word or ".org" in word:
                        syntax_tokens[i] = '@web'
                    elif len(word) > 1 and word[-1] is '.':
                        syntax_tokens[i] = word[:-1]
                    elif any(char.isdigit() for char in word):
                        syntax_tokens[i] = 'xnumx'
                    i += 1

                i = 0
                while i < len(entities1_words):
                    word = entities1_words[i]

                    if len(word) == 0 or word is ' ' or word is '_':
                        entities1_words.remove(word)
                        i -= 1
                    elif word[0] is not '@' and '@' in word:
                        entities1_words[i] = '@email'
                    elif word.startswith("http") or word.startswith("www") or ".com" in word or ".org" in word:
                        entities1_words[i] = '@web'
                    elif len(word) > 1 and word[-1] is '.':
                        entities1_words[i] = word[:-1]
                    elif any(char.isdigit() for char in word):
                        entities1_words[i] = 'xnumx'
                    i += 1

                i = 0
                while i < len(entities2_words):
                    word = entities2_words[i]

                    if len(word) == 0 or word is ' ' or word is '_':
                        entities2_words.remove(word)
                        i -= 1
                    elif word[0] is not '@' and '@' in word:
                        entities2_words[i] = '@email'
                    elif word.startswith("http") or word.startswith("www") or ".com" in word or ".org" in word:
                        entities2_words[i] = '@web'
                    elif len(word) > 1 and word[-1] is '.':
                        entities2_words[i] = word[:-1]
                    elif any(char.isdigit() for char in word):
                        entities2_words[i] = 'xnumx'
                    i += 1

                if len(syntax_tokens) <= 2*max_syntax_len or type is not 'train':   # when max_inbetween_len = 60, filter out 2464 noise

                    labels.append(label)
                    if len(entities1_words) > max_entity_len:
                        entities1_words = entities1_words[:max_entity_len]
                    if len(entities2_words) > max_entity_len:
                        entities2_words = entities2_words[:max_entity_len]

                    for word in syntax_tokens:
                        if word not in word_counts:
                            word_counts[word] = 1
                        else:
                            word_counts[word] += 1

                    for word in entities1_words:
                        if word not in word_counts:
                            word_counts[word] = 1
                        else:
                            word_counts[word] += 1

                    for word in entities2_words:
                        if word not in word_counts:
                            word_counts[word] = 1
                        else:
                            word_counts[word] += 1

                    entities1.append(entities1_words)
                    entities2.append(entities2_words)
                    chunks_inbetween.append(syntax_tokens)

        if type is 'test' and len(oov_label) > 0:
            print('Number of test datapoints thrown away because of its label did not seen in train:' + str(len(oov_label)))

        return entities1, entities2, labels, chunks_inbetween, word_counts, oov_label

    @classmethod
    def prepare_for_skipgram(cls, entities, contexts):

        entity_ids = []
        context_ids = []
        mention_ids = []
        for i in range(len(entities)):
            word = entities[i]
            context = contexts[i]
            for c in context:
                entity_ids.append(word)
                context_ids.append(c)
                mention_ids.append(i)
        return np.array(entity_ids), np.array(context_ids), np.array(mention_ids)


    ## NOTE: To understand the current negative sampling and replace it with a more simpler version. Keeping it as it is now.
    @classmethod
    def collect_negatives(cls, entities_for_sg, contexts_for_sg, entity_vocab, context_vocab):

        n_entities = entity_vocab.size()
        n_contexts = context_vocab.size()
        negatives = np.empty((n_entities, n_contexts))

        for i in range(n_entities):
            negatives[i,:] = np.arange(n_contexts)
        negatives[entities_for_sg, contexts_for_sg] = 0
        return negatives

    @classmethod
    def construct_indices(cls, mentions, contexts):

        entityToPatternsIdx = defaultdict(set)
        for men, ctxs in zip(list(mentions), list([list(c) for c in contexts])):
            for ctx in ctxs:
                tmp = entityToPatternsIdx[men]
                tmp.add(ctx)
                entityToPatternsIdx[men] = tmp

        patternToEntitiesIdx = defaultdict(set)
        for men, ctxs in zip(list(mentions), list([list(c) for c in contexts])):
            for ctx in ctxs:
                tmp = patternToEntitiesIdx[ctx]
                tmp.add(men)
                patternToEntitiesIdx[ctx] = tmp

        return entityToPatternsIdx, patternToEntitiesIdx
