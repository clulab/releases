# Author: Fan luo
#!/usr/bin/env python
# coding: utf-8 
 
import json
import logging
import random 
import os 
import pandas as pd 
import numpy as np
from typing import List, Dict
import re
import string
from pprint import pprint

from utils import *

from prettytable import PrettyTable
import ujson 
# import numpy
import spacy    
print(spacy.__version__)
import en_core_web_lg     
nlp = en_core_web_lg.load()  

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
 
import networkx as nx
from networkx.readwrite import json_graph
import itertools 
from more_itertools import pairwise
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from fuzzywuzzy import utils
from fuzzysearch import find_near_matches
import copy
import inflect

import json
import codecs

from networkx.algorithms import approximation as approx
 
reproducibility()
p = inflect.engine()
 
json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
json_path = 'crf_rsltn.json'   # cofre resolved
crf = json_load(json_path)

   

##### noun phrases extraction 
 
# consider noun_chunks that are substring of entity, execpt DATE, CARDINAL, ORDINAL entities
# - Eg: entity: `Louisville Cardinals of the Big East` type:  ORG
# 
# keeps noun_chunks: `Louisville Cardinals`, `Big East`
# 
# But do not keep chunk `January` from entity `January 2 2012`
# 
# - noun_chunks contains local entity keep rest and the chunk,
# 
# eg: 2013 sugar bowl, third single 
 
Local_NE_Types = [ 'NORP', 'GPE', 'DATE', 'LANGUAGE',  'CARDINAL', 'PERCENT', 'LOC', 'QUANTITY','ORDINAL'] 

def nphrases(text, known_phrases, prefix):   # prefix is same for text from same para, avoide merge nodes across paras

    # quote phrases    
    text = text.replace( "'s ", " ")
    text = re.sub(r'\" \"', '\"', text)  # replace " " with ", such as " " When in Rome " " ( 2010 )
    text = re.sub(r'\' \'', '\'', text)  # replace ' ' with '
    
    # replace double quote first: "A Lot About Livin' (And a Little 'bout Love)" 
    phrases_in_double_quotes = re.findall(r"\"(.*?)\"", text, re.DOTALL)
    phrases_in_double_quotes += re.findall(r"“(.*?)”", text, re.DOTALL)
    for i, phrase in enumerate(phrases_in_double_quotes):
        matches = find_near_matches('\"' + phrase.lower() + '\"', text.lower(), max_l_dist=0)
        matches += find_near_matches('“' + phrase.lower() + '“', text.lower(), max_l_dist=1)  
        matched_phrases = [text[m.start:m.end] for m in matches if m.matched!='']
        for matched_phrase in matched_phrases:
            text = text.replace(matched_phrase, " xxx ") 
    
    phrases_in_single_quotes = re.findall(r"'([^s].*?)'", text, re.DOTALL)   # [^s] do not match 's
    phrases_in_single_quotes += re.findall(r"’([^s].*?)’", text, re.DOTALL)
    for i, phrase in enumerate(phrases_in_single_quotes):
        matches = find_near_matches('\'' + phrase.lower() + '\'', text.lower(), max_l_dist=0)
        matches += find_near_matches('’' + phrase.lower() + '’', text.lower(), max_l_dist=0) 
        matched_phrases = [text[m.start:m.end] for m in matches if m.matched!='']
        for matched_phrase in matched_phrases:
            text = text.replace(matched_phrase, " xxx ") 
     
            
    phrases_in_quotes = phrases_in_single_quotes + phrases_in_double_quotes
    phrases_in_quotes = [lower(basic_normalize(phrase)) for phrase in phrases_in_quotes]
    phrases_in_quotes = [phrase for phrase in phrases_in_quotes if phrase!='' and phrase!='xxx']
             
    # ground to title phrase 
    # 2014 S/S is not recognized if Do not ground question phrases to titles
    if prefix == 'question' or prefix == 'answer':    
        # only ground to title phrase that exact matches substring in question, avoid expend phrase, such as 'album' in question to title '÷ (album)'
        phrases = [] 
        for phrase in known_phrases:  
            matches = find_match_with_tolerance(phrase, text, 0 ) 
            matched_phrases = [text[m.start:m.end] for m in matches if len(m.matched) > 2] 
            if len(matched_phrases) > 0:
                phrases.append(phrase)
        text, phrases = replace_phrases_in_text(text, phrases)
        for phrase in known_phrases:  
            matches = find_match_with_tolerance(phrase, text, 1 ) 
            matched_phrases = [text[m.start:m.end] for m in matches if len(m.matched) > 2] 
            if len(matched_phrases) > 0:
                phrases.append(phrase)
        text, phrases = replace_phrases_in_text(text, phrases)
        
    else:
        # spans that match any known phrases 
        phrases = inclusions(text, known_phrases)      # phrase'timofey samsonov' can be extracted from 'Timofey Petrovich Samsonov'  
        phrases = [remove_prefix(remove_prefix(phrase, 'question '), 'answer') for phrase in phrases]

        text, phrases = replace_phrases_in_text(text, phrases)
     
     
    # Named entities
    text = basic_normalize(text)
    doc = nlp(text) # 'New York' relaced with ' xxx ', so that chunk 'New York city' not be extracted 
    entity_phrases = []
    entity_phrases_tuples = []
    for entity in doc.ents:  
        label = entity.label_ 
        entity, e_start, e_end = phrase_cleaner(entity)
        if entity == None or entity.lower() in phrases + phrases_in_quotes:
            continue 
        entity_to_add = entity.lower()
        if(label in Local_NE_Types):  # and entity not in known_phrases):  known_phrases already be replaced  
            #  prefix 'question' is for nphrases extraction, would be removed before find_common_mapping
            # 'New York' in the question, add an adidtional edge from context '[p] New York' to it
            entity_to_add = prefix + ' ' + entity.lower()  
        elif NN_check(entity):
            entity_to_add = prefix + ' ' + entity.lower() 
        if entity_to_add not in entity_phrases:
            entity_phrases.append(entity_to_add)
            entity_phrases_tuples.append((entity.lower(), e_start, e_end, label))
 
    chunk_phrases = []
    chunk_phrases_tuples = []
    for chunk in doc.noun_chunks:
        chunk, c_start, c_end = phrase_cleaner(chunk)
        if(chunk):
            if all([chunk.lower() != phrase for phrase in [ent[0] for ent in entity_phrases_tuples if ent[3] not in Local_NE_Types]]) and all([chunk.lower() not in phrase for phrase in [ent[0] for ent in entity_phrases_tuples if ent[3] in Local_NE_Types]]):
                org_chunk = chunk.lower()
                
                if(len(chunk.split()) == 1 and (prefix == 'question' or prefix == 'answer')):  # single-word question noun
                    chunk = prefix + ' ' + chunk.lower()   # 'director' in the question, add an adidtional edge from context '[p] director' to it
                if(len(chunk.split()) > 1):      # multi-words context noun_phrase or question noun_phrase
                    chunk = chunk if chunk.startswith(prefix) else chunk.lower()  
                    rest_chunk = org_chunk
                    replaced_entities =[]
                    for i, (entity, e_start, e_end, _) in enumerate(entity_phrases_tuples):
                        if c_start <= e_start and c_end >= e_end: 
                            rest_chunk = rest_chunk.replace(entity, "").strip()
                            replaced_entities.append((entity_phrases[i], entity_phrases_tuples[i][0]))
                    if (len(replaced_entities) == 0 or (rest_chunk != org_chunk and any([replaced_entity[0].startswith(prefix) for replaced_entity in replaced_entities]))) and chunk.startswith(prefix)==False:
                        chunk = prefix + ' ' + chunk    # <prefix> New York city
                    
 
                    if rest_chunk != org_chunk: # and NN_check(rest_chunk) == False:      # '2012 game'
                    #  eg: 'american country music artist alan jackson album’
                        rest_chunk = basic_normalize(rest_chunk)
                        chunk_phrases.append(prefix + ' ' + rest_chunk) 
                        chunk_phrases_tuples.append((rest_chunk, c_start+1, c_end-1))
                    chunk_phrases.append(chunk) 
                elif(len(chunk.split()) == 1):        # single-word context noun_phrase
                    if NN_check(chunk):
                        chunk_phrases.append(prefix + ' ' + chunk.lower())  # such as 'group' and 'Michael'
                    else:
                        chunk_phrases.append(chunk.lower())   # such as 'cranberry'
                chunk_phrases_tuples.append((org_chunk, c_start, c_end))
 
    # remove entity 'Italian' when there is a chunk 'Italian physicist', Italian: NORP
    # Same for 'American actor'
    # however, 'only other Mexican' maintains, and not 'Mexican'
    nouns = entity_phrases_tuples + chunk_phrases_tuples 
    entities = []
    entities_tuples = []
    for i, (entity, e_start, e_end, _) in enumerate(entity_phrases_tuples):
        standalone_entity = True
        for another_noun_tuple in nouns:
            another_noun, n_start, n_end = another_noun_tuple[:3]
            if another_noun != entity and e_start >= n_start and e_end <= n_end:  # do not check  entity in another_noun directly, since singular_phrase('new york police') = new york polouse, which  is not in 'former new york police detective'
                standalone_entity = False
                break
        if standalone_entity: 
            entities.append(entity_phrases[i])
            entities_tuples.append(entity_phrases_tuples[i])
   
            
    return list( [(quote_p, "quote") for quote_p in set(phrases_in_quotes)] + 
                 [(ground_p, "ground") for ground_p in set(phrases)] +
                 [(entity, "entity", e_start, e_end) for entity, (_, e_start, e_end, _) in zip(entities, entities_tuples)] +
                 [(chunk, "chunk", c_start, c_end) for chunk, (_, c_start, c_end) in zip(chunk_phrases, chunk_phrases_tuples)] )

 

def noun_phrases_extraction(question, sentences, titles, answer):
    # add substring to titles_phrases_to_ground
    # ground question and answer to all titles , ground para_phrases to  all titles and question phrases, with source of phrases, keep qa phrases positions to be used in find_common_mapping
    
    # for each title containiing ( or ): (1) remove ( or )    (2) split title at ( and ), get first and multi-word following segments
    titles_phrases = [split_many(title, ['(', ')', ':', ',']) + [basic_normalize(''.join(ch for ch in title if ch not in ['(', ')', ':', ',']).lower())] for title in titles] 
     # remove dup, keep order
    titles_phrases = [[phrase for i, phrase in enumerate(phrases) if phrase not in phrases[:i]] for phrases in titles_phrases]  
    titles_phrases_to_ground = set(flatten(titles_phrases))
    
    labels_strip = ['CARDINAL', 'ORDINAL', 'DATE']
    for t_phrase in titles_phrases_to_ground.copy():
        normalized_t = basic_normalize(t_phrase)
        for entity in nlp(', '.join(normalized_t.split())).ents:  # connect with , otherwise some entity could not be recognized: '2012 louisville cardinals football team'
            if entity.label_ in labels_strip and normalized_t.startswith(entity.text):
                normalized_t = normalized_t[len(entity.text):]
            elif entity.label_ in labels_strip and normalized_t.endswith(entity.text):
                normalized_t = normalized_t[:-len(entity.text)]
        normalized_t = normalized_t.strip()
        if(len(normalized_t) > 0 and NN_check(normalized_t)==False):
            titles_phrases_to_ground.add(normalized_t) 

    question_phrases = nphrases(question, titles_phrases_to_ground, 'question')
    question_phrases_text = list(set([phrase[0] for phrase in question_phrases])) 
    answer_phrases =  nphrases(answer, titles_phrases_to_ground, 'answer')  
    answer_phrases_text =  list(set([phrase[0] for phrase in answer_phrases])) 
    
    paras_phrases = [] 
    sentences_coref_resolved = sentences.copy()
    to_match = list(titles_phrases_to_ground | set(question_phrases_text))  
    for i, title in enumerate(titles):
        if title in crf:      # if coreference exists 
            sentences_coref_resolved[i] = [' '.join(s_tokens) for s_tokens in crf[title]]
            
        para_phrases = [ [(titles_phrase, 'title') for titles_phrase in titles_phrases[i]] ]         # para_phrases[0] is the title phrases
        for sent in sentences_coref_resolved[i]:
            sent_phrases =  nphrases(sent, to_match, '[P'+ str(i)+']') 
            para_phrases.append(sent_phrases)                                
        paras_phrases.append(para_phrases)      
         
    question_phrases_text = [remove_prefix(s, 'question ') for s in question_phrases_text]
    answer_phrases_text = [remove_prefix(s, 'answer ') for s in answer_phrases_text]

    if(len(answer_phrases_text) == 0):
        answer_phrase, _, _  = phrase_cleaner(nlp(normalize_text(answer)))
        if answer_phrase: 
            answer_phrases_text = [answer_phrase]
        else:
            answer_phrases_text = [answer]
    

    return titles_phrases, question_phrases, question_phrases_text, paras_phrases, answer_phrases, answer_phrases_text
 

#### Graph Creation
# - add edge between nodes that are inclusive to each other without prefix
# - edges between any pair of nodes in one sentence , except single NN phrases, but not those match with any title phrases
# - As NN_phrases, noun_chunk that does not contain any entity directly connect to title, no full conntections with other phrases
# - connect title to most similar specific_entity, if no specific_entity, connect to the most similar one in each sentence 
# - if more than one title phrases per para, connect first to each sentence, with source of phrases 
def create_para_graph(paras_phrases):
    G = nx.Graph()     
    titles_phrases = list(flatten([para_phrases[0] for para_phrases in paras_phrases]))[::2]
    for pi, para_phrases in enumerate(paras_phrases):        # each para 
        title_phrases = [phrase[0] for phrase in para_phrases[0]]
        
        # add edge between inclusive phrases from same para: '[P6] louisville' and 'louisville cardinals of big east'
        flatten_para_phrases = list(flatten([[phrase[0] for phrase in s_p] for s_p in para_phrases]))
        noprefix_flatten_para_phrases = [remove_prefix(phrase, '[P', paraid = True) for phrase in flatten_para_phrases]
        for i, noprefix_phrase in enumerate(noprefix_flatten_para_phrases):
            phrase = flatten_para_phrases[i]
            noprefix_inclusion_phrases = inclusions(noprefix_phrase, noprefix_flatten_para_phrases)
            for noprefix_inclusion_phrase in noprefix_inclusion_phrases:
                inclusion_phrase = flatten_para_phrases[noprefix_flatten_para_phrases.index(noprefix_inclusion_phrase)]
                if phrase == inclusion_phrase:
                    continue 
                if(G.has_edge(inclusion_phrase, phrase) == False):
                    G.add_edge(inclusion_phrase, phrase, src = {(pi,)})

        # edges between any pair of nodes in one sentence , except single NN phrases that do not match with any title phrases
        for si, sent_phrases in enumerate(para_phrases):     # each sent
            # complete graph for each sent, including title_phrases
            if(len(sent_phrases) > 0): 
                sent_phrases_text = [phrase[0] for phrase in sent_phrases]
                
                if si == 0:
                    for node1, node2 in itertools.combinations(sent_phrases_text, 2):
                        if(node1 != node2 ):
                            if(G.has_edge(node1, node2)):
                                G[node1][node2]['src'].add((pi, 'title'))
                            else:
                                G.add_edge(node1, node2, src = {(pi, 'title')})
                else:
                    NN_phrases = [phrase[0] for phrase in sent_phrases if len(sum((find_match_with_tolerance(remove_prefix(phrase[0], '[P', paraid = True), s) for s in titles_phrases if len(s.split()) == len(remove_prefix(phrase[0], '[P', paraid = True).split())), [])) == 0 and NN_check(remove_stop_words(remove_prefix(phrase[0], '[P', paraid = True)))] #  a single NN unless it is a title phrase ‘delirium’,  use find_match_with_tolerance for typo, but for exact match, the len should be same to avoid 'film' which appears in many title phrases
                    no_entity_nounchunks = [phrase[0] for phrase in sent_phrases if phrase[1] == 'chunk' if len(nlp(phrase[0]).ents) == 0]
 

                    to_pair_connect = list(set(sent_phrases_text) - set(NN_phrases) -set(no_entity_nounchunks) - set(title_phrases))   # do not add additonal edges if sentence contains title phrases, such as 'tosca'
                    title_phrase =  title_phrases[0]
                    if len(to_pair_connect) == 1:           
                        phrase = to_pair_connect[0]
                        if(phrase != title_phrase):
                            if(G.has_edge(phrase, title_phrase)):
                                G[phrase][title_phrase]['src'].add((pi, 'title', si-1))
                            else:
                                G.add_edge(phrase, title_phrase, src = {(pi, 'title', si-1)})
                                
                    
                    for node1, node2 in itertools.combinations(to_pair_connect, 2):  # only applies when len(to_pair_connect) > 1
                        if(node1 != node2):
                            if(G.has_edge(node1, node2)):
                                G[node1][node2]['src'].add((pi, si-1))
                            else:
                                G.add_edge(node1, node2, src = {(pi, si-1)})
                               
                    
                    # add edge between the first title phrase to each single NN phrases
                    for phrase in NN_phrases + no_entity_nounchunks:   # such as '[P0] series'
                        if(phrase != title_phrase):
                            if(G.has_edge(phrase, title_phrase)):
                                G[phrase][title_phrase]['src'].add((pi, 'title', si-1))
                            else:
                                G.add_edge(phrase, title_phrase, src = {(pi, 'title', si-1)})
                    
                    # add edge between the first title phrase to most similar specific phrase, if none, connect with most similar phrase from the sentence
                    if title_phrase in sent_phrases_text.copy():
                        sent_phrases = [sent_phrase for sent_phrase in sent_phrases if(sent_phrase[0] != title_phrase)]
                        sent_phrases_text.remove(title_phrase)
    #                     print('title_phrase: ', title_phrase)
                    specific_entity_text = [phrase[0] for phrase in sent_phrases if phrase[1] in ["quote", "ground"] or specific_entity_check(phrase[0], titles_phrases)]
                    simi_phrase = ''
                    if(len(specific_entity_text) > 0):
                        simi_phrase, similarity = process.extractOne(title_phrase, specific_entity_text, scorer=fuzz.ratio)
                    elif(len(set(sent_phrases_text)-set(NN_phrases)-set(no_entity_nounchunks)) > 0):
                        simi_phrase, similarity = process.extractOne(title_phrase, list(set(sent_phrases_text)-set(NN_phrases)-set(no_entity_nounchunks)), scorer=fuzz.ratio)
 
                    if(simi_phrase != ''):
                        if(G.has_edge(title_phrase, simi_phrase)):
                            G[simi_phrase][title_phrase]['src'].add((pi, 'title', si-1))
                        else:
                            G.add_edge(simi_phrase, title_phrase, src = {(pi, 'title', si-1)}) 
 
    return G


 
#  subsgraphs have inclusion match with an question_phrase after remove prefix, with similar_in_length
# 'acidic bog' in paras_phrases VS 'bog' in question 
def create_relevant_graph(paras_phrases, question_phrases_text):
    G = create_para_graph(paras_phrases) #, question_phrases_text)
    Subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    RG = nx.Graph()    # relevant components:  components contain at least one question_phrase   
    for S in Subgraphs:
        s_nodes = list(S.nodes)
        s_nodes = [remove_prefix(node, '[P', paraid = True) for node in s_nodes]
        for phrase in question_phrases_text: 
            inclusive_phrases = inclusions(phrase, s_nodes)  # inclusions('bog', s_nodes): ['acidic bog', 'featherbed bog', 'Blanket bog']
            if len(inclusive_phrases) > 0:
                inclusive_phrases = set([inclusive_phrase for inclusive_phrase in inclusive_phrases if similar_in_length(phrase, inclusive_phrase) and inclusive_phrase])
                if len(inclusive_phrases) > 0: 
                    RG = nx.compose(RG, S)  # joint the relevant components 
                    break
        
    return RG
 
    
    
# add question node when has inclusion match, can ground to more than one question_phrase, if remove prefix has exact match or typo match, add edge directly, remove stopword in question_phrases_text, when inclusions to a single NN question phrase, instead of adding direct edge between node phrase and the included question phrase, add indirect edges via [Pi] single_NN; for single NN node, only ground to remove prefix. When node is a Date entity phrase, only match with exact match after remove prefix  
# not allow {'[P8] january 2': 'january 2 2012'}
def find_common_mapping(G, question_phrases, paras_phrases):   
    # fuzzy macth for common phrases: map pharse similar to question phrases to question phrase, then find common phrases
    
    flatten_paras_phrases = [ list(flatten([[phrase[0] for phrase in s_p] for s_p in para_p]))  for para_p in paras_phrases]
    common_phrases = set()
    mapping = {}
    typos = {}
    edges_to_add = []
    splitted = {}  # for debugging
    
    merged_question_phrases = {}
    question_phrases_nostop = {}
    question_phrases_dict = {question_phrase[0]:question_phrase[1:] for question_phrase in question_phrases}
    question_phrases_text = list(set(question_phrases_dict.keys())) 
    question_phrases_source = [question_phrases_dict[phrase_text][0] for phrase_text in question_phrases_text]
    question_phrases_text = [remove_prefix(s, 'question ') for s in question_phrases_text]

    
    for question_phrase in question_phrases_text:
        question_phrase_nostop =  remove_stop_words(question_phrase) 
        if question_phrase_nostop != question_phrase: 
            if(len(question_phrase_nostop.split()) > 0):
                question_phrases_nostop[question_phrase_nostop] = question_phrase 
            
    for phrase in G.nodes:
        if(phrase in question_phrases_text):    # has a exact match
            common_phrases.add(phrase)
            continue
        
        typo_match = False
        for q_phrase in question_phrases_text:
            if white_space_fix(singular_phrase(remove_punc(phrase))) == white_space_fix(singular_phrase(remove_punc(q_phrase))):    
                common_phrases.add(q_phrase)
                typos[phrase] = q_phrase
                typo_match = True
                break
        if typo_match:
            continue

        no_prefix_phrase = remove_prefix(phrase, '[P', paraid = True)
        if(no_prefix_phrase in question_phrases_text): 
            edges_to_add.append((phrase, no_prefix_phrase))   # [P6] episode, episode
            common_phrases.add(no_prefix_phrase)
            mapping[phrase] = no_prefix_phrase
            continue
        if(singular_phrase(no_prefix_phrase) in question_phrases_text):
            edges_to_add.append((phrase, singular_phrase(no_prefix_phrase)))   # [P6] episodes, episode
            common_phrases.add(singular_phrase(no_prefix_phrase))
            mapping[phrase] = singular_phrase(no_prefix_phrase)
            continue
        if(no_prefix_phrase in [singular_phrase(question_phrase) for question_phrase in question_phrases_text]):
            question_phrase = question_phrases_text[[singular_phrase(question_phrase) for question_phrase in question_phrases_text].index(no_prefix_phrase)]
            edges_to_add.append((phrase, question_phrase))     # [P6] episode, episodes
            common_phrases.add(question_phrase)
            mapping[phrase] = question_phrase 
            continue
        
        # For typo: Shani Gandi VS Shani Gandhi, 
        typo_match = False
        for question_phrase in question_phrases_text:
            if(fuzz.ratio(no_prefix_phrase, question_phrase) > 90 and len(no_prefix_phrase.split()) == len(question_phrase.split())): 
                matches = find_match_with_tolerance(no_prefix_phrase, question_phrase, 2)  # 2.1 mile VS 2.1 mi
                # '1965 birthday honours' not a typo of '1925 birthday honours'
                if all([c.isdigit() for c in ((set(no_prefix_phrase) - set(question_phrase)) | ( set(question_phrase) - set(no_prefix_phrase)))]) == False: 
                    typo_strs = [question_phrase[m.start:m.end] for m in matches if len(m.matched) > 2]  
                    if(len(typo_strs) > 0):
                        typo_match = True
                        common_phrases.add(question_phrase) 
                        if no_prefix_phrase == phrase:
                            typos[phrase] = question_phrase
                        else:
                            edges_to_add.append((phrase, question_phrase)) 
                            mapping[phrase] = question_phrase 
                        break
            
        if typo_match:
            continue
            
        if(len(nlp(no_prefix_phrase).ents) == 1 and nlp(no_prefix_phrase).ents[0].label_ == 'DATE' and nlp(no_prefix_phrase).ents[0].text == no_prefix_phrase):  # avoid date only phrase '[P8] january 2' to inclusive match to 'january 2 2012', but do not affect phrase contains date entity, such as 'tosca 1956 film'
            continue

        # check partial matched question_phrase 
        # when extract noun phrases, already map to best choice among question phrases and titles, here only match to question phrases           
        # check inclusions first, elif then inclusion_best_match
        included_question_phrases = inclusions(no_prefix_phrase, question_phrases_text)
        if len(included_question_phrases) == 0:
            included_question_phrases = inclusions(no_prefix_phrase, list(question_phrases_nostop.keys()))
        if len(included_question_phrases) > 0:    
            # still want ot match 'woman' VS 'bussinesswoman', 'performance' match 'organizations performance'
            included_question_phrases = set([included_phrase for included_phrase in included_question_phrases if similar_in_length(no_prefix_phrase, included_phrase) and included_phrase])
        
        for included_question_phrase in included_question_phrases:
            if(included_question_phrase): 
                
                if included_question_phrase in question_phrases_nostop:
                    included_question_phrase = question_phrases_nostop[included_question_phrase]
                    
                if(NN_check(no_prefix_phrase) and len(included_question_phrase.strip().split())!= 1):  # do not expand single NN word, such as 'practice' to ‘air employment practice committee’
        # 'louisville' is NNP, but would like it to match question phrase 'university of louisville'
                    continue
                
                included_question_phrase_source = question_phrases_source[question_phrases_text.index(included_question_phrase)]
                if(NN_check(included_question_phrase) or included_question_phrase_source=='chunk' ):  # "[P4] 2010 video game" VS 'video game' in question
                    if phrase.startswith('[P'):
                        intermediate_node = phrase[:5] + included_question_phrase
                    else:
                        most_freq_pid = np.argmax([para_p.count(phrase) for para_p in flatten_paras_phrases])
                        intermediate_node = '[P' + str(most_freq_pid) + '] ' + included_question_phrase
                    edges_to_add.append((phrase, intermediate_node))
                    edges_to_add.append((intermediate_node, included_question_phrase))  # add prefix
                    
                else: 
                    edges_to_add.append((phrase, included_question_phrase))
                common_phrases.add(included_question_phrase)
                mapping[phrase] = included_question_phrase

 
    G.add_edges_from(edges_to_add)  
 
    for question_phrase in question_phrases:
        for another_question_phrase in question_phrases:
            if(len(question_phrase) > 2 and len(another_question_phrase) > 2 ): # has start_id and end_idx
                phrase, _, p_start, p_end = question_phrase
                phrase = remove_prefix(phrase, 'question ')
                another_phrase, _, n_start, n_end = another_question_phrase
                another_phrase = remove_prefix(another_phrase, 'question ')
                if phrase != another_phrase and p_start >= n_start and p_end <= n_end:# do not check  entity in another_noun directly, since singular_phrase('new york police') = new york polouse, which  is not in 'former new york police detective'
                    if(another_phrase in merged_question_phrases):
                        merged_question_phrases[phrase] = merged_question_phrases[another_phrase]
                    else:
                        merged_question_phrases[phrase] = another_phrase
                    if phrase in common_phrases:
                        common_phrases.remove(phrase)
     
    G = nx.relabel_nodes(G, typos) 
    G = nx.relabel_nodes(G, merged_question_phrases) 
    
    
    return G, common_phrases, mapping, typos, merged_question_phrases
 
def connect_disjoint_subgraphs(G):
    Subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    disjoint = False
    if len(Subgraphs) > 1:
        disjoint = True
#         print("connect disjoint RG")
        representative_nodes = []  # more likely to include the representative_nodes in the final path  
        represnetive_score_nodes = []
        
        nodes = G.nodes 
        for node1, node2 in itertools.combinations(nodes, 2): 
            if(G.has_edge(node1, node2) == False and node1 != node2 and remove_prefix(node1, '[P', paraid = True) == remove_prefix(node2, '[P', paraid = True)):  
                G.add_edge(node1, node2, src = [('connect',)]) 
    
    return G, disjoint

def always_connect_disjoint_subgraphs(G):
    Subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    disjoint = False
    if len(Subgraphs) > 1:
        disjoint = True 
        representative_nodes = []  # more likely to include the representative_nodes in the final path  
        represnetive_score_nodes = []
        for S in Subgraphs: 
            S.remove_edges_from(nx.selfloop_edges(S)) 
            representative_node = sorted(S.degree, key=lambda x: x[1], reverse=True)[0]  # node with highest degree 
            representative_nodes.append(representative_node)   

        for node1, node2 in itertools.combinations([phrase[0] for phrase in representative_nodes], 2): # 0 is text, 1 is score
            if(node1 != node2):
                if(G.has_edge(node1, node2)):
                    G[node1][node2]['src'].append(('connect',)) 
                else:
                    G.add_edge(node1, node2, src = [('connect',)]) 
    
    return G, disjoint


def bridge_phrases_extraction(row):
    """
     function to compute identify bridge phrases with phrase graph. 
    """ 
    question = row["question"] 
    sentences = row['context']["sentences"].copy()
    titles = row['context']["title"] 
    answer = row["answer"]  
    titles_phrases, question_phrases, question_phrases_text, paras_phrases, answer_phrases, answer_phrases_text = noun_phrases_extraction(question, sentences, titles, answer)
    RG = create_relevant_graph(paras_phrases, question_phrases_text)  # this RG could still be disjoint

    RG, common_phrases, mapping, _, merged_question_phrases = find_common_mapping(RG, question_phrases, paras_phrases) # mapping matchs paras_phrases with question_phrases

    row['common_phrases'] = common_phrases
    
    assert len(set(merged_question_phrases.keys()) - set(question_phrases_text)) == 0
 
    # connect disjoint subgraphs
    RG, disjoint = connect_disjoint_subgraphs(RG)
    RG, disjoint = always_connect_disjoint_subgraphs(RG)
    row['disjoint'] = disjoint
    RG.remove_edges_from(nx.selfloop_edges(RG))       # remove (500 mile, 500 mile, {'src': [(9, 2)]})  

    # add weight to RG, to lower chance of [p4] video game -> video game -> [p8] video game 
    for u,v,d in RG.edges(data=True):
        if('src' not in d):     # {}
            d['weight']=5
        
    question_only_phrase = list(set(question_phrases_text).difference(common_phrases | set(merged_question_phrases.keys()))) 
    row['question_only_phrase'] = question_only_phrase

    if(len(common_phrases) > 1): 
        steiner_graph = approx.steinertree.steiner_tree(RG, common_phrases) 
        path_phrases = list(steiner_graph.nodes)  # to find the smallest subgraph covers all common_phrases  
 
        for i, phrase in enumerate(path_phrases):
            path_phrases[i] = remove_prefix(phrase, '[P', paraid = True)
        path_phrases = list(set(path_phrases))
        if(len(path_phrases) == 0):
            print(row['id'], 'no path found')   
        path_edges =  nx.get_edge_attributes(steiner_graph, 'src')      
  

    else: #  0 or 1 common phrases
        path_phrases = list(common_phrases)              
        path_edges = {}
        for phrase in common_phrases:
            path_edges[(phrase, None)] = []
            for pi, para_phrase in enumerate(paras_phrases):
                for si, sentence_phrases in enumerate(para_phrase):
                    if si > 0:  # not the title 
                        sentence_phrases = [ remove_prefix(sentence_phrase[0], '[P', paraid = True) for sentence_phrase in sentence_phrases]   
                        if any([find_match_with_tolerance(phrase, sentence_phrase) or find_match_with_tolerance(sentence_phrase, phrase) for sentence_phrase in sentence_phrases]):
                            path_edges[(phrase, None)].append((pi,si-1)) 
    path_edges = list(path_edges.items())
    path_sents = get_path_sents(path_edges)
    joint_context = basic_normalize(" ".join(list(flatten(row["context"]['title']))) + ' ' + " ".join(list(flatten(row["context"]['sentences']))) )
    joint_title = basic_normalize(" ".join(list(flatten(row["context"]['title'])))) 

 
    steiner_points = set(path_phrases) - set(question_phrases_text)
    _, steiner_points = phrase_postpreocess(steiner_points, joint_context, list(flatten(titles_phrases)))
    
    row['steiner_points'] = steiner_points  
        
    supporting_facts_phrases = []  
    supporting_facts_text = []
    supporting_facts_titles_phrases = []
    sp_sent_ids = row['supporting_facts']['sent_id'] 
    for spi, sp_title in enumerate(row['supporting_facts']['title']): 
        pi = titles.index(sp_title)
        sent_id = sp_sent_ids[spi]
        if pi < len(sentences) and sent_id < len(sentences[pi]):   # in case edge case 
            supporting_facts_text.append(sentences[pi][sent_id])
            sp_sent_phrases = paras_phrases[pi][sent_id+1]   # +1 since [0] is title phrase
            sp_sent_phrases = [remove_prefix(phrase[0], '[P', paraid = True) for phrase in sp_sent_phrases]
            supporting_facts_phrases.append(sp_sent_phrases)    
            
            t_phrases = titles_phrases[pi]
            t_phrases = [remove_prefix(phrase, '[P', paraid = True) for phrase in t_phrases]
            supporting_facts_titles_phrases.append(t_phrases) 
  
    # oracle bridge_phrases 
    bridge_phrases = annotate_bridge_phrases(supporting_facts_phrases, row['supporting_facts']['title'], supporting_facts_titles_phrases) 
    
    _, row['bridge_phrases'] = phrase_postpreocess(set(bridge_phrases) - set(question_phrases_text), joint_context, list(flatten(titles_phrases)))    
              
    supporting_facts_phrases, _ = phrase_postpreocess(list(flatten(supporting_facts_phrases)), joint_context, [])
    supporting_facts_titles_phrases, _ = phrase_postpreocess(list(flatten(supporting_facts_titles_phrases)), joint_title, [])
    row['supporting_facts_text'] = supporting_facts_text 
     
    
    return row 

