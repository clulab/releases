import json
import csv
import re
from collections import defaultdict
import networkx as nx
from networkx.algorithms import community
from math import log


from itertools import combinations

import numpy as np

from termcolor import colored

import statistics

import sys
from utils import helper

def argmax(l):
    return max(enumerate(l), key=lambda x: x[1])[0]

def build_graph(d):
    edge_dict = defaultdict(dict)
    G = nx.Graph()
    tokens = ['ROOT'] + d['token']
    heads = d['stanford_head']
    deprel= d['stanford_deprel']
    G.add_nodes_from([i for i in range(len(tokens))])
    for i in range(1, len(tokens)):
        G.add_edge(heads[i-1], i)
        edge_dict[heads[i-1]][i] = deprel[i-1]
        edge_dict[i][heads[i-1]] = "<"+deprel[i-1]
    return G, edge_dict


def rules_with_out_golds(candidates, origin, model_output):
    confs = defaultdict(dict)
    # In this case, we do not have access to the gold labels, so we are relying on predicted labels
    assert(len(model_output)==len(origin))
    subjects = defaultdict(list)
    objects = defaultdict(list)
    lineid2rule = defaultdict(dict) 
    # tuples = defaultdict(list)
    c = 0
    for i, item in enumerate(model_output):
        g, e = build_graph(origin[i])
        tokens = ['ROOT'] + origin[i]['token']
        postags = ['ROOT'] + origin[i]['stanford_pos']
        if item['predicted_label'].startswith('per'):
            subj_type = 'PERSON'
        else:
            subj_type = 'ORGANIZATION'
        if item['predicted_label'] != 'no_relation':
            if len(item['predicted_tags']) != 0 and not item['from_prev']:
                c += 1
                subj = list(range(origin[i]['subj_start']+1, origin[i]['subj_end']+2))
                obj = list(range(origin[i]['obj_start']+1, origin[i]['obj_end']+2))
                triggers = [j+1 for j, w in enumerate(origin[i]['token']) if j in item['predicted_tags'] and j+1 not in subj and j+1 not in obj]
                if triggers:
                    sp = []
                    op = []
                    trigger_head = triggers[argmax([g.degree[t] for t in triggers])]
                    for t in triggers:
                        for s in subj:
                            temp1 = nx.shortest_path(g, t, s)
                            for o in obj:
                                temp2 = nx.shortest_path(g, t, o)
                                if len(temp1+temp2)<len(sp+op) or sp == []:
                                    sp = temp1
                                    op = temp2
                                    trigger_head = t
                    if origin[i]['subj_type'] not in subjects[item['predicted_label']]:
                        subjects[item['predicted_label']].append(origin[i]['subj_type'])
                    if origin[i]['obj_type'] not in objects[item['predicted_label']]:
                        objects[item['predicted_label']].append(origin[i]['obj_type']) 

                    trigger = ''
                    prev = -1
                    for j in triggers:
                        if prev == -1:
                            trigger += '"%s"'%tokens[j]
                        elif j - prev == 1:
                            trigger += ' ' + '"%s"'%tokens[j]
                        else:
                            trigger += '(/.+/){,%d}'%(j-prev-1) + '"%s"'%tokens[j]
                        prev = j
                    l = [trigger, [postags[j] for j in triggers], [e[sp[j]][sp[j+1]] for j in range(len(sp)-1)], [e[op[j]][op[j+1]] for j in range(len(op)-1)]]
                    
                    if l not in candidates[item['predicted_label']] and len(sp)!=0 and len(op)!=0:
                        candidates[item['predicted_label']] += [l]
                    # elif len(triggers) > 3:
                    #     print (triggers, trigger, item['predicted_label'])
    return candidates, subjects, objects


def save_rule_dict(candidates, subjects, objects, name):
    res = defaultdict(dict)    
    output = dict()
    total = 0
    for label in candidates:
        cands = candidates[label]
        label = label.replace('/', '_slash_')
        output[label] = defaultdict(list)
        for i, c in enumerate(cands):
            trigger = c[0]
            subj = c[2]
            obj = c[3]
            if len(subj)>0 and len(obj)>0:
                output[label][trigger].append({'subj':subj, 'obj':obj})
                total += 1
    for label in output:
        label2 = label.replace('_slash_', '/')
        count = 0
        for trigger in output[label]:
            for rule in output[label][trigger]:
                count += 1
    print ("Generated %d rules."%total)
    with open('rules_%s.json'%name, 'w') as f:
        f.write(json.dumps(output))


    sod = defaultdict(dict)
    helper.ensure_dir('src/main/resources/grammars_%s/'%sys.argv[1], verbose=True)
    with open('src/main/resources/grammars_%s/master.yml'%sys.argv[1],'w') as f:
        f.write('''
taxonomy:
  - SUBJ_Person
  - SUBJ_Organization
  - OBJ_Person
  - OBJ_Organization
  - OBJ_Date
  - OBJ_Number
  - OBJ_Title
  - OBJ_Country
  - OBJ_Location
  - OBJ_City
  - OBJ_Misc
  - OBJ_State_or_province
  - OBJ_Duration
  - OBJ_Nationality
  - OBJ_Cause_of_death
  - OBJ_Criminal_charge
  - OBJ_Religion
  - OBJ_Url
  - OBJ_Ideology
  - Relation:
      - per:title
      - org:top_members/employees
      - per:employee_of
      - org:alternate_names
      - org:country_of_headquarters
      - per:countries_of_residence
      - org:city_of_headquarters
      - per:cities_of_residence
      - per:age
      - per:stateorprovinces_of_residence
      - per:origin
      - org:subsidiaries
      - org:parents
      - per:spouse
      - org:stateorprovince_of_headquarters
      - per:children
      - per:other_family
      - per:alternate_names
      - org:members
      - per:siblings
      - per:schools_attended
      - per:parents
      - per:date_of_death
      - org:member_of
      - org:founded_by
      - org:website
      - per:cause_of_death
      - org:political/religious_affiliation
      - org:founded
      - per:city_of_death
      - org:shareholders
      - org:number_of_employees/members
      - per:date_of_birth
      - per:city_of_birth
      - per:charges
      - per:stateorprovince_of_death
      - per:religion
      - per:stateorprovince_of_birth
      - per:country_of_birth
      - org:dissolved
      - per:country_of_death

rules:
  - import: grammars/entities.yml
    vars:
      # We need our entities before we can match events
      # Here we make use of the ${rulepriority} variable
      # used in the entities.yml rules
      rulepriority: "1"

            ''')
        for label in subjects:
            count = 0
            for subj in subjects[label]:
                for obj in objects[label]:
                    subj = subj[0]+subj[1:].lower()
                    obj = obj[0]+obj[1:].lower()
                    sod[label][subj+obj] = count
                    f.write('''
  - import: grammars_%s/%s.yml
    vars:
      label: %s
      rulepriority: "3+"
      subject_type: SUBJ_%s
      object_type: OBJ_%s
      count: "%d"

  '''%(name, label.replace('/', '_slash_')+'_unit', label, subj, obj, count))
                    count += 1
    print (json.dumps(sod))

if __name__ == "__main__":

    model_output = json.load(open('output_%s.json'%sys.argv[1]))
    origin = json.load(open('src/main/resources/data/train.json'))

    candidates = defaultdict(list)

    candidates, subjects, objects = rules_with_out_golds(candidates, origin, model_output)

    save_rule_dict(candidates, subjects, objects, sys.argv[1])






