import argparse
import json
import os
from collections import defaultdict

# Convert the GENIA data to BioNLP format
def parse_json_data(input_file, output_file, event_type):
    triggers = dict()
    rule_mappings = dict()

    pubmed = defaultdict(dict)
    with open(input_file) as f:
        raw = json.load(f)
        for entry in raw:
            sentence = entry['sentence']
            rule     = entry["rule"]
            trigger  = entry["trigger"]
            entity   = entry["entity"]

            eid = '%s%i%i'%(entity[0], entity[1][0], entity[1][1])
            temp = {'events':[{'trigger':trigger, 'rule': rule}], 'entity':entity}
            pubmed[sentence][eid] = temp
    i = 0
    for sentence in pubmed:
        i += 1
        with open("%s/%d.txt"%(output_file, i), "w") as txt:
            txt.write(sentence)
        j = 1
        k = len(pubmed[sentence].keys())+1
        l = 1
        for eid in pubmed[sentence]:
            entity = pubmed[sentence][eid]["entity"]
            with open("%s/%d.a1"%(output_file, i), "a") as a1:
                a1.write("T%d\tProtein %d %d\t%s\n"%(j, entity[1][0], entity[1][1], entity[0]))
            for event in pubmed[sentence][eid]["events"]:
                trigger = event["trigger"]
                rule = event["rule"]
                rule_mappings["%s/%d/E%d"%(output_file, i, l)] = rule 
                with open("%s/%d.a2"%(output_file, i), "a") as a2:
                    if "%s%d%d"%(trigger[0], trigger[1][0], trigger[1][1]) not in triggers:
                        triggers["%s%d%d"%(trigger[0], trigger[1][0], trigger[1][1])] = k
                        k += 1
                    a2.write("T%d\t%s %d %d\t%s\n"%(triggers["%s%d%d"%(trigger[0], trigger[1][0], trigger[1][1])], event_type, trigger[1][0], trigger[1][1], trigger[0]))
                    a2.write("E%d\t%s:T%d Theme:T%d\n"%(l, event_type, triggers["%s%d%d"%(trigger[0], trigger[1][0], trigger[1][1])], j))
                l += 1
            j += 1

    with open("%s/rule_mappings.json"%output_file, 'w') as f:
        f.write(json.dumps(rule_mappings))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('output')
    parser.add_argument('event')
    args = parser.parse_args()

    os.mkdir(args.output)
    parse_json_data(args.datadir, args.output, args.event)