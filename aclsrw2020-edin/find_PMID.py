import json
import glob, os


dic = {}
os.chdir("/data1/home/zheng/reach/output")
for file in glob.glob("*.uaz.sentences.json"):
    for sentence in json.load(open(file))['frames']:
        dic[sentence['text']] = file.split('.')[0]

events = ['ge_events.json', 'lo_events.json', 'ph_events.json']

for name in events:
    new_events = []
    jfile = json.load(open('/data1/home/zheng/reach/supplementary_material/'+name))
    for event in jfile:
        sentence = event['sentence']
        if sentence in dic:
            event['PMID'] = dic[sentence]
        else:
            print (sentence)
        new_events.append(event)
    with open("/data1/home/zheng/reach/supplementary_material/new_"+name, 'w') as f:
        f.write(json.dumps(new_events))