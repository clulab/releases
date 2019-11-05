import pandas as pd
import json
import os

test_file="../../data/rte/mednli/mli_test_v1.jsonl"
assert os.path.exists(test_file) is True
t=pd.read_json(test_file,lines=True)
out_path="../../data/rte/mednli/mli_test_lex.jsonl"
with open(out_path,'w') as outfile:
    outfile.write("")
for i,row in t.iterrows():
    with open(out_path, 'a+') as outfile:
        total = {'claim': row.sentence1,
                 'evidence':row.sentence2,
                 "label":row.gold_label}
        json.dump(total,outfile)
        outfile.write("\n")