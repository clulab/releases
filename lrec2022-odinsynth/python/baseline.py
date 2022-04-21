import argparse
from ast import literal_eval
from typing import Dict
import tqdm
import json

# Each line contains: {"text: {"sentences":[], "start": [], "end": [], "pattern": ""}}
def build_training_data_from_stepsspecs_file(filename: str = '/data/nlp/corpora/odinsynth/data/rules100k_unrolled/train_names'):
    with open(filename) as fin:
        lines = fin.readlines()
    
    lines = [[x.strip() for x in l.split('\t')] for l in lines]
    steps = [l[0] for l in lines]
    specs = [l[1] for l in lines]
    print(len(steps))

    with open('/data/nlp/corpora/odinsynth/data/rules100k_seq2seq/train.tsv', 'w+') as fout:
        for step_f, spec_f in tqdm.tqdm(list(zip(steps, specs))):
            with open(step_f) as fin:
                step_jsonlines = fin.readlines()
            with open(spec_f) as fin:
                spec_jsonlines = fin.readlines()
            spec_json    = json.loads(spec_jsonlines[0])
            doc_json     = json.loads(spec_jsonlines[1])
            
            spec_json['specs'].sort(key=lambda x: x['sentId'])
            
            correct_rule = json.loads(step_jsonlines[-1])['next_correct_rule']
            start = [x['start'] for x in spec_json['specs']]
            end   = [x['end'] for x in spec_json['specs']]
            sentences = [list(filter(lambda x: x['name'] == 'word', s['fields']))[0]['tokens'] for s in doc_json['sentences']]

            data_line = {
                "text": {
                    "sentences": sentences,
                    "start": start,
                    "end": end,
                    "correct_rule": correct_rule,

                }
            }
            _ = json.dump(data_line, fout)
            _ = fout.write('\n')

def main(config: Dict = {}):
    print(config)

# python baseline.py --train-path /data/nlp/corpora/odinsynth/data/TACRED/tacred/data/json/train_processed.json --test-path /data/nlp/corpora/odinsynth/data/TACRED/tacred/data/json/test_processed.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entry point of the baseline application experiments.")
    # parser = BaselineTransformerModel.add_model_specific_args(parser)
    
    parser.add_argument('--train-path', type=str, required=True, help="Train path for training")
    parser.add_argument('--test-path', type=str, required=True, help="Val path for testing")

    result = parser.parse_args()
    result = vars(result)

    main(result)
