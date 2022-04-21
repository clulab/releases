import torch
import os
from typing import Any, Dict
from fastapi import FastAPI
from model import PointwiseBM
from utils import extract_sentences_from_odinson_doc
from queryparser import QueryParser
import json

# Run with:
# uvicorn api:app --reload
#
# or with:
# uvicorn api:app --workers 4
# to avoid large memory usage
# or
# uvicorn api:app --workers 0 --limit-concurrency 4
# or
# uvicorn api:app --workers 1 --timeout-keep-alive 900 --port 8000
# 
# 
# Alternatively, use: CHECKPOINT_PATH='8_512' ./bash_scripts/start_servers.sh 8001
app = FastAPI()


# checkpoint_path = '/home/rvacareanu/projects/odinsynth/python/logs/odinsynth/version_921/checkpoints/best.ckpt' # 2_128
# checkpoint_path = '/home/rvacareanu/projects/odinsynth/python/logs/odinsynth/version_924/checkpoints/best.ckpt' # 2_256
# checkpoint_path = '/home/rvacareanu/projects/odinsynth/python/logs/odinsynth/version_927/checkpoints/best.ckpt' # 4_256
# checkpoint_path = '/home/rvacareanu/projects/odinsynth/python/logs/odinsynth/version_930/checkpoints/best.ckpt' # 4_512
# checkpoint_path = '/home/rvacareanu/projects/odinsynth/python/logs/odinsynth/version_933/checkpoints/best.ckpt' # 8_512

paths = {
    '2_128'    : '/home/rvacareanu/projects/odinsynth/python/logs/odinsynth/version_921/checkpoints/best.ckpt',
    '2_256'    : '/home/rvacareanu/projects/odinsynth/python/logs/odinsynth/version_924/checkpoints/best.ckpt',
    '4_256'    : '/home/rvacareanu/projects/odinsynth/python/logs/odinsynth/version_927/checkpoints/best.ckpt',
    '4_512'    : '/home/rvacareanu/projects/odinsynth/python/logs/odinsynth/version_930/checkpoints/best.ckpt',
    '8_512'    : '/home/rvacareanu/projects/odinsynth/python/logs/odinsynth/version_933/checkpoints/best.ckpt',
    'bert_base': '/home/rvacareanu/projects/odinsynth/python/logs/odinsynth/version_936/checkpoints/best.ckpt',
}

if 'CHECKPOINT_PATH' not in os.environ:
    print("No checkpoint specified. Was this intentional?")
    exit()

user_set_path = os.environ.get("CHECKPOINT_PATH")
checkpoint_path = paths.get(user_set_path, user_set_path) # The user can set either a name (i.e. '2_128') or a full path (i.e. '/home/rvacareanu/projects/odinsynth/python/logs/odinsynth/version_921/checkpoints/best.ckpt')

# Load for classical pretraining
model = PointwiseBM.load_from_checkpoint(checkpoint_path) #.to(torch.device('cuda:0'))
model = model.eval()


# # Load the RL trained agent
# from rl_agent import OdinsynthRLAgentWrapper
# device = torch.device('cuda:0')
# model = OdinsynthRLAgentWrapper(device).eval()
# model.load_state_dict(torch.load('/home/rvacareanu/projects/odinsynth/python/results/rl/from_pretrained_e2_q_network5.pt'))
# model = model.model

print(os.environ.get("CHECKPOINT_PATH"))

parser = QueryParser()

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# device = torch.device('cpu')

model = model.to(device)


@app.get("/")
def root():
    print('test')
    return {"message": "Hello, World!"}

"""
    Gets as input a dictionary that contains:
        - either 'doc' or 'sentences' (used to extract the sentences)
        - specs (used to extract the highlighted portions)
        - patterns (the patterns that will be scored)
        - current_pattern (the current pattern; used for scoring)

    Example:
    >>
    >> url = 'http://127.0.0.1:42183/score'
    >> import json
    >> import requests
    >> req = {
    >>    'sentences': [
    >>            ['Ohio', 'Republican', 'Rep.', 'Gillmor', 'found', 'dead', 'in', 'PERSON', 'apartment', 'DATE', ',', 'Republican', 'aide', 'says'], 
    >>            ['Ohio', 'Rep.', 'Gillmor', 'found', 'dead', 'in', 'PERSON', 'apartment', 'DATE', ',', 'Republican', 'aide', 'says']
    >>     ], 
    >>    'specs': [
    >>              {'docId': 'test', 'sentId': 0, 'start': 7, 'end': 10}, 
    >>              {'docId': 'test', 'sentId': 1, 'start': 6, 'end': 9}
    >>          ], 
    >>    'patterns': ['□', '□ □', '□?', '□*', '□+', '[□]'], 
    >>    'current_pattern': '□'
    >> }
    >> response = requests.post(url, data=json.dumps(req)) # should be like: 'http://<..>:<..>/score
    >> response.json() # example of values (can be different): [0.0003, 0.9995, 0.0002, 0.0002, 0.0002, 0.0002]
    >>
    >>
"""
@app.post("/score")
async def score(body: Dict[Any, Any] = None):
    print(body)
    if ('doc' not in body and 'sentences' not in body) or 'specs' not in body or 'patterns' not in body or 'current_pattern' not in body:
        return "Wrong format"

    # If the sentences were provided, no need to read the odinson document and parse it
    if 'sentences' in body:
        sentences = body['sentences']
    else:
        doc = json.loads(body['doc'])
        sentences = extract_sentences_from_odinson_doc(doc, 'word')
        
    specs = body['specs']
    if len(specs) == 0:
        return []
    else:
        if type(specs[0]) == str:
            specs = [json.loads(s) for s in specs]

    patterns = body['patterns']
    current_pattern = body['current_pattern']

    if(len(patterns) == 0):
        return list()


    specs_filtered     = [sp for sp in specs if int(sp['start']) < int(sp['end']) and int(sp['end']) >= 0]

    # Which sentences (index) have specs
    sentences_with_specs = [sp['sentId'] for sp in specs_filtered]
    
    sentences_filtered = [se for sp, se in zip(specs, sentences) if int(sp['start']) < int(sp['end']) and int(sp['end']) >= 0]
    sentences_filtered = [sentences_filtered[i] for i in sentences_with_specs] # Keep only the sentences with spec

    scores = []
    batch_size = 2
    chunked_patterns = [patterns[i:i + batch_size] for i in range(0, len(patterns), batch_size)]  
    for batch in chunked_patterns:
        data_list = []
        for pattern in batch:
            for sp, se in zip(specs_filtered, sentences_filtered):
                if int(sp['start']) < int(sp['end']) and int(sp['end']) >= 0:
                    data_list.append({
                        'text': [
                            ' '.join(se),
                            int(sp['start']),
                            int(sp['end']),
                            current_pattern,
                            pattern,
                            0,
                        ]
                    }) 
        prepared = model.collate_fn(model.tokenizer, model.symbols, model.symbol_tensors, parser, data_list)
        with torch.no_grad():
            score = model(prepared).squeeze(1)
            scores.append(score.reshape(-1, len(specs_filtered)).mean(dim=1).detach().cpu().numpy().tolist())
    
    return [y for x in scores for y in x]

    # # Score pattern by pattern
    # scores = []
    # for pattern in patterns:
    #     data_list = []
    #     for sp, se in zip(specs_filtered, sentences_filtered):
    #         data_list.append({
    #             'text': [
    #                 ' '.join(se),
    #                 int(sp['start']),
    #                 int(sp['end']),
    #                 current_pattern,
    #                 pattern,
    #                 0,
    #             ]
    #         })

    #     prepared = model.collate_fn(model.tokenizer, model.symbols, model.symbol_tensors, parser, data_list)
    #     score = model(prepared).mean().item()
    #     scores.append(score)

    # return scores


@app.get('/version')    
async def version():
    return checkpoint_path
