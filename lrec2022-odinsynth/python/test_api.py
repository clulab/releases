
import json
import requests

url = 'http://127.0.0.1:8000/score'

req = {
   'sentences': [
           ['Ohio', 'Republican', 'Rep.', 'Gillmor', 'found', 'dead', 'in', 'PERSON', 'apartment', 'DATE', ',', 'Republican', 'aide', 'says'], 
           ['Ohio', 'Rep.', 'Gillmor', 'found', 'dead', 'in', 'PERSON', 'apartment', 'DATE', ',', 'Republican', 'aide', 'says']
    ], 
   'specs': [
             {'docId': 'test', 'sentId': 0, 'start': 7, 'end': 10}, 
             {'docId': 'test', 'sentId': 1, 'start': 6, 'end': 9}
         ], 
   'patterns': ['□', '□ □', '□?', '□*', '□+', '[□]'], 
   'current_pattern': '□'
}

response = requests.post(url, data=json.dumps(req)) # should be like: 'http://<..>:<..>/score
response.json() # example of values (can be different): [0.0003, 0.9995, 0.0002, 0.0002, 0.0002, 0.0002]

print (response.json())




