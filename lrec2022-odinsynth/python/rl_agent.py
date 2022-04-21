from rl_utils import ProblemSpecification
from queryast import AstNode
from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from queryparser import QueryParser
from model import PointwiseBM




"""
'sentences': [
        ['Ohio', 'Republican', 'Rep.', 'Gillmor', 'found', 'dead', 'in', 'PERSON', 'apartment', 'DATE', ',', 'Republican', 'aide', 'says'], 
        ['Ohio', 'Rep.', 'Gillmor', 'found', 'dead', 'in', 'PERSON', 'apartment', 'DATE', ',', 'Republican', 'aide', 'says']
 ], 
'specs': [
          {'docId': 'test', 'sentId': 0, 'start': 7, 'end': 10}, 
          {'docId': 'test', 'sentId': 1, 'start': 6, 'end': 9}
      ], 


curl -X POST -H "Content-Type: application/json" -d '{"query": "[word=this] [word=is]", "sentences": ["this is a test", "this is a new test"], "specs": [{"sentence_id": 0, "start": 0, "end": 2}, {"sentence_id": 1, "start": 0, "end": 2}] }' 127.0.0.1:9000/isSolution
"""

"""
    RL Agent consisting of an underlying (pretrained) PointwiseBM
    Note that it holds a reference of the torch.device
    Will move the tensors to the right device before applying the model
"""
class OdinsynthRLAgentWrapper(nn.Module):
    def __init__(self, checkpoint_path = '/home/rvacareanu/projects/odinsynth/python/logs/odinsynth/version_921/checkpoints/best.ckpt') -> None:
        super().__init__()
        # self.model = PointwiseBM(PointwiseBM.load_from_checkpoint(checkpoint_path).hparams).to(device)#.eval()
        # self.model  = PointwiseBM.load_from_checkpoint('/home/rvacareanu/projects/odinsynth/python/logs/odinsynth/version_921/checkpoints/best.ckpt')
        self.model  = PointwiseBM.load_from_checkpoint(checkpoint_path)
        # for param in self.model.model.parameters():
            # param.requires_grad = False
        self.parser = QueryParser()

    """
    batch is a list of (current_node, [next_node_1, next_node_2, .., next_node_n], problem_specification) triples
    """
    def generalized_scorer_batched(self, batch, device, **kwargs):
        scores = []
        data_list = []
        spec_sizes = []
        # Iterate over the batch to construct the data structure that will be passed to the model
        for b in batch:
            specs     = b[2].specs
            sentences = b[2].sentences

            specs_filtered = [sp for sp in specs if int(sp['start']) < int(sp['end']) and int(sp['end']) >= 0]
        
            # Which sentences (index) have specs
            sentences_with_specs = [sp['sentId'] for sp in specs_filtered]
            
            sentences_filtered = [se for sp, se in zip(specs, sentences) if int(sp['start']) < int(sp['end']) and int(sp['end']) >= 0]
            sentences_filtered = [sentences_filtered[i] for i in sentences_with_specs] # Keep only the sentences with spec

            for next_node in b[1]:
                spec_sizes.append(len(specs_filtered))
                for sp, se in zip(specs_filtered, sentences_filtered):
                    data_list.append({
                            'text': [
                                ' '.join(se),
                                int(sp['start']),
                                int(sp['end']),
                                b[0],
                                next_node,
                                0,
                            ]
                        }
                    )

        prepared = self.model.collate_fn(self.model.tokenizer, self.model.symbols, self.model.symbol_tensors, self.parser, data_list)
        # prepared_on_device = {
            # 'input_ids'      : prepared['input_ids'],#.to(device),
            # 'attention_masks': prepared['attention_masks'],#.to(device),
            # 'token_type_ids' : prepared['token_type_ids'],#.to(device),
        # }

        # We are then batching over the previously constructed data structure
        # Needed to avoid cases with nodes that have thousands of possible
        # next nodes, resulting in huge batch (can happen when filling a word, 
        # if the vocabulary is huge)
        batch_size = kwargs.get('iterating_batch_size', 128)
        batched_result = []
        for i in range(0, prepared['input_ids'].shape[0], batch_size):
            batch_prepared_on_device = {
                'input_ids'      : prepared['input_ids'][i:i+batch_size].to(device),
                'attention_masks': prepared['attention_masks'][i:i+batch_size].to(device),
                'token_type_ids' : prepared['token_type_ids'][i:i+batch_size].to(device),
            }
            result = self.model(batch_prepared_on_device, return_logits=True, **kwargs)
            batched_result.append(result)
            
        # Concatenate everything, as if we ran the model without batching
        result = torch.cat(batched_result, dim=0)
        scores = torch.cat([x.mean(0) for x in result.split(spec_sizes, dim=0)], dim=0)

        return scores

    """
    batch is a list of (current_node, [next_node_1, next_node_2, .., next_node_n], problem_specification)
    """
    def generalized_score(self, batch, return_tensors, device, **kwargs):
        scores = []
        for b in batch:

            specs     = b[2].specs
            sentences = b[2].sentences

            specs_filtered = [sp for sp in specs if int(sp['start']) < int(sp['end']) and int(sp['end']) >= 0]
        
            # Which sentences (index) have specs
            sentences_with_specs = [sp['sentId'] for sp in specs_filtered]
            
            sentences_filtered = [se for sp, se in zip(specs, sentences) if int(sp['start']) < int(sp['end']) and int(sp['end']) >= 0]
            sentences_filtered = [sentences_filtered[i] for i in sentences_with_specs] # Keep only the sentences with spec

            chunk_size = kwargs.get('chunk_size', 64)
            chunked_patterns = [b[1][i:i + chunk_size] for i in range(0, len(b[1]), chunk_size)]
            chunked_result = []
            for chunked in chunked_patterns:                
                data_list = []
                for next_node in chunked:
                    for sp, se in zip(specs_filtered, sentences_filtered):
                        data_list.append({
                                'text': [
                                    ' '.join(se),
                                    int(sp['start']),
                                    int(sp['end']),
                                    b[0],
                                    next_node,
                                    0,
                                ]
                            }
                        )
                prepared = self.model.collate_fn(self.model.tokenizer, self.model.symbols, self.model.symbol_tensors, self.parser, data_list)
                prepared_on_device = {
                    'input_ids'      : prepared['input_ids'].to(device),
                    'attention_masks': prepared['attention_masks'].to(device),
                    'token_type_ids' : prepared['token_type_ids'].to(device),
                }

                # print(prepared_on_device['input_ids'].shape)
                result = self.model(prepared_on_device, return_logits=True, **kwargs)#.squeeze(1).reshape(-1, len(specs_filtered)).mean(dim=1)#.detach().cpu().numpy().tolist()
                chunked_result.append(result.squeeze(1).reshape(-1, len(specs_filtered)).mean(dim=1))
            if return_tensors:
                scores.append(torch.cat(chunked_result, dim=0))
            else:
                scores.append(torch.cat(chunked_result, dim=0).detach().cpu().numpy().tolist())
                
        return scores
        
    
    """
    :param obs -> A list of OdinsynthEnvStep objects

    The computation is done in a batch fashion

    :returns list or tensor with the computation. Depending on the data in **kwargs, 
             it returns list or tensor
             This different return types is actually necessary because, when returning
             a list, the lists inside the returned list contain a variable number of
             elements (this is because different nodes might have a different number
             of next nodes)
    """
    def forward(self, obs, device, **kwargs):
        # When the output is not stackable (tensors of different dimensions)
        # we have to return lists; 
        if 'return_list' in kwargs:
            result = self._fast_forward_with_lists(obs, device)
        # When we run the agent to obtain scores for particular (state, action) pairs
        # we short-circuit the system to only score those (that is, we do not 
        # run the system with (state, all_next_states))
        elif 'indices' in kwargs:
            result = self._fast_forward(obs, device, kwargs['indices'])
        else:
            raise ValueError("Unknown type of running")

        return result

    def old_forward(self, obs, device, **kwargs):
        # When the output is not stackable (tensors of different dimensions)
        # we have to return lists; 
        if 'return_list' in kwargs:
            result = self._forward_with_lists(obs, device)
        # When we run the agent to obtain scores for particular (state, action) pairs
        # we short-circuit the system to only score those (that is, we do not 
        # run the system with (state, all_next_states))
        elif 'indices' in kwargs:
            result = self._old_fast_forward(obs, device, kwargs['indices'])
        else:
            raise ValueError("Unknown type of running")
        
        return result


    """
    Returns the score of a particular observation, action pair
    """
    def _fast_forward(self, obs, device, indices):
        obs_filtered = [(x.query, [x.query.next_nodes_filtered(x.problem_specification.vocabulary)[idx]], x.problem_specification) for x, idx in zip(obs, indices)]
        result = self._batch_score(obs_filtered, device=device)
        return result

    """
    Returns the score of a particular observation, action pair
    """
    def _old_fast_forward(self, obs, device, indices):
        result = torch.stack([self._score(x.query, [x.query.next_nodes_filtered(x.problem_specification.vocabulary)[idx]], x.problem_specification, return_tensors=True, device=device)[0] for x, idx in zip(obs, indices)]).squeeze(1)
        return result

    """
    When we run forward on the target network to obtain
    the q values of the next observations, there can be
    observations that are final; 
    We handle this by returning 0
    Because of this, we return a list of numbers, as it 
    would be dubious to return a list of a mix of numbers
    and tensors
    """
    def _forward_with_lists(self, obs, device):
        result  = []
        for o in obs:
            # For the case when we run forward with the next_obs, there can be queries which are already finished,
            # so there is no next_node to score; We give a q_value of 0 in that case, as there is no more reward
            # to be obtained
            if o.query.is_valid_query() or len(o.query.next_nodes_filtered(o.problem_specification.vocabulary)) == 0:
                result.append(0)
            else:
                score = self._score(o.query, o.query.next_nodes_filtered(o.problem_specification.vocabulary), o.problem_specification, return_tensors=True, device=device)[0]
                result.append(score)
                
        result = [[0] if type(x) == int else x.tolist() for x in result]
        return result

    """
    See comments from _forward_with_lists
    """
    def _fast_forward_with_lists(self, obs, device):
        result  = []
        next_nodes  = []
        obs_filtered = []
        for o in obs:
            if o.query.is_valid_query() or len(o.query.next_nodes_filtered(o.problem_specification.vocabulary)) == 0:
                next_nodes.append(0)
            else:
                next_nodes.append(len(o.query.next_nodes_filtered(o.problem_specification.vocabulary)))
                obs_filtered.append(o)
        
        obs_filtered = [(x.query, x.query.next_nodes_filtered(x.problem_specification.vocabulary), x.problem_specification) for x in obs]
        result = self._batch_score(obs_filtered, device=device)

        # For the case when we run forward with the next_obs, there can be queries which are already finished,
        # so there is no next_node to score; We give a q_value of 0 in that case, as there is no more reward
        # to be obtained
        result = [x.tolist() if len(x) > 0 else [0] for x in result.split(next_nodes)]

        return result

    # Call the generalized score with the appropriate parameters
    # This method calls does the scoring functionality over a single 
    # (unrollecd, since we are expecting explicitly an AstNode, an Union
    # and a ProblemSpecification) observation
    # Batching can be obtained by repeteadly applying this method over
    # a list of observations
    def _score(self, current_node: AstNode, query: Union[AstNode, List[AstNode]], ps: ProblemSpecification, return_tensors=False, device=None, **kwargs) -> Union[List[float], float]:
        if isinstance(query, list):
            return self.generalized_score([(current_node.pattern(), [q.pattern() for q in query], ps)], return_tensors, device, **kwargs)
        else:
            return self.generalized_score([(current_node.pattern(), [query.pattern()], ps)], return_tensors, device, **kwargs)

    # Similar to _score, but this method operates over batches
    # It expects a list consisting of pairs such as:
    # (current_node: AstNode, query: List[AstNode], ps: ProblemSpecification)
    # This method should not call _score, but instead use the batching
    # technique to obtain a better performance
    def _batch_score(self, batch, return_tensors=False, device=None, **kwargs):
        return self.generalized_scorer_batched([(current_node.pattern(), [q.pattern() for q in query], ps) for (current_node, query, ps) in batch], device, **kwargs)



"""
    Simple network used for running the algorithms on typical gym environments
"""
class QNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_shape)

    def forward(self, x, **kwargs):
        x = torch.tensor(x).to(kwargs.get('device')).type(self.fc1.weight.data.type())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if kwargs.get('return_list', False):
            return x.tolist()
        elif 'indices' in kwargs:
            indices = torch.tensor(kwargs.get('indices')).to(kwargs.get('device')).view(-1, 1)
            return x.gather(1, indices).squeeze(1)
        else:
            return x


# orlaw = OdinsynthRLAgentWrapper().eval()
# ps1 = ProblemSpecification(
#      [
#         ['Ohio', 'Republican', 'Rep.', 'Gillmor', 'found', 'dead', 'in', 'PERSON', 'apartment', 'DATE', ',', 'Republican', 'aide', 'says'], 
#         ['Ohio', 'Rep.', 'Gillmor', 'found', 'dead', 'in', 'PERSON', 'apartment', 'DATE', ',', 'Republican', 'aide', 'says']
#     ],
#     [
#           {'docId': 'test', 'sentId': 0, 'start': 7, 'end': 10}, 
#           {'docId': 'test', 'sentId': 1, 'start': 6, 'end': 9}
#     ],
#     {
#         "word": ['PERSON', 'apartment', 'DATE'],
#         "lemma": ['person', 'apartment', 'date'],
#         "tag": ['NN', 'NNP', 'JJ'],
#     }
# )
# ps2 = ProblemSpecification(
#      [
#         ['Ohio', 'Republican', 'Rep.', 'Gillmor', 'found', 'dead', 'in', 'PERSON', 'apartment', 'DATE', ',', 'Republican', 'aide', 'says'], 
#         ['Ohio', 'Rep.', 'Gillmor', 'found', 'dead', 'in', 'PERSON', 'apartment', 'DATE', ',', 'Republican', 'aide', 'says'],
#         ['California', 'Rep.', 'Gillmor', 'found', 'dead', 'in', 'PERSON', 'apartment', 'DATE', ',', 'Republican', 'aide', 'says']
#     ],
#     [
#           {'docId': 'test', 'sentId': 0, 'start': 6, 'end': 10}, 
#           {'docId': 'test', 'sentId': 1, 'start': 5, 'end': 9},
#           {'docId': 'test', 'sentId': 2, 'start': 5, 'end': 9}
#     ],
#     {
#         "word": ['in', 'PERSON', 'apartment', 'DATE'],
#         "lemma": ['in', 'person', 'apartment', 'date'],
#         "tag": ['NN', 'NNP', 'JJ', 'IN'],
#     }
# )
# parser = QueryParser()

# class OdinsynthEnvStep:
#     def __init__(self, query: AstNode, problem_specification: ProblemSpecification):
#         self.query = query
#         self.problem_specification = problem_specification
    
#     def __str__(self):
#         return f"{self.query.pattern()} - {self.problem_specification}"

# obs0 = OdinsynthEnvStep(parser.parse('□'), ps1)
# obs1 = OdinsynthEnvStep(parser.parse('□'), ps2)
# obs2 = OdinsynthEnvStep(parser.parse('[word=□] □'), ps1)
# obs3 = OdinsynthEnvStep(parser.parse('[word=□] □'), ps2)
# obs4 = OdinsynthEnvStep(parser.parse('[word=in] □'), ps1)
# obs5 = OdinsynthEnvStep(parser.parse('[word=in] □'), ps2)

# import timeit
# from utils import init_random
# import numpy as np
# init_random(1)
# z = [obs0, obs1, obs2, obs3, obs4, obs5]
# # orlaw.forward([obs0, obs1, obs2, obs3, obs4, obs5], torch.device('cuda:0'), indices = [0, 0, 1, 0, 0])
# x = timeit.repeat(lambda: orlaw.forward(np.random.choice(z, 32), torch.device('cuda:0'), indices = [0, 0, 1, 0, 0]), number=5000, repeat=7)
# print(np.mean(x), np.std(x))
# x = timeit.repeat(lambda: orlaw.forward3(np.random.choice(z, 32), torch.device('cuda:0'), indices = [0, 0, 1, 0, 0]), number=5000, repeat=7)
# print(np.mean(x), np.std(x))
# print('--------------------------')
# x = timeit.repeat(lambda: orlaw.forward(np.random.choice(z, 16), torch.device('cuda:0'), indices = [0, 0, 1, 0, 0]), number=5000, repeat=7)
# print(np.mean(x), np.std(x))
# x = timeit.repeat(lambda: orlaw.forward3(np.random.choice(z, 16), torch.device('cuda:0'), indices = [0, 0, 1, 0, 0]), number=5000, repeat=7)
# print(np.mean(x), np.std(x))
# print('--------------------------')
# x = timeit.repeat(lambda: orlaw.forward(np.random.choice(z, 8), torch.device('cuda:0'), indices = [0, 0, 1, 0, 0]), number=5000, repeat=7)
# print(np.mean(x), np.std(x))
# x = timeit.repeat(lambda: orlaw.forward3(np.random.choice(z, 8), torch.device('cuda:0'), indices = [0, 0, 1, 0, 0]), number=5000, repeat=7)
# print(np.mean(x), np.std(x))
