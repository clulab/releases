from collections import defaultdict
import json
from rl_reward import RewardCalculator, RewardCalculatorParameters, SimpleRewardCalculator

import tqdm
from rl_utils import ProblemSpecification, OdinsynthEnvStep

from queryast import AstNode, HoleQuery
import numpy as np

import glob
import requests
from typing import List
import gym
from gym.spaces.discrete import Discrete
import math

"""
    Creates other OdinsynthEnv
"""
class OdinsynthEnvFactory:
    def __init__(
        self, 
        problem_specifications: List[ProblemSpecification]
    ):
        self.problem_specifications =problem_specifications

    def create_env(self, **kwargs):
        return OdinsynthEnv(np.random.choice(self.problem_specifications), **kwargs)

    @staticmethod
    def from_file(path: str, allowed_fields = {"word", "tag", "lemma"}, condition = None):
        # We only need the specs
        specs_path = sorted(glob.glob(f'{path}/specs/*.spec.json'))
        problem_specifications = []
        for s in tqdm.tqdm(specs_path):
            with open(s) as fin:
                lines = fin.readlines()
                specs = json.loads(lines[0])
                doc   = json.loads(lines[1])

                
                # Extract vocabulary from doc
                vocabulary = defaultdict(set)
                sentences = []
                if condition is None or condition(specs): # whether no condition was supplied or the condition matches
                # if max([x['end'] - x['start'] for x in specs['specs']]) < 5:
                    for sen, sp in zip(doc['sentences'], sorted(specs['specs'], key=lambda x: x['sentId'])):
                        for field in sen['fields']:
                            if field['name'] == 'word':
                                sentences.append(field['tokens'])
                            if field['name'] in allowed_fields:
                                vocabulary[field['name']].update(field['tokens'][sp['start']:sp['end']])

                    problem_specifications.append(ProblemSpecification(sentences, specs['specs'], vocabulary))

        return OdinsynthEnvFactory(problem_specifications)

    @staticmethod
    def get_train_test_factories(path, test_size = 0.25, random_state = 1, condition = lambda sp: max([x['end'] - x['start'] for x in sp['specs']]) < 5):
        from sklearn.model_selection import train_test_split
        factory = OdinsynthEnvFactory.from_file(path, condition = condition)
        train, test = train_test_split(factory.problem_specifications, test_size=test_size, random_state = random_state)
        return (OdinsynthEnvFactory(train), OdinsynthEnvFactory(test))

"""
    Wraps an OdinsynthEnv
    It allows the underlying environment to change at the end of the epoch.
    An alternative would have been to allow the underlying environment
    to mutate itself to a new environment, but it is not very elegant
    Passes the reward calculator object to the environment to calculate
    the reward
"""
class OdinsynthEnvWrapper(gym.Env):
    def __init__(self, factory: OdinsynthEnvFactory, reward_calculator = SimpleRewardCalculator(), env_params = {}):
        self.factory           = factory
        self.env               = self.factory.create_env(reward_calculator = reward_calculator)
        self.action_space      = self.env.action_space
        self.reward_calculator = reward_calculator
        self.env_params        = env_params

    # def _observation(self):
        # return self.env._observation()
        
    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env = self.factory.create_env(reward_calculator = self.reward_calculator, env_params = self.env_params)
        self.action_space = self.env.action_space
        

        return self.env.reset()

"""
    A space where the number of actions depend on the current state
"""
class StatefulOdinsynthDiscrete(gym.Space):
    def __init__(self, n, state, vocabulary):
        assert n >= 0
        self.n          = n
        self.state      = state
        self.vocabulary = vocabulary
        super(StatefulOdinsynthDiscrete, self).__init__((), np.int64)

    def sample(self):
        x = np.random.randint(len(self.state.next_nodes_filtered(self.vocabulary)))
        # print(self.state.pattern(), len(self.state.next_nodes(self.vocabulary)), x)
        return x

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.char in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n and as_int < len(self.state.next_nodes(self.vocabulary))

    def set_state(self, state):
        self.state = state
        # print("Set a new state", state.pattern(), self.state.pattern())

    def __repr__(self):
        return f"StatefulOdinsynthDiscrete({self.n}) with state {self.state}"

    def __eq__(self, other):
        return isinstance(other, Discrete) and self.n == other.n and self.state == other.state


"""
    An environment. Usually only single-used. Created with the factory
    A reward calculator object specifies how the reward is calculated.
    Note that when reset is called, a new environment is created (@see OdinsynthEnvWrapper).
    
"""
class OdinsynthEnv(gym.Env):
    def __init__(self, problem: ProblemSpecification, start_query: AstNode = HoleQuery(), reward_calculator: RewardCalculator = SimpleRewardCalculator(), env_params = {}):
        super().__init__()
        self.problem           = problem
        self.query: AstNode    = start_query
        self.current_valid_qp  = ""
        self.start_query       = start_query
        self.action_space      = StatefulOdinsynthDiscrete(100000, self.start_query, self.problem.vocabulary)
        self.observation_space = None # Observation space should not be needed for DQN
        self.steps             = 0
        self.previous_reward   = 0.0
        self.reward_calculator = reward_calculator
        self.env_params        = env_params

    def _observation(self):
        return self

    def call_odinson(self, endpoint = 'http://127.0.0.1:9001/isSolution'):
        response = requests.post(endpoint,
            headers = {"Content-Type": "application/json"},
            json = self.problem.construct_query(self.query.pattern().replace("\\\"", "")) # Replace for cases like "\"'\"" into "'"
        ).text
        jsonResponse = json.loads(response)
        return jsonResponse

    def step(self, action):
        self.steps += 1
        next_nodes = self.query.next_nodes_filtered(self.problem.vocabulary) # list(filter(lambda l: filter_query_holes(l), self.query.next_nodes(self.problem.vocabulary))) # self.query.next_nodes(self.problem.vocabulary)
        if len(next_nodes) <= action:
            raise ValueError(f"The action {action} is outside the range of valid actions, which is {len(next_nodes)}.\nCurrent node: {self.query.pattern()}\nProblem: self.problem\nNext nodes: [x.pattern() for x in next_nodes]")
        
        previous_query = self.query

        self.query = next_nodes[action]
        self.action_space.set_state(self.query)

        odinson_result = self.call_odinson()
        rcp = RewardCalculatorParameters(
                prev_query            = previous_query,
                current_query         = self.query,
                current_step          = self.steps,
                problem_specification = self.problem,
                is_solution           = odinson_result['solution'],
                is_compromised        = odinson_result['compromised'],
                odinson_metadata      = odinson_result,
                metadata              = {'previous_reward': self.previous_reward}
            )
        reward = self.reward_calculator(rcp)


        done = False        
        
        self.previous_reward = odinson_result['partial_reward']

        # We found the solution
        if odinson_result['solution']:
            done   = True
        # We cannot find the solution
        elif odinson_result['compromised']:
            done   = True
        elif self.query.is_valid_query():
            done   = True
        
        
        # The partial query is the same as the previous partial query, 
        # which means that nothing new was matched from the spec
        # (Note: This doesn't mean that we are lost; A solution might be found)
        # if odinson_result['partial_query'] == self.current_valid_qp:
            # print(4, OdinsynthEnvStep(self.query, self.problem), reward, done, {})
        return OdinsynthEnvStep(self.query, self.problem), reward, done, {'solution': odinson_result['solution'], 'compromised': odinson_result['compromised'], 'f1': odinson_result['partial_reward']}
  
    def reset(self):
        self.query = self.start_query
        self.action_space.set_state(self.query)
        self.steps = 0
        return OdinsynthEnvStep(self.query, self.problem)








# # init_random(1)
# # agent = DqnAgent(ModelCls=OdinsynthRLAgentWrapper, model_kwargs={}) # CatDqnAgent
# import tianshou as ts
# import torch

# oef        = OdinsynthEnvFactory.from_file("/data/nlp/corpora/odinsynth/data/toy/")
# train_envs = oef.create_env()


# test_envs  = oef.create_env()
# net        = OdinsynthRLAgentWrapper()
# optim      = torch.optim.Adam(net.parameters(), lr=1e-3)

# policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)
# train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 1), exploration_noise=True)
# test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

# result = ts.trainer.offpolicy_trainer(
#     policy, train_collector, test_collector,
#     max_epoch=10, step_per_epoch=10000, step_per_collect=1,
#     update_per_step=0.1, episode_per_test=100, batch_size=64,
#     train_fn=lambda epoch, env_step: policy.set_eps(0.1),
#     test_fn=lambda epoch, env_step: policy.set_eps(0.05),
#     stop_fn=lambda mean_rewards: mean_rewards >= 100)

# exit()
# import ray
# from ray.rllib.models import ModelCatalog
# ray.init()
# ModelCatalog.register_custom_model("my_model", OdinsynthRLAgentWrapper)

# oef   = OdinsynthEnvFactory.from_file("/data/nlp/corpora/odinsynth/data/toy/")
# model = OdinsynthRLAgentWrapper()

# config = {
#     "env": OdinsynthEnvFactory,  # or "corridor" if registered above
#     "env_config": {
#         "path": "/data/nlp/corpora/odinsynth/data/toy/",
#     },
#     # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
#     "num_gpus": 0,#int(os.environ.get("RLLIB_NUM_GPUS", "0")),
#     "model": {
#         "custom_model": "my_model",
#         "vf_share_layers": True,
#     },
#     "num_workers": 1,  # parallelism
#     "framework": "torch",
# }

# stop = {
#     "training_iteration": 10,
#     "timesteps_total": 10,
#     "episode_reward_mean": 0.1,
# }
# from ray import tune
# results = tune.run("DQN", config=config, stop=stop)




# # print(oe.action_space)
# # print(oe.observation_space)
# # print(oe.problem)
# # print(oe.metadata)
# # print(oe.query)