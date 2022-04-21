from dataclasses import dataclass
import math
from typing import Any, Optional
from rl_utils import ProblemSpecification
from queryast import AstNode
from typing import Callable

"""
Data classes and function classes to calculate the reward
Used in rl_environment.py
"""
@dataclass
class RewardCalculatorParameters:
    prev_query:            Optional[AstNode]
    current_query:         Optional[AstNode]
    current_step:          Optional[int]
    problem_specification: Optional[ProblemSpecification]
    is_solution:           Optional[bool]
    is_compromised:        Optional[bool]
    odinson_metadata:      Optional[dict]
    metadata:              Optional[dict]
    


class RewardCalculator:
    def calculate_reward(self, rcp: RewardCalculatorParameters, **kwargs):
        pass

    def __call__(self, rcp: RewardCalculatorParameters, **kwargs):
        return self.calculate_reward(rcp, **kwargs)

class SimpleRewardCalculator(RewardCalculator):
    def __init__(self):
        super().__init__()

    def calculate_reward(self, rcp: RewardCalculatorParameters, **kwargs):
        reward = -0.1
        
        if rcp.is_solution:
            reward = 7
        elif rcp.is_compromised:
            reward = -7
        elif rcp.current_query.is_valid_query():
            reward = -7

        return reward

class ParameterizedSimpleRewardCalculator(RewardCalculator):
    def __init__(self, step_reward, good_end_reward, bad_end_reward) -> None:
        super().__init__()
        self.step_reward     = step_reward
        self.good_end_reward = good_end_reward
        self.bad_end_reward  = bad_end_reward

    def calculate_reward(self, rcp: RewardCalculatorParameters, **kwargs):
        reward = self.step_reward
        
        if rcp.is_solution:
            reward = self.good_end_reward
        elif rcp.is_compromised:
            reward = self.bad_end_reward
        elif rcp.current_query.is_valid_query():
            reward = self.bad_end_reward

        return reward

"""
Like ParameterizedSimpleRewardCalculator, but the penalty per step increases
"""
class IncreasingStepPenaltyRewardCalculator(RewardCalculator):
    def __init__(self, step_reward, good_end_reward, bad_end_reward) -> None:
        super().__init__()
        self.step_reward     = step_reward
        self.good_end_reward = good_end_reward
        self.bad_end_reward  = bad_end_reward

    def calculate_reward(self, rcp: RewardCalculatorParameters, **kwargs):
        reward = math.log10(self.steps + 10) * self.step_reward
        
        if rcp.is_solution:
            reward = self.good_end_reward
        elif rcp.is_compromised:
            reward = self.bad_end_reward
        elif rcp.current_query.is_valid_query():
            reward = self.bad_end_reward


        return reward

    
"""
Like ParameterizedSimpleRewardCalculator, but reward partially during search (depending on how much we match)
"""
class PartialRewardCalculator(RewardCalculator):
    def __init__(self, step_reward, good_end_reward, bad_end_reward, partial_reward_multiplier) -> None:
        super().__init__()
        self.step_reward               = step_reward
        self.good_end_reward           = good_end_reward
        self.bad_end_reward            = bad_end_reward
        self.partial_reward_multiplier = partial_reward_multiplier

    def calculate_reward(self, rcp: RewardCalculatorParameters, **kwargs):
        reward = self.step_reward
        if rcp.odinson_metadata['partial_reward'] > rcp.metadata['previous_reward']:
            reward = self.step_reward + self.partial_reward_multiplier * rcp.odinson_metadata['partial_reward'] 

        if rcp.is_solution:
            reward = self.good_end_reward
        elif rcp.is_compromised:
            reward = self.bad_end_reward
        elif rcp.current_query.is_valid_query():
            reward = self.bad_end_reward

        return reward    
"""
Like PartialRewardCalculator, but the partial reward is the difference between last partial reward and current partial reward
"""
class PartialDeltaRewardCalculator(RewardCalculator):
    def __init__(self, step_reward, good_end_reward, bad_end_reward, partial_reward_multiplier) -> None:
        super().__init__()
        self.step_reward               = step_reward
        self.good_end_reward           = good_end_reward
        self.bad_end_reward            = bad_end_reward
        self.partial_reward_multiplier = partial_reward_multiplier

    def calculate_reward(self, rcp: RewardCalculatorParameters, **kwargs):
        reward = self.step_reward
        if rcp.odinson_metadata['partial_reward'] > rcp.metadata['previous_reward']:
            reward = self.step_reward + self.partial_reward_multiplier * (rcp.odinson_metadata['partial_reward'] - rcp.metadata['previous_reward'])

        if rcp.is_solution:
            reward = self.good_end_reward
        elif rcp.is_compromised:
            reward = self.bad_end_reward
        elif rcp.current_query.is_valid_query():
            reward = self.bad_end_reward

        return reward

"""
The lambda parameter specifies how to calculate the reward
"""
class LambdaRewardCalculator(RewardCalculator):
    def __init__(self, lambda_on_rcp: Callable[[RewardCalculatorParameters, Any], float]) -> None:
        super().__init__()
        self.func = lambda_on_rcp

    def calculate_reward(self, rcp: RewardCalculatorParameters, **kwargs):
        return self.func(rcp, **kwargs)