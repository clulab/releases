import math
"""
Scheduling functions
Receives as first agument the current step (i.e. the value that would
be on the x-axis) and returns the value according to its underlying function
(i.e. the value that would be on the y-axis)
"""
def cosine_with_hard_restarts_schedule_with_warmup(current_step: int, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return 0.0
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

def cosine_with_hard_restarts_schedule_with_warmup_decayed(current_step: int, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, decay_factor = 0.75, min_value = 0.05):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return min_value
    if int(float(num_cycles) * progress) > 0:
        return max(min_value, min_value + 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))) * (decay_factor ** int(float(num_cycles) * progress)))
    else:
        return max(min_value, min_value + 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

def triangular_schedule(current_step, cycle_length, min_value = 0.05, max_value = 1.0):
    slope = (max_value - min_value) / (cycle_length//2)
    cycle_position = current_step % cycle_length
    # Ascending
    if cycle_position < (cycle_length//2):
        return slope * cycle_position + min_value
    # Descending
    else:
        return -slope * (cycle_position - cycle_length//2) + max_value

def triangular_schedule_with_decay(current_step, cycle_length, min_value = 0.05, max_value = 1.0, decay_factor=0.75):
    slope = (max_value - min_value) / (cycle_length//2)
    cycle_position = current_step % cycle_length
    cycle_number   = current_step // cycle_length
    if cycle_number > 0:
        max_value = max_value * (decay_factor ** cycle_number)
        slope = (max_value - min_value) / (cycle_length//2)
    # Ascending
    if cycle_position < (cycle_length//2):
        return slope * cycle_position + min_value
    # Descending
    else:
        return -slope * (cycle_position - cycle_length//2) + max_value
        # return max(min_value, float(num_training_steps - current_step) / float(max(max_value, num_training_steps - (cycle_length//2))))

def linear_step(current_step, num_training_steps, min_value = 0.05, max_value = 1.0):
    slope = (max_value - min_value) / num_training_steps
    return max(slope * current_step + min_value, min(min_value, max_value))

# Like in cleanrl implementation
# Useful for 1:1 comparisons
def linear_schedule(current_step: int, start_e: float, end_e: float, duration: int):
    slope =  (end_e - start_e) / duration
    return max(slope * current_step + start_e, end_e)

# The current_step is unused, as the same value is returned regardless
# It was added to make every method similar: takes the current_step and potentially more data
def single_value(current_step: int, value: float):
    return value

# The key in the dictionary is the timestep when that scheduler will end
# To not be confused with the length
def multi_schedulers(current_step: int, schedulers_dict: dict, scheduler_params_dict: dict):
    prev = 0
    for (key, scheduler) in schedulers_dict.items():
        if current_step < key:
            return scheduler(current_step-prev, **scheduler_params_dict[key])
        else:
            prev += key
    return 0


# ---------------
# [] [] -> [] [] [], []? []
# Queue: {([] [] [], 3.0), ([]? [], 1.0)}
# ---------------



# [], [] [], 'whose', 'position', 'requires'
# [], [] [], 'whose', 'purpose' , 'is'
# [], [] [], 'whose', 'real'    , 'purpose', 'is'

# 'whose',
# 'position',
# 'requires',
# 'real',
# 'purpose',
# 'is'



# from rl_environment import OdinsynthEnvFactory, OdinsynthEnvWrapper
# from rl_agent import OdinsynthRLAgentWrapper
# import torch
# import torch.nn as nn
# import numpy as np

# init_random(4)
# device = torch.device('cuda:0')
# agent = OdinsynthRLAgentWrapper(device).eval()
# agent.load_state_dict(torch.load('/home/rvacareanu/projects/odinsynth/python/results/rl/from_pretrained_rules100k_q_network_54999.pt'))
# env = OdinsynthEnvWrapper(OdinsynthEnvFactory.from_file("/data/nlp/corpora/odinsynth/data/rules100k/")) #ProcessObsInputEnv(gym.make(args.gym_id))
# # env = OdinsynthEnvWrapper(OdinsynthEnvFactory.from_file("/data/nlp/corpora/odinsynth/data/toy/")) #ProcessObsInputEnv(gym.make(args.gym_id))
# obs = env.reset()
# print(obs.problem_specification)
# for sent, spec in zip(obs.problem_specification.sentences, sorted(obs.problem_specification.specs, key=lambda x: x['sentId'])):
#     print(sent[spec['start']:spec['end']])
# print("\n\n")
# print(obs.problem_specification)
# print(obs.query.pattern())

# logits = agent.forward([obs])[0][0]
# action = np.argmax(logits, axis=-1)
# next_obs, reward, done, _ = env.step(action)
# print(next_obs.query.pattern(), reward)


# logits = agent.forward([next_obs])[0][0]
# action = np.argmax(logits, axis=-1)
# next_obs, reward, done, _ = env.step(action)
# print(next_obs.query.pattern(), reward)


# logits = agent.forward([next_obs])[0][0]
# action = np.argmax(logits, axis=-1)
# next_obs, reward, done, _ = env.step(action)
# print(next_obs.query.pattern(), reward)

# logits = agent.forward([next_obs])[0][0]
# action = np.argmax(logits, axis=-1)
# next_obs, reward, done, _ = env.step(action)
# print(next_obs.query.pattern(), reward)

# logits = agent.forward([next_obs])[0][0]
# action = np.argmax(logits, axis=-1)
# next_obs, reward, done, _ = env.step(action)
# print(next_obs.query.pattern(), reward)

# logits = agent.forward([next_obs])[0][0]
# action = np.argmax(logits, axis=-1)
# next_obs, reward, done, _ = env.step(action)
# print(next_obs.query.pattern(), reward)

# logits = agent.forward([next_obs])[0][0]
# action = np.argmax(logits, axis=-1)
# next_obs, reward, done, _ = env.step(action)
# print(next_obs.query.pattern(), reward)

# logits = agent.forward([next_obs])[0][0]
# action = np.argmax(logits, axis=-1)
# next_obs, reward, done, _ = env.step(action)
# print(next_obs.query.pattern(), reward)

# logits = agent.forward([next_obs])[0][0]
# action = np.argmax(logits, axis=-1)
# next_obs, reward, done, _ = env.step(action)
# print(next_obs.query.pattern(), reward)

# logits = agent.forward([next_obs])[0][0]
# action = np.argmax(logits, axis=-1)
# next_obs, reward, done, _ = env.step(action)
# print(next_obs.query.pattern(), reward)

# logits = agent.forward([next_obs])[0][0]
# action = np.argmax(logits, axis=-1)
# next_obs, reward, done, _ = env.step(action)
# print(next_obs.query.pattern(), reward)

# logits = agent.forward([next_obs])[0][0]
# action = np.argmax(logits, axis=-1)
# next_obs, reward, done, _ = env.step(action)
# print(next_obs.query.pattern(), reward)

# logits = agent.forward([next_obs])[0][0]
# action = np.argmax(logits, axis=-1)
# next_obs, reward, done, _ = env.step(action)
# print(next_obs.query.pattern(), reward)

# logits = agent.forward([next_obs])[0][0]
# action = np.argmax(logits, axis=-1)
# next_obs, reward, done, _ = env.step(action)
# print(next_obs.query.pattern(), reward)

# logits = agent.forward([next_obs])[0][0]
# action = np.argmax(logits, axis=-1)
# next_obs, reward, done, _ = env.step(action)
# print(next_obs.query.pattern(), reward)

# logits = agent.forward([next_obs])[0][0]
# action = np.argmax(logits, axis=-1)
# next_obs, reward, done, _ = env.step(action)
# print(next_obs.query.pattern(), reward)









# total_timesteps = 20000

# schedulers_dict = {
#     1*total_timesteps/10: single_value,
#     8*total_timesteps/10: triangular_schedule_with_decay,
#     1*total_timesteps/10: single_value 
# }
# schedulers_params_dict = {
#     1*total_timesteps/10: {
#         'value': 0.0,
#     },
#     # starts at 1*total_timesteps/10 and ends at 8*total_timesteps/10, thus
#     # running for only 7*total_timesteps/10
#     8*total_timesteps/10: {
#         'cycle_length'      : 7*total_timesteps/10,
#         'min_value'         : 0.05,
#         'max_value'         : 1.0,
#         'decay_factor'      : 0.5,
#     },
#     # the key of the last one does not matter
#     1*total_timesteps/10: {
#         'value': 0.0,
#     },
# }
# # 

# min_value = 0.05
# max_value = 1.0
# print(schedulers_dict)
# print(schedulers_params_dict)
# result = []
# for i in range(total_timesteps):
#     result.append(multi_schedulers(i, schedulers_dict, schedulers_params_dict))


    
# import matplotlib.plt as plt
# plt.plot(range(total_timesteps), result)
# plt.hlines(min_value, 0, total_timesteps)
# plt.hlines(max_value, 0, total_timesteps)
# plt.show()
# plt.clf()



