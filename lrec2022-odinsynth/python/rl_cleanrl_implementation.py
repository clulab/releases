# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

from queryast import HoleQuery
from rl_utils import PrioritizedReplayBuffer, ReplayBuffer
from schedulers import cosine_with_hard_restarts_schedule_with_warmup_decayed, multi_schedulers, single_value, triangular_schedule_with_decay
from rl_environment import OdinsynthEnv, OdinsynthEnvFactory, OdinsynthEnvStep, OdinsynthEnvWrapper
from rl_agent import OdinsynthRLAgentWrapper
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import tqdm

import argparse
from distutils.util import strtobool
import numpy as np
import time
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="odinsynth",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=3e-6,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=20000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    
    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=50000,
                         help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=500,
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=32,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--start-e', type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.0001,
                        help="the ending epsilon for exploration")
    parser.add_argument('--exploration-fraction', type=float, default=0.6,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument('--learning-starts', type=int, default=20000,
                        help="timestep to start learning")
    parser.add_argument('--train-frequency', type=int, default=1,
                        help="the frequency of training")
    parser.add_argument('--accumulate-gradient', type=int, default=64,
                        help="for how many steps to accumulate the gradient")
    parser.add_argument('--test-every', type=int, default=10000,
                        help="test every time this number of steps has passed")
    parser.add_argument('--savepath', type=str, default="/home/rvacareanu/projects/odinsynth/python/results/rl/from_pretrained_default",
                        help="where to save")
    parser.add_argument('--datapath', type=str, default="/data/nlp/corpora/odinsynth/data/rules100k/",
                        help="where is the data")
    parser.add_argument('-ad', '--additonal-details', type=str, default=None,
                        help="More details about this run")
    parser.add_argument('--reward-step', type=float, default=-0.1,
                        help="Reward at every step")
    parser.add_argument('--reward-solution', type=float, default=7,
                        help="Reward if it is solution")
    parser.add_argument('--reward-partial-solution-multiplier', type=float, default=1,
                        help="Reward for a partial solution")
    parser.add_argument('--reward-compromised', type=float, default=-7,
                        help="Reward if compromised")
    parser.add_argument('--loss-type', type=str, default='l1',
                        help="What type of loss to use")
    parser.add_argument('--replay-buffer-type', type=str, default='per',
                        help="What type of replay buffer to use. One of {'per', 'normal'}")
    parser.add_argument('--log-weights', action='store_true',
                        help="Log the weights as histogram if the flag is set")
    parser.add_argument('--log-gradients', action='store_true',
                        help="Log the gradients as histogram if the flag is set. Before clip and after clip")

    args = parser.parse_args()

class DQNAgent():
    def __init__(self, hparams = {}) -> None:
        self.hparams = hparams
        self.device = torch.device('cuda:0')

        # The replay memory
        self.memory = replay_buffer_type[self.hparams['replay_buffer_type']](hparams['buffer_size'])

        # The networks
        self.q_network = OdinsynthRLAgentWrapper(device=self.device)
        self.target_network = OdinsynthRLAgentWrapper(device=self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # self.optimizer = optim.SGD(self.q_network.parameters(), lr=args.learning_rate)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        # self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr = 3e-5, max_lr = 1e-3, mode='triangular2', cycle_momentum = False, step_size_up=self.hparams['total_timesteps']//2)
        self.loss_fn = nn.MSELoss()

        train_env_factory, test_env_factory = OdinsynthEnvFactory.get_train_test_factories(hparams.get('datapath', '/data/nlp/corpora/odinsynth/data/rules100k/'), test_size = hparams.get('test_size', 0.005))

        self.train_env_factory = OdinsynthEnvFactory(train_env_factory.problem_specifications[:10000])
        self.test_env_factory  = OdinsynthEnvFactory(train_env_factory.problem_specifications[:1000])

        # Specify the rewards. Useful for exploration
        env_params = {
            'reward_step'            : self.hparams['reward_step'],
            'reward_solution'        : self.hparams['reward_solution'],
            'reward_partial_solution_multiplier': self.hparams['reward_partial_solution_multiplier'],
            'reward_compromised'     : self.hparams['reward_compromised'],
        }
        self.train_env = OdinsynthEnvWrapper(train_env_factory, env_params)
        self.test_env  = OdinsynthEnvWrapper(test_env_factory, env_params)
        self.loss_fn   = losses[self.hparams['loss_type']]

        self.beta = 0.6

    def train(self):
        
        # The key in the dictionary is the timestep when that scheduler will end
        # To not be confused with the length
        # This also mean that the key of the last one doesn't matter
        # The first line will apply until we reach 10 * self.hparams['total_timesteps']//100
        # exploitation = [(x * self.hparams['total_timesteps']//100, single_value) for x in range(0, 10)]
        schedulers_dict = {}
        schedulers_params_dict = {}
        for x in range(self.hparams['total_timesteps']//100):
            idx = (x + 1) * 1000
            schedulers_dict[idx] = single_value
            if (x+1) % 6 == 0:
                schedulers_params_dict[idx] = {'value': 0.15}
            else:
                schedulers_params_dict[idx] = {'value': 0   }


        episode_reward = 0
        total_reward = 0
        prior_eps = 1e-6
        obs = self.train_env.reset()
        for global_step in tqdm.tqdm(range(self.hparams['total_timesteps'])):
        # for global_step in range(self.hparams['total_timesteps']):
            # ALGO LOGIC: put action logic here
            epsilon = multi_schedulers(global_step, schedulers_dict, schedulers_params_dict) #linear_schedule(self.hparams['start_e'], self.hparams['end_e'], self.hparams['exploration_fraction']*self.hparams['total_timesteps'], global_step)
            fraction = min(global_step / self.hparams['total_timesteps'], 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            if random.random() < epsilon:
                # with torch.no_grad():
                    # values = self.q_network.forward([obs])
                    # values = torch.tensor(values)
                    # values = F.softmax(values, dim=-1)[0][0].detach().cpu().numpy()
                    # action = np.random.choice(list(range(values.shape[0])), 1, p=values)[0]
                action = self.train_env.action_space.sample()
            else:
                logits = self.q_network.forward([obs], return_list = True)[0] # the [0] indexing is done because we run the q_network with a list consisting of a single element
                action = np.argmax(logits, axis=-1)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, _ = self.train_env.step(action)
            episode_reward += reward
            
            # ALGO LOGIC: training.
            self.memory.store(obs, action, reward, next_obs, done)
            
            if global_step > self.hparams['learning_starts'] and global_step % self.hparams['train_frequency'] == 0:
                erb = self.memory.sample(self.hparams['batch_size'], {'beta': self.beta})
                with torch.no_grad():
                    tn_forward = self.target_network.forward(erb.next_obs, return_list = True)
                    target_max = torch.tensor([np.max(a[0]) for a in tn_forward]).to(self.device) # torch.max(target_network.forward(s_next_obses), dim=1)[0]
                    td_target  = torch.tensor(erb.rews).to(self.device) + self.hparams['gamma'] * target_max * (1 - torch.tensor(erb.done).to(self.device).int()).double()

                old_val = self.q_network.forward(erb.obs, indices=erb.acts).double()

                elementwise_loss = self.loss_fn(old_val, td_target, reduction='none')
                if self.hparams['replay_buffer_type'] == 'per':
                    loss = torch.mean(elementwise_loss * torch.tensor(erb.weights).to(self.device))
                else:
                    loss = torch.mean(elementwise_loss)
                    
                    
                # print('\n')
                # print('-----------')
                # print(tn_forward)
                # print(target_max)
                # print(td_target)
                # print(old_val)
                # print(loss)
                # print('-----------')
                # print('\n')

                # if global_step > 1010:
                #     exit()

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)

                # optimize the model
                loss.backward()

                if self.hparams['log_gradients']:
                    for tag, parm in self.q_network.named_parameters():
                        writer.add_histogram(f'Before_Clip/{tag}', parm.grad.data.cpu().numpy(), global_step)
                if self.hparams['log_weights']:
                    for tag, parm in self.q_network.named_parameters():
                        writer.add_histogram(f'Weight/{tag}', parm.data.cpu().numpy(), global_step)

                nn.utils.clip_grad_norm_(list(self.q_network.parameters()), self.hparams['max_grad_norm'])

                if self.hparams['log_gradients']:
                    for tag, parm in self.q_network.named_parameters():
                        writer.add_histogram(f'Before_Clip/{tag}', parm.grad.data.cpu().numpy(), global_step)

                self.optimizer.step()
                self.optimizer.zero_grad()
                # self.lr_scheduler.step()

                if self.hparams['replay_buffer_type'] == 'per':
                    self.memory.update({'loss_for_prior': elementwise_loss.detach().cpu().numpy(), 'prior_eps': prior_eps, 'indices': erb.indices})
                

                # update the target network
                if global_step % self.hparams['target_network_frequency'] == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook 
            obs = next_obs

            if done:
                # TRY NOT TO MODIFY: record rewards for plotting purposes
                total_reward += episode_reward

                writer.add_scalar("charts/episode_reward", episode_reward, global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)

                obs, episode_reward = self.train_env.reset(), 0

            if (global_step == 0 or global_step > self.hparams['learning_starts']) and global_step % self.hparams.get('test_every', 1000) == 0:
                # print(f"global_step={global_step}, episode_reward={total_reward/accumulation_counter}")
                solved_percentage, steps, mean_reward = self.test()
                torch.save(self.q_network.state_dict(), f"{self.hparams['savepath']}_q_network_{global_step}_{solved_percentage}.pt")
                torch.save(self.target_network.state_dict(), f"{self.hparams['savepath']}_target_network_{global_step}_{solved_percentage}.pt")
                writer.add_scalar("charts/test_result_solved", solved_percentage, global_step)
                writer.add_scalar("charts/test_result_steps", steps, global_step)
                writer.add_scalar("charts/test_result_meanreward", mean_reward, global_step)
                self.q_network.train()
                
                
        torch.save(self.q_network.state_dict(), f"{self.hparams['savepath']}_q_network.pt")
        torch.save(self.target_network.state_dict(), f"{self.hparams['savepath']}_target_network.pt")
        print(total_reward/self.hparams['total_timesteps'])
        self.train_env.close()
        writer.close()
        print(f"runs/{experiment_name}")

            

    def save_model(self):
        torch.save(self.q_network.state_dict(), f"{self.hparams['savepath']}_q_network5.pt")
        torch.save(self.target_network.state_dict(), f"{self.hparams['savepath']}_target_network5.pt")


    """
    Attempt to solve the environment by exploiting (taking 
    the next_state with the highest score) the q network trained so far
    Returns: (<solved or not>, <number of steps>, <total reward obtained>)
    """
    def solve_env(self, env: OdinsynthEnv, start_obs): 
        obs, reward, done, metadata = start_obs, None, False, None
        agent = self.q_network.eval()
        steps = 0
        total_reward = 0
        while not done:
            steps += 1
            with torch.no_grad():
                logits = agent.forward([obs], return_list = True)[0] # the [0] indexing is done because we run the q_network with a list consisting of a single element
            action = np.argmax(logits, axis=-1)
            obs, reward, done, metadata = env.step(action)
            total_reward += reward
            if metadata['solution']:
                done = True # Not necessary
                return (True, steps, total_reward)
            elif metadata['compromised']:
                done = True # Not necessary
                return (False, steps, total_reward)
            elif obs.query.is_valid_query():
                done = True # Not necessary
                return (False, steps, total_reward)


    """
        Test the q_network on the test environment
        Returns the number of solved environments, the mean number of steps per environment and the mean reward
    """
    def test(self):
        solved = []
        steps  = []
        reward = []
        for ps in self.test_env_factory.problem_specifications:
            result = self.solve_env(OdinsynthEnv(ps), OdinsynthEnvStep(HoleQuery(), ps))
            solved.append(result[0])
            steps.append(result[1])
            reward.append(result[2])

        return len([x for x in solved if x]) / len(solved), np.mean(steps), np.mean(reward)





from utils import init_random
init_random(1)

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
savepath_split = args.savepath.split('/')[-1]
writer = SummaryWriter(f"runs_hp/{experiment_name}_{savepath_split}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
print(f"runs_hp/{experiment_name}")

losses = {
    'mse': F.mse_loss,
    'l1' : F.l1_loss,
}

replay_buffer_type = {
    'per': PrioritizedReplayBuffer,
    'normal' : ReplayBuffer,
}

# if args.prod_mode:
#     import wandb
#     wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
#     writer = SummaryWriter(f"/tmp/{experiment_name}")

# if args.capture_video:
#     env = Monitor(env, f'videos/{experiment_name}')


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

"""
    :param current_step - the current step
    :param init_value   - the start value
    :param decay_steps  - the total number of steps
    :param decay_rate   - the rate at which to decay
    :param min_value    - the minimum value


    >>> exp_scheduler(1,     1.0, 10000, 0.1, 0.01)
    0.9997697679981565
    >>> exp_scheduler(2,     1.0, 10000, 0.1, 0.01)
    0.9995395890030878
    >>> exp_scheduler(10,    1.0, 10000, 0.1, 0.01)
    0.9977000638225533
    >>> exp_scheduler(100,   1.0, 10000, 0.1, 0.01)
    0.9772372209558107
    >>> exp_scheduler(1000,  1.0, 10000, 0.1, 0.01)
    0.7943282347242815
    >>> exp_scheduler(10000, 1.0, 10000, 0.1, 0.01)
    0.1

"""
def exp_scheduler(current_step, init_value, decay_steps, decay_rate, min_value):
    v = init_value * decay_rate ** (current_step/decay_steps)
    v = max(v, min_value)
    return v



print(vars(args))
dqn_agent = DQNAgent(vars(args))
dqn_agent.train()



