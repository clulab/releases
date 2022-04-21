import argparse
import os
import collections
from torch.utils.tensorboard.writer import SummaryWriter

import tqdm
from schedulers import multi_schedulers
from queryast import HoleQuery
import random

from torch.nn.utils.clip_grad import clip_grad_norm_
from rl_agent import OdinsynthRLAgentWrapper
import numpy as np
from rl_environment import OdinsynthEnv, OdinsynthEnvFactory, OdinsynthEnvStep, OdinsynthEnvWrapper
from rl_utils import SumSegmentTree, MinSegmentTree
from collections import deque
from typing import Deque, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self, 
        size: int, 
        batch_size: int = 32, 
        n_step: int = 1, 
        gamma: float = 0.99
    ):
        self.obs_buf: List[OdinsynthEnvStep] = [None] * size
        self.next_obs_buf: List[OdinsynthEnvStep] = [None] * size
        self.acts_buf: List[int] = [None] * size
        self.rews_buf: List[float] = [None] * size
        self.done_buf: List[bool] = [None] * size
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self, 
        obs: OdinsynthEnvStep, 
        act : int, 
        rew: float, 
        next_obs: OdinsynthEnvStep, 
        done: bool,
    ) -> Tuple[OdinsynthEnvStep, int, float, OdinsynthEnvStep, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for idx in idxs:
            s_lst.append(self.obs_buf[idx])
            a_lst.append(self.acts_buf[idx])
            r_lst.append(self.rews_buf[idx])
            s_prime_lst.append(self.next_obs_buf[idx])
            done_mask_lst.append(self.done_buf[idx])

        return ExperienceReplayBatch(
            np.array(s_lst), 
            np.array(a_lst),
            np.array(r_lst), 
            np.array(s_prime_lst),
            np.array(done_mask_lst),
            )
    
    def sample_batch_from_idxs(
        self, idxs: np.ndarray
    ):
        # for N-step Learning
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for idx in idxs:
            s_lst.append(self.obs_buf[idx])
            a_lst.append(self.acts_buf[idx])
            r_lst.append(self.rews_buf[idx])
            s_prime_lst.append(self.next_obs_buf[idx])
            done_mask_lst.append(self.done_buf[idx])

        return ExperienceReplayBatch(
            np.array(s_lst), 
            np.array(a_lst),
            np.array(r_lst), 
            np.array(s_prime_lst),
            np.array(done_mask_lst),
            )
    
    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size

class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6,
        n_step: int = 1, 
        gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(
            size, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(
        self, 
        obs: OdinsynthEnvStep, 
        act: int, 
        rew: float, 
        next_obs: OdinsynthEnvStep, 
        done: bool,
    ) -> Tuple[OdinsynthEnvStep, int, float, OdinsynthEnvStep, bool]:
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done)
        
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        
        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        obs = [self.obs_buf[i] for i in indices]
        next_obs = [self.next_obs_buf[i] for i in indices]
        acts = [self.acts_buf[i] for i in indices]
        rews = [self.rews_buf[i] for i in indices]
        done = [self.done_buf[i] for i in indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        return PrioritizedExperienceReplayBatch(obs, acts, rews, next_obs, done, weights, indices)
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight


ExperienceReplayBatch            = collections.namedtuple('ExperienceReplayBatch',            ['obs', 'acts', 'rews', 'next_obs', 'done']) 
PrioritizedExperienceReplayBatch = collections.namedtuple('PrioritizedExperienceReplayBatch', ['obs', 'acts', 'rews', 'next_obs', 'done', 'weights', 'indices']) 


class RainbowDQNAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
        self, 
        hparams = {},
        # memory_size: int,
        # batch_size: int,
        # target_update: int,
        # gamma: float = 0.99,
        # # PER parameters
        # alpha: float = 0.2,
        # beta: float = 0.6,
        # prior_eps: float = 1e-6,
        # # N-step Learning
        # n_step: int = 3,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        # action_dim = env.action_space.n
        self.hparams = hparams
        

        self.batch_size = self.hparams['batch_size']
        self.target_update = self.hparams['target_network_frequency']
        self.gamma = self.hparams['gamma']
        # NoisyNet: All attributes related to epsilon are removed
        
        # device: cpu / gpu
        self.device = torch.device('cuda:0')
        print(self.device)
        
        # PER
        # memory for 1-step Learning
        self.beta = self.hparams['beta']
        self.prior_eps = self.hparams['prior_eps']
        self.memory = PrioritizedReplayBuffer(
            size = self.hparams['buffer_size'], batch_size = self.batch_size, alpha=self.hparams['alpha']
        )
        
        # memory for N-step Learning
        self.use_n_step = True if self.hparams.get('n_step', 0) > 1 else False
        if self.use_n_step:
            self.n_step = self.hparams['n_step']
            self.memory_n = ReplayBuffer(
                self.hparams['buffer_size'], self.batch_size, n_step=self.hparams['n_step'], gamma=self.hparams['gamma']
            )
            
        # networks: dqn, dqn_target
        self.dqn = OdinsynthRLAgentWrapper(device=self.device)
        self.dqn_target = OdinsynthRLAgentWrapper(device=self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        train_env_factory, test_env_factory = OdinsynthEnvFactory.get_train_test_factories(hparams.get('datapath', '/data/nlp/corpora/odinsynth/data/rules100k/'), test_size = hparams.get('test_size', 0.005))

        self.train_env_factory = OdinsynthEnvFactory(train_env_factory.problem_specifications[:20000])
        self.test_env_factory  = test_env_factory

        self.train_env = OdinsynthEnvWrapper(train_env_factory)
        self.test_env  = OdinsynthEnvWrapper(test_env_factory)

        # self.epsilon = 0.9 # self.hparams['epsilon']

        
        # mode: train / test
        self.is_test = False

    def select_action(self, obs: OdinsynthEnvStep, epsilon) -> int:
        """Select an action from the input state."""
        if random.random() < epsilon:
            # with torch.no_grad():
                # values = self.dqn.forward([obs])
                # values = torch.tensor(values)
                # values = F.softmax(values, dim=-1)[0][0].detach().cpu().numpy()
                # action = np.random.choice(list(range(values.shape[0])), 1, p=values)[0]
            action = self.train_env.action_space.sample()
        else:
            logits = self.dqn.forward([obs])[0][0]
            action = np.argmax(logits, axis=-1)

        if not self.is_test:
            self.transition = [obs, action]
        
        return action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.train_env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples.weights
        ).to(self.device)
        indices = samples.indices
        
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        return loss.item()
        
    def train(self):
        """Train the agent."""
        self.is_test = False
        
        epsilon = 0.9 # We will change this value during training using a scheduling technique

        state = self.train_env.reset()
        update_cnt = 0
        losses = []
        scores = []
        score = 0

        # for frame_idx in range(1, self.hparams['total_timesteps'] + 1):
        for frame_idx in tqdm.tqdm(range(1, self.hparams['total_timesteps'] + 1)):
            action = self.select_action(state, epsilon)
            epsilon = linear_schedule(self.hparams['start_e'], self.hparams['end_e'], self.hparams['exploration_fraction'] * self.hparams['total_timesteps'], frame_idx)
            # epsilon = multi_schedulers(frame_idx, schedulers_dict, schedulers_params_dict) #linear_schedule(self.hparams['start_e'], self.hparams['end_e'], self.hparams['exploration_fraction']*self.hparams['total_timesteps'], global_step)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # PER: increase beta
            fraction = min(frame_idx / self.hparams['total_timesteps'], 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                state = self.train_env.reset()
                scores.append(score)
                score = 0

            # if training is ready
            if frame_idx > self.hparams['learning_starts']:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            if (frame_idx + 1) % 500 == 0:
                # print(f"global_step={global_step}, episode_reward={total_reward/accumulation_counter}")
                torch.save(self.dqn.state_dict(), f"{self.hparams['savepath']}_q_network_{frame_idx}.pt")
                torch.save(self.dqn_target.state_dict(), f"{self.hparams['savepath']}_target_network_{frame_idx}.pt")
                solved_percentage, steps = self.test()
                print(solved_percentage, steps)
                writer.add_scalar("charts/test_result_solved", solved_percentage, frame_idx)
                writer.add_scalar("charts/test_result_steps", steps, frame_idx)
                self.dqn.train()


            # plotting
            # if frame_idx % plotting_interval == 0:
                # self._plot(frame_idx, scores, losses)
                
        self.train_env.close()
        # writer.close()
        print(f"runs/{experiment_name}")
                
    def save_model(self):
        torch.save(self.dqn.state_dict(), f"{self.hparams['savepath']}_q_network5.pt")
        torch.save(self.dqn_target.state_dict(), f"{self.hparams['savepath']}_target_network5.pt")


    def solve_env(self, env: OdinsynthEnv, start_obs): 
        obs, reward, done, metadata = start_obs, None, False, None
        steps = 0
        while not done:
            steps += 1
            with torch.no_grad():
                logits = self.dqn.forward([obs])[0][0]
            action = np.argmax(logits, axis=-1)
            obs, reward, done, metadata = env.step(action)
            if metadata['solution']:
                done = True # Not necessary
                return (True, steps)
            elif metadata['compromised']:
                done = True # Not necessary
                return (False, steps)
            elif obs.query.is_valid_query():
                done = True # Not necessary
                return (False, steps)


    def test(self):
        solved = []
        steps  = []
        self.dqn.eval()
        for ps in self.test_env_factory.problem_specifications:
            result = self.solve_env(OdinsynthEnv(ps), OdinsynthEnvStep(HoleQuery(), ps))
            solved.append(result[0])
            steps.append(result[1])
        # results = [self.solve_env(OdinsynthEnv(ps), OdinsynthEnvStep(HoleQuery(), ps)) for ps in self.test_env_factory.problem_specifications]
        self.dqn.train()
        return len([x for x in solved if x]) / len(solved), np.mean(steps)

    def _compute_dqn_loss(self, samples: PrioritizedExperienceReplayBatch, gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        s_obs, s_actions, s_rewards, s_next_obses, s_dones, weights, indices = samples

        dqn_prediction = self.dqn.forward(s_next_obses)
        dqn_argmax     = [np.argmax(x[0]) for x in dqn_prediction]

        with torch.no_grad():
            tn_forward = self.dqn_target.forward(s_next_obses)
            tn_values = torch.tensor([nobse[0][a] for nobse, a in zip(tn_forward, dqn_argmax)]).to(self.device) # torch.max(target_network.forward(s_next_obses), dim=1)[0]
            target  = torch.tensor(s_rewards).to(self.device) + gamma * tn_values * (1 - torch.tensor(s_dones).to(self.device).int()).double()

        old_val = self.dqn.forward_batched_helper(s_obs, s_actions).double()
        elementwise_loss = F.mse_loss(old_val, target, reduction='none')

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="odinsynth",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=3e-5,
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
    parser.add_argument('--beta', type=float, default=0.4,
                        help="PER specific; How much importance sampling is used")
    parser.add_argument('--prior-eps', type=float, default=1e-6,
                        help="PER specific; Guarantees that every transition can be sampled")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="PER specific; How much prioritization is used")


args = parser.parse_args()
# if not args.seed:
    # args.seed = int(time.time())
from utils import init_random
import time
init_random(1)
print(vars(args))


experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
savepath_split = args.savepath.split('/')[-1]
writer = SummaryWriter(f"runs/{experiment_name}_{savepath_split}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))


dqn_agent = RainbowDQNAgent(vars(args))
solved_percentage, steps = dqn_agent.test()
print(solved_percentage, steps)
dqn_agent.train()

