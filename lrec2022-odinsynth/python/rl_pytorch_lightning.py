import random
import config
from rl_reward import ParameterizedSimpleRewardCalculator, PartialDeltaRewardCalculator, SimpleRewardCalculator
from shutil import copyfile
from schedulers import multi_schedulers, single_value
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import tqdm
import time
from collections import defaultdict


from pytorch_lightning import loggers as pl_loggers
from typing import Any, List, OrderedDict
from utils import init_random

from torch.utils.data.dataloader import DataLoader
from rl_environment import OdinsynthEnv, OdinsynthEnvFactory, OdinsynthEnvWrapper
from rl_agent import OdinsynthRLAgentWrapper, QNetwork
from rl_utils import ListDataset, PrioritizedReplayBuffer, ReplayBuffer, ReplayBufferDataset, ProblemSpecification, OdinsynthEnvStep
from queryast import HoleQuery
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from schedulers import linear_schedule
from transformers import AdamW

"""
Run the model on validation before training
Useful to log the metrics to have a better 
perspective on the trainingy dynamics
"""
class ValidateOnStart(pl.callbacks.Callback):
    def on_train_start(self, trainer, network):
        result = trainer.run_evaluation(test_mode=False)
        network.prev_solved       = result[0][0]['solved']
        network.prev_steps        = result[0][0]['steps']
        network.prev_total_reward = result[0][0]['total_reward']
        
        network.current_solved       = result[0][0]['solved']
        network.current_steps        = result[0][0]['steps']
        network.current_total_reward = result[0][0]['total_reward']

        network.initial_complete_status = result[0][0]['complete_status']
        
        return result

# """
# Call eval() again on the target network (pytorch-lightning
# will call train() at the end)
# """
# class EvalTargetOnBatchStart(pl.callbacks.Callback):
#     def on_train_batch_start(self, trainer, network):
#         network.target_network.eval()

# """
# Dropout can be a source of variability
# This callback turns off dropout on the q_network
# This callback is needed because the framework calls train()
# """
# class DropoutEvalOnBatchStart(pl.callbacks.Callback):
#     def on_train_batch_start(self, trainer, network):
#         network._set_qnetwork_dropout_on_eval()


class InitialLoggingBeforeTraining(pl.callbacks.Callback):
    def on_train_start(self, trainer, network):
        network.q_network.eval()
        network.initial_logging()
        network.q_network.train()
        network.target_network.eval()

"""
Before starting the training procedure, populate the replay buffer
This is necessary because when training we are sampling experiences 
from the replay buffer (So we need to have something in it to be able
to sample from)
"""
class PopulateBufferOnStart(pl.callbacks.Callback):
    def on_train_start(self, trainer, network):
        network.q_network.train()
        network.target_network.eval()
        network._set_qnetwork_dropout_on_eval()
        for x in tqdm.tqdm(range(network.hparams['learning_starts'])):
            network._populate_single()
            network._increment_custom_global_step()

class SaveBufferOnStart(pl.callbacks.Callback):
    def __init__(self, savepath):
        super().__init__()
        self.savepath = savepath

    def on_train_start(self, trainer, network):
        with open(self.savepath, 'wb') as fout:
            import pickle
            pickle.dump(network.memory, fout)
            

class LoadBufferOnStart(pl.callbacks.Callback):
    def __init__(self, loadpath):
        super().__init__()
        self.loadpath = loadpath

    def on_train_start(self, trainer, network):
        network.q_network.train()
        network.target_network.eval()
        network._set_qnetwork_dropout_on_eval()
        with open(self.loadpath, 'rb') as fin:
            import pickle
            buffer = pickle.load(fin)
        network.memory = buffer
        for x in tqdm.tqdm(range(network.hparams['learning_starts'])):
            network._increment_custom_global_step()


            


class DQN(pl.LightningModule):
    def __init__(self, hparams={}):
        super().__init__()
        self.hparams = hparams

        self.q_network = OdinsynthRLAgentWrapper() 
        self.target_network = OdinsynthRLAgentWrapper().eval() 

        # # Some comments for running this implementation on gym environments
        # self.q_network = QNetwork(4, 2)
        # self.target_network = QNetwork(4, 2).eval()
        # self.target_network.load_state_dict(self.q_network.state_dict())
        # self.target_network.eval()
        # # Some comments for running this implementation on gym environments

        # Some maps to allow for greater flexibility. Set the desired setting 
        # from the CLI and use the hparams and these dictionary to select it
        losses = {
            'mse': F.mse_loss,
            'l1' : F.l1_loss,
        }
        
        replay_buffer_type = {
            'per': PrioritizedReplayBuffer,
            'normal' : ReplayBuffer,
        }
        
        train_env_factory, test_env_factory = OdinsynthEnvFactory.get_train_test_factories(hparams.get('datapath', '/data/nlp/corpora/odinsynth/data/rules100k/'), test_size = hparams.get('test_size', 0.1), condition = None)

        self.train_env_factory = OdinsynthEnvFactory(train_env_factory.problem_specifications)
        self.test_env_factory  = OdinsynthEnvFactory(test_env_factory.problem_specifications)
        print(f"Total of {len(train_env_factory.problem_specifications)} train environments and {len(test_env_factory.problem_specifications)} test environments")


        # self.reward_calculator = ParameterizedSimpleRewardCalculator(0, 1, -1)
        self.reward_calculator = PartialDeltaRewardCalculator(-0.05, 5, -5, 0.5)
        # Specify additional parameters for the environment
        self.env_params = env_params = { }

        self.train_env = OdinsynthEnvWrapper(train_env_factory, reward_calculator = self.reward_calculator, env_params = self.env_params)
        # self.test_env  = OdinsynthEnvWrapper(test_env_factory,  reward_calculator = ParameterizedSimpleRewardCalculator(0, 10, -10), env_params = {})
        
        # NOTE No need to seed because when we sample from the action space we 
        # use np.random, which was alreaddy seeded
        # self.train_env.seed(self.hparams['seed'])
        # self.train_env.env.action_space.seed(self.hparams['seed'])
        # self.train_env.env.observation_space.seed(self.hparams['seed'])
        

        # # Some comments for running this implementation on gym environments
        # import gym
        # from rl_utils import ListBasedGymEnv
        # self.train_env = ListBasedGymEnv(gym.make('CartPole-v0'))
        # self.train_env.seed(self.hparams.get('seed', 1))
        # self.train_env.action_space.seed(self.hparams.get('seed', 1))
        # self.train_env.observation_space.seed(self.hparams.get('seed', 1))
        # # Some comments for running this implementation on gym environments


        self.loss_fn = losses[self.hparams['loss_type']]
        if self.hparams.get('buffer_load', None):
            with open(self.hparams['buffer_load'], 'rb') as fin:
                import pickle
                self.memory = pickle.load(fin)
        else:
            self.memory  = replay_buffer_type[self.hparams['replay_buffer_type']](hparams['buffer_size'], {'alpha': 0.6, 'beta': 0.4, 'total_timesteps': hparams['total_timesteps']})

        self.obs, self.reward, self.done, self.metadata = self.train_env.reset(), None, None, None #OdinsynthEnvStep(self.query, self.problem), reward, done, {'solution': odinson_result['solution'], 'compromised': odinson_result['compromised']}
        self.episode_reward, self.total_reward = 0, 0
        self.prior_eps = self.hparams['prior_eps']

        # The key in the dictionary is the timestep when that scheduler will end
        # To not be confused with the length
        # This also mean that the key of the last one doesn't matter
        # The first line will apply until we reach 10 * self.hparams['total_timesteps']//100
        # exploitation = [(x * self.hparams['total_timesteps']//100, single_value) for x in range(0, 10)]
        self.schedulers_dict = {}
        self.schedulers_params_dict = {}
        for x in range(self.hparams['total_timesteps']//100):
            idx = (x + 1) * 100
            self.schedulers_dict[idx] = single_value
            if (x+1) % 6 == 0:
                self.schedulers_params_dict[idx] = { 'value': 0.15 }
            else:
                self.schedulers_params_dict[idx] = { 'value': 0    }
        

        # # Some comments for running this implementation on gym environments
        # from schedulers import linear_step
        # self.schedulers_dict        = {self.hparams['total_timesteps']: linear_step}
        # self.schedulers_params_dict = {self.hparams['total_timesteps']: {'num_training_steps': self.hparams['exploration_fraction'] * self.hparams['total_timesteps'], 'min_value': 1.0, 'max_value': 0.01}}
        # # Some comments for running this implementation on gym environments


        ####### Some changes from the pytorch-lightning framework (2) #######
        # Use a custom global step. Needed to keep track
        # of the current step when getting the epsilon
        # during the buffer population (1/2)
        self.custom_global_step = 0
        # Use a custom logger. Useful to give specific global_step
        # values. Create a pl_loggers.TensorBoardLogger in order
        # to get the directory that would have been used by the
        # pytorch-lightning framework (2/2)
        logger = pl_loggers.TensorBoardLogger("runs_temp_temp/")
        self.custom_logger = SummaryWriter(logger.log_dir)
        print(f"Logging in {logger.log_dir}")

        self.prev_solved       = None
        self.prev_steps        = None
        self.prev_total_reward = None
        self.prev_f1           = None

        self.current_solved       = None
        self.current_steps        = None
        self.current_total_reward = None
        self.current_f1           = None

        self.initial_complete_status = None
        self.current_complete_status = None

        # A training step is one in which the optimizer.step() was called
        # When using accumulate_gradient, the step() call doesn't happen
        # every time we compute a loss
        # 'total_timesteps' in our case means the number of times a loss
        # will be computed
        # This convoluted situation is because of pytorch-lightning framework,
        # which sometimes uses the number of times a loss was computed, other
        # times uses the number of times step() is called
        self.num_training_steps = self.hparams['total_timesteps'] / self.hparams['accumulate_gradient'] 

        self.times = defaultdict(int)

        self.save_hyperparameters()

    """
    A custom log function provides us the flexibility of
    choosing the x value (custom_global_step)
    A custom global step is needed because the global_step is not 
    incremented during our initialization (and we might vary
    epsilon based on it); Our initialization consists of populating
    the replay buffer (and possibly other operations)
    """
    def custom_log(self, name, value, **kwargs):
        self.custom_logger.add_scalar(name, value, self.custom_global_step)

    def initial_logging(self):
        vdl = self.val_dataloader()
        outputs = []
        for ps in vdl:
            is_solved, steps, total_reward = self._solve_environment(OdinsynthEnv(ps), OdinsynthEnvStep(HoleQuery(), ps))
            outputs.append({'solved': is_solved, 'steps': steps, 'total_reward': total_reward})


        solved       = len([x['solved'] for x in outputs if x['solved']]) / len(outputs)
        steps        = np.mean([x['steps'] for x in outputs])
        total_reward = np.mean([x['total_reward'] for x in outputs])

        print(solved, steps, total_reward)


    def forward(self, obs, **kwargs):
        return self.q_network.forward(obs, device=self.device, **kwargs)

    """
    Method meant for internal use. It accesss the epsilon value
    It calls step on the epsilon scheduler before returning it
    """
    def __get_epsilon(self):
        epsilon = linear_schedule(self.custom_global_step, self.hparams.get('start_e', 0.15), self.hparams.get('end_e', 0.0), self.hparams.get('exploration_fraction', 0.6) * (self.hparams['learning_starts'] + self.hparams['total_timesteps'])) #linear_schedule(self.hparams['start_e'], self.hparams['end_e'], self.hparams['exploration_fraction']*self.hparams['total_timesteps'], global_step)
        # epsilon = multi_schedulers(self.custom_global_step, self.schedulers_dict, self.schedulers_params_dict) #linear_schedule(self.hparams['start_e'], self.hparams['end_e'], self.hparams['exploration_fraction']*self.hparams['total_timesteps'], global_step)
        self.custom_log('epsilon', epsilon, on_step=True, on_epoch=True)
        
        return epsilon

    @torch.no_grad()
    def _populate_single(self):
        # Choose an action
        current_epsilon = self.__get_epsilon()
        r = random.random()
        if r < current_epsilon:
            action = self.train_env.action_space.sample()
        else:
            logits = self.q_network.forward([self.obs], device=self.device, return_list = True)[0] # the [0] indexing is done because we run the q_network with a list consisting of a single element
            action = np.argmax(logits, axis=-1).item()

        # Play the action
        next_obs, self.reward, self.done, self.metadata = self.train_env.step(action)
        self.episode_reward += self.reward
        
        # Store the output in the experience replay buffer (memory)
        self.memory.store(self.obs, action, self.reward, next_obs, self.done)

        # Update the current state
        self.obs = next_obs

        
        # Some tracking and reset
        if self.done:
            self.total_reward += self.episode_reward
            self.custom_log('metrics/episode_reward', self.episode_reward, on_step=True, prog_bar=True)
            # print(f"global_step={self.custom_global_step}, episode_reward={self.episode_reward}, total_reward={self.total_reward}")
            self.obs, self.episode_reward = self.train_env.reset(), 0
            
            

    def _increment_custom_global_step(self):
        self.custom_global_step += 1

    """
    Dropout increases the variance
    This is not necessarily something desired in RL
    """
    def _set_qnetwork_dropout_on_eval(self):
        for (name, param) in self.q_network.named_parameters():
            if 'Dropout' in name:
                param.eval()

    """
    Attempt to solve the environment passed as parameter
    Used during testing. No .eval() called on the model 
    because the pytorch-lightning framework handles
    this when inside 
    TODO Maybe return dict, namedtuple or object?
    returns:
            (
                boolean -> whether this environment was solved
                int     -> the number of steps the agent spend before the end of the episode
                float   -> the total reward the agent received until the end of the episode
                float   -> the partial reward received at the end of this episode
            )
    """
    def _solve_environment(self, env: OdinsynthEnv, start_obs): 
        obs, reward, done, metadata = start_obs, None, False, None
        steps = 0
        total_reward = 0
        while not done:
            steps += 1
            with torch.no_grad():
                logits = self.q_network.forward([obs], device=self.device, return_list = True)[0] # the [0] indexing is done because we run the q_network with a list consisting of a single element
            action = np.argmax(logits, axis=-1)
            obs, reward, done, metadata = env.step(action)
            total_reward += reward
            if metadata['solution']:
                done = True # Not necessary
                return (True, steps, total_reward, metadata['f1'])
            elif metadata['compromised']:
                done = True # Not necessary
                return (False, steps, total_reward, metadata['f1'])
            elif obs.query.is_valid_query():
                done = True # Not necessary
                return (False, steps, total_reward, metadata['f1'])

    def _solve_environments_batched(self, envs: List[OdinsynthEnv], start_obs: List[OdinsynthEnvStep]):
        dones        = torch.zeros(len(envs)).bool()
        solved       = torch.zeros(len(envs)).bool()
        steps        = torch.zeros(len(envs))
        total_reward = torch.zeros(len(envs))
        env_startobs: List[List] = [[x[0], x[1]] for x in list(zip(envs, start_obs))]
        metadatas = {i:{} for i in range(len(envs))}
        
        while not dones.all():
            steps[dones.logical_not()] += 1
            # ndy -> not_don_yet
            ndy_idx  = [idx for idx, x in enumerate(env_startobs) if not (x[1].query.is_valid_query() or dones[idx])]
            ndy_obs  = [x[1] for idx, x in enumerate(env_startobs) if not (x[1].query.is_valid_query() or dones[idx])]

            with torch.no_grad():
                logits = self.q_network.forward(ndy_obs, device=self.device, return_list = True)
            actions = [np.argmax(individual_logits, axis=-1) for individual_logits in logits]
            for idx, action in enumerate(actions):
                env_idx = ndy_idx[idx]
                obs, reward, done, metadata = env_startobs[env_idx][0].step(action)
                total_reward[env_idx] += reward
                env_startobs[env_idx][1] = obs
                if metadata['solution']:
                    dones[env_idx]  = True
                    solved[env_idx] = True
                    metadatas[env_idx]['f1'] = metadata['f1']
                elif metadata['compromised'] or obs.query.is_valid_query():
                    dones[env_idx]  = True
                    solved[env_idx] = False
                    metadatas[env_idx]['f1'] = metadata['f1']


        return (solved, steps, total_reward, metadatas)


        
    def training_step(self, erb, idx) -> OrderedDict:
        # Pecularities of using the pytorch lightning framework
        # We might want dropout on eval and target_network on eval
        # The pytorch-lightning framework calls train on the whole module
        # Alternatives would be with callbacks, though this is dependent
        # on when is the call on train done
        self._increment_custom_global_step()
        self._set_qnetwork_dropout_on_eval()
        self.target_network.eval()

        # Add an experiene to the replay buffer in each training step, after .optimizer() call
        # This is necessary because if we use a big 'accumulate_gradient', say 1024, and a batch
        # size of 8, we would want to simulate a batch_size of 8*1024, but if we add experiences
        # to the buffer after each run with 8 experiences, the buffer will not look the same.
        # start = time.time()
        if self.custom_global_step % self.hparams['accumulate_gradient'] == 0:
            self._populate_single()
        # end = time.time()
        # self.times['_populate_single'] += end-start
        # Calculate the q values of the current observations using the (trainable) q_network
        # Do the q_network forward first to get the type of the tensor (e.g. precision 32, precision 16, etc)
        # start = time.time()
        old_val = self.q_network.forward(erb.obs, device=self.device, indices=erb.acts)
        # end = time.time()
        # self.times['old_val'] += end-start

        # Calculate the q value of next_observation with the target network
        # Do not compute the gradient when calculating the tensors for tn_forward
        # start = time.time()
        with torch.no_grad():
            tn_forward = self.target_network.forward(erb.next_obs, device=self.device, return_list = True) # Return list because next_nodes can produce lists of different lengths (and no point in padding)
            target_max = torch.tensor([np.max(a) for a in tn_forward]).to(self.device).type(dtype=old_val.type()) # torch.max
            rewards    = torch.tensor(erb.rews).to(self.device).type(dtype=old_val.type())
            mask       = (1 - torch.tensor(erb.done).to(self.device).type(dtype=old_val.type()))
            td_target  = rewards + self.hparams['gamma'] * target_max * mask

        # end = time.time()
        # self.times['td_target'] += end-start

        elementwise_loss = self.loss_fn(old_val, td_target, reduction='none')
        if self.hparams['replay_buffer_type'] == 'per':
            loss = torch.mean(elementwise_loss * torch.tensor(erb.weights).to(self.device).type(dtype=old_val.type()))
            self.memory.update({'loss_for_prior': elementwise_loss.detach().cpu().numpy(), 'prior_eps': self.prior_eps, 'indices': erb.indices})
        else:
            loss = torch.mean(elementwise_loss)


        # Update the target network
        # NOTE We use global_step instead of custom_global_step. We do not
        # count the steps spend while populating the buffer (would not change
        # anyway, since we are not training during that time)
        # Moreover, it would require special handling for accumulate_grad_batches
        # global_step is increased only after the real batch_size (batch_size * accumulate_grad_batches)
        # is reached
        if self.global_step % self.hparams['target_network_frequency'] == 0:
            # self.target_network.load_state_dict(self.q_network.state_dict())
            # Update if better
            if self.current_solved > self.prev_solved:
                self.target_network.load_state_dict(self.q_network.state_dict())
            elif self.current_solved == self.prev_solved and self.current_total_reward > self.prev_total_reward:
                self.target_network.load_state_dict(self.q_network.state_dict())
            elif self.current_solved == self.prev_solved and self.current_total_reward == self.prev_total_reward and self.current_steps > self.prev_steps:
                self.target_network.load_state_dict(self.q_network.state_dict())


        if self.custom_global_step % 100 == 0:
            self.custom_log('train_loss', loss.item(), on_step=True, on_epoch=True)

        return OrderedDict({
            "loss": loss,
        })


    def validation_step(self, ps: ProblemSpecification, idx):
        is_solved, steps, total_reward, f1 = self._solve_environment(OdinsynthEnv(ps, reward_calculator=self.reward_calculator, env_params = self.env_params), OdinsynthEnvStep(HoleQuery(), ps))
        return {'solved': is_solved, 'steps': steps, 'total_reward': total_reward, 'f1': f1}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        solved       = len([x['solved'] for x in outputs if x['solved']]) / len(outputs)
        steps        = np.mean([x['steps'] for x in outputs])
        total_reward = np.mean([x['total_reward'] for x in outputs])
        f1           = np.mean([x['f1'] for x in outputs])

        self.custom_log(f'metrics/solved', solved, prog_bar=True)
        self.custom_log(f'metrics/steps', steps, prog_bar=True)
        self.custom_log(f'metrics/total_reward', total_reward, prog_bar=True)
        self.custom_log(f'metrics/f1', f1, prog_bar=True)

        self.current_complete_status = [x['solved'] for x in outputs]
        all_data = np.array([[x['solved'], x['steps'], x['total_reward'], x['f1']] for x in outputs])
        np.save(self.custom_logger.get_logdir() + '/' + str(self.custom_global_step) + '_each_test_env', all_data)

        
        if self.initial_complete_status and self.current_complete_status:
            initial = set([x[0] for x in enumerate(self.initial_complete_status) if x[1]])
            current = set([x[0] for x in enumerate(self.current_complete_status)  if x[1]])
            total   = len(outputs)

            both_solved    = len(initial.intersection(current)) / total
            initial_solved = len(initial.difference(current)) / total
            new_solved     = len(current.difference(initial)) / total
            none_solved    = 1 - (len(initial.union(current)) / total)
            
            print('\n\nResult:', self.custom_global_step, solved, steps, total_reward, 
                    f'\nCompared to initial:', 
                    "{:.5f}".format(both_solved)    + '% (both)',        
                    "{:.5f}".format(initial_solved) + '% (only initial)',        
                    "{:.5f}".format(new_solved)     + '% (only current)',        
                    "{:.5f}".format(none_solved)    + '% (none)',        
            )

            self.custom_log(f'solved_stats/both', both_solved)
            self.custom_log(f'solved_stats/only_initial', initial_solved)
            self.custom_log(f'solved_stats/only_current', new_solved)
            self.custom_log(f'solved_stats/none', none_solved)
        
        else:
            print('\n\nResult:', self.custom_global_step, solved, steps, total_reward)


        # Update our internal tracking
        self.prev_solved       = self.current_solved
        self.prev_steps        = self.current_steps
        self.prev_total_reward = self.current_total_reward
        self.prev_f1           = self.current_total_reward

        self.current_solved       = solved
        self.current_steps        = steps
        self.current_total_reward = total_reward
        self.current_f1           = f1

        return {
            'solved'      : solved,
            'steps'       : steps,
            'total_reward': total_reward,
            'complete_status': [x['solved'] for x in outputs],
        }

    """
    Train dataloader. A dataloader over a dataset wrapping a replay buffer
    We set batch_size and batch_sample to None because we do the sampling inside the replay buffer
    We do this because: this is how it is usually done when not involving pytorch-lightning; and 
    some replay bufferes need to maintain an internal state (e.g. PER with the beta parameter). Thus,
    letting it handle sampling reveals the batch size, as the update happens once per batch size
    """ 
    def train_dataloader(self) -> DataLoader:
        dataset = ReplayBufferDataset(self.memory, self.hparams['batch_size'], self.hparams['total_timesteps'])
        return DataLoader(dataset = dataset, batch_size=None, batch_sampler=None, num_workers = self.hparams.get('train_workers', 0))

    """
    Validation dataloader. List of problem specification objects wrapped in a pytorch dataloader
    We set batch_size and batch_sampler to None because we use "_solve_environment" method which takes
    a single environment to solve
    Notice that the batch size is actually dynamic, because when we call the next_nodes on a node
    to score them (to get the most plausible expansion), the resulted list can have different lengths
    """
    def val_dataloader(self) -> DataLoader:
        dataset = ListDataset(self.test_env_factory.problem_specifications)
        return DataLoader(dataset = dataset, batch_size=None, batch_sampler=None, num_workers = self.hparams.get('validation_workers', 0))

    def configure_optimizers(self):
        named_params = list(self.q_network.named_parameters())
        params = [
            {
                'params': [p for n,p in named_params if not any(nd in n for nd in config.NO_DECAY)],
                'weight_decay': self.hparams['weight_decay'],
            },
            {
                'params': [p for n,p in named_params if any(nd in n for nd in config.NO_DECAY)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = AdamW(params, lr=self.hparams['learning_rate'])
        
        if 'use_scheduler' in self.hparams and self.hparams['use_scheduler']:     
            return (
                [optimizer], 
                [{
                    # 'scheduler': get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams['num_training_steps']//16, num_training_steps=self.hparams['num_training_steps'],),
                    # 'scheduler': get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams['num_training_steps']//2, num_training_steps=self.hparams['num_training_steps'],),
                    # 'scheduler': torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.hparams['learning_rate']/5, max_lr = self.hparams['learning_rate'], mode='triangular2', cycle_momentum=False, step_size_up=self.num_training_steps//2),
                    'scheduler': torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.hparams['learning_rate']/5, max_lr = self.hparams['learning_rate'], mode='triangular2', cycle_momentum=False, step_size_up=self.num_training_steps//2),
                    'interval': 'step',
                    'frequency': 1,
                    'strict': True,
                    'reduce_on_plateau': False, 
                }]
            )
        else:
            return optimizer


params = {
    'exp_name': 'ex_test', 
    'gym_id': 'odinsynth', 
    'learning_rate': 1e-5, 
    'seed': 1, 
    'total_timesteps': 1_000_000, 
    'torch_deterministic': True, 
    'cuda': True, 
    'buffer_size': 250_000, 
    'gamma': 0.999, 
    'target_network_frequency': 20_000, 
    'max_grad_norm': 0.5, 
    'batch_size': 128, 
    'start_e': 0.25, 
    'end_e': 0.0, 
    'exploration_fraction': 0.6, 
    'learning_starts': 250_000, 
    'train_frequency': 1, 
    'accumulate_gradient': 4, 
    'test_every': 20_000, 
    'datapath': '/data/nlp/corpora/odinsynth/data/rules100k/', 
    'additonal_details': '2_128, 100k, Clean RL', 
    'loss_type': 'mse', 
    'replay_buffer_type': 'per', 
    'log_weights': False, 
    'log_gradients': False, 
    'prior_eps': 1e-6,
    'weight_decay': 0.001,
    # 'buffer_load': 'runs_temp_temp/default/version_498/buffer.pkl', 
}

# Run the implementation of the DQN algorithm above on the custom
# implemented environment 
def rule_synthesizer_run(**kwargs):
    # Note that we artificially alter 'max_steps' and 'val_check_interval' to
    # keep into account the 'accumulate_grad_batches'
    # This is because we are interested in evaluating after k batches,
    # but a batch should really be considered as 'batch_size' * 'accumulate_grad_batches'
    
    this_run_params = params.copy()
    this_run_params.update(**kwargs)
    this_run_params['total_timesteps'] = this_run_params['total_timesteps'] * this_run_params.get('accumulate_gradient', 1)
    this_run_params['test_every']      = this_run_params['test_every']      * this_run_params.get('accumulate_gradient', 1)
    dqn = DQN(this_run_params)
    # logger = pl_loggers.TensorBoardLogger("runs_temp_temp/")
    training_params = {
        'gpus'                    : this_run_params.get('gpus', 1),
        # 'logger'                  : logger,
        'precision'               : this_run_params.get('precision', 16),
        'gradient_clip_val'       : this_run_params.get('max_grad_norm', 0.5),
        'accumulate_grad_batches' : this_run_params.get('accumulate_gradient', 1),
        'num_sanity_val_steps'    : this_run_params.get('num_sanity_val_steps', 50),
        'resume_from_checkpoint'  : this_run_params.get('resume_from_checkpoint', None),
        # pytorch-lightning weirdness; the number max number of steps has to do with global_step, but what 
        # is shown in tqdm progress bar is from each individual "small" batch (small means that we didn't 
        # accumulate over the number of times we specified). This is the same for 'val_check_interval' as well
        # Therefore, what happens is that the value displayed in the tqdm progress bar, once it reaches 
        # the value specified for 'val_check_interval', we perform a validation. 
        'max_steps'               : this_run_params.get('total_timesteps', 10000) / this_run_params.get('accumulate_gradient', 1), 
        'val_check_interval'      : this_run_params.get('test_every', 2000),
        'log_every_n_steps'       : 100,
        # 'profiler'                :'simple',
    }
    print(this_run_params)
    print(training_params)
    lrm = LearningRateMonitor(logging_interval='step')
    es  = EarlyStopping(
        monitor  = 'solved',
        patience = 39,
        mode     = 'max'
    )
    cp = ModelCheckpoint(
        monitor    = 'solved',
        save_top_k = 11,
        mode       = 'max',
        save_last  = True,
        filename   = '{step}-{solved:.5f}-{steps:.5f}-{total_reward:.5f}'
    )


    vos   = ValidateOnStart() # Run evaluation on start
    pbos  = PopulateBufferOnStart() # Populate the buffer for experience replay on start
    sbos  = SaveBufferOnStart(f"{dqn.custom_logger.log_dir}/buffer.pkl")
    # lbos  = LoadBufferOnStart('runs_temp_temp/default/version_498/buffer.pkl')
    # ilbt  = InitialLoggingBeforeTraining() #

    train_dl = dqn.train_dataloader()
    val_dl   = dqn.val_dataloader()

    trainer = pl.Trainer(
        **training_params,
        callbacks = [lrm, es, cp, vos, pbos],
        # progress_bar_refresh_rate=0,
    )

    print("The checkpoints are saved at:", trainer.logger.log_dir)

    trainer.fit(dqn, train_dataloader = train_dl, val_dataloaders = val_dl)
    copyfile(cp.best_model_path, f'{cp.dirpath}/best.ckpt')
    dqn.custom_logger.close()
    return cp.best_model_score


# Load two models and run them over the test dataset
# Log some statistics about them
# 
def compare_rule_synthesizer_models():
    device = torch.device('cuda:0')
    dqn_trained   = DQN.load_from_checkpoint('/home/rvacareanu/projects/odinsynth/python/lightning_logs/version_180/checkpoints/step=step=156-solved=solved=0.233-steps=step=steps=9.573-total_reward=total_reward=-4.600.ckpt').eval()
    dqn_trained = dqn_trained.to(device)
    init_random(1)
    dqn_untrained = DQN(params).eval()
    dqn_untrained = dqn_untrained.to(device)
    test_env = dqn_untrained.val_dataloader()
    untrained = []
    trained   = []
    import tqdm
    for ps in tqdm.tqdm(test_env):
        result_untrained = dqn_untrained._solve_environment(OdinsynthEnv(ps), OdinsynthEnvStep(HoleQuery(), ps))
        result_trained   =   dqn_trained._solve_environment(OdinsynthEnv(ps), OdinsynthEnvStep(HoleQuery(), ps))
        untrained.append(result_untrained)
        trained.append(result_trained)
    
    untrained = np.array(untrained)
    trained   = np.array(trained)
    np.save('results/rl/pytorch-lightning/untrained.npy', untrained)
    np.save('results/rl/pytorch-lightning/trained.npy', trained)
    

# Run the implementation of the DQN algorithm above on a gym environment
# (Useful for validation purposes; Some environments from gym are known
# to be reasonably easy to solve by the DQN framework)
def gym_run():
    params['buffer_size']              = 10000
    params['target_network_frequency'] = 500
    params['learning_starts']          = 10000
    params['total_timesteps']          = 20000
    params['test_every']               = 200000
    params['learning_rate']            = 7e-4
    params['batch_size']               = 128
    params['loss_type']                = 'mse'
    params['replay_buffer_type']       = 'per'

    dqn = DQN(params)

    training_params = {
        'gpus'                     : params.get('gpus', 1),
        # 'logger'                   : logger,
        'precision'                : params.get('precision', 32),
        'gradient_clip_val'        : params.get('max_grad_norm', 0.5),
        'accumulate_grad_batches'  : params.get('accumulate_gradient', 1),
        'num_sanity_val_steps'     : params.get('num_sanity_val_steps', 0),
        'resume_from_checkpoint'   : params.get('resume_from_checkpoint', None),
        'max_steps'                : params.get('total_timesteps', 100000),
        'val_check_interval'       : params.get('test_every', 200000),
        'log_every_n_steps'        : 1,
        'progress_bar_refresh_rate': 0,
        # 'profiler'                :'simple',
    }

    pbos  = PopulateBufferOnStart()

    trainer = pl.Trainer(
        **training_params,
        callbacks = [pbos],
    )

    train_dl = dqn.train_dataloader()

    init_random(1)
    trainer.fit(dqn, train_dataloader = train_dl)

init_random(1)
# compare_rule_synthesizer_models()
rule_synthesizer_run()
# gym_run()

