from collections import OrderedDict
import pickle
import os
import sys
import time

import gymnasium as gym
from gymnasium import wrappers
import numpy as np
import torch
from climb.infrastructure import pytorch_util as ptu
from climb.infrastructure.utils import sample_trajectory, sample_trajectories, sample_n_trajectories
from climb.infrastructure.logger import Logger
from climb.envs.envs_utils import register_custom_envs

class RL_Trainer(object):
    def __init__(self, params):
        #############
        ## INIT
        #############
        self.params = params
        self.logger = Logger(self.params['logdir'])

        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############
        register_custom_envs(self.params['env_name'])
        self.env = gym.make(self.params['env_name'])
        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params['agent_params']['discrete'] = discrete

        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        #############
        ## AGENT
        #############
        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy):
        """
        :param n_iter:  number of iterations
        :param collect_policy:
        :param eval_policy:
        """
        self.total_envsteps = 0
        self.start_time = time.time()

        print_period = 1000
        for itr in range(n_iter + 1):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************"%itr)

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            use_batchsize = self.params['batch_size']
            if itr==0:
                use_batchsize = self.params['batch_size_initial']
            paths, envsteps_this_batch = (
                self.collect_training_trajectories(
                    itr, collect_policy, use_batchsize)
            )

            self.total_envsteps += envsteps_this_batch

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")
            all_logs = self.train_agent() 
            if self.logmetrics:
                self.perform_logging(itr, paths, eval_policy, all_logs)
                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))
    ####################################
    ####################################

    def collect_training_trajectories(self, collect_policy, batch_size):
        """
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
        """
        # print('Collecting train data...')
        paths, envsteps_this_batch = sample_trajectories(
            self.env,
            collect_policy,
            batch_size,
            self.params['ep_len']
        )
        return paths, envsteps_this_batch

    def train_agent(self):
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            obs_batch, act_batch, rew_batch, nobs_batch, term_batch = self.agent.sample(self.params['train_batch_size'])
            train_log = self.agent.train(obs_batch, act_batch, rew_batch, nobs_batch, term_batch)
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################
    def perform_logging(self, itr, paths, eval_policy, all_logs):
        last_log = all_logs[-1]
        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()
