from collections import OrderedDict
import pickle
import os
import sys
import time

import gymnasium as gym
from typing import Union
from gymnasium import wrappers
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
from climb.infrastructure import pytorch_util as ptu
from climb.infrastructure.utils import sample_trajectory, sample_trajectories, sample_n_trajectories, sample_trajectories_climber, sample_trajectory_climber
from climb.infrastructure.logger import Logger
from climb.envs.envs_utils import register_custom_envs
from climb.agents.ppo_agent import PPOAgent
from climb.agents.exploration_agent import ExplorationOrExploitationAgent
from climb.infrastructure.utils import Path, Path_climb
import pdb

class RL_Trainer(object):
    def __init__(self, params):
        #############
        ## INIT
        #############
        self.params = params
        self.logger = Logger(self.params['logdir'])
        self.saving_video = False

        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############
        register_custom_envs(self.params['env_name'])

        if self.params['xml_file'] is not None:
            xml_file = os.path.abspath(self.params['xml_file'])
            self.env = gym.make(self.params['env_name'], xml_file=xml_file, keyframe=self.params['keyframe'], rand_start_keyframe=self.params['rand_start_keyframe'], render_mode="rgb_array")
        else:
            self.env = gym.make(self.params['env_name'], render_mode="rgb_array")
        
        if self.params['rand_start_keyframe']:
            self.env = RecordVideo(self.env, "videos/todo", episode_trigger=lambda episode_id: self.saving_video, name_prefix=self.params['env_name']+'_rand_start')
        elif self.params['keyframe'] is not None:
            self.env = RecordVideo(self.env, "videos/keyframe", episode_trigger=lambda episode_id: self.saving_video, name_prefix=self.params['env_name']+'_'+self.params['keyframe'])
        else:
            self.env = RecordVideo(self.env, "videos/", episode_trigger=lambda episode_id: self.saving_video, name_prefix=self.params['env_name'])
            # self.env = RecordVideo(self.env, "videos/", episode_trigger=lambda episode_id: episode_id % 1000 == 1, name_prefix=self.params['env_name'])

        
        self._last_obs = self.env.reset(seed=self.params['seed'])[0]
        self._last_episode_starts = True
        self.num_timesteps = 0

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
        # agent_class = self.params['agent_class']
        if self.params['rnd']:
            self.agent: Union[PPOAgent] = ExplorationOrExploitationAgent(self.env, self.params['agent_params'], ep_len=self.params['ep_len'])
        else:
            self.agent: Union[PPOAgent] = PPOAgent(self.env, self.params['agent_params'], ep_len=self.params['ep_len'])

        if self.params['load_params'] is not None:
            checkpoint = torch.load(self.params['load_params'])
            mean_net_state_dict = {k.replace('mean_net.', ''): v for k, v in checkpoint['actor'].items() if k.startswith('mean_net')}
            self.agent.actor.mean_net.load_state_dict(mean_net_state_dict)
            # self.agent.actor.std = nn.Parameter(checkpoint['actor']['std'])
            self.agent.critic.load_state_dict(checkpoint['critic'])


    def run_training_loop(self):
        """
        :param n_iter:  number of iterations
        :param collect_policy:
        :param eval_policy:
        """
        # self.total_envsteps = 0
        self.start_time = time.time()

        # print_period = 1000
        # for itr in range(n_iter + 1):
        #     if itr % print_period == 0:
        #         print("\n\n********** Iteration %i ************"%itr)

        # decide if metrics should be logged


        # use_batchsize = self.params['batch_size']
        paths = []
        while self.num_timesteps <= self.params['max_training_timesteps']:
            path = self.collect_rollouts(self.env, self.agent.rollout_buffer, self.params['ep_len'])
            paths.extend(path)
            self.agent._update_current_progress_remaining(self.num_timesteps, self.params['max_training_timesteps'])
            self.agent.train()
    
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif self.num_timesteps % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False
            if self.logmetrics:
                print(f"============= num_timesteps: {self.num_timesteps} =============")
                self.perform_logging(self.num_timesteps, paths, self.agent.actor, None)
                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], self.num_timesteps))
                paths = []
    
    ####################################
    ####################################

    def collect_rollouts(self, env, rollout_buffer, n_rollout_steps):
        self.agent.eval_mode()
        n_steps = 0
        rollout_buffer.reset()

        # upward_reward = []
        # reward_motion = []

        if self.params['env_name'] == 'Climber-v0':
            obs, acs, rews, next_obs, terminals, heights, goal_rewards = [], [], [], [], [], [], []
        else:
            obs, acs, rews, next_obs, terminals = [], [], [], [], []
        self.env.reset(seed=self.params['seed'])
        self.params['seed'] += 1
        paths = []
        while n_steps < n_rollout_steps:

            with torch.no_grad():
                _last_obs = ptu.from_numpy(self._last_obs)
                actions, log_probs = self.agent.actor.get_action(self._last_obs)
                values = self.agent.critic(_last_obs)

            # clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            new_obs, rewards, dones, truncated, infos = env.step(actions)
            if self.params['env_name'] == 'Climber-v0':
                height = infos['z_position']
                # reward_motion.append(infos['reward_motion'])
                goal_reward = infos['goal_reward']

            # upward_reward.append(infos['reward_upward'])

            self.num_timesteps += 1

            n_steps += 1

            if self.params['agent_params']['discrete']:
                actions = actions.reshape(-1, 1)
            if (
                dones
                and infos.get("terminal_observation") is not None
                and infos.get("TimeLimit.truncated", False)
            ):
                terminal_obs = torch.tensor(infos["terminal_observation"])[0]
                with torch.no_grad():
                    terminal_value = self.agent.critic(terminal_obs)[0]
                rewards += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                dones,
                self._last_episode_starts,
                values.cpu(),
                log_probs.cpu(),
            )

            obs.append(self._last_obs)
            acs.append(actions)
            if self.params['env_name'] == 'Climber-v0':
                heights.append(height)
                goal_rewards.append(goal_reward)
            rews.append(rewards)
            next_obs.append(new_obs)
            if dones or n_steps >= n_rollout_steps:
                terminals.append(1)
                # upward_reward = []
                goal_reward = []
                
                if self.params['env_name'] == 'Climber-v0':
                    paths.append(Path_climb(obs, acs, rews, next_obs, terminals, heights, goal_rewards))
                    obs, acs, rews, next_obs, terminals, heights, goal_rewards = [], [], [], [], [], [], []
                else:
                    paths.append(Path(obs, acs, rews, next_obs, terminals))
                    obs, acs, rews, next_obs, terminals = [], [], [], [], []
                self.env.reset(seed=self.params['seed'])
                self.params['seed'] += 1
            else:
                terminals.append(0)

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.agent.critic(ptu.from_numpy(new_obs))
        if not self.params['rnd']:
            rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        # print(f"motion rew mean: {np.array(reward_motion).mean():.3f}")
        # print(f"motion rew std: {np.array(reward_motion).std():.3f}")
        # print(f"motion rew max: {np.array(reward_motion).max():.3f}")
        # print(f"motion rew min: {np.array(reward_motion).min():.3f}")
        return paths


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
        self.agent.eval_mode()
        self.env.unwrapped.cur_mode = "eval"
        # record one episode
        self.saving_video = True
        if self.params['env_name'] == 'Climber-v0':
            sample_trajectory_climber(self.env, eval_policy, self.params['ep_len'], render=False, render_mode="rgb_array")
        else:
            sample_trajectory(self.env, eval_policy, self.params['ep_len'], render=False, render_mode="rgb_array")
        self.saving_video = False

        last_log = None if all_logs is None else all_logs[-1]
        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        if self.params['env_name'] == 'Climber-v0':
            eval_paths, eval_envsteps_this_batch = sample_trajectories_climber(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])
        else:
            eval_paths, eval_envsteps_this_batch = sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])
        
        self.env.unwrapped.cur_mode = "train"
        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            if self.params['env_name'] == 'Climber-v0':
                train_max_height = [path["height"].max() for path in paths]
                eval_max_height = [eval_path["height"].max() for eval_path in eval_paths]
                train_goal_reward = [path["goal_reward"].mean() for path in paths]
                eval_goal_reward = [eval_path["goal_reward"].mean() for eval_path in eval_paths]

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
            if self.params['env_name'] == 'Climber-v0':
                logs["Eval_AverageMaxHeight"] = np.mean(eval_max_height)
                logs["Eval_AverageGoalReward"] = np.mean(eval_goal_reward)
                if self.params['rand_start_keyframe'] and np.mean(eval_max_height) > self.env.unwrapped.height_trigger[self.env.unwrapped.start_keyframe_max]:
                    self.env.unwrapped.start_keyframe_max += 1
                    print("==============================================")
                    print("TRIGGER HEIGHT REACHED! Adding another starting keyframe")
                    print("==============================================")

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)
            if self.params['env_name'] == 'Climber-v0':
                logs["Train_AverageMaxHeight"] = np.mean(train_max_height)
                logs["Train_AverageGoalReward"] = np.mean(train_goal_reward)

            # logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time

            if last_log is not None:
                logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            # logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()
