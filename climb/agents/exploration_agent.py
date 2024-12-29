from collections import OrderedDict

# from rob831.hw4_part2.critics.dqn_critic import DQNCritic
from climb.infrastructure import pytorch_util as ptu
from climb.infrastructure import buffers
from climb.critics.ppo_critic import PPOCritic
# from rob831.hw4_part2.infrastructure.replay_buffer import ReplayBuffer

# from rob831.hw4_part2.infrastructure.utils import *
# from rob831.hw4_part2.policies.argmax_policy import ArgMaxPolicy

# from rob831.hw4_part2.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from climb.exploration.rnd_model import RNDModel
# from .dqn_agent import DQNAgent
from .ppo_agent import PPOAgent
from climb.policies.ppo_policy import PPOMLPPolicy
import numpy as np
import torch
import torch.nn as nn
import pdb

class ExplorationOrExploitationAgent(PPOAgent):
    def __init__(self, env, agent_params, ep_len, normalize_rnd=True, rnd_gamma=0.99):
        super(ExplorationOrExploitationAgent, self).__init__(env, agent_params, ep_len)
        
        # self.replay_buffer = MemoryOptimizedReplayBuffer(100000, 1, float_obs=True)
        self.num_exploration_steps = agent_params['num_exploration_steps']
        # self.offline_exploitation = agent_params['offline_exploitation']

        self.exploitation_critic = PPOCritic(
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['learning_rate']
        )
        self.exploration_critic = PPOCritic(
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['learning_rate']
        )
        
        self.exploration_model = RNDModel(agent_params)
        self.explore_weight_schedule = agent_params['explore_weight_schedule']
        self.exploit_weight_schedule = agent_params['exploit_weight_schedule']
        
        # self.actor = ArgMaxPolicy(self.exploration_critic)
        self.critic = self.exploration_critic

        # self.eval_policy = PPOMLPPolicy(
        #     self.agent_params['ac_dim'],
        #     self.agent_params['ob_dim'],
        #     self.agent_params['n_layers'],
        #     self.agent_params['size'],
        #     action_space_bound=self.action_space_bound,
        #     discrete=self.agent_params['discrete'],
        #     learning_rate=self.agent_params['learning_rate'],
        # )
        self.exploit_rew_shift = agent_params['exploit_rew_shift']
        self.exploit_rew_scale = agent_params['exploit_rew_scale']
        # self.eps = agent_params['eps']

        self.running_rnd_rew_std = 1
        self.normalize_rnd = normalize_rnd
        self.rnd_gamma = rnd_gamma
        self.t = 0

    def train(self):
        self.exploitation_critic.train()
        self.exploration_critic.train()
        self.exploration_model.train()
        self.actor.train()
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining) 
        
        for _ in range(self.policy_updates_per_rollout):
            for rollout_data in self.rollout_buffer.get():
                ac_na = rollout_data.actions
                ob_no = rollout_data.observations
                # next_ob_no = np.concatenate((rollout_data.observations[1:], rollout_data.observations[-1:,]))
                next_ob_no = np.concatenate((self.rollout_buffer.observations[1:], self.rollout_buffer.observations[-1:]))[self.rollout_buffer.indices,:]
                re_n = ptu.to_numpy(rollout_data.rewards)
                terminal_n = ptu.to_numpy(rollout_data.dones)
                # print(self.rollout_buffer.indices)
                log = {}

                if self.t > self.num_exploration_steps:
                    # TODO: After exploration is over, set the actor to optimize the extrinsic critic
                    #HINT: Look at method ArgMaxPolicy.set_critic
                    # self.actor.set_critic(self.exploitation_critic)
                    self.critic = self.exploitation_critic

                # Get Reward Weights
                # TODO: Get the current explore reward weight and exploit reward weight
                #       using the schedule's passed in (see __init__)
                # COMMENT: you can simply use explore_weight = 1, and exploit_weight = 0
                explore_weight = self.explore_weight_schedule.value(self.t)
                exploit_weight = self.exploit_weight_schedule.value(self.t)

                # Run Exploration Model #
                # TODO: Evaluate the exploration model on s to get the exploration bonus
                # HINT: Normalize the exploration bonus, as RND values vary highly in magnitude.
                # HINT: Normalize using self.running_rnd_rew_std, and keep an exponential moving average
                # of self.running_rnd_rew_std using self.rnd_gamma.
                # expl_bonus = None

                expl_bonus = self.exploration_model.forward_np(next_ob_no)
                self.running_rnd_rew_std = self.rnd_gamma * self.running_rnd_rew_std + (1 - self.rnd_gamma) * np.std(expl_bonus)
                expl_bonus /= self.running_rnd_rew_std

                # Reward Calculations #
                # TODO: Calculate mixed rewards, which will be passed into the exploration critic
                # HINT: See doc for definition of mixed_reward
                # mixed_reward = None

                mixed_reward = exploit_weight * re_n + explore_weight * expl_bonus

                # TODO: Calculate the environment reward
                # HINT: For part 1, env_reward is just 're_n'
                #       After this, env_reward is 're_n' shifted by self.exploit_rew_shift,
                #       and scaled by self.exploit_rew_scale
                # env_reward = None
                env_reward = (re_n + self.exploit_rew_shift) * self.exploit_rew_scale

                # Update Critics And Exploration Model #

                # TODO 1): Update the exploration model (based off s')
                # TODO 2): Update the exploration critic (based off mixed_reward)
                # TODO 3): Update the exploitation critic (based off env_reward)
                # expl_model_loss = None
                # exploration_critic_loss = None
                # exploitation_critic_loss = None
                expl_model_loss = self.exploration_model.update(next_ob_no)

                # exploration_critic_loss = self.exploration_critic.update(ob_no, ac_na, next_ob_no, mixed_reward, terminal_n)
                self.rollout_buffer.rewards = mixed_reward
                last_value = self.exploration_critic(ptu.from_numpy(self.rollout_buffer.observations[-1:,:])).squeeze().detach()
                last_done = self.rollout_buffer.dones[-1,0]
                self.rollout_buffer.compute_returns_and_advantage(last_values=last_value, dones=last_done)
                for _ in range(self.agent_params['critic_updates_per_policy_update']):
                    values = self.exploration_critic(ob_no).squeeze(1)
                    value_loss = self.MseLoss(values.squeeze(), ptu.from_numpy(self.rollout_buffer.returns[self.rollout_buffer.indices]).squeeze())
                    # print(f"explore values mean {values.mean():.2f}")
                    # print(f"target mean {rollout_data.returns.mean():.2f}\n")
                    # print(f"values std {values.std():.2f}")
                    # print(f"target std {rollout_data.returns.std():.2f}\n")
                    # print(f"values max {values.argmax()} {values.max():.2f} (actual: {rollout_data.returns[values.argmax()]:.2f})")
                    # print(f"target max {rollout_data.returns.argmax()} {rollout_data.returns.max():.2f}\n")
                    # print(f"values min {values.argmin()} {values.min():.2f} (actual: {rollout_data.returns[values.argmin()]:.2f})")
                    # print(f"target min {rollout_data.returns.argmin()} {rollout_data.returns.min():.2f}\n\n")
                    exploration_critic_loss = self.vf_coef * value_loss
                    self.exploration_critic.optimizer.zero_grad()
                    exploration_critic_loss.backward()
                    self.exploration_critic.optimizer.step()

                # exploitation_critic_loss = self.exploitation_critic.update(ob_no, ac_na, next_ob_no, env_reward, terminal_n)
                self.rollout_buffer.rewards = env_reward
                last_value = self.exploitation_critic(ptu.from_numpy(self.rollout_buffer.observations[-1:,:])).squeeze().detach()
                last_done = self.rollout_buffer.dones[-1,0]
                self.rollout_buffer.compute_returns_and_advantage(last_values=last_value, dones=last_done)
                for _ in range(self.agent_params['critic_updates_per_policy_update']):
                    values = self.exploitation_critic(ob_no).squeeze(1)
                    value_loss = self.MseLoss(values.squeeze(), ptu.from_numpy(self.rollout_buffer.returns[self.rollout_buffer.indices]).squeeze())
                    # print(f"exploit values mean {values.mean():.2f}")
                    # print(f"target mean {rollout_data.returns.mean():.2f}\n")
                    # print(f"values std {values.std():.2f}")
                    # print(f"target std {rollout_data.returns.std():.2f}\n")
                    # print(f"values max {values.argmax()} {values.max():.2f} (actual: {rollout_data.returns[values.argmax()]:.2f})")
                    # print(f"target max {rollout_data.returns.argmax()} {rollout_data.returns.max():.2f}\n")
                    # print(f"values min {values.argmin()} {values.min():.2f} (actual: {rollout_data.returns[values.argmin()]:.2f})")
                    # print(f"target min {rollout_data.returns.argmin()} {rollout_data.returns.min():.2f}\n\n")
                    exploitation_critic_loss = self.vf_coef * value_loss
                    self.exploitation_critic.optimizer.zero_grad()
                    exploitation_critic_loss.backward()
                    self.exploitation_critic.optimizer.step()

                
                distribution, entropy = self.actor(ob_no)
                # epsilon = 1e-6
                # clipped_actions = torch.clamp(actions, -(self.action_space_bound - epsilon), self.action_space_bound - epsilon)
                log_prob = distribution.log_prob(ac_na).sum(-1)
                advantages = ptu.from_numpy(self.rollout_buffer.advantages[self.rollout_buffer.indices].squeeze())
                if self.standardize_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                self.actor.optimizer.zero_grad()
                policy_loss.backward()
                self.actor.optimizer.step()

                # Logging #
                # log['Exploitation Critic Loss'] = exploitation_critic_loss
                # log['Exploration Critic Loss'] = exploration_critic_loss
                # log['Exploration Model Loss'] = expl_model_loss

                # print(f"p loss:{policy_loss.item():.2f}, "
                #     f"v loss:{value_loss.item():.2f}, "
                #     f"ratio max:{ratio.max():.2f}, "
                #     f"std:{torch.exp(self.actor.logstd).mean():.2f}, "
                #     f"v mean:{self.rollout_buffer.value_mean:.2f}, "
                #     f"v std:{self.rollout_buffer.value_std:.2f}, ")
                self.t += 1
        # return log


    # def step_env(self):
    #     """
    #         Step the env and store the transition
    #         At the end of this block of code, the simulator should have been
    #         advanced one step, and the replay buffer should contain one more transition.
    #         Note that self.last_obs must always point to the new latest observation.
    #     """
    #     if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
    #         self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

    #     perform_random_action = np.random.random() < self.eps or self.t < self.learning_starts

    #     if perform_random_action:
    #         action = self.env.action_space.sample()
    #     else:
    #         processed = self.replay_buffer.encode_recent_observation()
    #         action = self.actor.get_action(processed)

    #     next_obs, reward, done, info = self.env.step(action)
    #     self.last_obs = next_obs.copy()

    #     if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
    #         self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

    #     if done:
    #         self.last_obs = self.env.reset()
