import numpy as np
import torch
import torch.nn as nn
import tqdm

from collections import OrderedDict

from climb.agents.base_agent import BaseAgent
from climb.policies.ppo_policy import PPOMLPPolicy
from climb.critics.ppo_critic import PPOCritic
from climb.infrastructure import pytorch_util as ptu
from climb.infrastructure import buffers
from climb.infrastructure.utils import get_schedule_fn
import pdb

device = ptu.device
class PPOAgent(BaseAgent):

    def __init__(self, env, agent_params, ep_len):
        super().__init__()
        self.agent_params = agent_params
        self.action_space_bound = torch.tensor(env.action_space.high)
        self.policy_updates_per_rollout = self.agent_params['policy_updates_per_rollout']
        self.critic_updates_per_policy_update = self.agent_params['critic_updates_per_policy_update']
        self.batch_size = self.agent_params['batch_size']
        self.max_grad_norm = self.agent_params['max_grad_norm']
        self.ent_coef = self.agent_params['ent_coef']
        self.vf_coef = self.agent_params['vf_coef']
        self.clip_range = self.agent_params['clip_range']
        self.clip_range = get_schedule_fn(self.clip_range)
        self.clip_range_vf = self.agent_params['clip_range_vf']
        self._current_progress_remaining = 1.0
        self.MseLoss = nn.MSELoss()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        self.actor = PPOMLPPolicy(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            action_space_bound=self.action_space_bound,
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
        )
        self.critic = PPOCritic(
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['learning_rate']
        )

        self.rollout_buffer = buffers.RolloutBuffer(buffer_size=ep_len, \
                                                    observation_space=env.observation_space, \
                                                    action_space=env.action_space, \
                                                    device=ptu.device, \
                                                    gamma=self.gamma, \
                                                    n_envs=1, \
                                                    normalize_value=self.agent_params['normalize_value_est'])

    def train(self):
        self.train_mode()
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining) 
        
        for _ in range(self.policy_updates_per_rollout):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                observations = rollout_data.observations
                if self.actor.discrete:
                    actions = rollout_data.actions.long().flatten()
                
                distribution, entropy = self.actor(observations)
                # epsilon = 1e-6
                # clipped_actions = torch.clamp(actions, -(self.action_space_bound - epsilon), self.action_space_bound - epsilon)
                log_prob = distribution.log_prob(actions).sum(-1)
                advantages = rollout_data.advantages
                if self.standardize_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                # print(f"ratio mean {ratio.mean()}")
                # print(f"ratio max {ratio.max():.2f}")
                # print(f"ratio min {ratio.min()}")

                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                
                for _ in range(self.agent_params['critic_updates_per_policy_update']):
                    values = self.critic(observations).squeeze(1)
                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + torch.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    value_loss = self.MseLoss(values_pred.squeeze(), rollout_data.returns)
                    # print(f"values mean {values.mean():.2f}")
                    # print(f"target mean {rollout_data.returns.mean():.2f}\n")
                    # print(f"values std {values.std():.2f}")
                    # print(f"target std {rollout_data.returns.std():.2f}\n")
                    # print(f"values max {values.argmax()} {values.max():.2f} (actual: {rollout_data.returns[values.argmax()]:.2f})")
                    # print(f"target max {rollout_data.returns.argmax()} {rollout_data.returns.max():.2f}\n")
                    # print(f"values min {values.argmin()} {values.min():.2f} (actual: {rollout_data.returns[values.argmin()]:.2f})")
                    # print(f"target min {rollout_data.returns.argmin()} {rollout_data.returns.min():.2f}\n")
                    critic_loss = self.vf_coef * value_loss
                    self.critic.optimizer.zero_grad()
                    critic_loss.backward()
                    if self.max_grad_norm is not None:
                        grad_norm_critic = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic.optimizer.step()
                
                # loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                actor_loss = policy_loss + self.ent_coef * entropy_loss
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                if self.max_grad_norm is not None:
                    grad_norm_actor = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()
                
        print(f"p loss:{policy_loss.item():.2f}, "
              f"v loss:{value_loss.item():.2f}, "
              f"ratio max:{ratio.max():.2f}, "
              f"std:{torch.exp(self.actor.logstd).mean():.2f}, "
              f"v mean:{self.rollout_buffer.value_mean:.2f}, "
              f"v std:{self.rollout_buffer.value_std:.2f}, ")
            #   f"cri grad:{grad_norm_critic:.2f}, "
            #   f"act grad:{grad_norm_actor:.2f}")
    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def zero_all_grad(self):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
    
    def step_all(self):
        self.actor.optimizer.step()
        self.critic.optimizer.step()
    
    def clip_gradient(self, max_norm):
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm)

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # TODO Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)

        v_s = self.critic.forward_np(ob_no)
        v_sp1 = self.critic.forward_np(next_ob_no)
        q_sa = re_n + self.gamma * v_sp1 * (1 - terminal_n)
        adv_n = q_sa - v_s

        if self.standardize_advantages:
            adv_n = (adv_n - adv_n.mean()) / adv_n.std()
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
    
    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, filepath)

    