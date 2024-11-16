import abc
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import utils
from climb.infrastructure import pytorch_util as ptu

class PPOCritic(nn.Module):

    def __init__(self,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 **kwargs
        ):

        super().__init__(**kwargs)

        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training

        self.value_net = ptu.build_mlp(input_size=self.ob_dim, output_size=1, n_layers=self.n_layers, size=self.size).to(ptu.device)
        self.optimizer = optim.Adam(
                self.value_net.parameters(),
                self.learning_rate
            )

    def forward(self, observations):
        return self.value_net(observations)
    
    # def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
    #     """
    #         Update the parameters of the critic.

    #         let sum_of_path_lengths be the sum of the lengths of the paths sampled from
    #             Agent.sample_trajectories
    #         let num_paths be the number of paths sampled from Agent.sample_trajectories

    #         arguments:
    #             ob_no: shape: (sum_of_path_lengths, ob_dim)
    #             ac_na: length: sum_of_path_lengths. The action taken at the current step.
    #             next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
    #             reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
    #                 the reward for each timestep
    #             terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
    #                 at that timestep of 0 if the episode did not end

    #         returns:
    #             training loss
    #     """
    #     # Copied from HW3
    #     for grad_steps in range(self.num_grad_steps_per_target_update * self.num_target_updates):
    #         if grad_steps % self.num_grad_steps_per_target_update == 0:
    #             V_s_next = self(next_ob_no) * (1 - terminal_n)
    #             target = (reward_n + self.gamma * V_s_next).detach()
    #         loss = self.loss(self(ob_no), target)
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #     return loss.item()
