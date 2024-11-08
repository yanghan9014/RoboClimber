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
