import abc
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from climb.policies.base_policy import BasePolicy
from climb.infrastructure import pytorch_util as ptu
from torch import distributions
from torch.distributions import TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform
import pdb

device = ptu.device

# class TruncatedNormal(torch.distributions.Normal):
#     def __init__(self, loc, scale, lower_bound=-0.4, upper_bound=0.4):
#         super().__init__(loc, scale)
#         self.lower_bound = lower_bound
#         self.upper_bound = upper_bound
#         self._z_lower = self.cdf(lower_bound)
#         self._z_upper = self.cdf(upper_bound)
#         self._z_log_delta = torch.log(self._z_upper - self._z_lower + 1e-6)  # Normalizing factor

#     def log_prob(self, value):
#         log_prob = super().log_prob(value)  # Standard normal log-prob
#         # Apply truncation by subtracting normalization factor
#         return (log_prob - self._z_log_delta).sum(-1)

#     def sample(self, sample_shape=torch.Size()):
#         # Rejection sampling to enforce truncation
#         samples = super().sample(sample_shape)
#         samples = torch.clamp(samples, self.lower_bound, self.upper_bound)
#         return samples

#     def entropy(self):
#         # Approximate entropy for truncated normal
#         base_entropy = super().entropy()
#         return (base_entropy - self._z_log_delta).sum(-1)

class PPOMLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 action_space_bound=1.0,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
        ):

        super().__init__(**kwargs)

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline
        self.action_space_bound = action_space_bound

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers,
                                      size=self.size,
                                      output_activation="tanh")
            self.logstd = nn.Parameter(
                torch.full((self.ac_dim,), 0.0, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def get_action(self, obs: np.ndarray) -> np.ndarray:

        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        action_distribution, _ = self.forward(observation)
        action = action_distribution.sample()
        # epsilon = 1e-6
        # clipped_actions = torch.clamp(action, -(self.action_space_bound - epsilon), self.action_space_bound - epsilon)
        # log_prob(acttion).sum(-1) is required when using distributions.Normal
        return ptu.to_numpy(action.squeeze()), action_distribution.log_prob(action).sum(-1) 

    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            action_distribution = distributions.Normal(loc=batch_mean, scale=torch.exp(self.logstd))
            # action_distribution = TruncatedNormal(loc=batch_mean, scale=torch.exp(self.logstd), lower_bound=-self.action_space_bound, upper_bound=self.action_space_bound)
            entropy = action_distribution.entropy()

            # Define the transformation to map to action bounds [-action_bound, action_bound]
            transforms = [TanhTransform(cache_size=1), AffineTransform(loc=0.0, scale=self.action_space_bound)]
            action_distribution = TransformedDistribution(action_distribution, transforms)

            return action_distribution, entropy
    
    def update(self):

        raise NotImplementedError
