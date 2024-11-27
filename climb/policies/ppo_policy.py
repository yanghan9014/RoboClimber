import abc
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from climb.policies.base_policy import BasePolicy
from climb.infrastructure import pytorch_util as ptu
from torch import distributions
from torch.distributions.transforms import TanhTransform
import pdb

device = ptu.device


class TruncatedNormal(distributions.Distribution):
    def __init__(self, mean, scale_tril, min_val, max_val):
        self.mea = mean
        self.scale_tril = scale_tril
        self.min_val = min_val
        self.max_val = max_val
        self._normal = distributions.MultivariateNormal(mean, scale_tril=scale_tril)
        
        # Calculate the total probability mass in the truncated region
        self.total_prob = self._normal.cdf(max_val) - self._normal.cdf(min_val)

    def sample(self, sample_shape=torch.Size()):
        """
        Sample from the truncated normal distribution.
        """
        size = sample_shape
        tmp = self._normal.sample(size)  # Sample from MultivariateNormal
        
        # Check which values are valid (inside the truncation bounds)
        valid = (tmp >= self.min_val) & (tmp <= self.max_val)
        
        # Gather the valid samples
        valid_samples = tmp[valid]
        
        # Rescale to the desired mean and std
        valid_samples.mul_(self.std).add_(self.mea)
        
        return valid_samples

    def log_prob(self, value):
        """
        Compute the log-probability for the truncated normal distribution.
        """
        # Ensure the value is within the truncated bounds
        if torch.any((value < self.min_val) | (value > self.max_val)):
            return torch.full_like(value, float('-inf'))
        
        # Calculate the log-probability for the normal distribution
        log_prob = self._normal.log_prob(value)
        
        # Normalize the PDF to the truncated region (uniform-like inside the truncated region)
        log_prob -= torch.log(self.total_prob)
        
        return log_prob

    def cdf(self, value):
        """
        Compute the CDF for the truncated normal distribution.
        """
        # Calculate the CDF for the normal distribution
        cdf_val = self._normal.cdf(value)
        
        # Normalize the CDF to the truncated region
        return (cdf_val - self._normal.cdf(self.min_val)) / self.total_prob

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
                                      n_layers=self.n_layers, size=self.size)
            self.std = nn.Parameter(
                0.2*torch.ones(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.std.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.std], self.mean_net.parameters()),
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
        action = action_distribution.rsample()
        # print('###################')
        # action_tanh = torch.tanh(action)
        action_scaled = action * self.action_space_bound
        
        epsilon = 1e-6
        clipped_actions = torch.clamp(action, -(self.action_space_bound - epsilon), self.action_space_bound - epsilon)
        # return ptu.to_numpy(action.squeeze()), action_distribution.log_prob(clipped_actions)
        # return ptu.to_numpy(clipped_actions.squeeze()), action_distribution.log_prob(clipped_actions)
        return ptu.to_numpy(action_scaled.squeeze()), action_distribution.log_prob(action)
    
    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            # scale_tril = torch.diag(self.std)
            scale_tril = torch.diag(self.std)
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            base_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            # base_distribution = TruncatedNormal(batch_mean, batch_scale_tril, batch_mean-1, batch_mean+1)
            # base_distribution = torch.distributions.Normal(batch_mean, torch.exp(self.std))

            # print(base_distribution)
            # base_distribution = torch.distributions.Normal(batch_mean, torch.exp(self.std))
            # print(base_distribution)
            entropy = base_distribution.entropy()

            # Apply tanh transformation to enforce [-1, 1] range
            tanh_transform = TanhTransform()
            transformed_distribution = distributions.TransformedDistribution(base_distribution, [tanh_transform])
            action_distribution = distributions.TransformedDistribution(transformed_distribution, [
                torch.distributions.transforms.AffineTransform(loc=0.0, scale=self.action_space_bound)
            ])
            # action_distribution = distributions.TransformedDistribution(base_distribution, [
                # torch.distributions.transforms.AffineTransform(loc=0.0, scale=self.action_space_bound)
            # ])
            # return base_distribution, entropy
            # return action_distribution, entropy
            return transformed_distribution, entropy
    
    def update(self):

        raise NotImplementedError
