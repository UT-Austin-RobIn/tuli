import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal


class SquashedGaussianSample(NamedTuple):
    action: Tensor
    log_prob: Tensor
    pre_tanh: Tensor


class TanhDiagonalGaussian:
    """Diagonal Gaussian followed by (1 - action_margin) * tanh, so actions stay strictly inside [-1, 1].

    PPO stores the bounded action and recomputes its log probability through the exact inverse tanh.
    """

    def __init__(self, mean, logstd, *, logstd_min=-5.0, logstd_max=0.0, action_margin=1.0e-6):
        self.mean = mean
        self.logstd = torch.clamp(logstd, min=logstd_min, max=logstd_max)
        self.base_dist = Normal(self.mean, torch.exp(self.logstd))
        self.action_margin = max(float(action_margin), float(torch.finfo(mean.dtype).eps))
        self.inverse_epsilon = max(float(torch.finfo(mean.dtype).eps), 1.0e-7)

    @property
    def action_scale(self):
        return self.mean.new_tensor(1.0 - self.action_margin)

    def _squash(self, pre_tanh):
        return self.action_scale * torch.tanh(pre_tanh)

    def deterministic_action(self):
        return self._squash(self.mean)

    def sample(self):
        pre_tanh = self.base_dist.sample()
        return SquashedGaussianSample(self._squash(pre_tanh), self.log_prob_from_pre_tanh(pre_tanh), pre_tanh)

    def rsample(self):
        pre_tanh = self.base_dist.rsample()
        return SquashedGaussianSample(self._squash(pre_tanh), self.log_prob_from_pre_tanh(pre_tanh), pre_tanh)

    def inverse(self, action):
        normalized = torch.clamp(action / self.action_scale, min=-1.0 + self.inverse_epsilon,
                                 max=1.0 - self.inverse_epsilon)
        return 0.5 * (torch.log1p(normalized) - torch.log1p(-normalized))

    def log_prob_from_pre_tanh(self, pre_tanh):
        # Operation order is load-bearing: it fixes autograd's gradient summation order and so bit-exact training.
        base = self.base_dist.log_prob(pre_tanh)
        # log(1 - tanh(z)^2) = 2 * (log 2 - z - softplus(-2z)), finite for large |z|.
        log_tanh_jacobian = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        return (base - (torch.log(self.action_scale) + log_tanh_jacobian)).sum(dim=-1)

    def log_prob(self, action):
        return self.log_prob_from_pre_tanh(self.inverse(action))
