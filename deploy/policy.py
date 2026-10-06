import numpy as np
import torch
import torch.nn as nn

from .bounded_policy import TanhDiagonalGaussian

BANGING_PROPRIOCEPTION_OBSERVATION_KEYS = (
    "robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel", "robot0_eef_pos",
)
HAMMERING_PROPRIOCEPTION_TARGET_KEYS = (*BANGING_PROPRIOCEPTION_OBSERVATION_KEYS, "hammer_target_xy")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Actor-critic with a tanh-squashed diagonal Gaussian policy."""

    def __init__(self, obs_dim, action_dim, *, actor_logstd_init=0.0,
                 actor_logstd_min=-5.0, actor_logstd_max=0.0, squash_action_margin=1.0e-6):
        super().__init__()
        if not actor_logstd_min <= actor_logstd_init <= actor_logstd_max:
            raise ValueError("actor_logstd_init must lie inside [actor_logstd_min, actor_logstd_max]")
        self.actor_logstd_init = float(actor_logstd_init)
        self.actor_logstd_min = float(actor_logstd_min)
        self.actor_logstd_max = float(actor_logstd_max)
        self.squash_action_margin = float(squash_action_margin)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), self.actor_logstd_init))

    def policy_metadata(self):
        return {
            "version": 1,
            "action_distribution": "tanh_normal",
            "actor_logstd_init": self.actor_logstd_init,
            "actor_logstd_min": self.actor_logstd_min,
            "actor_logstd_max": self.actor_logstd_max,
            "squash_action_margin": self.squash_action_margin,
        }

    @torch.no_grad()
    def clamp_logstd_(self):
        return self.actor_logstd.clamp_(min=self.actor_logstd_min, max=self.actor_logstd_max)

    def effective_logstd(self):
        return torch.clamp(self.actor_logstd, min=self.actor_logstd_min, max=self.actor_logstd_max)

    def _distribution(self, mean):
        return TanhDiagonalGaussian(
            mean, self.effective_logstd().expand_as(mean),
            logstd_min=self.actor_logstd_min, logstd_max=self.actor_logstd_max,
            action_margin=self.squash_action_margin)

    def deterministic_action(self, x):
        return self._distribution(self.actor_mean(x)).deterministic_action()

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        probs = self._distribution(self.actor_mean(x))
        if action is None:
            action = probs.sample().action
        log_prob = probs.log_prob(action)
        # The squashed Gaussian has no closed-form entropy; estimate it from a fresh sample.
        entropy = -probs.rsample().log_prob
        return action, log_prob, entropy, self.critic(x)
