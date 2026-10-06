import numpy as np
import torch

from .policy import BANGING_PROPRIOCEPTION_OBSERVATION_KEYS, HAMMERING_PROPRIOCEPTION_TARGET_KEYS, Agent


class SavedPolicy:
    """A saved banging (24-input) or hammering (26-input) policy acting on raw, unnormalized observations."""

    def __init__(self, checkpoint, device="cpu"):
        keys = checkpoint.get("observation_keys") or checkpoint["env_metadata"]["observation_keys"]
        policy = checkpoint["policy_metadata"]
        if (tuple(keys) not in (BANGING_PROPRIOCEPTION_OBSERVATION_KEYS, HAMMERING_PROPRIOCEPTION_TARGET_KEYS)
                or policy["version"] != 1 or policy["action_distribution"] != "tanh_normal"):
            raise ValueError("expected a proprioceptive tanh-normal banging or hammering checkpoint")
        self.observation_keys = tuple(keys)
        self.observation_dim = 24 if self.observation_keys == BANGING_PROPRIOCEPTION_OBSERVATION_KEYS else 26
        self.agent = Agent(self.observation_dim, 3, **{k: policy[k] for k in (
            "actor_logstd_init", "actor_logstd_min", "actor_logstd_max", "squash_action_margin")}).to(device)
        self.agent.load_state_dict(checkpoint["agent"])
        self.agent.requires_grad_(False)
        self.agent.eval()
        self.device = device
        self.obs_mean = np.asarray(checkpoint["obs_rms"]["mean"], dtype=np.float64)
        self.obs_var = np.asarray(checkpoint["obs_rms"]["var"], dtype=np.float64)

    @classmethod
    def load(cls, path, device="cpu"):
        return cls(torch.load(path, map_location=device, weights_only=False), device)

    def normalize(self, observation):
        normalized = (np.asarray(observation, dtype=np.float64) - self.obs_mean) / np.sqrt(self.obs_var + 1.0e-8)
        return torch.as_tensor(np.clip(normalized, -10.0, 10.0).astype(np.float32), device=self.device)

    @torch.no_grad()
    def sample_action(self, observation):
        action, _, _, _ = self.agent.get_action_and_value(self.normalize(observation).reshape(1, -1))
        return action

    @torch.no_grad()
    def deterministic_action(self, observation):
        return self.agent.deterministic_action(self.normalize(observation).reshape(1, -1))
