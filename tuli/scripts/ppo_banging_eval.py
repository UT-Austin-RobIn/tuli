import os
import sys
import argparse
import robosuite

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal

from envs.banging_env import BangingEnv

robosuite.environments.REGISTERED_ENVS["BangingEnv"] = BangingEnv


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def make_robosuite_env(idx, capture_video, run_name, gamma, task=None,
                       object_type="plastic_cube", reward_scale=99.5):
    import robosuite
    from robosuite.wrappers import GymWrapper
    from robosuite.controllers import load_composite_controller_config

    def thunk():
        controller_config = load_composite_controller_config(robot="Panda")
        controller_config["body_parts"]["right"]["type"] = "OSC_POSITION"
        controller_config["body_parts"]["right"]["output_max"] = [0.05, 0.05, 0.05]
        controller_config["body_parts"]["right"]["output_min"] = [-0.05, -0.05, -0.05]

        env = robosuite.make(
            "BangingEnv",
            robots=["Panda"],
            controller_configs=controller_config,
            has_renderer=capture_video,
            has_offscreen_renderer=False,
            control_freq=20,
            horizon=500,
            use_object_obs=True,
            use_contact_obs=True,
            use_camera_obs=False,
            object_type=object_type,
            reward_scale=reward_scale,
        )
        env = GymWrapper(env, keys=[
            "robot0_joint_pos_cos",
            "robot0_joint_pos_sin",
            "robot0_joint_vel",
            "robot0_eef_pos",
            "cube_pos",
            "cube_quat",
            "cube_table_contact_binary",
        ])

        env.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": 20,
            "semantics.async": False,
        }

        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        return env

    return thunk


def plot_contact_forces(force_history, episode_idx, output_dir=None):
    """Plot contact force over time for one episode."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(force_history, linewidth=0.8)
    ax.axhline(np.mean(force_history), color="r", linestyle="--", label=f"mean={np.mean(force_history):.2f}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Contact Force (normalized)")
    ax.set_title(f"Episode {episode_idx} — Contact Forces")
    ax.legend()
    ax.set_ylim(bottom=0)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"forces_ep{episode_idx:03d}.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def evaluate(model_path, object_type, reward_scale, eval_episodes=5,
             render=False, device="cpu", output_dir=None):
    envs = gym.vector.SyncVectorEnv([
        make_robosuite_env(0, render, "eval", 0.99,
                           object_type=object_type, reward_scale=reward_scale)
    ])
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    force_history = []

    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, reward, _, _, infos = envs.step(actions.cpu().numpy())

        force_history.append(float(reward[0]))

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    ep_return = info["episode"]["r"]
                    ep_idx = len(episodic_returns)
                    print(f"episode={ep_idx}, return={ep_return:.3f}, "
                          f"mean_force={np.mean(force_history):.4f}, "
                          f"max_force={np.max(force_history):.4f}")
                    plot_contact_forces(force_history, ep_idx, output_dir)
                    episodic_returns.append(ep_return)
                    force_history = []

        obs = next_obs

    print(f"\n{object_type} — {eval_episodes} episodes: "
          f"mean={np.mean(episodic_returns):.3f}, std={np.std(episodic_returns):.3f}")
    envs.close()
    return episodic_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--object-type", default="plastic_cube")
    parser.add_argument("--reward-scale", type=float, default=99.5)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    evaluate(
        args.model_path,
        object_type=args.object_type,
        reward_scale=args.reward_scale,
        eval_episodes=args.episodes,
        render=args.render,
        device=args.device,
        output_dir=args.output_dir,
    )
