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


def _find_obs_rms(env):
    while env is not None and not hasattr(env, "obs_rms"):
        env = getattr(env, "env", None)
    return env.obs_rms if env is not None else None


def _lookat_quat(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    z = -fwd
    right = np.cross(up, z); right /= np.linalg.norm(right)
    up_cam = np.cross(z, right)
    R = np.column_stack([right, up_cam, z])

    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z_ = (R[1, 0] - R[0, 1]) / s
    else:
        k = int(np.argmax(np.diag(R)))
        if k == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z_ = (R[0, 2] + R[2, 0]) / s
        elif k == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z_ = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z_ = 0.25 * s
    return np.array([w, x, y, z_], dtype=np.float64)


_CAMERA_EYE    = (1.47, -1.2, 1.15)   # 1/3 closer along the eye→target line
_CAMERA_TARGET = (0.0,   0.0, 0.95)
_CAMERA_W, _CAMERA_H = 512, 384


def set_obs_rms(vec_env, rms_dict):
    for e in vec_env.envs:
        rms = _find_obs_rms(e)
        rms.mean  = np.array(rms_dict["mean"])
        rms.var   = np.array(rms_dict["var"])
        rms.count = rms_dict["count"]


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


_save_video_trigger     = lambda ep: ep == 0
_save_video_name_prefix = "rl-video"


class _DropGripperAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        low  = env.action_space.low[:-1]
        high = env.action_space.high[:-1]
        self.action_space = gym.spaces.Box(low, high, dtype=env.action_space.dtype)
    def action(self, action):
        return np.concatenate([action, np.zeros(1, dtype=action.dtype)])


def make_robosuite_env(idx, live_render, save_video, video_dir,
                       reward_divisor=622.2,
                       material="hard", drop_gripper=False, osc_kp=150,
                       min_delta_fn=20.0, min_air_steps=5,
                       joint_torque_limits={2: 25, 4: 25}):
    import robosuite
    from robosuite.wrappers import GymWrapper
    from robosuite.controllers import load_composite_controller_config

    def thunk():
        controller_config = load_composite_controller_config(robot="Panda")
        controller_config["body_parts"]["right"]["type"] = "OSC_POSITION"
        controller_config["body_parts"]["right"]["output_max"] = [0.05, 0.05, 0.05]
        controller_config["body_parts"]["right"]["output_min"] = [-0.05, -0.05, -0.05]
        controller_config["body_parts"]["right"]["kp"] = osc_kp

        env = robosuite.make(
            "BangingEnv",
            robots=["Panda"],
            controller_configs=controller_config,
            has_renderer=live_render,
            has_offscreen_renderer=save_video,
            control_freq=20,
            horizon=500,
            use_object_obs=True,
            use_contact_obs=True,
            use_camera_obs=False,
            reward_divisor=reward_divisor,
            material=material,
            min_delta_fn=min_delta_fn,
            min_air_steps=min_air_steps,
            joint_torque_limits=joint_torque_limits,
            # hard_reset=True corrupts the offscreen GL framebuffer on reset; soft reset is required for video eval.
            hard_reset=False,
        )
        env = GymWrapper(env, keys=[
            "robot0_joint_pos_cos",
            "robot0_joint_pos_sin",
            "robot0_joint_vel",
            "robot0_eef_pos",
            "cube_pos",
            "cube_quat",
            "cube_table_contact_binary",
            "contact_force",
        ])

        env.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": 20,
            "semantics.async": False,
        }
        env.render_mode = "rgb_array"

        if save_video:
            robosuite_env = env.env
            cam_pos  = np.array(_CAMERA_EYE, dtype=np.float64)
            cam_quat = _lookat_quat(_CAMERA_EYE, _CAMERA_TARGET)

            def _rgb_render(_rs=robosuite_env, _pos=cam_pos, _quat=cam_quat):
                cid = _rs.sim.model.camera_name2id("frontview")
                _rs.sim.model.cam_pos[cid]  = _pos
                _rs.sim.model.cam_quat[cid] = _quat
                _rs.sim.forward()
                return _rs.sim.render(camera_name="frontview",
                                      width=_CAMERA_W, height=_CAMERA_H)[::-1]
            env.render = _rgb_render

        if save_video:
            env = gym.wrappers.RecordVideo(env, video_dir,
                                           episode_trigger=_save_video_trigger,
                                           name_prefix=_save_video_name_prefix,
                                           disable_logger=True)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if drop_gripper:
            env = _DropGripperAction(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        return env

    return thunk


def plot_contact_forces(force_history, episode_idx, output_dir=None):
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


def evaluate(model_path, reward_divisor, material="hard",
             eval_episodes=5, render=False, save_video=False,
             device="cpu", output_dir=None, osc_kp=150):
    model_dir = os.path.dirname(model_path) or "."
    run_dir   = os.path.dirname(model_dir) if os.path.basename(model_dir) == "model" else model_dir
    video_dir = os.path.join(run_dir, "videos")
    ckpt_stem = os.path.splitext(os.path.basename(model_path))[0]

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    action_dim = ckpt["agent"]["actor_logstd"].shape[1]
    drop_gripper = (action_dim == 3)
    print(f"checkpoint action_dim={action_dim}  → drop_gripper={drop_gripper}")

    global _save_video_trigger, _save_video_name_prefix
    _save_video_trigger     = lambda ep, n=eval_episodes: ep < n
    _save_video_name_prefix = ckpt_stem

    envs = gym.vector.SyncVectorEnv([
        make_robosuite_env(0, render, save_video, video_dir, osc_kp=osc_kp,
                           reward_divisor=reward_divisor,
                           material=material,
                           drop_gripper=drop_gripper)
    ])
    if save_video:
        print(f"saving videos to {video_dir}")
    agent = Agent(envs).to(device)
    agent.load_state_dict(ckpt["agent"])
    set_obs_rms(envs, ckpt["obs_rms"])
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    force_history = []

    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, reward, _, _, infos = envs.step(actions.cpu().numpy())

        force_history.append(float(reward[0]))

        if "episode" in infos and "_episode" in infos:
            mask  = np.asarray(infos["_episode"], dtype=bool)
            ep_rs = np.asarray(infos["episode"]["r"])
            for i in np.where(mask)[0]:
                ep_return = float(ep_rs[i])
                ep_idx = len(episodic_returns)
                print(f"episode={ep_idx}, return={ep_return:.3f}, "
                      f"mean_force={np.mean(force_history):.4f}, "
                      f"max_force={np.max(force_history):.4f}")
                plot_contact_forces(force_history, ep_idx, output_dir)
                episodic_returns.append(ep_return)
                force_history = []

        obs = next_obs

    print(f"\n{eval_episodes} episodes: "
          f"mean={np.mean(episodic_returns):.3f}, std={np.std(episodic_returns):.3f}")
    envs.close()

    if save_video and os.path.isdir(video_dir):
        import glob
        for f in glob.glob(os.path.join(video_dir, "*-episode-*.mp4")):
            new = f.replace("-episode-", "_episode_")
            os.rename(f, new)
    return episodic_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--reward-divisor", type=float, default=622.2)
    parser.add_argument("--material", choices=["hard", "soft"], default="hard",
                        help="Material preset (match the training config).")
    parser.add_argument("--osc-kp", type=float, default=150,
                        help="OSC controller stiffness (match the training config)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="open live MuJoCo viewer")
    parser.add_argument("--save-video", action="store_true", help="save MP4s to {model}_videos/")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    evaluate(
        args.model_path,
        reward_divisor=args.reward_divisor,
        material=args.material,
        eval_episodes=args.episodes,
        render=args.render,
        save_video=args.save_video,
        device=args.device,
        output_dir=args.output_dir,
        osc_kp=args.osc_kp,
    )
