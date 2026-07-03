import os
import sys
import argparse
import random
import time
import datetime
import yaml
import robosuite

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from envs.hammering_env import HammeringEnv
from utils.ppo_logger import PPOLogger, build_iter_stats

robosuite.environments.REGISTERED_ENVS["HammeringEnv"] = HammeringEnv


_CONFIG_DEFAULTS = {
    "bound_coef": 0.0,
    "adaptive_kl": False,
    "desired_kl":  0.01,
    "lr_min":      1.0e-5,
    "lr_max":      1.0e-2,
    "vec_env_type": "sync",
    "drop_gripper": True,
    "osc_kp": 150,
    "min_air_steps": 5,
    "min_delta_fn": 0.0,
    "joint_torque_limits": None,

    "reward_divisor": 622.2,
    "material": "hard",

    "reward_mode":    "target",   # "target" | "repeat"
    "target_pos":     [0.0, 0.0], # world-frame XY of target (target mode) / initial anchor (repeat)
    "target_radius":  0.03,       # target cylinder radius (target mode)
    "target_height":  0.005,
    "repeat_sigma":   0.05,       # proximity σ for repeat mode
}


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    flat = {}
    for section in cfg.values():
        flat.update(section)

    for k, v in _CONFIG_DEFAULTS.items():
        flat.setdefault(k, v)

    flat["batch_size"]     = flat["num_envs"] * flat["num_steps"]
    flat["minibatch_size"] = flat["batch_size"] // flat["num_minibatches"]
    flat["num_iterations"] = flat["total_timesteps"] // flat["batch_size"]

    if flat.get("num_checkpoints") is not None:
        flat["save_interval"] = max(1, flat["num_iterations"] // flat["num_checkpoints"])
    else:
        flat.setdefault("save_interval", flat["num_iterations"])

    return argparse.Namespace(**flat)


class _DropGripperAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        low  = env.action_space.low[:-1]
        high = env.action_space.high[:-1]
        self.action_space = gym.spaces.Box(low, high, dtype=env.action_space.dtype)

    def action(self, action):
        return np.concatenate([action, np.zeros(1, dtype=action.dtype)])


def make_robosuite_env(reward_divisor=622.2,
                       material="hard", min_air_steps=5,
                       min_delta_fn=0.0, joint_torque_limits=None,
                       horizon=500, control_freq=20,
                       drop_gripper=False, osc_kp=150,
                       reward_mode="target",
                       target_pos=(0.0, 0.0),
                       target_radius=0.03,
                       target_height=0.005,
                       repeat_sigma=0.05):
    import robosuite
    from robosuite.wrappers import GymWrapper
    from robosuite.controllers import load_composite_controller_config

    def thunk():
        import robosuite as _rs
        from envs.hammering_env import HammeringEnv as _HE
        _rs.environments.REGISTERED_ENVS["HammeringEnv"] = _HE

        controller_config = load_composite_controller_config(robot="Panda")
        controller_config["body_parts"]["right"]["type"] = "OSC_POSITION"
        controller_config["body_parts"]["right"]["output_max"] = [0.05, 0.05, 0.05]
        controller_config["body_parts"]["right"]["output_min"] = [-0.05, -0.05, -0.05]
        controller_config["body_parts"]["right"]["kp"] = osc_kp

        env = robosuite.make(
            "HammeringEnv",
            robots=["Panda"],
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=False,
            control_freq=control_freq,
            horizon=horizon,
            use_object_obs=True,
            use_contact_obs=True,
            use_camera_obs=False,
            reward_divisor=reward_divisor,
            material=material,
            min_air_steps=min_air_steps,
            min_delta_fn=min_delta_fn,
            joint_torque_limits=joint_torque_limits,
            reward_mode=reward_mode,
            target_pos=tuple(target_pos),
            target_radius=target_radius,
            target_height=target_height,
            repeat_sigma=repeat_sigma,
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
            "hammer_target_xy",
        ])
        env.metadata = {"render_fps": control_freq}

        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if drop_gripper:
            env = _DropGripperAction(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        return env

    return thunk


def _find_obs_rms(env):
    while env is not None and not hasattr(env, "obs_rms"):
        env = getattr(env, "env", None)
    return env.obs_rms if env is not None else None


def get_obs_rms(vec_env):
    if isinstance(vec_env, gym.vector.SyncVectorEnv):
        rms = _find_obs_rms(vec_env.envs[0])
        mean, var, count = rms.mean, rms.var, rms.count
    else:
        results = vec_env.call("get_wrapper_attr", "obs_rms")
        rms = results[0]
        mean, var, count = rms.mean, rms.var, rms.count
    return {
        "mean":  np.array(mean),
        "var":   np.array(var),
        "count": float(count),
    }


def set_obs_rms(vec_env, rms_dict):
    from gymnasium.wrappers.utils import RunningMeanStd
    mean  = np.array(rms_dict["mean"],  dtype=np.float64)
    var   = np.array(rms_dict["var"],   dtype=np.float64)
    count = float(rms_dict["count"])

    def _make():
        r = RunningMeanStd(shape=mean.shape)
        r.mean, r.var, r.count = mean.copy(), var.copy(), count
        return r

    vec_env.set_attr("obs_rms", [_make() for _ in range(vec_env.num_envs)])


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


def warm_start_from_checkpoint(agent, ckpt):
    src = ckpt["agent"]
    dst = agent.state_dict()
    full, partial, skipped = [], [], []
    for k, v in src.items():
        if k not in dst:
            skipped.append(k)
            continue
        t = dst[k]
        if t.shape == v.shape:
            dst[k] = v
            full.append(k)
        elif (v.ndim == 2 and t.ndim == 2
              and t.shape[0] == v.shape[0] and t.shape[1] >= v.shape[1]):
            new_t = torch.zeros_like(t)
            new_t[:, :v.shape[1]] = v.to(new_t.dtype)
            dst[k] = new_t
            partial.append(k)
        else:
            print(f"[warm-start] skip {k}: shape {tuple(v.shape)} !-> "
                  f"{tuple(t.shape)} (unhandled)", flush=True)
            skipped.append(k)
    agent.load_state_dict(dst)
    return full, partial, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to YAML config file")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="optional path to a banging checkpoint to warm-start from")
    cli_args, _ = parser.parse_known_args()
    args = load_config(cli_args.config)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.name}_{ts}"

    logger = PPOLogger(run_name, args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    env_fns = [make_robosuite_env(reward_divisor=args.reward_divisor,
                                  material=args.material,
                                  min_air_steps=args.min_air_steps,
                                  min_delta_fn=args.min_delta_fn,
                                  joint_torque_limits=args.joint_torque_limits,
                                  horizon=args.horizon,
                                  control_freq=args.control_freq,
                                  drop_gripper=args.drop_gripper,
                                  osc_kp=args.osc_kp,
                                  reward_mode=args.reward_mode,
                                  target_pos=args.target_pos,
                                  target_radius=args.target_radius,
                                  target_height=args.target_height,
                                  repeat_sigma=args.repeat_sigma)
               for _ in range(args.num_envs)]
    if args.vec_env_type == "async":
        print(f"[vec_env] AsyncVectorEnv with {args.num_envs} subprocess workers "
              f"(expect ~{args.num_envs * 2}s startup)", flush=True)
        envs = gym.vector.AsyncVectorEnv(env_fns, context="spawn")
    else:
        envs = gym.vector.SyncVectorEnv(env_fns)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if cli_args.pretrained:
        ckpt = torch.load(cli_args.pretrained, map_location=device, weights_only=False)
        full, partial, skipped = warm_start_from_checkpoint(agent, ckpt)
        print(f"[warm-start] {cli_args.pretrained}: "
              f"copied {len(full)}, partial-copied {len(partial)} {partial}, "
              f"skipped {len(skipped)} {skipped}", flush=True)

        if "obs_rms" in ckpt:
            new_dim  = int(np.array(envs.single_observation_space.shape).prod())
            src_mean = np.asarray(ckpt["obs_rms"]["mean"], dtype=np.float64)
            src_var  = np.asarray(ckpt["obs_rms"]["var"],  dtype=np.float64)
            pad = new_dim - src_mean.shape[0]
            padded = {
                "mean":  np.concatenate([src_mean, np.zeros(pad)]) if pad > 0 else src_mean,
                "var":   np.concatenate([src_var,  np.ones(pad)])  if pad > 0 else src_var,
                "count": float(ckpt["obs_rms"]["count"]),
            }
            set_obs_rms(envs, padded)
            print(f"[warm-start] installed obs_rms (dim {src_mean.shape[0]}->{new_dim}, "
                  f"count={padded['count']:.1f}) into {envs.num_envs} envs", flush=True)
        else:
            print("[warm-start] checkpoint has no obs_rms; using fresh normalization",
                  flush=True)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    bang_counts = np.zeros(args.num_envs)
    total_episodes = 0

    logger.print_training_header(args)

    for iteration in range(1, args.num_iterations + 1):
        iter_start = time.time()
        logger.print_iter_header(iteration, args.num_iterations, global_step, args.total_timesteps)

        if not args.adaptive_kl and args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        iter_returns, iter_lengths, iter_bangs = [], [], []
        iter_delta_fns, iter_air_factors, iter_new_contacts = [], [], []
        iter_terminations, iter_truncations = 0, 0
        steps_per_policy_log = max(1, 1000 // args.num_envs)

        collect_start = time.time()
        inference_time, env_step_time = 0.0, 0.0
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            t0 = time.time()
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            inference_time += time.time() - t0

            if step > 0 and step % steps_per_policy_log == 0:
                with torch.no_grad():
                    pol_mean = agent.actor_mean(next_obs).mean(0).cpu().numpy()
                    pol_std  = torch.exp(agent.actor_logstd).cpu().numpy().flatten()
                    val_mean = float(value.mean().item())
                logger.print_policy_snapshot(global_step, pol_mean, pol_std, val_mean)

            t0 = time.time()
            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            env_step_time += time.time() - t0

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            if "new_contact" in infos:
                nc = np.asarray(infos["new_contact"], dtype=np.float64)
                bang_counts += nc
                iter_new_contacts.append(nc)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done).to(device)

            iter_terminations += int(np.sum(terminations))
            iter_truncations  += int(np.sum(truncations))

            if "delta_fn" in infos:
                iter_delta_fns.append(np.asarray(infos["delta_fn"], dtype=np.float64))
            if "air_factor" in infos:
                iter_air_factors.append(np.asarray(infos["air_factor"], dtype=np.float64))

            if "episode" in infos and "_episode" in infos:
                mask  = np.asarray(infos["_episode"], dtype=bool)
                ep_rs = np.asarray(infos["episode"]["r"])
                ep_ls = np.asarray(infos["episode"]["l"])
                for i in np.where(mask)[0]:
                    total_episodes += 1
                    ep_return = float(ep_rs[i])
                    ep_length = int(ep_ls[i])
                    ep_bangs  = int(bang_counts[i])
                    iter_returns.append(ep_return)
                    iter_lengths.append(ep_length)
                    iter_bangs.append(ep_bangs)
                    term_kind = "term" if terminations[i] else "trunc"
                    logger.log_episode(global_step, total_episodes, i,
                                       ep_return, ep_length, ep_bangs, term_kind)
                    bang_counts[i] = 0.0
        collect_time = time.time() - collect_start

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs        = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs   = logprobs.reshape(-1)
        b_actions    = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns    = returns.reshape(-1)
        b_values     = values.reshape(-1)

        print(f"  PPO update ({args.update_epochs} epochs × {args.num_minibatches} mb)", flush=True)
        update_start = time.time()

        b_inds = np.arange(args.batch_size)
        clipfracs, vf_clipfracs, epoch_clipfracs, epoch_kls = [], [], [], []
        pg_losses, v_losses, v_losses_unclipped = [], [], []
        entropy_losses, bound_losses, grad_norms = [], [], []
        ratios_all = []
        early_stopped_at = None
        approx_kl = torch.tensor(0.0)
        old_approx_kl = torch.tensor(0.0)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_action_mean = agent.actor_mean(b_obs[mb_inds])
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                    ratios_all.append(ratio.detach())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss_raw = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    with torch.no_grad():
                        vf_clipfracs.append(((v_loss_clipped > v_loss_unclipped).float().mean()).item())
                else:
                    v_loss = v_loss_raw

                entropy_loss = entropy.mean()
                bound_loss   = (torch.relu(mb_action_mean.abs() - 1.0) ** 2).mean()

                loss = (pg_loss
                        - args.ent_coef  * entropy_loss
                        + args.vf_coef   * v_loss
                        + args.bound_coef * bound_loss)

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                v_losses_unclipped.append(v_loss_raw.item())
                entropy_losses.append(entropy_loss.item())
                bound_losses.append(bound_loss.item())
                grad_norms.append(float(grad_norm))

            mb_clipfrac = float(np.mean(clipfracs[-args.num_minibatches:]))
            epoch_clipfracs.append(mb_clipfrac)
            epoch_kls.append(float(approx_kl.item()))
            logger.print_epoch(epoch + 1, args.update_epochs,
                               approx_kl.item(), mb_clipfrac,
                               pg_loss.item(), v_loss.item(),
                               bound_loss.item(), float(grad_norm))

            if args.target_kl is not None and approx_kl > args.target_kl:
                early_stopped_at = epoch + 1
                logger.print_early_stop(approx_kl.item(), args.target_kl)
                break
        update_time = time.time() - update_start

        if args.adaptive_kl and epoch_kls:
            mean_kl = float(np.mean(epoch_kls))
            cur_lr = optimizer.param_groups[0]["lr"]
            if mean_kl > 2.0 * args.desired_kl:
                new_lr = cur_lr / 1.5
            elif mean_kl < 0.5 * args.desired_kl and mean_kl > 0.0:
                new_lr = cur_lr * 1.5
            else:
                new_lr = cur_lr
            new_lr = float(np.clip(new_lr, args.lr_min, args.lr_max))
            optimizer.param_groups[0]["lr"] = new_lr

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        value_returns_corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if var_y > 0 else float("nan")
        ratios_cat = torch.cat(ratios_all)

        log_start = time.time()
        sps = int(global_step / (time.time() - start_time))
        iter_time = time.time() - iter_start

        stats = build_iter_stats(
            agent=agent,
            b_obs=b_obs, b_actions=b_actions, b_advantages=b_advantages,
            y_pred=y_pred, y_true=y_true, value_returns_corr=value_returns_corr,
            pg_losses=pg_losses, v_losses=v_losses,
            v_losses_unclipped=v_losses_unclipped,
            entropy_losses=entropy_losses, bound_losses=bound_losses,
            epoch_kls=epoch_kls, old_approx_kl=old_approx_kl.item(),
            approx_kl_final=approx_kl.item(),
            clipfracs=clipfracs, epoch_clipfracs=epoch_clipfracs,
            vf_clipfracs=vf_clipfracs, ratios_cat=ratios_cat,
            explained_var=explained_var,
            grad_norms=grad_norms, max_grad_norm=args.max_grad_norm,
            rewards_np=rewards.cpu().numpy(),
            iter_returns=iter_returns, iter_lengths=iter_lengths,
            iter_bangs=iter_bangs,
            iter_delta_fns=iter_delta_fns, iter_air_factors=iter_air_factors,
            iter_terminations=iter_terminations,
            iter_truncations=iter_truncations,
            total_episodes=total_episodes,
            learning_rate=optimizer.param_groups[0]["lr"],
            iter_time=iter_time, collect_time=collect_time,
            update_time=update_time, inference_time=inference_time,
            env_step_time=env_step_time,
            log_time=0.0,
            sps=sps, start_time=start_time,
            num_envs=args.num_envs, num_steps=args.num_steps,
            iteration=iteration, early_stopped_at=early_stopped_at,
            iter_new_contacts=iter_new_contacts, min_delta_fn=args.min_delta_fn,
        )
        log_time = time.time() - log_start
        stats["perf"]["log_time"] = log_time
        logger.log_iter(global_step, stats)

        sat_rate = stats["policy"]["action_saturation_rate"]
        logger.print_iter_summary(
            iteration=iteration, total_iters=args.num_iterations,
            iter_time=iter_time, sps=sps,
            collect_time=collect_time, update_time=update_time,
            total_episodes=total_episodes,
            iter_returns=iter_returns, iter_bangs=iter_bangs,
            rewards_np=rewards.cpu().numpy(), sat_rate=sat_rate,
            early_stopped_at=early_stopped_at,
            supra_rate=stats["charts"].get("suprathreshold_strike_rate"),
            strike_count=stats["charts"].get("strike_count"),
        )

        if iteration % args.save_interval == 0 and args.save_model:
            model_dir = f"runs/{run_name}/model"
            os.makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/{args.name}_{iteration}.cleanrl_model"
            torch.save({"agent": agent.state_dict(), "obs_rms": get_obs_rms(envs)}, model_path)
            print(f"model saved to {model_path}")

    if args.save_model:
        model_dir = f"runs/{run_name}/model"
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/{args.name}_final.cleanrl_model"
        torch.save({"agent": agent.state_dict(), "obs_rms": get_obs_rms(envs)}, model_path)
        print(f"model saved to {model_path}")

    envs.close()
    logger.close()
