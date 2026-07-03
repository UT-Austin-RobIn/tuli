import os
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


class PPOLogger:
    def __init__(self, run_name, args, log_root="runs"):
        self.run_name = run_name
        self.run_dir  = f"{log_root}/{run_name}"
        os.makedirs(self.run_dir, exist_ok=True)

        self.start_time = time.time()

        if getattr(args, "track", False):
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )

        self.writer = SummaryWriter(self.run_dir)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])
            ),
        )

        self.log_file = open(f"{self.run_dir}/training.log", "w", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, self.log_file)
        sys.stderr = _Tee(sys.__stderr__, self.log_file)
        print(f"logging to {self.run_dir}/training.log", flush=True)

    # ----- console formatting ------------------------------------------------

    def print_training_header(self, args):
        print(f"\n{'='*70}")
        print(f"  run: {self.run_name}")
        print(f"  {args.num_iterations} iterations × {args.batch_size:,} steps = "
              f"{args.total_timesteps:,} total")
        print(f"  rollout: {args.num_envs} envs × {args.num_steps} steps")
        print(f"  update : {args.update_epochs} epochs × {args.num_minibatches} "
              f"minibatches (mb size {args.minibatch_size})")
        print(f"{'='*70}\n", flush=True)

    def print_iter_header(self, iteration, total_iters, global_step, total_steps):
        print(f"[iter {iteration}/{total_iters}] step {global_step:,}/"
              f"{total_steps:,} | collecting rollout...", flush=True)

    def print_policy_snapshot(self, global_step, pol_mean, pol_std, val_mean):
        m = np.array2string(pol_mean, precision=3, suppress_small=True)
        s = np.array2string(pol_std,  precision=3, suppress_small=True)
        print(f"    [step {global_step:>9,}] policy mean={m} std={s}  "
              f"V̂={val_mean:+.3f}", flush=True)

    def print_epoch(self, epoch, total_epochs, approx_kl, mb_clipfrac,
                    pg_loss, v_loss, bound_loss, grad_norm):
        print(f"    epoch {epoch:>2d}/{total_epochs}  kl={approx_kl:.4f}  "
              f"clipfrac={mb_clipfrac:.3f}  pg={pg_loss:+.4f}  vf={v_loss:.4f}  "
              f"bnd={bound_loss:.4f}  |g|={grad_norm:.3f}", flush=True)

    def print_early_stop(self, approx_kl, target_kl):
        print(f"    early stop: kl {approx_kl:.4f} > target_kl {target_kl}",
              flush=True)

    def print_iter_summary(self, *, iteration, total_iters, iter_time, sps,
                           collect_time, update_time, total_episodes,
                           iter_returns, iter_bangs, rewards_np, sat_rate,
                           early_stopped_at, supra_rate=None, strike_count=None):
        if iter_returns:
            ret_mean = float(np.mean(iter_returns))
            ret_max  = float(np.max(iter_returns))
            bang_mean = float(np.mean(iter_bangs))
            bang_max  = int(np.max(iter_bangs))
            ep_summary = (f"{len(iter_returns)} eps  "
                          f"return mean={ret_mean:.3f} max={ret_max:.3f}  "
                          f"bangs mean={bang_mean:.1f} max={bang_max}")
        else:
            ep_summary = (f"no eps  rew_mean={rewards_np.mean():.4f} "
                          f"rew_nonzero={(rewards_np > 0).mean():.3f}")
        sat_tag = f"  sat={sat_rate:.2f}" if sat_rate > 0.1 else ""
        kl_tag  = f"  early-stopped@epoch{early_stopped_at}" if early_stopped_at else ""
        if strike_count is None:
            supra_tag = ""
        elif strike_count == 0:
            supra_tag = "  supra=n/a(0 strikes)"
        else:
            supra_tag = f"  supra={supra_rate:.2f}({strike_count} strikes)"
        print(f"[iter {iteration}/{total_iters}] done in {iter_time:.1f}s  "
              f"SPS={sps}  collect={collect_time:.1f}s update={update_time:.1f}s  "
              f"total_eps={total_episodes}  |  {ep_summary}{sat_tag}{supra_tag}{kl_tag}\n",
              flush=True)

    # ----- scalar writers ----------------------------------------------------

    def log_episode(self, global_step, ep_idx, env_idx, ep_return, ep_length,
                    ep_bangs, term_kind):
        print(f"    ep#{ep_idx:<5d} env{env_idx}  return={ep_return:7.3f}  "
              f"bangs={ep_bangs:3d}  len={ep_length}  [{term_kind}]", flush=True)
        self.writer.add_scalar("charts/episodic_return",        ep_return, global_step)
        self.writer.add_scalar("charts/episodic_length",        ep_length, global_step)
        self.writer.add_scalar("charts/bang_count_per_episode", ep_bangs,  global_step)

    def log_iter(self, global_step, stats):
        for section, metrics in stats.items():
            for name, value in metrics.items():
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue
                self.writer.add_scalar(f"{section}/{name}", float(value), global_step)

    def close(self):
        self.writer.close()
        try:
            self.log_file.close()
        except Exception:
            pass


def stats_losses(*, pg_losses, v_losses, v_losses_unclipped, entropy_losses,
                 bound_losses, epoch_kls, old_approx_kl, approx_kl_final,
                 clipfracs, epoch_clipfracs, vf_clipfracs, ratios_cat,
                 explained_var):
    d = {
        "policy_loss":          float(np.mean(pg_losses)),
        "value_loss":           float(np.mean(v_losses)),
        "value_loss_unclipped": float(np.mean(v_losses_unclipped)),
        "entropy":              float(np.mean(entropy_losses)),
        "bound_loss":           float(np.mean(bound_losses)),
        "approx_kl":            float(np.mean(epoch_kls)),
        "approx_kl_final":      float(approx_kl_final),
        "old_approx_kl":        float(old_approx_kl),
        "clipfrac":             float(np.mean(clipfracs)),
        "clipfrac_final_epoch": float(epoch_clipfracs[-1]),
        "explained_variance":   float(explained_var),
        "ratio_mean":           float(ratios_cat.mean().item()),
        "ratio_max":            float(ratios_cat.max().item()),
        "ratio_min":            float(ratios_cat.min().item()),
    }
    if vf_clipfracs:
        d["vf_clipfrac"] = float(np.mean(vf_clipfracs))
    return d


def stats_policy(*, b_action_mean, b_actions, logstd):
    logstd_np = logstd.detach().cpu().numpy().flatten()
    std_np    = np.exp(logstd_np)
    d = {
        "action_std_mean":        float(std_np.mean()),
        "action_std_min":         float(std_np.min()),
        "action_std_max":         float(std_np.max()),
        "action_logstd_mean":     float(logstd_np.mean()),
        "mu_abs_mean":            float(b_action_mean.abs().mean().item()),
        "mu_abs_max":             float(b_action_mean.abs().max().item()),
        "action_abs_mean":        float(b_actions.abs().mean().item()),
        "action_saturation_rate": float((b_actions.abs() >= 0.999).float().mean().item()),
        "mu_out_of_bounds_rate":  float((b_action_mean.abs() > 1.0).float().mean().item()),
    }
    for i, s in enumerate(std_np):
        d[f"action_std_dim{i}"] = float(s)
    return d


def stats_value(*, y_pred, y_true, b_advantages, value_returns_corr):
    return {
        "value_mean":          float(np.mean(y_pred)),
        "value_std":           float(np.std(y_pred)),
        "returns_mean":        float(np.mean(y_true)),
        "returns_std":         float(np.std(y_true)),
        "returns_max":         float(np.max(y_true)),
        "advantages_mean":     float(b_advantages.mean().item()),
        "advantages_std":      float(b_advantages.std().item()),
        "advantages_abs_mean": float(b_advantages.abs().mean().item()),
        "value_returns_corr":  value_returns_corr,
    }


def stats_grad(*, grad_norms, max_grad_norm):
    arr = np.asarray(grad_norms, dtype=np.float64)
    return {
        "global_norm_mean": float(arr.mean()),
        "global_norm_max":  float(arr.max()),
        "clip_rate":        float((arr >= max_grad_norm).mean()),
    }


def stats_charts(*, rewards_np, iter_returns, iter_lengths, iter_bangs,
                 iter_delta_fns, iter_air_factors,
                 iter_terminations, iter_truncations, total_episodes,
                 learning_rate, iter_new_contacts=None, min_delta_fn=None):
    d = {
        "learning_rate":        learning_rate,
        "rollout_reward_mean":  float(rewards_np.mean()),
        "rollout_reward_max":   float(rewards_np.max()),
        "rollout_reward_sum":   float(rewards_np.sum()),
        "reward_nonzero_rate":  float((rewards_np > 0).mean()),
        "reward_nonzero_count": int((rewards_np > 0).sum()),
        "terminations":         iter_terminations,
        "truncations":          iter_truncations,
        "episodes_this_iter":   len(iter_returns),
        "total_episodes":       total_episodes,
    }
    if iter_returns:
        d["episodic_return_mean"] = float(np.mean(iter_returns))
        d["episodic_return_max"]  = float(np.max(iter_returns))
        d["episodic_return_min"]  = float(np.min(iter_returns))
        d["episodic_return_std"] = float(np.std(iter_returns))
        d["episodic_length_mean"] = float(np.mean(iter_lengths))
        d["bang_count_mean"]      = float(np.mean(iter_bangs))
        d["bang_count_max"]       = int(np.max(iter_bangs))
    if iter_delta_fns:
        deltas = np.concatenate(iter_delta_fns)
        pos = deltas[deltas > 0]
        if pos.size:
            d["delta_fn_mean_when_active"] = float(pos.mean())
            d["delta_fn_max"]              = float(deltas.max())
            d["delta_fn_active_rate"]      = float((deltas > 0).mean())
    if iter_air_factors:
        af = np.concatenate(iter_air_factors)
        strike = af[af > 0]
        d["strike_rate"]           = float((af > 0).mean())
        d["air_factor_mean_all"]   = float(af.mean())
        if strike.size:
            d["air_factor_on_strike"] = float(strike.mean())
    if iter_delta_fns and iter_new_contacts:
        deltas = np.concatenate(iter_delta_fns)
        ncs    = np.concatenate(iter_new_contacts).astype(bool)
        strike_deltas = deltas[ncs] if ncs.shape == deltas.shape else deltas[:0]
        d["strike_count"] = int(strike_deltas.size)
        if strike_deltas.size and min_delta_fn is not None:
            d["suprathreshold_strike_rate"] = float((strike_deltas >= min_delta_fn).mean())
        else:
            d["suprathreshold_strike_rate"] = float("nan")
    return d


def stats_perf(*, iter_time, collect_time, update_time, inference_time,
               env_step_time, log_time, sps, start_time, num_envs, num_steps):
    return {
        "SPS":            sps,
        "iter_time":      iter_time,
        "collect_time":   collect_time,
        "update_time":    update_time,
        "inference_time": inference_time,
        "env_step_time":  env_step_time,
        "log_time":       log_time,
        "wall_time":      time.time() - start_time,
        "collect_fps":    (num_envs * num_steps) / max(collect_time, 1e-6),
    }


def stats_info(*, iteration, epochs_completed, early_stopped_at):
    d = {"iteration": iteration, "epochs_completed": epochs_completed}
    if early_stopped_at is not None:
        d["kl_early_stop_epoch"] = early_stopped_at
    return d


def build_iter_stats(*, agent, b_obs, b_actions, b_advantages, y_pred, y_true,
                     value_returns_corr,
                     pg_losses, v_losses, v_losses_unclipped, entropy_losses,
                     bound_losses, epoch_kls, old_approx_kl, approx_kl_final,
                     clipfracs, epoch_clipfracs, vf_clipfracs, ratios_cat,
                     explained_var, grad_norms, max_grad_norm,
                     rewards_np, iter_returns, iter_lengths, iter_bangs,
                     iter_delta_fns, iter_air_factors,
                     iter_terminations, iter_truncations, total_episodes,
                     learning_rate, iter_time, collect_time, update_time,
                     inference_time, env_step_time, log_time, sps, start_time,
                     num_envs, num_steps, iteration, early_stopped_at,
                     iter_new_contacts=None, min_delta_fn=None):
    with torch.no_grad():
        b_action_mean = agent.actor_mean(b_obs).detach()
    return {
        "losses": stats_losses(
            pg_losses=pg_losses, v_losses=v_losses,
            v_losses_unclipped=v_losses_unclipped,
            entropy_losses=entropy_losses, bound_losses=bound_losses,
            epoch_kls=epoch_kls, old_approx_kl=old_approx_kl,
            approx_kl_final=approx_kl_final, clipfracs=clipfracs,
            epoch_clipfracs=epoch_clipfracs, vf_clipfracs=vf_clipfracs,
            ratios_cat=ratios_cat, explained_var=explained_var,
        ),
        "policy": stats_policy(
            b_action_mean=b_action_mean, b_actions=b_actions,
            logstd=agent.actor_logstd,
        ),
        "value": stats_value(
            y_pred=y_pred, y_true=y_true, b_advantages=b_advantages,
            value_returns_corr=value_returns_corr,
        ),
        "grad": stats_grad(
            grad_norms=grad_norms, max_grad_norm=max_grad_norm,
        ),
        "charts": stats_charts(
            rewards_np=rewards_np, iter_returns=iter_returns,
            iter_lengths=iter_lengths, iter_bangs=iter_bangs,
            iter_delta_fns=iter_delta_fns,
            iter_air_factors=iter_air_factors,
            iter_terminations=iter_terminations,
            iter_truncations=iter_truncations,
            total_episodes=total_episodes, learning_rate=learning_rate,
            iter_new_contacts=iter_new_contacts, min_delta_fn=min_delta_fn,
        ),
        "perf": stats_perf(
            iter_time=iter_time, collect_time=collect_time,
            update_time=update_time, inference_time=inference_time,
            env_step_time=env_step_time, log_time=log_time,
            sps=sps, start_time=start_time,
            num_envs=num_envs, num_steps=num_steps,
        ),
        "info": stats_info(
            iteration=iteration,
            epochs_completed=len(epoch_kls),
            early_stopped_at=early_stopped_at,
        ),
    }
