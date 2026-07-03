import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config

from envs.banging_env import BangingEnv

robosuite.environments.REGISTERED_ENVS["BangingEnv"] = BangingEnv

# match ppo_banging.py's training plant exactly
OSC_KP = 150
OSC_DELTA = 0.05
TORQUE_LIMITS = {2: 25, 4: 25}
CONTROL_FREQ = 20
HORIZON = 500


def make_env(material):
    controller_config = load_composite_controller_config(robot="Panda")
    controller_config["body_parts"]["right"]["type"] = "OSC_POSITION"
    controller_config["body_parts"]["right"]["output_max"] = [OSC_DELTA] * 3
    controller_config["body_parts"]["right"]["output_min"] = [-OSC_DELTA] * 3
    controller_config["body_parts"]["right"]["kp"] = OSC_KP

    return robosuite.make(
        "BangingEnv",
        robots=["Panda"],
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        control_freq=CONTROL_FREQ,
        horizon=HORIZON,
        use_object_obs=True,
        use_contact_obs=True,
        use_camera_obs=False,
        reward_divisor=1.0,      # irrelevant: we record raw Fn, not reward
        material=material,
        min_air_steps=5,
        use_air_gate=True,
        min_delta_fn=0.0,        # record everything; thresholds are chosen later
        joint_torque_limits=TORQUE_LIMITS,
    )


class BangPolicy:
    MAX_DOWN_STEPS = 60  # bail-out if the arm never reaches the table

    def __init__(self, rng):
        self.rng = rng
        self.reset()

    def reset(self):
        self.phase = "down"
        self.phase_steps = 0
        self.up_steps = 0
        self.xy = np.zeros(2)

    def act(self, info):
        self.phase_steps += 1
        if self.phase == "down":
            if info.get("new_contact", False) or self.phase_steps > self.MAX_DOWN_STEPS:
                self.phase = "up"
                self.phase_steps = 0
                self.up_steps = int(self.rng.integers(4, 13))  # varies drop height/impact speed
                self.xy = self.rng.uniform(-0.15, 0.15, size=2)
            dz = -1.0
        else:
            if self.phase_steps >= self.up_steps:
                self.phase = "down"
                self.phase_steps = 0
            dz = 1.0
        return np.array([self.xy[0] * 0.2, self.xy[1] * 0.2, dz, 0.0])


class PressPolicy:
    def __init__(self, rng):
        self.rng = rng
        self.reset()

    def reset(self):
        self.t = 0
        self.in_press = False

    def act(self, info):
        self.t += 1
        contact_now = info.get("air_run", 1) == 0
        if not self.in_press:
            if contact_now:
                self.in_press = True
            return np.array([0.0, 0.0, -1.0, 0.0])
        if not contact_now:
            return np.array([0.0, 0.0, -1.0, 0.0])
        dz = -0.55 + 0.45 * np.sin(2.0 * np.pi * self.t / 20.0)  # sinusoidal press depth in [-1,-0.1], ~1 Hz
        return np.array([0.0, 0.0, dz, 0.0])


def record(material, mode, n_steps, seed):
    rng = np.random.default_rng(seed)
    env = make_env(material)
    policy = BangPolicy(rng) if mode == "bang" else PressPolicy(rng)

    fn = np.zeros(n_steps, dtype=np.float64)
    delta_fn = np.zeros(n_steps, dtype=np.float64)
    new_contact = np.zeros(n_steps, dtype=bool)
    air_ok = np.zeros(n_steps, dtype=bool)

    env.reset()
    policy.reset()
    info = {}
    for t in range(n_steps):
        obs, reward, done, info = env.step(policy.act(info))
        fn[t] = float(env._total_outer_fn)
        delta_fn[t] = float(info["delta_fn"])
        new_contact[t] = bool(info["new_contact"])
        air_ok[t] = info["air_factor"] > 0.0
        if done:
            env.reset()
            policy.reset()
            info = {}
        if (t + 1) % 2000 == 0:
            print(f"  [{material}/{mode}] {t + 1:,}/{n_steps:,} steps  "
                  f"strikes={int(new_contact[:t + 1].sum()):,}  "
                  f"Fn max={fn[:t + 1].max():.1f}", flush=True)
    env.close()
    return fn, delta_fn, new_contact, air_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", choices=("hard", "soft"), required=True)
    parser.add_argument("--mode", choices=("bang", "press"), default="bang")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    fn, delta_fn, new_contact, air_ok = record(
        args.material, args.mode, args.steps, args.seed
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        total_normal_force=fn,
        delta_fn=delta_fn,
        new_contact=new_contact,
        air_factor=air_ok,
        material=args.material,
        mode=args.mode,
        seed=args.seed,
    )

    strikes = delta_fn[new_contact]
    print(f"\n[done] {args.out}: {fn.size:,} steps, "
          f"{int(new_contact.sum()):,} strikes")
    if strikes.size:
        print(f"  strike |dFn|: p50={np.percentile(strikes, 50):.1f}  "
              f"p95={np.percentile(strikes, 95):.1f}  "
              f"max={strikes.max():.1f} N")


if __name__ == "__main__":
    main()
