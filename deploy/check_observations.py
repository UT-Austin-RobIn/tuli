"""Read-only robot check: build the policy inputs from live Deoxys state and print the actions. Sends no commands.

python -m deploy.check_observations --policy finetune --target 0.41 0.0
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .frozen_policy import SavedPolicy
from .gp import BoundedGP, FeatureMap, rank_poses
from .observation import ACTION_SCALE, FLANGE_OFFSET, SIM_BASE, build_observation, grip_site_base, link7_pose

POLICIES = Path(__file__).resolve().parent / "policies"
CHECKPOINTS = {"banging": "banging_ckpt17", "scratch": "scratch_ckpt109",
               "finetune": "finetune_ckpt28", "gp_ucb": "banging_ckpt17"}
OUT_OF_RANGE = 5.0  # normalized input beyond this many training standard deviations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--target", type=float, nargs=2, metavar=("X", "Y"),
                        help="target center in the robot base frame (m); needed for scratch, finetune, gp_ucb")
    parser.add_argument("--interface-cfg", default="charmander.yml")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out", default=None, help="save the readings to this .npz")
    args = parser.parse_args()
    if args.policy != "banging" and args.target is None:
        parser.error(f"--target is needed for {args.policy}")
    return args


def wait_for_state(robot, timeout=10.0):
    start = time.time()
    while robot.state_buffer_size == 0 or not robot.check_nonzero_configuration():
        if time.time() - start > timeout:
            raise SystemExit("no robot state: is auto_arm.sh running on the NUC, and is no other program using the arm?")
        time.sleep(0.05)


def tool_offset(q, O_T_EE):
    """Robot-reported EE position against our kinematics: (offset along the flange z, sideways error)."""
    L7 = link7_pose(q)
    flange, z = L7[:3, 3] + L7[:3, 2] * FLANGE_OFFSET, L7[:3, 2]
    d = O_T_EE[:3, 3] - flange
    along = float(d @ z)
    return along, float(np.linalg.norm(d - along * z))


def main():
    args = parse_args()
    from deoxys import config_root
    from deoxys.franka_interface import FrankaInterface

    policy = SavedPolicy.load(POLICIES / f"{CHECKPOINTS[args.policy]}.cleanrl_model")
    target_base = None if args.target is None else np.asarray(args.target, dtype=np.float64)
    gp_pick = None
    if target_base is not None:
        gp, meta = BoundedGP.load(POLICIES / "gp_ucb_ckpt37.npz")
        low, high = np.asarray(meta["target_low"]) - SIM_BASE[:2], np.asarray(meta["target_high"]) - SIM_BASE[:2]
        if np.any(target_base < low) or np.any(target_base > high):
            print(f"WARNING target {target_base} is outside the trained target centers {low} .. {high} (base frame)")
        if args.policy == "gp_ucb":
            poses = json.loads((POLICIES / "start_poses.json").read_text())["poses"]
            ee_xyz = np.array([p["ee_xyz"] for p in poses])
            order, mean = rank_poses(gp, FeatureMap(ee_xyz, meta["target_low"], meta["target_high"]), ee_xyz,
                                     target_base + SIM_BASE[:2])
            gp_pick = int(order[0])
            print(f"GP-UCB start pose {gp_pick} (predicted value {mean[gp_pick]:.3f}): qpos {np.round(poses[gp_pick]['qpos'], 4).tolist()}")
            gp_pick = np.asarray(poses[gp_pick]["qpos"])

    robot = FrankaInterface(os.path.join(config_root, args.interface_cfg), use_visualizer=False)
    wait_for_state(robot)
    print(f"policy {args.policy}: {policy.observation_dim} inputs; reading for {args.seconds:.0f} s at 20 Hz; no commands are sent")

    log = {k: [] for k in ("t", "q", "dq", "O_T_EE", "obs", "action")}
    worst_side, along_seen = 0.0, []
    start = next_tick = time.time()
    step = 0
    while time.time() - start < args.seconds:
        q, dq, O_T_EE = robot.last_q.copy(), robot.last_dq.copy(), robot.last_eef_pose.copy()
        obs = build_observation(q, dq, target_base if policy.observation_dim == 26 else None)
        action = policy.deterministic_action(obs).numpy().ravel()
        z = (obs.astype(np.float64) - policy.obs_mean) / np.sqrt(policy.obs_var + 1.0e-8)
        along, side = tool_offset(q, O_T_EE)
        along_seen.append(along)
        worst_side = max(worst_side, side)
        for k, v in zip(log, (time.time() - start, q, dq, O_T_EE, obs, action)):
            log[k].append(v)
        if step % 10 == 0:
            grip = grip_site_base(q)
            flagged = [i for i in np.flatnonzero(np.abs(z) > OUT_OF_RANGE)]
            print(f"t {time.time() - start:5.1f}s | grip (base) {np.round(grip, 4)} | action {np.round(action, 3)} "
                  f"-> goal {np.round(grip + ACTION_SCALE * action, 4)} | |dq|max {np.abs(dq).max():.3f} "
                  f"| robot EE offset {along * 100:.2f} cm, side err {side * 1000:.2f} mm"
                  + (f" | OUT OF TRAINING RANGE inputs {flagged}" if flagged else ""))
            if gp_pick is not None and step == 0:
                print(f"  joint distance to the GP start pose: {np.round(gp_pick - q, 3).tolist()}")
        step += 1
        next_tick += 0.05
        time.sleep(max(0.0, next_tick - time.time()))

    print(f"robot EE offset along flange z: {np.mean(along_seen) * 100:.2f} cm (Franka hand default 10.34 cm); "
          f"largest sideways error {worst_side * 1000:.2f} mm (should be under 1 mm)")
    if args.out:
        np.savez(args.out, **{k: np.asarray(v) for k, v in log.items()}, policy=args.policy,
                 target_base=np.full(2, np.nan) if target_base is None else target_base)
        print(f"saved {args.out}")
    robot.close()


if __name__ == "__main__":
    main()
