# Deploying the paper policies on a Franka

Self-contained: `pip install -r deploy/requirements.txt` (torch, numpy, scipy), then import from the repo root.
Modules: `policy.py` (network, input order), `bounded_policy.py` (tanh-squashed actions), `frozen_policy.py`
(load a checkpoint, normalize raw inputs), `gp.py` (GP-UCB start-pose ranking). No simulator packages.
These files were checked to reproduce the paper policies' actions and GP pose choices exactly.

## Policies (`policies/`, seed 101; sources and hashes in `manifest.json`)

| File | Method | Paper panel (20 targets) |
|---|---|---|
| `banging_ckpt17.cleanrl_model` | banging; low level of GP-UCB | — |
| `scratch_seed101_ckpt79.cleanrl_model` | scratch PPO | 20/20 |
| `finetune_seed101_ckpt43.cleanrl_model` | fine-tuned PPO | 20/20 |
| `gp_ucb_seed101_ckpt30.npz` | GP-UCB start-pose model | 19/20 |
| `start_poses.json` | the 999 start poses: joint angles `qpos` and simulated end-effector XYZ `ee_xyz` (GP inputs) | — |

```python
import json
import numpy as np
from deploy.frozen_policy import SavedPolicy
from deploy.gp import BoundedGP, FeatureMap, rank_poses

policy = SavedPolicy.load("deploy/policies/finetune_seed101_ckpt43.cleanrl_model")
action = policy.deterministic_action(observation)  # raw 26-vector in, (1, 3) in [-1, 1] out

poses = json.load(open("deploy/policies/start_poses.json"))["poses"]
qpos, ee_xyz = [p["qpos"] for p in poses], np.array([p["ee_xyz"] for p in poses])
gp, meta = BoundedGP.load("deploy/policies/gp_ucb_seed101_ckpt30.npz")
features = FeatureMap(ee_xyz, meta["target_low"], meta["target_high"])
pose_index = rank_poses(gp, features, ee_xyz, target_xy)[0][0]       # then move to qpos[pose_index]
low_level = SavedPolicy.load("deploy/policies/banging_ckpt17.cleanrl_model")
action = low_level.sample_action(observation[:24])                   # GP-UCB used sampled actions
```

## Interface the policies were trained with

| Item | Value |
|---|---|
| Frames | Sim world = robot base + (−0.56, 0, 0.912) m, no rotation. Observations and targets are in sim world. |
| Observation | `cos q` (7), `sin q` (7), `dq` (7), end-effector XYZ (3), [target XY (2) for scratch / fine-tune] |
| End-effector point | robosuite grip site: 0.2035 m along link-7 z (9.65 cm past the flange) |
| Action | `action × 0.05` m position change in the base frame, every 50 ms (20 Hz); gripper held; at most 500 steps |
| Deterministic / sampled | scratch and fine-tune: `deterministic_action`; GP-UCB low level: `sample_action` |
| Start pose | scratch, fine-tune: `[0, 0.1963, 0, −2.618, 0, 2.9416, 0.7854]`; banging: any bank pose; GP-UCB: top-ranked bank pose |
| Target area (target centers, base frame) | X 0.29–0.53 m, Y −0.17–0.17 m (sim world X −0.27 to −0.03, Y −0.17 to 0.17) |
| Table | top 8.7 cm below the robot base; 0.8 × 0.8 m, centered 0.56 m in front |
| Cube | 8 cm, 0.256 kg, rigidly held; center about 2.4 cm below the grip site |
| Success | first contact point inside the 3 cm target circle with a ≥ 50 N normal-force rise after ≥ 5 airborne steps; the episode ends there |

## Controller (training: robosuite OSC_POSITION)

Goal position = current position + action each step; the orientation goal is reset to the current orientation each
step; kp 150 for position and orientation, kd = 2√kp, inertia-weighted; nullspace pull toward the start pose (kp 10).
Deoxys OSC_POSITION holds a fixed downward orientation instead; OSC_POSE with a zero rotation delta matches the
training behavior. Rotation Kp, interpolation, state smoothing and the nullspace target need to be set to match.
