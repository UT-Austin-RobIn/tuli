# tuli

Training Banging -> hammering policy under hard / soft contact materials.

## Setup

```bash
conda env create -f environment.yml    # once
conda activate tuli
cd ~/tuli
```

W&B logging is on by default (`logging: track` in each config; project `tuli`).

## Train

Phase 1 — banging (one checkpoint per material):

```bash
python tuli/scripts/ppo_banging.py configs/banging_hard.yaml
python tuli/scripts/ppo_banging.py configs/banging_soft.yaml
```

Phase 2 — hammering, scratch vs warm-start:

```bash
python tuli/scripts/ppo_hammering.py configs/hammering_target.yaml
python tuli/scripts/ppo_hammering.py configs/hammering_repeat.yaml

# warm-start from a phase-1 banging checkpoint
python tuli/scripts/ppo_hammering.py configs/hammering_target.yaml \
    --pretrained runs/<banging_run>/model/<name>_final.cleanrl_model
```

Checkpoints: `runs/<name>_<timestamp>/model/<name>_{<iter>,final}.cleanrl_model`.

## Eval

```bash
python tuli/scripts/ppo_banging_eval.py runs/<run>/model/<name>_final.cleanrl_model \
    --material hard --episodes 5 --save-video     # or --render for live viewer
```

Match `--material` to what the checkpoint was trained on.

## Recalibrate reward scale

Recalibrate if the physics of simulation (material of cube / table) change. 
Divisor = p95 of nonzero |ΔFn| on a scripted hard bang trace:

```bash
python tuli/scripts/record_banging_trace.py --material hard --mode bang --out traces/hard_bang.npz
python tuli/scripts/calibrate_reward_scale.py traces/hard_bang.npz
```

Update `reward_divisor` in every config at once — it is **shared** across
materials.

## Recalibrating minimum force (`min_delta_fn`)

`--mode press` records the force during press. Namely, when the arm descend, hold
contact, and sinusoidally modulate pressure.

```bash
python tuli/scripts/record_banging_trace.py --material hard --mode press --out traces/hard_press.npz
python tuli/scripts/record_banging_trace.py --material soft --mode press --out traces/soft_press.npz
python tuli/scripts/record_banging_trace.py --material soft --mode bang  --out traces/soft_bang.npz
python tuli/scripts/calibrate_reward_scale.py traces/hard_press.npz   # repeat per trace, read p99
```

Minimum force value above p99 was picked for both materials to avoid the policy learning to press. 
Check to make sure that this value is below the soft-material lower tail so the soft policy doesn't get zero rewards.
Currently at 20 N. 

## WIP

1. Testing if both air gate and floor force are needed for banging policy.
2. Experimenting with hammering policy.
3. Creating evals to compare the hammering policy training process. 
