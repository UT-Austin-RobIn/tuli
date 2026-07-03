import argparse
import os
import sys
import numpy as np


def percentiles(arr, ps=(50, 75, 90, 95, 99, 99.5, 100)):
    return {p: float(np.percentile(arr, p)) for p in ps}


def run(data_path, out_dir=None, zero_threshold=1e-3):
    d = np.load(data_path, allow_pickle=True)
    if "total_normal_force" not in d.files:
        raise KeyError(f"{data_path} has no 'total_normal_force' key; "
                       f"available: {list(d.files)}")
    fn = np.asarray(d["total_normal_force"], dtype=np.float64)
    n_contacts = np.asarray(d.get("n_contacts", np.zeros_like(fn)), dtype=np.int64)
    print(f"[input] {data_path}: {fn.size:,} steps  "
          f"Fn max={fn.max():.1f}  Fn p95={np.percentile(fn, 95):.1f}  "
          f"contact-steps={(fn > 0).sum():,}")

    delta = np.abs(np.diff(fn, prepend=fn[0]))
    pos = delta[delta > zero_threshold]

    print("\n" + "=" * 70)
    print(" per-step |ΔFn| distribution  (nonzero = above "
          f"{zero_threshold:g} N)")
    print("=" * 70)
    print(f"  total steps: {delta.size:,}")
    print(f"  nonzero:     {pos.size:,}  "
          f"({100.0 * pos.size / delta.size:.1f}% of all steps)")
    if pos.size == 0:
        print("  NO contact transients detected — calibration failed.")
        return None

    ps = percentiles(pos, (50, 75, 90, 95, 99, 99.5, 100))
    for p, v in ps.items():
        print(f"  p{p:<5g}: {v:>8.2f} N")
    print(f"  mean : {pos.mean():.2f} N")
    print(f"  std  : {pos.std():.2f} N")

    print("\n  over ALL steps (including zero):")
    all_ps = percentiles(delta, (50, 75, 90, 95, 99, 99.5, 100))
    for p, v in all_ps.items():
        print(f"  p{p:<5g}: {v:>8.2f} N")

    print("\n" + "=" * 70)
    print(" RECOMMENDATION")
    print("=" * 70)
    p95 = ps[95]
    p90 = ps[90]
    p99 = ps[99]
    print(f"  reward_divisor ≈ {p95:.1f}   "
          f"(p95 of nonzero |ΔFn|)")
    print(f"  alternatives: p90={p90:.1f} (sharper gradient, more saturation), "
          f"p99={p99:.1f} (flatter gradient, less saturation)")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "delta_fn_samples.npy"), delta)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(9, 6))
            axes[0].plot(fn, linewidth=0.5)
            axes[0].set_title("Fn (raw normal force)")
            axes[0].set_ylabel("N")
            axes[1].hist(pos, bins=80, log=True)
            axes[1].axvline(p95, color="r", linestyle="--", label=f"p95 = {p95:.1f}")
            axes[1].axvline(p90, color="orange", linestyle=":", label=f"p90 = {p90:.1f}")
            axes[1].axvline(p99, color="purple", linestyle=":", label=f"p99 = {p99:.1f}")
            axes[1].set_xlabel("|ΔFn| per step (N)")
            axes[1].set_ylabel("count (log)")
            axes[1].set_title("|ΔFn| distribution (nonzero steps)")
            axes[1].legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "delta_fn_calibration.png"), dpi=150)
            plt.close(fig)
            print(f"  plot → {out_dir}/delta_fn_calibration.png")
        except ImportError:
            pass
    return p95


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", nargs="?", default="data.npz",
                        help="path to .npz with total_normal_force")
    parser.add_argument("--out", default="runs/reward_scale_calibration",
                        help="dir for saved samples + plot (pass '' to skip)")
    parser.add_argument("--zero-threshold", type=float, default=1e-3,
                        help="|ΔFn| below this is treated as 'no transient'")
    args = parser.parse_args()
    run(args.data_path, out_dir=(args.out or None),
        zero_threshold=args.zero_threshold)
