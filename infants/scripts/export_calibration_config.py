#!/usr/bin/env python3
"""Build calibration_markers.yaml from stereo-calib run outputs.

Finds the latest run_* folder under each calibration pair directory,
reads calibration_results.json, and combines with Qualisys mocap anchors.
Missing inputs are written as null in the output YAML.

Example:
  python infants/scripts/export_calibration_config.py --folder 26_06_29_infant_017
  python infants/scripts/export_calibration_config.py \\
      --folder /home/robotlearning2/infants/data/calibration_data/26_06_29_infant_017
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

DATA_ROOT = Path("/home/robotlearning2/infants/data/calibration_data")

# Pair folder names used across record/prepare scripts.
PAIR_DIRS = {
    "cam_L": ("left_to_qualisys",),
    "cam_R": ("right_to_qualisys",),
    "cam_M": ("left_to_mid", "right_to_left"),
}

DEFAULT_MOCAP_ANCHORS = {
    "cam_L": "31039",
    "cam_R": "31041",
}

CAMERA_META = {
    "cam_L": {
        "frame_id": "cam_L_color_optical_frame",
        "color": [1.0, 0.0, 0.0],
    },
    "cam_M": {
        "frame_id": "cam_M_color_optical_frame",
        "color": [0.0, 1.0, 0.0],
        "parent": "cam_L",
    },
    "cam_R": {
        "frame_id": "cam_R_color_optical_frame",
        "color": [0.0, 0.0, 1.0],
    },
}


def resolve_session_dir(folder: str) -> Path:
    path = Path(folder).expanduser()
    if not path.is_absolute():
        path = DATA_ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Calibration session folder not found: {path}")
    return path


def find_pair_dir(session_dir: Path, pair_names: tuple[str, ...]) -> Path | None:
    for name in pair_names:
        candidate = session_dir / name
        if candidate.is_dir():
            return candidate
    return None


def find_latest_run_dir(pair_dir: Path) -> Path | None:
    runs = [
        run_dir
        for run_dir in pair_dir.glob("run_*")
        if (run_dir / "calibration_results.json").is_file()
    ]
    if not runs:
        return None
    return max(runs, key=lambda p: p.name)


def load_stereo_rt(calib_json: Path) -> tuple[list, list] | tuple[None, None]:
    try:
        with open(calib_json) as f:
            data = json.load(f)
        rot = data.get("rot")
        trans = data.get("trans")
        if rot is None or trans is None:
            return None, None
        if trans and isinstance(trans[0], list):
            trans = [row[0] for row in trans]
        return rot, trans
    except (OSError, json.JSONDecodeError, KeyError, TypeError, IndexError):
        return None, None


def _looks_like_mocap_xml(path: Path) -> bool:
    if not path.is_file():
        return False
    head = path.read_text(encoding="utf-8", errors="ignore")[:300]
    return "<calibration" in head or "<?xml" in head


def find_mocap_calibration(session_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        return path if _looks_like_mocap_xml(path) else None

    # Canonical name: {session_folder}_mocap_calibration (with or without extension)
    session_stem = f"{session_dir.name}_mocap_calibration"
    preferred = [
        session_dir / session_stem,
        *sorted(session_dir.glob(f"{session_stem}.*")),
    ]
    for candidate in preferred:
        if _looks_like_mocap_xml(candidate):
            return candidate

    patterns = (
        "*_mocap_calibration",
        "*_mocap_calibration.*",
        "*mocap_calibration*",
        "qualisys_calibration*",
        "*.qca",
        "*.qca.txt",
    )
    for pattern in patterns:
        for candidate in sorted(session_dir.glob(pattern)):
            if _looks_like_mocap_xml(candidate):
                return candidate
    return None


def mocap_transform_from_serial(
    xml_path: Path, serial: str
) -> list[list[float]] | None:
    try:
        root = ET.parse(str(xml_path)).getroot()
    except (OSError, ET.ParseError):
        return None

    for cam in root.iter("camera"):
        if cam.get("serial") != serial:
            continue
        tf = cam.find("transform")
        if tf is None:
            return None

        try:
            return [
                [
                    float(tf.get("r11")),
                    float(tf.get("r21")),
                    float(tf.get("r31")),
                    float(tf.get("x")) / 1000.0,
                ],
                [
                    float(tf.get("r12")),
                    float(tf.get("r22")),
                    float(tf.get("r32")),
                    float(tf.get("y")) / 1000.0,
                ],
                [
                    float(tf.get("r13")),
                    float(tf.get("r23")),
                    float(tf.get("r33")),
                    float(tf.get("z")) / 1000.0,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        except (TypeError, ValueError):
            return None
    return None


def build_config(
    session_dir: Path,
    mocap_path: Path | None,
    mocap_anchors: dict[str, str],
    num_markers: int,
    sphere_radius: float,
) -> tuple[dict, dict[str, Path | None], list[str]]:
    sources: dict[str, Path | None] = {}
    cameras: dict[str, dict] = {}
    warnings: list[str] = []

    for cam_name, pair_names in PAIR_DIRS.items():
        entry = {**CAMERA_META[cam_name], "stereo_R": None, "stereo_T": None}
        if cam_name in mocap_anchors:
            entry["T_mc_wrt_mcR"] = None

        pair_dir = find_pair_dir(session_dir, pair_names)
        if pair_dir is None:
            warnings.append(
                f"{cam_name}: missing pair folder ({' or '.join(pair_names)})"
            )
            sources[cam_name] = None
            cameras[cam_name] = entry
            continue

        run_dir = find_latest_run_dir(pair_dir)
        if run_dir is None:
            warnings.append(
                f"{cam_name}: no run_* with calibration_results.json in {pair_dir.name}/"
            )
            sources[cam_name] = None
            cameras[cam_name] = entry
            continue

        stereo_r, stereo_t = load_stereo_rt(run_dir / "calibration_results.json")
        sources[cam_name] = run_dir
        if stereo_r is None or stereo_t is None:
            warnings.append(
                f"{cam_name}: could not read rot/trans from {run_dir.name}/calibration_results.json"
            )
        else:
            entry["stereo_R"] = stereo_r
            entry["stereo_T"] = stereo_t

        if cam_name in mocap_anchors:
            if mocap_path is None:
                warnings.append(f"{cam_name}: missing mocap calibration file")
            else:
                transform = mocap_transform_from_serial(
                    mocap_path, mocap_anchors[cam_name]
                )
                if transform is None:
                    warnings.append(
                        f"{cam_name}: mocap anchor {mocap_anchors[cam_name]} "
                        f"not found in {mocap_path.name}"
                    )
                else:
                    entry["T_mc_wrt_mcR"] = transform

        cameras[cam_name] = entry

    config = {
        "num_markers": num_markers,
        "sphere_radius": sphere_radius,
        "cameras": cameras,
    }
    return config, sources, warnings


def write_config(
    output_path: Path,
    session_dir: Path,
    config: dict,
    sources: dict[str, Path | None],
    mocap_path: Path | None,
    warnings: list[str],
) -> None:
    mocap_line = mocap_path.name if mocap_path else "(missing)"
    header = (
        "# Calibration chain for trial visualization (markers + multi-camera alignment).\n"
        f"# Auto-generated by export_calibration_config.py\n"
        f"# Session: {session_dir.name}\n"
        f"# Mocap calibration: {mocap_line}\n"
        "# Sources:\n"
    )
    for cam_name in PAIR_DIRS:
        run_dir = sources.get(cam_name)
        if run_dir is None:
            header += f"#   {cam_name}: (missing)\n"
        else:
            header += f"#   {cam_name}: {run_dir.relative_to(session_dir)}\n"
    if warnings:
        header += "# Missing / incomplete:\n"
        for warning in warnings:
            header += f"#   - {warning}\n"
    header += (
        "#\n"
        "# Usage:\n"
        f"#   python infants/scripts/run_trial_viz.py --markers "
        f"--calib-config {output_path}\n\n"
    )

    with open(output_path, "w") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export calibration_markers.yaml from latest stereo-calib runs."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Session folder name (e.g. 26_06_29_infant_017) or absolute path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output YAML path (default: <session>/calibration_markers.yaml)",
    )
    parser.add_argument(
        "--mocap-calibration",
        type=Path,
        help="Qualisys mocap calibration XML (.txt/.qca). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--mocap-anchor-l",
        default=DEFAULT_MOCAP_ANCHORS["cam_L"],
        help=f"Qualisys serial for cam_L anchor (default: {DEFAULT_MOCAP_ANCHORS['cam_L']})",
    )
    parser.add_argument(
        "--mocap-anchor-r",
        default=DEFAULT_MOCAP_ANCHORS["cam_R"],
        help=f"Qualisys serial for cam_R anchor (default: {DEFAULT_MOCAP_ANCHORS['cam_R']})",
    )
    parser.add_argument("--num-markers", type=int, default=1)
    parser.add_argument("--sphere-radius", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = resolve_session_dir(args.folder)
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else session_dir / "calibration_markers.yaml"
    )
    mocap_path = find_mocap_calibration(session_dir, args.mocap_calibration)
    mocap_anchors = {
        "cam_L": args.mocap_anchor_l,
        "cam_R": args.mocap_anchor_r,
    }

    config, sources, warnings = build_config(
        session_dir,
        mocap_path,
        mocap_anchors,
        args.num_markers,
        args.sphere_radius,
    )
    write_config(output_path, session_dir, config, sources, mocap_path, warnings)

    print(f"[OK] Wrote {output_path}")
    for cam_name in PAIR_DIRS:
        run_dir = sources.get(cam_name)
        if run_dir is None:
            print(f"  {cam_name}: (missing)")
        else:
            print(f"  {cam_name}: {run_dir}")
    for warning in warnings:
        print(f"[WARN] {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
