#!/usr/bin/env python3
"""Copy Qualisys trial files from Windows into each Linux trial_* folder.

For every trial_00N under a session, fetches matching C3D / TSV / Miqus AVI
(and Miqus AVIs) from:

  windows:D:/Roberto_project/{infant}/

into:

  data/{session}/trial_00N/

Never touches trial_ros.bag. Missing Windows files print a red warning and
the rest of the session still transfers.

Naming matches existing sessions, e.g. 2026-07-17_10-57-17 / trial_001:

  26_07_17_020_1.c3d
  26_07_17_020_1.tsv
  26_07_17_020_1_Miqus_1_31039.avi
  26_07_17_020_1_Miqus_10_31041.avi
  26_07_17_020_1_Miqus_3_31043.avi

Also accepts *_infant_015_2.* and unnumbered trial-1 stems (*_015.c3d).

Usage:
  python infants/scripts/organize_trial_session.py \\
      --session 2026-07-17_10-57-17 --infant 020
  python infants/scripts/organize_trial_session.py \\
      --session data/2026-07-17_10-57-17 --infant 020 --dry-run
"""
from __future__ import annotations

import argparse
import atexit
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path("/home/robotlearning2/infants/data")
DEFAULT_WINDOWS_HOST = "windows"
WINDOWS_PROJECT_ROOT = "D:/Roberto_project"

SKIP_REMOTE_NAMES = {"calibration", "calibration_data"}
COPY_SUFFIXES = {".c3d", ".tsv", ".avi"}
MIQUS_CAMERAS = ("Miqus_1_31039", "Miqus_10_31041", "Miqus_3_31043")

# 26_07_17_020_1.c3d  |  26_06_24_infant_015_2.tsv  |  26_06_24_infant_015.c3d
FILE_RE = re.compile(
    r"^(?P<date>\d{2}_\d{2}_\d{2})"
    r"(?:_infant)?_(?P<id>\d+)"
    r"(?:_(?P<trial>\d+))?"
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)
CALIB_NAME_RE = re.compile(
    r"left_to_qualisys|right_to_qualisys|left_to_mid|mocap_calibration",
    re.IGNORECASE,
)


def red(msg: str) -> None:
    print(f"\033[31m[WARN] {msg}\033[0m", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--session",
        required=True,
        help="Linux session folder (name under data/ or absolute path), "
        "e.g. 2026-07-17_10-57-17",
    )
    p.add_argument(
        "--infant",
        required=True,
        help="Infant id on Windows, e.g. 020 → "
        f"{DEFAULT_WINDOWS_HOST}:{WINDOWS_PROJECT_ROOT}/020/",
    )
    p.add_argument(
        "--windows-host",
        default=DEFAULT_WINDOWS_HOST,
        help=f"SSH/scp host alias (default: {DEFAULT_WINDOWS_HOST})",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files that already exist in the trial folder",
    )
    p.add_argument("--dry-run", action="store_true", help="Print only; no scp")
    return p.parse_args()


def resolve_session_dir(session: str) -> Path:
    path = Path(session).expanduser()
    if not path.is_absolute():
        path = DATA_ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Session folder not found: {path}")
    trials = sorted(p for p in path.glob("trial_*") if p.is_dir())
    if not trials:
        raise FileNotFoundError(f"No trial_* folders under {path}")
    return path


def infant_ids_to_try(infant: str) -> list[str]:
    raw = infant.strip().strip("/")
    ids = [raw]
    if raw.isdigit():
        padded = f"{int(raw):03d}"
        if padded not in ids:
            ids.append(padded)
        stripped = str(int(raw))
        if stripped not in ids:
            ids.append(stripped)
    return ids


def ssh_control_opts(control_path: str | None) -> list[str]:
    if not control_path:
        return []
    return [
        "-o", f"ControlPath={control_path}",
        "-o", "ControlMaster=no",
    ]


def start_ssh_master(host: str) -> str:
    """Ask for the Windows password once; later ssh/scp reuse this socket."""
    sock_dir = Path(tempfile.mkdtemp(prefix="ssh-mux-"))
    control_path = str(sock_dir / "ctl")
    print(f"[INFO] Connecting to {host} (enter password once)...")
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "ControlMaster=yes",
            "-o",
            f"ControlPath={control_path}",
            "-o",
            "ControlPersist=yes",
            host,
            "exit",
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Could not open SSH connection to {host}")

    def stop() -> None:
        subprocess.run(
            ["ssh", "-o", f"ControlPath={control_path}", "-O", "exit", host],
            capture_output=True,
        )
        try:
            Path(control_path).unlink(missing_ok=True)
            sock_dir.rmdir()
        except OSError:
            pass

    atexit.register(stop)
    return control_path


def list_windows_files(
    host: str, remote_dir: str, control_path: str | None = None
) -> list[str] | None:
    win_path = remote_dir.replace("/", "\\")
    mux = ssh_control_opts(control_path)
    commands = (
        ["ssh", *mux, host, f'cmd /c dir /b "{win_path}"'],
        ["ssh", *mux, host, f"ls -1 '{remote_dir}'"],
    )
    for cmd in commands:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            continue
        names = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if names:
            return names
    return None


def date_prefix_from_session(session_dir: Path) -> str | None:
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", session_dir.name)
    if not m:
        return None
    return f"{m.group(1)[2:]}_{m.group(2)}_{m.group(3)}"


def guessed_filenames(date_prefix: str, infant_int: int, trial_n: int) -> list[str]:
    """Candidate Windows names when remote listing is unavailable."""
    ids = [f"{infant_int:03d}"]
    stems = []
    for iid in ids:
        stems.append(f"{date_prefix}_{iid}_{trial_n}")
        stems.append(f"{date_prefix}_infant_{iid}_{trial_n}")
        if trial_n == 1:
            stems.append(f"{date_prefix}_{iid}")
            stems.append(f"{date_prefix}_infant_{iid}")
    names: list[str] = []
    seen: set[str] = set()
    for stem in stems:
        for suffix in (".c3d", ".tsv"):
            names.append(f"{stem}{suffix}")
        for cam in MIQUS_CAMERAS:
            names.append(f"{stem}_{cam}.avi")
    out = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def trial_number_from_dir(trial_dir: Path) -> int | None:
    m = re.search(r"(\d+)$", trial_dir.name)
    return int(m.group(1)) if m else None


def parse_remote_name(name: str, infant_int: int) -> tuple[int, str] | None:
    """Return (trial_number, filename) if this is a trial mocap file for infant."""
    if name.lower() in SKIP_REMOTE_NAMES or CALIB_NAME_RE.search(name):
        return None
    suffix = Path(name).suffix.lower()
    if suffix not in COPY_SUFFIXES:
        return None
    m = FILE_RE.match(name)
    if not m:
        return None
    try:
        file_id = int(m.group("id"))
    except ValueError:
        return None
    if file_id != infant_int:
        return None
    trial_s = m.group("trial")
    trial_n = int(trial_s) if trial_s else 1
    return trial_n, name


def scp_file(
    host: str,
    remote_dir: str,
    filename: str,
    dest_dir: Path,
    dry_run: bool,
    control_path: str | None = None,
) -> bool:
    src = f"{host}:{remote_dir}/{filename}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  scp  {src}  ->  {dest_dir}/")
    if dry_run:
        return True
    proc = subprocess.run(
        ["scp", *ssh_control_opts(control_path), src, str(dest_dir)],
        text=True,
    )
    if proc.returncode != 0:
        red(f"scp failed for {filename} (exit {proc.returncode})")
        return False
    return True


def expected_roles(names: list[str]) -> dict[str, bool]:
    found = {
        "c3d": False,
        "tsv": False,
    }
    for cam in MIQUS_CAMERAS:
        found[cam] = False
    for name in names:
        lower = name.lower()
        if lower.endswith(".c3d"):
            found["c3d"] = True
        elif lower.endswith(".tsv"):
            found["tsv"] = True
        for cam in MIQUS_CAMERAS:
            if cam.lower() in lower and lower.endswith(".avi"):
                found[cam] = True
    return found


def main() -> int:
    args = parse_args()
    session_dir = resolve_session_dir(args.session)
    infant_raw = args.infant.strip().strip("/")
    infant_int = int(infant_raw) if infant_raw.isdigit() else None
    if infant_int is None:
        print(f"[ERROR] --infant must be numeric (got {infant_raw!r})", file=sys.stderr)
        return 1

    trials = sorted(p for p in session_dir.glob("trial_*") if p.is_dir())
    trial_by_n = {}
    for trial_dir in trials:
        n = trial_number_from_dir(trial_dir)
        if n is None:
            red(f"Could not parse trial number from {trial_dir.name}; skipping")
            continue
        trial_by_n[n] = trial_dir

    print(f"[INFO] Session: {session_dir}")
    print(f"[INFO] Trials:  {', '.join(d.name for d in trials)}")
    if args.dry_run:
        print("[INFO] Dry run — no files will be copied.")

    control_path = None
    if not args.dry_run:
        try:
            control_path = start_ssh_master(args.windows_host)
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

    remote_dir = None
    remote_names: list[str] = []
    for infant_id in infant_ids_to_try(infant_raw):
        candidate = f"{WINDOWS_PROJECT_ROOT}/{infant_id}"
        print(f"[INFO] Listing {args.windows_host}:{candidate}/")
        names = list_windows_files(args.windows_host, candidate, control_path)
        if names is None:
            red(f"Could not list {args.windows_host}:{candidate}")
            continue
        remote_dir = candidate
        remote_names = names
        break

    guess = False
    if remote_dir is None:
        remote_dir = f"{WINDOWS_PROJECT_ROOT}/{infant_ids_to_try(infant_raw)[0]}"
        date_prefix = date_prefix_from_session(session_dir)
        if date_prefix is None:
            print(
                f"[ERROR] Could not list Windows files and cannot guess names "
                f"from session folder {session_dir.name}",
                file=sys.stderr,
            )
            return 1
        red(
            f"Falling back to guessed filenames from session date {date_prefix} "
            f"under {args.windows_host}:{remote_dir}/"
        )
        guess = True
        by_trial = {
            n: guessed_filenames(date_prefix, infant_int, n) for n in trial_by_n
        }
        ignored = 0
    else:
        by_trial = defaultdict(list)
        ignored = 0
        for name in remote_names:
            parsed = parse_remote_name(name, infant_int)
            if parsed is None:
                ignored += 1
                continue
            trial_n, fname = parsed
            by_trial[trial_n].append(fname)

    copied = 0
    skipped = 0
    failed = 0

    for n in sorted(trial_by_n):
        trial_dir = trial_by_n[n]
        files = by_trial.get(n, [])
        print(f"\n===== {trial_dir.name} =====")
        if not files:
            red(
                f"No C3D/TSV/AVI for trial {n} on Windows "
                f"({remote_dir}/)"
            )
            continue

        roles = expected_roles(files)
        if not roles["c3d"]:
            red(f"{trial_dir.name}: no .c3d on Windows")
        if not roles["tsv"]:
            red(f"{trial_dir.name}: no .tsv on Windows")
        for cam in MIQUS_CAMERAS:
            if not roles[cam]:
                red(f"{trial_dir.name}: no {cam}.avi on Windows")

        for filename in sorted(files):
            dest = trial_dir / filename
            if dest.exists() and not args.force:
                print(f"  SKIP  {filename} (already in {trial_dir.name})")
                skipped += 1
                continue
            ok = scp_file(
                args.windows_host,
                remote_dir,
                filename,
                trial_dir,
                args.dry_run,
                control_path,
            )
            if ok:
                copied += 1
            else:
                failed += 1

    extra = sorted(n for n in by_trial if n not in trial_by_n)
    if extra and not guess:
        print()
        red(
            "Windows has trial files with no matching Linux trial_* folder: "
            + ", ".join(f"{n} ({len(by_trial[n])} files)" for n in extra)
        )

    print()
    print(
        f"[OK] Done. copied={copied} skipped={skipped} failed={failed} "
        f"(ignored {ignored} non-trial names on Windows)"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
