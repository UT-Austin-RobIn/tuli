#!/usr/bin/env python3
"""Reorganize Qualisys calibration files into the layout expected by prepare scripts.

Linux RealSense bags and Qualisys exports live under:

  ~/infants/data/calibration_data/{session}/

With --infant, fetches from Windows and organizes in one step:

  windows:D:/Roberto_project/{infant}/calibration
    -> {session}/left_to_qualisys/ ...
    -> {session}/right_to_qualisys/ ...
    -> {session}/{session}_mocap_calibration.txt

Without --infant, organizes flat Qualisys files already sitting under
{session}/ (never overwrites ros.bag).

Target layout:

  {session}/
    {session}_mocap_calibration.txt
    left_to_qualisys/{session}_left_to_qualisys_Miqus_1_31039.avi
    left_to_qualisys/{session}_left_to_qualisys.tsv
    right_to_qualisys/...
    left_to_mid/          # ros.bag only from Linux; never touched here
    _unused/

Usage:
  python infants/scripts/organize_calibration_session.py --folder 26_07_30_infant_002
  python infants/scripts/organize_calibration_session.py --folder 26_06_25_infant_015 --infant 015
  python infants/scripts/organize_calibration_session.py --folder 26_07_30_infant_002 --dry-run
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

DATA_ROOT = Path("/home/robotlearning2/infants/data/calibration_data")
DEFAULT_WINDOWS_HOST = "windows"
WINDOWS_PROJECT_ROOT = "D:/Roberto_project"
WINDOWS_CALIB_DIR = "calibration"

PAIR_MIQUS = {
    "left_to_qualisys": "Miqus_1_31039",
    "right_to_qualisys": "Miqus_10_31041",
}

AVI_RE = re.compile(
    r"^(?P<pair>left_to_qualisys|right_to_qualisys)"
    r"(?P<trail>\d*)_"
    r"(?P<miqus>Miqus_\d+_\d+)\.avi$",
    re.IGNORECASE,
)
TSV_RE = re.compile(
    r"^(?P<pair>left_to_qualisys|right_to_qualisys)(?P<trail>\d*)\.tsv$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--folder",
        required=True,
        help="Session folder name, e.g. 26_07_30_infant_002",
    )
    p.add_argument(
        "--infant",
        help="Infant id on Windows (e.g. 015). scp from "
        f"{DEFAULT_WINDOWS_HOST}:{WINDOWS_PROJECT_ROOT}/{{infant}}/calibration "
        "into the session folder, then organize into left/right_to_qualisys/.",
    )
    p.add_argument(
        "--windows-host",
        default=DEFAULT_WINDOWS_HOST,
        help=f"SSH/scp host alias for Windows (default: {DEFAULT_WINDOWS_HOST})",
    )
    p.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Do not scp from Windows even if --infant is set",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="calibration_data root (default: %(default)s)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print only; no changes")
    return p.parse_args()


def normalize_miqus(raw: str) -> str:
    m = re.search(r"Miqus_(\d+)_(\d+)", raw, re.I)
    if not m:
        return raw
    return f"Miqus_{m.group(1)}_{m.group(2)}"


def move_path(src: Path, dst: Path, dry_run: bool) -> None:
    print(f"  MOVE  {src}  ->  {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    shutil.move(str(src), str(dst))


def copy_path(src: Path, dst: Path, dry_run: bool) -> None:
    print(f"  COPY  {src}  ->  {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def iter_files(*dirs: Path):
    """Yield files under each directory tree (recursive)."""
    seen: set[Path] = set()
    for d in dirs:
        if not d.is_dir():
            continue
        for path in sorted(d.rglob("*")):
            if not path.is_file():
                continue
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            yield path


def loose_tsv_matches_pair(name: str, pair: str) -> bool:
    lower = name.lower()
    return lower.endswith(".tsv") and pair in lower


def loose_avi_matches_pair(name: str, pair: str, needed_miqus: str) -> bool:
    lower = name.lower()
    return (
        lower.endswith(".avi")
        and pair in lower
        and needed_miqus.lower() in lower
    )


def flatten_windows_calibration_folder(session_dir: Path, dry_run: bool) -> None:
    """scp -r copies the remote folder itself; lift files up and drop the extra dir."""
    nested = session_dir / WINDOWS_CALIB_DIR
    if not nested.is_dir():
        return
    print(f"[INFO] Flattening {nested.name}/ into {session_dir}")
    for item in sorted(nested.iterdir()):
        dest = session_dir / item.name
        if dest.exists():
            print(f"  SKIP  {item.name} (already present)")
            if not dry_run and item.is_file():
                item.unlink()
            continue
        move_path(item, dest, dry_run)
    if dry_run:
        return
    leftover = [p for p in nested.iterdir()]
    if leftover:
        print(f"[WARN] Could not remove {nested}; leftover: {[p.name for p in leftover]}")
        return
    nested.rmdir()
    print(f"[OK] Removed extra directory {nested.name}/")


def fetch_from_windows(
    infant: str,
    session_dir: Path,
    host: str,
    dry_run: bool,
) -> None:
    """scp Qualisys calibration dump from Windows into the Linux session folder."""
    infant = infant.strip()
    if not infant:
        raise ValueError("--infant must be non-empty")

    remote_dir = f"{WINDOWS_PROJECT_ROOT}/{infant}/{WINDOWS_CALIB_DIR}"
    session_dir.mkdir(parents=True, exist_ok=True)
    src = f"{host}:{remote_dir}"
    dst = str(session_dir)
    print(f"[INFO] Fetching {src}  ->  {dst}/")
    if dry_run:
        print("[INFO] Dry run — skipping scp.")
        flatten_windows_calibration_folder(session_dir, dry_run)
        return

    # Same form as: scp -r windows:D:/Roberto_project/050/calibration ./
    proc = subprocess.run(
        ["scp", "-r", src, dst],
        capture_output=False,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"scp failed (exit {proc.returncode}). "
            f"Ensure SSH works: scp -r {src} {dst}"
        )
    flatten_windows_calibration_folder(session_dir, dry_run=False)
    print(f"[OK] Windows files copied into {session_dir}")


def find_qca(*dirs: Path) -> Path | None:
    candidates: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for pattern in ("*.qca.txt", "*.qca", "*mocap_calibration*", "*calibration*.txt"):
            for path in d.rglob(pattern):
                if path.is_file() and "_unused" not in path.parts:
                    candidates.append(path)
    xmlish = []
    for path in sorted(set(candidates)):
        head = path.read_text(encoding="utf-8", errors="ignore")[:400]
        if "<calibration" in head or "<?xml" in head:
            xmlish.append(path)
    if not xmlish:
        return None

    def score(p: Path) -> tuple[int, str]:
        name = p.name.lower()
        if name.endswith(".qca.txt") or name.endswith(".qca"):
            return (0, name)
        if "mocap_calibration" in name:
            return (1, name)
        return (2, name)

    return sorted(xmlish, key=score)[0]


def find_avi_for_pair(
    session_dir: Path, pair: str, needed_miqus: str, *search_dirs: Path
) -> Path | None:
    pair_dir = session_dir / pair
    canonical = pair_dir / f"{session_dir.name}_{pair}_{needed_miqus}.avi"
    if canonical.is_file():
        return canonical
    if pair_dir.is_dir():
        for path in sorted(pair_dir.glob(f"*_{needed_miqus}.avi")):
            return path
    for path in iter_files(*search_dirs):
        m = AVI_RE.match(path.name)
        if m:
            if m.group("pair").lower() != pair:
                continue
            if normalize_miqus(m.group("miqus")) == needed_miqus:
                return path
            continue
        if loose_avi_matches_pair(path.name, pair, needed_miqus):
            return path
    return None


def find_tsv_for_pair(
    session_dir: Path, pair: str, *search_dirs: Path
) -> Path | None:
    pair_dir = session_dir / pair
    canonical = pair_dir / f"{session_dir.name}_{pair}.tsv"
    if canonical.is_file():
        return canonical
    if pair_dir.is_dir():
        matches = sorted(pair_dir.glob(f"*{pair}*.tsv"))
        if matches:
            return matches[0]
    for path in iter_files(*search_dirs):
        m = TSV_RE.match(path.name)
        if m and m.group("pair").lower() == pair:
            return path
        if loose_tsv_matches_pair(path.name, pair):
            return path
    return None


def print_next_commands(session: str) -> None:
    print("[OK] Session is laid out for prepare / stereo-calib / export.")
    print()
    print("Next commands (copy-paste):")
    print()
    print("  cd ~/infants")
    print("  source ~/envs/infants/bin/activate")
    print(
        f"  python infants/scripts/prepare_calibration_images.py "
        f"--folder_name {session} --type left_to_qualisys"
    )
    print(
        f"  python infants/scripts/prepare_calibration_images.py "
        f"--folder_name {session} --type right_to_qualisys"
    )
    print(
        f"  python infants/scripts/prepare_calibration_image_rs.py "
        f"--folder_name {session}"
    )
    print()
    print("  cd ~/stereo-calib")
    print("  source .venv/bin/activate")
    base = f"/home/robotlearning2/infants/data/calibration_data/{session}"
    print(f"  python examples/perform_calibration.py --data-path {base}/left_to_qualisys")
    print(f"  python examples/perform_calibration.py --data-path {base}/right_to_qualisys")
    print(f"  python examples/perform_calibration.py --data-path {base}/left_to_mid")
    print()
    print("  cd ~/infants")
    print("  source ~/envs/infants/bin/activate")
    print(f"  python infants/scripts/export_calibration_config.py --folder {session}")
    print()


def main() -> int:
    args = parse_args()
    session = args.folder.strip().strip("/")
    data_root = args.data_root.resolve()
    session_dir = data_root / session

    if not session_dir.is_dir():
        if args.infant or args.dry_run:
            print(f"[INFO] Creating session dir: {session_dir}")
            if not args.dry_run:
                session_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(
                f"[ERROR] Session folder not found: {session_dir}\n"
                "Record Linux bags first, or pass --infant to fetch from Windows.",
                file=sys.stderr,
            )
            return 1

    print(f"[INFO] Session folder: {session_dir}")
    if args.dry_run:
        print("[INFO] Dry run — no files will be changed.")

    flatten_windows_calibration_folder(session_dir, args.dry_run)

    if args.infant and not args.skip_fetch:
        try:
            fetch_from_windows(args.infant, session_dir, args.windows_host, args.dry_run)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

    search_dirs = (session_dir,)
    unused = session_dir / "_unused"
    errors: list[str] = []
    warnings: list[str] = []
    handled: set[Path] = set()

    def mark(path: Path) -> None:
        handled.add(path.resolve())

    for pair in ("left_to_qualisys", "right_to_qualisys", "left_to_mid"):
        pair_dir = session_dir / pair
        if not args.dry_run:
            pair_dir.mkdir(parents=True, exist_ok=True)
        bag = pair_dir / "ros.bag"
        if bag.is_file():
            mark(bag)
            print(f"  OK    {pair}/ros.bag  (Linux recording — left untouched)")
        else:
            warnings.append(
                f"No Linux ros.bag at {pair}/ros.bag. "
                f"Windows scp never provides this — only record_for_calibration does."
            )

    # Park extra Miqus AVIs
    for path in iter_files(*search_dirs):
        if path.resolve() in handled:
            continue
        m = AVI_RE.match(path.name)
        if not m:
            continue
        pair = m.group("pair").lower()
        miqus = normalize_miqus(m.group("miqus"))
        if pair in PAIR_MIQUS and miqus != PAIR_MIQUS[pair]:
            move_path(path, unused / path.name, args.dry_run)
            mark(path)
            warnings.append(f"Extra AVI for {pair} -> _unused/: {path.name}")

    for pair, needed_miqus in PAIR_MIQUS.items():
        pair_dir = session_dir / pair
        avi = find_avi_for_pair(session_dir, pair, needed_miqus, *search_dirs)
        if avi is None:
            errors.append(
                f"No AVI for {pair} with camera {needed_miqus} "
                f"(expected under {session_dir})"
            )
        else:
            dst = pair_dir / f"{session}_{pair}_{needed_miqus}.avi"
            if avi.resolve() != dst.resolve():
                move_path(avi, dst, args.dry_run)
            else:
                print(f"  OK    {pair}/{dst.name}")
            mark(avi)
            mark(dst)

        tsv = find_tsv_for_pair(session_dir, pair, *search_dirs)
        if tsv is None:
            errors.append(
                f"No TSV for {pair} (expected under {session_dir})"
            )
        else:
            dst = pair_dir / f"{session}_{pair}.tsv"
            if tsv.resolve() != dst.resolve():
                move_path(tsv, dst, args.dry_run)
            else:
                print(f"  OK    {pair}/{dst.name}")
            mark(tsv)
            mark(dst)

    mocap_dst = session_dir / f"{session}_mocap_calibration.txt"
    qca = find_qca(*search_dirs)
    if qca is None:
        errors.append(
            f"No Qualisys mocap calibration (.qca / .qca.txt) under {session_dir}/"
        )
    elif qca.resolve() == mocap_dst.resolve():
        print(f"  OK    {mocap_dst.name}")
        mark(mocap_dst)
    else:
        if not args.dry_run:
            unused.mkdir(parents=True, exist_ok=True)
        copy_path(qca, mocap_dst, args.dry_run)
        mark(mocap_dst)
        # If a duplicate sat in the session root, park it after copying canonical name.
        if qca.parent.resolve() == session_dir.resolve() and qca.name != mocap_dst.name:
            move_path(qca, unused / qca.name, args.dry_run)
            mark(qca)
        else:
            mark(qca)

    # Park leftover dumps from search dirs (never touch ros.bag)
    for path in iter_files(*search_dirs):
        if path.resolve() in handled:
            continue
        if path.name == mocap_dst.name or path.name == "ros.bag":
            continue
        lower = path.name.lower()
        if lower.endswith((".qtm", ".avi", ".tsv")) or lower.endswith(".qca") or lower.endswith(
            ".qca.txt"
        ):
            move_path(path, unused / path.name, args.dry_run)
            mark(path)
            warnings.append(f"Parked leftover -> _unused/: {path.name}")

    print()
    for w in warnings:
        print(f"[WARN] {w}")
    if errors:
        print()
        for e in errors:
            print(f"[ERROR] {e}")
        print()
        print("[FAIL] Fix the Qualisys file errors above, then re-run this script.")
        if args.infant:
            print(
                "Re-fetch from Windows, then re-run:\n"
                f"  python infants/scripts/organize_calibration_session.py "
                f"--folder {session} --infant {args.infant}"
            )
        else:
            print(
                "Place Qualisys files under the session folder, or fetch automatically:\n"
                f"  python infants/scripts/organize_calibration_session.py "
                f"--folder {session} --infant XXX"
            )
        return 1

    print()
    if any("ros.bag" in w for w in warnings):
        print(
            "[OK] Qualisys files are organized. "
            "Missing Linux ros.bag(s) must be restored/re-recorded before prepare."
        )
        print()
        return 0

    print_next_commands(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
