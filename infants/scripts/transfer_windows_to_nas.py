#!/usr/bin/env python3
"""Copy a folder from the Windows machine to the Synology NAS mount on Ubuntu.

Usage:
    python transfer_windows_to_nas.py WINDOWS_PATH NAS_DEST_NAME

Example:
    python transfer_windows_to_nas.py \
        'D:\\Roberto project\\014' \
        2026-06-09_14-02-01

    # Creates: ~/synology-tuli/2026-06-09_14-02-01/014/...

Requires:
    - NAS mounted at ~/synology-tuli
    - SSH access to Windows: ssh "ut austin"@192.168.253.101
    - paramiko (pip install paramiko)
"""
import argparse
import getpass
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import paramiko


DEFAULT_WINDOWS_HOST = "192.168.253.101"
DEFAULT_WINDOWS_USER = "ut austin"
DEFAULT_NAS_ROOT = Path.home() / "synology-tuli"


def normalize_windows_path(path: str) -> str:
    path = path.replace("\\", "/")
    if re.match(r"^[A-Za-z]:", path):
        return path
    if re.match(r"^/[A-Za-z]:", path):
        return path[1:]
    return path


def nas_is_mounted(nas_root: Path) -> bool:
    result = subprocess.run(
        ["mountpoint", "-q", str(nas_root)],
        check=False,
    )
    return result.returncode == 0


def download_dir(sftp, remote_dir: str, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(remote_dir):
        remote_path = f"{remote_dir.rstrip('/')}/{entry.filename}"
        local_path = local_dir / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            download_dir(sftp, remote_path, local_path)
        else:
            print(f"  {entry.filename}")
            sftp.get(remote_path, str(local_path))


def transfer(windows_folder: str, nas_dest_name: str, host: str, user: str, nas_root: Path):
    win_src = normalize_windows_path(windows_folder)
    nas_dest = nas_root / nas_dest_name

    if not nas_is_mounted(nas_root):
        print(f"NAS is not mounted at {nas_root}", file=sys.stderr)
        print(f"Run: sudo mount -t nfs 192.168.253.1:/volume1/tuli {nas_root}", file=sys.stderr)
        sys.exit(1)

    nas_dest.mkdir(parents=True, exist_ok=True)

    folder_name = Path(win_src).name
    local_target = nas_dest / folder_name

    password = os.environ.get("WINDOWS_PASSWORD")
    if not password:
        password = getpass.getpass(f"{user}@{host}'s password: ")

    print(f"From:  {user}@{host}:{win_src}/")
    print(f"To:    {local_target}/\n")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=30)

    try:
        sftp = client.open_sftp()
        try:
            download_dir(sftp, win_src, local_target)
        finally:
            sftp.close()
    finally:
        client.close()

    print(f"\nDone. Files are at: {local_target}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy a folder from Windows to the Synology NAS mount."
    )
    parser.add_argument(
        "windows_folder",
        help=r"Path on Windows, e.g. D:\Roberto project\014",
    )
    parser.add_argument(
        "nas_dest_name",
        help=f"Folder name to create under NAS root (default: {DEFAULT_NAS_ROOT})",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("WINDOWS_HOST", DEFAULT_WINDOWS_HOST),
        help=f"Windows host (default: {DEFAULT_WINDOWS_HOST})",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("WINDOWS_USER", DEFAULT_WINDOWS_USER),
        help=f"Windows SSH username (default: {DEFAULT_WINDOWS_USER!r})",
    )
    parser.add_argument(
        "--nas-root",
        type=Path,
        default=Path(os.environ.get("NAS_ROOT", DEFAULT_NAS_ROOT)),
        help=f"Local NAS mount point (default: {DEFAULT_NAS_ROOT})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    transfer(
        args.windows_folder,
        args.nas_dest_name,
        args.host,
        args.user,
        args.nas_root.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
