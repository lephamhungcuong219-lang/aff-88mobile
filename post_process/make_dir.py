#!/usr/bin/env python3
"""Create a job directory and its standard subdirectories.

Usage:
    python make_dir.py <dir_path>

Creates (parents included, like mkdir -p):
    <dir_path>/input
    <dir_path>/output
    <dir_path>/temp

Stdout: exactly 'done' or 'fail'.
Stderr: the reason for a failure (human-readable).
Exit code: 0 for success, 1 for failure.

Directories that already exist count as success.
"""

import argparse
import sys
from pathlib import Path

SUBDIRS = ("input", "output", "temp")


def fail(reason: str) -> int:
    """Report a failure: reason to stderr, 'fail' to stdout."""
    print(f"[make_dir] {reason}", file=sys.stderr)
    print("fail")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a directory with input/output/temp subdirectories (like mkdir -p)."
    )
    parser.add_argument("dir_path", type=Path, help="Path of the base directory to create")
    args = parser.parse_args()

    base = args.dir_path.expanduser().resolve()

    for path in (base, *(base / name for name in SUBDIRS)):
        if path.exists() and not path.is_dir():
            return fail(f"Path exists but is not a directory: {path}")
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return fail(f"Could not create {path}: {exc}")

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
