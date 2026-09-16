#!/usr/bin/env python3
"""Restore local-only WeDetect helpers without overwriting different files."""

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party_overlays" / "WeDetect"
DESTINATION = ROOT / "InstanceRecognition" / "ThirdParty" / "WeDetect"
EXPECTED_HEAD = "dd302dba0069ace1b05816bafbc3fa1dbd6aa68c"
FILES = (
    "LOCAL_REPRODUCTION.md",
    "requirements-local.txt",
    "run_local_examples.sh",
    "verify_local_reproduction.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate and report; do not write files")
    args = parser.parse_args()
    if not (DESTINATION / ".git").exists():
        parser.error("WeDetect submodule is missing; run git submodule update --init --recursive first")
    head = subprocess.check_output(["git", "-C", str(DESTINATION), "rev-parse", "HEAD"], text=True).strip()
    if head != EXPECTED_HEAD:
        parser.error("WeDetect is not at the recorded commit; check local changes before restoring")

    # Preflight every file before writing anything.
    planned = []
    for name in FILES:
        source, destination = SOURCE / name, DESTINATION / name
        if not source.is_file():
            parser.error("Missing overlay source: " + str(source))
        content = source.read_bytes()
        if destination.is_symlink():
            parser.error("Refusing to write through a symlink: " + str(destination))
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != content:
                parser.error("Refusing to overwrite a different local file: " + str(destination))
            print("UNCHANGED " + name)
        else:
            planned.append((source, destination, content))
            print(("WOULD RESTORE " if args.check else "RESTORE ") + name)
    if not args.check:
        for source, destination, content in planned:
            with destination.open("xb") as output:
                output.write(content)
            destination.chmod(source.stat().st_mode & 0o777)
    print("CHECK OK" if args.check else "RESTORE OK")


if __name__ == "__main__":
    main()
