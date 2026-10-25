#!/usr/bin/env python3
"""Create a new, locked and verified Parquet archive backup snapshot."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.archive_backup import backup_archive  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("archive"))
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(backup_archive(args.root, args.destination), indent=2))


if __name__ == "__main__":
    main()
