"""Create a verified, independent archive snapshot without copying orphans."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

from research.archive import (
    _fsync_directory, archive_lock, file_path, read_catalog, verify_entry,
)


def separate_roots(root, backup_root):
    root, backup_root = Path(root).resolve(), Path(backup_root).resolve()
    if root == backup_root or root in backup_root.parents or backup_root in root.parents:
        raise ValueError("backup must be a separate archive copy, preferably on another device")
    return root, backup_root


def verify_copy(root, backup_root, entry, backup_entry):
    if backup_entry != entry:
        raise ValueError("backup is missing, stale or corrupt")
    original, copy = file_path(root, entry), file_path(backup_root, backup_entry)
    if copy.is_file() and original.samefile(copy):
        raise ValueError("backup must contain independent files, not hard links")
    verify_entry(root, entry)
    verify_entry(backup_root, backup_entry)


def backup_archive(root, destination):
    """Publish a new snapshot; never overwrite an existing backup directory.

    A shared source lock excludes archive writers throughout the copy. The
    catalog is published with the files by one directory rename. A backup on the
    same device survives file damage but does not protect against device loss.
    """
    root, destination = separate_roots(root, destination)
    if destination.exists():
        raise FileExistsError("backup destination already exists; choose a new snapshot")
    if not (root / "catalog.json").is_file():
        raise ValueError("source archive catalog is missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".archive-backup-", dir=destination.parent))
    try:
        with archive_lock(root):
            entries = read_catalog(root)
            if not entries:
                raise ValueError("source archive is empty")
            for entry in entries:
                target = staging / entry["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file_path(root, entry), target)
                with target.open("rb") as stream:
                    os.fsync(stream.fileno())
                verify_copy(root, staging, entry, entry)
                for directory in (target.parent, *target.parent.parents):
                    _fsync_directory(directory)
                    if directory == staging:
                        break
            shutil.copyfile(root / "catalog.json", staging / "catalog.json")
            with (staging / "catalog.json").open("rb") as stream:
                os.fsync(stream.fileno())
            _fsync_directory(staging)
            # The destination is a new snapshot, not an existing mutable backup.
            if destination.exists():
                raise FileExistsError("backup destination was created concurrently")
            staging.rename(destination)
            _fsync_directory(destination.parent)
        return {
            "files": len(entries), "rows": sum(e["rows"] for e in entries),
            "bytes": sum(e["bytes"] for e in entries),
            "same_device": root.stat().st_dev == destination.stat().st_dev,
            "destination": str(destination),
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)  # Only this call's unpublished temporary copy.
