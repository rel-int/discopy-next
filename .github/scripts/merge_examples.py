"""Merge the Hypothesis example databases of the shards of a proptest run.

Every shard starts from the same database and writes only under the keys of
the cells it runs, which no other shard runs, so the merged database is the
union of what the shards end with, minus what any shard deleted: without
the deletions, a copy a shard never touched would bring back an entry the
shard that owns it had pruned, on every run.

Each shard directory holds the database as ``examples/`` and the paths it
deleted, relative to it, one per line as ``deleted.txt``.

Usage: python merge_examples.py OUT SHARD...
"""

import pathlib
import shutil
import sys


def deleted(shard):
    """The paths a shard deleted from the database it started with."""
    path = shard / "deleted.txt"
    lines = path.read_text().splitlines() if path.exists() else []
    return {pathlib.PurePosixPath(line) for line in lines if line}


def entries(shard):
    """The paths of the database a shard ended with."""
    root = shard / "examples"
    return {pathlib.PurePosixPath(path.relative_to(root).as_posix())
            for path in root.rglob("*") if path.is_file()}


def merge(out, shards):
    """Write the union of the shards' databases minus their deletions."""
    gone = set().union(*map(deleted, shards))
    for shard in shards:
        for entry in sorted(entries(shard) - gone):
            target = out / entry
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(shard / "examples" / entry, target)


if __name__ == "__main__":
    out, *shards = map(pathlib.Path, sys.argv[1:])
    merge(out, shards)
