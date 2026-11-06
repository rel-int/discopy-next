"""Replay the history's merges of CHANGELOG.md through changelog-merge.py.

Usage: ``python3 -I replay.py [--since 2026-10-14] [--ref origin/main] [-v]``
from a full clone. Two sets of merge commits reachable from the ref:

- conflicted: those whose ``git show --remerge-diff`` names CHANGELOG.md
  alone, compared with the hand resolution that was committed;
- clean: those where both parents changed CHANGELOG.md since the merge base
  and git merged it without a conflict, compared with the committed file,
  both through the driver and through its unit merge alone.

Each line of output is a count; ``-v`` adds one line per commit that is not
identical, with the lines the driver dropped (``-``) and added (``+``)
against the committed file, and ``resurrected`` for any line of the base that
the driver kept although one side deleted it.
"""
import argparse
import collections
import importlib.util
import os
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "changelog_merge", os.path.join(HERE, "changelog-merge.py"))
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)
PATH = "CHANGELOG.md"


def git(*args):
    """The standard output of a git command, or None when it fails."""
    proc = subprocess.run(["git", *args], capture_output=True, check=False)
    return proc.stdout.decode(errors="surrogateescape") \
        if proc.returncode == 0 else None


def show(rev):
    """CHANGELOG.md at a revision, or None."""
    return git("show", f"{rev}:{PATH}")


def merges(ref, since):
    """The merge commits reachable from ref since the date, newest first."""
    out = git("log", ref, "--merges", f"--since={since}", "--format=%H")
    if out is None:
        sys.exit(f"replay: git log {ref} failed")
    return out.split()


def run_driver(base, ours, theirs):
    """The driver's result on the three texts, and its exit status."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = [os.path.join(tmp, name) for name in "OAB"]
        for path, text in zip(paths, (base, ours, theirs)):
            with open(path, "w", encoding="utf-8", newline="",
                      errors="surrogateescape") as file:
                file.write(text)
        status = DRIVER.main(*paths)
        with open(paths[1], encoding="utf-8", newline="",
                  errors="surrogateescape") as file:
            return file.read(), status


def compare(got, want):
    """identical, multiset (same lines, another order) or differs."""
    if got == want:
        return "identical"
    if collections.Counter(got.splitlines()) == collections.Counter(
            want.splitlines()):
        return "multiset"
    return "differs"


def classify(got, want, base, ours, theirs):
    """Why the driver's result differs from the committed file.

    ``driver lost``: a bullet or link of main, or one the branch added, that
    the committed file kept and the driver did not; ``driver resurrected``: a
    base unit one side deleted and the other kept as is; ``driver
    duplicated``: a unit more often than on either side; ``hand edit``: the
    committed file wrote or dropped units of its own, or kept one that a
    side deleted; ``blank lines`` else.
    """
    have, need, old, mine, main = (collections.Counter(
        unit for unit in DRIVER.units(text) if unit.strip())
        for text in (got, want, base, ours, theirs))
    kept = [u for u in need if u not in have and u in old
            and (u in mine) != (u in main)]
    lost = [u for u in need if u not in have and u not in kept and (
        u in main or (u in mine and u not in old))]
    back = [u for u in have if u in old and u not in need and (
        (u not in main and mine[u] == old[u])
        or (u not in mine and main[u] == old[u]))]
    dup = [u for u in have if have[u] > max(mine[u], main[u])]
    marks = [u for u in have if u.startswith(("<<<<<<<", "=======",
                                              ">>>>>>>"))]
    dup = [u for u in dup if u not in marks]
    hand = kept + [u for u in need if u not in mine and u not in main] + [
        u for u in have if u not in need and u in mine and (
            u in main or u not in old) and u not in marks]
    reasons = [(name, units) for name, units in (
        ("driver lost", lost), ("driver resurrected", back),
        ("driver duplicated", dup), ("conflict", marks),
        ("hand edit", hand)) if units]
    return reasons or [("blank lines", [])]


def detail(commit, got, want, base, ours, theirs):
    """One line naming why the driver's result differs, and the units."""
    reasons = classify(got, want, base, ours, theirs)
    return f"{commit[:8]} " + ", ".join(
        f"{name} {len(units)}" for name, units in reasons) + "".join(
        f"\n    {name}: {unit.splitlines()[0][:72]}"
        for name, units in reasons if name.startswith("driver")
        for unit in units[:4])


def triple(commit):
    """The base, ours and theirs of a merge commit, or None."""
    bases = (git("merge-base", "--all", f"{commit}^1", f"{commit}^2")
             or "").split()
    if len(bases) != 1:
        return None
    texts = [show(rev) for rev in (bases[0], f"{commit}^1", f"{commit}^2")]
    return None if None in texts else texts


def main():
    """Print the counts of both replays."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2026-10-14")
    parser.add_argument("--ref", default="origin/main")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    conflicted, clean, notes = collections.Counter(), collections.Counter(), []
    for commit in merges(args.ref, args.since):
        names = set(git("show", "--remerge-diff", "--name-only", "--format=",
                        commit).split())
        want = show(commit)
        texts = triple(commit)
        if texts is None or want is None:
            conflicted["skipped"] += names == {PATH}
            continue
        base, ours, theirs = texts
        if names == {PATH}:
            got, status = run_driver(base, ours, theirs)
            try:
                DRIVER.merge(base, ours, theirs)
            except Exception:  # pylint: disable=broad-exception-caught
                conflicted["crashed"] += 1  # counted, not raised
            verdict = compare(got, want)
            conflicted[verdict] += 1
            conflicted[f"exit {status}"] += 1
            if verdict == "differs":
                notes.append(f"conflicted exit {status} "
                             + detail(commit, got, want, *texts))
                for name, _ in classify(got, want, *texts):
                    conflicted[name] += 1
        elif PATH not in names and ours != base and theirs != base \
                and ours != theirs:
            got, _ = run_driver(base, ours, theirs)
            clean["driver " + compare(got, want)] += 1
            try:
                merged = DRIVER.merge(base, ours, theirs)
            except Exception:  # pylint: disable=broad-exception-caught
                clean["units alone crashed"] += 1
                continue
            if merged is None:
                clean["units alone declined"] += 1
                continue
            units = merged[0]
            clean["units alone " + compare(units, want)] += 1
            if compare(units, want) == "differs":
                notes.append("clean units alone "
                             + detail(commit, units, want, *texts))
    total = sum(conflicted[k] for k in ("identical", "multiset", "differs"))
    print(f"conflicted {total} ({conflicted['skipped']} skipped, not one "
          f"merge base): identical {conflicted['identical']}, "
          f"multiset {conflicted['multiset']}, differs "
          f"{conflicted['differs']}; driver exit 0 {conflicted['exit 0']}, "
          f"exit 1 {conflicted['exit 1']}, crashed {conflicted['crashed']}")
    print("conflicted differs because: " + ", ".join(
        f"{name} {conflicted[name]}" for name in (
            "hand edit", "blank lines", "conflict", "driver lost",
            "driver resurrected",
            "driver duplicated")))
    total = sum(v for k, v in clean.items() if k.startswith("driver"))
    print(f"clean {total}: driver identical {clean['driver identical']}, "
          f"units alone identical {clean['units alone identical']}, "
          f"multiset {clean['units alone multiset']}, "
          f"differs {clean['units alone differs']}, "
          f"declined (a release) {clean['units alone declined']}, "
          f"crashed {clean['units alone crashed']}")
    if args.verbose:
        print("\n".join(notes))


if __name__ == "__main__":
    sys.exit(main())
