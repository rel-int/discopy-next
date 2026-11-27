"""Check that a pull request touching ``discopy/`` adds a changelog fragment.

Runs from ``changelog.yml`` with the files the pull request changes against
its merge base, one ``<status>\\t<path>`` line each as ``git diff --name-status
--no-renames`` prints them, on standard input, and ``LABELS`` set to
``no-changelog`` when the pull request carries that label, matched exactly.
A fragment is ``changelog.d/<slug>.<section>[.<n>].md``, ``<section>`` one
of the ``[[tool.towncrier.type]]`` directories of ``pyproject.toml``. Every
fragment the pull request adds or edits must be named so, whatever it
touches; one must be added when it touches ``discopy/``, unless it carries
the ``no-changelog`` label.
"""

import os
import re
import sys

LABEL = "no-changelog"
SECTIONS = ("added", "changed", "deprecated", "removed", "fixed", "project")
FRAGMENT = re.compile(
    r"changelog\.d/[a-z0-9][a-z0-9_-]*\.(?:%s)(?:\.[0-9]+)?\.md"
    % "|".join(SECTIONS))
IGNORED = {"changelog.d/README.md", "changelog.d/template.md.jinja"}


def problems(changes, labels):
    """What stops the pull request, given its ``(status, path)`` changes."""
    found = [f"{path}: not changelog.d/<slug>.<section>.md, <section> one "
             f"of {', '.join(SECTIONS)}"
             for status, path in changes
             if path.startswith("changelog.d/") and status != "D"
             and path not in IGNORED and not FRAGMENT.fullmatch(path)]
    added = [path for status, path in changes
             if status == "A" and FRAGMENT.fullmatch(path)]
    touches = any(path.startswith("discopy/") for _, path in changes)
    if touches and not added and LABEL not in labels:
        found.append(
            "discopy/ changed and no changelog.d/ fragment was added: add "
            f"one, or label the pull request {LABEL} if nobody using "
            "DisCoPy would notice the change")
    return found


def main():
    changes = [tuple(line.split("\t")[:2]) for line in sys.stdin
               if line.strip()]
    labels = set(filter(None, os.environ.get("LABELS", "").split(",")))
    found = problems([(s[0], path.strip()) for s, path in changes], labels)
    for problem in found:
        print(f"::error::{problem}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
