"""The extras and dependency groups of ``pyproject.toml``, for the suites that
check what one of them installs."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

with open(Path(__file__).parents[2] / "pyproject.toml", "rb") as file:
    PYPROJECT = tomllib.load(file)

EXTRAS = PYPROJECT["project"]["optional-dependencies"]
GROUPS = PYPROJECT["dependency-groups"]


def installs(table, name, extras=EXTRAS):
    """The names the extra or dependency group ``name`` of ``table`` installs,
    with those of the groups it includes and of the ``extras`` of discopy it
    requires."""
    names = set()
    for entry in table[name]:
        if isinstance(entry, dict):
            names |= installs(table, entry["include-group"], extras)
            continue
        requirement = Requirement(entry)
        if requirement.name == "discopy":
            names = names.union(*(installs(extras, extra, extras)
                                  for extra in requirement.extras))
        else:
            names.add(requirement.name)
    return names
