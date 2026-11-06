"""The extras and dependency groups of ``pyproject.toml``, for the suites that
check what one of them installs."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

with open(Path(__file__).parents[2] / "pyproject.toml", "rb") as file:
    PYPROJECT = tomllib.load(file)

EXTRAS = PYPROJECT["project"]["optional-dependencies"]
GROUPS = PYPROJECT["dependency-groups"]


def installs(extras, extra):
    """The names an extra installs, with those of the extras it includes."""
    names = set()
    for requirement in map(Requirement, extras[extra]):
        if requirement.name == "discopy":
            names = names.union(
                *(installs(extras, name) for name in requirement.extras))
        else:
            names.add(requirement.name)
    return names
