"""Tests for the check that a pull request touching discopy/ adds a
changelog fragment."""

import pytest


@pytest.mark.parametrize("changes, labels, expected", [
    ([("M", "discopy/cat.py"), ("A", "changelog.d/fix-x.fixed.md")], set(), 0),
    ([("M", "discopy/cat.py")], set(), 1),
    ([("M", "discopy/cat.py")], {"no-changelog"}, 0),
    ([("M", "discopy/cat.py"), ("M", "changelog.d/old.fixed.md")], set(), 1),
    ([("M", "discopy/cat.py"), ("A", "changelog.d/x.fixed.2.md")], set(), 0),
    ([("M", "docs/index.rst")], set(), 0),
    ([("D", "changelog.d/released.added.md"), ("M", "CHANGELOG.md")],
     set(), 0),
])
def test_a_fragment_is_needed_only_for_discopy(
        changelog_fragment, changes, labels, expected):
    assert len(changelog_fragment.problems(changes, labels)) == expected


@pytest.mark.parametrize("path", [
    "changelog.d/x.md",
    "changelog.d/x.feature.md",
    "changelog.d/X.added.md",
    "changelog.d/x.added.txt",
    "changelog.d/sub/x.added.md",
])
def test_a_misnamed_fragment_fails_even_with_the_label(
        changelog_fragment, path):
    found = changelog_fragment.problems(
        [("A", path), ("M", "discopy/cat.py")], {"no-changelog"})
    assert len(found) == 1 and found[0].startswith(path)


def test_the_sections_are_those_towncrier_compiles(changelog_fragment):
    import pathlib
    import tomllib
    pyproject = pathlib.Path(__file__).parents[2] / "pyproject.toml"
    with open(pyproject, "rb") as file:
        types = tomllib.load(file)["tool"]["towncrier"]["type"]
    assert {t["directory"] for t in types} == set(changelog_fragment.SECTIONS)
