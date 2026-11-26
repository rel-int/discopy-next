"""
The API docs show short names: an autosummary entry is the name of its
object in the current module, a module reference keeps its last part, and a
page is titled with ``name``, the last part of its name: ``objname`` is the
full name of a module its package does not import, and Sphinx's default
template titles a page with ``fullname``, so a list that generates pages
names its template.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SOURCES = sorted([
    *(ROOT / "docs" / "api").glob("*.rst"),
    *(ROOT / "docs" / "_templates").glob("*.rst"),
    ROOT / "docs" / "index.rst",
    *(ROOT / "discopy").rglob("*.py")])

AUTOSUMMARY = re.compile(r"^( *)\.\. autosummary::\n((?:\1 +.*\n|\n)*)", re.M)


def entries(text):
    """ The entries of every autosummary list in a text. """
    for match in AUTOSUMMARY.finditer(text):
        for line in match.group(2).splitlines():
            line = line.strip()
            if line and not line.startswith(":"):
                yield line


@pytest.mark.parametrize(
    "path", SOURCES, ids=lambda path: str(path.relative_to(ROOT)))
def test_short_names(path):
    text = path.read_text(encoding="utf-8")
    assert [name for name in entries(text) if "." in name] == []
    assert [block.group(0) for block in AUTOSUMMARY.finditer(text)
            if ":toctree:" in block.group(2)
            and ":template:" not in block.group(2)] == []
    assert re.findall(r":mod:`discopy\.[\w.]+`", text) == []
    if path.parent.name == "_templates":
        assert text.startswith("{{ name | escape | underline}}")
