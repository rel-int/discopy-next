"""
The API docs show short names, and a longer one only where two short names
would read the same: an autosummary entry is the name of its object in the
current module, a page is titled with ``name``, the last part of its name,
and a reference to a module or an object keeps its last part, or names as
much of its path as tells it apart from another reference of its paragraph,
as the bases of a class do, e.g. those of ``ribbon.Diagram`` read
``pivotal.Diagram, balanced.Diagram``. ``objname`` is the full name of a
module its package does not import, and Sphinx's default template titles a
page with ``fullname``, so a list that generates pages names its template.
"""

import importlib.util
import re
from pathlib import Path

import pytest

from discopy import cat, ribbon, symmetric

ROOT = Path(__file__).parent.parent
SOURCES = sorted([
    *(ROOT / "docs" / "api").glob("*.rst"),
    *(ROOT / "docs" / "_templates").glob("*.rst"),
    ROOT / "docs" / "index.rst",
    *(ROOT / "discopy").rglob("*.py")])

AUTOSUMMARY = re.compile(r"^( *)\.\. autosummary::\n((?:\1 +.*\n|\n)*)", re.M)
REFERENCE = re.compile(
    r":(mod|class|meth|func|obj|attr|exc):"
    r"`(?:([^`<]+?)\s*<([\w.]+)>|([~!.]?)([\w.]+))`")


def entries(text):
    """ The entries of every autosummary list in a text. """
    for match in AUTOSUMMARY.finditer(text):
        for line in match.group(2).splitlines():
            line = line.strip()
            if line and not line.startswith(":"):
                yield line


def collisions(text):
    """ The texts that two references of one paragraph read the same. """
    for paragraph in re.split(r"\n\s*\n", text):
        targets = {}
        for role, title, titled, tilde, target in REFERENCE.findall(paragraph):
            shown = title or (
                target.rsplit(".", 1)[-1] if tilde == "~" else target)
            if not title and role in ("meth", "func"):
                shown = shown.removesuffix("()") + "()"
            targets.setdefault(shown, set()).add(titled or target)
        yield from (shown for shown, found in targets.items()
                    if len(found) > 1)


@pytest.mark.parametrize(
    "path", SOURCES, ids=lambda path: str(path.relative_to(ROOT)))
def test_short_names(path):
    text = path.read_text(encoding="utf-8")
    assert [name for name in entries(text) if "." in name] == []
    assert [block.group(0) for block in AUTOSUMMARY.finditer(text)
            if ":toctree:" in block.group(2)
            and ":template:" not in block.group(2)] == []
    assert [match.group(0) for match in REFERENCE.finditer(text)
            if (match.group(5) or match.group(2) or "").startswith(
                "discopy.") and match.group(4) != "~"] == []
    assert list(collisions(text)) == []
    if path.parent.name == "_templates":
        assert text.startswith("{{ name | escape | underline}}")


def bases(cls):
    """ The bases of a class as the docs extension shows them. """
    spec = importlib.util.spec_from_file_location(
        "bases", ROOT / "docs" / "_ext" / "bases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = list(cls.__dict__.get("__orig_bases__", cls.__bases__))
    module.process_bases(None, cls.__name__, cls, {}, result)
    return result


def test_bases():
    assert bases(ribbon.Diagram)[:2] == [
        ":class:`pivotal.Diagram <discopy.pivotal.Diagram>`",
        ":class:`balanced.Diagram <discopy.feedback.closed.balanced.Diagram>`"]
    assert bases(symmetric.Box)[0] == (
        ":class:`balanced.Box <discopy.balanced.Box>`")
    assert bases(symmetric.Box)[1] is symmetric.Diagram
    assert bases(cat.Box) == list(cat.Box.__bases__)
