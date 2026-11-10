"""Check that the Python code blocks of README.md run and its links resolve."""

import re
from pathlib import Path

import matplotlib

README = Path(__file__).parent.parent / "README.md"
FENCE = re.compile(r"```.*?```", re.DOTALL)
LINK = re.compile(r"\]\(([^)]*)\)")
HEADING = re.compile(r"^#+ (.+)$", re.MULTILINE)


def test_readme(monkeypatch):
    from fixtures.readme import run_code_blocks
    monkeypatch.chdir(README.parent)
    matplotlib.use("Agg")
    run_code_blocks(README)


def test_readme_links_resolve():
    text = FENCE.sub("", README.read_text(encoding="utf-8"))
    anchors = {"#" + re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
               for heading in HEADING.findall(text)}
    broken = [target for target in LINK.findall(text)
              if not target.startswith(("http://", "https://"))
              and target not in anchors
              and not (README.parent / target).exists()]
    assert broken == []
