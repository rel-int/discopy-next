"""Extract the Python code blocks from README.md and check they run."""

from pathlib import Path

import matplotlib

README = Path(__file__).parent.parent / "README.md"


def test_readme(monkeypatch):
    from fixtures.readme import run_code_blocks
    monkeypatch.chdir(README.parent)
    matplotlib.use("Agg")
    run_code_blocks(README)
