"""Extract the Python code blocks from discopy/optyx/README.md and check they
run. They need every extra optyx interfaces with: graphix, perceval and pyzx,
so ``--skip-extra`` skips this file with the rest of ``test/optyx``."""

from pathlib import Path

import matplotlib

README = Path(__file__).parent.parent.parent / "discopy" / "optyx" / "README.md"


def test_readme(monkeypatch):
    from fixtures.readme import run_code_blocks
    monkeypatch.chdir(README.parent)
    matplotlib.use("Agg")
    run_code_blocks(README)
