"""Check that docs/export_notebooks.py runs the notebooks of its own tree."""

from pathlib import Path

from docs import export_notebooks

ROOT = Path(__file__).resolve().parents[1]

PROBE = """\
```python {{.marimo}}
import discopy
assert discopy.__file__.startswith({root!r}), discopy.__file__
```
"""


def test_notebook_imports_the_discopy_of_its_tree(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    notebook = tmp_path / "probe.md"
    notebook.write_text(PROBE.format(root=str(ROOT / "discopy")))

    export_notebooks.export(notebook, check=True)
