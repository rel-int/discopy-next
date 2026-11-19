"""Check that docs/export_notebooks.py runs the notebooks it is asked for,
on the discopy of its own tree."""

from pathlib import Path

from docs import export_notebooks
from fixtures.pyproject import EXTRAS, installs

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


def test_every_name_must_match_a_notebook(monkeypatch, capsys):
    monkeypatch.setattr(export_notebooks, "render", lambda *_, **__: True)
    monkeypatch.setattr("sys.argv", [
        "export_notebooks.py", "--check", "examples", "fixpoint"])

    assert export_notebooks.main() == 1
    assert "fixpoint" in capsys.readouterr().err


def test_the_docs_extra_installs_what_the_optyx_notebooks_need():
    """The docs build installs the optyx extra and joblib, so that it renders
    the notebooks of ``optyx/`` rather than their fallback link pages."""
    assert installs(EXTRAS, "optyx") | {"joblib"} <= installs(EXTRAS, "docs")
