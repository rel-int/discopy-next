"""Check that docs/export_notebooks.py runs the notebooks it is asked for,
on the discopy of its own tree."""

import tarfile
from pathlib import Path

import pytest

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


SPAWN = """\
# A process pool — as cotengra's optimizer starts

```python {.marimo}
import marimo as mo
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(
        1, mp_context=multiprocessing.get_context("spawn")) as pool:
    assert list(pool.map(abs, [-1])) == [1]
```
"""


@pytest.mark.slow
def test_a_spawned_worker_can_import_the_notebook(tmp_path):
    """A spawned worker re-imports the notebook as its main module, which
    must be Python and not Markdown (#251)."""
    notebook = tmp_path / "spawn.md"
    notebook.write_text(SPAWN)

    export_notebooks.export(notebook, check=True)


def test_every_name_must_match_a_notebook(monkeypatch, capsys):
    monkeypatch.setattr(export_notebooks, "render", lambda *_, **__: True)
    monkeypatch.setattr("sys.argv", [
        "export_notebooks.py", "--check", "examples", "fixpoint"])

    assert export_notebooks.main() == 1
    assert "fixpoint" in capsys.readouterr().err


@pytest.mark.parametrize("names, runs", [
    ([], False), (["readme_example"], True)])
def test_an_unexecuted_notebook_runs_only_when_named(
        monkeypatch, names, runs):
    rendered = []
    monkeypatch.setattr(export_notebooks, "render", lambda notebook, **_: (
        rendered.append(export_notebooks.name_of(notebook)) or True))
    monkeypatch.setattr("sys.argv", ["export_notebooks.py", "--check", *names])

    assert export_notebooks.main() == 0
    assert ("optyx/readme_example" in rendered) is runs


def test_the_docs_build_links_to_an_unexecuted_notebook(monkeypatch):
    pages = {}
    monkeypatch.delenv(export_notebooks.SKIP, raising=False)
    monkeypatch.setattr(export_notebooks, "export", lambda *_, **__: None)
    monkeypatch.setattr(export_notebooks, "write_page", lambda nb, **kw: (
        pages.setdefault(export_notebooks.name_of(nb), kw)))

    export_notebooks.generate()

    assert pages["optyx/readme_example"] == {
        "rendered": False, "skipped": True}
    assert pages["optyx/fixpoints"] == {"rendered": True}


def test_the_docs_extra_installs_what_the_optyx_notebooks_need():
    """The docs build installs the optyx extra and joblib, so that it renders
    the notebooks of ``optyx/`` rather than their fallback link pages."""
    assert installs(EXTRAS, "optyx") | {"joblib"} <= installs(EXTRAS, "docs")


@pytest.mark.parametrize("published", [True, False])
def test_a_build_that_runs_none_embeds_those_main_rendered(
        tmp_path, monkeypatch, published):
    """Read the Docs runs no notebook: it embeds the HTML the workflow pushed
    to the rendered-notebooks branch, and links to the source of a notebook
    the branch lacks, of an unexecuted one, or of all when it is not there."""
    branch = tmp_path / "discopy-next-rendered-notebooks"
    for name in ["diagrams", "optyx/fixpoints", "optyx/readme_example"]:
        (branch / f"{name}.html").parent.mkdir(parents=True, exist_ok=True)
        (branch / f"{name}.html").write_text(f"<p>{name}</p>")
    (branch / "COMMIT").write_text("0123456789abcdef\n")
    tarball = tmp_path / "rendered-notebooks.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(branch, branch.name)
    pages = {}
    monkeypatch.setenv(export_notebooks.SKIP, "1")
    monkeypatch.setenv(export_notebooks.RENDERED, (
        tarball if published else tmp_path / "missing.tar.gz").as_uri())
    monkeypatch.setattr(export_notebooks, "HTML_DIR", tmp_path / "html")
    monkeypatch.setattr(export_notebooks, "export", pytest.fail)
    monkeypatch.setattr(export_notebooks, "write_page", lambda nb, **kw: (
        pages.setdefault(export_notebooks.name_of(nb), kw)))

    export_notebooks.generate()

    embedded = {"rendered": True, "commit": "0123456789abcdef"}
    skipped = {"rendered": False, "skipped": True}
    assert pages["diagrams"] == pages["optyx/fixpoints"]\
        == (embedded if published else skipped)
    assert pages["qnlp"] == pages["optyx/readme_example"] == skipped
    assert (tmp_path / "html" / "optyx" / "fixpoints.html").exists()\
        is published
