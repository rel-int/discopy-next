#!/usr/bin/env python
"""Render the marimo notebooks in ``docs/notebooks`` for the documentation.

For every ``docs/notebooks/*.md`` marimo notebook, and those of
:mod:`discopy.optyx` in ``docs/notebooks/optyx``, this module:

1. runs it and exports the *computed* result to a self-contained HTML file in
   ``docs/_static/notebooks/<name>.html`` (via ``marimo export html``), and
2. writes a small reStructuredText page ``docs/notebooks/<name>.rst`` that
   embeds that HTML in an ``<iframe>`` so Sphinx picks it up in the toctree,

where ``<name>`` is the path of the notebook in ``docs/notebooks`` without its
suffix, e.g. ``qnlp`` or ``optyx/bose_hubbard``.

Both artefacts are generated (they are *not* committed): ``conf.py`` calls
:func:`generate` from a ``builder-inited`` hook, so ``sphinx-build`` renders the
notebooks as part of a normal docs build. If a notebook cannot be executed
(e.g. an optional dependency is missing on Read the Docs) the build still
succeeds -- the page falls back to a link to the notebook source.

It can also be run as a script::

    python docs/export_notebooks.py            # render every notebook
    python docs/export_notebooks.py qnlp        # render a single notebook
    python docs/export_notebooks.py --check     # only check they execute
    python docs/export_notebooks.py --check --jobs 4  # four at a time

``--check`` fails when a notebook does not execute, unless it is one of
:data:`KNOWN_BROKEN`, which fail on a bug that an issue tracks.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory

DOCS = Path(__file__).resolve().parent
NOTEBOOKS = DOCS / "notebooks"
HTML_DIR = DOCS / "_static" / "notebooks"
SUBDIRECTORIES = ("optyx", )
REPOSITORY = os.environ.get(
    "READTHEDOCS_GIT_CLONE_URL", "https://github.com/discopy/discopy")
SOURCE = "{}/blob/{}".format(
    REPOSITORY.removesuffix(".git"),
    os.environ.get("READTHEDOCS_GIT_COMMIT_HASH", "main"))
ISSUES = "https://github.com/rel-int/discopy-next/issues/"

# Notebooks that fail to execute on a bug of their own, each tracked by an
# issue, so that ``--check`` does not block on them; ``generate`` already falls
# back to a link page for any notebook that fails to execute.
KNOWN_BROKEN = {
    "optyx/feed_forward_example": ISSUES + "247",
    "optyx/photon_distinguishability": ISSUES + "247",
    "optyx/readme_example": ISSUES + "247",
    "optyx/test_light_cones": ISSUES + "249",
}

IFRAME = """\
.. raw:: html

    <iframe class="marimo-notebook" src="{up}_static/notebooks/{name}.html"
            title="{title}" loading="lazy"
            style="width: 100%; height: 90vh; border: none;"></iframe>
"""

THEME_STYLE = """\
<style type="text/css">
/* DisCoPy SVGs turn the elements tagged by drawing.backend.DARK_MODE_STYLE
   white under a dark prefers-color-scheme; rendered inline in the notebook,
   they read the browser preference rather than the notebook theme, so these
   higher-specificity rules retag them by marimo's theme class instead. */
.light [id^="dark-stroke"] path { stroke: #000000 !important; }
.light [id^="dark-fill"] g, .light [id^="dark-fill"] use
{ fill: #000000 !important; stroke: #000000 !important; }
.dark [id^="dark-stroke"] path { stroke: #ffffff !important; }
.dark [id^="dark-fill"] g, .dark [id^="dark-fill"] use
{ fill: #ffffff !important; stroke: #ffffff !important; }
</style>"""

FALLBACK = """\
.. note::

    This notebook could not be rendered in this build (a dependency needed to
    execute it may be missing). You can read its source on GitHub:
    `{name}.md <{source}/docs/notebooks/{name}.md>`_.
"""


def notebooks() -> list[Path]:
    """Every notebook, those of :data:`SUBDIRECTORIES` after the others."""
    return sorted(NOTEBOOKS.glob("*.md")) + [
        notebook for directory in SUBDIRECTORIES
        for notebook in sorted((NOTEBOOKS / directory).glob("*.md"))]


def name_of(notebook: Path) -> str:
    """The path of ``notebook`` in ``docs/notebooks``, without its suffix."""
    return notebook.relative_to(NOTEBOOKS).with_suffix("").as_posix()


def title_of(notebook: Path) -> str:
    """The first level-one heading of ``notebook``, outside of code cells.

    Falls back to the ``title:`` field of the YAML front-matter, which marimo
    derives from the name of the file, and then to the name of the file.
    """
    lines = notebook.read_text().splitlines()
    closing = [i for i, line in enumerate(lines) if i and line == "---"]
    end = closing[0] + 1 if lines[:1] == ["---"] and closing else 0
    front, body, fenced = lines[1:end - 1] if end else [], lines[end:], False
    for line in body:
        fenced ^= line.startswith("```")
        if not fenced and line.startswith("# "):
            return line[2:].strip()
    for line in front:
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip()
    return notebook.stem


def export(notebook: Path, *, check: bool) -> None:
    """Run ``notebook`` and export the computed HTML (unless ``check``)."""
    with TemporaryDirectory() as scratch:
        if check:
            output = Path(scratch) / f"{notebook.stem}.html"
        else:
            output = HTML_DIR / f"{name_of(notebook)}.html"
            output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "marimo", "export", "html", notebook.name,
             "-o", str(output), "-f"],
            cwd=notebook.parent, check=True)
        add_theme_style(output, title_of(notebook))


def add_theme_style(output: Path, title: str) -> None:
    """Insert :data:`THEME_STYLE` and ``title`` in an exported notebook."""
    text = re.sub("<title>.*?</title>",
                  lambda _: f"<title>{escape(title)}</title>",
                  output.read_text(), count=1, flags=re.DOTALL)
    output.write_text(text.replace("</head>", THEME_STYLE + "</head>", 1))


def write_page(notebook: Path, *, rendered: bool) -> None:
    """Write the reStructuredText page for ``notebook``."""
    title, name = title_of(notebook), name_of(notebook)
    body = (IFRAME if rendered else FALLBACK).format(
        name=name, up="../" * (name.count("/") + 1), title=escape(title),
        source=SOURCE)
    (NOTEBOOKS / f"{name}.rst").write_text(
        f"{title}\n{'=' * len(title)}\n\n" + body)


def generate(*, strict: bool = False) -> None:
    """Render every notebook to HTML and write its page (used by ``conf.py``).

    When ``strict`` is false a notebook that fails to execute falls back to a
    link page instead of aborting the build.
    """
    for notebook in notebooks():
        try:
            export(notebook, check=False)
            write_page(notebook, rendered=True)
        except (subprocess.CalledProcessError, OSError):
            if strict:
                raise
            write_page(notebook, rendered=False)


def render(notebook: Path, *, check: bool) -> bool:
    """Export ``notebook`` and write its page, saying whether it executed."""
    print(f"rendering {name_of(notebook)}.md ...", flush=True)
    try:
        export(notebook, check=check)
    except subprocess.CalledProcessError:
        return False
    if not check:
        write_page(notebook, rendered=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*",
                        help="names or stems to render (default: all)")
    parser.add_argument("--check", action="store_true",
                        help="only check the notebooks execute, write nothing")
    parser.add_argument("--jobs", type=int, default=1,
                        help="how many notebooks to run at a time")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    names = set(args.notebooks)
    selected = [nb for nb in notebooks()
                if not names or {nb.stem, name_of(nb)} & names]
    if not selected:
        print("no matching notebooks found", file=sys.stderr)
        return 1

    with ThreadPoolExecutor(args.jobs) as pool:
        executed = list(pool.map(
            lambda notebook: render(notebook, check=args.check), selected))
    failures = []
    for notebook, ok in zip(selected, executed):
        name = name_of(notebook)
        if not ok and args.check and name in KNOWN_BROKEN:
            print(f"{name}: known failure, see {KNOWN_BROKEN[name]}",
                  file=sys.stderr)
        elif not ok:
            failures.append(name)
    if failures:
        print(f"failed to execute: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
