"""
Check that the paper builds: its listings run, its figures are drawn by the
Typst and the TikZ backends, its LaTeX is written from its Typst, and both
compile where their compiler is installed.
"""

import doctest
import re
import shutil

import matplotlib
import pytest

from paper import build, transpile
from paper.transpile import Unsupported


@pytest.fixture(scope="module")
def names():
    """ The names of the figures, once the paper is built. """
    matplotlib.use("Agg")
    return build.build()


def test_the_transpiler_does_what_its_docstrings_say():
    assert doctest.testmod(transpile).failed == 0


@pytest.mark.slow
def test_the_listings_run_and_every_figure_is_drawn_twice(names):
    assert names == sorted(set(names), key=names.index)
    for name in names:
        typst = (build.PAPER / "figures" / f"{name}.typ").read_text("utf-8")
        tikz = (build.PAPER / "figures" / f"{name}.tikz").read_text("utf-8")
        assert "canvas(" in typst and "#set page" not in typst
        assert tikz.startswith("\\begin{tikzpicture}")


@pytest.mark.slow
def test_every_reference_and_citation_of_the_latex_resolves(names):
    assert names
    latex = (build.PAPER / "main.tex").read_text("utf-8")
    bibliography = (
        build.PAPER.parent / "docs" / "discopy.bib").read_text("utf-8")
    labels = set(re.findall(r"\\label\{([^}]+)\}", latex))
    keys = set(re.findall(r"^@\w+\{([^,]+),", bibliography, re.MULTILINE))
    assert set(re.findall(r"\\cref\{([^}]+)\}", latex)) <= labels
    assert set(re.findall(r"\\cite\{([^}]+)\}", latex)) <= keys


def test_the_transpiler_refuses_what_it_does_not_read():
    for refused in ["#strike[this]", "$a / b$", "$integral f$", "#link[x]",
                    "$50% x$", "$a ~ b$", "${x}$",
                    '#link(<label>)[x]', "a \\ b", "half a $ sign"]:
        with pytest.raises(Unsupported):
            transpile.inline(refused)
    with pytest.raises(Unsupported, match="no blank line"):
        transpile.blocks("A paragraph\n= and a heading\n")
    for refused in ["#table()", "==== Too deep", "=No space"]:
        with pytest.raises(Unsupported):
            transpile.block_to_latex(refused, build.PAPER)


@pytest.mark.slow
def test_the_typst_compiles(names):
    typst = pytest.importorskip("typst")
    assert names
    try:
        pdf = build.typst()
    except typst.TypstError as error:
        if "failed to download package" in str(error):
            pytest.skip("CeTZ cannot be downloaded.")
        raise
    assert pdf.startswith(b"%PDF")


@pytest.mark.slow
def test_the_latex_compiles_with_nothing_undefined(names):
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed.")
    assert names
    pdf = build.latex()
    assert pdf.read_bytes().startswith(b"%PDF")
    log = pdf.with_suffix(".log").read_text("utf-8", errors="replace")
    assert "undefined" not in log.lower()


def test_the_transpiler_escapes_what_latex_reads_as_markup():
    assert transpile.inline("snake_case & 100% {x} a~b")\
        == r"snake\_case \& 100\% \{x\} a~b"
    assert transpile.inline('#link("https://a.b/c_d#e")[text_1]')\
        == r"\href{https://a.b/c_d\#e}{text\_1}"
