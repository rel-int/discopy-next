"""Tests for the Typst AST: the Typst mathematics of a LaTeX label."""

import pytest

from discopy.drawing.typst_ast import (
    LATEX_FUNCTIONS, LATEX_SYMBOLS, latex_math)


def compiled(math):
    """ Compile a page holding the Typst mathematics ``math``. """
    typst = pytest.importorskip("typst")
    return typst.compile(
        f"#set page(width: auto, height: auto)\n${math}$".encode(),
        format="svg")


@pytest.mark.parametrize("command", sorted(LATEX_SYMBOLS))
def test_every_latex_symbol_has_its_typst(command):
    compiled(latex_math(f"x \\{command} y"))


@pytest.mark.parametrize("command", sorted(LATEX_FUNCTIONS))
def test_every_latex_function_has_its_typst(command):
    compiled(latex_math(f"\\{command}{{a}}{{b}}"))


@pytest.mark.parametrize("latex, typst", [
    ("<=", "\\< ="),
    ("->", "- \\>"),
    ("xy", "x y"),
    ("a/b", "a \\/ b"),
    ("f(x, y)", "f \\( x \\, y \\)"),
    ("\\phi_1", "phi.alt_(1)"),
    ("\\rho_\\star", "rho_(star.op)"),
    ("x^\\frac{a}{b}", "x^(frac(a, b))"),
    ("\\left( x \\right.", "\\( x"),
    ("\\operatorname{Tr}(f)", 'op("Tr") \\( f \\)'),
    ("\\text{a \"b\"}", '"a \\"b\\""'),
    ("f''", "f''"),
    ("^2", '""^(2)'),
    ("\\int_a^b f(x)\\,dx = \\sqrt{2}",
     "integral_(a)^(b) f \\( x \\) thin d x = sqrt(2)"),
])
def test_latex_math(latex, typst):
    assert latex_math(latex) == typst
    compiled(typst)


@pytest.mark.parametrize("latex, message", [
    ("\\mapstoo", "No Typst for the LaTeX command \\\\mapstoo"),
    ("{x", "Unbalanced braces"),
    ("x}", "Unbalanced braces"),
    ("\\text{x", "Unbalanced braces"),
    ("\\text x", "Missing the braces"),
    ("\\frac{a}", "Missing an argument"),
])
def test_latex_math_raises(latex, message):
    with pytest.raises(ValueError, match=message):
        latex_math(latex)
