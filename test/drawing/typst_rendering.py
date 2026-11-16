"""Tests for the Typst/CeTZ drawing backend."""

import os

import pytest

from discopy.monoidal import Ty, Box
from discopy.symmetric import Ty as STy, Swap
from discopy.rigid import Ty as RTy, Cup, Cap, Id
from discopy.braided import Braid
from discopy.frobenius import Ty as FTy, Spider
from discopy.quantum.zx import Z, X, H

TYPST_FOLDER = "test/drawing/typst/"


def _normalize_typst(text):
    """Strip trailing whitespace from each line for comparison."""
    return "\n".join(line.rstrip() for line in text.splitlines())


def compiled(source, **params):
    """Compile Typst source, or skip when CeTZ cannot be downloaded."""
    typst = pytest.importorskip("typst")
    try:
        return typst.compile(source.encode(), **params)
    except typst.TypstError as error:
        if "failed to download package" in str(error):
            pytest.skip("CeTZ cannot be downloaded.")
        raise


def drawn(diagram, **params):
    """Draw with Typst, or skip when CeTZ cannot be downloaded."""
    typst = pytest.importorskip("typst")
    try:
        return diagram.draw(format="typst", **params)
    except typst.TypstError as error:
        if "failed to download package" in str(error):
            pytest.skip("CeTZ cannot be downloaded.")
        raise


def typst_and_compare(file):
    """Decorator that compares to_typst() output against a reference."""
    def decorator(func):
        def wrapper():
            diagram = func()
            source = diagram.to_typst().render()
            true_path = os.path.join(TYPST_FOLDER, file)
            if not os.path.exists(true_path):
                with open(true_path, "w") as f:
                    f.write(source)
                return
            with open(true_path) as f:
                assert _normalize_typst(source) == _normalize_typst(f.read())
        return wrapper
    return decorator


@typst_and_compare("box.typ")
def test_box_to_typst():
    x, y = Ty("x"), Ty("y")
    return Box("f", x, y)


@typst_and_compare("composition.typ")
def test_composition_to_typst():
    x, y = Ty("x"), Ty("y")
    f = Box("f", x, y)
    g = Box("g", y, x)
    return f >> g


@typst_and_compare("tensor.typ")
def test_tensor_to_typst():
    x, y = Ty("x"), Ty("y")
    f = Box("f", x, y)
    g = Box("g", y, x)
    return f @ g


@typst_and_compare("swap.typ")
def test_swap_to_typst():
    x = STy("x")
    return Swap(x, x)


@typst_and_compare("braid.typ")
def test_braid_to_typst():
    x = Ty("x")
    return Braid(x, x)


@typst_and_compare("cup.typ")
def test_cup_to_typst():
    n = RTy("n")
    return Cup(n, n.r)


@typst_and_compare("cap.typ")
def test_cap_to_typst():
    n = RTy("n")
    return Cap(n.r, n)


@typst_and_compare("spider.typ")
def test_spider_to_typst():
    x = FTy("x")
    # Explicit colour: the `unfuse` doctest sets `Spider.color` globally.
    return Spider(2, 1, x, color="black")


@typst_and_compare("snake_equation.typ")
def test_snake_equation_to_typst():
    n = RTy("n")
    return Id(n.r) >> Cap(n.r, n) @ Id(n.r) >> Id(n.r) @ Cup(n, n.r)


@typst_and_compare("zx_z_spider.typ")
def test_zx_z_to_typst():
    return Z(2, 1)


@typst_and_compare("zx_x_spider.typ")
def test_zx_x_to_typst():
    return X(2, 1)


@typst_and_compare("hadamard.typ")
def test_hadamard_to_typst():
    return H


def test_to_typst_returns_document():
    """to_typst() returns a Document AST with correct structure."""
    x, y = Ty("x"), Ty("y")
    f = Box("f", x, y)
    doc = f.to_typst()
    source = doc.render()
    assert 'import "@preview/cetz' in source
    assert 'canvas(' in source
    assert "import draw: *" in source


def test_to_typst_includes_mathematical_labels():
    """Box labels with LaTeX syntax remain intact."""
    x, y = Ty("x"), Ty("y")
    f = Box("$f$", x, y)
    source = f.to_typst().render()
    assert "[$f$]" in source


def test_typst_source_is_deterministic():
    """Multiple calls give identical output."""
    x, y = Ty("x"), Ty("y")
    f = Box("f", x, y)
    assert f.to_typst().render() == f.to_typst().render()


def test_typst_backend_in_drawing_init():
    """The Typst backend is exported from discopy.drawing."""
    from discopy.drawing import Typst
    assert Typst is not None


def test_typst_compilation():
    """Compilation to SVG works when typst-py is installed."""
    try:
        import typst  # noqa: F401
    except ImportError:
        pytest.skip("typst package not installed")
    from discopy.monoidal import Ty, Box
    x, y = Ty("x"), Ty("y")
    f = Box("f", x, y)
    svg = drawn(f, show=False)
    assert isinstance(svg, bytes)
    assert svg.startswith(b"<svg") or svg.startswith(b"<?xml")


def test_typst_output_to_file(tmp_path):
    """``output`` writes Typst source for ``.typ`` and SVG bytes otherwise."""
    x, y = Ty("x"), Ty("y")
    f = Box("f", x, y)
    source = f.draw(format="typst", path=tmp_path / "f.typ", show=False)
    assert (tmp_path / "f.typ").read_text() == source
    assert 'import "@preview/cetz' in source
    svg = drawn(f, path=tmp_path / "f.svg", show=False)
    assert (tmp_path / "f.svg").read_bytes() == svg


def test_typst_colors_and_labels():
    """Colours, transparent fills and spider labels reach the CeTZ source."""
    x = FTy("x")
    # A phase gives the spider a label, drawn white on black, black on red.
    dark = Spider(2, 1, x, 0.5, color="black").to_typst().render()
    assert 'fill: rgb("#ffffff")' in dark
    light = Spider(2, 1, x, 0.5, color="red").to_typst().render()
    assert 'rgb("#e8a5a5")' in light
    assert 'fill: rgb("#000000")' in light
    transparent = Box("f", x, x, color="none").to_typst().render()
    assert "fill: none" in transparent


def test_typst_color_helpers():
    """``format_color`` and ``color_expr`` handle names, hexcodes and none."""
    from discopy.drawing.backend import Typst
    assert Typst.format_color(None) == "none"
    assert Typst.format_color("#123456") == "#123456"
    assert Typst.format_color("red") == "#e8a5a5"
    assert Typst.color_expr("none") == "none"
    assert Typst.color_expr("#123456") == 'rgb("#123456")'
    assert Typst.color_expr("white") == "white"


def test_wire_bezier_points():
    """A wire only bends when it is neither horizontal nor vertical."""
    from discopy.drawing.backend import wire_bezier_points
    assert wire_bezier_points((0, 0), (0, 1), True) is None
    assert wire_bezier_points((0, 0), (1, 0), True) is None
    assert wire_bezier_points((0, 0), (3, 3), True) == ((2, 0), (3, 1))
    assert wire_bezier_points((0, 0), (3, 3), False) == ((0, 2), (1, 3))


def test_typst_fontsize_and_text_colour():
    """A non-default fontsize scales the text by its ratio to the default."""
    x, y = Ty("x"), Ty("y")
    source = Box("f", x, y).to_typst(fontsize=24).render()
    assert 'text("f", size: 2.4em)' in source
    assert compiled(source, format="svg")


def test_typst_ast_nodes():
    """The AST nodes render the Typst syntax they stand for."""
    from discopy.drawing.typst_ast import Import, TypstNode
    assert Import("cetz", members=["canvas"]).render() \
        == '#import "cetz": canvas'
    assert Import("fletcher", alias="f").render() == '#import "fletcher" as f'
    assert Import("cetz").render() == '#import "cetz"'
    with pytest.raises(NotImplementedError):
        TypstNode().render()


def test_typst_show_returns_ipython_svg():
    """With no path and ``show``, ``output`` hands IPython an SVG to display."""
    pytest.importorskip("IPython")
    x, y = Ty("x"), Ty("y")
    svg = drawn(Box("f", x, y), show=True)
    assert type(svg).__name__ == "SVG"


def test_typst_bezier_ends_where_the_curve_does():
    """A bezier of the backend is start, two controls, end; CeTZ takes the
    two ends first."""
    from discopy.drawing.backend import Typst
    backend = Typst()
    backend.draw_bezier([(0, 1), (0, 0.5), (1, 0.5), (1, 0)])
    assert backend.body[0].render() == (
        'bezier((0, 1), (1, 0), (0, 0.5), (1, 0.5), '
        'stroke: rgb("#000000") + 1pt)')
    x = Ty("x")
    strands = [line for line in Braid(x, x).to_typst().render().splitlines()
               if "bezier" in line]
    assert any("bezier((1.5, 1), (0.5, 0)," in line for line in strands)


def test_typst_names_are_not_markup():
    """A name is a string literal, whatever characters it holds."""
    x = Ty("x")
    source = Box('f]#x "y"', x, x).to_typst().render()
    assert r'"f]#x \"y\""' in source
    assert compiled(source, format="svg")


def test_typst_colours_given_as_numbers():
    from discopy.drawing.backend import Typst
    assert Typst.format_color([1, 0, 0]) == "#ff0000"
    assert Typst.format_color("#00ff00") == "#00ff00"
    assert Typst.format_color(None) == Typst.format_color("none") == "none"
    assert Typst.format_color("tab:blue") == "#1f77b4"
    assert Typst.format_color("0.5") == "#808080"
    assert Typst.format_color("white") == "#ffffff"


def test_to_typst_leaves_the_drawing_as_it_was():
    x, y = Ty("x"), Ty("y")
    drawing = Box("f", x, y).to_drawing()
    nodes, source = set(drawing.positions), drawing.to_typst().render()
    assert set(drawing.positions) == nodes
    assert drawing.to_typst().render() == source


def test_typst_frames_dual_rails():
    """A dual rail cup is drawn with its fold inside the frame."""
    from discopy import ribbon
    x = ribbon.Ty("x")
    drawing = ribbon.Braid(x, x).trace().to_ribbons().to_drawing()
    framed = drawing.relabel_nodes()
    framed.add_box_corners()
    framed.frame_dual_rail()
    source = drawing.to_typst().render()
    assert f"({framed.width:g}, {framed.height:g})" in source


def test_typst_draws_the_nodes_of_a_controlled_gate():
    """A CX has a black control dot and a circled plus as its target."""
    from discopy.quantum.gates import CX
    source = CX.to_typst().render()
    circles = [line for line in source.splitlines() if "circle(" in line]
    assert any('fill: rgb("#000000")' in line for line in circles)
    assert any('fill: rgb("#ffffff"), stroke: rgb("#000000")' in line
               for line in circles)
    assert compiled(source, format="svg")


def test_typst_keeps_the_arrow_of_a_measurement():
    from discopy.quantum.circuit import Measure
    source = Measure().to_typst().render()
    assert source.count('mark: (end: ">", fill: black)') == 1
    assert compiled(source, format="svg")


def test_typst_import_quotes_its_path():
    from discopy.drawing.typst_ast import Import
    assert Import('a"b').render() == '#import "a' + chr(92) + '"b"'


def test_typst_from_every_way_of_drawing(tmp_path):
    """A drawing and an equation draw with Typst as a diagram does."""
    from io import BytesIO, StringIO
    from discopy.monoidal import Equation
    x, y = Ty("x"), Ty("y")
    f = Box("f", x, y)
    paths = [tmp_path / name for name in ("f.typ", "d.typ", "e.typ")]
    source = f.draw(format="typst", path=paths[0])
    assert f.to_drawing().draw(format="typst", path=paths[1]) == source
    assert "canvas" in Equation(f, f).draw(format="typst", path=paths[2])
    binary, text = BytesIO(), StringIO()
    svg = drawn(f, path=binary, show=False)
    assert binary.getvalue() == svg
    assert drawn(f, path=text, show=False).decode() == text.getvalue()


def test_typst_wire_bends_as_matplotlib_does():
    """Out of the source when asked to, into the target otherwise."""
    from discopy.drawing.backend import Typst, wire_bezier_points
    for bend_out in (False, True):
        first, _ = wire_bezier_points((0, 3), (3, 0), bend_out)
        assert first == ((2, 3) if bend_out else (0, 1))
    backend = Typst()
    backend.draw_wire((0, 3), (3, 0))
    backend.draw_wire((0, 3), (3, 0), style="->")
    assert backend.body[0].render().startswith("bezier(")
    assert backend.body[1].render().startswith("line(")


def test_typst_strings_and_imports():
    from discopy.drawing.typst_ast import Import, Str
    assert Str("a\nb\tc").render() == '"a' + chr(92) + 'nb' + chr(92) + 'tc"'
    with pytest.raises(ValueError):
        Import("x", members=["a"], alias="y").render()


def test_typst_array_colours_and_spider_sizes():
    import numpy as np
    from discopy.drawing.backend import Typst
    backend = Typst()
    backend.draw_polygon(
        (0, 0), (1, 0), (1, 1), facecolor=np.array([1., 0, 0]),
        edgecolor=np.array([0, 0, 1.]))
    assert 'fill: rgb("#ff0000"), stroke: rgb("#0000ff")'\
        in backend.body[0].render()
    assert "size: 2.4em" in Z(1, 2, 0.5).to_typst(
        fontsize_spiders=24).render()


def test_typst_labels_that_are_numbers():
    """A ket is labelled by its digits, which are integers."""
    from discopy.quantum.gates import Ket
    source = Ket(0, 1).to_typst().render()
    assert '"0"' in source and '"1"' in source
    assert compiled(source, format="svg")


def test_typst_draws_the_symbol_of_an_equation_as_its_label_alone():
    from discopy.monoidal import Equation
    x = Ty("x")
    source = Equation(Box("f", x, x), Box("g", x, x)).to_drawing()\
        .to_typst().render()
    assert 'text("=", fill: rgb("#000000")' in source
    assert "circle(" not in source
    assert compiled(source, format="svg")
