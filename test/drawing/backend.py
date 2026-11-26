import os
import re
from io import BytesIO

import numpy as np
import pytest
from pytest import mark, raises

from discopy.drawing.backend import TikZ
from discopy.monoidal import Ty, Box
from discopy.braided import Braid
from discopy.frobenius import Ty as FTy, Spider
from discopy.quantum.zx import Z, X, H
from discopy.rigid import Ty as RTy, Cup, Cap, Id
from discopy.symmetric import Ty as STy, Swap


def test_draw_raises_on_relabeled_box(tmp_path):
    """ Drawing a relabeled box over an existing baseline raises. """
    x, y = Ty('x'), Ty('y')
    path = tmp_path / "box.svg"
    Box('f', x, y).draw(doctest=path, show=False)
    with raises(ValueError, match="Drawing differs"):
        Box('g', x, y).draw(doctest=path, show=False)


def test_tikz_doctest_compares_against_its_baselines(tmp_path):
    f, path = Box('f', Ty('x'), Ty('y')), tmp_path / "box.tikz"
    params = dict(doctest=str(path), to_tikz=True, use_tikzstyles=True)
    f.draw(**params)
    f.draw(**params)
    assert sorted(file.name for file in tmp_path.iterdir())\
        == ["box.tikz", "box.tikzstyles"]
    for baseline in [path, path.with_suffix(".tikzstyles")]:
        expected = baseline.read_text()
        baseline.write_text("not tikz\n")
        with raises(ValueError, match="Drawing differs"):
            f.draw(**params)
        baseline.write_text(expected)


def test_draw_discard_grounds_every_wire(tmp_path):
    """ A `Discard` on several objects draws one ground per wire, as the
    tensor of single-wire discards does, see rel-int/discopy-next#239. """
    from discopy.quantum.circuit import Discard, bit, qubit

    def strokes(diagram, name):
        path = tmp_path / f"{name}.tikz"
        diagram.draw(path=str(path), to_tikz=True)
        return sum(line.startswith("\\draw")
                   for line in path.read_text().splitlines())

    assert strokes(Discard(bit @ qubit), "multi") \
        == strokes(Discard(bit) @ Discard(qubit), "tensored")


def test_tikz_wire_separates_its_style_keys():
    tikz = TikZ()
    tikz.draw_wire((0, 0), (1, 2), style="->")
    assert "->, looseness=" in tikz.edgelayer[-1]


def test_tikz_twist_in_an_xcolor_name(tmp_path):
    from discopy import ribbon
    path = tmp_path / "twist.tikz"
    for colour, back in [("RoyalPurple", "RoyalPurple!60!black"),
                         ("red!50", "red!50!white!60!black")]:
        ribbon.Diagram.twist(ribbon.Ty('x')).to_ribbons(colour=colour)\
            .draw(path=str(path), to_tikz=True)
        tikz = path.read_text()
        assert f"fill={colour}," in tikz and f"fill={back}," in tikz


def test_tikzstyle_of_a_hexcode_has_no_hash(tmp_path):
    path = tmp_path / "box.tikz"
    Box('f', Ty('x'), Ty('x'), color='#336699').draw(
        path=str(path), to_tikz=True, use_tikzstyles=True)
    tikz, styles = path.read_text(), path.with_suffix(".tikzstyles")\
        .read_text()
    assert "style=336699_box" in tikz
    assert "\\tikzstyle{336699_box}" in styles
    assert "#" not in tikz + styles


def test_tikz_spider_styles_are_keyed_by_name_fill_and_shape(tmp_path):
    from discopy.monoidal import Equation
    x, path = Ty('x'), tmp_path / "spiders.tikz"
    red = Box('f(x)', x, x, draw_as_spider=True, color='red')
    blue = Box('g(x)', x, x, draw_as_spider=True, color='blue',
               shape='rectangle')
    Equation(red, blue).draw(
        path=str(path), to_tikz=True, use_tikzstyles=True)
    tikz, styles = path.read_text(), path.with_suffix(".tikzstyles")\
        .read_text()
    assert styles.splitlines()[1:4] == [
        "\\tikzstyle{symbol_red_circle}="
        f"[fill={TikZ.format_color('red')}, shape=circle]",
        "\\tikzstyle{symbol_none_circle}=[fill=none, shape=circle]",
        "\\tikzstyle{symbol_blue_rectangle}="
        f"[fill={TikZ.format_color('blue')}, shape=rectangle]"]
    for name in ["red_circle", "none_circle", "blue_rectangle"]:
        assert f"[style=symbol_{name}, " in tikz


def test_tikzstyles_sit_beside_their_drawing(tmp_path):
    folder = tmp_path / "dir.v2"
    folder.mkdir()
    for name in ["diagram.tikz", "diagram"]:
        Box('f', Ty('x'), Ty('y')).draw(
            path=folder / name, to_tikz=True, use_tikzstyles=True)
    assert sorted(file.name for file in tmp_path.iterdir()) == ["dir.v2"]
    assert sorted(file.name for file in folder.iterdir())\
        == ["diagram", "diagram.tikz", "diagram.tikzstyles"]


@mark.parametrize("mode", ["w", "wb"])
def test_svg_to_an_open_file_has_the_dark_mode_style(tmp_path, mode):
    f, path = Box('f', Ty('x'), Ty('y')), tmp_path / "box.svg"
    with open(path, mode) as file:
        f.draw(path=file, format="svg", show=False)
    assert path.read_text(encoding="utf-8") == f.to_svg()


def test_svg_to_a_bytes_buffer_has_the_dark_mode_style():
    f, buffer = Box('f', Ty('x'), Ty('y')), BytesIO()
    f.draw(path=buffer, format="svg", show=False)
    assert buffer.getvalue().decode("utf-8") == f.to_svg()


@mark.parametrize("use_tikzstyles", [False, True])
@mark.parametrize("colour", [
    "tab:blue", "C0", "0.5", "navy", (.1, .2, .3), [.1, .2, .3],
    np.array([.1, .2, .3])])
def test_tikz_spells_a_matplotlib_only_colour_by_its_rgb(
        tmp_path, colour, use_tikzstyles):
    """ A colour that matplotlib reads and xcolor does not is written as its
    RGB values, see rel-int/discopy-next#466. """
    from matplotlib.colors import to_rgb
    from discopy.monoidal import Equation
    red, green, blue = (round(255 * x) for x in to_rgb(colour))
    rgb = f"{{rgb,255: red,{red}; green,{green}; blue,{blue}}}"
    x, path = Ty('x'), tmp_path / "colour.tikz"
    Equation(Box('f', x, x, color=colour),
             Box('g', x, x, color=colour, draw_as_spider=True)).draw(
        path=str(path), to_tikz=True, use_tikzstyles=use_tikzstyles)
    styles = path.with_suffix(".tikzstyles")
    tikz = path.read_text() + (styles.read_text() if use_tikzstyles else "")
    assert tikz.count(f"fill={rgb}") == 2
    assert all(re.fullmatch(r"\w+", name)
               for name in re.findall(r"style=([^,\]]*)", tikz))


@mark.parametrize("colour", [(.1, .2, .3), [.1, .2, .3], np.array([.1, .2, .3])])
def test_matplotlib_draws_an_unhashable_colour(colour):
    """ A colour matplotlib reads need not be hashable, e.g. a list or an
    array, see rel-int/discopy-next#482. """
    from discopy.monoidal import Equation
    from discopy.quantum.gates import Bra, Ket
    x, ket, bra, buffer = Ty('x'), Ket(0), Bra(1), BytesIO()
    ket.color = bra.color = colour
    Equation(
        Box('f', x, x, color=colour),
        Box('g', x, x, color=colour, draw_as_spider=True), ket, bra).draw(
        path=buffer, format="svg", show=False)
    assert buffer.getvalue().decode("utf-8").count("fill: #1a334c") == 4


@mark.parametrize("colour", ["purple", "lightgray", "RoyalPurple", "red!50"])
def test_tikz_keeps_an_xcolor_name(colour):
    assert TikZ.format_color(colour) == colour


def test_triangle_flips_with_dagger_and_transpose_not_conjugate():
    from discopy import pivotal
    from discopy.drawing.backend import Backend
    x = pivotal.Ty('x')
    f = pivotal.Box('f', x, x, shape="triangle")

    def apex_down(diagram):
        drawing = diagram.to_drawing()
        drawing.add_box_corners()
        node, = (n for n in drawing.positions if n.kind == "box")
        base, _, apex = Backend.triangle_corners(drawing.positions, node)
        return apex.y < base.y
    assert [apex_down(g) for g in (f, f.r, f.dagger(), f.conjugate())]\
        == [True, False, False, True]


def test_controlled_triangle_draws(tmp_path):
    from discopy.quantum.gates import Controlled, QuantumGate
    from discopy.quantum.circuit import qubit
    gate = QuantumGate('U', qubit, qubit, [1, 0, 0, 1], shape="triangle")
    Controlled(gate).draw(path=str(tmp_path / "controlled.png"), show=False)


def test_tikz_background_has_the_edge_of_its_face(tmp_path):
    """ The background is white on white: LaTeX draws no black frame. """
    for use_tikzstyles, background in [
            (False, "\\draw [-, fill={white}, draw={white}] "),
            (True, "\\draw [style=white_white_box] ")]:
        path = tmp_path / "box.tikz"
        Box('f', Ty('x'), Ty('x')).draw(
            path=str(path), to_tikz=True, use_tikzstyles=use_tikzstyles)
        polygons = [line for line in path.read_text().splitlines()
                    if line.startswith("\\draw [") and "fill" in line
                    or line.startswith("\\draw [style=")]
        assert polygons[0].startswith(background)
        assert all(not line.startswith(background) for line in polygons[1:])


def test_tikz_spider_label_is_not_scaled_with_its_node(tmp_path):
    from discopy.monoidal import Equation
    x, path = Ty('x'), tmp_path / "equation.tikz"
    Equation(Box('f', x, x), Box('g', x, x)).draw(
        path=str(path), to_tikz=True)
    symbol, = [line for line in path.read_text().splitlines()
               if "{=}" in line]
    assert "scale" not in symbol and "label={center:{=}}" in symbol
    assert symbol.endswith("{};")


def test_tikz_escapes_a_label_outside_mathematics(tmp_path):
    from discopy.monoidal import Equation
    x, path = Ty('x_1'), tmp_path / "labels.tikz"
    spider = Box('50% & #1', x, x, draw_as_spider=True)
    Equation(Box('feed_forward', x, x), spider,
             symbol="$a_1 \\mapsto b$").draw(path=str(path), to_tikz=True)
    tikz = path.read_text()
    for label in ["{feed\\_forward}", "{x\\_1}", "{$a_1 \\mapsto b$}",
                  "label={center:{50\\% \\& \\#1}}"]:
        assert label in tikz
    assert TikZ.escape("(n >> s) << n") == "(n >{}> s) <{}< n"
    assert TikZ.escape(">>> a\\_b \\$x_y\\$") == ">{}>{}> a\\_b \\$x\\_y\\$"


def test_typst_reads_a_label_between_dollars_as_latex(tmp_path):
    from pytest import importorskip
    from discopy.drawing.backend import Typst
    from discopy.monoidal import Equation
    importorskip("typst")
    assert Typst.math_label("$a$ and $b$") == ("$a$ and $b$", False)
    assert Typst.math_label("$\\text{\\$5}$") == ('"$5"', True)
    f, g = Box('f', Ty('x'), Ty('y')), Box('$\\phi_1$', Ty('x'), Ty('y'))
    equation = Equation(f, g, symbol="$\\mapsto$")
    equation.draw(format="typst", path=tmp_path / "a.svg")
    source = equation.to_drawing().to_typst().render()
    assert "[$arrow.r.bar$]" in source and "[$phi.alt_(1)$]" in source
    with raises(ValueError, match="No Typst for the LaTeX command"):
        Equation(f, g, symbol="$\\nosuch$").to_drawing().to_typst()


def test_typst_draws_its_labels_over_its_boxes_and_wires():
    x, y = Ty('x'), Ty('a_long_wire_name')
    diagram = Box('hidden', x, y) >> Box('query', y, x) @ Box('k', Ty(), x)
    lines = diagram.to_typst().render().splitlines()
    labels = [i for i, line in enumerate(lines) if "content(" in line]
    shapes = [i for i, line in enumerate(lines)
              if line.lstrip().startswith(("line(", "bezier("))]
    assert labels and max(shapes) < min(labels)


def test_typst_fills_regions_frames_and_ribbons():
    pytest.importorskip("typst")
    from discopy import ribbon
    from discopy.egraph import Join
    from discopy.markov import Ty as MTy, Box as MBox
    from discopy.drawing.backend import Typst
    x, y = MTy('x'), ribbon.Ty('y')
    join = Join(MBox('f', x, x), MBox('g', x, x))
    ribbons = (ribbon.Braid(y, y) >> ribbon.Braid(y, y)).trace(
        n=2).to_ribbons()
    for diagram, bands in [(join, False), (ribbons, True)]:
        drawing = diagram.to_drawing().framed()
        cells = len(list(Typst.region_cells(drawing)))
        shapes = Typst().draw(drawing).to_document().render().count(
            "merge-path")
        assert cells and (shapes > cells if bands else shapes == cells)
        diagram.draw(format="typst", path=None, show=False)


@mark.parametrize("shape", ["rectangle", "triangle_up", "triangle_down",
                            "plus"])
def test_typst_draws_the_shape_of_a_spider(shape):
    pytest.importorskip("typst")
    x = Ty('x')

    def source(shape):
        return Box('', x, x @ x, draw_as_spider=True, shape=shape)\
            .to_typst().render().splitlines()
    calls, discs = source(shape), source("circle")
    primitive = "rect(" if shape == "rectangle" else "line("
    added = [call.strip() for call in calls if call not in discs]
    assert "circle(" not in "".join(calls)
    assert len(added) == (2 if shape == "plus" else 1)
    assert all(call.startswith(primitive) for call in added)
    colour = "stroke" if shape == "plus" else "fill"
    assert all(f'{colour}: rgb("#000000")' in call for call in added)
    Box('', x, x @ x, draw_as_spider=True, shape=shape).draw(
        format="typst", path=None, show=False)


@mark.parametrize("name", ["box.typ", "box.svg"])
def test_typst_doctest_compares_against_its_baseline(tmp_path, name):
    pytest.importorskip("typst")
    f, path = Box('f', Ty('x'), Ty('y')), tmp_path / name
    f.draw(format="typst", doctest=path)
    baseline = path.read_bytes()
    f.draw(format="typst", doctest=path)
    assert [file.name for file in tmp_path.iterdir()] == [name]
    with raises(ValueError, match="Drawing differs"):
        Box('g', Ty('x'), Ty('y')).draw(format="typst", doctest=path)
    assert path.read_bytes() == baseline
    assert f.draw(format="typst", path=path, compare=True) is not None
    with raises(ValueError, match="ends in .typ or .svg"):
        f.draw(format="typst", doctest=tmp_path / "box.png")


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


def test_left_to_right_follows_the_category_and_keeps_arrows(tmp_path):
    from discopy.monoidal import Bubble, Diagram
    from discopy.quantum.circuit import Measure
    from discopy.utils import factory

    @factory
    class Body(Diagram):
        draw_left_to_right = True

    class Primitive(Box, Body):
        pass

    x = Ty('x')
    f = Primitive('f', x, x)
    body = f >> Primitive('g', x, x @ x)
    assert body.to_svg() == body.to_svg(left_to_right=True)\
        != body.to_svg(left_to_right=False)
    assert Bubble(f[::-1], dom=x, cod=x).to_drawing().asymmetry() == 0.125

    path = tmp_path / "measure.tikz"
    Measure().draw(to_tikz=True, left_to_right=True, path=str(path))
    assert "->" in path.read_text()


def test_bubble_of_args_drawn_in_different_directions():
    from discopy.monoidal import Bubble, Diagram
    from discopy.utils import factory

    @factory
    class Body(Diagram):
        draw_left_to_right = True

    class Primitive(Box, Body):
        pass

    x = Ty('x')
    with raises(ValueError):
        Bubble(Box('f', x, x), Primitive('g', x, x)).to_drawing()


def test_bubble_of_coloured_args_drawn_in_another_direction():
    from discopy.monoidal import Bubble, Colour, Diagram, Wire
    from discopy.utils import factory

    @factory
    class Body(Diagram):
        draw_left_to_right = True

    class Primitive(Box, Body):
        pass

    x = Ty(Wire('x', Colour('red'), Colour('blue')))
    bubble = Bubble(Primitive('f', x, x), Primitive('g', x, x))
    assert len(bubble.to_drawing().boxes[0].nested) == 2


def test_left_to_right_labels_legend_and_dagger():
    from discopy.config import BOX_DRAWING_ATTRIBUTES
    from discopy.drawing.backend import Backend
    from discopy.monoidal import Bubble, Colour, Diagram, Wire
    from discopy.utils import factory, text_width

    @factory
    class Body(Diagram):
        draw_left_to_right = True

    class Primitive(Box, Body):
        pass

    name = 'a very long function name'
    x = Ty(Wire('x', Colour('red'), Colour('blue')))
    long = Primitive(name, x, x).to_drawing()
    assert long.height >= text_width(name) + 1 > 1 == long.width
    assert sorted(Backend.region_colours(
        Bubble(Primitive('f', x, x), dom=Ty(), cod=Ty()).to_drawing())
    ) == ['blue', 'red']
    spider = Primitive(name, x, x, draw_as_spider=True).to_drawing()
    assert spider.height == 1

    class Recorder(Backend):
        def __init__(self):
            super().__init__()
            self.heights = {}

        def draw_text(self, text, i, j, **params):
            self.heights[text] = j

        def output(self, path=None, show=True, **params):
            pass

    y = Ty('y')
    bubble = Bubble(
        Primitive('f', y, y), Primitive('g', y, y), frame_colour="red")
    assert bubble.dagger().frame_colour == "red"
    default = BOX_DRAWING_ATTRIBUTES['frame_colour']
    BOX_DRAWING_ATTRIBUTES['frame_colour'] = lambda _: "blue"
    try:
        assert Bubble(Primitive('f', y, y)).frame_colour == "blue"
    finally:
        BOX_DRAWING_ATTRIBUTES['frame_colour'] = default
    up, down = Recorder(), Recorder()
    bubble.to_drawing().draw(backend=up)
    bubble.to_drawing().dagger().draw(backend=down)
    assert up.heights['f'] > up.heights['g']
    assert down.heights['f'] < down.heights['g']


def test_bubble_drawn_in_another_direction_fits_its_framed_args():
    from discopy.ribbon import DualRailCup, Ty as RTy

    class Across(Box):
        draw_left_to_right = True

    x = RTy('x')
    arg = DualRailCup(x @ x, x.r @ x.r)
    drawing = Across.from_args(arg, dom=Ty(), cod=Ty()).to_drawing()
    nested, = drawing.boxes[0].nested
    framed = arg.to_drawing().framed()
    assert framed.width != arg.to_drawing().width
    assert nested.width == framed.width
    assert drawing.height >= nested.width + 1


def test_left_to_right_separates_args_and_lifts_labels():
    from discopy.drawing.backend import Backend
    from discopy.monoidal import Bubble, Diagram
    from discopy.utils import factory

    @factory
    class Body(Diagram):
        draw_left_to_right = True

    class Primitive(Box, Body):
        pass

    class Recorder(Backend):
        def __init__(self):
            super().__init__()
            self.beziers, self.labels, self.wires = [], [], []

        def draw_bezier(self, points, adaptive=True):
            self.beziers.append(([tuple(p) for p in points], adaptive))

        def draw_text(self, text, i, j, **params):
            if params.get("verticalalignment") in ("top", "bottom"):
                self.labels.append(j)
            else:
                self.boxes[text] = j

        def output(self, path=None, show=True, **params):
            pass

    Recorder.boxes = {}
    x = Ty('x')
    recorder = Recorder()
    Bubble(Primitive('f', x, x), Primitive('g', x, x)).draw(backend=recorder)
    separators = [ps for ps, adaptive in recorder.beziers if not adaptive
                  and len({y for _, y in ps}) == 1]
    assert len(separators) == 1
    assert recorder.boxes['g'] < separators[0][0][1] < recorder.boxes['f']

    recorder = Recorder()
    (Primitive('f', x, x) >> Primitive('g', x, x)).draw(backend=recorder)
    wire_y = recorder.beziers[0][0][0][1]
    assert recorder.labels and all(j > wire_y for j in recorder.labels)


def test_left_to_right_label_padding_and_contrast():
    from discopy.drawing.backend import Backend
    from discopy.monoidal import Colour, Diagram, Wire
    from discopy.utils import factory

    @factory
    class Body(Diagram):
        draw_left_to_right = True

    class Primitive(Box, Body):
        pass

    class Recorder(Backend):
        def __init__(self):
            super().__init__()
            self.labels, self.wire_y = [], None

        def draw_bezier(self, points, adaptive=True):
            self.wire_y = points[0][1]

        def draw_text(self, text, i, j, **params):
            if params.get("verticalalignment") in ("top", "bottom"):
                self.labels.append((j, params["color"]))

        def output(self, path=None, show=True, **params):
            pass

    x = Ty(Wire('x', Colour('white'), Colour('black')))
    recorder = Recorder()
    Primitive('f', x, x).draw(backend=recorder, textpad=(.5, .0625))
    assert recorder.labels
    for j, color in recorder.labels:
        assert j >= recorder.wire_y + .5 - 1e-9
        assert color == "black"


def test_left_to_right_labels_above_wires_in_a_daggered_box():
    from discopy.drawing.backend import Backend
    from discopy.monoidal import Bubble, Colour, Diagram, Wire
    from discopy.utils import factory

    @factory
    class Body(Diagram):
        draw_left_to_right = True

    class Primitive(Box, Body):
        pass

    class Recorder(Backend):
        def __init__(self):
            super().__init__()
            self.labels, self.wires = [], []

        def draw_bezier(self, points, adaptive=True):
            if len({y for _, y in points}) == 1:
                self.wires.append(points[0][1])

        def draw_text(self, text, i, j, **params):
            if params.get("verticalalignment") in ("top", "bottom"):
                self.labels.append((j, params["color"]))

        def output(self, path=None, show=True, **params):
            pass

    x = Ty(Wire('x', Colour('white'), Colour('black')))
    bubble = Bubble(Primitive('f', x, x), dom=Ty(), cod=Ty())
    recorder = Recorder()
    bubble.to_drawing().dagger().draw(backend=recorder)
    assert recorder.labels
    for j, color in recorder.labels:
        assert any(j - .125 <= y < j for y in recorder.wires)
        assert color == "white"


def test_labels_of_args_turned_back_top_to_bottom():
    from discopy.drawing.backend import Backend
    from discopy.monoidal import Colour, Wire

    class Across(Box):
        draw_left_to_right = True

    class Recorder(Backend):
        def __init__(self):
            super().__init__()
            self.labels, self.wires = [], []

        def draw_bezier(self, points, adaptive=True):
            if len({x for x, _ in points}) == 1:
                self.wires.append(points[0][0])

        def draw_text(self, text, i, j, **params):
            if params.get("verticalalignment") in ("top", "bottom"):
                self.labels.append((i, params["color"]))

        def output(self, path=None, show=True, **params):
            pass

    x = Ty(Wire('x', Colour('white'), Colour('black')))
    recorder = Recorder()
    Across.from_args(Box('f', x, x), dom=Ty(), cod=Ty()).draw(
        backend=recorder)
    assert recorder.labels and recorder.wires
    for i, color in recorder.labels:
        assert any(0 < i - wire <= .125 for wire in recorder.wires)
        assert color == "white"


def test_label_contrast_through_three_turns_reflected_outside():
    from discopy.drawing.backend import Backend
    from discopy.monoidal import Bubble, Colour, Diagram, Wire
    from discopy.utils import factory

    @factory
    class Body(Diagram):
        draw_left_to_right = True

    class Primitive(Box, Body):
        pass

    class Across(Box):
        draw_left_to_right = True

    class Recorder(Backend):
        def __init__(self):
            super().__init__()
            self.colours = []

        def draw_text(self, text, i, j, **params):
            if params.get("verticalalignment") in ("top", "bottom"):
                self.colours.append(params["color"])

        def output(self, path=None, show=True, **params):
            pass

    x = Ty(Wire('x', Colour('white'), Colour('black')))
    recorder = Recorder()
    inside = Across.from_args(Bubble(Primitive('f', x, x)))
    Bubble(inside).to_drawing().dagger().draw(backend=recorder)
    assert recorder.colours and set(recorder.colours) == {"white"}


def test_left_to_right_labels_stand_on_their_point_in_tikz(tmp_path):
    from discopy.monoidal import Diagram
    from discopy.utils import factory

    @factory
    class Body(Diagram):
        draw_left_to_right = True

    class Primitive(Box, Body):
        pass

    path = tmp_path / "ltr.tikz"
    Primitive('f', Ty('x'), Ty('x')).draw(
        to_tikz=True, fontsize_types=2, path=str(path))
    tikz = path.read_text()
    assert "anchor=south west" in tikz and "anchor=north west" not in tikz


def test_label_contrast_through_two_turns_reflected():
    from types import SimpleNamespace
    from discopy.drawing.backend import Matplotlib, Point, Transposed

    graph = SimpleNamespace(width=1, height=1)
    outside = Transposed(Matplotlib(), Point(0, 0), graph)
    inside = Transposed(outside, Point(0, 0), graph, flip=True)
    assert inside.turns == 2 and not inside.cod_is_after()
    typ = SimpleNamespace(dom="white", cod="black")
    assert inside.label_background(typ) == "black"


def test_left_to_right_frame_keeps_its_name_inside():
    from discopy.drawing.backend import Backend
    from discopy.monoidal import Diagram
    from discopy.utils import factory

    @factory
    class Body(Diagram):
        draw_left_to_right = True

    class Primitive(Box, Body):
        pass

    class Recorder(Backend):
        def __init__(self):
            super().__init__()
            self.alignments = {}

        def draw_text(self, text, i, j, **params):
            self.alignments[text] = params.get("verticalalignment")

        def output(self, path=None, show=True, **params):
            pass

    n = Ty('n')
    y, x = Primitive('y', Ty(), n), Primitive('x', Ty(), n)
    with_ = Primitive('W', Ty(), n, args=(y, x), drawing_name="W",
                      draw_as_frame=True, draw_vertically=True)
    recorder = Recorder()
    with_.bubble(dom=Ty(), cod=n).draw(backend=recorder)
    assert recorder.alignments["W"] == "top"
    assert recorder.alignments["n"] == "bottom"


def test_frame_name_drawn_without_wire_labels():
    from discopy.drawing.backend import Backend

    class Recorder(Backend):
        def __init__(self):
            super().__init__()
            self.texts = []

        def draw_text(self, text, i, j, **params):
            self.texts.append(text)

        def output(self, path=None, show=True, **params):
            pass

    x = Ty('x')
    recorder = Recorder()
    Box('f', x, x).bubble(
        dom=x, cod=x, drawing_name="g", draw_as_frame=True
    ).draw(backend=recorder, wire_labels=False)
    assert "g" in recorder.texts and "x" not in recorder.texts


def test_left_to_right_frame_keeps_its_name_inside_when_reflected():
    from discopy.drawing.backend import Backend
    from discopy.monoidal import Diagram
    from discopy.utils import factory

    @factory
    class Body(Diagram):
        draw_left_to_right = True

    class Primitive(Box, Body):
        pass

    class Recorder(Backend):
        def __init__(self):
            super().__init__()
            self.texts, self.frames = {}, []

        def draw_text(self, text, i, j, **params):
            self.texts[text] = (j, params.get("verticalalignment"))

        def draw_polygon(self, *points, **params):
            self.frames.append([y for _, y in points])

        def output(self, path=None, show=True, **params):
            pass

    n = Ty('n')
    frame = Primitive('W', Ty(), n, args=(
        Primitive('y', Ty(), n), Primitive('x', Ty(), n)), drawing_name="W",
        draw_as_frame=True, draw_vertically=True)
    for dagger in (False, True):
        recorder = Recorder()
        drawing = frame.bubble(dom=Ty(), cod=n).to_drawing()
        (drawing.dagger() if dagger else drawing).draw(backend=recorder)
        j, alignment = recorder.texts["W"]
        bottom, top = min(recorder.frames[0]), max(recorder.frames[0])
        assert alignment == ("bottom" if dagger else "top")
        assert bottom < j < top


def test_frame_sides_have_an_edge_between_two_colours():
    from discopy.drawing.backend import Backend
    from discopy.monoidal import Colour, Wire

    class Recorder(Backend):
        def __init__(self):
            super().__init__()
            self.widths = []

        def draw_wire(self, source, target, *args, **params):
            self.widths.append(params.get("linewidth"))

        def draw_text(self, text, i, j, **params):
            pass

        def output(self, path=None, show=True, **params):
            pass

    x = Ty('x')
    framed, recorder = Recorder(), Recorder()
    Box('f', x, x).bubble(dom=x, cod=x, draw_as_frame=True).draw(
        backend=framed)
    assert framed.widths and 0 not in framed.widths
    Box('f', x, x).to_drawing().slot(None).draw(backend=recorder)
    assert 0 in recorder.widths
    coloured = Ty(Wire('x', Colour('red'), Colour('blue')))
    coloured_slot = Recorder()
    Box('f', coloured, coloured).to_drawing().slot(None).draw(
        backend=coloured_slot)
    assert 0 not in coloured_slot.widths


def test_left_to_right_args_in_the_variants_of_a_box():
    from discopy import pivotal
    from discopy.drawing.backend import Backend
    from discopy.utils import Point, factory

    @factory
    class Body(pivotal.Diagram):
        draw_left_to_right = True

    class Primitive(pivotal.Box, Body):
        pass

    class Recorder(Backend):
        def __init__(self):
            super().__init__()
            self.polygons, self.texts = [], {}

        def draw_polygon(self, *points, **params):
            self.polygons.append(points)

        def draw_text(self, text, i, j, **params):
            self.texts.setdefault(text, []).append(Point(i, j))

        def output(self, path=None, show=True, **params):
            pass

    x, y = pivotal.Ty('x'), pivotal.Ty('y')
    f, g = Primitive('f', x, y), Primitive('g', y, x)
    W = pivotal.Box('W', x, y, args=(f, g), drawing_name="W")
    chopped = {}
    for name, diagram in [("W", W), ("dagger", W[::-1]),
                          ("transpose", W.r), ("conjugate", W[::-1].r)]:
        recorder = Recorder()
        diagram.draw(backend=recorder, asymmetry=.25)
        outline, = (p for p in recorder.polygons if len(p) == 5)
        xs, ys = [p[0] for p in outline], [p[1] for p in outline]
        chopped[name] = next(
            (cx == max(xs), cy == max(ys))
            for cx in (min(xs), max(xs)) for cy in (min(ys), max(ys))
            if (cx, cy) not in outline)
        (f_at, ), (g_at, ) = recorder.texts["f"], recorder.texts["g"]
        turned = name in ("transpose", "conjugate")
        assert (f_at.y < g_at.y) == turned
        f_input = min(  # The label of the wire into f, its x or its y.
            recorder.texts["x" if name in ("W", "transpose") else "y"],
            key=lambda p: abs(p.y - f_at.y))
        assert (f_input.x > f_at.x) == turned
    assert chopped == {  # Whether the corner cut off is right, top.
        "W": (True, False), "dagger": (True, True),
        "transpose": (False, True), "conjugate": (False, False)}
