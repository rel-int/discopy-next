import re
from io import BytesIO

import numpy as np
from pytest import mark, raises

from discopy.drawing.backend import TikZ
from discopy.monoidal import Ty, Box


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
