from pytest import raises

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
