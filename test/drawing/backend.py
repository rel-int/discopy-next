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


def test_tikz_wire_separates_its_style_keys():
    tikz = TikZ()
    tikz.draw_wire((0, 0), (1, 2), style="->")
    assert "->, looseness=" in tikz.edgelayer[-1]
