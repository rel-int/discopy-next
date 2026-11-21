import pytest
from hypothesis import given, settings

from discopy import monoidal
from discopy.monoidal import Ty, Box
from discopy.drawing.layout import OpenQP, to_drawing

pytest.importorskip("osqp")

x = Ty('x')


def assert_isotopic(drawing):
    framed = drawing.relabel_nodes()
    framed.add_box_corners()
    shape = framed.shape()
    assert not drawing.crossings()
    assert not shape["through"] and not shape["overlaps"]
    for _, order in shape["order"]:
        assert order == tuple(sorted(order))


@settings(max_examples=20, deadline=None)
@given(monoidal.Diagram.strategy())
def test_layout_is_isotopic(diagram):
    drawing = to_drawing(diagram)
    assert (drawing.dom, drawing.cod) == (diagram.dom, diagram.cod)
    assert len(drawing.boxes) == len(diagram.boxes)
    assert_isotopic(drawing)


def test_boxes_stretch_rather_than_wires_bend():
    f, g = Box('f', x, x ** 3), Box('g', x ** 2, x)
    u, v = Box('u', Ty(), x ** 3), Box('v', x ** 6, Ty())
    diagram = u >> g @ f >> x @ f @ x @ x >> v
    drawing = to_drawing(diagram)
    assert diagram.to_drawing().width == 6.5
    assert drawing.width == 6  # the six inputs of v, a unit apart
    assert all(drawing.positions[a].x == drawing.positions[b].x
               for a, b in drawing.wires())
    assert_isotopic(drawing)


def test_width_against_straightness():
    f, g = Box('f', x, x @ x), Box('g', x @ x, x)
    diagram = f @ x >> x @ g
    straight, narrow = (
        to_drawing(diagram, width=w) for w in (1, 10))
    assert narrow.width < straight.width
    assert any(narrow.positions[a].x != narrow.positions[b].x
               for a, b in narrow.wires())
    assert_isotopic(narrow)


def test_unit_on_the_nose():
    qp = OpenQP.functor()(Box('f', x, x @ x))
    assert OpenQP.id(x) >> qp == qp == qp >> OpenQP.id(x @ x)


def test_one_box_at_a_time():
    with pytest.raises(NotImplementedError):
        OpenQP.from_drawing((Box('f', x, x) @ Box('g', x, x)).to_drawing())


def test_dagger_reflects_waypoints():
    drawing = to_drawing(Box('f', x, x @ x) >> Box('g', x @ x, x))
    assert drawing.wire_nodes
    assert drawing.dagger().dagger() == drawing


def test_weights_are_not_negative():
    qp = OpenQP.functor()(Box('f', x, x))
    for weights in [dict(width=-1), dict(bend=-1), dict(square=-1)]:
        with pytest.raises(ValueError):
            qp.solve(**weights)
