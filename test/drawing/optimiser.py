from math import pi

import numpy as np
import pytest
from hypothesis import given, settings

from discopy import monoidal, pivotal, symmetric
from discopy.monoidal import Ty, Box
from discopy.drawing.optimiser import (
    OpenQP, PlanarMap, to_drawing, to_pivotal_drawing)

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


def test_a_picture_has_no_dagger():
    drawing = to_drawing(Box('f', x, x @ x) >> Box('g', x @ x, x))
    assert drawing.wire_nodes
    with pytest.raises(NotImplementedError):
        drawing.dagger()


def test_waypoints_are_drawn_as_curves(tmp_path):
    drawing = to_drawing(Box('f', x, x @ x) >> Box('g', x @ x, x))
    waypoints = [(u, v) for u, v in drawing.edges
                 if "wire" in (u.kind, v.kind)]
    drawing.draw(path=str(tmp_path / "out.tikz"), to_tikz=True)
    assert (tmp_path / "out.tikz").read_text().count(
        ".. controls") == len(waypoints) > 0


@pytest.mark.parametrize("params", [{}, {"format": "typst"}])
def test_every_backend_draws_waypoints(tmp_path, params):
    if params:
        pytest.importorskip("typst")
    drawing = to_drawing(Box('f', x, x @ x) >> Box('g', x @ x, x))
    path = tmp_path / ("out.svg" if params else "out.png")
    drawing.draw(path=str(path), **params)
    assert path.exists() and path.stat().st_size


def test_weights_are_not_negative():
    qp = OpenQP.functor()(Box('f', x, x))
    for weights in [dict(width=-1), dict(bend=-1), dict(square=-1)]:
        with pytest.raises(ValueError):
            qp.solve(**weights)


px, py = pivotal.Ty('x'), pivotal.Ty('y')


def test_a_waypoint_on_an_end_draws_nothing(tmp_path):
    drawing = to_pivotal_drawing(
        px @ pivotal.Cap(px.r, px) >> pivotal.Cup(px, px.r) @ px)
    curves = [(u, v) for u, v in drawing.edges if "wire" in (u.kind, v.kind)
              and drawing.positions[u] != drawing.positions[v]]
    drawing.draw(path=str(tmp_path / "out.tikz"), to_tikz=True)
    assert (tmp_path / "out.tikz").read_text().count(
        ".. controls") == len(curves) > 0


def test_snake_is_pulled_straight():
    drawing = to_pivotal_drawing(
        px @ pivotal.Cap(px.r, px) >> pivotal.Cup(px, px.r) @ px)
    xs = [p.x for p in drawing.positions.values()]
    assert max(xs) - min(xs) < .05


@pytest.mark.parametrize("box", [
    pivotal.Box('f', px, py), pivotal.Box('g', px @ px, py)])
def test_transpose_is_upside_down(box):
    drawing = to_pivotal_drawing(box.transpose())
    angle, = (b.angle for b in drawing.boxes if b.name == box.name)
    assert abs(abs(angle) - pi) < .2


def test_words_turn_to_the_verb():
    from discopy.lambek.pregroup import Ty, Word, Cup
    s, n = Ty('s'), Ty('n')
    sentence = Word('Alice', n) @ Word('loves', n.r @ s @ n.l)\
        @ Word('Bob', n) >> Cup(n, n.r) @ s @ Cup(n.l, n)
    planar = PlanarMap.from_drawing(to_drawing(sentence))
    z = planar.solve()
    assert planar.is_valid(z) and planar.energy(z) < planar.energy(
        planar.start) / 10
    alice, loves, bob = (np.arctan2(z[4 * k + 3], z[4 * k + 2])
                         for k in range(3))
    assert alice > pi / 2 and bob < -pi / 2 and abs(loves) < .1


def test_crossing_wires_are_not_valid():
    f = pivotal.Box('f', px @ px, px @ px)
    planar = PlanarMap.from_drawing(to_drawing(f))
    assert planar.is_valid(planar.start)
    z, top = planar.start.copy(), planar.offset("top")
    z[top], z[top + 1] = z[top + 1], z[top]
    assert not planar.is_valid(z)


def test_a_wire_crossing_itself_is_not_valid():
    snake = PlanarMap.from_drawing(to_drawing(
        px @ pivotal.Cap(px.r, px) >> pivotal.Cup(px, px.r) @ px))
    z, points = snake.start.copy(), snake.offset("points")
    z[points: points + 2] = .3, .4
    assert not snake.is_valid(z)


def test_the_empty_diagram():
    drawing = to_pivotal_drawing(pivotal.Id())
    assert not drawing.boxes and not drawing.positions


def test_what_it_does_not_draw():
    a = symmetric.Ty('a')
    with pytest.raises(NotImplementedError):
        PlanarMap.from_drawing(to_drawing(symmetric.Swap(a, a)))
    with pytest.raises(NotImplementedError):
        PlanarMap.from_drawing(to_drawing(
            pivotal.Cap(px, px.l) >> pivotal.Cup(px, px.l)))
