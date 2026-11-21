from math import pi

import numpy as np
import pytest

from discopy import pivotal, symmetric
from discopy.drawing import layout
from discopy.drawing.planar import PlanarMap, to_drawing

pytest.importorskip("osqp")

x, y = pivotal.Ty('x'), pivotal.Ty('y')


def test_snake_is_pulled_straight():
    drawing = to_drawing(x @ pivotal.Cap(x.r, x) >> pivotal.Cup(x, x.r) @ x)
    xs = [p.x for p in drawing.positions.values()]
    assert max(xs) - min(xs) < .05


@pytest.mark.parametrize("box", [
    pivotal.Box('f', x, y), pivotal.Box('g', x @ x, y)])
def test_transpose_is_upside_down(box):
    drawing = to_drawing(box.transpose())
    angle, = (b.angle for b in drawing.boxes if b.name == box.name)
    assert abs(abs(angle) - pi) < .2


def test_words_turn_to_the_verb():
    from discopy.lambek.pregroup import Ty, Word, Cup
    s, n = Ty('s'), Ty('n')
    sentence = Word('Alice', n) @ Word('loves', n.r @ s @ n.l)\
        @ Word('Bob', n) >> Cup(n, n.r) @ s @ Cup(n.l, n)
    planar = PlanarMap.from_drawing(layout.to_drawing(sentence))
    z = planar.solve()
    assert planar.is_valid(z) and planar.energy(z) < planar.energy(
        planar.start) / 10
    alice, loves, bob = (np.arctan2(z[4 * k + 3], z[4 * k + 2])
                         for k in range(3))
    assert alice > pi / 2 and bob < -pi / 2 and abs(loves) < .1


def test_crossing_wires_are_not_valid():
    f = pivotal.Box('f', x @ x, x @ x)
    planar = PlanarMap.from_drawing(layout.to_drawing(f))
    assert planar.is_valid(planar.start)
    z, top = planar.start.copy(), planar.offset("top")
    z[top], z[top + 1] = z[top + 1], z[top]
    assert not planar.is_valid(z)


def test_what_it_does_not_draw():
    a = symmetric.Ty('a')
    with pytest.raises(NotImplementedError):
        PlanarMap.from_drawing(layout.to_drawing(symmetric.Swap(a, a)))
    with pytest.raises(NotImplementedError):
        PlanarMap.from_drawing(layout.to_drawing(
            pivotal.Cap(x, x.l) >> pivotal.Cup(x, x.l)))
