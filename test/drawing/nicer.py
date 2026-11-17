from discopy.monoidal import Ty, Box
from discopy.drawing.nicer import (
    WEIGHTS, features, hand_score, make_nicer, rectangles, wires)


x = Ty('x')
f, g, h = Box('f', x, x), Box('g', x, x), Box('h', x @ x, x)
s, e = Box('s', Ty(), x), Box('e', x, Ty())
diagrams = [
    f @ g, f @ x @ x >> x @ g @ x >> x @ x @ f >> h @ x >> h,
    f @ f @ f >> f @ f @ f, s @ s >> f @ x >> e @ e, Box('long_name', x, x)]


def test_features_are_finite_costs():
    for diagram in diagrams:
        costs = features(diagram.to_drawing())
        assert set(costs) == set(WEIGHTS)
        assert all(0 <= cost < float("inf") for cost in costs.values())


def test_wires_and_rectangles():
    drawing = (f @ g).foliation().to_drawing()
    assert len(wires(drawing)) == 4
    (l0, b0, r0, t0), (l1, b1, r1, t1) = rectangles(drawing)
    assert b0 == b1 and t0 == t1 and r0 <= l1
    assert features(Box('a' * 40, x, x).to_drawing())["label_overlap"] == 0


def test_make_nicer_keeps_the_diagram():
    interchangers = lambda d: d.to_staircases().normal_form(
        normalize=type(d).normalize)
    for diagram in diagrams[:3]:
        nicer = make_nicer(diagram)
        assert sorted(map(str, nicer.boxes))\
            == sorted(map(str, diagram.boxes))
        assert interchangers(nicer) == interchangers(diagram)
        assert hand_score(nicer.to_drawing()) >= hand_score(
            diagram.to_drawing())


def test_make_nicer_takes_a_score():
    assert make_nicer(f @ g, score=lambda d: d.height) == f @ g
    assert make_nicer(f @ g, score=lambda d: -d.height)\
        == (f @ g).foliation()
