from discopy.ribbon import *


def test_to_ribbons_width():
    x = Ty('x')

    # By default the two wires of each ribbon are four times closer.
    drawing = Diagram.twist(x).to_ribbons().to_drawing()
    dom = sorted(drawing.positions[n].x for n in drawing.dom_nodes)
    assert round(dom[1] - dom[0], 3) == 0.25

    # width=1 leaves the wires at the usual minimal width.
    drawing = Diagram.twist(x).to_ribbons(width=1).to_drawing()
    dom = sorted(drawing.positions[n].x for n in drawing.dom_nodes)
    assert round(dom[1] - dom[0], 3) == 1.0


def test_to_ribbons_default_and_zero_width():
    from discopy import config

    x = Ty('x')
    twist = Diagram.twist(x)

    # width=None pulls the default width from discopy.config.
    assert twist.to_ribbons() == twist.to_ribbons(
        config.DRAWING_DEFAULT["ribbon_width"])

    # width=0 returns the diagram as is, i.e. without dual rails.
    assert twist.to_ribbons(width=0) == twist


def test_to_ribbons_trace_width():
    x = Ty('x')

    # The looped ribbon of a trace stays compressed all the way around, even
    # though its wires are rotated (the adjoint keeps the region between).
    drawing = Braid(x, x).trace(left=True).to_ribbons().to_drawing()
    rows = {}
    for node, point in drawing.positions.items():
        if node.kind in ("box_dom", "box_cod"):
            rows.setdefault(round(point.y, 3), []).append(round(point.x, 3))
    for xs in rows.values():
        xs = sorted(xs)
        gaps = [round(xs[i + 1] - xs[i], 3) for i in range(len(xs) - 1)]
        assert gaps == [0.25, 1.0, 0.25]


def test_to_ribbons_gadgets():
    x = Ty('x')
    trace = Braid(x, x).trace(left=True)

    # Cups, caps, braids and twists each become a single dual rail box.
    boxes = trace.to_ribbons(width=None).boxes
    assert any(isinstance(b, DualRailCap) for b in boxes)
    assert any(isinstance(b, DualRailCup) for b in boxes)
    assert any(isinstance(b, DualRailBraid) for b in boxes)

    cup, = (b for b in boxes if isinstance(b, DualRailCup))
    assert len(cup.dom) == 4 and len(cup.cod) == 0
    assert isinstance(cup.dagger(), DualRailCap)


def test_to_ribbons_gadgets_eval_repr():
    x = Ty('x')
    boxes = Braid(x, x).trace(left=True).to_ribbons().boxes
    for box in boxes + [box.dagger() for box in boxes]:
        assert eval(repr(box), Diagram.environment()) == box


def test_to_ribbons_box():
    from discopy.balanced import double_rail
    x, y = Ty('x'), Ty('y')
    # A generator is doubled into a box on the rails of its ribbons.
    f, g = Box('f', x, x @ y), Box('g', x @ y, x)
    doubled, = f.to_ribbons(width=None).boxes
    assert doubled.dom == double_rail(x)
    assert doubled.cod == double_rail(x @ y)
    # It still composes with the doubled wires (and gadgets) around it.
    assert (f >> x @ Twist(y) >> g).to_ribbons(width=None)


def test_Kauffman(monkeypatch):
    monkeypatch.setattr(Ty, "l", property(lambda self: self))
    monkeypatch.setattr(Ty, "r", property(lambda self: self))
    x, A = Ty('x'), Box('A', Ty(), Ty())

    class Polynomial(Diagram):
        def braid(x, y):
            return (A @ x @ y) + (Cup(x, y) >> A.dagger() >> Cap(x, y))

    Kauffman = Functor(ob_map={x: x}, ar_map={}, cod=Polynomial)

    assert Kauffman(Braid(x, x))\
        == (A @ x @ x) + (Cup(x, x) >> A.dagger() >> Cap(x, x))


def test_rotate():
    x, y = Ty('x'), Ty('y')
    assert Twist(x).r == Twist(x.r)
    assert Braid(x, y).dagger().r == Braid(x.r, y.r).dagger()
    rotated = (Box('f', Ty(), y @ x) >> Braid(x, y).dagger()).r
    assert Diagram(rotated.inside, rotated.dom, rotated.cod) == rotated


def test_rotation_is_well_typed():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x @ y, x @ y)
    snake = Cap(x, x.r) @ x >> x @ Cup(x.r, x)
    for diagram in (f >> Braid(x, y), snake, f >> Twist(x) @ y,
                    f >> Twist(x)[::-1] @ y):
        for rotated in (diagram.r, diagram.l,
                        diagram.to_ribbons().r, diagram.to_ribbons().l):
            assert Diagram(rotated.inside, rotated.dom, rotated.cod)\
                == rotated
    assert DualRailCup(x @ x, y @ y, is_dagger=True).r\
        == DualRailCap(y.r @ y.r, x.r @ x.r, is_dagger=True)


def test_cup_joins_two_distinct_wires():
    from pytest import raises
    n = Ty('n')
    for i in range(2):
        with raises(ValueError, match="distinct"):
            Id(n @ n.r).cup(i, i)


def test_to_hypergraph():
    # A twist is not the identity: it stays a box of the hypergraph.
    x = Ty('x')
    assert Diagram.twist(x).to_hypergraph().boxes == (Diagram.twist(x), )


def test_functor_factory():
    x, y = Ty('x'), Ty('y')
    assert Diagram.functor_factory is Functor
    assert Diagram.functor_factory({x: y}, {})(Twist(x)) == Twist(y)
    assert Diagram.functor_factory({x: y}, {})(Braid(x, x)) == Braid(y, y)


def test_sum_factory():
    x = Ty('x')
    f = Box('f', x, x)
    assert Diagram.sum_factory is Sum and type(f + f) is Sum


def test_dual_rail_delay():
    x = Ty('x')
    for diagram in [Cup(x, x.r), Cap(x, x.l), Twist(x), Braid(x, x)]:
        assert diagram.delay().to_ribbons() == diagram.to_ribbons().delay()


def test_Bubble_to_ribbons():
    x = Ty('x')
    f = Box('f', x, x)
    assert f.bubble(name="N").to_ribbons() == f.to_ribbons().bubble(name="N")


def test_feedback_to_ribbons_is_built_from_its_argument():
    x = Ty('x')
    g = Box('g', x @ x, x @ x)
    rails = g.to_ribbons()
    assert Trace(g).to_ribbons() == rails.trace(n=2)
    assert Head(g).to_ribbons() == rails.head
    assert Tail(g).to_ribbons() == rails.tail
    assert g.delay().to_ribbons() == rails.delay()


def test_structural_boxes():
    from discopy.testable import assert_structural_boxes
    assert_structural_boxes(Diagram)
