from pytest import raises

from discopy import bang
from discopy.bang import *
from discopy.utils import AxiomError, dumps, loads


def test_structural_boxes():
    from discopy.testable import assert_structural_boxes
    assert_structural_boxes(Diagram)


def test_LinearCategory_is_monoidal_not_biclosed():
    from discopy import biclosed, rigid, tensor
    from discopy.abc import LinearCategory
    assert issubclass(Diagram, LinearCategory)
    for category in (biclosed.Diagram, rigid.Diagram, tensor.Diagram):
        assert not issubclass(category, LinearCategory)
        assert not hasattr(category, "bang")


def test_bang_is_one_wire_and_a_bubble():
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x, y), Box('g', y, x)
    assert f.bang() == Bang(f) and f.bang().dom == x.bang() == Ty(BangOb(x))
    assert (x @ y).bang() != x.bang() @ y.bang()
    assert len((x @ y).bang()) == 1 and (x @ y).bang().inside[0].arg == x @ y
    assert (f >> g).bang() != f.bang() >> g.bang()
    assert Id(x).bang() != Id(x.bang()) and Bang(Id(x)).arg == Id(x)
    assert Bang(f).dagger() == Bang(f[::-1])
    assert Ty().bang() == Ty(BangOb(Ty())) != Ty()


def test_structural_primitives_of_a_linear_category():
    x, y = Ty('x'), Ty('y')
    assert Dereliction(x).dom == x.bang() and Dereliction(x).cod == x
    assert Digging(x).cod == x.bang().bang()
    assert Contraction(x).cod == x.bang() @ x.bang()
    assert Weakening(x).dom == x.bang() and Weakening(x).cod == Ty()
    assert Exchange(x, y).dom == x.bang() @ y == Exchange(x, y, True).cod
    assert Exchange(x, y)[::-1] == Exchange(x, y, True)
    assert Laxator(x, y).dom == x.bang() @ y.bang()
    assert Laxator(x, y).cod == (x @ y).bang()
    assert Laxator().dom == Ty() and Laxator().cod == Ty().bang()
    for method, box in [
            (Diagram.dereliction, Dereliction), (Diagram.digging, Digging),
            (Diagram.contraction, Contraction), (Diagram.weakening, Weakening)]:
        assert method(x) == box(x)
    assert Diagram.exchange(x, y, True) == Exchange(x, y, True)
    assert Diagram.laxator(x, y) == Laxator(x, y)
    assert Diagram.laxator() == Laxator()
    promotion = Digging(x) >> Box('g', x.bang(), y).bang()
    assert (promotion.dom, promotion.cod) == (x.bang(), y.bang())
    with raises(ValueError):
        Laxator(x)
    from discopy.utils import AxiomError
    for box in (Dereliction(x), Digging(x), Contraction(x), Weakening(x),
                Laxator()):
        with raises(AxiomError):
            box.dagger()


def test_bang_reads_back():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    namespace = vars(bang) | dict(x=x, y=y, f=f, bang=bang)
    for thing in (
            x.bang(), (x @ y).bang().bang(), Ty().bang(), f.bang(), (f >> f[::-1]).bang(), Bang(f).bang(),
            Dereliction(x @ y), Digging(x.bang()), Contraction(x),
            Weakening(x), Exchange(x, y), Exchange(x, y @ x, True),
            Laxator(x, y), Laxator()):
        assert eval(repr(thing), namespace) == thing
        assert eval(str(thing), namespace) == thing
        assert loads(dumps(thing)) == thing
    assert str(x.bang()) == "x.bang()"
    assert str((x @ y).bang()) == "(x @ y).bang()"


def test_Functor_sends_the_bang_through_the_codomain():
    from discopy import tensor
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    F = Functor({x: y, y: x}, {f: f[::-1]})
    assert F(x.bang()) == y.bang() and F(f.bang()) == f[::-1].bang()
    assert F(Dereliction(x)) == Dereliction(y)
    assert F(Exchange(x, y, True)) == Exchange(y, x, True)
    assert F(Laxator(x, y)) == Laxator(y, x) and F(Laxator()) == Laxator()
    assert F(Contraction(x) >> f.bang() @ Weakening(x)) == (
        Contraction(y) >> f[::-1].bang() @ Weakening(y))
    G = Functor(
        ob_map={x: tensor.Dim(2), y: tensor.Dim(3),
                x.bang(): tensor.Dim(4), y.bang(): tensor.Dim(5)},
        ar_map={Dereliction(x): tensor.Box('e', tensor.Dim(4), tensor.Dim(2)),
                f.bang(): tensor.Box('!f', tensor.Dim(4), tensor.Dim(5))},
        cod=tensor.Diagram)
    assert G(x.bang()) == tensor.Dim(4)
    assert G(Dereliction(x)) == tensor.Box('e', tensor.Dim(4), tensor.Dim(2))
    assert G(f.bang()) == tensor.Box('!f', tensor.Dim(4), tensor.Dim(5))


def test_bang_is_drawn_with_an_exclamation_mark():
    x, y = Ty('x'), Ty('y')
    drawing = (Contraction(x) >> Box('f', x, y).bang() @ Dereliction(x))\
        .to_drawing()
    assert [str(wire) for wire in drawing.dom.inside] == ["!x"]
    assert Ty(BangOb(x @ y)).to_drawing().inside[0].name == "!(x @ y)"
    assert {"Δ", "ε"} <= {box.drawing_name for box in drawing.boxes}


def test_wiring():
    x, y = Ty('x'), Ty('y')
    assert Diagram.wiring([1, 0], x @ y.bang()) == Exchange(y, x, True)
    assert Diagram.wiring([0, 1, 0, 1], x.bang() @ y.bang())\
        == Contraction(x) @ Contraction(y)\
        >> x.bang() @ Exchange(x, y.bang()) @ y.bang()
    assert Diagram.wiring([], x.bang()) == Weakening(x)
    for wiring, dom in [([1, 0], x @ y), ([0, 0], x @ y), ([1], x @ y)]:
        with raises(AxiomError):
            Diagram.wiring(wiring, dom)


def test_terms_share_banged_variables():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    g = Box('g', x.bang() @ y @ x.bang(), z)
    term = (x.bang() @ y)(lambda a, b: g(a, b, a))
    assert term.freevars == [] and term.stack == x.bang() @ y
    assert term.eval() == Contraction(x) @ y\
        >> x.bang() @ Exchange(x, y) >> g
    assert term.normal_form() == term
    with raises(ValueError):
        (x @ y)(lambda a, b: Box('h', x @ y @ x, z)(a, b, a))


def test_kappa_weakens_unused_banged_variables():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    g, h = Box('g', x.bang() @ x.bang(), z), Box('h', x @ y, z)
    term = (x.bang() @ y.bang() @ x.bang())(lambda a, b, c: g(a, c))
    assert term.stack == x.bang() @ y.bang() @ x.bang()
    assert term.eval() == x.bang() @ Weakening(y) @ x.bang() >> g
    a = Variable('a', x)
    right = (y @ y.bang())(lambda c, d: Constant(h)(a, c))
    assert right.freevars == [a] and right.offset == 1
    assert right.eval() == x @ y @ Weakening(y) >> h
    b, d = Variable('b', y), Variable('d', y.bang())
    after = Kappa((d, b), x(lambda a: Constant(h)(a, b)))
    assert after.stack == x @ y.bang() @ y
    assert after.eval() == x @ Weakening(y) @ y >> h
    f, s = Box('f', x, y), Box('s', Ty(), x.bang())
    redex = x.bang()(lambda b: Constant(f)(a), offset=1)(Constant(s))
    assert redex.normal_form().eval() == x @ s >> x @ Weakening(x) >> f
    with raises(ValueError):
        (x @ x)(lambda b, c: Constant(f)(c))


def test_a_term_with_no_box_is_a_bang_term():
    x, y = Ty('x'), Ty('y')
    for term, diagram in [
            ((x.bang() @ y)(lambda a, b: b), Weakening(x) @ y),
            (x.bang()(lambda a: a), Id(x.bang()))]:
        assert isinstance(term, Kappa) and term.eval() == diagram
        assert type(eval(str(term))) is Kappa and eval(str(term)) == term
    with raises(ValueError):
        (x @ y)(lambda a, b: b)
