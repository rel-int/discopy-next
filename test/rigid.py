from pytest import raises

from discopy.monoidal import Colour
from discopy.rigid import *


def test_Ob_init():
    with raises(TypeError) as err:
        Wire('x', z='y')


def test_Ty_over_under():
    x, y = Ty('x'), Ty('y')
    assert x.over(y) == x @ y.l == x << y
    assert y.under(x) == x.r @ y == x >> y


def test_Box_hash_winding():
    """
    Rigid boxes that differ only by their winding number must not be equal
    nor hash equally, see https://github.com/discopy/discopy/pull/387
    """
    x = Ty('x')
    f = Box('f', x, x)
    assert f != f.rotate() and hash(f) != hash(f.rotate())
    assert f == Box('f', x, x) and hash(f) == hash(Box('f', x, x))


def test_Wire_unwind():
    red, blue = Colour("red"), Colour("blue")
    x = Wire("x", dom=red, cod=blue)
    assert x.r.r.unwind() == x


def test_Ty_z():
    with raises(ValueError):
        Ty('x', 'y').z
    with raises(ValueError):
        Ty().z
    assert Ty('x').l.z == -1


def test_Nat_r():
    assert Nat(2).r == Nat(2)


def test_Diagram_cups():
    with raises(TypeError) as err:
        Diagram.cups('x', Ty('x'))
    with raises(TypeError) as err:
        Diagram.cups(Ty('x'), 'x')


def test_Diagram_caps():
    with raises(TypeError) as err:
        Diagram.caps('x', Ty('x'))
    with raises(TypeError) as err:
        Diagram.caps(Ty('x'), 'x')


def test_Diagram_normal_form():
    x = Ty('x')
    assert Id(x).transpose(left=True).normal_form() == Id(x.l)
    assert Id(x).transpose().normal_form() == Id(x.r)

    f = Box('f', Ty('a'), Ty('b') @ Ty('c'))
    assert f.normal_form() == f
    assert f.transpose().transpose(left=True).normal_form() == f
    assert f.transpose(left=True).transpose().normal_form() == f
    diagram = f\
        .transpose(left=True).transpose(left=True).transpose().transpose()
    assert diagram.normal_form() == f

    Eckmann_Hilton = Box('s0', Ty(), Ty()) @ Box('s1', Ty(), Ty())
    with raises(NotImplementedError) as err:
        Eckmann_Hilton.normal_form()
    assert str(err.value) == messages.NOT_CONNECTED.format(Eckmann_Hilton)


def test_Cup_init():
    with raises(TypeError):
        Cup('x', Ty('y'))
    with raises(TypeError):
        Cup(Ty('x'), 'y')
    t = Ty('n', 's')
    with raises(ValueError) as err:
        Cup(t, t.r)
    with raises(ValueError) as err:
        Cup(Ty(), Ty())


def test_Cap_init():
    with raises(TypeError):
        Cap('x', Ty('y'))
    with raises(TypeError):
        Cap(Ty('x'), 'y')
    t = Ty('n', 's')
    with raises(ValueError) as err:
        Cap(t, t.l)
    with raises(ValueError) as err:
        Cap(Ty(), Ty())


def test_Cup_Cap_adjoint():
    n = Ty('n')
    assert Cap(n, n.l).l == Cup(n.l.l, n.l)
    assert Cap(n, n.l).r == Cup(n, n.r)
    assert Cup(n, n.r).l == Cap(n, n.l)
    assert Cup(n, n.r).r == Cap(n.r.r, n.r)


def test_AxiomError():
    n, s = Ty('n'), Ty('s')
    with raises(AxiomError) as err:
        Cup(n, n)
    with raises(AxiomError) as err:
        Cup(n, s)
    with raises(AxiomError) as err:
        Cup(n, n.l)
    with raises(AxiomError) as err:
        Cup(n, n.r).dagger()
    with raises(AxiomError) as err:
        Cap(n, n.l).dagger()
    with raises(AxiomError) as err:
        Cup(n, n.l.l)
    with raises(AxiomError) as err:
        Cap(n, n.l.l)


def test_id_adjoint():
    assert Id(Ty('n')).r == Id(Ty('n').r)
    assert Id(Ty('n')).l == Id(Ty('n').l)
    assert Id().l == Id() == Id().r


def test_sum_adjoint():
    x = Ty('x')
    two, boxes = Box('two', x, x), Box('boxes', x, x)
    two_boxes = two + boxes
    assert two_boxes.l == two.l + boxes.l
    assert two_boxes.l.r == two_boxes


def test_curry_uncurry():
    x, y, z = map(Ty, "xyz")
    f = Box('f', x @ y, z)
    assert f.curry(n=0) == f == f.uncurry(n=0)
    assert f.curry().uncurry().normal_form() == f
    assert f.curry(left=False).uncurry(left=False).normal_form() == f
    assert f.curry(n=2).uncurry(n=2).normal_form() == f
    with raises(ValueError):
        f.curry(n=3)
    with raises(ValueError):
        f.uncurry(n=2)


def test_curry_zero():
    x = Ty('x')
    f = Box('f', x @ x, x)
    assert f.curry(0) == f == f.curry(0, left=False)


def test_Functor():
    """ The functor of a rigid diagram rotates, so a boundary keeps its z. """
    x, y = Ty('x'), Ty('y')
    assert Diagram.Functor is Functor
    assert Diagram.Functor({x: y}, {})(x.r) == y.r


def test_Ob_repr():
    assert repr(Wire('a', z=42)) == "rigid.Wire('a', z=42)"
    assert repr(Wire('a', dom=Colour('red'))) == (
        "rigid.Wire('a', dom=monoidal.Colour('red'), "
        "cod=monoidal.Colour('none'))")


def test_Ob_str():
    a = Wire('a')
    assert str(a) == "a" and str(a.r) == "a.r" and str(a.l) == "a.l"


def test_Ob_dagger():
    from discopy import biclosed, braided
    assert braided.Wire('a').dagger() == braided.Wire('a')
    assert biclosed.Wire('a').dagger() == biclosed.Wire('a')
    with raises(AxiomError):
        Wire('a').dagger()


def test_Wire_strategy():
    """ Rigid wires wind both ways, pivotal ones by parity, self-dual
    ones not at all. """
    from hypothesis import find
    from discopy import frobenius, pivotal
    assert find(Wire.strategy(), lambda wire: wire.z == -1).z == -1
    assert find(pivotal.Wire.strategy(), lambda wire: wire.z).z == 1
    assert find(frobenius.Wire.strategy(), lambda wire: True).z == 0
    assert len(find(frobenius.Ty.strategy(), lambda ty: len(ty) == 1)) == 1


def test_functor_factory():
    """ The functor of a rigid diagram rotates, so a boundary keeps its z. """
    x, y = Ty('x'), Ty('y')
    assert Diagram.Functor is Functor
    assert Diagram.Functor({x: y}, {})(x.r) == y.r
