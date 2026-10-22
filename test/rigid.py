from pytest import raises

from discopy import messages
from discopy.monoidal import Colour
from discopy.rigid import *


def test_Ob_init():
    with raises(TypeError) as err:
        Wire('x', z='y')


def test_Ob_eq():
    assert Wire('a') == Wire('a').l.r and Wire('a') != 'a'


def test_Ob_hash():
    a = Wire('a')
    assert {a: 42}[a] == 42


def test_Ty_over_under():
    x, y = Ty('x'), Ty('y')
    assert x.over(y) == x @ y.l == x << y
    assert y.under(x) == x.r @ y == x >> y


def test_Box_hash():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    assert f == f @ Id()
    assert hash(f) == hash(f @ Id())
    assert hash(f) == hash(Id() @ f)
    assert f @ Id() in {f}
    assert {f: 42}[f @ Id()] == 42


def test_Box_hash_winding():
    """
    Rigid boxes that differ only by their winding number must not be equal
    nor hash equally, see https://github.com/discopy/discopy/pull/387
    """
    x = Ty('x')
    f = Box('f', x, x)
    assert f != f.rotate() and hash(f) != hash(f.rotate())
    assert f == Box('f', x, x) and hash(f) == hash(Box('f', x, x))


def test_Box_dagger_keeps_winding():
    from discopy import rigid
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    for box in [f.r, f.l, f.r.r, f.l.l]:
        assert box[::-1].z == box.z and box[::-1][::-1] == box
        assert eval(repr(box[::-1])) == box[::-1]
    assert f.r[::-1] == f[::-1].r and f.l[::-1] == f[::-1].l


def test_Ob_repr():
    assert repr(Wire('a', z=42)) == "rigid.Wire('a', z=42)"


def test_Ob_str():
    a = Wire('a')
    assert str(a) == "a" and str(a.r) == "a.r" and str(a.l) == "a.l"


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


def test_Diagram_cups_caps_of_unequal_lengths():
    x = Ty('x')
    for left, right in [(Ty(), x.r), (x, Ty()), (x @ x, x.r)]:
        with raises(AxiomError):
            Diagram.cups(left, right)
        with raises(AxiomError):
            Diagram.caps(right, left)


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


def test_functor_factory():
    """ The functor of a rigid diagram rotates, so a boundary keeps its z. """
    x, y = Ty('x'), Ty('y')
    assert Diagram.functor_factory is Functor
    assert Diagram.functor_factory({x: y}, {})(x.r) == y.r


def test_Layer_merge_to_identity():
    x = Ty('x')
    with raises(AxiomError):  # a snake is no layer, see #599.
        Layer(Cap(x, x.l), x).merge(Layer(x, Cup(x.l, x)))


def test_rotated_box_serialisation():
    from hypothesis import find, given, settings
    from discopy.utils import dumps, loads
    for z in [-2, -1, 1, 2]:
        assert find(Box.strategy(), lambda box, z=z: box.z == z).z == z

    from discopy import pivotal
    f = pivotal.Box('f', pivotal.Ty('x'), pivotal.Ty('y'))
    for box in [f.r.delay(), f.delay().l.l]:
        assert loads(dumps(box)) == box

    @settings(max_examples=50)
    @given(Box.strategy())
    def round_trip(box):
        assert loads(dumps(box)) == box
    round_trip()
