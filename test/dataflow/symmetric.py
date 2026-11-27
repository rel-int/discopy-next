import pickle

from pytest import raises

from discopy import feedback, python, stream
from discopy.dataflow.symmetric import *
from discopy.utils import AxiomError, dumps, loads


def test_invalid_inputs():
    with raises(ValueError):
        HeadOb(Wire('x').delay())
    with raises(ValueError):
        TailOb(Wire('x').delay())
    with raises(ValueError):
        TailOb(Wire('x'))


def test_Head_Tail_FollowedBy_to_tree():
    from discopy.dataflow import symmetric
    x, y = Ty(Wire('x', is_constant=False)), Ty('y')
    f = Box('f', x, y)
    for term in (x.head, x.tail, x.head.delay(), f.head, f.tail, f.tail.delay(2),
                 FollowedBy(x), FollowedBy(x, is_dagger=True).delay()):
        assert loads(dumps(term)) == term
        assert pickle.loads(pickle.dumps(term)) == term
        assert eval(repr(term)) == term


def test_Head_Tail_dagger():
    from itertools import product
    x, y = (Ty(Wire(name, is_constant=False)) for name in "xy")
    f, g = Box('f', x, y), Box('g', y, x)
    for arg, part, steps in product((f, f >> g), ("head", "tail"), (0, 2)):
        term = getattr(arg, part).delay(steps)
        dagger = getattr(arg.dagger(), part).delay(steps)
        assert term.dagger() == term[::-1] == dagger
        assert type(term.dagger()) is type(term)
        assert term.dagger().dagger() == term
    assert (f.head >> g.head).dagger() == g.dagger().head >> f.dagger().head


def test_FollowedBy_delay():
    x = Ty('x')
    for is_dagger in (False, True):
        fby = FollowedBy(x, is_dagger)
        assert fby.delay().delay() == fby.delay(2)\
            == FollowedBy(x, is_dagger, time_step=2)
        assert fby.delay(2).dom == fby.dom.delay(2)
        assert fby.delay(2).cod == fby.cod.delay(2)


def test_Functor_sends_equal_heads_to_equal_images():
    """ The head and tail of a daggered wire are named after its string,
    which shows the dagger, so that they are not those of the wire it
    daggers, which a functor sends elsewhere (next#445). """
    x, y, z = map(Ty, "xyz")
    w = Ty(Wire('w', is_constant=False))
    F = Functor({x: y @ z, w: Ty(Wire('v', is_constant=False))}, {})
    types = [x.head, x[::-1].head, w.tail, w[::-1].tail, w.head]
    assert len(set(types)) == len(types)
    assert F(x.head) == (y @ z).head != (z[::-1] @ y[::-1]).head\
        == F(x[::-1]).head == F(x[::-1].head)
    assert F(w[::-1].tail) == F(w[::-1]).tail
    for term in types:
        assert eval(str(term), {'x': x, 'w': w}) == term


def test_functor_preserves_the_boundary_of_FollowedBy():
    x, y = (Ty(Wire(name, is_constant=False)) for name in "xy")
    F, identity = Functor({x: y}, {}), Functor(lambda t: t, lambda f: f)
    for fby in (FollowedBy(x), FollowedBy(x, is_dagger=True),
                FollowedBy(x, is_dagger=True).delay()):
        assert identity(fby) == fby
        assert (F(fby).dom, F(fby).cod) == (F(fby.dom), F(fby.cod))
    iso = FollowedBy(x) >> FollowedBy(x).dagger()
    assert F(iso) == FollowedBy(y) >> FollowedBy(y).dagger()


def test_stream_boxes_raise_into_a_codomain_without_streams():
    x = Ty('x')
    f = Box('f', x, x)
    F = Functor({x: int}, {f: lambda n: n + 1}, cod=python.Function)
    assert F(f.delay())(1) == F(f)(1) == 2
    for term in (x.head, f.head, f.tail, FollowedBy(x), f.head.delay()):
        with raises(AxiomError):
            F(term)
    G = Functor({x: int, x.head: int}, {}, cod=python.Function)
    assert G(x.head) == G(x)


def test_HeadOb_TailOb_dagger():
    x = Ty(Wire('x', is_constant=False))
    for term in (x.head, x.tail, x.head.delay(2), x.tail.delay()):
        assert term.dagger().dagger() == term
        assert type(term.dagger().inside[0]) is type(term.inside[0])
    with raises(TypeError):
        Wire('x', is_constant=1)
