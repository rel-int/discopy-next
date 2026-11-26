from pytest import raises

from discopy.abc import ClosedCategory, MarkovCategory
from discopy.closed.symmetric import *
from discopy.utils import AxiomError


def test_one_exponential_on_either_side():
    x, y = Ty('x'), Ty('y')
    assert x << y == x ** y == y >> x
    assert Eval(x << y).dom == (x << y) @ y
    assert Eval(x << y, left=False).dom == y @ (x << y)
    f = Box('f', y @ x, x)
    assert (f.curry().dom, f.curry().cod) == (y, x << x)
    assert (f.curry(left=False).dom, f.curry(left=False).cod) == (x, x << y)


def test_closed_and_not_markov():
    assert issubclass(Diagram, ClosedCategory)
    assert not issubclass(Diagram, MarkovCategory)
    with raises(AttributeError):
        Diagram.copy


def test_functor_keeps_swaps_and_evaluations():
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x @ y, x), Box('g', y @ x, y)
    F = Functor({x: y, y: x}, {f: g})
    assert F(Swap(x, y)) == Swap(y, x)
    assert F(Eval(x << y, left=False)) == Eval(y << x, left=False)
    assert F(f.curry()) == g.curry()


def test_curry_has_no_dagger():
    with raises(AxiomError):
        Box('f', Ty('x', 'y'), Ty('x')).curry().dagger()
