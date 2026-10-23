from pytest import raises

from discopy.utils import AxiomError
from discopy.pivotal import *


def test_trace():
    with raises(AxiomError):
        Box('f', 'x', 'y').trace()


def test_Sum():
    f = Box('f', 'x', 'x')
    assert Sum([f]) == f
    assert isinstance(f + f, Sum) and (f + f).terms == (f, f)
    image = Functor(lambda x: x, {f: f + f})(f)
    assert isinstance(image, Sum) and image.terms == (f, f)


def test_Box_is_traced():
    from discopy import feedback
    x = Ty('x')
    assert isinstance(Box('f', x, x), feedback.Box)


def test_subs_keeps_winding_number_and_time_step():
    from sympy.abc import phi
    from discopy.grammar.pregroup import Ty as P, Word
    x = Ty('x')
    f = Box('f', x, x @ x, data=phi)
    assert f.delay().r.subs(phi, 1) == f.subs(phi, 1).delay().r
    assert f.lambdify(phi)(1).r[::-1] == f.r[::-1].lambdify(phi)(1)
    n, s = P('n'), P('s')
    word = Word('w', n @ s, dom=n.r, data=phi)
    assert word.l.subs(phi, 1) == Word('w', n @ s, dom=n.r, data=1).l
