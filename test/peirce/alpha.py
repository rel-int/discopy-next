from itertools import product

from pytest import raises

from discopy.peirce.alpha import *
from discopy.tensor import Tensor
from discopy.utils import AxiomError

p, q = Box('p'), Box('q')


def test_colours():
    assert p.colour == white and p.negative().colour == black
    assert (~p).colour == white and (~p).arg == p.negative()
    assert p.negative().negative() == p and (~p).negative().colour == black


def test_cut_changes_colour():
    with raises(AxiomError):
        Cut(p, dom=p.dom, cod=p.cod)
    assert Cut(p.negative()) == ~p


def test_dunders():
    assert p & q == p @ q
    assert p | q == ~(~p & ~q)
    assert p.relative_sum(q) == ~(~p >> ~q)
    assert p.par(q) == ~(~p @ ~q)


def test_model():
    for P, Q in product((False, True), repeat=2):
        M = Model(ar={'p': P, 'q': Q})
        truth = lambda d: bool(M(d).array.all())
        assert truth(p & q) == (P and Q) and truth(p | q) == (P or Q)
        assert truth(~p) == (not P) and truth(~~p) == P
        assert truth(p.negative()) == (not P)
        assert truth(p.relative_sum(q)) == truth(p.par(q)) == (P or Q)
        assert isinstance(M(p), Tensor)
    assert Model(ar={'p': True}) != Model(ar={'p': False})
    assert Model(ar={'p': 1, 'q': 0}) == Model(ar={'q': 0, 'p': 1})


def test_cut_checks_both_boundaries():
    with raises(AxiomError, match="changes the colour"):
        Cut(p.negative(), cod=p.negative().cod)
