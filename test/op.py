from pytest import raises

from discopy import op
from discopy.abc import Nat
from discopy.axioms import assert_axioms
from discopy.matrix import Matrix
from discopy.python.finset import Function


def test_axioms():
    assert_axioms(*(level[Matrix[int]] for level in (
        op.Category, op.Monoidal, op.Symmetric, op.Markov, op.Comarkov)))


def test_FinSet():
    FinSet, one, two = op.Comarkov[Function], Nat(1), Nat(2)
    assert FinSet.merge(one) == FinSet(Function([0, 0], 1, 2))
    assert FinSet.swap(one, two) == FinSet(Function.swap(two, one))
    assert FinSet.noise(two).dom == Nat(0) and FinSet.noise(two).cod == two
    assert op.Markov.copy(one).inside == FinSet.merge(one)


def test_transparency():
    for arrow in (op.Markov[Matrix[int]].copy(Nat(1)), op.Markov.copy(Nat(1))):
        assert eval(repr(arrow), type(arrow).environment() | {
            "Function": Function, "Nat": Nat}) == arrow


def test_wrong_category():
    with raises(TypeError):
        op.Comarkov[Function](Matrix[int].id(1))
    with raises(TypeError):
        op.Comarkov[Matrix[int]].id(1) >> op.Markov[Matrix[int]].id(1)
