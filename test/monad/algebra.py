# -*- coding: utf-8 -*-

from pytest import raises

from discopy.kleisli.channel import Channel
from discopy.kleisli.monad import Maybe, Powerset, Subdistribution
from discopy.monad import free
from discopy.monad.algebra import Algebra, Homomorphism


zero = Algebra[Maybe]("zero", int, lambda x: 0 if x is None else x)
one = Algebra[Maybe]("one", int, lambda x: 1 if x is None else x)


def test_Algebra_of_Maybe_is_a_pointed_type():
    assert zero(None) == 0 and zero(5) == 5
    assert all(zero.unitality(x) for x in range(3))
    assert all(zero.associativity(x) for x in (None, 0, 5))


def test_Algebra_eq_and_hash():
    assert zero == Algebra[Maybe]("zero", int, lambda x: x or 0)
    assert hash(zero) == hash(Algebra[Maybe]("zero", int, lambda x: x or 0))
    assert zero != one != Algebra[Powerset]("one", int, sum) != "one"
    assert Algebra[Maybe].free(int) == Algebra[Maybe].free(int)
    assert repr(zero) == "monad.algebra.Algebra[Maybe]('zero', int)"
    assert str(zero) == "zero"


def test_Algebra_free_satisfies_the_laws():
    """ The laws of the free algebra are the laws of the monad. """
    nested = frozenset({
        frozenset({frozenset({1}), frozenset()}),
        frozenset({frozenset({2, 3})})})
    assert Algebra[Powerset].free(int).associativity(nested)
    assert Algebra[Powerset].free(int).unitality(frozenset({1, 2}))


def test_Algebra_law_fails():
    """ The sum of a set is no powerset algebra: union forgets duplicates. """
    total = Algebra[Powerset]("sum", int, sum)
    assert total.unitality(5)
    assert not total.associativity(frozenset({
        frozenset({1}), frozenset({1, 2})}))


def test_Homomorphism_of_pointed_types():
    Pointed = Homomorphism[Maybe]
    double = Pointed(lambda x: 2 * x, zero, zero)
    increment = Pointed(lambda x: x + 1, zero, one)
    assert all(double.preservation(x) for x in (None, 3))
    assert increment.preservation(None) and increment.preservation(3)
    assert not Pointed(lambda x: x + 1, zero, zero).preservation(None)
    assert (Pointed.id(zero) >> double >> increment)(3) == 7
    with raises(TypeError):
        Pointed(lambda x: x, zero, Algebra[Powerset]("any", bool, any))
    with raises(TypeError):
        double >> free.Homomorphism[Maybe](lambda x: x, int, int)
    assert repr(Pointed.id(zero)).startswith(
        "monad.algebra.Homomorphism[Maybe](")


def test_Homomorphism_from_free():
    """
    The inclusion of free algebras is a functor: it sends the free extension
    of a channel to a homomorphism, and composites to composites.
    """
    Free = free.Homomorphism[Subdistribution]
    Kleisli = Channel[Subdistribution]
    flip = Kleisli(lambda x: frozenset({(x, .5), (not x, .5)}), bool, bool)
    step = Homomorphism[Subdistribution].from_free(Free.from_channel(flip))
    mm = frozenset({
        (frozenset({(True, 1.)}), .5), (frozenset({(False, .5)}), .25)})
    assert step.preservation(mm)
    assert step.dom == step.cod == Algebra[Subdistribution].free(bool)
    assert (step >> step)(frozenset({(True, 1.)}))\
        == Homomorphism[Subdistribution].from_free(
            Free.from_channel(flip) >> Free.from_channel(flip))(
                frozenset({(True, 1.)}))
    with raises(TypeError):
        Homomorphism[Maybe].from_free(Free.from_channel(flip))
