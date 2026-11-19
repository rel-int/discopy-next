# -*- coding: utf-8 -*-

from pytest import raises

from discopy.group import Cyclic, Symmetric
from discopy.kleisli.channel import Channel
from discopy.kleisli.monad import Maybe, Powerset, Subdistribution
from discopy.monad import free
from discopy.monad.algebra import Algebra, Homomorphism, Operation, Writer
from discopy.kleisli.monad import Monad
from discopy.python.function import Function


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


def test_Algebra_product_and_power_satisfy_the_laws():
    disjunction = Algebra[Powerset]("any", bool, any)
    conjunction = Algebra[Powerset]("all", bool, all)
    both = Algebra[Powerset].product(disjunction, conjunction)
    assert both.carrier == tuple[bool, bool] and str(both) == "any × all"
    pairs = [(x, y) for x in (False, True) for y in (False, True)]
    assert all(both.unitality(pair) for pair in pairs)
    assert both.associativity(frozenset({
        frozenset(pairs[:2]), frozenset(pairs[1:]), frozenset()}))
    assert both(frozenset()) == (False, True)
    cube = zero.power(3)
    assert cube == Algebra[Maybe].product(zero, zero, zero)
    assert cube.unitality((1, 2, 3)) and cube.associativity(None)
    assert cube(None) == (0, 0, 0) and cube((1, 2, 3)) == (1, 2, 3)
    unit = Algebra[Maybe].product()
    assert (str(unit), unit.carrier, unit(None)) == ("1", tuple[()], ())
    with raises(TypeError):
        Algebra[Maybe].product(zero, disjunction)
    with raises(ValueError):
        zero.power(-1)


def test_Algebra_product_projections_are_homomorphisms():
    pair = Algebra[Maybe].product(zero, one)
    for i, factor in enumerate((zero, one)):
        projection = Homomorphism[Maybe](
            lambda xs, i=i: xs[i], pair, factor)
        assert projection.preservation(None)
        assert projection.preservation((5, 7))


def test_Operation_acts_on_every_algebra():
    choice = Operation[Powerset](frozenset({0, 1}), 2)
    assert choice(Algebra[Powerset]("any", bool, any))((False, True))
    assert not choice(Algebra[Powerset]("all", bool, all))((False, True))
    assert choice(Algebra[Powerset].free(int))(
        (frozenset({1}), frozenset({2}))) == frozenset({1, 2})
    default = Operation[Maybe](None, 0)
    assert default(zero)(()) == 0 and default(one)(()) == 1
    assert choice == Operation[Powerset](frozenset({1, 0}), 2) != default
    assert hash(choice) == hash(Operation[Powerset](frozenset({0, 1}), 2))
    assert repr(choice)\
        == "monad.algebra.Operation[Powerset](frozenset({0, 1}), 2)"
    with raises(TypeError):
        choice(zero)


def test_Operation_commutes_with_homomorphisms():
    disjunction = Algebra[Powerset]("any", bool, any)
    conjunction = Algebra[Powerset]("all", bool, all)
    negation = Homomorphism[Powerset](
        lambda x: not x, disjunction, conjunction)
    for t in (frozenset(), frozenset({0}), frozenset({0, 2})):
        operation = Operation[Powerset](t, 3)
        assert all(operation.naturality(negation, (x, y, False))
                   for x in (False, True) for y in (False, True))
    increment = Homomorphism[Maybe](lambda x: x + 1, zero, one)
    assert Operation[Maybe](None, 2).naturality(increment, (3, 4))
    assert Operation[Maybe](1, 2).naturality(increment, (3, 4))
    assert not Operation[Maybe](None, 0).naturality(
        Homomorphism[Maybe](lambda x: x + 1, zero, zero), ())
    with raises(TypeError):
        Operation[Powerset](frozenset(), 0).naturality(increment, ())


def test_Writer_is_a_monad():
    W = Writer(Symmetric[3])
    g, h = Symmetric[3]((1, 2, 0)), Symmetric[3]((1, 0, 2))
    assert W == Writer(Symmetric[3]) != Writer(Cyclic[3])
    assert repr(W) == "Writer(group.Symmetric[3])"
    assert W.unit(str)("egg") == (Symmetric[3].id(), "egg")
    assert W.mult(str)((g, (h, "egg"))) == (h >> g, "egg")
    assert W(Function(len, str, int))((g, "egg")) == (g, 3)


def test_Writer_algebras_are_actions():
    W = Writer(Symmetric[3])
    elements = list(Symmetric[3].elements())
    for action in (Symmetric[3].action(), Symmetric[3].action().tensor(
            Symmetric[3].action())):
        algebra = W.from_action(action)
        points = range(action.size)
        assert W.to_action(algebra, action.size) == action
        assert all(algebra.unitality(x) for x in points)
        assert all(algebra.associativity((g, (h, x)))
                   for g in elements for h in elements for x in points)
    sign = Algebra[W](
        "sign", int, lambda gx: (gx[1] + len(gx[0].word())) % 2)
    assert all(sign.associativity((g, (h, x)))
               for g in elements for h in elements for x in range(2))
    back = W.from_action(W.to_action(sign, 2))
    assert back != sign and all(
        back((g, x)) == sign((g, x)) for g in elements for x in range(2))
    with raises(TypeError):
        W.from_action(Cyclic[3].action())
    with raises(TypeError):
        W.to_action(zero, 3)


def test_Writer_homomorphisms_are_equivariant():
    W = Writer(Symmetric[3])
    elements = list(Symmetric[3].elements())
    points = W.from_action(Symmetric[3].action())
    pairs = W.from_action(Symmetric[3].action().tensor(
        Symmetric[3].action()))
    first = Homomorphism[W](lambda xy: xy // 3, pairs, points)
    assert all(first.preservation((g, xy))
               for g in elements for xy in range(9))
    constant = Homomorphism[W](lambda xy: 0, pairs, points)
    assert not all(constant.preservation((g, xy))
                   for g in elements for xy in range(9))


def test_Writer_hashes_as_the_monad_it_equals():
    W = Writer(Symmetric[3])
    def identity(x):
        return x

    same = Monad(W.name, identity, identity, identity, identity)
    assert W == same and hash(W) == hash(same)
