from itertools import product
from operator import matmul

import pytest

import discopy
from discopy import markov, para
from discopy.actegory import Action, Para, dirac
from discopy.kleisli.monad import Maybe, Subdistribution, merge
from discopy.kleisli.multiplicative import Channel
from discopy.python.multiplicative import Function, Ty
from discopy.utils import AxiomError, untuplify

Stochastic = Channel[Subdistribution]
ACT = Action.deterministic(Stochastic)
B = Ty(bool)
neg = Function(lambda b: not b, bool, bool)
const = Function(lambda b: True, bool, bool)
copy = Function.copy(B)
coin = Stochastic(lambda: frozenset({(False, .5), (True, .5)}), (), bool)


def gate(bias: float) -> Channel:
    """
    A channel reading the data and a weight, ``x and not w`` flipped with
    ``bias``: asymmetric in its two arguments, unlike a function of ``x != w``,
    so that swapping data and weight, or the order of two weights, shows.
    """
    return Stochastic(lambda x, w: frozenset(
        {(x and not w, 1 - bias), (not (x and not w), bias)}),
        (bool, bool), bool)


def dist(channel, *xs) -> dict:
    """ The distribution of a channel on a point, as a dict. """
    return dict(merge(channel(*xs)))


def close(left: dict, right: dict) -> bool:
    """ Two distributions equal up to ``1e-9``. """
    return left.keys() == right.keys() and all(
        abs(left[x] - right[x]) < 1e-9 for x in left)


def smear(bias: float) -> Channel:
    """ A channel on the data alone, flipped with ``bias``. """
    return Stochastic(
        lambda x: frozenset({(x, 1 - bias), (not x, bias)}), bool, bool)


def layers():
    return [Para(B, B, gate(bias), B, ACT) for bias in (.1, .2, .3)]


def negate(ty: Ty) -> Function:
    return Function(lambda *xs: untuplify(tuple(not x for x in xs)), ty, ty)


SHAPES = [
    ("x", "y", "z", "p", "q"), ("xy", "z", "x", "pq", "s"),
    ("", "x", "y", "p", ""), ("x", "x", "", "", "p")]


def symmetric(f: Para) -> para.Markov:
    return para.Markov[markov.Diagram](f.dom, f.cod, f.inside, f.param)


@pytest.mark.parametrize("shape", SHAPES)
def test_self_action_is_para(shape):
    """ Para of the self-action is ``para``, whose copar is empty here. """
    a, b, c, P, Q = (markov.Ty(*name) for name in shape)
    act = Action.right(markov.Diagram)
    f = Para(a, b, markov.Box("f", a @ P, b), P, act)
    g = Para(b, c, markov.Box("g", b @ Q, c), Q, act)
    r = markov.Box("r", markov.Ty("s"), P)
    rr = markov.Box("rr", markov.Ty("s"), P @ Q)
    unit = Para.id(act, a)
    Markov = para.Markov[markov.Diagram]
    assert (f >> g).param == P @ Q == symmetric(f >> g).param
    assert Markov.equation_factory(
        symmetric(f >> g), symmetric(f) >> symmetric(g))
    assert Markov.equation_factory(symmetric(unit), Markov.id(a))
    assert Markov.equation_factory(
        symmetric(unit >> f), Markov.id(a) >> symmetric(f))
    assert Markov.equation_factory(
        symmetric(f.reparam(r)), symmetric(f).reparam(r))
    assert Markov.equation_factory(
        symmetric((f >> g).reparam(rr)),
        (symmetric(f) >> symmetric(g)).reparam(rr))
    assert act.unitality(f.inside) and act.multiplicativity(f.inside, r, r)
    env = {**vars(discopy), "Para": Para, "Action": Action}
    assert eval(repr(f), env) == f


def test_action_laws():
    f, g = smear(.3), smear(.1)
    assert ACT.unitality(gate(.1)) and ACT.unitality(f)
    assert ACT.multiplicativity(f, neg, copy)
    assert ACT.multiplicativity(gate(.1), copy, neg)
    assert ACT.functoriality(f, g, neg, neg)
    assert ACT.functoriality(f, g, Function.id(B), copy)
    assert ACT.functoriality(f, g, neg, const)
    assert ACT.identities(B, B) and ACT.identities(Ty(bool, bool), Ty())


def test_dirac_acts_deterministically():
    f = smear(.25)
    assert dirac(f, neg)(True, True) == frozenset({
        ((True, False), .75), ((False, False), .25)})
    assert dirac(f, Function.discard(B))(True, False) == f(True)


def test_object_laws():
    """ ``X ◁ I = X`` and ``(X ◁ P) ◁ Q = X ◁ (P @ Q)`` on objects. """
    x, P, Q = Ty(bool), Ty(int), Ty(str)
    assert ACT.ob(x, Ty()) == x
    assert ACT.ob(ACT.ob(x, P), Q) == ACT.ob(x, P @ Q)


def negated(f, r):
    return dirac(f, r) >> dirac(Stochastic.id(Ty()), negate(f.cod @ r.cod))


def twisted(f, r):
    return dirac(f, r >> negate(r.cod))


def doubled(f, r):
    return dirac(dirac(f, r), r)


def test_non_actions_are_caught():
    """ Each non-action breaks the laws it should and keeps the others. """
    f, g, r = smear(.3), smear(.1), Function.id(B)
    assert not Action(Function, Stochastic, matmul, negated).unitality(
        gate(.1))
    twist = Action(Function, Stochastic, matmul, twisted)
    assert ACT.functoriality(f, g, r, r)
    assert not twist.functoriality(f, g, r, r)
    assert twist.unitality(f) and twist.multiplicativity(f, neg, neg)
    assert not twist.identities(B, B)


def test_non_multiplicative_action():
    """ ``X ◁ P = X @ P @ P`` is only not multiplicative, even on objects. """
    fat = Action(
        Function, Stochastic, lambda x, p: x @ p @ p, doubled)
    f, g = smear(.3), smear(.1)
    assert fat.unitality(f) and fat.identities(B, B)
    assert fat.functoriality(f, g, neg, neg)
    assert fat.functoriality(f, g, neg, const)
    assert not fat.multiplicativity(f, neg, copy)
    x, P, Q = Ty(bool), Ty(int), Ty(str)
    assert fat.ob(x, Ty()) == x
    assert fat.ob(fat.ob(x, P), Q) != fat.ob(x, P @ Q)


def two_layers(first: float, second: float):
    """ Two gates given two weights by hand, as a distribution. """
    one, two = gate(first), gate(second)
    return lambda x, w, v: dict(merge(
        (z, p * q) for y, p in one(x, w) for z, q in two(y, v)))


def test_untied_weights_are_distinct():
    one, two, _ = layers()
    untied, by_hand = one >> two, two_layers(.1, .2)
    assert untied.param == B @ B
    for xs in product([False, True], repeat=3):
        assert close(dist(untied.inside, *xs), by_hand(*xs))
    swapped = (two >> one).inside
    assert any(
        not close(dist(swapped, *xs), by_hand(*xs))
        for xs in product([False, True], repeat=3))


def test_tying_is_copy_in_parameters():
    one, two, _ = layers()
    untied = one >> two
    tied = untied.reparam(copy)
    by_hand = two_layers(.1, .2)
    assert tied.param == B
    for x, w in product([False, True], repeat=2):
        assert close(dist(tied.inside, x, w), by_hand(x, w, w))
    assert Para.equation_factory(tied, tied.reparam(Function.id(B)))


def test_reparam_is_contravariant():
    one, _, _ = layers()
    stacked = one.reparam(neg).reparam(const)
    assert Para.equation_factory(stacked, one.reparam(const >> neg))
    assert not Para.equation_factory(stacked, one.reparam(neg >> const))


def test_weight_sampled_once_or_twice():
    one, two, _ = layers()
    untied, tied = one >> two, (one >> two).reparam(copy)
    skip = Stochastic.id(B)
    once = skip @ coin >> tied.inside
    copied = skip @ (coin >> Stochastic.copy(B)) >> untied.inside
    twice = skip @ coin @ coin >> untied.inside
    for x in (False, True):
        assert close(dist(once, x), dist(copied, x))
    assert any(
        not close(dist(once, x), dist(twice, x)) for x in (False, True))


def test_copy_is_natural_only_for_deterministic_maps():
    assert Function.equation_factory(neg >> copy, copy >> neg @ neg)
    assert Function.equation_factory(const >> copy, copy >> const @ const)
    discard = Function.discard(B)
    assert Function.equation_factory(
        discard >> Function.copy(Ty()), copy >> discard @ discard)
    once = coin >> Stochastic.copy(B)
    assert not Stochastic.equation_factory(once, coin @ coin)
    noise = smear(.3)
    assert not Stochastic.equation_factory(
        noise >> Stochastic.copy(B), Stochastic.copy(B) >> noise @ noise)


def test_composition_laws():
    one, two, three = layers()
    unit = Para.id(ACT, B)
    assert Para.equation_factory((one >> two) >> three, one >> (two >> three))
    assert (one >> two >> three).param == B @ B @ B
    assert Para.equation_factory(unit >> one, one, one >> unit)
    assert not Para.equation_factory(one >> two, one >> one)
    both = one >> two
    assert Para.equation_factory(
        both.reparam(neg >> copy), both.reparam(copy).reparam(neg))


def test_errors():
    one, *_ = layers()
    integers = Para.id(ACT, Ty(int))
    with pytest.raises(AxiomError):
        one >> integers
    with pytest.raises(AxiomError):
        one >> Para.id(Action.deterministic(Channel[Maybe]), B)
    with pytest.raises(TypeError):
        one.reparam(smear(.1))
    with pytest.raises(AxiomError):
        one.reparam(copy)
    with pytest.raises(AxiomError):
        Para(B, B, gate(.1), Ty(), ACT)
    with pytest.raises(TypeError):
        Para(B, B, gate(.1), bool, ACT)


def test_repr():
    """
    The actions round-trip by their named constructors, so that two separately
    built ones are equal. A ``Para`` of channels holds lambdas, which have no
    repr to evaluate: its round trip is that of the self-action above.
    """
    env = {**vars(discopy), "Action": Action, "Maybe": Maybe,
           "Subdistribution": Subdistribution}
    for act in (Action.right(markov.Diagram), ACT, Action.deterministic(
            Channel[Maybe])):
        assert eval(repr(act), env) == act
    assert repr(ACT) == "Action.deterministic(" \
        "kleisli.multiplicative.Channel[Subdistribution])"
    assert Action.right(markov.Diagram) != ACT
