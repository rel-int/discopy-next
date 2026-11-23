# -*- coding: utf-8 -*-

from pytest import raises

from discopy import cat
from discopy.enriched import Arrow, Functor
from discopy.kleisli.monad import List, Maybe, Powerset, Subdistribution
from discopy.matrix import Matrix
from discopy.monoidal import Box, Diagram, Ty
from discopy.python import Function
from discopy.sum import Sum
from discopy.utils import AxiomError

x, y, z = map(cat.Ob, "xyz")
f, g = cat.Box('f', x, y), cat.Box('g', x, y)
h, k = cat.Box('h', y, z), cat.Box('k', y, z)
Sums = Arrow[cat.Arrow, List]


def test_List():
    assert List.unit(int)(1) == [1]
    assert List.mult(int)([[1], [], [2, 3]]) == [1, 2, 3]
    assert List(Function(lambda n: 2 * n, int, int))([1, 2]) == [2, 4]


def test_then_is_the_list_of_composites():
    m, n = Sums([f, g], x, y), Sums([h, k], y, z)
    assert (m >> n) == Sums([f >> h, f >> k, g >> h, g >> k], x, z)
    assert Sums.id(x) == Sums([cat.Id(x)], x, x)
    assert Sums.id(x) >> m == m == m >> Sums.id(y)
    assert (Sums([], x, y) >> n) == Sums([], x, z)
    with raises(AxiomError):
        m >> m
    with raises(TypeError):
        m >> Arrow[cat.Arrow, Maybe](h, y, z)


def test_Maybe_arrow_composes_or_fails():
    Partial = Arrow[cat.Arrow, Maybe]
    assert (Partial(f, x, y) >> Partial(h, y, z)).inside == f >> h
    assert (Partial(None, x, y) >> Partial(h, y, z)).inside is None
    assert (Partial(f, x, y) >> Partial(None, y, z)).inside is None


def test_Function_arrows_compose_pointwise():
    succ = Function(lambda n: n + 1, int, int)
    double = Function(lambda n: 2 * n, int, int)
    Lists = Arrow[Function, List]
    m = Lists([succ, double], int, int) >> Lists([succ], int, int)
    assert [function(3) for function in m.inside] == [5, 7]
    assert m.to_channel()(3) == [5, 7]


def test_extension_to_cat_is_the_sum():
    F = Functor(lambda x: x, lambda f: f, dom=Sums, cod=cat.Arrow)
    m, n = Sums([f, g], x, y), Sums([h, k], y, z)
    assert F(m) == Sum((f, g)) == f + g
    assert F(Sums([], x, y)) == cat.Arrow.zero(x, y)
    assert F(m >> n) == F(m) >> F(n)\
        == Sum((f >> h, f >> k, g >> h, g >> k))
    assert F(Sums.id(x)) == cat.Id(x)


def test_extension_to_Matrix_sums_the_images():
    """
    With ``f = [[1, 2]]``, ``g = [[0, 1]]``, ``h = [[1], [1]]`` and
    ``k = [[2], [0]]``: ``f + g = [[1, 3]]`` and ``h + k = [[3], [1]]``, so the
    image of the composite is ``1 * 3 + 3 * 1 = 6``, the sum of the four
    composites ``3 + 2 + 1 + 0``.
    """
    ar_map = {f: [1, 2], g: [0, 1], h: [1, 1], k: [2, 0]}
    F = Functor({x: 1, y: 2, z: 1}, ar_map, dom=Sums, cod=Matrix[int])
    G = cat.Functor({x: 1, y: 2, z: 1}, ar_map, cod=Matrix[int])
    m, n = Sums([f, g], x, y), Sums([h, k], y, z)
    assert F(m) == Matrix[int]([1, 3], 1, 2) == G(f + g)
    assert F(n) == Matrix[int]([3, 1], 2, 1) == G(h + k)
    assert F(m >> n) == Matrix[int]([6], 1, 1) == F(m) >> F(n)
    assert F(Sums([], x, y)) == Matrix[int].zero(1, 2)
    assert F(Sums.id(y)) == Matrix[int].id(2)


def test_algebra_of_another_monad():
    for category in (cat.Arrow, Matrix[int]):
        with raises(TypeError):
            category.algebra(Powerset, x, y)
    F = Functor(lambda x: x, lambda f: f,
                dom=Arrow[cat.Arrow, Powerset], cod=cat.Arrow)
    with raises(TypeError):
        F(Arrow[cat.Arrow, Powerset](frozenset({f}), x, y))


def test_algebra_laws():
    total = Matrix[int].algebra(List, 1, 1)
    one, two = Matrix[int]([1], 1, 1), Matrix[int]([2], 1, 1)
    assert total.unitality(two)
    assert total.associativity([[one, two], [], [two]])
    assert total([one, two, two]) == Matrix[int]([5], 1, 1)


def test_tensor_of_coins():
    a, b = Ty('a'), Ty('b')
    u, v, w = Box('u', a, b), Box('v', a, b), Box('w', b, a)
    Coins = Arrow[Diagram, Subdistribution]
    coin = Coins(frozenset({(u, .5), (v, .5)}), a, b)
    sure = Coins(frozenset({(w, 1.)}), b, a)
    assert coin @ sure == Coins(
        frozenset({(u @ w, .5), (v @ w, .5)}), a @ b, b @ a)
    assert (coin @ coin).inside == frozenset({
        (u @ u, .25), (u @ v, .25), (v @ u, .25), (v @ v, .25)})


def test_to_channel_is_a_functor():
    succ = Function(lambda n: n + 1, int, int)
    double = Function(lambda n: 2 * n, int, int)
    Coins = Arrow[Function, Subdistribution]
    m = Coins(frozenset({(succ, .5), (double, .5)}), int, int)
    n = Coins(frozenset({(succ, 1.)}), int, int)
    assert (m >> n).to_channel()(3) == frozenset({(5, .5), (7, .5)})\
        == (m.to_channel() >> n.to_channel())(3)
    assert Coins.id(int).to_channel()(3) == frozenset({(3, 1.)})


def test_to_channel_of_another_category():
    with raises(TypeError):
        Sums([f], x, y).to_channel()
