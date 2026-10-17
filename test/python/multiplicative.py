# -*- coding: utf-8 -*-

import itertools
from typing import List
from pytest import raises

from discopy.biclosed import *
from discopy.python import Function, Ty, exp
from discopy.utils import AxiomError


def test_Function():
    x, y, z = Ty(complex), Ty(bool), Ty(float)
    f = Function(dom=y, cod=exp(z, x),
                 inside=lambda y: lambda x: abs(x) ** 2 if y else 0)
    g = Function(dom=x @ y, cod=z, inside=lambda x, y: f(y)(x))

    assert f.uncurry().curry()(True)(1j) == f(True)(1j)
    assert f.uncurry(left=False).curry(left=False)(True)(1j) == f(True)(1j)
    assert g.curry().uncurry()(1j, True) == g(1j, True)
    assert g.curry(left=False).uncurry(left=False)(1j, True) == g(1j, True)


def test_tensor_unit_is_none():
    """ `None` is the unit of `tensor`, so it never swallows the operands
    that follow it. """
    f, g = Function.id((int, )), Function.id((bool, ))
    assert f.tensor() == f.tensor(None) == f
    assert f.tensor(None, g).dom == f.tensor(g).dom == Ty(int, bool)
    assert f.tensor(g, None).dom == Ty(int, bool)


def test_fixed_point():
    from math import sqrt
    phi = Function(lambda x: 1 + x / (1 + x), float, float).fix()
    assert phi.dom == Ty() and phi.cod == Ty(float)
    assert phi() == (1 + sqrt(5)) / 2
    pair = Function(lambda a, x, y: (a, x), (int, int, int), (int, int))
    assert pair.fix(2).dom == Ty(int) and pair.fix(2)(3) == (3, 3)
    with raises(AxiomError):
        Function(lambda x, y: x, (int, int), str).fix()


def test_trace():
    f = Function(lambda x, y: (x + y, 1), (int, int), (int, int))
    assert f.trace().dom == Ty(int) == f.trace().cod and f.trace()(5) == 6
    g = Function(lambda x, y, z: (x, y, z), (int, int, int), (int, int, int))
    assert g.trace(2).dom == Ty(int) == g.trace(2).cod and g.trace(2)(7) == 7
    with raises(NotImplementedError):
        Function.id(int).trace(left=True)
    with raises(AxiomError):
        Function(lambda x, y: (x, "y"), (int, int), (int, str)).trace()


def test_trace_axioms():
    """ The laws of traced categories, on feedback with a unique fixed point
    so that the iteration reaches it whatever the seed. """
    X = Ty(int)
    f = Function(lambda a, x: (a + x, min(x + 1, a)), X @ X, X @ X)
    g = Function(lambda a, x, y: (a + x + y, min(x + 1, a), min(y + 1, x)),
                 X @ X @ X, X @ X @ X)
    h = Function(lambda x: x + 1, X, X)

    def eq(*fs):
        inputs = itertools.product(range(4), repeat=len(fs[0].dom))
        return all(fs[0].is_parallel(k) for k in fs) and all(
            len({k(*xs) for k in fs}) == 1 for xs in inputs)

    assert f.trace(0) == f and eq(g.trace(2), g.trace().trace())  # Vanishing
    assert eq(Function.swap(X, X).trace(), Function.id(X))  # Yanking
    assert eq((f >> X @ h).trace(), (X @ h >> f).trace())  # Sliding
    assert eq((h @ X >> f).trace(), h >> f.trace())  # Left-naturality
    assert eq((f >> h @ X).trace(), f.trace() >> h)  # Right-naturality
    assert eq((X @ f).trace(), X @ f.trace())  # Superposing


def test_list_generic_in_function():
    func = Function(sum, List[int], int)
    assert func([1, 2, 3]) == 6


def test_hash():
    f = Function(lambda x: x, int, int)
    assert hash(f) == hash(Function(f.inside, int, int)) and {f: 0}[f] == 0
    assert f != Function(f.inside, int, str)


def test_Functor_into_Function_folds_with_matmul():
    from discopy import monoidal
    from discopy.monoidal import Ty, Box, Functor

    # python.Function.ob is the free monoid List[type], so a Functor folds the
    # image of a type with the monoid product @, not tuple + (issue #728).
    assert Function.ob is monoidal.List[type]
    x = Ty('x')
    g = Box('g', x @ x, x)
    F = Functor(ob_map={x: bool}, ar_map={g: lambda a, b: a and b}, cod=Function)
    assert F(x @ x) == F(x) @ F(x) == monoidal.List[type](bool, bool)
    assert F(Ty()) == monoidal.List[type]()
    assert F(g)(True, True) is True and F(g)(True, False) is False


def test_type_checking():
    f = Function(lambda n: (n, n), Ty(int), Ty(int, int, int))
    with raises(ValueError):
        f(1, 2)  # Two arguments for a domain of one.
    with raises(RuntimeError):
        f(1)  # Two outputs for a codomain of three.


def test_fix_vanishing_and_range():
    f = Function(lambda x, y: (x + y, 1), (int, int), (int, int))
    assert f.fix(0) == f and f.trace(0) == f
    for n in (-1, 3):
        with raises(ValueError):
            f.fix(n)
        with raises(ValueError):
            f.trace(n)


def test_trace_seed():
    """ The loop of a trace starts from the default instance of its type,
    not from a default argument of the function inside, and iterates until
    two iterates are equal, as many times as it takes (#91). """
    def step(a, y=0):
        return (y, min(a, y + 1))

    def memory(a, y=7):
        return (y, y)

    f = Function(step, Ty(int, int), Ty(int, int))
    assert f.trace()(5) == 5 and f.trace()(5000) == 5000
    assert Function(memory, Ty(int, int), Ty(int, int)).trace()(5) == 0


def test_trace_uniformity():
    """ The trace is uniform along a map preserving the seed of the loop,
    e.g. a copy, as the iterates of the double of ``f`` are the copies of
    those of ``f``, but not along ``h`` with ``h(int()) == 1``. """
    A = B = M = Ty(int)
    f = Function(lambda a, m: (m, min(a, m + 1)), A @ M, B @ M)
    for n in (2, 3):
        h = Function.copy(M, n)
        g = A @ M @ Function.discard(M ** (n - 1)) >> f >> B @ h
        assert all((f >> B @ h)(a, m) == (A @ h >> g)(a, m)
                   for a, m in itertools.product(range(4), repeat=2))
        assert all(f.trace()(a) == g.trace(n)(a) for a in range(8))
    h = Function(lambda m: m + 1, M, M)
    f = Function(lambda a, m: (m, m), A @ M, B @ M)
    g = Function(lambda a, m: (m - 1, m), A @ M, B @ M)
    assert all((f >> B @ h)(a, m) == (A @ h >> g)(a, m)
               for a, m in itertools.product(range(4), repeat=2))
    assert f.trace()(0) == 0 != -1 == g.trace()(0)


def test_trace_uniformity_Square():
    """ The image of a commuting square of :class:`markov.Square` under a
    functor sending each box to a monotone Boolean function, so that the
    iteration from ``False`` reaches the least fixed point. """
    from hypothesis import given, settings, strategies as st

    from discopy import markov
    from discopy.utils import untuplify

    def monotone(box, data):
        gates = [(data.draw(st.sampled_from([any, all])), data.draw(st.lists(
            st.booleans(), min_size=len(box.dom), max_size=len(box.dom))))
            for _ in box.cod]
        return Function(lambda *xs: untuplify(tuple(
            gate(x for x, read in zip(xs, mask) if read)
            for gate, mask in gates)), len(box.dom) * (bool, ),
            len(box.cod) * (bool, ))

    @settings(max_examples=50, deadline=None)
    @given(st.data(), st.booleans())
    def check(data, written):
        f, h, g = data.draw(
            markov.Square[markov.Diagram].strategy(written=written))
        boxes = {box: monotone(box, data) for box in f.boxes}
        F = markov.Functor(
            ob_map=lambda _: bool, ar_map=boxes.__getitem__, cod=Function)
        left, right = F(f).trace(len(h.dom)), F(g).trace(len(h.cod))
        for xs in itertools.product((False, True), repeat=len(left.dom)):
            assert left(*xs) == right(*xs)
    check()
