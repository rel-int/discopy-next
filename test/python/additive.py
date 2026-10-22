# -*- coding: utf-8 -*-


def test_additive_Function():
    from discopy.interaction import Ty, Diagram
    from discopy.python.additive import Function, Id, Swap, Merge, Noise

    X, xs = Function.ob(int), []
    m, e = Function.merge(X, n=2), Function.noise(X)
    assert Merge == Function.merge and Noise == Function.noise
    assert (e.dom, e.cod) == (Function.ob(), X)

    def f_inside(m, n=0):
        xs.append(m)
        return 3 * m + 1 if m % 2 else m // 2, 0 if n == 1 and m == 2 else 1

    f = Function(f_inside, X @ X, X @ X)
    g = Function(lambda m: m // 2, X, X)

    # This converges if https://en.wikipedia.org/wiki/Collatz_conjecture holds.
    assert f.trace()(42) == 1 and xs == [42, 21, 64, 32, 16, 8, 4, 2]

    eq = lambda *fs: all(fs[0].is_parallel(f) for f in fs) and all(
        len(set(f(42, i) for f in fs)) == 1 for i in range(len(fs[0].dom)))

    assert eq(Swap(X, X) >> m, m)
    assert eq(X @ e >> m, Id(X), e @ X >> m)
    assert eq(m @ X >> m, X @ m >> m, Function.merge(X, n=3))
    assert eq(Function.merge(X @ X), X @ Swap(X, X) @ X >> m @ m)

    assert eq(Swap(X, X).trace(), Id(X))  # Yanking
    assert eq((f >> X @ g).trace(), (X @ g >> f).trace())  # Sliding
    assert eq((g @ X >> f).trace(), g >> f.trace())  # Left-naturality
    assert eq((f >> g @ X).trace(), f.trace() >> g)  # Right-naturality

    IntTy, D = Ty[Function.ob], Diagram[Function]
    assert eq(D.id(IntTy(X, X)).transpose().inside, Id(X @ X))


def test_trace_unequal_arity():
    from discopy.python.additive import Function

    # f : A + U -> B + C + U, traced over U: tag 0 (A) loops back to U,
    # any other tag (C) exits. len(dom) == 2 != 3 == len(cod), so the
    # traced summand sits at a different tag on each side.
    def inside(obj, tag=0):
        return obj, 2 if tag == 0 else 1

    f = Function(inside, (int, int), (int, int, int))
    assert f.trace()(7) == (7, 1)


def test_trace_takes_the_keyword_form():
    from pytest import raises
    from discopy.python.additive import Function
    from discopy.utils import AxiomError

    def inside(obj, tag=0):
        return obj, 2 if tag == 0 else 1

    f = Function(inside, (int, int), (int, int, int))
    assert f.trace(mem=int)(7) == f.trace(cod=(int, int))(7) == (7, 1)
    with raises(TypeError):
        f.trace(1, mem=int)
    with raises(AxiomError):
        f.trace(mem=str)


def test_Hypergraph_pass_token():
    from pytest import raises
    from discopy.comarkov import Ty, Box, Diagram
    from discopy.hypergraph import Functor, Hypergraph
    from discopy.python.additive import Function
    from discopy.utils import AxiomError

    x, X = Ty('x'), Function.ob(int)
    f, g = Box('f', x @ x, x @ x), Box('g', x, x)
    seen = []

    def step(n, tag):
        seen.append((n, tag))
        return (n, 0) if n == 1 else (3 * n + 1 if n % 2 else n // 2, 1)

    F = Functor({x: int}, {
        f: Function(step, X @ X, X @ X),
        g: Function(lambda n: n + 1, X, X)}, cod=Function)

    # The trace is absorbed into the wiring, run as a while loop.
    loop = F(f.trace().to_hypergraph())
    assert len(loop.boxes) == 1 and not loop.is_causal
    assert loop(6) == F(f).trace()(6) == 1
    assert seen[:9] == [(6, 0), (3, 1), (10, 1), (5, 1), (16, 1),
                        (8, 1), (4, 1), (2, 1), (1, 1)]

    # The merge of two branches, either of which the token takes.
    merged = F((g @ x >> Diagram.merge(x)).to_hypergraph())
    assert merged(1, 0) == 2 and merged(1, 1) == 1

    # The domain is a disjoint union: one token, one wire.
    assert F((g @ g).to_hypergraph())(1, 1) == (2, 1)

    # A wire consumed twice is a copy, which a disjoint union does not have.
    copy = Hypergraph[Function](X, X @ X, (), ((0, ), (), (0, 0)))
    with raises(AxiomError, match="is not right-monogamous"):
        copy(1)
