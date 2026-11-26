# -*- coding: utf-8 -*-

import time

from pytest import raises

from discopy import closed, comarkov, compact, frobenius, markov
from discopy.para import (
    BOUND, Cell, Closed, Comarkov, Compact, Feedback, Hypergraph, Markov,
    Symmetric, Traced, UpToPermutation, permutations, same)
from discopy import python
from discopy.python import Function
from discopy.symmetric import Box, Diagram, Ty
from discopy.utils import AxiomError

x, y, z, p, q = map(Ty, "xyzpq")
f = Symmetric(x, y, Box('f', x @ p, y), p)
g = Symmetric(y, z, Box('g', y @ q, z), q)


def test_errors():
    with raises(AxiomError):
        Symmetric(x, y, Box('f', x @ p, y), q)
    with raises(AxiomError):
        Symmetric(x, z, Box('f', x @ p, y), p)
    with raises(AxiomError):
        g >> f
    with raises(AxiomError):
        f.reparam(Box('r', q, q))


def test_symmetric_axioms():
    assert f >> Symmetric.id(y) == f == Symmetric.id(x) >> f
    assert (f @ g.dom >> f.cod @ g).is_parallel(f @ g)
    swap = Symmetric.swap(x, y)
    assert (swap >> Symmetric.swap(y, x)).inside.simplify()\
        == Diagram.id(x @ y)
    assert Symmetric.permutation((1, 0), (x, y)).inside.simplify()\
        == swap.inside
    assert Symmetric.braid(x, y) == swap


def test_permutation():
    perm = Symmetric.permutation([1, 2, 0], (x, y, z))
    assert perm.inside == Diagram.permutation([1, 2, 0], x @ y @ z)
    assert perm.param == perm.copar == Ty()
    i, s, fl = python.Ty(int), python.Ty(str), python.Ty(float)
    perm = Symmetric[Function].permutation([1, 2, 0], (i, s, fl))
    assert perm.dom == i @ s @ fl and perm.cod == s @ fl @ i
    assert perm.inside(1, "a", 2.5) == ("a", 2.5, 1)


def test_trace():
    t = Traced(x @ y, z @ y, Box('t', x @ y @ p, z @ y), p)
    assert t.trace(0) == t
    inside = x @ Diagram.swap(p, y) >> t.inside
    assert t.trace() == Traced(x, z, inside.trace(), p)
    u = Traced(y @ x, y @ z, Box('u', y @ x @ p, y @ z), p)
    assert u.trace(left=True) == Traced(x, z, u.inside.trace(left=True), p)


def test_reparam():
    r, s = Box('r', q, p), Box('s', p, q)
    assert f.reparam(r).param == q
    assert f.reparam(r).reparam(s) == f.reparam(s >> r)
    assert (f @ g).reparam(r @ s).param == q @ p


def test_markov():
    X = markov.Ty('x')
    assert Markov.copy(X, 3) == Markov.lift(markov.Diagram.copy(X, 3))
    assert Markov.copy(X).param == markov.Ty()


def test_comarkov():
    X = comarkov.Ty('x')
    assert Comarkov.merge(X, 3) == Comarkov.lift(comarkov.Diagram.merge(X, 3))
    assert Comarkov.noise(X) == Comarkov.lift(comarkov.Noise(X))
    assert Comarkov.merge(X).param == comarkov.Ty()
    X = frobenius.Ty('x')
    assert Hypergraph.merge(X) == Hypergraph.lift(frobenius.Spider(2, 1, X))


def test_closed():
    a, b, c, P = map(closed.Ty, "abcP")
    k = Closed(a @ b, c, closed.Box('k', a @ b @ P, c), P)
    assert Closed.ev(c, b).param == closed.Ty()
    left, right = k.curry(left=True), k.curry(left=False)
    assert k.curry() == left
    assert k.curry(0) == k == k.curry(0, left=False)
    assert (left.dom, left.cod, left.param) == (a, c << b, P)
    assert (right.dom, right.cod, right.param) == (b, a >> c, P)
    assert left.inside\
        == (a @ closed.Diagram.swap(P, b) >> k.inside).curry(left=True)
    with raises(ValueError):
        Closed(a @ b, c, closed.Box('k', a @ b, c @ P), copar=P).curry()


def test_closed_curry_stays_in_the_domain():
    a, b, c, P = map(closed.Ty, "abcP")
    k = Closed(a @ b, c, closed.Box('k', a @ b @ P, c), P)
    for n, left in [(3, True), (3, False), (-1, True), (-1, False)]:
        with raises(ValueError, match="off a @ b"):
            k.curry(n, left=left)


def test_feedback():
    x, y, z, P = map(markov.Ty, "xyzP")
    f = Feedback(x @ y.delay(), z @ y,
                 markov.Box('f', x @ y.delay() @ P, z @ y), P)
    fb = f.feedback()
    assert (fb.dom, fb.cod, fb.param) == (x, z, P)
    assert fb == f.feedback(1) == f.feedback(mem=y) == f.feedback(cod=z)
    assert f.delay().param == P.delay()
    with raises(NotImplementedError):
        f.feedback(left=True)


def test_compact():
    x, y = map(compact.Ty, "xy")
    assert Compact.cups(x, x.r) == Compact.lift(compact.Diagram.cups(x, x.r))
    assert Compact.caps(x, x.l).param == compact.Ty()
    k = Compact(x @ y, x,
                compact.Box('k', x @ y @ compact.Ty('P'), x),
                compact.Ty('P'))
    assert k.curry(left=True).cod == x << y
    P = compact.Ty('P')
    h = Compact(x @ y, x, compact.Box('h', x @ y, x @ P), copar=P)
    assert (h.curry().cod, h.curry().copar) == (x << y, P)


def test_hypergraph():
    X = frobenius.Ty('x')
    assert Hypergraph.spiders(1, 2, X)\
        == Hypergraph.lift(frobenius.Diagram.spiders(1, 2, X))


def test_python():
    f, ff = python.Ty(float), python.Ty(float, float)
    inside = Function(lambda a, w, b: w * a + b, f ** 3, f)
    layer = Symmetric[Function](f, f, inside, ff)
    with raises(TypeError):
        Symmetric[Function]((float, ), (float, ), inside, (float, float))
    pair = layer @ layer
    assert pair.inside(1., 2., 10., 0., 3., 5.) == (10., 11.)


def test_copar():
    m, n = Ty('m'), Ty('n')
    t = Symmetric(x, y, Box('t', x @ p, y @ m), p, m)
    with raises(AxiomError):
        Symmetric(x, y, Box('t', x @ p, y @ m), p, n)
    with raises(AxiomError):
        Symmetric(x, y, Box('t', x @ p, y @ m), p)
    with raises(AxiomError):
        t.recopar(Box('c', n, n))
    assert t >> Symmetric.id(y) == t == Symmetric.id(x) >> t
    assert t.reparam(Box('r', q, p)).copar == m
    assert t.recopar(Box('c', m, n)).param == p
    assert (t @ t).param == p @ p and (t @ t).copar == m @ m


def test_copar_trace():
    m = Ty('m')
    t = Traced(x @ y, z @ y, Box('t', x @ y @ p, z @ y @ m), p, m)
    trace = t.trace()
    assert (trace.dom, trace.cod, trace.param, trace.copar) == (x, z, p, m)
    u = Traced(y @ x, y @ z, Box('u', y @ x @ p, y @ z @ m), p, m)
    assert u.trace(left=True).copar == m


def test_copar_feedback():
    x, y, z, m, P = map(markov.Ty, "xyzmP")
    f = Feedback(x @ y.delay(), z @ y,
                 markov.Box('f', x @ y.delay() @ P, z @ y @ m), P, m)
    fb = f.feedback()
    assert (fb.dom, fb.cod, fb.param, fb.copar) == (x, z, P, m)
    assert f.delay().copar == m.delay()


def test_copar_python():
    f = python.Ty(float)
    add = Function(lambda a, s: (a + s, a), f ** 2, f ** 2)
    cell = Symmetric[Function](f, f, add, f, f)
    network = cell >> cell
    assert network.param == network.copar == f ** 2
    assert network.inside(2., 1., 10.) == (13., 2., 3.)


def terms():
    m, X = Ty('m'), frobenius.Ty('x')
    a, b, c, P = map(closed.Ty, "abcP")
    return (f, f >> g, f @ g, Symmetric.id(x),
            Symmetric(x, y, Box('t', x @ p, y @ m), p, m),
            Hypergraph.spiders(1, 2, X),
            Closed(a @ b, c, closed.Box('k', a @ b @ P, c), P))


def test_tree():
    from discopy.utils import dumps, loads
    for term in terms():
        loaded = loads(dumps(term))
        assert type(loaded) is type(term) and loaded == term


def test_hash():
    for term in terms():
        assert hash(term) == hash(type(term)(*vars(term).values()))
    assert len({f, Symmetric(x, y, Box('f', x @ p, y), p), g}) == 2


p1, p2, q1, q2 = map(Ty, ("p1", "p2", "q1", "q2"))
r1, r2, s1, s2 = (
    Box("r1", p1, p), Box("r2", p2, p1), Box("s1", q1, q), Box("s2", q2, q1))


def test_cell_errors():
    with raises(AxiomError):
        Cell(f, f.reparam(r1), Box("r1'", p1, p))
    with raises(AxiomError):
        Cell(f, Symmetric(x, z, Box('f', x @ p, z), p), Diagram.id(p))
    with raises(AxiomError):
        Cell.reparam(f, Box('r', q, q))
    with raises(AxiomError):
        Cell.reparam(f, r1).then(Cell.id(f))


def test_cell_wrong_order():
    assert (f >> g).param == f.param @ g.param
    cod = f.reparam(r1) >> g.reparam(s1)
    assert Cell(f >> g, cod, r1 @ s1).cod == cod
    with raises(AxiomError):
        Cell(f >> g, cod, s1 @ r1)
    with raises(AxiomError):
        Cell(f @ g, f.reparam(r1) @ g.reparam(s1), s1 @ r1)
    with raises(AxiomError, match="does not compose"):
        Cell.id(g) >> Cell.id(f)


def test_cell_then_up_to_the_quotient():
    def looped(para):
        dom, param = para.dom, para.param
        inside = Diagram.swap(dom, param) >> Diagram.swap(param, dom)
        return Symmetric(dom, para.cod, inside >> para.inside, param)

    assert looped(f) != f and same(looped(f), f)
    alpha = Cell.reparam(f, r1)
    other = looped(alpha.cod)
    assert other != alpha.cod and same(other, alpha.cod)
    beta = Cell.reparam(other, r2)
    assert alpha.then(beta).cod == other.reparam(r2)
    assert alpha.then(beta).inside == r2 >> r1


def test_cell_round_trip():
    import discopy
    from discopy import para
    cell = Cell.reparam(f, r1)
    assert eval(repr(cell), {**vars(discopy), **vars(para)}) == cell


def test_cell_vertical():
    alpha = Cell.reparam(f, r1)
    beta = Cell.reparam(alpha.cod, r2)
    gamma = Cell.reparam(beta.cod, Box("r3", Ty("p3"), p2))
    assert alpha.then(beta).inside == r2 >> r1
    assert alpha.then(beta).cod == f.reparam(r1).reparam(r2)
    assert alpha.then(beta).then(gamma) == alpha.then(beta.then(gamma))
    assert Cell.id(f).then(alpha) == alpha == alpha.then(Cell.id(alpha.cod))


def test_cell_horizontal_and_interchange():
    a1, a2 = Cell.reparam(f, r1), Cell.reparam(f.reparam(r1), r2)
    b1, b2 = Cell.reparam(g, s1), Cell.reparam(g.reparam(s1), s2)
    assert (a1 >> b1).inside == r1 @ s1
    assert (a1 >> b1).cod == f.reparam(r1) >> g.reparam(s1)
    assert (a1 @ b1).cod == f.reparam(r1) @ g.reparam(s1)
    left, right = (a1 >> b1).then(a2 >> b2), a1.then(a2) >> b1.then(b2)
    assert (left.dom, left.cod) == (right.dom, right.cod)
    assert left.inside.to_hypergraph() == right.inside.to_hypergraph()
    left, right = (a1 @ b1).then(a2 @ b2), a1.then(a2) @ b1.then(b2)
    assert left.inside.to_hypergraph() == right.inside.to_hypergraph()


def test_cell_invertible():
    two = Symmetric(x, y, Box('t', x @ p @ q, y), p @ q)
    swap = Cell.reparam(two, Diagram.swap(q, p))
    assert swap.is_invertible and Cell.id(two).is_invertible
    assert not Cell.reparam(two, Box('r', p @ q, p @ q)).is_invertible
    assert not Cell.reparam(f, r1).is_invertible
    h = Symmetric(x, y, Box('h', x, y), Ty())
    assert not Cell.reparam(h, Box('discard', p, Ty())).is_invertible
    assert Cell.id(h).is_invertible
    assert list(permutations(Diagram, p @ q, q @ p))
    assert not list(permutations(Diagram, p @ q, p @ p))
    assert not list(permutations(Diagram, p, Ty()))
    assert next(permutations(Diagram, p @ p, p @ p)) == Diagram.id(p @ p)
    assert len(list(permutations(Diagram, p @ p, p @ p))) == 2


def shifted(n):
    """ A cell reparametrising by the cyclic shift of n wires. """
    wires = Ty(*"abcdefghij"[:n])
    shift = Diagram.permutation([*range(1, n), 0], wires)
    box = Symmetric(x, y, Box('t', x @ shift.cod, y), shift.cod)
    return Cell.reparam(box, shift)


def test_bound():
    assert BOUND == 6
    assert shifted(BOUND).is_invertible
    assert not shifted(BOUND + 1).is_invertible
    wide = Ty(*"abcdefg")
    assert len(list(permutations(Diagram, wide, wide))) == 1
    assert not list(permutations(Diagram, wide, wide[1:] @ wide[:1]))
    assert Cell.id(shifted(BOUND + 1).dom).is_invertible


def test_permutations_are_lazy():
    perms = permutations(Diagram, p @ q @ p, p @ q @ p)
    assert iter(perms) is perms
    assert next(perms) == Diagram.id(p @ q @ p)


def test_laws_up_to_permutation():
    h = Symmetric(z, x, Box('h', z @ p1, x), p1)
    k = Symmetric(x, y, Box('k', x @ q1, y), q1)
    left, right = (f >> g) @ (h >> k), (f @ h) >> (g @ k)
    assert left.param != right.param
    assert UpToPermutation(left) == UpToPermutation(right)
    left = (f @ g) >> Symmetric.swap(y, z)
    right = Symmetric.swap(x, y) >> (g @ f)
    assert left.param == p @ q and right.param == q @ p
    assert UpToPermutation(left) == UpToPermutation(right)
    assert UpToPermutation(left) != UpToPermutation(f @ g)


def test_the_seven_laws_hold():
    h = Symmetric(z, x, Box('h', z @ p1, x), p1)
    k = Symmetric(x, y, Box('k', x @ q1, y), q1)
    assert Symmetric.bifunctoriality(f, h, g, k)
    assert Symmetric.braid_naturality(f, g)
    scalar = Symmetric(Ty(), Ty(), Box('s', q1, Ty()), q1)
    assert Symmetric.scalar_sliding(scalar, f)
    t = Traced(x @ y, x @ z, Box('t', x @ y @ p, x @ z), p)
    u = Traced(x, x, Box('u', x @ q, x), q)
    assert Traced.trace_dinaturality_left(t, u)
    assert Traced.trace_dinaturality_right(
        Traced(y @ x, z @ x, Box('t', y @ x @ p, z @ x), p), u)
    a, b, c = map(compact.Ty, "abc")
    cf = Compact(a, b, compact.Box('f', a @ compact.Ty('p'), b),
                 compact.Ty('p'))
    cg = Compact(b, c, compact.Box('g', b @ compact.Ty('q'), c),
                 compact.Ty('q'))
    assert Compact.transpose_contravariance(cf, cg)


def test_laws_are_not_trivial():
    extra = Symmetric(y, y, Box('e', y, y))
    law = Symmetric.equation_factory(f, f >> extra).modulo(UpToPermutation)
    assert not law
    swap = Symmetric.equation_factory(f @ g, g @ f)
    assert not swap.modulo(UpToPermutation)


def test_up_to_permutation_is_an_equivalence():
    two = Symmetric(x, y, Box('t', x @ p @ q, y), p @ q)
    flip = two.reparam(Diagram.swap(q, p))
    left, right = UpToPermutation(two), UpToPermutation(flip)
    assert left == right and right == left and left == left
    assert hash(left) == hash(right)
    assert left != UpToPermutation(Symmetric(
        z, y, Box('t', z @ p @ q, y), p @ q))
    assert left != UpToPermutation(Symmetric(
        x, z, Box('t', x @ p @ q, z), p @ q))
    assert left != UpToPermutation(Symmetric(
        x, y, Box('t', x @ p @ q, y @ p), p @ q, p))
    assert left != UpToPermutation(Symmetric(
        x, y, Box('t2', x @ p @ q, y), p @ q))
    assert left != two


def test_up_to_permutation_needs_the_same_wires():
    left = Symmetric(x, y, Box('t', x @ p @ q, y), p @ q)
    right = Symmetric(x, y, Box('t', x @ p @ p, y), p @ p)
    assert UpToPermutation(left) != UpToPermutation(right)
    assert not Symmetric.equation_factory(left, right).modulo(UpToPermutation)


def test_up_to_permutation_over_functions():
    ff = python.Ty(float, float)
    inside = Function(lambda a, w, b: w * a + b, python.Ty(float) ** 3,
                      python.Ty(float))
    layer = Markov[Function](python.Ty(float), python.Ty(float), inside, ff)
    assert UpToPermutation(layer) == UpToPermutation(layer)
    assert same(layer, layer)


def test_copy_is_not_a_permutation():
    X, P = markov.Ty('X'), markov.Ty('P')
    box = Markov(X, X, markov.Box('c', X @ P @ P, X), P @ P)
    forget = markov.Diagram.discard(P) @ P >> markov.Diagram.copy(P)
    cell = Cell.reparam(box, forget)
    assert cell.cod.param == box.param and not cell.is_invertible
    assert UpToPermutation(cell.cod) != UpToPermutation(box)
    assert UpToPermutation(box) == UpToPermutation(box)


def test_weight_tying_is_a_cell():
    X, P = markov.Ty('X'), markov.Ty('P')
    layer = Markov(X, X, markov.Box('c', X @ P, X), P)
    network = layer >> layer
    tying = Cell.reparam(network, markov.Diagram.copy(P))
    assert tying.cod.param == P and network.param == P @ P
    assert not tying.is_invertible


def test_laws_with_coparameters():
    a = Ty('a')
    f, g, h, k = (
        Symmetric(dom, cod, Box(name, dom @ a, cod @ a), a, a)
        for name, dom, cod in (
            ('f', x, y), ('g', y, z), ('h', z, x), ('k', x, y)))
    left, right = (f >> g) @ (h >> k), (f @ h) >> (g @ k)
    assert left != right and left.param == right.param == a @ a @ a @ a
    assert UpToPermutation(left) == UpToPermutation(right)
    assert Symmetric.bifunctoriality(f, h, g, k)
    assert Symmetric.braid_naturality(f, g)


def test_coparameters_are_permuted_not_changed():
    m, n = Ty('m'), Ty('n')
    t = Symmetric(x, y, Box('t', x @ p, y @ m @ n), p, m @ n)
    flip = t.recopar(Diagram.swap(m, n))
    assert UpToPermutation(t) == UpToPermutation(flip) and t != flip
    other = t.recopar(Box('c', m @ n, m @ n))
    assert UpToPermutation(t) != UpToPermutation(other)
    twice = t.reparam(Box('r', q, p))
    assert UpToPermutation(t) != UpToPermutation(twice)
    assert not Symmetric.equation_factory(t, other).modulo(UpToPermutation)


def test_many_wires_are_decided_without_search():
    a = Ty('a')
    for n in (3, 7, 12):
        wires = a ** n
        box = Symmetric(x, y, Box('t', x @ wires, y @ wires), wires, wires)
        moved = box.reparam(Diagram.swap(a, a) @ a ** (n - 2))
        assert moved != box
        assert UpToPermutation(box) == UpToPermutation(moved)
        assert UpToPermutation(box) == UpToPermutation(
            box.reparam(Diagram.id(wires)))
        other = Symmetric(x, y, Box('s', x @ wires, y @ wires), wires, wires)
        assert UpToPermutation(box) != UpToPermutation(other)


def test_hypergraph_laws_with_equal_param_and_copar():
    a = frobenius.Ty('a')
    boxes = [frobenius.Box(name, a @ a, a @ a @ a if name == 'f' else a @ a)
             for name in "fghk"]
    f, g, h, k = (Hypergraph(a, a, box, a, box.cod[1:])
                  for box in boxes)
    assert (f >> g) @ (h >> k) != (f @ h) >> (g @ k)
    assert Hypergraph.bifunctoriality(f, g, h, k)
    assert Hypergraph.bifunctoriality(f, h, g, k)
    assert Hypergraph.braid_naturality(f, g)
    assert not issubclass(Hypergraph, Feedback)


def test_compact_laws_with_adjoint_parameters():
    a = compact.Ty('a')
    for params in (a @ a.r @ a, a @ a.r @ a @ a.r @ a @ a @ a):
        f, g, h, k = (
            Compact(a, a, compact.Box(name, a @ params, a @ a), params, a)
            for name in "fghk")
        assert Compact.bifunctoriality(f, g, h, k)
        assert Compact.bifunctoriality(f, h, g, k)
    wrong = Compact(a, a, compact.Box('k', a @ a.r, a @ a), a.r, a)
    left, right = (f >> g) @ (h >> k), (f @ h) >> (g @ wrong)
    assert UpToPermutation(left) != UpToPermutation(right)


def test_feedback_never_pairs_parameters_with_coparameters():
    X, A = map(markov.Ty, "XA")
    f = Feedback(X, X, markov.Box('f', X, X))
    g = Feedback(X, X, markov.Box('g', X @ A, X @ A), A, A)
    h = Feedback(X, X, markov.Box('h', X @ A, X), A)
    k = Feedback(X, X, markov.Box('k', X, X @ A), copar=A)
    left, right = (f @ g) >> (h @ k), (f >> h) @ (g >> k)
    assert left != right
    assert (left.param, left.copar) == (A @ A, A @ A)
    assert UpToPermutation(left) == UpToPermutation(right)
    assert Feedback.bifunctoriality(f, g, h, k)


def test_feedback_laws_with_a_memory():
    X, Y, Z, M = map(markov.Ty, "XYZM")
    f, g, h, k = (
        Feedback(dom, cod, markov.Box(name, dom @ M, cod @ M), M, M)
        for name, dom, cod in (
            ('f', X, Y), ('g', Y, Z), ('h', Z, X), ('k', X, Y)))
    assert Feedback.bifunctoriality(f, h, g, k)
    assert Feedback.braid_naturality(f, g)


def test_the_boundary_stays_in_place():
    a, b = (Symmetric(x, y, Box(name, x, y)) for name in "ab")
    both = a @ b
    for moved in (both >> Symmetric.swap(y, y),
                  Symmetric.swap(x, x) >> both):
        assert UpToPermutation(both) != UpToPermutation(moved)
        assert UpToPermutation(moved) != UpToPermutation(both)


def test_data_tells_boxes_apart():
    assert Box('f', x, y) != Box('f', x, y, data=1)
    one, two = (Symmetric(x, y, Box('t', x @ p, y, data=n), p)
                for n in (1, 2))
    assert UpToPermutation(one) != UpToPermutation(two)
    assert UpToPermutation(one) == UpToPermutation(one.reparam(Diagram.id(p)))


def test_ports_of_a_box_are_ordered():
    left = Symmetric(x @ y, z, Box('t', x @ y, z))
    right = Symmetric.swap(x, y) >> Symmetric(y @ x, z, Box('t', y @ x, z))
    assert UpToPermutation(left) != UpToPermutation(right)
    u, v = Ty('u'), Ty('v')
    a, b = Symmetric(u, x, Box('a', u, x)), Symmetric(v, x, Box('b', v, x))
    t = Symmetric(x @ x, y, Box('t', x @ x, y))
    straight, crossed = (a @ b) >> t, (a @ b) >> Symmetric.swap(x, x) >> t
    assert UpToPermutation(straight) != UpToPermutation(crossed)


def test_a_boundary_port_is_not_a_parameter_port():
    box = Box('t', x @ x, y)
    left = Symmetric(x, y, box, x)
    right = Symmetric(x, y, Diagram.swap(x, x) >> box, x)
    assert UpToPermutation(left) != UpToPermutation(right)
    box = Box('t', x, y @ y)
    left = Symmetric(x, y, box, Ty(), y)
    right = Symmetric(x, y, box >> Diagram.swap(y, y), Ty(), y)
    assert UpToPermutation(left) != UpToPermutation(right)


def test_adjoint_parameters_are_not_plain_ones():
    a = compact.Ty('a')
    plain = Compact(a, a, compact.Box('t', a @ a, a), a)
    adjoint = Compact(a, a, compact.Box('t', a @ a.r, a), a.r)
    assert UpToPermutation(plain) != UpToPermutation(adjoint)
    left = Compact(a, a, compact.Box('t', a @ a @ a.r, a), a @ a.r)
    right = Compact(a, a, compact.Box('t', a @ a.r @ a, a), a.r @ a)
    assert UpToPermutation(left).boundary == UpToPermutation(right).boundary
    assert UpToPermutation(left) != UpToPermutation(right)


def test_unused_parameters_are_decided_fast():
    a = Ty('a')
    wires = a ** 12

    def build(box):
        return Symmetric(x, y, box @ Diagram.id(wires), wires, wires)

    base = build(Box('t', x, y))
    others = (build(Box('s', x, y)), build(Box('t', x, y, data=1)))
    start = time.perf_counter()
    for other in others:
        assert UpToPermutation(base) != UpToPermutation(other)
    assert UpToPermutation(base) == UpToPermutation(
        base.reparam(Diagram.swap(a, a) @ a ** 10))
    assert time.perf_counter() - start < 2
