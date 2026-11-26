from itertools import product
from random import Random

import pytest

import discopy
from discopy import markov, para, python
from discopy.markov import Box, Diagram, Ty
from discopy.neural.datatype import (
    Algebra, Datatype, LaxHomomorphism, chain)
from discopy.utils import AxiomError
from fixtures.datatype import Int, Net, P, Para, S, check_grads

A = Ty("A")
LISTS, TREES = Datatype.list(A, S), Datatype.tree(A, S)
nil, cons = LISTS.constructors
leaf, node = TREES.constructors


def as_list(n):
    """ The notebook's list diagram, written independently. """
    diagram = A ** n @ nil
    for i in range(n, 0, -1):
        diagram = diagram >> A ** (i - 1) @ cons
    return diagram


def random_list(rng):
    return [rng.randint(-9, 9) for _ in range(rng.randint(0, 7))]


def random_tree(rng, depth=4):
    if depth == 0 or rng.random() < 0.3:
        return ("leaf", rng.randint(-9, 9))
    return ("node", random_tree(rng, depth - 1), random_tree(rng, depth - 1))


def tree_value(tree):
    """ The term and the data in pre-order, i.e. the leaves from the left. """
    if tree[0] == "leaf":
        return ("Leaf", ), [tree[1]]
    (left, xs), (right, ys) = tree_value(tree[1]), tree_value(tree[2])
    return ("Node", left, right), xs + ys


def leaves(tree):
    return tree_value(tree)[1]


# 1. terms


@pytest.mark.parametrize("n", range(6))
def test_list_terms(n):
    term = LISTS.term(chain(n))
    assert term == as_list(n)
    assert (term.dom, term.cod) == (A ** n, S)
    assert len(term.boxes) == n + 1


def test_tree_terms():
    value = ("Node", ("Node", ("Leaf", ), ("Leaf", )), ("Leaf", ))
    term = TREES.term(value)
    assert term == (leaf @ leaf >> node) @ leaf >> node
    assert (term.dom, term.cod) == (A ** 3, S)
    assert TREES.term(("Leaf", )) == leaf
    small, big = ("Leaf", ), ("Node", ("Leaf", ), ("Leaf", ))
    assert TREES.substitute(node, TREES.term(small), TREES.term(big))\
        == TREES.term(("Node", small, big))


def test_data_and_arity():
    assert [(LISTS.data(b), LISTS.arity(b)) for b in (nil, cons)]\
        == [(Ty(), 0), (A, 1)]
    assert [(TREES.data(b), TREES.arity(b)) for b in (leaf, node)]\
        == [(A, 0), (Ty(), 2)]
    assert LISTS.carrier == S and LISTS.constructor("Cons") == cons


def test_data_before_children():
    B, C = Ty("B"), Ty("C")
    lists = Datatype(Box("Nil", B, S), Box("Cons", A @ S, S))
    assert lists.term(("Cons", ("Nil", ))).dom == A @ B
    assert lists.term(("Cons", ("Cons", ("Nil", )))).dom == A @ A @ B
    trees = Datatype(Box("Leaf", C, S), Box("Node", A @ S @ S, S))
    value = ("Node", ("Node", ("Leaf", ), ("Leaf", )), ("Leaf", ))
    assert trees.term(value).dom == A @ A @ C @ C @ C


def test_long_list_term():
    n, rng = 200, Random(6)
    term, xs = LISTS.term(chain(n)), [rng.randint(-9, 9) for _ in range(n)]
    assert term.dom == A ** n and len(term.boxes) == n + 1
    expected = 0
    for x in reversed(xs):
        expected = x + 2 * expected
    assert list_algebra(param=False)(term)(*xs) == expected


def test_repr():
    scope = vars(discopy) | {"Datatype": Datatype}
    for datatype in (LISTS, TREES):
        assert eval(repr(datatype), scope) == datatype


@pytest.mark.parametrize("call, args, error", [
    *[(LISTS.term, [value], ValueError) for value in (
        ("Snoc", ), ("Cons", ), ("Nil", ("Nil", )),
        ("Cons", ("Nil", ), ("Nil", )), "Nil", (), ["Nil"])],
    (Datatype, [Box("c", S @ A, S)], ValueError),  # carrier before data
    (Datatype, [Box("c", S @ A @ S, S)], ValueError),  # carrier in between
    (Datatype, [Box("a", Ty(), S), Box("b", S, A)], ValueError),
    (Datatype, [Box("a", Ty(), S), Box("b", Ty("T"), Ty("T"))], ValueError),
    (Datatype, [Box("a", Ty(), S @ S)], ValueError),  # codomain not atomic
    (Datatype, [], ValueError),  # no constructors
    (Datatype, [Box("a", Ty(), S), Box("a", S, S)], ValueError),  # same name
    (Datatype, ["Nil"], TypeError),
    (chain, [-1], ValueError)])
def test_term_errors(call, args, error):
    with pytest.raises(error):
        call(*args)


# 2. folds into python functions


def list_algebra(param=True):
    if not param:
        return Algebra(
            LISTS, {nil: python.Function(lambda: 0, python.Ty(), Int),
                    cons: python.Function(
                        lambda a, s: a + 2 * s, Int @ Int, Int)},
            {A: Int, S: Int}, python.Function)
    return Algebra(
        LISTS, {nil: Net(python.Ty(), Int, python.Function(
                    lambda p: p, Int, Int), param=Int),
                cons: Net(Int @ Int, Int, python.Function(
                    lambda a, s, p: a + p * s, Int ** 3, Int), param=Int)},
        {A: Int, S: Int}, Net)


def tree_algebra():
    return Algebra(
        TREES, {leaf: Net(Int, Int, python.Function(
                    lambda a, p: a + p, Int ** 2, Int), param=Int),
                node: Net(Int @ Int, Int, python.Function(
                    lambda left, right, p: 2 * left - right + p, Int ** 3,
                    Int), param=Int)},
        {A: Int, S: Int}, Net)


def list_recursion(xs, p):
    return p if not xs else xs[0] + p * list_recursion(xs[1:], p)


def tree_recursion(tree, p):
    if tree[0] == "leaf":
        return tree[1] + p
    return 2 * tree_recursion(tree[1], p) - tree_recursion(tree[2], p) + p


def test_parametric_list_fold():
    rng, algebra = Random(0), list_algebra()
    for xs in [[]] + [random_list(rng) for _ in range(40)]:
        p = rng.randint(-5, 5)
        fold = algebra(LISTS.term(chain(len(xs)))).inside
        assert fold(*xs, p) == list_recursion(xs, p)


def test_plain_list_fold():
    rng, algebra = Random(1), list_algebra(param=False)

    def recursion(xs):
        return 0 if not xs else xs[0] + 2 * recursion(xs[1:])

    for xs in [[]] + [random_list(rng) for _ in range(40)]:
        assert algebra(LISTS.term(chain(len(xs))))(*xs) == recursion(xs)


def test_tree_fold():
    rng, algebra = Random(2), tree_algebra()
    for _ in range(40):
        tree, p = random_tree(rng), rng.randint(-5, 5)
        value, xs = tree_value(tree)
        fold = algebra(TREES.term(value)).inside
        assert fold(*xs, p) == tree_recursion(tree, p)


def test_algebra_errors():
    cell = Net(Int @ Int, Int, python.Function(
        lambda a, s, p, q: a, Int ** 4, Int), param=Int @ Int)
    algebra = list_algebra()
    with pytest.raises(ValueError):  # missing image
        Algebra(LISTS, {nil: algebra.ar_map[nil]}, algebra.ob_map, Net)
    with pytest.raises(ValueError, match=r"Nil.*Cons"):  # names each one
        Algebra(LISTS, {nil: algebra.ar_map[nil], cons: cell},
                algebra.ob_map, Net)
    plain = list_algebra(param=False)
    with pytest.raises(ValueError, match="not an arrow"):  # wrong category
        Algebra(LISTS, plain.ar_map, plain.ob_map, Net)
    with pytest.raises(ValueError, match="atom"):  # missing atom of the data
        Algebra(LISTS, algebra.ar_map, {S: Int}, Net)
    with pytest.raises(ValueError, match="atom"):  # missing carrier
        Algebra(LISTS, algebra.ar_map, {A: Int}, Net)


def test_algebra_param():
    assert list_algebra(param=False).param is None
    assert list_algebra().param == Int


# 3. folds into parametric diagrams


def diagram_algebra():
    return Algebra(
        LISTS, {nil: Para(Ty(), S, Box("initial", P, S), param=P),
                cons: Para(A @ S, S, Box("cell", A @ S @ P, S), param=P)},
        {A: A, S: S}, Para)


def tree_diagram_algebra():
    return Algebra(
        TREES, {leaf: Para(A, S, Box("leaf", A @ P, S), param=P),
                node: Para(S @ S, S, Box("node", S @ S @ P, S), param=P)},
        {A: A, S: S}, Para)


@pytest.mark.parametrize("n", range(6))
def test_diagram_fold(n):
    algebra, term = diagram_algebra(), LISTS.term(chain(n))
    untied = algebra.functor(term)
    assert untied.param == P ** (n + 1)
    tied = algebra(term)
    assert tied.param == P and (tied.dom, tied.cod) == (A ** n, S)
    assert tied == untied.reparam(Diagram.copy(P, n + 1))
    assert tied == algebra.functor(as_list(n)).reparam(Diagram.copy(P, n + 1))
    assert tied.inside == A ** n @ Diagram.copy(P, n + 1) >> untied.inside


def test_diagram_tree_fold():
    algebra = tree_diagram_algebra()
    term = TREES.term(("Node", ("Node", ("Leaf", ), ("Leaf", )), ("Leaf", )))
    k = len(term.boxes)
    assert k == 5
    untied, tied = algebra.functor(term), algebra(term)
    assert untied.param == P ** k and tied.param == P
    assert tied == untied.reparam(Diagram.copy(P, k))


# 4. folds into networks, run by torch


def test_network_list_fold():
    torch = pytest.importorskip("torch")
    from discopy.neural.network import Box as Layer, Dim, Dims, Network
    from discopy.neural.torch import Module
    torch.manual_seed(0)
    x, h, H0, W = Dims(3), Dims(5), Dims(5), Dims(Dim(8, 5))

    def rnn(a, s, w):
        return torch.tanh(torch.cat([a, s], -1) @ w)

    algebra = diagram_algebra()
    implement = markov.Functor(
        ob_map={A: x, S: h, P: H0 @ W},
        ar_map={
            Box("initial", P, S): Network.id(H0) @ Network.discard(W),
            Box("cell", A @ S @ P, S):
                x @ h @ Network.discard(H0) @ W
                >> Layer("cell", x @ h @ W, h, module=rnn)},
        cod=Network)

    def recursion(xs, h0, w):
        return h0 if not xs else rnn(xs[0], recursion(xs[1:], h0, w), w)

    for n in (0, 1, 3, 6):
        tied = algebra(LISTS.term(chain(n)))
        module = Module(
            para.Markov[Network](x ** n, h, param=H0 @ W,
                                 inside=implement(tied.inside)),
            torch.randn(5), torch.randn(8, 5) / 3)
        assert [tuple(p.shape) for p in module.parameters()] == [(5, ), (8, 5)]
        xs = [torch.randn(3) for _ in range(n)]
        out = module(*xs)
        h0, w = [p.detach().clone().requires_grad_() for p in module.weights]
        ref = recursion(xs, h0, w)
        assert torch.allclose(out, ref, atol=1e-6)
        out.sum().backward()
        ref.sum().backward()
        check_grads(module, (h0, w))


def test_network_tree_fold():
    torch = pytest.importorskip("torch")
    from discopy.neural.network import Box as Layer, Dim, Dims, Network
    from discopy.neural.torch import Module
    torch.manual_seed(1)
    x, W = Dims(4), Dims(Dim(8, 4))

    def leaf_cell(a, w):
        return torch.tanh(torch.cat([a, a], -1) @ w)

    def node_cell(left, right, w):
        return torch.tanh(torch.cat([left, right], -1) @ w)

    algebra = tree_diagram_algebra()
    implement = markov.Functor(
        ob_map={A: x, S: x, P: W},
        ar_map={
            Box("leaf", A @ P, S):
                Layer("leaf", x @ W, x, module=leaf_cell),
            Box("node", S @ S @ P, S):
                Layer("node", x @ x @ W, x, module=node_cell)},
        cod=Network)

    def recursion(tree, xs, w):
        if tree[0] == "leaf":
            return leaf_cell(xs.pop(0), w)
        left = recursion(tree[1], xs, w)
        return node_cell(left, recursion(tree[2], xs, w), w)

    one = ("leaf", 0)
    shapes = [one,
              ("node", ("node", one, one), ("node", one, one)),
              ("node", one, ("node", one, ("node", one, one))),
              ("node", ("node", ("node", one, one), one), one)]
    for tree in shapes:
        value, data = tree_value(tree)
        k = len(data)
        tied = algebra(TREES.term(value))
        module = Module(
            para.Markov[Network](x ** k, x, param=W,
                                 inside=implement(tied.inside)),
            torch.randn(8, 4) / 3)
        assert [tuple(p.shape) for p in module.parameters()] == [(8, 4)]
        xs = [torch.randn(4) for _ in range(k)]
        out = module(*xs)
        (w, ) = [p.detach().clone().requires_grad_() for p in module.weights]
        ref = recursion(tree, list(xs), w)
        assert torch.allclose(out, ref, atol=1e-6)
        out.sum().backward()
        ref.sum().backward()
        check_grads(module, (w, ))


# 5. functoriality


def test_untied_image_is_composite():
    """ The image of ``A @ t >> Cons`` is ``A @ image(t) >> image(Cons)``. """
    algebra, rng = list_algebra(), Random(3)
    cons_image = algebra.ar_map[cons]
    for _ in range(30):
        xs = random_list(rng)
        m, a = len(xs), rng.randint(-9, 9)
        ps = [rng.randint(-5, 5) for _ in range(m + 2)]
        inner = algebra.functor(LISTS.term(chain(m)))
        outer = algebra.functor(LISTS.term(("Cons", chain(m))))
        assert outer.param == inner.param @ cons_image.param
        expected = cons_image.inside(
            a, inner.inside(*xs, *ps[:m + 1]), ps[m + 1])
        assert outer.inside(a, *xs, *ps) == expected


def test_untied_fold_numerics():
    """ Every constructor box reads its own parameter, from the left. """
    algebra, rng = list_algebra(), Random(4)
    for _ in range(30):
        xs = random_list(rng)
        ps = [rng.randint(-5, 5) for _ in range(len(xs) + 1)]
        image = algebra.functor(LISTS.term(chain(len(xs)))).inside
        expected = ps[0]
        for x, p in zip(reversed(xs), ps[1:]):
            expected = x + p * expected
        assert image(*xs, *ps) == expected


def test_tree_functoriality():
    algebra = tree_algebra()
    rng = Random(5)
    for _ in range(30):
        left, right = random_tree(rng, 3), random_tree(rng, 3)
        (lv, lx), (rv, rx) = tree_value(left), tree_value(right)
        both = algebra.functor(TREES.term(("Node", lv, rv)))
        lf = algebra.functor(TREES.term(lv))
        rf = algebra.functor(TREES.term(rv))
        assert both.param == lf.param @ rf.param @ algebra.ar_map[node].param
        ps = [rng.randint(-5, 5) for _ in range(len(both.param))]
        k = len(lf.param)
        l_out = lf.inside(*lx, *ps[:k])
        r_out = rf.inside(*rx, *ps[k:-1])
        assert both.inside(*lx, *rx, *ps) == 2 * l_out - r_out + ps[-1]


# 6. lax algebras and lax homomorphisms


def test_lift_compares_merged_copies():
    """ The lifts compare up to merging copies: copies of copies are one. """
    f = Para(Ty(), S, Box("f", P ** 3, S), param=P ** 3)
    left = f.reparam(Diagram.copy(P, 3))
    right = f.reparam(Diagram.copy(P) >> Diagram.copy(P) @ P)
    assert para.same(left, right)


LEAF = ("Leaf", )
LIST_VALUES = [chain(m) for m in range(5)]
TREE_VALUES = [LEAF, ("Node", LEAF, LEAF),
               ("Node", ("Node", LEAF, LEAF), LEAF)]
CASES = [(diagram_algebra, LISTS, value) for value in LIST_VALUES]\
    + [(tree_diagram_algebra, TREES, value) for value in TREE_VALUES]


def nodes(value):
    """ Every subterm of a term. """
    yield value
    for child in value[1:]:
        yield from nodes(child)


def depths(value, depth=0):
    """ The depth of each box of a term, in the order of its boxes. """
    return [d for child in value[1:] for d in depths(child, depth + 1)]\
        + [depth]


def spider(outs, n=2):
    """ The map P ** n -> P ** len(outs), output i reads input outs[i]. """
    return Diagram.from_function(list(outs), P ** n)


def shift(n):
    """ The map P ** n -> P ** n @ P that rotates the parameters. """
    return spider([*range(1, n), 0, 0], n)


def untied(algebra, datatype, value, n, read):
    """ The untied image of a term, box i reading the copy read(depth i). """
    reads = [read(depth) for depth in depths(value)]
    return algebra.functor(datatype.term(value)).reparam(
        spider(reads, n))


@pytest.mark.parametrize("make, datatype, value", CASES)
def test_tied_is_lax_homomorphism(make, datatype, value):
    algebra = make()
    fold = LaxHomomorphism(algebra, Diagram.copy(P))(value)
    assert para.same(fold, algebra(datatype.term(value)))
    for sub in nodes(value):
        square = algebra.associativity(
            datatype.constructor(sub[0]), *map(datatype.term, sub[1:]))
        assert markov.Equation(square.inside, Diagram.copy(P, 2))


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("make, datatype, value", CASES)
def test_lax_homomorphisms_are_not_unique(make, datatype, value, n):
    algebra = make()
    fold = LaxHomomorphism(algebra, shift(n))(value)
    assert fold.param == P ** n
    assert para.same(
        fold, untied(algebra, datatype, value, n, lambda d: d % n))
    tied = algebra(datatype.term(value))
    assert para.same(tied, fold.reparam(Diagram.copy(P, n)))
    if n == 1:
        assert para.same(fold, tied)


@pytest.mark.parametrize("m", range(5))
def test_same_boundary_different_maps(m):
    algebra = diagram_algebra()
    alt, first = (LaxHomomorphism(algebra, spider(outs))(chain(m))
                  for outs in ([1, 0, 0], [0, 1, 0]))
    assert (alt.dom, alt.cod) == (first.dom, first.cod)
    assert alt.param == first.param
    assert (para.UpToPermutation(alt) != para.UpToPermutation(first))\
        == (m >= 1)
    tied = algebra(LISTS.term(chain(m)))
    assert para.same(first, tied.reparam(Diagram.from_function([0], P @ P)))


SHAPES = [
    (cons, [LISTS.term(chain(1))], diagram_algebra, LISTS),
    (A @ cons >> cons, [LISTS.term(chain(0))], diagram_algebra, LISTS),
    (Diagram.id(S), [LISTS.term(chain(1))], diagram_algebra, LISTS),
    (node, [TREES.term(LEAF)] * 2, tree_diagram_algebra, TREES),
    (leaf @ S >> node, [TREES.term(LEAF)], tree_diagram_algebra, TREES)]


@pytest.mark.parametrize("shape, children, make, datatype", SHAPES)
def test_lax_algebra_units(shape, children, make, datatype):
    algebra, holes = make(), shape.dom.count(S)
    square = algebra.associativity(shape, *children)
    assert para.same(
        square.cod, algebra(datatype.substitute(shape, *children)))
    right = algebra.lift_reparam(shape, *[algebra.unitality()] * holes)\
        >> para.Reparam.id(algebra(shape))
    right = right.then(algebra.associativity(shape, *[Diagram.id(S)] * holes))
    left = para.Reparam.id(algebra(shape)) >> algebra.unitality()
    left = left.then(algebra.associativity(Diagram.id(S), shape))
    for alpha in (left, right):
        assert para.same(alpha.cod, algebra(shape))
        assert markov.Equation(alpha.inside, Diagram.id(P))


def pastings(algebra, datatype, shape, children, grandchildren):
    """ The two pastings of the associativity of the lax algebra. """
    holes = [algebra.lift(child, *map(algebra, terms), param=algebra.param)
             for child, terms in zip(children, grandchildren)]
    squares = [algebra.associativity(child, *terms)
               for child, terms in zip(children, grandchildren)]
    terms = [term for group in grandchildren for term in group]
    substituted = [datatype.substitute(child, *group)
                   for child, group in zip(children, grandchildren)]
    left = (para.Reparam.id(algebra.lift(shape, *holes))
            >> algebra.associativity(shape, *children)).then(
        algebra.associativity(datatype.substitute(shape, *children), *terms))
    right = (algebra.lift_reparam(shape, *squares)
             >> para.Reparam.id(algebra(shape))
             ).then(algebra.associativity(shape, *substituted))
    return left, right


@pytest.mark.parametrize("make, datatype, shape, children, grandchildren", [
    (diagram_algebra, LISTS, cons, [cons], [[LISTS.term(chain(0))]]),
    (tree_diagram_algebra, TREES, node, [leaf, node],
     [[], [TREES.term(LEAF)] * 2])])
def test_lax_algebra_associativity(
        make, datatype, shape, children, grandchildren):
    left, right = pastings(
        make(), datatype, shape, children, grandchildren)
    assert para.same(left.dom, right.dom) and para.same(left.cod, right.cod)
    assert markov.Equation(left.inside, right.inside)
    assert markov.Equation(left.inside, Diagram.copy(P, 3))


def test_only_the_copy_is_forced():
    two = Ty("p") @ Ty("p")
    algebra = Algebra(
        LISTS, {nil: Para(Ty(), S, Box("initial", two, S), param=two),
                cons: Para(A @ S, S, Box("cell", A @ S @ two, S), param=two)},
        {A: A, S: S}, Para)
    square = algebra.associativity(cons, A @ cons >> cons)
    built = []
    for outs in product(range(2), repeat=4):
        try:
            para.Reparam(square.dom, Diagram.from_function(outs, two),
                         square.cod)
        except AxiomError:
            continue
        built.append(outs)
    assert built == [(0, 1, 0, 1)]


KAPPAS = list(product(range(2), repeat=3))
SQUARES = {(0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)}
UNIT = {(0, 1, 0), (0, 1, 1)}
CLOSED = [(A @ nil >> cons, [], ("Cons", ("Nil", ))),
          (A @ cons >> cons, [("Nil", )], ("Cons", ("Cons", ("Nil", ))))]


def holds(algebra, kappa):
    """ Whether the squares at two closed shapes hold for kappa. """
    hom = LaxHomomorphism(algebra, kappa)
    for shape, children, value in CLOSED:
        folds = [hom(child) for child in children]
        dom = algebra.lift(shape, *folds, param=hom.param)\
            >> algebra(shape)
        try:
            para.Reparam(dom, kappa, hom(value))
        except AxiomError:
            return False
    return True


@pytest.mark.parametrize("outs", KAPPAS)
def test_the_eight_spiders(outs):
    algebra, kappa = diagram_algebra(), spider(outs)
    hom = LaxHomomorphism(algebra, kappa)
    assert holds(algebra, kappa) == (outs in SQUARES)
    assert bool(hom.unitality()) == (outs in UNIT)
    assert bool(hom.multiplication()) == (outs in SQUARES)
    tied = algebra(LISTS.term(chain(2)))
    projections = [
        para.same(hom(chain(2)),
                  tied.reparam(Diagram.from_function([i], P @ P)))
        for i in range(2)]
    survivors = {(0, 0, 0): [True, False], (0, 1, 0): [True, False],
                 (0, 1, 1): [False, True], (1, 1, 1): [False, True]}
    assert projections == survivors.get(outs, [False, False])


def test_copy_and_rotations():
    algebra = diagram_algebra()
    assert holds(algebra, Diagram.copy(P))
    assert not holds(algebra, shift(2))
    for n in (2, 3):
        assert not LaxHomomorphism(algebra, shift(n)).unitality()
    assert LaxHomomorphism(algebra, Diagram.copy(P)).unitality()
    assert LaxHomomorphism(algebra, Diagram.copy(P)).multiplication()


def test_well_typed_negative_and_box_witness():
    algebra = diagram_algebra()
    square = algebra.associativity(cons, LISTS.term(chain(0)))
    with pytest.raises(AxiomError):
        para.Reparam(square.dom, Box("k", P, P @ P), square.cod)
    hom = LaxHomomorphism(algebra, spider([0, 1, 0]))
    top, fold = hom.top(chain(1)), hom(chain(1))
    assert hom.preservation(chain(1)).cod == fold
    para.Reparam(top, hom.kappa, fold)
    with pytest.raises(AxiomError):
        para.Reparam(top, shift(2), fold)


def test_oplax_is_rejected_at_constructors_only():
    algebra, trees = diagram_algebra(), tree_diagram_algebra()
    discard = Diagram.discard(P)
    projections = [Diagram.id(P) @ discard, discard @ Diagram.id(P)]
    nil_square = algebra.associativity(nil).dom
    para.Reparam(algebra(LISTS.term(chain(0))), discard @ P, nil_square)
    cons_square = algebra.associativity(cons, LISTS.term(chain(0))).dom
    node_square = trees.associativity(
        node, TREES.term(LEAF), TREES.term(LEAF)).dom
    for projection in projections:
        with pytest.raises(AxiomError):
            para.Reparam(
                algebra(LISTS.term(chain(1))), projection, cons_square)
        with pytest.raises(AxiomError):
            para.Reparam(trees(TREES.term(("Node", LEAF, LEAF))),
                         projection, node_square)


def fits(top, fold):
    """ The spiders kappa such that fold is the square of top. """
    found = set()
    for outs in KAPPAS:
        try:
            para.Reparam(top, spider(outs), fold)
        except AxiomError:
            continue
        found.add(outs)
    return found


def test_one_kappa_for_all_constructors():
    algebra, q = diagram_algebra(), P @ P
    nil_fold = algebra.ar_map[nil].reparam(Diagram.from_function([0], q))
    nil_top = algebra.lift(nil, param=q) >> algebra.ar_map[nil]
    cons_top = algebra.lift(cons, nil_fold) >> algebra.ar_map[cons]
    cons_fold = cons_top.reparam(Diagram.from_function([0, 1, 1], q))
    fits_nil, fits_cons = fits(nil_top, nil_fold), fits(cons_top, cons_fold)
    assert fits_nil == {outs for outs in KAPPAS if outs[2] == 0}
    assert fits_cons == {outs for outs in KAPPAS
                         if outs[0] == 0 and outs[2] == 1}
    assert not fits_nil & fits_cons


def test_strict_needs_an_empty_parameter_by_types():
    empty = Algebra(
        LISTS, {nil: Para(Ty(), S, Box("initial", Ty(), S)),
                cons: Para(A @ S, S, Box("cell", A @ S, S))},
        {A: A, S: S}, Para)
    strict = LaxHomomorphism(empty, Diagram.id(Ty()))
    for m in range(4):
        assert strict.preservation(chain(m)).is_invertible
        assert para.same(strict(chain(m)), empty(LISTS.term(chain(m))))
    algebra = diagram_algebra()
    with pytest.raises(ValueError):
        LaxHomomorphism(algebra, Diagram.id(P))
    tied = LaxHomomorphism(algebra, Diagram.copy(P))
    assert not tied.preservation(chain(1)).is_invertible


def test_box_reparametrisation_lifts_into_one_hole_only():
    algebra, trees = diagram_algebra(), tree_diagram_algebra()
    r = Box("r", Ty("R"), P)
    alpha = para.Reparam.of(algebra(LISTS.term(chain(0))), r)
    lifted = algebra.lift_reparam(cons, alpha)
    assert lifted.inside == r
    assert para.same(lifted.cod, algebra.lift(cons, alpha.cod))
    leaf = para.Reparam.of(trees(TREES.term(LEAF)), r)
    with pytest.raises(AxiomError):
        trees.lift_reparam(node, leaf, leaf)


@pytest.mark.parametrize("deterministic", [False, True])
def test_a_box_in_kappa_fails_both_axioms(deterministic):
    """ A box is neither discarded (h >> ! != !) nor copied. """
    box = Box("h", P, P, is_deterministic=deterministic)
    hom = LaxHomomorphism(diagram_algebra(), Diagram.copy(P) >> P @ box)
    assert not hom.unitality() and not hom.multiplication()


def test_lax_errors():
    plain, algebra = list_algebra(param=False), diagram_algebra()
    trees = tree_diagram_algebra()
    for call in (plain.unitality, lambda: plain.associativity(cons),
                 lambda: plain.lift(cons)):
        with pytest.raises(ValueError, match="parametric"):
            call()
    with pytest.raises(ValueError, match="parametric"):
        LaxHomomorphism(plain, Diagram.copy(Ty()))
    with pytest.raises(ValueError, match="Expected a"):
        LaxHomomorphism(algebra, "kappa")
    with pytest.raises(ValueError, match="map into"):
        LaxHomomorphism(algebra, Box("k", P, P))
    fold = trees(TREES.term(LEAF))
    wide = fold.reparam(Diagram.from_function([0], P @ P))
    with pytest.raises(ValueError, match="one parameter"):
        trees.lift(node, fold, wide)
    with pytest.raises(ValueError, match="got a reparametrisation"):
        trees.lift(node, fold, para.Reparam.id(fold))
    with pytest.raises(ValueError, match="got a map"):
        trees.lift_reparam(node, para.Reparam.id(fold), fold)
    with pytest.raises(ValueError, match="at least one"):
        trees.lift_reparam(node)
    with pytest.raises(ValueError, match="Expected a param"):
        algebra.lift(nil)
    with pytest.raises(ValueError, match="same inside"):
        trees.lift_reparam(node, para.Reparam.id(fold), para.Reparam.of(
            fold, Box("r", Ty("R"), P)))
    with pytest.raises(ValueError, match="Expected 1 maps"):
        algebra.lift(cons, fold, fold)
    with pytest.raises(ValueError, match="Expected 1 maps"):
        algebra.associativity(cons)
    hom = LaxHomomorphism(algebra, Diagram.copy(P))
    for value in (("Foo", ), ("Cons", ), ("Nil", ("Nil", )), "Nil", ()):
        with pytest.raises(ValueError):
            hom(value)
