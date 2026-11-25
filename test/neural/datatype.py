from random import Random

import pytest

import discopy
from discopy import markov, para, python
from discopy.markov import Box, Diagram, Ty
from discopy.neural.datatype import Algebra, Datatype, chain

A, S, P = Ty("A"), Ty("S"), Ty("P")
LISTS, TREES = Datatype.list(A, S), Datatype.tree(A, S)
nil, cons = LISTS.constructors
leaf, node = TREES.constructors
Int = python.Ty(int)


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


@pytest.mark.parametrize("value", [
    ("Snoc", ), ("Cons", ), ("Nil", ("Nil", )),
    ("Cons", ("Nil", ), ("Nil", ))])
def test_term_errors(value):
    with pytest.raises(ValueError):
        LISTS.term(value)


@pytest.mark.parametrize("value", ["Nil", (), ["Nil"]])
def test_term_not_a_tuple(value):
    with pytest.raises(ValueError):
        LISTS.term(value)


def test_bad_constructors():
    with pytest.raises(ValueError):  # the carrier before data
        Datatype(Box("c", S @ A, S))
    with pytest.raises(ValueError):  # the carrier in the middle
        Datatype(Box("c", S @ A @ S, S))
    with pytest.raises(ValueError):  # the codomain is not the carrier
        Datatype(Box("a", Ty(), S), Box("b", S, A))
    with pytest.raises(ValueError):  # two carriers
        Datatype(Box("a", Ty(), S), Box("b", Ty("T"), Ty("T")))
    with pytest.raises(ValueError):  # codomain not atomic
        Datatype(Box("a", Ty(), S @ S))
    with pytest.raises(ValueError):  # no constructors
        Datatype()
    with pytest.raises(ValueError):  # same name twice
        Datatype(Box("a", Ty(), S), Box("a", S, S))
    with pytest.raises(TypeError):
        Datatype("Nil")
    with pytest.raises(ValueError):
        chain(-1)


# 2. folds into python functions


def list_algebra(param=True):
    if not param:
        return Algebra(
            LISTS, {nil: python.Function(lambda: 0, python.Ty(), Int),
                    cons: python.Function(
                        lambda a, s: a + 2 * s, Int @ Int, Int)},
            {A: Int, S: Int}, python.Function)
    Net = para.Markov[python.Function]
    return Algebra(
        LISTS, {nil: Net(python.Ty(), Int, python.Function(
                    lambda p: p, Int, Int), param=Int),
                cons: Net(Int @ Int, Int, python.Function(
                    lambda a, s, p: a + p * s, Int ** 3, Int), param=Int)},
        {A: Int, S: Int}, Net)


def tree_algebra():
    Net = para.Markov[python.Function]
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
    Net = para.Markov[python.Function]
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

Para = para.Markov[Diagram]


def diagram_algebra():
    return Algebra(
        LISTS, {nil: Para(Ty(), S, Box("initial", P, S), param=P),
                cons: Para(A @ S, S, Box("cell", A @ S @ P, S), param=P)},
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
    algebra = Algebra(
        TREES, {leaf: Para(A, S, Box("leaf", A @ P, S), param=P),
                node: Para(S @ S, S, Box("node", S @ S @ P, S), param=P)},
        {A: A, S: S}, Para)
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

    def grad(p):
        return torch.zeros_like(p) if p.grad is None else p.grad

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
        assert torch.allclose(grad(module.weights[0]), grad(h0), atol=1e-6)
        assert torch.allclose(grad(module.weights[1]), grad(w), atol=1e-6)


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

    algebra = Algebra(
        TREES, {leaf: Para(A, S, Box("leaf", A @ P, S), param=P),
                node: Para(S @ S, S, Box("node", S @ S @ P, S), param=P)},
        {A: A, S: S}, Para)
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
        assert torch.allclose(module.weights[0].grad, w.grad, atol=1e-6)


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
