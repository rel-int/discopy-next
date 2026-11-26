# -*- coding: utf-8 -*-

"""
Equivariant layers as weight tying: the orbits of a group on pairs of
points, the dense layer they give and the networks built from it.
"""

import itertools
from functools import partial

import numpy
from pytest import importorskip, raises

from discopy import para
from discopy.group import Action, Cyclic, Product, Symmetric, Wreath
from discopy.neural.network import Box, Dim, Dims, Network
from discopy.python.finset import Permutation

TORUS = Product[Cyclic[3], Cyclic[3]]
LINES = Wreath[Symmetric[2], Symmetric[2]]
ACTIONS = (Cyclic[6].action(), Symmetric[4].action(), TORUS.action(),
           LINES.action(), Wreath[Cyclic[2], Symmetric[3]].action())


def modules():
    """ The two modules that need PyTorch, or skip the test. """
    return importorskip("torch"), importorskip("discopy.neural.equivariant")


def brute(dom, cod):
    """
    The orbit of each pair ``(y, x)`` by flood fill along the generators,
    numbered in the order the pairs are met, independently of
    :meth:`Action.orbits`.
    """
    label = {}
    for start in itertools.product(range(cod.size), range(dom.size)):
        if start in label:
            continue
        label[start], stack = len(set(label.values())), [start]
        while stack:
            y, x = stack.pop()
            for g, h in zip(cod.images, dom.images):
                if (g[y], h[x]) not in label:
                    label[g[y], h[x]] = label[start]
                    stack.append((g[y], h[x]))
    return label


def brute_tying(dom, cod):
    label = brute(dom, cod)
    return numpy.array([[label[y, x] for x in range(dom.size)]
                        for y in range(cod.size)])


def count(dom, cod):
    return int(tying(dom, cod).max()) + 1


def tying(dom, cod):
    return importorskip("discopy.neural.equivariant").tying(dom, cod)


def test_counts():
    """
    The number of weights, from the known number of orbits: the partitions
    of 3 points and of 4 points for the symmetric group.
    """
    importorskip("torch")
    sym = Symmetric[5].action()
    assert count(sym, sym) == 2
    assert count(sym.tensor(sym), sym) == 5
    assert count(sym.tensor(sym), sym.tensor(sym)) == 15
    cyclic = Cyclic[6].action()
    assert count(cyclic, cyclic) == 6
    free = Action.trivial(Cyclic[6], 6)
    assert count(free, free) == 36
    torus = TORUS.action()
    assert torus.size == 9 and count(torus, torus) == 9
    for action in (sym, cyclic, torus):
        invariant = Action.trivial(action.group)
        assert count(action, invariant) == 1
        assert tying(action, invariant).shape == (1, action.size)


def test_tying():
    """
    The entries are the orbit numbers, as a flood fill finds them, with
    the rows indexed by ``cod`` and the columns by ``dom``.
    """
    importorskip("torch")
    sym, cyclic = Symmetric[5].action(), Cyclic[6].action()
    assert (tying(sym, sym) == 1 - numpy.eye(5, dtype=int)).all()
    shifts = [[(x - y) % 6 for x in range(6)] for y in range(6)]
    assert (tying(cyclic, cyclic) == numpy.array(shifts)).all()
    free = Action.trivial(Cyclic[6], 6)
    assert (tying(free, free) == numpy.arange(36).reshape(6, 6)).all()
    for action in ACTIONS:
        for other in (action, action.tensor(action)):
            assert (tying(other, action) == brute_tying(other, action)).all()
            assert (tying(action, other) == brute_tying(action, other)).all()
    with raises(TypeError):
        tying(Cyclic[3].action(), Cyclic[2].action())


def test_tying_orientation():
    """
    A non-square pair is indexed ``[y, x]``: a fixed input point is tied to
    one weight for each of its rows, a fixed output point to one for each of
    its columns.
    """
    importorskip("torch")
    cyclic = Cyclic[6].action()
    pairs = cyclic.tensor(cyclic)
    assert tying(cyclic, pairs).shape == (36, 6)
    assert (tying(cyclic, pairs) == brute_tying(cyclic, pairs)).all()
    assert (tying(pairs, cyclic) == brute_tying(pairs, cyclic)).all()
    fixed = Action.trivial(Cyclic[6], 2)
    columns = tying(fixed, cyclic)
    assert columns.shape == (6, 2)
    assert (columns == columns[0]).all() and columns[0, 0] != columns[0, 1]
    rows = tying(cyclic, fixed)
    assert rows.shape == (2, 6)
    assert (rows == rows[:, :1]).all() and rows[0, 0] != rows[1, 0]


def test_tying_trivial():
    """
    A trivial action tensors with any action of its group: the orbits of the
    pairs are those of the action, once for each point of the trivial one.
    """
    importorskip("torch")
    action = LINES.action()
    trivial = Action.trivial(LINES, 3)
    orbits = len(action.orbits())
    single = tying(action, Action.trivial(LINES))[0]
    for y, row in enumerate(tying(action, trivial)):
        assert (row == single + y * orbits).all()


def test_act():
    torch, equivariant = modules()
    p, q = Permutation((1, 2, 0)), Permutation((1, 0, 2))
    v = torch.tensor([[10., 1.], [20., 2.], [30., 3.]])
    assert equivariant.act(p, v).tolist() == [[30., 3.], [10., 1.], [20., 2.]]
    assert equivariant.act(Permutation.id(3), v).equal(v)
    batch = torch.randn(4, 5, 3, 2)
    both = equivariant.act(q, equivariant.act(p, batch))
    assert both.equal(equivariant.act(p >> q, batch))
    assert both[..., (p >> q)[1], :].equal(batch[..., 1, :])


def test_equivariant_and_not():
    """ Tied weights commute with every generator, a dense matrix does not. """
    torch, equivariant = modules()
    torch.manual_seed(0)
    for action in ACTIONS:
        for cod in (action, action.tensor(action)):
            dense = equivariant.Dense(action, cod)
            w = torch.randn(dense.orbits, 2, 3, dtype=torch.float64)
            b = torch.randn(dense.bias_orbits, 2, dtype=torch.float64)
            v = torch.randn(4, action.size, 3, dtype=torch.float64)
            tied = partial(dense, w=w, b=b)
            assert equivariant.equivariance(tied, action, cod, v) < 1e-5
        free = Action.trivial(action.group, action.size)
        dense = equivariant.Dense(free, free)
        w = torch.randn(dense.orbits, 1, 1, dtype=torch.float64)
        v = torch.randn(4, action.size, 1, dtype=torch.float64)
        assert equivariant.equivariance(
            partial(dense, w=w), action, action, v) > 1e-2


def test_equivariance():
    torch, equivariant = modules()
    rotations = Cyclic[3].action()
    v = torch.tensor([[1.], [2.], [4.]])
    assert equivariant.equivariance(lambda v: v, rotations, rotations) == 0.0
    assert equivariant.equivariance(
        lambda v: v.flip(0), rotations, rotations, v) == 3.0
    assert equivariant.equivariance(
        lambda v: v.flip(0), rotations, rotations, v, 2 * v) == 6.0
    with raises(TypeError):
        equivariant.equivariance(
            lambda v: v, rotations, Cyclic[2].action(), v)


def explicit(torch, dom, cod, v, w, b=None):
    """
    The explicit sum over points and channels, with the orbits of a flood
    fill, and the bias if any.
    """
    label = brute(dom, cod)
    bias = brute(Action.trivial(cod.group), cod)
    return torch.tensor([[[
        (0 if b is None else b[bias[y, 0], d].item()) + sum(
            w[label[y, x], d, c].item() * v[n, x, c].item()
            for x in range(dom.size) for c in range(3))
        for d in range(2)] for y in range(cod.size)]
        for n in range(v.shape[0])], dtype=torch.float64)


def test_reference():
    """ The layer is the explicit sum over points and channels. """
    torch, equivariant = modules()
    torch.manual_seed(1)
    for action in (Cyclic[3].action(), Symmetric[3].action()):
        for dom in (action, action.tensor(action)):
            dense = equivariant.Dense(dom, action)
            w = torch.randn(dense.orbits, 2, 3, dtype=torch.float64)
            b = torch.randn(dense.bias_orbits, 2, dtype=torch.float64)
            v = torch.randn(2, dom.size, 3, dtype=torch.float64)
            actual = dense(v, w, b)
            assert actual.shape == (2, action.size, 2)
            torch.testing.assert_close(
                actual, explicit(torch, dom, action, v, w, b))
            torch.testing.assert_close(
                dense(v, w), explicit(torch, dom, action, v, w))
            assert dense.orbits == len(set(brute(dom, action).values()))


def test_layer():
    """
    The legs drop the unit dimensions, the shapes follow the channels, and
    the weights are 2 on the diagonal and 3 elsewhere.
    """
    torch, equivariant = modules()
    action = Symmetric[4].action()
    full = equivariant.layer(action, action, (3, 2), bias=True)
    assert full.dom == Dims(Dim(4, 3)) and full.cod == Dims(Dim(4, 2))
    assert [dim.inside for dim in full.param.inside] == [(2, 2, 3), (2,)]
    assert full.inside.boxes[0].name == "W"
    one = equivariant.layer(action, action, name="V")
    assert one.dom == one.cod == Dims(4) and one.param == Dims(2)
    assert one.inside.boxes[0].name == "V"
    w, v = torch.tensor([2., 3.]), torch.tensor([1., 2., 4., 8.])
    module = importorskip("discopy.neural.torch").Module(one, w)
    assert module(v).tolist() == [2 * x + 3 * (15 - x) for x in v.tolist()]
    sums = equivariant.layer(action, Action.trivial(action.group))
    assert sums.cod == Dims(1) and sums.param == Dims(1)


def test_channels():
    """
    A layer to an invariant output with channels and a bias, compiled to a
    module, is the dense layer called directly with the same weights.
    """
    torch, equivariant = modules()
    torch.manual_seed(4)
    action = Symmetric[4].action()
    invariant = Action.trivial(action.group, 1)
    composite = equivariant.layer(
        action, invariant, channels=(2, 3), bias=True)
    weights = [torch.randn(dim.inside, dtype=torch.float64)
               for dim in composite.param.inside]
    module = importorskip("discopy.neural.torch").Module(composite, *weights)
    dense = equivariant.Dense(action, invariant)
    v = torch.randn(5, 4, 2, dtype=torch.float64)
    expected = dense(v, weights[0].reshape(dense.orbits, 3, 2),
                     weights[1].reshape(dense.bias_orbits, 3))
    assert module(v).shape == (5, 3)
    torch.testing.assert_close(module(v), expected[:, 0, :])


def network(equivariant, torch, action, middle=None, fast=False):
    """
    equivariant >> tanh >> equivariant >> tanh >> invariant, the middle
    layer being on the action ``middle`` if given.
    """
    points = Dims(action.size)
    tanh = para.Markov[Network].lift(
        Box("tanh", points, points, module=torch.tanh))
    layer = partial(equivariant.layer, fast=fast)
    invariant = Action.trivial(action.group)
    middle = middle or action
    return layer(action, action) >> tanh >> layer(middle, middle) >> tanh\
        >> layer(action, invariant)


def compiled(torch, composite):
    """ The module of a network with random weights. """
    weights = [torch.randn(dim.inside, dtype=torch.float64)
               for dim in composite.param.inside]
    return importorskip("discopy.neural.torch").Module(composite, *weights)


def equivariance_of(equivariant, module, action, v):
    """ The invariance of the output: the group acts trivially on it. """
    one = Action.trivial(action.group)
    return equivariant.equivariance(
        lambda u: module(u[..., 0])[..., None, None], action, one, v)


def test_composite():
    """ The tutorial network: 78, 13 and 5 weights, an invariant output. """
    torch, equivariant = modules()
    torch.manual_seed(2)
    trivial = Action.trivial(Cyclic[6], 6)
    for fast in (False, True):
        for action, size in ((trivial, 78), (Cyclic[6].action(), 13),
                             (Symmetric[6].action(), 5)):
            module = compiled(
                torch, network(equivariant, torch, action, fast=fast))
            assert sum(p.numel() for p in module.parameters()) == size
            v = torch.randn(5, 6, 1, dtype=torch.float64)
            assert module(v[..., 0]).shape == (5, )
            assert equivariance_of(equivariant, module, action, v) < 1e-9
            module(v[..., 0]).pow(2).sum().backward()
            assert all(p.grad.abs().sum() > 0 for p in module.parameters())


def test_composite_untied():
    """
    The negative control: the same network on rotations, with its middle
    layer built from a trivial action, i.e. untied, is not invariant.
    """
    torch, equivariant = modules()
    torch.manual_seed(5)
    action = Cyclic[6].action()
    untied = network(equivariant, torch, action, Action.trivial(Cyclic[6], 6))
    module = compiled(torch, untied)
    v = torch.randn(5, 6, 1, dtype=torch.float64)
    assert equivariance_of(equivariant, module, action, v) > 1e-2


def test_wreath():
    """ Wreath products and their products have equivariant layers. """
    torch, equivariant = modules()
    torch.manual_seed(3)
    for group, orbits in ((LINES, 3), (Product[LINES, LINES], 9)):
        action = group.action()
        dense = equivariant.Dense(action, action)
        w = torch.randn(dense.orbits, 1, 1, dtype=torch.float64)
        v = torch.randn(3, action.size, 1, dtype=torch.float64)
        assert dense.orbits == orbits
        assert equivariant.equivariance(
            partial(dense, w=w), action, action, v) < 1e-9


def test_fast_layer():
    """
    The fast layer has the parameters and the outputs of the dense one,
    through a module, with a bias, channels and every kind of realisation.
    """
    torch, equivariant = modules()
    module = importorskip("discopy.neural.torch").Module
    torch.manual_seed(6)
    symmetric, rotations = Symmetric[4].action(), Cyclic[6].action()
    pairs = Symmetric[3].action().tensor(Symmetric[3].action())
    for dom, cod, channels in (
            (symmetric, symmetric, (3, 2)), (rotations, rotations, (2, 2)),
            (symmetric, Action.trivial(symmetric.group), (2, 3)),
            (Symmetric[3].action(), pairs, (1, 2))):
        slow = equivariant.layer(dom, cod, channels, bias=True)
        fast = equivariant.layer(dom, cod, channels, bias=True, fast=True)
        assert (slow.dom, slow.cod, slow.param) == (
            fast.dom, fast.cod, fast.param)
        weights = [torch.randn(dim.inside, dtype=torch.float64)
                   for dim in slow.param.inside]
        v = torch.randn(5, *slow.dom.inside[0].inside, dtype=torch.float64)
        torch.testing.assert_close(
            module(fast, *weights)(v), module(slow, *weights)(v))


def test_unbatched():
    """
    An unbatched input gives the batched output without its batch axis,
    down to a single point with one channel, which has no axis at all.
    """
    torch, equivariant = modules()
    torch.manual_seed(6)
    action = Symmetric[5].action()
    for cod, shape in ((Action.trivial(action.group), ()), (action, (5, ))):
        for bias in (False, True):
            module = compiled(torch, equivariant.layer(action, cod, bias=bias))
            v = torch.randn(action.size, dtype=torch.float64)
            assert module(v).shape == shape
            torch.testing.assert_close(module(v), module(v[None])[0])
