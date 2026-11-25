# -*- coding: utf-8 -*-

"""
Equivariant layers factorised along the construction of the group: each
realisation is checked against the dense layer, in outputs and gradients.
"""

import random
from functools import partial

from pytest import importorskip, raises

from discopy.group import Action, Cyclic, Product, Symmetric, Wreath
from discopy.python.finset import Permutation

LINES = Wreath[Symmetric[2], Symmetric[2]]
CELLS = Wreath[Wreath[Symmetric[3], Symmetric[3]], Cyclic[2]]
C2, C3, C4, S3 = (Cyclic[2], Cyclic[3], Cyclic[4], Symmetric[3])
CHAINS = (
    Symmetric[2].action(), Symmetric[3].action(), Symmetric[5].action(),
    Cyclic[2].action(), Product[Symmetric[3], Cyclic[2]].action(),
    Product[Symmetric[2], Symmetric[3]].action(),
    Product[Product[Cyclic[2], Symmetric[2]], Symmetric[3]].action(),
    Wreath[Symmetric[2], Symmetric[2]].action(),
    Wreath[Symmetric[3], Symmetric[2]].action(),
    Wreath[Symmetric[2], Cyclic[2]].action(),
    Wreath[Wreath[Symmetric[2], Symmetric[2]], Symmetric[3]].action(),
    Product[LINES, LINES].action(),
    Wreath[Symmetric[3], Cyclic[2]].action(power=True),
    Wreath[Symmetric[2], Symmetric[3]].action(power=True),
    CELLS.action(power=True))
CHANNELS = (1, 2, 3, 5)


def conjugate(action):
    """ The action with its first two points swapped, by relabelling. """
    n = action.size
    sigma = [1, 0] + list(range(2, n))
    images = []
    for g in action.images:
        moved = [0] * n
        for x in range(n):
            moved[sigma[x]] = sigma[g[x]]
        images.append(Permutation(tuple(moved), n))
    return Action(action.group, n, tuple(images))


def modules():
    """ The modules that need PyTorch, or skip the test. """
    return (importorskip("torch"), importorskip("discopy.neural.equivariant"),
            importorskip("discopy.neural.factorised"))


def tying(dom, cod):
    return importorskip("discopy.neural.equivariant").tying(dom, cod)


def draw(torch, fast, seed, dtype=None, channels=None):
    """
    Seeded inputs, weights, biases and an upstream gradient for a
    realisation, channels given or drawn from ``CHANNELS``, the first three
    tracked.
    """
    dtype = dtype or torch.float64
    rng = random.Random(seed)
    c, d = channels or (rng.choice(CHANNELS), rng.choice(CHANNELS))
    generator = torch.Generator().manual_seed(seed)

    def normal(*shape):
        return torch.randn(*shape, generator=generator, dtype=dtype)

    v = normal(2, 3, fast.dom.size, c).requires_grad_()
    w = normal(fast.orbits, d, c).requires_grad_()
    b = normal(fast.bias_orbits, d).requires_grad_()
    return v, w, b, normal(2, 3, fast.cod.size, d)


def gradients(torch, module, v, w, b, g):
    """ The output and its gradients with respect to ``v``, ``w``, ``b``. """
    out = module(v, w, b)
    return (out, ) + torch.autograd.grad(out, (v, w, b), g)


def agrees(torch, equivariant, fast, seed=0, dtype=None, channels=None):
    """
    A realisation numbers its orbits as :func:`tying`, is equal to
    :class:`Dense` in outputs and gradients, and is equivariant.
    """
    dense = equivariant.Dense(fast.dom, fast.cod)
    assert (fast.tying() == tying(fast.dom, fast.cod)).all()
    assert (fast.orbits, fast.bias_orbits) == (
        dense.orbits, dense.bias_orbits)
    v, w, b, g = draw(torch, fast, seed, dtype, channels)
    for got, want in zip(gradients(torch, fast, v, w, b, g),
                         gradients(torch, dense, v, w, b, g)):
        assert (got - want).abs().max() <= 1e-5 * (1 + want.abs().max())
    tied = partial(fast, w=w.detach(), b=b.detach())
    assert equivariant.equivariance(
        tied, fast.dom, fast.cod, v.detach()) <= 1e-5


def test_pooling():
    """ A single point on either side is Dense, for any action. """
    torch, equivariant, factorised = modules()
    point = Action.trivial(Cyclic[3])
    pairs = (
        (Action.trivial(Cyclic[3], 4), point), (point, point),
        (Cyclic[6].action(), Action.trivial(Cyclic[6])),
        (Action.trivial(Symmetric[4]), Symmetric[4].action().tensor(
            Symmetric[4].action())),
        (S3.action().tensor(S3.action()), Action.trivial(S3)))
    for action in CHAINS:
        trivial = Action.trivial(action.group)
        pairs += ((action, trivial), (trivial, action))
    for seed, (dom, cod) in enumerate(pairs):
        agrees(torch, equivariant, factorised.Pooling(dom, cod), seed)
    fast = factorised.Pooling(Action.trivial(Cyclic[3], 4), point)
    assert fast.orbits == 4 and fast.bias_orbits == 1
    symmetric = S3.action()
    with raises(TypeError):
        factorised.Pooling(symmetric, symmetric)
    with raises(TypeError):
        factorised.Pooling(symmetric, Action.trivial(Cyclic[3]))


def test_realise():
    """ A side of one point is Pooling, else Dense, equal to Dense. """
    torch, equivariant, factorised = modules()
    symmetric, rotations = Symmetric[4].action(), Cyclic[5].action()
    pairs = symmetric.tensor(symmetric)
    for seed, (dom, cod, kind) in enumerate((
            (symmetric, Action.trivial(symmetric.group), "Pooling"),
            (Action.trivial(symmetric.group), pairs, "Pooling"),
            (symmetric, symmetric, "Dense"),
            (rotations, rotations, "Dense"),
            (symmetric, pairs, "Dense"),
            (pairs, pairs, "Dense"))):
        fast = factorised.realise(dom, cod)
        assert type(fast).__name__ == kind
        agrees(torch, equivariant, fast, seed)


def test_realise_refused():
    """ What has no single point is Dense, and nothing raises. """
    torch, equivariant, factorised = modules()
    symmetric = S3.action()
    cube = symmetric.tensor(symmetric).tensor(symmetric)
    rotations = Cyclic[4].action()
    twice = (
        Wreath[C3, Symmetric[2]].action(),
        Wreath[Symmetric[2], C3].action(),
        Wreath[Product[C2, C2], Symmetric[2]].action(),
        Product[Cyclic[1], C3].action(),
        cube, Action(S3, 3, symmetric.images[::-1]),
        conjugate(Cyclic[5].action()), conjugate(LINES.action()))
    for seed, (dom, cod) in enumerate(tuple((a, a) for a in twice) + (
            (rotations, rotations.tensor(rotations)), (symmetric, cube),
            (symmetric, conjugate(symmetric)))):
        fast = factorised.realise(dom, cod)
        assert type(fast) is equivariant.Dense
        assert (fast.tying() == tying(dom, cod)).all()
        if seed < 4:
            agrees(torch, equivariant, fast, seed)


def test_tying_device():
    """ The tables are read from a module in double precision or on CUDA. """
    torch, equivariant, factorised = modules()
    symmetric, rotations = S3.action(), Cyclic[5].action()
    fasts = (
        factorised.Pooling(symmetric, Action.trivial(S3)),
        factorised.Pooling(Action.trivial(S3), symmetric),
        equivariant.Dense(rotations, rotations))
    for fast in fasts:
        want = tying(fast.dom, fast.cod)
        assert (fast.double().tying() == want).all()
        if torch.cuda.is_available():
            assert (fast.cuda().tying() == want).all()
