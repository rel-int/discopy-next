# -*- coding: utf-8 -*-

"""
Equivariant layers factorised along the construction of the group: each
realisation is checked against the dense layer, in outputs and gradients.
"""

import itertools
import random
import time
from functools import partial

import numpy
from pytest import importorskip, mark, raises

from discopy.group import Action, Cyclic, Product, Symmetric, Wreath
from discopy.python.finset import Permutation

TORUS = Product[Cyclic[3], Cyclic[3]]
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
CYCLICS = (
    Cyclic[3].action(), Cyclic[4].action(), Cyclic[7].action(),
    Product[C2, C3].action(), Product[C3, C4].action(),
    Product[C4, S3].action(), Product[S3, C4].action(),
    Product[Product[C2, C3], S3].action(),
    Product[C3, Product[C3, C3]].action(),
    Wreath[C3, C2].action(power=True))
CHANNELS = (1, 2, 3, 5)
PAIRS = ((2, 3), (3, 2), (1, 1))
CELLS4 = Wreath[Wreath[Wreath[S3, S3], S3], S3]
POWERS = (
    Wreath[Product[Symmetric[2], Symmetric[2]], C2].action(power=True),
    Wreath[Product[C3, Symmetric[2]], C2].action(power=True),
    Product[C3, Wreath[S3, C2]].action(
        C3.action(), Wreath[S3, C2].action(power=True)))


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


def brute_tying(dom, cod):
    """ The orbits of the pairs by flood fill along the generators. """
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
    return numpy.array([[label[y, x] for x in range(dom.size)]
                        for y in range(cod.size)])


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


def test_kronecker():
    """ The chains of every product, wreath and power are Dense. """
    torch, equivariant, factorised = modules()
    for seed, action in enumerate(CHAINS):
        fast = factorised.Kronecker(action, action)
        assert isinstance(fast, equivariant.Realisation)
        agrees(torch, equivariant, fast, seed)


def test_cyclic():
    """ Cyclic factors, mixed with chains, direct and by transform. """
    torch, equivariant, factorised = modules()
    for seed, action in enumerate(CYCLICS):
        for fft in (False, True):
            fast = factorised.Kronecker(action, action, fft)
            assert any(f.cyclic for f in fast.structure.factors)
            agrees(torch, equivariant, fast, seed)


def test_cyclic_tying():
    """ The tie numbering of the power action is that of the orbits. """
    _, _, factorised = modules()
    action = Wreath[C3, C2].action(power=True)
    fast = factorised.Kronecker(action, action)
    assert (fast.tying() == tying(action, action)).all()
    assert (fast.tying() == brute_tying(action, action)).all()


def test_cyclic_selection():
    """ The size picks the path, and more than three axes need the transform. """
    torch, equivariant, factorised = modules()
    Kronecker = factorised.Kronecker
    action = Product[C3, C4].action()
    assert Kronecker.fft_from == 64 and Kronecker(action, action).fft is None
    always = type("Always", (Kronecker, ), {"fft_from": 1})
    never = type("Never", (Kronecker, ), {"fft_from": 10 ** 6})
    for kind in (always, never):
        agrees(torch, equivariant, kind(action, action), 3)
    many = Product[Product[C3, C3], Product[C3, C3]].action()
    with raises(TypeError):
        Kronecker(many, many, fft=False)
    agrees(torch, equivariant, Kronecker(many, many), 4)


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


def test_partitions():
    """ Set partitions of one or two points on each side are Dense. """
    torch, equivariant, factorised = modules()
    seed = 0
    for n in (2, 3, 4, 5):
        base = Symmetric[n].action()
        for p, q in itertools.product((1, 2), repeat=2):
            dom = base if p == 1 else base.tensor(base)
            cod = base if q == 1 else base.tensor(base)
            fast = factorised.Partitions(dom, cod)
            assert (fast.p, fast.q) == (p, q)
            agrees(torch, equivariant, fast, seed)
            seed += 1
            if n >= 4:
                assert fast.orbits == {2: 2, 3: 5, 4: 15}[p + q]


def test_partitions_numbering():
    """ Fewer points than coordinates leave out the partitions. """
    _, _, factorised = modules()
    assert [len(factorised.partitions(4, n)) for n in (1, 2, 3, 4)] == [
        1, 8, 14, 15]
    base = Symmetric[2].action()
    both = base.tensor(base)
    assert factorised.Partitions(both, both).orbits == 8
    assert (factorised.Partitions(base, both).tying()
            == tying(base, both)).all()


def test_partitions_refused():
    """ Other tensor powers and groups have no partitions. """
    _, _, factorised = modules()
    base = Symmetric[3].action()
    cube = base.tensor(base).tensor(base)
    for dom, cod in ((cube, base), (base, cube), (cube, cube),
                     (Cyclic[3].action(), Cyclic[3].action()),
                     (Symmetric[1].action(), Symmetric[1].action()),
                     (Wreath[Symmetric[2], Symmetric[2]].action(),
                      Wreath[Symmetric[2], Symmetric[2]].action()),
                     (base, Symmetric[4].action())):
        with raises(TypeError):
            factorised.Partitions(dom, cod)


def test_float32():
    """ Single precision agrees up to the size of the output. """
    torch, equivariant, factorised = modules()
    action, cyclic = Symmetric[5].action(), Cyclic[7].action()
    for fast in (factorised.Kronecker(action, action),
                 factorised.Kronecker(cyclic, cyclic, False),
                 factorised.Kronecker(cyclic, cyclic, True),
                 factorised.Pooling(action, Action.trivial(action.group))):
        dense = equivariant.Dense(fast.dom, fast.cod)
        v, w, b, _ = draw(torch, fast, 7, torch.float32)
        got, want = fast(v, w, b), dense(v, w, b)
        assert (got - want).abs().max() <= 1e-5 * (1 + want.abs().max())


def test_buffers():
    """ The weights are gathered only if the tie is not the identity. """
    torch, _, factorised = modules()
    symmetric = Symmetric[3].action()
    power = Wreath[Symmetric[3], Cyclic[2]].action(power=True)
    assert factorised.Kronecker(symmetric, symmetric).tie is None
    assert factorised.Kronecker(power, power).tie.tolist() == [0, 1, 1, 2]
    assert not hasattr(factorised.Kronecker(symmetric, symmetric), "mobius")
    assert factorised.Kronecker(symmetric, symmetric).chains == (2, )
    assert factorised.Partitions(symmetric, symmetric).mobius.dtype\
        == torch.get_default_dtype()


def test_structure():
    """ The structures of the natural actions, and what has none. """
    _, _, factorised = modules()
    structure, Kronecker = factorised.structure, factorised.Kronecker
    symmetric = Symmetric[3].action()
    assert structure(symmetric).tie == (0, 1)
    assert structure(Symmetric[1].action()) is None
    power = structure(Wreath[Symmetric[3], Cyclic[2]].action(power=True))
    assert power.tie == ((0, 1), (1, 2))
    assert power.shape == (3, 3)
    assert structure(Cyclic[4].action()).tie == (0, 1, 2, 3)
    assert structure(TORUS.action()).shape == (3, 3)
    assert structure(Cyclic[1].action()) is None
    assert structure(Product[Cyclic[1], Cyclic[3]].action()) is None
    refused = (
        Wreath[Cyclic[3], Symmetric[2]].action(),
        Wreath[Symmetric[2], Cyclic[3]].action(),
        Action(Symmetric[3], 3, symmetric.images[::-1]))
    for action in refused:
        assert structure(action) is None
        with raises(TypeError):
            Kronecker(action, action)
    with raises(TypeError):
        Kronecker(symmetric, refused[-1])
    with raises(TypeError):
        factorised.Pooling(symmetric, symmetric)
    with raises(TypeError):
        factorised.Pooling(symmetric, Action.trivial(Cyclic[3]))


def test_realise():
    """ The realisation is chosen by structure, and equal to Dense. """
    torch, equivariant, factorised = modules()
    symmetric, rotations = Symmetric[4].action(), Cyclic[5].action()
    pairs = symmetric.tensor(symmetric)
    for seed, (dom, cod, kind) in enumerate((
            (symmetric, Action.trivial(symmetric.group), "Pooling"),
            (Action.trivial(symmetric.group), pairs, "Pooling"),
            (symmetric, symmetric, "Kronecker"),
            (rotations, rotations, "Kronecker"),
            (symmetric, pairs, "Partitions"),
            (pairs, pairs, "Partitions"))):
        fast = factorised.realise(dom, cod)
        assert type(fast).__name__ == kind
        agrees(torch, equivariant, fast, seed)


def test_realise_refused():
    """ What has no realisation is Dense, and no other raises. """
    _, equivariant, factorised = modules()
    symmetric = Symmetric[3].action()
    cube = symmetric.tensor(symmetric).tensor(symmetric)
    rotations = Cyclic[4].action()
    twice = (
        Wreath[C3, Symmetric[2]].action(),
        Wreath[Symmetric[2], C3].action(),
        Wreath[Product[C2, C2], Symmetric[2]].action(),
        Product[Cyclic[1], C3].action(),
        cube, Action(Symmetric[3], 3, symmetric.images[::-1]))
    for dom, cod in tuple((a, a) for a in twice) + (
            (rotations, rotations.tensor(rotations)), (symmetric, cube)):
        fast = factorised.realise(dom, cod)
        assert type(fast) is equivariant.Dense
        with raises(TypeError):
            factorised.Kronecker(dom, cod)
        with raises(TypeError):
            factorised.Partitions(dom, cod)


def test_scale(monkeypatch):
    """ No orbit of pairs is ever computed: sizes beyond a dense matrix. """
    torch, equivariant, factorised = modules()
    base = Symmetric[64].action()
    start = time.perf_counter()
    fast = factorised.Partitions(base.tensor(base), base.tensor(base))
    assert fast.orbits == 15 and time.perf_counter() - start < 5

    def refuse(*_):
        raise AssertionError("The orbits of pairs were computed.")

    monkeypatch.setattr(Action, "tensor", refuse)
    monkeypatch.setattr(equivariant, "tying", refuse)
    monkeypatch.setattr(factorised, "tying", refuse, raising=False)
    for action, orbits in (
            (Symmetric[1024].action(), 2), (Cyclic[4096].action(), 4096),
            (Product[Cyclic[64], Cyclic[64]].action(), 4096)):
        start = time.perf_counter()
        fast = factorised.realise(action, action)
        assert type(fast) is factorised.Kronecker and fast.orbits == orbits
        assert time.perf_counter() - start < 5
        out = fast(torch.ones(1, action.size, 1), torch.ones(orbits, 1, 1))
        assert out.shape == (1, action.size, 1)


def test_scale_refusal(monkeypatch):
    """ A relabelled action is refused by size, without its tensor power. """
    _, _, factorised = modules()

    def refuse(*_):
        raise AssertionError("The tensor of actions was computed.")

    action = conjugate(Symmetric[1024].action())
    assert action != Symmetric[1024].action()
    monkeypatch.setattr(Action, "tensor", refuse)
    start = time.perf_counter()
    assert factorised.structure(action) is None
    assert factorised.Partitions.degree(action) is None
    assert time.perf_counter() - start < 5


def test_realise_conjugate():
    """ A relabelled action has no structure: Dense, equal to the dense. """
    torch, equivariant, factorised = modules()
    canonical = (
        Cyclic[5].action(), Product[C3, S3].action(), LINES.action(),
        Wreath[S3, C2].action(power=True))
    for seed, action in enumerate(canonical):
        other = conjugate(action)
        assert other != action and factorised.structure(other) is None
        for dom, cod in ((other, other), (action, other), (other, action)):
            fast = factorised.realise(dom, cod)
            assert type(fast) is equivariant.Dense
            dense = equivariant.Dense(dom, cod)
            assert (fast.tying() == dense.tying()).all()
            agrees(torch, equivariant, fast, seed)
    for dom, cod in ((S3.action(), conjugate(S3.action())),
                     (conjugate(S3.action()), S3.action())):
        assert type(factorised.realise(dom, cod)) is equivariant.Dense


def test_channels():
    """ Fixed numbers of channels, every realisation. """
    torch, equivariant, factorised = modules()
    symmetric, rotations = S3.action(), Cyclic[5].action()
    pairs = symmetric.tensor(symmetric)
    trivial = Action.trivial(S3)
    fasts = (
        factorised.Kronecker(symmetric, symmetric),
        factorised.Kronecker(rotations, rotations, False),
        factorised.Kronecker(rotations, rotations, True),
        factorised.Pooling(symmetric, trivial),
        factorised.Pooling(trivial, pairs),
        factorised.Partitions(symmetric, pairs),
        factorised.Partitions(pairs, pairs))
    for seed, fast in enumerate(fasts):
        for channels in PAIRS:
            agrees(torch, equivariant, fast, seed, channels=channels)


@mark.slow
def test_cases():
    """ Wider products and powers over a base with several factors. """
    torch, equivariant, factorised = modules()
    for seed, action in enumerate(POWERS + (
            Product[CELLS4, S3].action(), )):
        fast = factorised.Kronecker(action, action)
        agrees(torch, equivariant, fast, seed)
    for seed, action in enumerate(POWERS):
        for fft in (False, True):
            agrees(torch, equivariant, factorised.Kronecker(
                action, action, fft), seed)


@mark.slow
def test_cases_slow():
    """
    A product with the symmetric group of nine points, on 81 points: the
    dense reference computes the orbits of all the pairs of points, so a
    bigger base, e.g. ``CELLS4`` with its 729 points, takes minutes.
    """
    torch, equivariant, factorised = modules()
    action = Product[Wreath[S3, S3], Symmetric[9]].action()
    agrees(torch, equivariant, factorised.Kronecker(action, action), 0)


def test_cyclic_default():
    """ The default path of a cyclic group is Dense on both sides of 64. """
    torch, equivariant, factorised = modules()
    for n in (60, 63, 64, 96):
        action = Cyclic[n].action()
        fast = factorised.Kronecker(action, action)
        assert fast.fft is None
        agrees(torch, equivariant, fast, n)


def test_imprimitive_never_builds_power(monkeypatch):
    """ Recognising an imprimitive action does not build the power one. """
    _, _, factorised = modules()
    group = Wreath[Symmetric[8], Symmetric[8]]
    action, build = group.action(), group.action

    def guard(*args, power=False, **kwargs):
        assert not power
        return build(*args, power=power, **kwargs)

    monkeypatch.setattr(group, "action", guard)
    start = time.perf_counter()
    assert factorised.structure(action) is not None
    assert factorised.Kronecker(action, action).dom == action
    assert time.perf_counter() - start < 5


def test_float32_large():
    """ Single precision stays within 1e-4 of the double precision Dense. """
    torch, equivariant, factorised = modules()
    cyclic = Cyclic[256].action()
    lines = Wreath[Symmetric[8], Symmetric[8]].action()
    for seed, fast in enumerate((
            factorised.Kronecker(cyclic, cyclic, True),
            factorised.Kronecker(lines, lines))):
        dense = equivariant.Dense(fast.dom, fast.cod)
        v, w, b, g = draw(torch, fast, seed)
        want = gradients(torch, dense, v, w, b, g)
        args = [t.detach().float().requires_grad_() for t in (v, w, b)]
        got = gradients(torch, fast, *args, g.float())
        for x, y in zip(got, want):
            assert (x.double() - y).abs().max() <= 1e-4 * y.abs().max()


def test_tying_device():
    """ The tables are read from a module in double precision or on CUDA. """
    torch, _, factorised = modules()
    symmetric, rotations = S3.action(), Cyclic[5].action()
    pairs = symmetric.tensor(symmetric)
    fasts = (
        factorised.Kronecker(symmetric, symmetric),
        factorised.Kronecker(rotations, rotations),
        factorised.Pooling(symmetric, Action.trivial(S3)),
        factorised.Pooling(Action.trivial(S3), symmetric),
        factorised.Partitions(symmetric, pairs),
        importorskip("discopy.neural.equivariant").Dense(rotations, rotations))
    for fast in fasts:
        want = tying(fast.dom, fast.cod)
        assert (fast.double().tying() == want).all()
        if torch.cuda.is_available():
            assert (fast.cuda().tying() == want).all()
