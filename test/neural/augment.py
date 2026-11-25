# -*- coding: utf-8 -*-

"""
Augmentation and symmetrisation: seeded draws of group elements, the
Reynolds average against the orbit means of a matrix, and the homomorphisms
of actions.
"""

import itertools
import random
from functools import partial

import numpy
from pytest import importorskip, raises

from discopy import para
from discopy.group import Action, Cyclic, Product, Symmetric, Wreath
from discopy.kleisli.channel import Channel
from discopy.kleisli.monad import Seed
from discopy.neural.network import Box, Dim, Dims, Network

TORUS = Product[Cyclic[3], Cyclic[3]]
LINES = Wreath[Symmetric[2], Symmetric[2]]
ACTIONS = (Cyclic[6].action(), Symmetric[4].action(), TORUS.action(),
           LINES.action(), Wreath[Cyclic[2], Symmetric[3]].action())


def modules():
    """ The modules that need PyTorch, or skip the test. """
    return importorskip("torch"), importorskip("discopy.neural.augment")


def brute(dom, cod):
    """
    The orbit of each pair ``(y, x)`` by flood fill along the generators,
    independently of :meth:`Action.orbits`.
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


def orbit_means(matrix, dom, cod):
    """ The matrix with each entry replaced by its mean on its orbit. """
    label = brute(dom, cod)
    total, size = {}, {}
    for (y, x), k in label.items():
        total[k] = total.get(k, 0.) + matrix[y, x]
        size[k] = size.get(k, 0) + 1
    return numpy.array([[total[label[y, x]] / size[label[y, x]]
                         for x in range(dom.size)] for y in range(cod.size)])


def test_sample_is_pure():
    _, augment = modules()
    for group in (Cyclic[6], Symmetric[4], TORUS):
        draw = augment.sample(group)
        g, after = draw(420)
        assert draw(420) == (g, after)
        assert after != 420 and draw(after)[1] != after
        assert g in set(group.elements())
        assert len({draw(seed)[1] for seed in range(20)}) == 20


CHI_SQUARE = {2: 13.816, 5: 20.515}
""" The 0.999 quantiles of the chi-square law, by degrees of freedom. """


def chi_square(group, draws):
    """ The chi-square statistic of the elements drawn against uniform. """
    counts = dict.fromkeys(group.elements(), 0)
    for g in draws:
        counts[g] += 1
    expected = len(draws) / group.order()
    return sum((n - expected) ** 2 / expected for n in counts.values())


def test_sample_is_uniform():
    """
    The chi-square statistic against the uniform law is below the 0.999
    quantile, on consecutive seeds and on the chain of next seeds.
    """
    _, augment = modules()
    size = 6000
    for group in (Cyclic[3], Symmetric[3]):
        draw, chain, seed = augment.sample(group), [], 0
        for _ in range(size):
            g, seed = draw(seed)
            chain.append(g)
        threshold = CHI_SQUARE[group.order() - 1]
        assert chi_square(group, chain) < threshold
        assert chi_square(
            group, [draw(s)[0] for s in range(size)]) < threshold


def test_kleisli_composition():
    """
    Composing the augmentation with itself threads the seed through two draws,
    and a draw of ``sample`` composes with an arrow of the same monad.
    """
    torch, augment = modules()
    torch.manual_seed(7)
    dom, cod = Cyclic[6].action(), Action(Cyclic[6], 2, ((1, 0), ))
    arrow = augment.augment(dom, cod)
    x = torch.randn(6, 2, dtype=torch.float64)
    y = torch.randn(2, 3, dtype=torch.float64)
    for composite in (arrow.then(arrow), arrow >> arrow):
        for seed in range(5):
            (a, b), middle = arrow((x, y))(seed)
            (c, d), after = arrow((a, b))(middle)
            (u, v), last = composite((x, y))(seed)
            assert u.equal(c) and v.equal(d) and last == after
    draw = augment.sample(Symmetric[3])
    pick = Channel[Seed](lambda _: draw, tuple, object)
    pair = Channel[Seed](
        lambda g: lambda seed: ((g, draw(seed)[0]), draw(seed)[1]),
        object, tuple)
    for seed in range(5):
        g, middle = draw(seed)
        (first, second), after = pick.then(pair)(())(seed)
        assert first == g and (second, after) == draw(middle)


def test_augment_pure_and_one_element():
    """
    The same element acts on the input and the label: with the same action
    on both, a pair ``(v, v)`` stays a pair of equal tensors, equal to the
    action of an element on ``v``, and the same seed gives the same pair.
    """
    torch, augment = modules()
    equivariant = importorskip("discopy.neural.equivariant")
    torch.manual_seed(0)
    action = Cyclic[6].action()
    arrow = augment.augment(action, action)
    v = torch.randn(3, 6, 2, dtype=torch.float64)
    images = [equivariant.act(action(g), v) for g in Cyclic[6].elements()]
    outputs = []
    for seed in range(12):
        (x, y), after = arrow((v, v))(seed)
        (x2, y2), after2 = arrow((v, v))(seed)
        assert x.equal(y) and x.equal(x2) and y.equal(y2) and after == after2
        assert any(x.equal(image) for image in images)
        outputs.append(x)
    assert any(not x.equal(outputs[0]) for x in outputs)
    with raises(TypeError):
        augment.augment(action, Cyclic[2].action())


def test_augment_preserves_the_relation():
    """ If ``y = f(x)`` for an equivariant ``f`` then so is ``y' = f(x')``. """
    torch, augment = modules()
    equivariant = importorskip("discopy.neural.equivariant")
    torch.manual_seed(1)
    for action in ACTIONS:
        for cod in (action, Action.trivial(action.group)):
            dense = equivariant.Dense(action, cod)
            w = torch.randn(dense.orbits, 2, 3, dtype=torch.float64)
            x = torch.randn(4, action.size, 3, dtype=torch.float64)
            arrow = augment.augment(action, cod)
            for seed in range(5):
                (u, v), _ = arrow((x, dense(x, w)))(seed)
                torch.testing.assert_close(v, dense(u, w))


def test_augmentation_box():
    """
    The box has a leg for the key, the input and the label, composes and
    runs, and a composite draws what the composite of the arrows draws.
    """
    torch, augment = modules()
    module = importorskip("discopy.neural.torch").Module
    torch.manual_seed(2)
    dom, cod = Cyclic[6].action(), Symmetric[3].action()
    box = augment.augmentation(
        dom, Action.trivial(Cyclic[6], 2), name="aug")
    assert box.name == "aug"
    assert box.dom == box.cod == Dims(1) @ Dims(6) @ Dims(2)
    box = augment.augmentation(dom, dom)
    assert box.dom == box.cod == Dims(1) @ Dims(6) @ Dims(6)
    x = torch.randn(6, dtype=torch.float64)
    y = torch.randn(6, dtype=torch.float64)
    key = torch.tensor(42)
    network = box >> box
    key2, u, v = module(network)(key, x, y)
    arrow = augment.augment(dom, dom)
    pair = (x[:, None], y[:, None])
    (a, b), after = arrow(pair)(42)
    (c, d), after = arrow((a, b))(after % 2 ** 63)
    assert key2.item() == after % 2 ** 63 and key2.dtype == key.dtype
    assert u.equal(c[:, 0]) and v.equal(d[:, 0])
    assert augment.augmentation(cod, cod).dom == Dims(1) @ Dims(3) @ Dims(3)


def test_augmentation_box_with_batch_and_single_point():
    """ Leading axes ride along and a one-point leg has no axis. """
    torch, augment = modules()
    one = Action.trivial(Cyclic[3], 1)
    box = augment.augmentation(Cyclic[3].action(), one)
    assert box.cod == Dims(1) @ Dims(3) @ Dims(1)
    x, y = torch.randn(5, 3), torch.randn(5)
    key, u, v = box.forward(torch.tensor([7]), x, y)
    assert key.shape == (1, ) and u.shape == x.shape and v.equal(y)
    assert sorted(u[0].tolist()) == sorted(x[0].tolist())


def test_augmentation_box_one_element_per_batch():
    """
    One element acts on every row of a batch, through two different actions
    on the input and the label, the one that ``sample`` draws from the seed.
    """
    torch, augment = modules()
    equivariant = importorskip("discopy.neural.equivariant")
    dom, cod = Cyclic[6].action(), Action(Cyclic[6], 2, ((1, 0), ))
    box = augment.augmentation(dom, cod)
    x = torch.arange(6.) + 100 * torch.arange(5.)[:, None]
    y = torch.arange(2.) + 100 * torch.arange(5.)[:, None]
    for seed in range(8):
        key, u, v = box.forward(torch.tensor(seed), x, y)
        g, after = augment.sample(Cyclic[6])(seed)
        expected_x = torch.stack([
            equivariant.act(dom(g), row[:, None])[:, 0] for row in x])
        expected_y = torch.stack([
            equivariant.act(cod(g), row[:, None])[:, 0] for row in y])
        assert u.equal(expected_x) and v.equal(expected_y)
        assert key.item() == after % augment.MODULUS


def test_augmentation_box_refuses_bad_keys():
    torch, augment = modules()
    box = augment.augmentation(Cyclic[3].action(), Cyclic[3].action())
    x = torch.randn(3)
    with raises(ValueError, match=r"shape \(2,\)"):
        box.forward(torch.tensor([1, 2]), x, x)
    with raises(ValueError, match="non-negative"):
        box.forward(torch.tensor(-1), x, x)


def test_symmetrise_is_orbit_mean():
    """ The exact Reynolds operator of a matrix is its orbit means. """
    torch, augment = modules()
    equivariant = importorskip("discopy.neural.equivariant")
    torch.manual_seed(3)
    for action in ACTIONS:
        for dom in (action, action.tensor(action)):
            w = torch.randn(action.size, dom.size, dtype=torch.float64)
            v = torch.randn(3, dom.size, 1, dtype=torch.float64)
            reynolds = augment.symmetrise(lambda u: w @ u, dom, action)
            expected = orbit_means(w.numpy(), dom, action)
            actual = reynolds(v)
            assert actual.shape == (3, action.size, 1)
            numpy.testing.assert_allclose(
                actual.numpy(), expected @ v.numpy(), atol=1e-9)
            dense, tying = equivariant.Dense(dom, action), equivariant.tying(
                dom, action)
            weights = torch.zeros(dense.orbits, 1, 1, dtype=torch.float64)
            for y, x in numpy.ndindex(action.size, dom.size):
                weights[tying[y, x]] = float(expected[y, x])
            torch.testing.assert_close(actual, dense(v, weights))


def test_symmetrise_projects():
    """
    The symmetrised map is equivariant, idempotent, and the identity on an
    equivariant map.
    """
    torch, augment = modules()
    equivariant = importorskip("discopy.neural.equivariant")
    torch.manual_seed(4)
    for action in ACTIONS:
        w = torch.randn(action.size, action.size, dtype=torch.float64)
        v = torch.randn(3, action.size, 2, dtype=torch.float64)
        once = augment.symmetrise(lambda u: w @ u, action, action)
        twice = augment.symmetrise(once, action, action)
        assert equivariant.equivariance(once, action, action, v) < 1e-9
        assert (once(v) - w @ v).abs().max() > 1e-3
        torch.testing.assert_close(twice(v), once(v))
        dense = equivariant.Dense(action, action)
        tied = partial(dense, w=torch.randn(
            dense.orbits, 2, 2, dtype=torch.float64))
        torch.testing.assert_close(
            augment.symmetrise(tied, action, action)(v), tied(v))


def test_symmetrise_samples():
    """ Monte-Carlo converges to the exact average, purely in the seed. """
    torch, augment = modules()
    torch.manual_seed(5)
    action = Symmetric[4].action()
    w = torch.randn(4, 4, dtype=torch.float64)
    v = torch.randn(3, 4, 1, dtype=torch.float64)

    def error(*args):
        approx = augment.symmetrise(lambda u: w @ u, action, action, *args)
        exact = augment.symmetrise(lambda u: w @ u, action, action)
        assert approx(v).equal(approx(v))
        return (approx(v) - exact(v)).abs().max().item()
    assert error(3000, 0) < error(8, 0) / 4
    estimate = augment.symmetrise(lambda u: w @ u, action, action, 8, 1)
    other = augment.symmetrise(lambda u: w @ u, action, action, 8, 2)
    assert not estimate(v).equal(other(v))


def test_symmetrise_samples_unbiased():
    """ One sample per seed averages to the exact operator. """
    torch, augment = modules()
    torch.manual_seed(8)
    action = Symmetric[3].action()
    w = torch.randn(3, 3, dtype=torch.float64)
    v = torch.randn(3, 1, dtype=torch.float64)
    exact = augment.symmetrise(lambda u: w @ u, action, action)(v)
    mean = torch.stack([
        augment.symmetrise(lambda u: w @ u, action, action, 1, seed)(v)
        for seed in range(3000)]).mean(0)
    assert (mean - exact).abs().max() < .15


def test_symmetrise_samples_share_a_prefix():
    """ The draws of ``k`` samples are the first of ``k + 1`` samples. """
    torch, augment = modules()
    equivariant = importorskip("discopy.neural.equivariant")
    torch.manual_seed(9)
    action = Symmetric[4].action()
    w = torch.randn(4, 4, dtype=torch.float64)
    v = torch.randn(4, 2, dtype=torch.float64)
    draw, terms, seed = augment.sample(action.group), [], 3
    for _ in range(6):
        g, seed = draw(seed)
        terms.append(equivariant.act(
            action(g.dagger()), w @ equivariant.act(action(g), v)))
        count = len(terms)
        estimate = augment.symmetrise(
            lambda u: w @ u, action, action, count, 3)(v)
        torch.testing.assert_close(estimate, sum(terms) / count)


def test_reynolds_needs_the_inverse():
    """
    Control: with ``g`` instead of ``g^-1`` on the output the average is a
    different operator, on a non-abelian group as on the others.
    """
    torch, augment = modules()
    equivariant = importorskip("discopy.neural.equivariant")
    torch.manual_seed(10)
    action = Symmetric[4].action()
    w = torch.randn(4, 4, dtype=torch.float64)
    v = torch.randn(4, 2, dtype=torch.float64)
    wrong = sum(
        equivariant.act(action(g), w @ equivariant.act(action(g), v))
        for g in action.group.elements()) / action.group.order()
    right = augment.symmetrise(lambda u: w @ u, action, action)(v)
    expected = orbit_means(w.numpy(), action, action) @ v.numpy()
    numpy.testing.assert_allclose(right.numpy(), expected, atol=1e-9)
    assert (wrong - right).abs().max() > 1e-3


def test_symmetrise_refuses():
    torch, augment = modules()
    big = Symmetric[9].action()
    with raises(ValueError, match="samples"):
        augment.symmetrise(lambda v: v, big, big)
    estimate = augment.symmetrise(lambda v: v, big, big, 4, 0)
    v = torch.randn(9, 1)
    torch.testing.assert_close(estimate(v), v)
    with raises(ValueError):
        augment.symmetrise(lambda v: v, big, big, 0)
    with raises(TypeError):
        augment.symmetrise(
            lambda v: v, Cyclic[3].action(), Cyclic[2].action())


def test_homomorphisms():
    _, augment = modules()
    action = Cyclic[3].action()
    pairs = action.tensor(action)
    for dom, cod, function, expected in (
            (pairs, action, lambda p: p // 3, True),
            (pairs, action, lambda p: p % 3, True),
            (pairs, action, lambda p: (p + 1) // 3, False),
            (action, action, lambda x: x, True),
            (action, Action.trivial(Cyclic[3], 3), lambda x: x, False),
            (action, Action.trivial(Cyclic[3]), lambda x: 0, True)):
        assert augment.is_homomorphism(function, dom, cod) is expected
    first = augment.homomorphism(lambda p: p // 3, pairs, action)
    assert all(first.preservation((g, x))
               for g in Cyclic[3].elements() for x in range(9))
    assert not augment.homomorphism(
        lambda p: (p + 1) // 3, pairs, action).preservation(
            (Cyclic[3](1), 2))


def test_orbit_index_is_a_homomorphism():
    """ A point to its orbit is a homomorphism into the trivial action. """
    _, augment = modules()
    for action in ACTIONS:
        orbits = action.orbits()
        index = {x: k for k, orbit in enumerate(orbits) for x in orbit}
        trivial = Action.trivial(action.group, len(orbits))
        assert augment.is_homomorphism(index.__getitem__, action, trivial)
        if len(orbits) < action.size:
            assert not augment.is_homomorphism(
                lambda x: x, action, Action.trivial(action.group, action.size))


def test_generators_suffice():
    """
    Every function of four points agrees with the check on all elements, and
    the homomorphisms of the rotations to themselves are the four rotations.
    """
    _, augment = modules()
    action = Cyclic[4].action()
    found = []
    for values in itertools.product(range(4), repeat=4):
        brute_force = all(
            values[action(g)[x]] == action(g)[values[x]]
            for g in Cyclic[4].elements() for x in range(4))
        assert augment.is_homomorphism(
            values.__getitem__, action, action) is brute_force
        if brute_force:
            found.append(values)
    assert sorted(found) == [
        tuple((x + c) % 4 for x in range(4)) for c in range(4)]
    with raises(TypeError):
        augment.is_homomorphism(lambda x: x, action, Cyclic[2].action())


def test_generators_suffice_with_several_generators():
    """
    The check on the generators is the check on all elements, on the
    permutations of three points (all 27 maps of points, only the identity
    commutes) and on the torus (the identity, its mutations, random maps).
    """
    _, augment = modules()
    rng = random.Random(0)
    symmetric = Symmetric[3].action()
    torus = TORUS.action()
    size = torus.size
    identity = tuple(range(size))
    mutations = [identity[:i] + (j, ) + identity[i + 1:]
                 for i in range(size) for j in range(size)]
    randoms = [tuple(rng.randrange(size) for _ in range(size))
               for _ in range(200)]
    cases = ((symmetric, itertools.product(range(3), repeat=3)),
             (torus, [identity] + mutations + randoms))
    for action, candidates in cases:
        found = []
        for values in candidates:
            brute_force = all(
                values[action(g)[x]] == action(g)[values[x]]
                for g in action.group.elements() for x in range(action.size))
            assert augment.is_homomorphism(
                values.__getitem__, action, action) is brute_force
            found.extend([values] * brute_force)
        assert tuple(range(action.size)) in found
        assert len(action.group.generators()) > 1


def network(equivariant, torch, action):
    """ equivariant >> tanh >> equivariant >> tanh >> invariant. """
    points = Dims(action.size)
    tanh = para.Markov[Network].lift(
        Box("tanh", points, points, module=torch.tanh))
    layer, invariant = equivariant.layer, Action.trivial(action.group)
    return layer(action, action) >> tanh >> layer(action, action) >> tanh\
        >> layer(action, invariant)


def parametric(torch, name, module, size, param):
    """ A parametric box on the points of a layer of ``size`` points. """
    x = Dims(size)
    return para.Markov[Network](x, x, param=param, inside=Box(
        name, x @ param, x, module=module))


def test_strictness():
    """
    A network of tied layers is invariant for every value of its weights,
    i.e. a strict homomorphism of parametric maps, an untied one is not, and
    the sample matters: a layer equivariant at the zero weights is caught.
    """
    torch, augment = modules()
    equivariant = importorskip("discopy.neural.equivariant")
    for group in (Symmetric[5], Cyclic[6]):
        one, action = Action.trivial(group), group.action()
        tied = network(equivariant, torch, action)
        assert augment.strictness(tied, action, one) < 1e-9
        dense = parametric(
            torch, "W", lambda v, w: v @ w, action.size,
            Dims(Dim(action.size, action.size)))
        assert augment.strictness(dense, action, action) > 1e-2
        bias = parametric(torch, "b", torch.add, action.size,
                          Dims(action.size))
        shifted = equivariant.layer(action, action) >> bias
        assert augment.strictness(shifted, action, action) > 1e-2
        assert augment.strictness(shifted, action, action, draws=0) == 0.0
        zero = augment.strictness(
            equivariant.layer(action, action), action, action)
        assert zero < 1e-9


def test_strictness_seed():
    """ The same seed gives the same float, another one is still small. """
    torch, augment = modules()
    equivariant = importorskip("discopy.neural.equivariant")
    action = Cyclic[6].action()
    one = Action.trivial(Cyclic[6])
    tied = network(equivariant, torch, action)
    dense = parametric(torch, "W", lambda v, w: v @ w, 6, Dims(Dim(6, 6)))
    assert augment.strictness(dense, action, action, seed=1)\
        == augment.strictness(dense, action, action, seed=1)
    assert augment.strictness(dense, action, action, seed=1)\
        != augment.strictness(dense, action, action, seed=2)
    assert all(augment.strictness(tied, action, one, seed=s) < 1e-9
               for s in range(3))
