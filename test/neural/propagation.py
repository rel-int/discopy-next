# -*- coding: utf-8 -*-

import random

import numpy as np
from pytest import importorskip, raises

from discopy.neural import propagation
from discopy.neural.network import Dim, Dims, Trace
from discopy.neural.propagation import (
    belief, diameter, exclusive, marginals, reply, sum_product)


def random_tree(rng, n_factors=5, max_arity=3):
    """ A factor table whose incidence graph is a tree: every new factor
    holds one node there already and new ones. """
    table, n_nodes = [], 1
    for _ in range(n_factors):
        fresh = rng.randrange(max_arity)
        table.append(sorted(
            [rng.randrange(n_nodes), *range(n_nodes, n_nodes + fresh)]))
        n_nodes += fresh
    return tuple(map(tuple, table))


def random_tensors(rng, factors, states, batch=()):
    return [rng.uniform(.1, 2., batch + len(factor) * (states, ))
            for factor in factors]


def gap(beliefs, exact):
    return max(float(abs(b - e).max()) for b, e in zip(beliefs, exact))


def test_sum_product_is_the_contraction_on_trees():
    rng, numbers = random.Random(0), np.random.default_rng(0)
    for _ in range(20):
        factors, states = random_tree(rng), rng.choice([2, 3])
        tensors = random_tensors(numbers, factors, states)
        network, exact = sum_product(factors, states), marginals(
            factors, *tensors)
        assert network.cod == Dims(states) ** len(exact)
        for rounds in (diameter(factors), diameter(factors) + 3):
            beliefs = network.to_function(np.ndarray, rounds)(*tensors)
            assert gap(np.atleast_2d(beliefs), exact) < 1e-6


def test_sum_product_needs_its_rounds():
    factors = ((0, 1), (1, 2), (2, 3), (3, 4))
    tensors = random_tensors(np.random.default_rng(0), factors, 2)
    network, exact = sum_product(factors, 2), marginals(factors, *tensors)
    gaps = [gap(network.to_function(rounds=rounds)(*tensors), exact)
            for rounds in range(1, 5)]
    assert diameter(factors) == 8 and gaps[0] > 1e-3 and gaps[-1] < 1e-12
    assert gaps == sorted(gaps, reverse=True)


def test_a_reply_leaves_out_the_node_it_is_sent_to(monkeypatch):
    factors = ((0, 1), (1, 2), (1, 3))
    tensors = random_tensors(np.random.default_rng(0), factors, 3)
    exact, rounds = marginals(factors, *tensors), diameter(factors)

    def echo(tensor, *messages):
        return tuple(sent + message for sent, message in zip(
            reply(tensor, *messages), messages))

    right = sum_product(factors, 3).to_function(rounds=rounds)(*tensors)
    monkeypatch.setattr(propagation, "reply", echo)
    wrong = sum_product(factors, 3).to_function(rounds=rounds)(*tensors)
    assert gap(right, exact) < 1e-12 < 1e-3 < gap(wrong, exact)


def test_sum_product_approximates_on_a_cycle():
    factors = ((0, 1), (0, 2), (1, 2))
    tensors = random_tensors(np.random.default_rng(0), factors, 2)
    beliefs = sum_product(factors, 2).to_function(rounds=50)(*tensors)
    assert 1e-6 < gap(beliefs, marginals(factors, *tensors)) < .1


def test_sum_product_on_a_batch():
    factors = ((0, ), (0, 1, 2), (2, 3))
    batch = random_tensors(np.random.default_rng(0), factors, 2, (4, ))
    network = sum_product(factors, 2)
    assert isinstance(network, Trace) and len(network.mem) == 6
    assert network.dom == Dims(2, Dim(2, 2, 2), Dim(2, 2))
    beliefs = network.to_function(rounds=diameter(factors))(*batch)
    assert all(node.shape == (4, 2) for node in beliefs)
    for row in range(4):
        exact = marginals(factors, *(tensor[row] for tensor in batch))
        assert gap([node[row] for node in beliefs], exact) < 1e-12


def test_messages():
    assert exclusive(*np.array([1, 2, 4])) == (6, 5, 3)
    assert exclusive(np.array(5)) == (0, )
    impossible = np.array([-np.inf, 0.])
    assert all(np.array_equal(message, impossible) for message in exclusive(
        impossible, impossible, impossible))
    assert np.allclose(belief(np.log([1., 3.])), [.25, .75])
    tensor = np.array([[1., 0.], [1., 2.]])
    first, second = reply(tensor, np.log([1., 3.]), np.zeros(2))
    assert np.allclose(np.exp(first), [.25, .75])
    assert np.allclose(np.exp(second), [.4, .6])
    assert diameter(((0, ), )) == 1 and diameter(((0, 1), (1, 2), (0, 2))) == 3
    with raises(ValueError, match="not connected"):
        diameter(((0, 1), (2, 3)))


def test_sum_product_torch():
    torch = importorskip("torch")
    factors = ((0, 1), (1, 2))
    tensors = [torch.tensor(tensor, requires_grad=True) for tensor in
               random_tensors(np.random.default_rng(0), factors, 2)]
    network = sum_product(factors, 2)
    beliefs = network.to_function(torch.Tensor, diameter(factors))(*tensors)
    exact = marginals(factors, *tensors)
    assert gap([b.detach() for b in beliefs], [e.detach() for e in exact])\
        < 1e-12
    gradients = [torch.randn(2, dtype=torch.float64) for _ in beliefs]
    expected = torch.autograd.grad(
        network.to_function(rounds=1)(*tensors), tensors, gradients)
    backprop = network.to_lens(torch.Tensor, rounds=1).put(
        *(tensor.detach() for tensor in tensors), *gradients)
    assert gap(backprop, expected) < 1e-9
