# -*- coding: utf-8 -*-

"""
Belief propagation: the sum-product algorithm as a network with a trace.

A constraint satisfaction problem is a tensor network: a tensor per factor,
a wire per node, and contracting it sums over the assignments, weighted.
That contraction is :func:`marginals`, one ``einsum`` per node. Sum-product
:cite:p:`Pearl88` computes it by passing messages instead: a round is a
network on the messages, :func:`sum_product` is its trace, and
:meth:`~discopy.neural.network.Network.unroll` runs it for a number of
rounds.

The law is that on a factor table whose incidence graph is a tree, the
rounds of :func:`sum_product` give the :func:`marginals`, exactly, as soon
as there are as many rounds as the :func:`diameter` of the tree. On a graph
with cycles they give an approximation.

The messages are logarithms, so that the zero a trace starts from is the
uniform message and a node adds what its factors send it.

Summary
-------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    sum_product
    marginals
    diameter
    exclusive
    belief
    reply
    weigh

Example
-------
A path of three nodes with two states each, the middle one in both factors:

>>> import numpy as np
>>> factors = ((0, 1), (1, 2))
>>> tensors = np.array([[1., 2.], [3., 4.]]), np.array([[5., 6.], [7., 8.]])
>>> network = sum_product(factors, states=2)
>>> network.dom, network.cod
(Dims(Dim(2, 2), Dim(2, 2)), Dims(Dim(2), Dim(2), Dim(2)))
>>> beliefs = network.to_function(rounds=diameter(factors))(*tensors)
>>> assert np.allclose(beliefs, marginals(factors, *tensors))
>>> np.round(beliefs[1], 3)
array([0.328, 0.672])
>>> network.arg.draw(
...     doctest="docs/_static/neural/sum-product.svg", figsize=(8, 6))

.. image:: /_static/neural/sum-product.svg
    :align: center
"""

from __future__ import annotations

from string import ascii_letters

from discopy.monoidal import Dim
from discopy.neural.network import Box, Dims, Network, Trace, namespace
from discopy.neural.wiring import memberships, table_of


def exclusive(*messages) -> tuple:
    """
    What a node sends each of its factors: the sum of the messages of all
    the others, zero when there is none. The others are summed rather than
    the factor's own subtracted from the total, which an impossible state,
    of logarithm minus infinity, would turn into no number at all.

    Parameters:
        messages : The message of each factor of the node, a logarithm.

    Example
    -------
    >>> import numpy as np
    >>> [int(sent) for sent in exclusive(*np.array([1, 2, 4]))]
    [6, 5, 3]
    """
    zero = namespace(messages[0]).zeros_like(messages[0])
    return tuple(sum(messages[:i] + messages[i + 1:], zero)
                 for i in range(len(messages)))


def normalise(weights):
    """ Divide positive weights by their sum along the last axis. """
    return weights / weights.sum(-1)[..., None]


def weigh(message):
    """
    The weights of a message, i.e. its exponential, up to a factor: the
    largest logarithm is taken off first, so that a sharp message, of
    logarithms too low to exponentiate, keeps its ratios.

    Parameters:
        message : The logarithms of positive weights, along the last axis.

    Example
    -------
    >>> import numpy as np
    >>> weigh(np.array([-801., -800.])).round(3).tolist()
    [0.368, 1.0]
    """
    array = namespace(message)
    return array.exp(message - array.amax(message, -1)[..., None])


def belief(*messages):
    """
    The belief of a node: the product of the messages of all its factors,
    normalised, i.e. the softmax of their sum.

    Parameters:
        messages : The message of each factor of the node, a logarithm.

    Example
    -------
    >>> import numpy as np
    >>> belief(np.log([1., 3.]), np.log([2., 2.]))
    array([0.25, 0.75])
    """
    return normalise(weigh(sum(messages)))


def reply(tensor, *messages) -> tuple:
    """
    What a factor sends each of its nodes: its tensor contracted with the
    messages of all the others, normalised, as a logarithm.

    Parameters:
        tensor : The tensor of the factor, positive, one axis per node.
        messages : The message of each node of the factor, a logarithm,
            at most as many as the letters an ``einsum`` names axes with.

    Example
    -------
    >>> import numpy as np
    >>> tensor, zero = np.array([[1., 0.], [1., 2.]]), np.zeros(2)
    >>> [np.exp(message).tolist() for message in reply(tensor, zero, zero)]
    [[0.25, 0.75], [0.5, 0.5]]
    """
    if len(messages) > len(ascii_letters):
        raise ValueError(f"A factor holds at most {len(ascii_letters)} "
                         f"nodes, got {len(messages)}.")
    array, axes = namespace(tensor), ascii_letters[:len(messages)]
    weights = list(map(weigh, messages))

    def to(slot):
        others = [k for k in range(len(messages)) if k != slot]
        return array.log(normalise(array.einsum(
            ",".join(["..." + axes] + ["..." + axes[k] for k in others])
            + "->..." + axes[slot], tensor, *(weights[k] for k in others))))

    return tuple(map(to, range(len(messages))))


def sum_product(factors, states: int) -> Network:
    """
    The sum-product algorithm on a factor table as a network with a trace,
    from the tensor of each factor to the belief of each node: the traced
    legs are the messages from the factors to the nodes, one per member of
    each factor, and a round sends them through :func:`exclusive` on every
    node, :func:`reply` on every factor and :func:`belief` on every node.

    Parameters:
        factors : The factor table, see
            :func:`discopy.neural.wiring.table_of`, of any arities.
        states : The number of states of a node, at least two.

    Note
    ----
    Its :meth:`~discopy.neural.network.Network.unroll` runs the rounds from
    the uniform message, on tensors with any leading dimensions, e.g. a
    batch of problems on one table. A learned propagator is this network
    with other boxes on its factors.
    """
    table = table_of(factors)
    nodes = memberships(table)
    message = Dims(states)
    tensors = Dims(*(Dim(*len(factor) * (states, )) for factor in table))
    ports = [(index, node) for index, factor in enumerate(table)
             for node in factor]
    state = message ** len(ports)

    def send(box, *legs) -> list:
        sent = box(*legs)
        return list(sent) if isinstance(sent, (list, tuple)) else [sent]

    @Network.from_callable(tensors @ state, message ** len(nodes) @ state)
    def body(*legs):
        old = dict(zip(ports, legs[len(table):]))
        asked, new = {}, {}
        for node, of in enumerate(nodes):
            box = Box('node', message ** len(of), message ** len(of),
                      module=exclusive)
            asked.update(zip(((index, node) for index in of), send(
                box, *(old[index, node] for index in of))))
        for index, factor in enumerate(table):
            box = Box('factor', tensors[index] @ message ** len(factor),
                      message ** len(factor), module=reply)
            new.update(zip(((index, node) for node in factor), send(
                box, legs[index], *(asked[index, node] for node in factor))))
        beliefs = [
            send(Box('belief', message ** len(of), message, module=belief),
                 *(new[index, node] for index in of))[0]
            for node, of in enumerate(nodes)]
        return (*beliefs, *(new[port] for port in ports))

    return Trace(body, mem=state)


def marginals(factors, *tensors) -> list:
    """
    The marginal of each node in the tensor network of a factor table: the
    contraction of all the tensors but for the wire of the node,
    normalised, along a path of ``opt_einsum`` when it is installed. This
    is what :func:`sum_product` computes on a tree.

    Parameters:
        factors : The factor table, see
            :func:`discopy.neural.wiring.table_of`.
        tensors : The tensor of each factor, positive, one axis per node.

    Example
    -------
    >>> import numpy as np
    >>> tensor = np.array([[1., 0.], [1., 2.]])
    >>> [marginal.tolist() for marginal in marginals(((0, 1), ), tensor)]
    [[0.25, 0.75], [0.5, 0.5]]
    """
    table = table_of(factors)
    if len(tensors) != len(table):
        raise ValueError(
            f"Expected {len(table)} tensors, got {len(tensors)}.")
    operands = [x for pair in zip(tensors, map(list, table)) for x in pair]
    try:
        # pylint: disable-next=import-outside-toplevel  # an optional extra
        from opt_einsum import contract
    except ImportError:
        contract = namespace(tensors[0]).einsum
    return [normalise(contract(*operands, [node]))
            for node in range(len(memberships(table)))]


def diameter(factors) -> int:
    """
    The diameter of the incidence graph of a factor table, i.e. the longest
    of the shortest paths between its nodes and factors, each membership
    one step: the number of rounds after which :func:`sum_product` is exact
    on a tree.

    Parameters:
        factors : The factor table, see
            :func:`discopy.neural.wiring.table_of`, connected.

    Example
    -------
    >>> diameter(((0, 1), (1, 2)))
    4
    """
    table = table_of(factors)
    nodes = memberships(table)
    edges = {("node", node): [("factor", index) for index in of]
             for node, of in enumerate(nodes)}
    edges.update({("factor", index): [("node", node) for node in factor]
                  for index, factor in enumerate(table)})

    def eccentricity(start):
        seen, frontier, steps = {start}, [start], -1
        while frontier:
            frontier = [y for x in frontier for y in edges[x]
                        if y not in seen and not seen.add(y)]
            steps += 1
        if len(seen) < len(edges):
            raise ValueError("The factor table is not connected.")
        return steps

    return max(map(eccentricity, edges))
