# -*- coding: utf-8 -*-

"""
Recursive message passing as a diagram: a closed map drawn by
:func:`~discopy.neural.signature.from_incidence` or
:func:`~discopy.neural.signature.from_relation` interpreted as a network with
a trace.

The map is a morphism of a compact closed category, whose boxes are the
nodes and the factors, sent to the :class:`~discopy.interaction.Int`
construction over :class:`~discopy.neural.network.Network`: a wire swaps the
legs of its two ends, a self-wired pair of ports is one directed leg, the
state of a node, and the trace over the shared legs is one *round*, the
synchronous round of :class:`~discopy.neural.recursion.Recursion`. Unrolling
the trace from zeros runs the first rounds of a cycle from its initial state.

The boxes hold adaptors of the modules the network already owns, so that the
parameters are the same objects.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Fire
    Reply

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        closed
        legs
        select
        reads
        boxes
        interpret
        of

Example
-------
>>> import torch  # doctest: +EXTRA
>>> from discopy.neural.recursion import PeerNet
>>> triangle = PeerNet((4, 6, 8, 5), ((1, 2), (0, 2), (0, 1)), 3, 2, 3, 1)
>>> trace = of(triangle)
>>> len(trace.mem)
9
>>> s = triangle.initial(torch.ones(2, 3, 3))
>>> given, memory = legs(triangle, s)
>>> after = trace.arg.to_function(torch.Tensor)(*given, *memory)
>>> expected = legs(triangle, triangle.round(s))[1]
>>> all(torch.allclose(x, y) for x, y in zip(after, expected + expected))
True
"""

from __future__ import annotations

from itertools import accumulate

import torch
from torch import nn

from discopy import frobenius
from discopy.neural.cell import Site
from discopy.neural.network import Box, Dims, Function, Network, Trace
from discopy.neural.recursion import FactorNet, PeerNet
from discopy.neural.signature import (
    NODE, Orbit, Signature, Sym, from_incidence, from_relation)
from discopy.neural.wiring import memberships, ports, where

MESSAGE, STATE = frobenius.Ty("message"), frobenius.Ty("state")
""" The roles of the ports of a node and a factor: what a neighbour says, and
what a node remembers. """


class Fire(nn.Module):
    """
    A node module of a recursive network as the module of a box: the clue,
    the answer, one message per port then the state in, one belief per port
    then the state out, so that the belief is copied to every port.

    Parameters:
        site : The :class:`~discopy.neural.cell.Site` of the network, which
               the adaptor holds and does not copy.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.neural.cell import Site, Widths
    >>> site = Site(Widths(4, 6, 8, 5))
    >>> clue, answer = torch.rand(2, 4), torch.rand(2, 5)
    >>> messages, state = torch.rand(2, 3, 4), torch.rand(2, 6)
    >>> *beliefs, state = Fire(site)(clue, answer, *messages.unbind(1), state)
    >>> len(beliefs), tuple(state.shape)
    (3, (2, 6))
    """
    def __init__(self, site: Site):
        super().__init__()
        self.site = site

    def forward(self, clue, answer, *messages_state):
        """
        Parameters:
            clue : The clue of the node, ``(batch, dim)``.
            answer : The answer of the node, ``(batch, y_dim)``.
            messages_state : One message per port, each ``(batch, dim)``,
                             then the state, ``(batch, state_dim)``.

        Returns:
            One belief per port, the same one, then the new state.
        """
        # pylint: disable=arguments-differ
        *messages, state = messages_state
        belief, state = Site.forward(
            self.site, torch.stack(messages, -2)[:, None], state[:, None],
            clue[:, None], answer[:, None])
        return (belief[:, 0], ) * len(messages) + (state[:, 0], )


class Reply(nn.Module):
    """
    A factor module of a recursive network as the module of a box: one
    belief per member in, one reply per member out.

    Parameters:
        unit : The :class:`~discopy.neural.cell.DeepSet` of the network,
               which the adaptor holds and does not copy.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.neural.cell import DeepSet, Widths
    >>> unit = DeepSet(Widths(4, 6, 8, 5))
    >>> replies = Reply(unit)(*torch.rand(3, 2, 4))
    >>> len(replies), tuple(replies[0].shape)
    (3, (2, 4))
    """
    def __init__(self, unit: nn.Module):
        super().__init__()
        self.unit = unit

    def forward(self, *messages):
        """
        Parameters:
            messages : One belief per member, each ``(batch, dim)``.

        Returns:
            One reply per member, each ``(batch, dim)``.
        """
        replies = self.unit(torch.stack(messages, -2)[:, None])
        return tuple(replies[:, 0].unbind(-2))


def closed(net: PeerNet | FactorNet) -> frobenius.CMap:
    """
    The closed map a recursive network is wired by, drawn from the tables
    it gathers along: one node box per node and, for a
    :class:`~discopy.neural.recursion.FactorNet`, one factor box per factor,
    named as the network names its factor module.

    The peers table of a :class:`~discopy.neural.recursion.PeerNet` must be
    symmetric and irreflexive, as a relation drawn by
    :func:`~discopy.neural.signature.from_relation` is.

    Parameters:
        net : A :class:`~discopy.neural.recursion.PeerNet` or a
              :class:`~discopy.neural.recursion.FactorNet`.

    Raises:
        ValueError : If the peers table is not symmetric.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.neural.recursion import PeerNet
    >>> from discopy.neural.wiring import from_map
    >>> net = PeerNet((4, 6, 8, 5), ((1, 2), (0, 2), (0, 1)), 3, 2, 3, 1)
    >>> from_map(closed(net))
    ((1, 2), (0, 2), (0, 1))
    """
    node = Signature((Orbit(MESSAGE, 1, Sym.PERM), Orbit(STATE, traced=True)))
    if isinstance(net, PeerNet):
        return from_relation(
            net.peers.tolist(),  # ty: ignore[call-non-callable]
            node)
    factor = Signature((Orbit(MESSAGE, 1, Sym.PERM), ))
    table = net.to_units.view(  # ty: ignore[call-non-callable]
        net.n_units, net.arity).tolist()
    return from_incidence(memberships(table, net.n_cells), node, factor,
                          factor_name=net.factor)


def legs(net: PeerNet | FactorNet, s) -> tuple[tuple, tuple]:
    """
    The state of a recursive network as the legs of its diagram: ``(given,
    memory)``, the clue and the answer of every node, and the legs of the
    trace, in the order of :func:`of`.

    A node's legs are the messages at its real ports, then its state, and a
    factor's legs are the beliefs of its members. The padding of a
    :class:`~discopy.neural.recursion.FactorNet` has no leg, and both of its
    layouts give the same legs.

    Parameters:
        net : The network.
        s : Its state, :class:`~discopy.neural.recursion.State` or
            :class:`~discopy.neural.recursion.PortState`.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.neural.recursion import PeerNet
    >>> net = PeerNet((4, 6, 8, 5), ((1, 2), (0, 2), (0, 1)), 3, 2, 3, 1)
    >>> given, memory = legs(net, net.initial(torch.ones(2, 3, 3)))
    >>> len(given), len(memory)
    (6, 9)
    """
    nodes = s.state.shape[1]
    given = tuple(leg for i in range(nodes)
                  for leg in (s.clue[:, i], s.answer[:, i]))
    if isinstance(net, PeerNet):
        rows = net.peers.tolist()  # ty: ignore[call-non-callable]
        inputs = [[s.units[:, peer] for peer in row] for row in rows]
        members = ()
    else:
        units = s.units[:, net.to_units].view(
            len(s.units), net.n_units, net.arity, -1) \
            if s.units.dim() == 3 else s.units
        degrees = [net.degree] * nodes if net.real_degree is None \
            else net.real_degree.long(  # ty: ignore[call-non-callable]
            ).tolist()
        inputs = [[s.cells[:, i, k] for k in range(degree)]
                  for i, degree in enumerate(degrees)]
        members = tuple(units[:, f, j] for f in range(net.n_units)
                        for j in range(net.arity))
    states = tuple(leg for i, row in enumerate(inputs)
                   for leg in (*row, s.state[:, i]))
    return given, states + members


def select(cmap: frobenius.CMap, index: int, row: tuple, at: dict
           ) -> list[int]:
    """
    The positions of the ports of a box that carry a leg: first those wired
    to another box, then the first of every self-wired pair, which is the
    port its state goes out of.

    Parameters:
        cmap : The closed map.
        index : The index of the box.
        row : The numbers of its ports, see
              :func:`~discopy.neural.wiring.ports`.
        at : The box and position of every port number, see
             :func:`~discopy.neural.wiring.where`.

    Raises:
        ValueError : If a port is wired to a port of its own box and is not
                     a state, i.e. a traced pair.

    Example
    -------
    >>> pair = Signature((Orbit(MESSAGE, 1), Orbit(STATE, traced=True)))
    >>> cmap = from_relation(((1, ), (0, )), pair)
    >>> rows = ports(cmap)
    >>> select(cmap, 0, rows[0], where(rows))
    [0, 1]
    """
    mates = [at[cmap.edges[port]] for port in row]
    loops = [position for position, (other, _) in enumerate(mates)
             if other == index]
    if any(cmap.boxes[index].cod[position] != STATE for position in loops):
        raise ValueError(f"box {index} has a port wired to its own box "
                         "that is not a state")
    return [position for position, (other, _) in enumerate(mates)
            if other != index] + [
        position for position in loops if position < mates[position][1]]


def reads(cmap: frobenius.CMap, rows: tuple, slots: list, dims: list
          ) -> list[int]:
    """
    The leg each leg of the diagram reads: for the leg of a slot, the leg
    written at the other end of its wire, or its own leg for a state.

    Parameters:
        cmap : The closed map.
        rows : The numbers of the ports of every box.
        slots : The box and position of every leg, see :func:`select`.
        dims : The legs of every slot.

    Example
    -------
    >>> pair = Signature((Orbit(MESSAGE, 1), Orbit(STATE, traced=True)))
    >>> cmap = from_relation(((1, ), (0, )), pair)
    >>> slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    >>> reads(cmap, ports(cmap), slots, [Dims(1)] * 4)
    [2, 1, 0, 3]
    """
    at = where(rows)
    number = {slot: n for n, slot in enumerate(slots)}
    starts = list(accumulate((len(dim) for dim in dims), initial=0))
    mates = [at[cmap.edges[rows[index][position]]]
             for index, position in slots]
    sources = [n if mate[0] == slot[0] else number[mate]
               for n, (slot, mate) in enumerate(zip(slots, mates))]
    return [starts[source] + leg for source, dim in zip(sources, dims)
            for leg in range(len(dim))]


def boxes(cmap: frobenius.CMap, ar_map: dict, slots: list, dims: list,
          given: list) -> tuple[list[Box], list[int]]:
    """
    The boxes of the diagram, one per box of the map, and the route that
    sends the legs they read to their inputs: first the boundary legs a box
    reads, then the legs of its slots.

    Parameters:
        cmap : The closed map.
        ar_map : The module of the boxes, for each name.
        slots : The box and position of every leg, see :func:`select`.
        dims : The legs of every slot.
        given : The legs every box reads from the boundary.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> pair = Signature((Orbit(MESSAGE, 1), Orbit(STATE, traced=True)))
    >>> cmap = from_relation(((1, ), (0, )), pair)
    >>> slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    >>> parts, route = boxes(cmap, {NODE: abs}, slots, [Dims(1)] * 4,
    ...                      [Dims(), Dims()])
    >>> len(parts), route
    (2, [0, 1, 2, 3])
    """
    offsets = list(accumulate((len(dim) for dim in given), initial=0))
    starts = list(accumulate((len(dim) for dim in dims), initial=0))
    parts, route = [], []
    for index, box in enumerate(cmap.boxes):
        own = [n for n, slot in enumerate(slots) if slot[0] == index]
        mine = Dims().tensor(*(dims[n] for n in own))
        parts.append(Box(box.name, given[index] @ mine, mine,
                         module=ar_map[box.name]))
        route += range(offsets[index], offsets[index + 1])
        route += (offsets[-1] + starts[n] + leg
                  for n in own for leg in range(len(dims[n])))
    return parts, route


def interpret(cmap: frobenius.CMap, ob_map: dict, ar_map: dict,
              inputs: dict | None = None) -> Network:
    """
    The network with a trace a closed map is sent to: one box per box of the
    map, with one leg in and one out per port wired to another box and per
    traced pair of ports, in the order of the ports, and one trace over all
    the legs, whose body routes every leg out to the leg in at the other end
    of its wire, and observes the legs it wrote.

    The box of a name is filled by the module ``ar_map[name]``, one object
    for every box of that name, after the legs ``inputs[name]`` it reads
    from the boundary of the network, e.g. the clue of a node.

    Parameters:
        cmap : A closed map, e.g. of :func:`~discopy.neural.signature.\
from_incidence`.
        ob_map : The legs of a port, for each role.
        ar_map : The module of the boxes, for each name.
        inputs : The legs a box reads from the boundary, for each name.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> pair = Signature((Orbit(MESSAGE, 1), Orbit(STATE, traced=True)))
    >>> swap = lambda m, s: (s + m, s + 1)
    >>> trace = interpret(from_relation(((1, ), (0, )), pair),
    ...                   {MESSAGE: Dims(1), STATE: Dims(1)}, {NODE: swap})
    >>> trace.dom, len(trace.mem)
    (Dims(), 4)
    """
    inputs = inputs or {}
    rows = ports(cmap)
    at = where(rows)
    slots = [(index, position) for index, row in enumerate(rows)
             for position in select(cmap, index, row, at)]
    dims = [ob_map[cmap.boxes[index].cod[position]]
            for index, position in slots]
    given = [inputs.get(box.name, Dims()) for box in cmap.boxes]
    parts, route = boxes(cmap, ar_map, slots, dims, given)
    mem = Dims().tensor(*dims)
    back = reads(cmap, rows, slots, dims)
    body = Function(Dims().tensor(*given) @ mem, route)\
        >> Network.id(Dims()).tensor(*parts)\
        >> Function(mem, back + back)
    return Trace(body, mem=mem)


def of(net: PeerNet | FactorNet) -> Network:
    """
    The network with a trace of a recursive network: the closed map of its
    wiring, with the node module and the factor module of the network as the
    modules of its boxes, under the names the network knows them by.

    Nothing is assigned to the network, and no random number is drawn. A
    :class:`~discopy.neural.recursion.PeerNet` must have a symmetric peers
    table, see :func:`closed`.

    Parameters:
        net : A :class:`~discopy.neural.recursion.PeerNet` or a
              :class:`~discopy.neural.recursion.FactorNet`.

    Raises:
        ValueError : If the peers table is not symmetric, or the factor
                     module of the network is named as the nodes are.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.neural.recursion import FactorNet
    >>> path = FactorNet((4, 6, 8, 5), ((0, 1), (1, 2)), 3, 2, 3, 1)
    >>> [box.name for box in of(path).arg.boxes if box.module is not None]
    ['cell', 'cell', 'cell', 'unit', 'unit']
    """
    widths = net.widths
    ob_map = {MESSAGE: Dims(widths.dim), STATE: Dims(widths.state_dim)}
    ar_map = {NODE: Fire(net.cell)}
    if isinstance(net, FactorNet):
        if net.factor == NODE:
            raise ValueError(f"the factor module is named {NODE!r}, "
                             "as the nodes are")
        ar_map[net.factor] = Reply(getattr(net, net.factor))
    return interpret(closed(net), ob_map, ar_map,
                     {NODE: Dims(widths.dim, widths.y_dim)})
