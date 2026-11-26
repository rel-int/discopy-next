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

The image has one box, and so one module call, per node and per factor:
:func:`stack` sends it to a round with a number of boxes that does not grow
with the graph.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Fire
    Reply
    Gather

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
        buckets
        pooled
        wires
        stacked
        pack
        on_nodes
        routes
        stack

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

from bisect import bisect_right
from itertools import accumulate

import torch
from torch import nn

from discopy import frobenius
from discopy.neural.cell import Site
from discopy.neural.network import Box, Dim, Dims, Function, Network, Trace
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
            clue : The clue of the node, ``(batch, dim)``, or of a stack of
                   nodes, ``(batch, nodes, dim)``.
            answer : The answer of the node, ``(batch, y_dim)``.
            messages_state : One message per port, each ``(batch, dim)``,
                             then the state, ``(batch, state_dim)``, with the
                             same node axis as the clue.

        Returns:
            One belief per port, the same one, then the new state.
        """
        # pylint: disable=arguments-differ
        *messages, state = messages_state
        rows, degree, dim = len(clue), len(messages), clue.shape[-1]
        joined = torch.stack(messages, -2).reshape(rows, -1, degree, dim)
        belief, state = Site.forward(
            self.site, joined, state.reshape(rows, -1, state.shape[-1]),
            clue.reshape(rows, -1, dim),
            answer.reshape(rows, -1, answer.shape[-1]))
        shape = clue.shape[:-1]
        return (belief.reshape(*shape, -1), ) * degree\
            + (state.reshape(*shape, -1), )


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
            messages : One belief per member, each ``(batch, dim)``, or of a
                       stack of factors, ``(batch, factors, dim)``.

        Returns:
            One reply per member, each of the shape of a belief.
        """
        joined = torch.stack(messages, -2)
        rows, (arity, dim) = len(joined), joined.shape[-2:]
        replies = self.unit(joined.reshape(rows, -1, arity, dim))
        return tuple(replies.reshape(joined.shape).unbind(-2))


class Gather(nn.Module):
    """
    The wiring between two layers of stacked boxes as one module: every
    output leg is read off the node axis of the input legs, which it
    concatenates, by an index. A leg of one node has no node axis.

    Parameters:
        wired : For each output leg, the input legs it reads, increasing, and
                the position of each of its nodes on their concatenation.
        sizes : The nodes of each input leg.

    Attributes:
        index : The positions of all the output legs, concatenated.
        offsets : Where the index of each output leg starts and ends.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> a, b = torch.tensor([[[1.], [2.]]]), torch.tensor([[3.]])
    >>> gather = Gather((((0, 1), (2, 1)), ), (2, 1))
    >>> gather(a, b)[0].tolist()
    [[[3.0], [2.0]]]
    >>> gather
    Gather((((0, 1), (2, 1)),), (2, 1))
    """
    index: torch.Tensor

    def __init__(self, wired: tuple, sizes: tuple[int, ...]):
        super().__init__()
        self.wired, self.sizes = tuple(wired), tuple(sizes)
        self.sources = tuple(sources for sources, _ in self.wired)
        self.whole = tuple(
            len(sources) == 1 and index == tuple(range(sizes[sources[0]]))
            for sources, index in self.wired)
        self.offsets = tuple(accumulate(
            (len(index) for _, index in self.wired), initial=0))
        self.register_buffer("index", torch.tensor(
            [place for _, index in self.wired for place in index],
            dtype=torch.long), persistent=False)

    def __repr__(self):
        return f"Gather({self.wired!r}, {self.sizes!r})"

    def forward(self, *tensors):
        """
        A leg that is read whole, as it is, is returned itself, not a copy.

        Parameters:
            tensors : One tensor per input leg, ``(batch, nodes, width)``, or
                      ``(batch, width)`` when it has one node.

        Returns:
            One tensor per output leg, of the same shape convention.
        """
        # pylint: disable=arguments-differ
        inputs = [tensor.reshape(len(tensor), size, -1)
                  for tensor, size in zip(tensors, self.sizes)]
        pools, outputs = {}, []
        for n, (sources, whole) in enumerate(zip(self.sources, self.whole)):
            if whole:
                outputs.append(tensors[sources[0]])
                continue
            if sources not in pools:
                pools[sources] = torch.cat([inputs[s] for s in sources], 1)
            read = pools[sources][:, self.index[self.offsets[n]:
                                                self.offsets[n + 1]]]
            outputs.append(read[:, 0] if read.shape[1] == 1 else read)
        return tuple(outputs)


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


def routes(layer: list[Box]) -> list[int]:
    """
    The leg each input of a layer of boxes reads, in the domain the legs they
    read from the boundary then those of the memory, in the order of their
    outputs.

    Parameters:
        layer : The boxes, each reading the legs it writes after those it
                reads from the boundary.

    Example
    -------
    >>> a = Box("a", Dims(2, 2), Dims(3))
    >>> b = Box("b", Dims(4, 4, 4), Dims(5, 6))
    >>> routes([a, b])
    [0, 2, 1, 3, 4]
    """
    given = list(accumulate(
        (len(box.dom) - len(box.cod) for box in layer), initial=0))
    own = list(accumulate((len(box.cod) for box in layer), initial=0))
    return [leg for n in range(len(layer))
            for leg in (*range(given[n], given[n + 1]),
                        *range(given[-1] + own[n], given[-1] + own[n + 1]))]


def boxes(cmap: frobenius.CMap, ar_map: dict, slots: list, dims: list,
          given: list) -> list[Box]:
    """
    The boxes of the diagram, one per box of the map, each reading the legs
    it is given and writing those of its slots.

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
    >>> len(boxes(cmap, {NODE: abs}, slots, [Dims(1)] * 4,
    ...           [Dims(), Dims()]))
    2
    """
    parts = []
    for index, box in enumerate(cmap.boxes):
        mine = Dims().tensor(*(dim for dim, slot in zip(dims, slots)
                               if slot[0] == index))
        parts.append(Box(box.name, given[index] @ mine, mine,
                         module=ar_map[box.name]))
    return parts


def interpret(cmap: frobenius.CMap, ob_map: dict, ar_map: dict,
              inputs: dict | None = None) -> Network:
    """
    The network with a trace a closed map is sent to: one box per box of the
    map, with one leg in and one out per port wired to another box and per
    traced pair of ports, and one trace over all the legs, whose body routes
    every leg out to the leg in at the other end of its wire, and observes
    the legs it wrote.

    The box of a name is filled by the module ``ar_map[name]``, after the
    legs ``inputs[name]`` it reads from the boundary, e.g. the clue of a node.

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
    parts = boxes(cmap, ar_map, slots, dims, given)
    mem = Dims().tensor(*dims)
    back = reads(cmap, rows, slots, dims)
    body = Function(Dims().tensor(*given) @ mem, routes(parts))\
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


def buckets(trace: Network) -> tuple[list[Box], list[list[int]]]:
    """
    The boxes of the round of a trace, as :func:`interpret` draws it, and the
    indices of those that share a name, a module and the legs they read and
    write, i.e. one degree of a node or a factor, in order of appearance: the
    buckets of the nodes, then those of the factors, as :func:`interpret`
    draws their boxes, so that the layout is that of the network.

    Parameters:
        trace : A trace drawn by :func:`interpret` or :func:`of`.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.neural.recursion import FactorNet
    >>> path = FactorNet((4, 6, 8, 5), ((0, 1), (1, 2)), 3, 2, 3, 1)
    >>> buckets(of(path))[1]
    [[0, 2], [1], [3, 4]]
    """
    parts = [box for box in trace.arg.boxes if box.module is not None]
    found: dict = {}
    for index, box in enumerate(parts):
        found.setdefault(
            (box.name, box.dom, box.cod, id(box.module)), []).append(index)
    return parts, list(found.values())


def pooled(sizes: tuple[int, ...], found: list[tuple[int, int]]) -> tuple:
    """
    How one stacked leg reads the legs of the layer before: the legs it
    reads, increasing, and the position of each of its nodes on their
    concatenation along the node axis.

    Parameters:
        sizes : The nodes of each leg.
        found : For each node, the leg it reads and its place on that leg.

    Example
    -------
    >>> pooled((2, 1), [(1, 0), (0, 1)])
    ((0, 1), (2, 1))
    """
    sources = tuple(sorted({leg for leg, _ in found}))
    starts = dict(zip(sources, accumulate(
        (sizes[leg] for leg in sources), initial=0)))
    return sources, tuple(starts[leg] + place for leg, place in found)


def wires(trace: Network) -> tuple[tuple, tuple[int, ...]]:
    """
    The wiring of the stacked round of a trace: ``(reads, sizes)``, the
    arguments of :class:`Gather`. The ``k`` legs of one role of a bucket are
    one leg with ``k`` nodes, and every one of its nodes reads the leg its
    box reads in the round of the trace, wherever that leg went.

    Parameters:
        trace : A trace drawn by :func:`interpret` or :func:`of`.

    Raises:
        ValueError : If the trace is not one drawn by :func:`interpret`, whose
                     body ends with the routing of the legs back.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.neural.recursion import FactorNet
    >>> path = FactorNet((4, 6, 8, 5), ((0, 1), (1, 2)), 3, 2, 3, 1)
    >>> wired, sizes = wires(of(path))
    >>> len(wired), sizes
    (7, (2, 2, 1, 1, 1, 2, 2))
    >>> wired[1]
    ((1,), (0, 1))
    """
    last = trace.arg.boxes[-1] if trace.arg.boxes else None
    if not isinstance(last, Function) or len(last.func) != 2 * len(trace.mem):
        raise ValueError("expected a trace drawn by interpret or of, "
                         "whose body ends with the routing of its legs")
    parts, groups = buckets(trace)
    back = list(last.func)[:len(trace.mem)]
    own = list(accumulate((len(box.cod) for box in parts), initial=0))
    firsts = list(accumulate(
        (len(parts[group[0]].cod) for group in groups), initial=0))
    at = {box: (j, place) for j, group in enumerate(groups)
          for place, box in enumerate(group)}
    sizes = tuple(len(group) for group in groups
                  for _ in parts[group[0]].cod.inside)

    def origin(box, leg):
        source = back[own[box] + leg]
        owner = bisect_right(own, source) - 1
        j, place = at[owner]
        return firsts[j] + source - own[owner], place

    wired = tuple(
        pooled(sizes, [origin(box, leg) for box in group])
        for group in groups for leg in range(len(parts[group[0]].cod)))
    return wired, sizes


def stacked(tensors: tuple, widths: list[int], groups: list[list[int]]
            ) -> tuple:
    """
    The legs of a tensor per leg of each box, one per leg of each bucket:
    the ``k`` tensors of a role stacked along a new axis after the batch,
    except that the one tensor of a bucket of one box is left as it is.

    Parameters:
        tensors : One tensor per leg of every box.
        widths : The legs of each box.
        groups : The boxes of each bucket, see :func:`buckets`.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> ones = tuple(torch.ones(5, 3) for _ in range(4))
    >>> [tuple(t.shape) for t in stacked(ones, [2, 2], [[0, 1]])]
    [(5, 2, 3), (5, 2, 3)]
    >>> [tuple(t.shape) for t in stacked(ones, [2, 2], [[0], [1]])]
    [(5, 3), (5, 3), (5, 3), (5, 3)]
    """
    starts = list(accumulate(widths, initial=0))
    return tuple(
        torch.stack([tensors[starts[box] + leg] for box in group], 1)
        if len(group) > 1 else tensors[starts[group[0]] + leg]
        for group in groups for leg in range(widths[group[0]]))


def pack(trace: Network, given: tuple, memory: tuple) -> tuple[tuple, tuple]:
    """
    The legs of a trace as the legs of its :func:`stack`: for each bucket the
    legs it reads from the boundary, then those of the memory, each the
    legs of its boxes stacked along an axis after the batch.

    Parameters:
        trace : A trace drawn by :func:`interpret` or :func:`of`.
        given : One tensor per leg of its domain, e.g. from :func:`legs`.
        memory : One tensor per leg of its memory.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.neural.recursion import FactorNet
    >>> path = FactorNet((4, 6, 8, 5), ((0, 1), (1, 2)), 3, 2, 3, 1)
    >>> given, memory = pack(of(path), *legs(
    ...     path, path.initial(torch.ones(2, 3, 3))))
    >>> [tuple(leg.shape) for leg in memory]
    [(2, 2, 4), (2, 2, 6), (2, 4), (2, 4), (2, 6), (2, 2, 4), (2, 2, 4)]
    """
    parts, groups = buckets(trace)
    return (stacked(given, [len(box.dom) - len(box.cod) for box in parts],
                    groups),
            stacked(memory, [len(box.cod) for box in parts], groups))


def on_nodes(box: Box, k: int) -> Box:
    """
    The box on ``k`` nodes: the module of ``box``, one object, on legs with a
    node axis in front of their shape. One node has no node axis, since a
    dimension of one is dropped from a shape: the box is its own.

    Parameters:
        box : A box whose module is the same on every node.
        k : The number of nodes.

    Example
    -------
    >>> on_nodes(Box("f", Dims(2), Dims(3), module=abs), 4)
    neural.network.Box('f', Dims(Dim(4, 2)), Dims(Dim(4, 3)))
    >>> on_nodes(Box("f", Dims(2), Dims(3), module=abs), 1).dom
    Dims(Dim(2))
    """
    if k == 1:
        return box
    return Box(box.name,
               Dims(*(Dim(k) @ leg for leg in box.dom.inside)),
               Dims(*(Dim(k) @ leg for leg in box.cod.inside)),
               module=box.module)


def stack(trace: Network) -> Network:
    """
    The trace with the boxes of each bucket stacked: the functor that sends
    a layer of ``k`` boxes ``f`` of one module and one shape, ``f ** k``, to
    one box on legs of ``k`` nodes, the module called once on them, and the
    wiring between the layers, a permutation or a copy of legs, to an index
    gather on the node axis, :class:`Gather`.

    The ``k`` legs of a role are one leg ``Dim(k, width)``, see :func:`pack`
    for the tensors and :func:`buckets` for the layout.

    The module of a box must run on legs with node axes between the batch
    and the width, as :class:`Fire` and :class:`Reply` do. A bucket of one
    node has the leg ``Dim(width)`` with no node axis, since a dimension of
    one is dropped: its tensors are not stacked, and its type is its shape.

    Parameters:
        trace : A trace drawn by :func:`interpret` or :func:`of`.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.neural.recursion import FactorNet
    >>> grid = FactorNet(
    ...     (4, 6, 8, 5), ((0, 1), (2, 3), (0, 2), (1, 3)), 3, 2, 3, 1)
    >>> names = lambda t: [b.name for b in t.arg.boxes if b.module]
    >>> len(names(of(grid))), names(stack(of(grid)))
    (8, ['cell', 'unit', 'gather'])
    >>> len(of(grid).mem), len(stack(of(grid)).mem)
    (20, 5)
    """
    parts, groups = buckets(trace)
    layer = [on_nodes(parts[group[0]], len(group)) for group in groups]
    given = Dims().tensor(*(box.dom[:len(box.dom) - len(box.cod)]
                            for box in layer))
    mem = Dims().tensor(*(box.cod for box in layer))
    gather = Box("gather", mem, mem, module=Gather(*wires(trace)))
    body = Function(given @ mem, routes(layer))\
        >> Network.id(Dims()).tensor(*layer) >> gather\
        >> Function(mem, list(range(len(mem))) * 2)
    return Trace(body, mem=mem)
