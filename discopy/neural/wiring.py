# -*- coding: utf-8 -*-

"""
The tables a recursive network gathers along, read off a factor graph.

A *factor table* lists, for each factor, the nodes it holds, increasing: the
nine rows, nine columns and nine boxes of a sudoku, the lines of an n-queens
board. It is a hypergraph on the nodes, and everything a network needs to
route messages along it is a table of indices:

* :func:`primal` is its primal graph, the nodes each node shares a factor
  with, which a network wired straight from node to node gathers along;
* :func:`routing` is its incidence graph, a port per membership, flattened
  into the two gathers that take node beliefs onto factor ports and factor
  replies back onto node ports, padded where the degrees differ;
* :func:`from_map` reads a factor table back out of the closed map that
  :func:`~discopy.neural.signature.from_incidence` or
  :func:`~discopy.neural.signature.from_relation` draws, so a wiring drawn
  with DisCoPy compiles to the same tables;
* :func:`decompose` and :func:`families` read a wiring invariant under a
  group of symmetries as signed sums of :class:`Block`, the partitions the
  group's orbitals are made of, which :mod:`discopy.neural.geometric` sums
  over by reductions rather than gathers.

The tables are tuples of integers, so this module imports no tensor
framework: a network turns them into index tensors once, when it is built.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Routing
    Block

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        table_of
        memberships
        primal
        routing
        from_map
        chains
        unravel
        depths
        orbitals
        decompose
        families

Example
-------
>>> factors = ((0, 1), (1, 2))
>>> primal(factors)
((1,), (0, 2), (1,))
>>> routing(factors).to_nodes
(0, 0, 1, 2, 3, 3)
"""

from __future__ import annotations

import itertools
import math
from typing import NamedTuple

from discopy import group as groups
from discopy.frobenius import CMap
from discopy.neural.signature import NODE
from discopy.utils import factory_name


def table_of(factors) -> tuple[tuple[int, ...], ...]:
    """
    A factor table as a tuple of tuples of integers, checked: every factor
    increasing, with no node twice.

    Parameters:
        factors : For each factor, the nodes it holds, e.g. a list of lists
                  or an integer tensor of shape ``(n_factors, arity)``.

    Raises:
        ValueError : If a factor is not sorted or holds a node twice.

    Example
    -------
    >>> table_of([[0, 1], [1, 2]])
    ((0, 1), (1, 2))
    >>> table_of([[1, 0]])
    Traceback (most recent call last):
        ...
    ValueError: factor 0 is not sorted and distinct: (1, 0)
    """
    table = tuple(tuple(int(node) for node in factor) for factor in factors)
    for index, factor in enumerate(table):
        if list(factor) != sorted(set(factor)):
            raise ValueError(
                f"factor {index} is not sorted and distinct: {factor}")
    return table


def memberships(factors, n_nodes: int | None = None
                ) -> tuple[tuple[int, ...], ...]:
    """
    For each node, the factors it belongs to, increasing.

    Parameters:
        factors : The factor table, see :func:`table_of`.
        n_nodes : The number of nodes, by default one more than the largest
                  node any factor holds.

    Raises:
        ValueError : If a node belongs to no factor, or a factor holds a
                     node that does not exist.

    Example
    -------
    >>> memberships(((0, 1), (1, 2)))
    ((0,), (0, 1), (1,))
    >>> memberships(((0, 1), ), 3)
    Traceback (most recent call last):
        ...
    ValueError: node 2 belongs to no factor
    """
    table = table_of(factors)
    if n_nodes is None:
        n_nodes = 1 + max((max(factor, default=-1) for factor in table),
                          default=-1)
    found = [[] for _ in range(n_nodes)]
    for index, factor in enumerate(table):
        for node in factor:
            if not 0 <= node < n_nodes:
                raise ValueError(f"factor {index} holds node {node}, "
                                 f"which does not exist")
            found[node].append(index)
    for node, of in enumerate(found):
        if not of:
            raise ValueError(f"node {node} belongs to no factor")
    return tuple(map(tuple, found))


def primal(factors, n_nodes: int | None = None
           ) -> tuple[tuple[int, ...], ...]:
    """
    The primal graph of a factor table: for every node, the nodes it shares
    a factor with, increasing, itself excluded.

    This is the peer table of a network wired straight from node to node,
    e.g. the 20 peers of a sudoku cell from its row, column and box.

    Parameters:
        factors : The factor table, see :func:`table_of`.
        n_nodes : The number of nodes, see :func:`memberships`.

    Example
    -------
    >>> rows, columns = ((0, 1), (2, 3)), ((0, 2), (1, 3))
    >>> primal(rows + columns)
    ((1, 2), (0, 3), (0, 3), (1, 2))
    """
    table = table_of(factors)
    return tuple(
        tuple(sorted({other for index in of for other in table[index]}
                     - {node}))
        for node, of in enumerate(memberships(table, n_nodes)))


class Routing(NamedTuple):
    """
    The two gathers of a factor graph, flat, and what a masked pool needs.

    A factor of arity ``a`` has ``a`` ports, one per member, in increasing
    node order; the factor ports are numbered factor-major, so the port of
    the ``j``-th member of factor ``i`` is ``i * arity + j``. A node has one
    port per factor it belongs to, in increasing factor order; a node with
    fewer factors than the largest :attr:`degree` repeats its last factor on
    the remaining ports, which :attr:`valid` masks.

    Parameters:
        to_factors : The node behind each factor port, ``n_factors * arity``.
        to_nodes : The factor port behind each node port, ``n_nodes *
                   degree``.
        degree : The ports of a node, the largest number of factors any node
                 belongs to.
        arity : The ports of a factor, the same for every factor.
        valid : ``None`` when every node has :attr:`degree` factors, else for
                each node a ``1.0`` per real port then a ``0.0`` per padded
                one.
        degrees : The number of factors each node belongs to.
    """
    to_factors: tuple[int, ...]
    to_nodes: tuple[int, ...]
    degree: int
    arity: int
    valid: tuple[tuple[float, ...], ...] | None
    degrees: tuple[int, ...]


def routing(factors, n_nodes: int | None = None) -> Routing:
    """
    The :class:`Routing` of a factor table.

    Parameters:
        factors : The factor table, see :func:`table_of`, every factor of
                  the same arity.
        n_nodes : The number of nodes, see :func:`memberships`.

    Raises:
        ValueError : If the factors do not all have the same arity, a factor
                     is not sorted and distinct, or a node belongs to no
                     factor.

    Example
    -------
    A path of three nodes, the middle one in both factors:

    >>> path = routing(((0, 1), (1, 2)))
    >>> path.to_factors, path.to_nodes
    ((0, 1, 1, 2), (0, 0, 1, 2, 3, 3))
    >>> path.degree, path.arity, path.degrees
    (2, 2, (1, 2, 1))
    >>> path.valid
    ((1.0, 0.0), (1.0, 1.0), (1.0, 0.0))
    >>> routing(((0, 1), (1, 2, 3)))
    Traceback (most recent call last):
        ...
    ValueError: factors of unequal arity: [2, 3]
    """
    table = table_of(factors)
    arities = sorted({len(factor) for factor in table})
    if len(arities) != 1:
        raise ValueError(f"factors of unequal arity: {arities}")
    arity = arities[0]
    found = memberships(table, n_nodes)
    degrees = tuple(map(len, found))
    degree = max(degrees)
    slots = {(node, index): index * arity + slot
             for index, factor in enumerate(table)
             for slot, node in enumerate(factor)}
    to_nodes = tuple(
        slots[node, index] for node, of in enumerate(found)
        for index in of + (of[-1], ) * (degree - len(of)))
    valid = None if min(degrees) == degree else tuple(
        (1.0, ) * real + (0.0, ) * (degree - real) for real in degrees)
    return Routing(tuple(node for factor in table for node in factor),
                   to_nodes, degree, arity, valid, degrees)


def from_map(cmap: CMap, node_name: str = NODE
             ) -> tuple[tuple[int, ...], ...]:
    """
    The table a closed map wires: for every box that is not a node, the
    nodes at its ports, in port order -- the factor table of a map drawn by
    :func:`~discopy.neural.signature.from_incidence`; when every box is a
    node, for every node the nodes at its ports -- the peer table of a map
    drawn by :func:`~discopy.neural.signature.from_relation`.

    A port wired back to its own box is a traced loop, not a membership, and
    is skipped. Nodes are numbered in the order their boxes appear.

    Parameters:
        cmap : A closed map, every box of it with an empty domain.
        node_name : The name of the node boxes.

    Raises:
        ValueError : If the map has a boundary, or a port joins two boxes
                     neither of which is a node.

    Example
    -------
    The sudoku of side four: every cell in its row, its column and its box.

    >>> from discopy.frobenius import Ty
    >>> from discopy.neural.signature import (
    ...     Orbit, Signature, Sym, from_incidence, from_relation)
    >>> message, state = Ty("message"), Ty("state")
    >>> cell = Signature((Orbit(message, 3, Sym.PERM),
    ...                   Orbit(state, traced=True)))
    >>> unit = Signature((Orbit(message, 4, Sym.PERM), ))
    >>> incidence = tuple(
    ...     (i // 4, 4 + i % 4, 8 + 2 * (i // 8) + i % 4 // 2)
    ...     for i in range(16))
    >>> sudoku = from_incidence(incidence, cell, unit)
    >>> from_map(sudoku)[:5]
    ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15), \
(0, 4, 8, 12))
    >>> units = tuple(
    ...     tuple(i for i in range(16) if index in incidence[i])
    ...     for index in range(12))
    >>> assert routing(from_map(sudoku)) == routing(units)

    The peer table comes back out of the primal graph's own map:

    >>> peer = Signature((Orbit(message, 7, Sym.PERM),
    ...                   Orbit(state, traced=True)))
    >>> clique = from_relation(primal(units), peer)
    >>> assert from_map(clique) == primal(units) == primal(from_map(sudoku))
    """
    if cmap.dom or cmap.cod:
        raise ValueError("the map is not closed")
    ports, box_at, start = [], {}, 0
    for index, box in enumerate(cmap.boxes):
        offsets = CMap.logical_order(len(box.dom), len(box.cod))
        ports.append(tuple(start + offset for offset in offsets))
        box_at.update((start + offset, index) for offset in offsets)
        start += len(offsets)
    nodes = {index: number for number, index in enumerate(
        i for i, box in enumerate(cmap.boxes) if box.name == node_name)}
    factors = [i for i in range(len(cmap.boxes)) if i not in nodes] \
        or list(nodes)
    table = []
    for index in factors:
        others = [box_at[cmap.edges[port]] for port in ports[index]]
        others = [other for other in others if other != index]
        if not all(other in nodes for other in others):
            raise ValueError(
                f"box {index} is wired to a box that is not a node")
        table.append(tuple(nodes[other] for other in others))
    return tuple(table)


def chains(group) -> tuple[tuple[int, ...], ...]:
    """
    The axes of the natural action of a group, grouped into chains, outer
    factor first: ``(n, )`` for a symmetric group, the top's axis followed
    by the base's chain for an imprimitive wreath product and the chains of
    both factors for a direct product.

    A chain of wreath products of symmetric groups acts on its points
    numbered row-major, outer factor first, and a product of chains on the
    tensor of their axes: two points of a chain *agree to depth* ``d`` when
    their first ``d`` coordinates along it are equal, and the orbitals of
    the group, its orbits on pairs of points, are the tuples of depths, one
    per chain, see :func:`orbitals`.

    Parameters:
        group : A group built from symmetric groups, and cyclic groups of
                order at most two, by :class:`~discopy.group.Product` and
                :class:`~discopy.group.Wreath`.

    Only the imprimitive action of a wreath product is read, on the ``k``
    blocks of ``n`` points, not its power action on ``n ** k`` tuples.

    Raises:
        TypeError : If a factor has more than two orbitals on pairs of its
                    points, e.g. a cyclic group of order three, or a wreath
                    product holds a direct product.

    Example
    -------
    >>> from discopy.group import Product, Symmetric, Wreath
    >>> lines = Wreath[Symmetric[3], Symmetric[3]]
    >>> chains(Product[lines, lines])
    ((3, 3), (3, 3))
    """
    if issubclass(group, groups.Product):
        return chains(group.left) + chains(group.right)
    if issubclass(group, groups.Wreath):
        base, top = chains(group.base), chains(group.top)
        if len(base) != 1 or len(top) != 1:
            raise TypeError(f"{factory_name(group)} has a product inside a "
                            "wreath: its orbitals are not agreements")
        return (top[0] + base[0], )
    if issubclass(group, groups.Symmetric) or (
            issubclass(group, groups.Cyclic) and group.n <= 2):
        return ((group.n, ), )
    raise TypeError(f"{factory_name(group)} has more than two orbitals on "
                    "pairs: its orbitals are not agreements")


class Block(NamedTuple):
    """
    The partition of the points of a tensor of shape ``shape`` by their
    coordinates along the axes ``keep``: one block per value of the kept
    coordinates, numbered row-major, each running over the other axes.

    Parameters:
        shape : The shape of the points.
        keep : The axes the points of a block agree on, increasing.

    Example
    -------
    The rows, the columns and the boxes of a four by four sudoku, its cells
    laid out as ``(band, row, stack, column)``:

    >>> rows, boxes = Block((2, 2, 2, 2), (0, 1)), Block((2, 2, 2, 2), (0, 2))
    >>> rows.members()[1], boxes.members()[1]
    ((4, 5, 6, 7), (2, 3, 6, 7))
    >>> rows.count, rows.size, rows.free
    (4, 4, (2, 3))
    """
    shape: tuple[int, ...]
    keep: tuple[int, ...]

    @property
    def free(self) -> tuple[int, ...]:
        """ The axes a block runs over. """
        return tuple(a for a in range(len(self.shape)) if a not in self.keep)

    @property
    def order(self) -> tuple[int, ...]:
        """ The kept axes followed by the free ones. """
        return self.keep + self.free

    @property
    def size(self) -> int:
        """ The number of points in each block. """
        return math.prod(self.shape[a] for a in self.free)

    @property
    def count(self) -> int:
        """ The number of blocks. """
        return math.prod(self.shape[a] for a in self.keep)

    def members(self) -> tuple[tuple[int, ...], ...]:
        """ Every block, as the increasing tuple of its points. """
        found: dict = {}
        for point in range(math.prod(self.shape)):
            coordinates = unravel(point, self.shape)
            found.setdefault(tuple(coordinates[a] for a in self.keep),
                             []).append(point)
        return tuple(tuple(found[key]) for key in sorted(found))

    @classmethod
    def of(cls, layout, depth: tuple[int, ...]) -> Block:
        """
        The block of the points agreeing to ``depth`` or deeper along the
        chains of ``layout``.

        Parameters:
            layout : The chains, see :func:`chains`.
            depth : One depth per chain.

        Example
        -------
        >>> Block.of(((3, 3), (3, 3)), (1, 1))
        Block(shape=(3, 3, 3, 3), keep=(0, 2))
        """
        keep, start = [], 0
        for chain, agreed in zip(layout, depth):
            keep += range(start, start + agreed)
            start += len(chain)
        return cls(sum(layout, ()), tuple(keep))


def unravel(point: int, shape) -> tuple[int, ...]:
    """
    The coordinates of a point numbered row-major in ``shape``.

    Parameters:
        point : The number of the point.
        shape : The shape of the points.

    Example
    -------
    >>> unravel(5, (2, 3))
    (1, 2)
    """
    coordinates = []
    for n in reversed(shape):
        point, x = divmod(point, n)
        coordinates.append(x)
    return tuple(reversed(coordinates))


def depths(v: int, w: int, layout) -> tuple[int, ...]:
    """
    How deep two points agree along each chain, i.e. the orbital of the
    pair.

    Parameters:
        v : A point.
        w : Another point.
        layout : The chains, see :func:`chains`.

    Example
    -------
    Two cells of a sudoku in the same row and the same box:

    >>> depths(0, 1, ((3, 3), (3, 3)))
    (2, 1)
    """
    shape = sum(layout, ())
    first, second = unravel(v, shape), unravel(w, shape)
    result, start = [], 0
    for chain in layout:
        agreed = 0
        while agreed < len(chain) \
                and first[start + agreed] == second[start + agreed]:
            agreed += 1
        result.append(agreed)
        start += len(chain)
    return tuple(result)


def orbitals(layout) -> tuple[tuple[int, ...], ...]:
    """
    The orbitals of the group, one depth per chain.

    Parameters:
        layout : The chains, see :func:`chains`.

    Example
    -------
    >>> len(orbitals(((3, 3), (3, 3))))
    9
    """
    return tuple(itertools.product(*(range(len(c) + 1) for c in layout)))


def decompose(relation, layout) -> tuple[tuple[int, Block], ...]:
    """
    A relation invariant under the group as a signed sum of blocks:
    ``(coefficient, block)`` pairs such that ``w`` is related to ``v`` iff
    the coefficients of the blocks of ``v`` that hold ``w`` add up to one.

    The relation is a union of orbitals, and the orbital of the pairs that
    agree to depths *exactly* ``d`` is, by Möbius inversion along each
    chain, the alternating sum of the blocks agreeing to ``d`` plus zero or
    one more axis along each chain.

    Parameters:
        relation : Per point, the points it is related to.
        layout : The chains of a group the relation is invariant under.

    Raises:
        ValueError : If the relation is not on the points of the layout,
                     relates a point to one twice, or splits an orbital.

    Example
    -------
    The peers of a sudoku cell are its row, its column and its box, less
    the three cells of its row in its box, the three of its column and
    itself:

    >>> from discopy.neural.wiring import primal
    >>> sudoku = ((3, 3), (3, 3))
    >>> units = [b.members() for b in (Block.of(sudoku, (2, 0)),
    ...     Block.of(sudoku, (0, 2)), Block.of(sudoku, (1, 1)))]
    >>> peers = primal(sum(units, ()))
    >>> for coefficient, block in decompose(peers, sudoku):
    ...     print(coefficient, block.keep)
    1 (2, 3)
    1 (0, 2)
    -1 (0, 2, 3)
    1 (0, 1)
    -1 (0, 1, 2)
    -1 (0, 1, 2, 3)
    """
    size = math.prod(sum(layout, ()))
    if len(relation) != size:
        raise ValueError(
            f"{len(relation)} points for a layout of {size} points")
    inside: dict = {}
    for v, related in enumerate(relation):
        if len(set(related)) != len(related) \
                or not all(0 <= w < size for w in related):
            raise ValueError(f"point {v} is related to {tuple(related)}, "
                             f"not to distinct points of {size}")
        related = set(related)
        for w in range(len(relation)):
            orbital = depths(v, w, layout)
            if inside.setdefault(orbital, w in related) != (w in related):
                raise ValueError(f"the relation splits the orbital {orbital}")
    coefficients: dict = {}
    for orbital in (o for o, held in inside.items() if held):
        for bump in itertools.product((0, 1), repeat=len(layout)):
            deeper = tuple(d + b for d, b in zip(orbital, bump))
            if all(d <= len(c) for d, c in zip(deeper, layout)):
                coefficients[deeper] = coefficients.get(deeper, 0)\
                    + (-1) ** sum(bump)
    return tuple((c, Block.of(layout, d))
                 for d, c in sorted(coefficients.items()) if c)


def families(factors, layout) -> tuple[Block, ...]:
    """
    A factor table as families of blocks, in the order the table lists
    them: every factor is a block, and the factors of a family come
    together, every block of it in order.

    Parameters:
        factors : The factor table.
        layout : The chains, see :func:`chains`.

    Raises:
        ValueError : If a factor holds a point off the layout, or does not
                     start a family of blocks.

    Example
    -------
    >>> sudoku = ((3, 3), (3, 3))
    >>> units = [b.members() for b in (Block.of(sudoku, (2, 0)),
    ...     Block.of(sudoku, (0, 2)), Block.of(sudoku, (1, 1)))]
    >>> [b.keep for b in families(sum(units, ()), sudoku)]
    [(0, 1), (2, 3), (0, 2)]
    """
    found, cursor, table = [], 0, table_of(factors)
    size = math.prod(sum(layout, ()))
    if any(point >= size for factor in table for point in factor):
        raise ValueError(f"a factor holds a point off the {size} points")
    while cursor < len(table):
        first = table[cursor]
        if len(first) < 2:
            raise ValueError(f"factor {cursor} has fewer than two members")
        family = Block.of(layout, tuple(min(agreed) for agreed in zip(*(
            depths(first[0], w, layout) for w in first[1:]))))
        members = family.members()
        if table[cursor:cursor + len(members)] != members:
            raise ValueError(
                f"factor {cursor} does not start a family of blocks")
        found.append(family)
        cursor += len(members)
    return tuple(found)
