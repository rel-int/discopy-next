# -*- coding: utf-8 -*-

"""
What a generator promises: its legs, grouped into orbits, and the symmetry
each orbit carries.

A functor into :mod:`discopy.neural` preserves swaps, copies, discards and
traces strictly and for free, because they are wiring: a permutation of
tensors. What it cannot preserve for free is a box whose legs carry a
symmetry -- a spider, the members of a constraint unit over nine cells.
Those stay boxes, and their equations hold **iff the module satisfies
them**. A :class:`Signature` is where that promise is written down: it says
how many legs a box has, which of them are one orbit under a group, and
which are traced; whether the module keeps it is measured, not assumed.

A signature is the single source of truth for the *port layout of one
generator*, which would otherwise be agreed on by hand:

* :attr:`Signature.cod` is the type of the abstract box, so that
  :func:`from_incidence` and :func:`from_relation` draw a wiring out of a
  family's combinatorics alone;
* :meth:`Signature.positions` and :meth:`Signature.loops` say which ports
  carry a role and which pairs are traced, so that a module filling the box
  reads its legs off the signature rather than off a count of its own;
* :meth:`Signature.generators` gives the group an equivariance check runs a
  module against -- the check itself is the notebooks' business.

The boxes and the maps live in :mod:`discopy.frobenius`: every box a builder
wires has an empty domain, so every wire joins two codomain ports, which a
map accepts only between self-dual roles. Interpreting such a map as a
:class:`~discopy.neural.network.Network` is the business of a later level,
through the :class:`~discopy.interaction.Int`-construction over networks,
where a role read and written on one port is a pair of one shape.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Sym
    Orbit
    Signature

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        leg_generators
        from_incidence
        from_relation

Example
-------

>>> from discopy.frobenius import Ty
>>> message, state, given = Ty("message"), Ty("state"), Ty("given")
>>> cell = Signature((
...     Orbit(message, 3, Sym.PERM), Orbit(state, traced=True),
...     Orbit(given, traced=True)))
>>> print(cell.cod)
message @ message @ message @ state @ state @ given @ given
>>> cell.positions(state)
(3, 4)
>>> cell.loops()
((3, 4), (5, 6))
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

from discopy import frobenius
from discopy.python.finset import Permutation
from discopy.utils import assert_isatomic

NODE = "cell"
""" The name the node boxes of a generated wiring get by default. """

RELATION = "unit"
""" The name the relation boxes of a generated wiring get by default. """


class Sym(StrEnum):
    """
    The symmetry an orbit of legs carries, i.e. the group the module at
    that site must be equivariant under.

    * :attr:`NONE` : the legs are distinguishable, no equation.
    * :attr:`PERM` : the symmetric group, e.g. the members of a constraint
      unit or the legs of a spider.
    * :attr:`CYCLIC` : the cyclic group, e.g. the legs of a planar node.

    Example
    -------
    >>> Sym.PERM, str(Sym.PERM)
    (Sym.PERM, 'perm')
    """

    NONE = "none"
    PERM = "perm"
    CYCLIC = "cyclic"

    def __repr__(self):
        return f"Sym.{self.name}"


@dataclass(frozen=True)
class Orbit:
    """
    A family of legs playing the same role.

    An orbit has ``arity`` legs, each carrying the atomic type ``role``; a
    traced orbit has each leg twice, the outgoing copy followed by the
    incoming one, which :meth:`Signature.loops` wires together.

    Parameters:
        role : The atomic type of one leg.
        arity : The number of legs.
        sym : The group the legs are an orbit under.
        traced : Whether each leg is a self-wired pair of ports.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> print(Orbit(Ty("state"), traced=True).cod)
    state @ state
    >>> Orbit(Ty("peer"), 3, Sym.PERM)
    Orbit(role=frobenius.Ty(frobenius.Wire('peer')), arity=3, sym=Sym.PERM, \
traced=False)
    """

    role: frobenius.Ty
    arity: int = 1
    sym: Sym = Sym.NONE
    traced: bool = False

    def __post_init__(self):
        assert_isatomic(self.role, frobenius.Ty)
        if self.arity < 0:
            raise ValueError(self.arity)

    @property
    def copies(self) -> int:
        """ How many times each leg appears: two when traced, one else. """
        return 2 if self.traced else 1

    @property
    def n_ports(self) -> int:
        """ The number of ports of the orbit. """
        return self.arity * self.copies

    @property
    def cod(self) -> frobenius.Ty:
        """ The ports of the orbit, as a type. """
        return self.role ** self.n_ports


@dataclass(frozen=True)
class Signature:
    """
    The legs of one generator, as a tuple of orbits, in port order.

    Parameters:
        orbits : The orbits, in the order their ports appear.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> peer, hidden, memory = Ty("peer"), Ty("hidden"), Ty("memory")
    >>> clique = Signature((
    ...     Orbit(peer, 4, Sym.PERM), Orbit(hidden, traced=True),
    ...     Orbit(memory, traced=True)))
    >>> clique.positions(hidden), clique.positions(memory)
    ((4, 5), (6, 7))
    >>> clique.loops()
    ((4, 5), (6, 7))
    """

    orbits: tuple[Orbit, ...]

    @property
    def cod(self) -> frobenius.Ty:
        """ The codomain of the abstract box: every port, in order. """
        return frobenius.Ty().tensor(*(orbit.cod for orbit in self.orbits))

    @property
    def roles(self) -> tuple[frobenius.Ty, ...]:
        """ The role of each port, in order. """
        return tuple(self.cod)

    def box(self, name: str) -> frobenius.Box:
        """
        The abstract box of this signature: no domain, one port per leg.

        The box lives in :mod:`discopy.frobenius`, since every box a
        builder wires has an empty domain, so every wire joins two codomain
        ports, and a map accepts such a wire only between adjoint types:
        the roles have to be self-dual.

        Parameters:
            name : The name of the box, which is also the key its module is
                   looked up under.

        Example
        -------
        >>> from discopy.frobenius import Ty
        >>> unit = Signature((Orbit(Ty("message"), 3, Sym.PERM), ))
        >>> print(unit.box("unit"))
        unit
        >>> print(unit.box("unit").cod)
        message @ message @ message
        """
        return frobenius.Box(name, frobenius.Ty(), self.cod)

    def positions(self, role: frobenius.Ty) -> tuple[int, ...]:
        """
        Where a role sits in the port order.

        For a traced orbit the outgoing copies come first, so the ``i``-th
        and the ``arity + i``-th entries are the two ends of one loop.

        Parameters:
            role : The atomic role to locate.
        """
        return tuple(
            i for i, other in enumerate(self.roles) if other == role)

    def loops(self) -> tuple[tuple[int, int], ...]:
        """
        The traced pairs of ports, as positions in the port order.

        Two readings of one pair, worth keeping apart. *Structurally* a
        self-wired pair is the trace of the compact target -- wiring, which
        a functor preserves strictly. *Dynamically* it is a persistent state
        channel: what a box writes on one end it reads back on the other one
        round later, delayed feedback rather than a fixed point.
        """
        result, cursor = [], 0
        for orbit in self.orbits:
            if orbit.traced:
                result += [(cursor + i, cursor + orbit.arity + i)
                           for i in range(orbit.arity)]
            cursor += orbit.n_ports
        return tuple(result)

    def loop_wires(self, index: int) -> list:
        """
        The wires closing the traced ports of a box onto themselves, as
        pairs of ``(index, position)`` for
        :meth:`~discopy.cmap.CMap.from_wiring`.

        A loop is a trace, and the trace of a compact map is wiring: the
        same map comes out of tracing the boundary of an open one.

        Parameters:
            index : The index of the box in the map.

        Example
        -------
        >>> from discopy.frobenius import Box, CMap, Ty
        >>> g = Box("g", Ty("x"), Ty("x"))
        >>> loop = Signature((Orbit(Ty("x"), traced=True), ))
        >>> loop.loop_wires(0)
        [((0, 0), (0, 1))]
        >>> CMap.from_box(g).trace() == CMap.from_wiring(
        ...     (g, ), loop.loop_wires(0))
        True
        """
        return [((index, source), (index, target))
                for source, target in self.loops()]

    def resize(self, index: int, arity: int) -> Signature:
        """
        The same signature with the arity of one orbit changed, which is
        how one shared module serves sites of different degree.

        Parameters:
            index : The position of the orbit in ``orbits``, from zero.
            arity : Its new arity.

        Raises:
            IndexError : If there is no orbit at that position.
        """
        if not 0 <= index < len(self.orbits):
            raise IndexError(
                f"no orbit {index} in a signature of {len(self.orbits)}")
        return Signature(self.orbits[:index] + (
            replace(self.orbits[index], arity=arity), )
            + self.orbits[index + 1:])

    def generators(self) -> list[Permutation]:
        """
        The generators of the symmetry group of the signature, as
        permutations of its ports.

        A permutation acts on the *legs* of one orbit and on every copy of
        each leg alike, so a traced orbit stays traced: it is the tensor of
        the identity on the ports before the orbit, one copy of the leg
        permutation per copy of the orbit, and the identity after. The
        identity of the group generated is the equation the module at this
        site must satisfy.

        Example
        -------
        >>> from discopy.frobenius import Ty
        >>> unit = Signature((Orbit(Ty("message"), 3, Sym.PERM), ))
        >>> [tuple(permutation.inside) for permutation in unit.generators()]
        [(1, 0, 2), (1, 2, 0)]
        """
        result, cursor, n_ports = [], 0, len(self.cod)
        for orbit in self.orbits:
            after = Permutation.id(n_ports - cursor - orbit.n_ports)
            for legs in leg_generators(orbit.sym, orbit.arity):
                result.append(Permutation.id(cursor).tensor(
                    *(Permutation(legs), ) * orbit.copies, after))
            cursor += orbit.n_ports
        return result


def leg_generators(sym: Sym, arity: int) -> list[tuple[int, ...]]:
    """
    The generators of a symmetry group, as permutations of legs.

    This is the group itself, before it acts on anything:
    :meth:`Signature.generators` is the same group acting on ports.

    Parameters:
        sym : The symmetry the legs carry.
        arity : The number of legs.

    Example
    -------
    >>> leg_generators(Sym.PERM, 3)
    [(1, 0, 2), (1, 2, 0)]
    >>> leg_generators(Sym.CYCLIC, 3)
    [(1, 2, 0)]
    >>> leg_generators(Sym.NONE, 3), leg_generators(Sym.PERM, 1)
    ([], [])
    >>> leg_generators("perms", 3)
    Traceback (most recent call last):
        ...
    ValueError: 'perms' is not a valid Sym
    """
    sym = Sym(sym)
    if arity < 2 or sym == Sym.NONE:
        return []
    rotation = tuple(range(1, arity)) + (0, )
    if sym == Sym.CYCLIC:
        return [rotation]
    swap = (1, 0) + tuple(range(2, arity))
    return [swap, rotation]


def from_incidence(incidence: tuple, node: Signature,
                   relation: Signature | Mapping[str, Signature],
                   node_name: str = NODE,
                   relation_name: str | tuple[str, ...] = RELATION
                   ) -> frobenius.CMap:
    """
    The bipartite incidence graph of a family of nodes and the relations
    they belong to, as a closed map in :mod:`discopy.frobenius`: one node
    box per node with one incidence port per relation it belongs to plus
    its traced loops, one relation box per relation with one port per
    member, and a wire from each node to each of its relations.

    Neither the degrees nor the sizes need to be uniform: a node of degree
    ``d`` gets the node signature with its first orbit resized to ``d``,
    and likewise a relation of ``m`` members -- one shared module still
    fills every site of a name. A module shared across degrees must answer
    any number of legs alike, reading its degree off the tensors it is
    handed, since a layer built for one degree fails on another; one that
    pools its messages with a mean keeps its scale independent of the
    degree.

    Parameters:
        incidence : Per node, the indices of the relations it belongs to;
                    relations are numbered from ``0``.
        node : The signature of a node box, whose first orbit is the
               incidence orbit; its declared arity is a default, resized
               per node.
        relation : The signature of a relation box, whose first orbit is
                   the membership orbit, resized per relation; or a
                   mapping from relation name to signature when the
                   relations do not all share one.
        node_name : The name every node box carries.
        relation_name : The name every relation box carries, or one name
                        per relation -- boxes of one name share one
                        module, so a relation playing a different part,
                        e.g. a graph-level readout wired to every node,
                        is a relation with a name of its own.

    Raises:
        ValueError : If a node belongs to a relation that does not exist, or
                     if the names are not one per relation.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> message, given = Ty("message"), Ty("given")
    >>> node = Signature((Orbit(message, 2, Sym.PERM),
    ...                   Orbit(given, traced=True)))
    >>> unit = Signature((Orbit(message, 3, Sym.PERM), ))
    >>> square = from_incidence(((0, 1), (0, 1), (0, 1)), node, unit)
    >>> len(square.boxes), square.n_ports // 2
    (5, 9)

    A graph-level readout is one more relation every node belongs to,
    under its own name -- a generator, not a feature of the model:

    >>> shape = from_incidence(
    ...     ((0, 1), (0, 1), (1, )), node, unit,
    ...     relation_name=("unit", "readout"))
    >>> [box.name for box in shape.boxes]
    ['cell', 'cell', 'cell', 'unit', 'readout']
    """
    n_nodes = len(incidence)
    for index, relations in enumerate(incidence):
        for other in relations:
            if not isinstance(other, int) or other < 0:
                raise ValueError(f"node {index} belongs to relation {other}, "
                                 "which does not exist")
    size = Counter(other for relations in incidence for other in relations)
    n_relations = 1 + max(size, default=-1)
    names = (relation_name, ) * n_relations \
        if isinstance(relation_name, str) else tuple(relation_name)
    if len(names) != n_relations:
        raise ValueError(f"{len(names)} names for {n_relations} relations")
    relations_of = dict.fromkeys(names, relation) \
        if isinstance(relation, Signature) else dict(relation)

    nodes = [node.resize(0, len(relations)) for relations in incidence]
    units = [relations_of[names[index]].resize(0, size[index])
             for index in range(n_relations)]

    free = [0] * n_relations
    wires: list = []
    for index, relations in enumerate(incidence):
        for position, other in enumerate(relations):
            wires.append(
                ((index, position), (n_nodes + other, free[other])))
            free[other] += 1
        wires += nodes[index].loop_wires(index)
    for other in range(n_relations):
        wires += units[other].loop_wires(n_nodes + other)

    boxes = tuple(sig.box(node_name) for sig in nodes) + tuple(
        sig.box(names[index]) for index, sig in enumerate(units))
    return frobenius.CMap.from_wiring(boxes, wires)


def from_relation(relation: tuple, node: Signature, node_name: str = NODE
                  ) -> frobenius.CMap:
    """
    The graph of a binary relation between nodes, as a closed map in
    :mod:`discopy.frobenius`: one node box per node with one port per
    related node plus its traced loops, and a wire between each related
    pair. No relation boxes.

    The relation must be symmetric and irreflexive, a node having one port
    per related node where a self-edge would need two; the degrees need
    not be uniform, a node related to ``d`` others gets the node signature
    with its first orbit resized to ``d``, and the one shared module still
    fills every site (see :func:`from_incidence`).

    Parameters:
        relation : Per node, the indices of the nodes it is related to.
        node : The signature of a node box, whose first orbit is the
               relation orbit; its declared arity is a default, resized
               per node.
        node_name : The name every node box carries.

    Raises:
        ValueError : If the relation is not symmetric or not irreflexive,
                     relates a node to one that does not exist or to the
                     same node more than once.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> node = Signature((Orbit(Ty("peer"), 2, Sym.PERM),
    ...                   Orbit(Ty("state"), traced=True)))
    >>> triangle = from_relation(((1, 2), (0, 2), (0, 1)), node)
    >>> len(triangle.boxes), triangle.n_ports // 2
    (3, 6)
    >>> [len(box.cod) for box in from_relation(
    ...     ((1, ), (0, 2), (1, )), node).boxes]
    [3, 4, 3]
    """
    nodes = [node.resize(0, len(others)) for others in relation]

    wires: list = []
    for index, others in enumerate(relation):
        for other in others:
            if not isinstance(other, int) or not 0 <= other < len(relation):
                raise ValueError(f"node {index} is related to node {other}, "
                                 "which does not exist")
            if other == index:
                raise ValueError(f"node {index} is related to itself")
            if others.count(other) > 1:
                raise ValueError(f"node {index} is related to node {other} "
                                 "more than once")
            if index not in relation[other]:
                raise ValueError("the relation is not symmetric")
            if index < other:
                wires.append(((index, others.index(other)),
                              (other, relation[other].index(index))))
        wires += nodes[index].loop_wires(index)

    boxes = tuple(sig.box(node_name) for sig in nodes)
    return frobenius.CMap.from_wiring(boxes, wires)
