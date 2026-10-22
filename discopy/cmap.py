
"""
An implementation of open `combinatorial maps
<https://en.wikipedia.org/wiki/Combinatorial_map>`_.
See :cite:`DelpeuchVicary22` for a comprehensive overview of combinatorial
maps in relation to string diagrams.

A combinatorial map is fully described by a pair of permutations :math:`v` and
:math:`e` acting on a set of ports :math:`P` (also called darts or
half-edges) where:

* :math:`v` is an arbitrary permutation whose decomposition induces a node for
  each cycle, giving an orientation on ports;
* :math:`e` is a fixpoint-free involution, hence its cycle decomposition
  only contains transpositions which are to be understood as wires of the map.

A map morphism from :math:`(P, v, e)` to :math:`(P', v', e')` is then defined
as a function :math:`f : P \\rightarrow P'` such that:

* :math:`f` defines a homomorphism of the underlying graph:
  :math:`e; f = f; e'`;
* :math:`f` respects orientation: :math:`v; f = f; v'`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    PortKind
    Port
    CMap
"""

from __future__ import annotations

import operator
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property, reduce
from inspect import isclass
from io import BytesIO
from itertools import accumulate
from math import inf, lcm
from typing import TYPE_CHECKING, ClassVar, Literal

from discopy import hypergraph, messages
from discopy.abc import (
    BalancedCategory,
    BraidedCategory,
    CompactCategory,
    DaggerCategory,
    NamedGeneric,
    Pregroup,
    RigidCategory,
    SymmetricCategory,
    TracedCategory,
)
from discopy.cat import Ob, Sum
from discopy.python.finset import Permutation
from discopy.utils import (
    AxiomError,
    from_tree,
    resolve_factory,
    assert_isatomic,
    assert_istraceable,
    assert_isinstance,
    classproperty,
    factory_name,
    get_origin,
    unbiased,
)

if TYPE_CHECKING:
    from discopy.monoidal import Box, Diagram, Ty


class PortKind(StrEnum):
    """ The four kinds of ports in a :class:`CMap`. """

    INPUT = "input"
    OUTPUT = "output"
    DOM = "dom"
    COD = "cod"

    @property
    def is_negative(self) -> bool:
        """ Whether the port is a box input or map output. """
        return self == "dom" or self == "output"

    @property
    def is_positive(self) -> bool:
        """ Whether the port is a map input or box output. """
        return self == "input" or self == "cod"

    @property
    def is_boundary(self) -> bool:
        """ Whether the port belongs to the map boundary. """
        return self == "input" or self == "output"

    @property
    def is_input(self) -> bool:
        """ Whether the port is drawn on the input side. """
        return self == "input" or self == "dom"

    @property
    def is_output(self) -> bool:
        """ Whether the port is drawn on the output side. """
        return self == "cod" or self == "output"


@dataclass(frozen=True)
class Port:
    """
    A port in a combinatorial map.

    Parameters:
        kind : The kind of boundary or box port.
        i : The position within its boundary or box side.
        obj : The type carried by the port.
        depth : The box index, with inputs at ``-inf`` and outputs at ``+inf``.
        side : The vertical side on which the port is drawn.
    """
    kind: PortKind
    i: int
    obj: Ob
    depth: float
    side: Literal["up", "down"]


class CMap[C0: Pregroup, C1: CMap](
    CompactCategory[C0, C1], DaggerCategory[C0, C1], NamedGeneric['category']
):
    r"""
    An open combinatorial map, i.e. a diagram represented as a bijection
    between its ports.

    Contrary to the abstract definition, which has unstructured nodes arising
    from the given orientation permutation, we take DisCoPy boxes as nodes and
    derive a canonical clockwise port orientation on boxes: every box of arity
    :math:`m` and coarity :math:`n` maps to a :math:`(m+n)`-cycle in the
    generated permutation, consisting of contiguous port indices.
    Additionally, we allow two kinds of scalars:

    * `scalar loops` arising from composing cups and caps, parametrized by an
      atomic type;
    * `scalar boxes`, i.e. boxes with empty domain and codomain

    As for the open structure, we represent the map boundary by a virtual apex
    node, whose signature is the dagger of the that of the overall map.

    By default, `CMap` defines the free compact category over a set of boxes,
    but we also want to be able to encode weaker structure, disallowing cups
    and caps or even traced structure altogether.
    We therefore further distinguish port sides by assigning a negative
    polarity on domain ports and a positive polarity on codomain ports
    by equipping the map with a polarity assignment
    :math:`m : P \rightarrow \{-1, +1\}`.

    Following :class:`Hypergraph`, the map is parametrised by a category.
    The functor used by :meth:`from_diagram` is read from
    ``category.functor_factory``; :meth:`Diagram.to_map` parameterises
    ``CMap`` with the concrete diagram category automatically.
    A map is always compact, whatever the category that hosts it, so that
    every compact operation is available when manipulating maps. It is
    :meth:`to_diagram` that needs the structure of ``category``:

    * cups and caps, i.e. same-polarity pairings :math:`e; m = m` (see
      :attr:`is_monogamous`), are made explicit by :meth:`make_monogamous`,
      which needs a category with cups and caps;
    * traces, i.e. cycles and loops (see :attr:`is_acyclic`), are made
      explicit by :meth:`make_causal`, which needs a traced category.

    Parameters:
        dom : The domain of the map.
        cod : The codomain of the map.
        boxes : The boxes inside the map.
        edges : A fixpoint-free involution on ports.
        loops : The types of closed wire components with no ports.
        check : Whether to :meth:`validate` the involution and the wire types.

    Example
    -------
    >>> from discopy.compact import Ty, Box, CMap
    >>> from discopy.python.finset import Permutation
    >>> x, y, z = map(Ty, "xyz")
    >>> f, g = map(CMap.from_box, [
    ...     Box("f", x @ y, x @ z),
    ...     Box("g", z @ z, z),
    ... ])
    >>> cm = f @ z >> x @ g
    >>> # apex: 10 : x, 11 : z ⊢ 2 : x, 1 : y, 0 : z
    >>> # f:    3 : x, 4 : y ⊢ 6 : x, 5 : z
    >>> # g:    7 : z, 8 : z ⊢ 9 : z
    >>> cm.edges == Permutation.from_cycles([
    ...     (0, 3), (1, 4), (2, 8), (5, 7), (6, 10), (9, 11)], 12)
    True
    >>> cm.orientation == Permutation.from_cycles([
    ...     (2, 1, 0, 10, 11), (3, 4, 5, 6), (7, 8, 9)], 12)
    True
    >>> cm.draw(
    ...     doctest="docs/_static/cmap/simple-cmap.dot",
    ...     port_indices=True,
    ...     show=False,
    ... )

    .. graphviz:: /_static/cmap/simple-cmap.dot
        :align: center

    Swaps affect the edge permutation but leave the vertex permutation
    fixed:

    >>> f, g = map(CMap.from_box, [
    ...     Box("f", x @ y, z @ x),
    ...     Box("g", z @ z, z),
    ... ])
    >>> cm = (f >> CMap.swap(z, x)) @ z >> x @ g
    >>> cm.draw(
    ...     doctest="docs/_static/cmap/swapped-cmap.dot",
    ...     port_indices=True,
    ...     show=False,
    ... )

    .. graphviz:: /_static/cmap/swapped-cmap.dot
        :align: center
    """

    category: ClassVar[Diagram] = None
    functor = classproperty(lambda cls: cls.category.functor_factory)
    ob = classproperty(lambda cls: cls.category.ob)

    dom: C0
    cod: C0
    loops: tuple[C0, ...]
    edges: Permutation

    def __init__(
            self, dom: C0, cod: C0, boxes: tuple[Box, ...],
            edges: Iterable[int],
            loops: tuple[C0, ...] = (), *, check: bool = True):
        assert_isinstance(dom, self.category.ob)
        assert_isinstance(cod, self.category.ob)
        for box in boxes:
            assert_isinstance(box, self.category)
        for loop in loops:
            assert_isatomic(loop, self.category.ob)
        self.dom, self.cod, self.boxes = dom, cod, tuple(boxes)
        self.loops = tuple(loops)

        self.edges = Permutation(edges, self.n_ports)
        if check:
            self.validate()

    @cached_property
    def ports(self) -> list[Port]:
        """ The ports in canonical orientation order. """
        def port(kind, i, obj, depth):
            if not kind.is_boundary:
                depth += 0.5 if kind.is_input else -0.5
            return Port(
                kind, i=i, obj=obj, depth=depth,
                side="up" if kind.is_input else "down")

        inputs = [port(PortKind.INPUT, i=i, obj=obj, depth=-inf)
                  for i, obj in enumerate(self.dom)]
        box_ports = reduce(operator.iadd, [
            [
              port(kind, i=i, obj=obj, depth=depth)
              for i, obj in indexed_typ
            ]
            for depth, box in enumerate(self.boxes)
            for kind, indexed_typ in [
                (PortKind.DOM, tuple(enumerate(box.dom))),
                (PortKind.COD, tuple(reversed(tuple(enumerate(box.cod)))))
            ]
        ], [])
        outputs = [port(PortKind.OUTPUT, i=i, obj=obj, depth=inf)
                   for i, obj in enumerate(self.cod)]
        return inputs + box_ports + outputs

    @property
    def n_ports(self) -> int:
        """ The number of ports. """
        return len(self.dom) + sum(
            len(box.dom) + len(box.cod) for box in self.boxes) + len(self.cod)

    @cached_property
    def _box_port_indices(self) -> tuple[tuple[int, ...], ...]:
        """ The consecutive port indices belonging to each box. """
        result, start = [], len(self.dom)
        for box in self.boxes:
            stop = start + len(box.dom) + len(box.cod)
            result.append(tuple(range(start, stop)))
            start = stop
        return tuple(result)

    @staticmethod
    def logical_order(arity: int, coarity: int) -> tuple[int, ...]:
        """
        The offsets of the ports of a box in logical order, i.e. its domain
        ports followed by its codomain ports, within the clockwise order
        which stores the codomain ports reversed.

        Parameters:
            arity : The number of domain ports.
            coarity : The number of codomain ports.

        Example
        -------
        >>> CMap.logical_order(2, 3)
        (0, 1, 4, 3, 2)
        """
        return tuple(range(arity)) + tuple(
            reversed(range(arity, arity + coarity)))

    @property
    def faces(self) -> Permutation:
        """ The face permutation, computed as ``edges ; orientation``. """
        return self.edges.then(self.orientation)

    @property
    def n_vertices(self) -> int:
        """ The number of vertices, including the boundary apex if present. """
        return len(self.boxes) + bool(len(self.dom) or len(self.cod))

    @property
    def n_edges(self) -> int:
        """ The number of edges. """
        return self.n_ports // 2 + len(self.loops)

    @property
    def n_faces(self) -> int:
        """ The number of faces, including closed scalar components. """
        portless_boxes = sum(
            not len(box.dom) and not len(box.cod) for box in self.boxes)
        return len(self.faces.cycles()) + portless_boxes\
            + len(self.loops)

    @property
    def euler_characteristic(self) -> int:
        """
        Euler characteristic ``V - E + F`` with boundary at infinity.

        For maps with non-empty domain or codomain, the input and output ports
        are treated as one virtual boundary/apex, ordered clockwise as inputs
        left-to-right followed by outputs right-to-left. Fully closed maps have
        no boundary apex.

        >>> from discopy.symmetric import Ty, Box, Swap
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box("f", x @ y, z)
        >>> f.to_map().euler_characteristic
        2
        >>> (Swap(y, x) >> f).to_map().euler_characteristic
        0
        """
        if len(self.connected_components) != 1:
            raise ValueError(messages.NOT_CONNECTED.format(self))
        if not self.n_ports and not self.boxes and not self.loops:
            return 2
        return self.n_vertices - self.n_edges + self.n_faces

    @property
    def is_scalar(self) -> bool:
        """
        Whether the map is a scalar, i.e. an endomorphism of the unit.

        Scalars are the maps with no boundary, so they are closed under
        tensor:

        >>> from discopy.compact import Ty, Box, CMap
        >>> x = Ty("x")
        >>> loop = CMap.caps(x.r, x) >> CMap.cups(x.r, x)
        >>> box = Box("s", Ty(), Ty()).to_map()
        >>> assert loop.is_scalar and box.is_scalar
        >>> assert (loop @ box).is_scalar
        >>> assert not CMap.id(x).is_scalar
        """
        return not self.dom and not self.cod

    @property
    def genus(self) -> int:
        r"""
        The genus of a connected map, i.e. the number of handles of the
        smallest orientable surface it embeds in without crossings.

        This is :math:`(2 - \chi) / 2` for the Euler characteristic
        :math:`\chi` of :attr:`euler_characteristic`, which it defers to and
        which raises :class:`ValueError` on a map that is not connected.

        A map with no vertex, i.e. a loop, has genus zero: a circle does
        embed in the sphere, but Euler's formula only counts it once a
        vertex has subdivided it, which a loop by definition has not. This
        is the only degenerate case -- a closed map is a scalar and has a
        genus like any other.

        >>> from discopy.symmetric import Ty, Box, Swap
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box("f", x @ y, z)
        >>> f.to_map().genus
        0
        >>> (Swap(y, x) >> f).to_map().genus
        1
        """
        characteristic = self.euler_characteristic
        return 0 if not self.n_vertices else (2 - characteristic) // 2

    @property
    def is_planar(self) -> bool:
        """
        Whether the combinatorial map is planar, i.e. all of its components
        have :attr:`genus` zero.
        """
        return all(
            component.genus == 0
            for component in self.connected_components)

    @property
    def orientation(self) -> Permutation:
        """
        The closed orientation permutation.

        The first cycle is the boundary apex, when the boundary is non-empty.
        Each following non-empty cycle is the contiguous port interval of a
        box in canonical order: domain ports, then codomain ports.

        >>> from discopy.compact import Ty, Box, CMap
        >>> from discopy.python.finset import Permutation
        >>> x, y, z = map(Ty, "xyz")
        >>> f, g = Box('f', x @ y, x @ z), Box('g', z @ z, z)
        >>> cm = (f @ z >> x @ g).to_map()
        >>> assert cm.orientation == Permutation.from_cycles([
        ...     (2, 1, 0, 10, 11), # boundary
        ...     (3, 4, 5, 6),      # f
        ...     (7, 8, 9),         # g
        ... ], 12), f"got {cm.orientation.cycles()!r}"
        """
        boundary = (self.boundary_cycle, ) if self.boundary_cycle else ()
        return Permutation.from_cycles(
            boundary + self._box_port_indices, len(self.ports))

    @property
    def boundary_cycle(self) -> tuple[int, ...]:
        """ The clockwise cycle of the virtual boundary apex. """
        inputs = tuple(range(len(self.dom)))
        outputs = tuple(range(self.n_ports - len(self.cod), self.n_ports))
        return tuple(reversed(inputs)) + outputs

    def validate(self):
        """ Validate the edges involution and the types of each wire. """
        ports = self.ports
        if not self.edges.is_fixpoint_free_involution():
            raise ValueError(messages.NOT_AN_INVOLUTION.format(self.edges))

        for i, j in enumerate(self.edges):
            if i > j:
                continue
            type(self).validate_wire(ports[i], ports[j])

    @classmethod
    def assert_isrigid(cls):
        """ Assert that :attr:`category` has cups and caps. """
        if not issubclass(cls.category, RigidCategory):
            raise AxiomError(messages.NOT_RIGID.format(
                factory_name(cls.category)))

    @classmethod
    def assert_istraced(cls):
        """ Assert that :attr:`category` has traces. """
        if not issubclass(cls.category, TracedCategory):
            raise AxiomError(messages.NOT_TRACED.format(
                factory_name(cls.category)))

    @property
    def connected_components(self) -> list[CMap]:
        """ The connected components, with the boundary component first. """
        if not self.n_ports:
            # Avoid recursively rebuilding the same portless component.
            if len(self.boxes) + len(self.loops) <= 1:
                return [self]
            components = [
                type(self)(self.ob(), self.ob(), (box, ), (), check=False)
                for box in self.boxes]
            components += [
                type(self)(
                    self.ob(), self.ob(), (), (), loops=(loop, ), check=False)
                for loop in self.loops]
            return components

        component_of = self.edges.coequalizer(self.orientation)
        boundary = set(range(len(self.dom))) | set(range(
            self.n_ports - len(self.cod), self.n_ports))
        boundary_component = component_of[next(iter(boundary))]\
            if boundary else None

        ports_by_component: dict[int, list[int]] = {}
        for port, component in component_of.items():
            ports_by_component.setdefault(component, []).append(port)

        boxes_by_component: dict[int, list[tuple[int, Box]]] = {}
        portless_boxes: list[Box] = []
        for box_index, box in enumerate(self.boxes):
            box_ports = self._box_port_indices[box_index]
            if not box_ports:
                portless_boxes.append(box)
                continue
            component = component_of[box_ports[0]]
            boxes_by_component.setdefault(component, []).append((
                box_index, box))

        if len(ports_by_component) == 1 and not portless_boxes\
                and not self.loops:
            return [self]

        def make_component(component: int) -> CMap:
            dom = self.dom if component == boundary_component else self.ob()
            cod = self.cod if component == boundary_component else self.ob()
            boxes = tuple(box for _, box in boxes_by_component.get(
                component, ()))

            kept_ports = []
            if component == boundary_component:
                kept_ports += list(range(len(self.dom)))
            for box_index, _ in boxes_by_component.get(component, ()):
                kept_ports += list(self._box_port_indices[box_index])
            if component == boundary_component:
                kept_ports += list(range(
                    self.n_ports - len(self.cod), self.n_ports))
            mapping = {old: new for new, old in enumerate(kept_ports)}
            edges = Permutation.from_transpositions(
                ((mapping[i], mapping[j])
                 for i, j in enumerate(self.edges)
                 if i < j and i in mapping and j in mapping),
                len(kept_ports))
            return type(self)(dom, cod, boxes, edges, check=False)

        ordered_components = sorted(
            ports_by_component,
            key=lambda component: (
                component != boundary_component,
                min(ports_by_component[component])))
        components = [make_component(component)
                      for component in ordered_components]
        components += [
            type(self)(self.ob(), self.ob(), (box, ), (), check=False)
            for box in portless_boxes]
        components += [
            type(self)(
                self.ob(), self.ob(), (), (), loops=(loop, ), check=False)
            for loop in self.loops]
        return components

    def splice(
            self, edges: Permutation,
            glue: Permutation,
            ports: list[Port]) -> tuple[Permutation, tuple]:
        """
        Compute the edges and scalars created by a gluing operation.
        """
        components = edges.coequalizer(glue)
        removed = {port for port in range(len(glue)) if glue[port] != port}
        removed_by_component: dict[int, list[int]] = {}
        for port in removed:
            removed_by_component.setdefault(components[port], []).append(port)
        kept = [i for i in range(len(edges)) if i not in removed]
        mapping = {old: new for new, old in enumerate(kept)}
        surviving: dict[int, list[int]] = {}
        for port, component in components.items():
            if port not in removed:
                surviving.setdefault(component, []).append(port)

        edge_pairs = [
            tuple(sorted(mapping[port] for port in ports))
            for ports in surviving.values() if len(ports) == 2]
        scalars, scalar_components = [], set()
        for component, removed_ports in removed_by_component.items():
            if component in surviving or component in scalar_components:
                continue
            scalar = ports[removed_ports[0]].obj
            scalar = scalar if isinstance(scalar, self.category.ob)\
                else self.ob(scalar)
            scalars.append(
                scalar.r if getattr(scalar, "z", 0) % 2 else scalar)
            scalar_components.add(component)
        return (
            Permutation.from_transpositions(edge_pairs, len(kept)),
            tuple(scalars)
        )

    @classmethod
    def validate_equal_types(cls, source: Port, target: Port):
        """ Validate a wire between equal types. """
        if not source.obj == target.obj:
            raise AxiomError(messages.NOT_ADJOINT.format(
                source.obj, target.obj))

    @classmethod
    def validate_adjoint_types(cls, source: Port, target: Port):
        """ Validate a wire between adjoint types. """
        adjoint_types = getattr(source.obj, "r", None) == target.obj\
            or source.obj == getattr(target.obj, "r", None)
        if not adjoint_types:
            raise AxiomError(messages.NOT_ADJOINT.format(
                source.obj, target.obj))

    @classmethod
    def validate_wire(cls, source: Port, target: Port):
        """
        Validate type compatibility for a wire between two ports.

        Raises:
            AxiomError : If the types or orientations are incompatible.
        """
        if source.kind.is_positive and target.kind.is_negative:
            cls.validate_equal_types(source, target)
        elif target.kind.is_positive and source.kind.is_negative:
            cls.validate_equal_types(target, source)
        else:
            cls.validate_adjoint_types(source, target)

    @property
    def is_monogamous(self) -> bool:
        """
        Checks monogamy, i.e. every wire connects a positive to a negative
        port, so that the map has no cups or caps. This is the analogue of
        :attr:`Hypergraph.is_monogamous`, in which case the map lives in a
        traced category.

        >>> from discopy.compact import Ty, CMap
        >>> x = Ty("x")
        >>> assert CMap.id(x).is_monogamous
        >>> assert not CMap.cups(x, x.r).is_monogamous
        """
        ports = self.ports
        return all(
            ports[i].kind.is_positive != ports[j].kind.is_positive
            for i, j in enumerate(self.edges) if i < j)

    @property
    def box_edges(self) -> Iterable[tuple[int, int]]:
        """
        The directed wires from the codomain of a box to the domain of
        another, as pairs of source and target port indices.
        """
        ports = self.ports
        for i, j in enumerate(self.edges):
            if i > j:
                continue
            source, target = (i, j) if ports[i].kind.is_positive else (j, i)
            if ports[source].kind == PortKind.COD\
                    and ports[target].kind == PortKind.DOM:
                yield source, target

    @cached_property
    def box_ranks(self) -> tuple[int, ...] | None:
        """
        The rank of each box, i.e. the longest directed path of
        :attr:`box_edges` that reaches it, or ``None`` if there is a cycle.
        """
        ports = self.ports
        dependents = [[] for _ in self.boxes]
        indegree = [0] * len(self.boxes)
        for source, target in self.box_edges:
            i = int(ports[source].depth + 0.5)
            j = int(ports[target].depth - 0.5)
            dependents[i].append(j)
            indegree[j] += 1
        ranks = [0] * len(self.boxes)
        ready = [i for i, degree in enumerate(indegree) if not degree]
        seen = 0
        while ready:
            source = ready.pop()
            seen += 1
            for target in dependents[source]:
                ranks[target] = max(ranks[target], ranks[source] + 1)
                indegree[target] -= 1
                if not indegree[target]:
                    ready.append(target)
        return tuple(ranks) if seen == len(self.boxes) else None

    @property
    def is_acyclic(self) -> bool:
        """
        Whether the directed wiring has no cycles or scalar loops, i.e.
        whether :attr:`box_edges` admits a topological sort.

        >>> from discopy.compact import Ty, Box
        >>> x = Ty("x")
        >>> f = Box("f", x, x).to_map()
        >>> assert f.is_acyclic
        >>> assert not f.trace().is_acyclic
        """
        return not self.loops and self.box_ranks is not None

    @property
    def is_topologically_ordered(self) -> bool:
        """
        Whether every directed wire points forward in the box order, i.e.
        every wire from the codomain of a box to the domain of another goes
        to a box of greater :attr:`Port.depth`.

        >>> from discopy.compact import Ty, Box
        >>> x = Ty("x")
        >>> f, g = Box("f", x, x), Box("g", x, x)
        >>> assert (f.to_map() >> g.to_map()).is_topologically_ordered
        >>> snakes = f.transpose(left=True) >> g.transpose(left=True)
        >>> assert not snakes.to_map().is_topologically_ordered
        """
        ports = self.ports
        return all(
            int(ports[i].depth + 0.5) < int(ports[j].depth - 0.5)
            for i, j in self.box_edges)

    def reorder(self, order: Iterable[int]) -> CMap:
        """ Relabel ports to put boxes in the given order. """
        order = tuple(order)
        boxes = tuple(self.boxes[i] for i in order)
        mapping = list(range(self.n_ports))
        start = len(self.dom)
        for old in order:
            old_ports = self._box_port_indices[old]
            for source, target in zip(
                    old_ports, range(start, start + len(old_ports))):
                mapping[source] = target
            start += len(old_ports)
        edges = self.edges.conjugate(Permutation(mapping))
        return type(self)(
            self.dom, self.cod, boxes, edges,
            loops=self.loops, check=False)

    def topological_order(self) -> CMap:
        """
        Reorder the boxes so that every directed wire points forward.

        This relabels the box order without touching the wiring. It is the
        identity on :attr:`is_causal` maps.

        Raises:
            AxiomError : If the map has a directed cycle, i.e. it is not
                :attr:`is_acyclic`, so that no such order exists.

        >>> from discopy.compact import Ty, Box
        >>> x = Ty("x")
        >>> f, g = Box("f", x, x), Box("g", x, x)
        >>> snakes = (f.transpose(left=True) >> g.transpose(left=True))
        >>> assert not snakes.to_map().is_topologically_ordered
        >>> assert snakes.to_map().topological_order().boxes == (g, f)
        >>> ordered = (f.to_map() >> g.to_map()) @ f.to_map()
        >>> assert ordered.topological_order() is ordered
        >>> f.to_map().trace().topological_order()
        Traceback (most recent call last):
        ...
        discopy.utils.AxiomError: ... has a directed cycle, ...
        """
        ranks = self.box_ranks
        if ranks is None:
            raise AxiomError(messages.NOT_ACYCLIC.format(self))
        if self.is_topologically_ordered:
            return self
        return self.reorder(tuple(sorted(
            range(len(self.boxes)), key=lambda i: (ranks[i], i))))

    @property
    def is_causal(self) -> bool:
        """
        Checks causality, i.e. the map is :attr:`is_monogamous` with no
        loops and :attr:`is_topologically_ordered`, which implies
        :attr:`is_acyclic`. A causal map lives in a symmetric category,
        i.e. it can be drawn using only swaps.

        >>> from discopy.compact import Ty, Box, CMap
        >>> x = Ty("x")
        >>> f = Box("f", x, x).to_map()
        >>> assert (f >> f).is_causal
        >>> assert not f.trace().is_causal
        >>> assert not CMap.cups(x, x.r).is_causal
        """
        return not self.loops and self.is_monogamous\
            and self.is_topologically_ordered

    def __repr__(self):
        return factory_name(type(self))\
            + f"(dom={self.dom!r}, cod={self.cod!r}, " \
              f"boxes={self.boxes!r}, edges={self.edges!r}, " \
              f"loops={self.loops!r})"

    def __eq__(self, other: object):
        return isinstance(other, CMap) and (
            self.dom, self.cod, self.boxes, self.edges, self.loops
        ) == (
            other.dom, other.cod, other.boxes, other.edges, other.loops)

    def __hash__(self):
        return hash((
            self.dom, self.cod, self.boxes, self.edges, self.loops))

    def to_tree(self) -> dict:
        """
        Serialise a map, with the name of its category.

        >>> from discopy.utils import dumps, loads
        >>> from discopy.compact import Ty, Box, CMap
        >>> x = Ty('x')
        >>> cmap = CMap.caps(x, x.r) >> Box('f', x, x).to_map() @ x.r
        >>> assert loads(dumps(cmap)) == cmap
        """
        return {
            'factory': factory_name(get_origin(type(self))),
            'category': factory_name(self.category),
            'dom': self.dom.to_tree(),
            'cod': self.cod.to_tree(),
            'boxes': [box.to_tree() for box in self.boxes],
            'edges': [int(i) for i in self.edges],
            'loops': [loop.to_tree() for loop in self.loops]}

    @classmethod
    def from_tree(cls, tree: dict) -> CMap:
        """ Decode a serialised map, see :meth:`to_tree`. """
        factory = cls[resolve_factory(tree['category'])]
        return factory(
            from_tree(tree['dom']), from_tree(tree['cod']),
            tuple(map(from_tree, tree['boxes'])), tree['edges'],
            loops=tuple(map(from_tree, tree['loops'])))

    @classmethod
    def id(cls, dom=None) -> CMap:
        """ The identity map, with each input wired to its output. """
        dom = cls.ob() if dom is None else dom
        n_ports = 2 * len(dom)
        edge = Permutation.from_transpositions(
            ((i, i + len(dom)) for i in range(len(dom))), n_ports)
        return cls(dom, dom, (), edge, check=False)

    @classmethod
    def from_box(cls, box: Box) -> CMap:
        """ Embed a box, wiring its boundary to fresh box ports. """
        left = len(box.dom)
        right = len(box.cod)
        n_ports = 2 * (left + right)
        edge = Permutation.from_transpositions(
            [(i, left + i) for i in range(left)]
            + [(2 * left + right - i - 1, 2 * left + right + i)
               for i in range(right)],
            n_ports)
        return cls(box.dom, box.cod, (box, ), edge, check=False)

    @classmethod
    def from_glued(cls, dom: Ty, cod: Ty,
                   images: Iterable[tuple[CMap, int]]) -> CMap:
        """
        Glue a sequence of maps onto a scan of open wires, in one pass.

        Each wire of the result is a connected component of the wires of the
        ``images``, computed by union-find as they are glued. This is the
        colimit of the diagram of gluings, i.e. the same map as the iterated
        :meth:`then` of the ``images`` whiskered at their offsets, but built
        once rather than rebuilt at every step.

        Parameters:
            dom : The domain of the result.
            cod : The codomain of the result.
            images : Each map to glue, together with the offset at which its
                domain meets the scan.

        >>> from discopy.compact import Ty, Box, CMap
        >>> x = Ty("x")
        >>> f, g = map(CMap.from_box, [Box("f", x, x), Box("g", x, x)])
        >>> CMap.from_glued(x, x, [(f, 0), (g, 0)]) == f >> g
        True
        >>> CMap.from_glued(Ty(), Ty(), [
        ...     (CMap.caps(x.r, x), 0), (CMap.cups(x.r, x), 0)]).loops == (x, )
        True
        """
        wires, ends, objects = [], [], []

        def fresh(obj):
            wires.append(len(wires))
            ends.append([])
            objects.append(obj)
            return len(wires) - 1

        def find(wire):
            while wires[wire] != wire:
                wires[wire] = wires[wires[wire]]
                wire = wires[wire]
            return wire

        def union(source, target):
            source, target = sorted([find(source), find(target)])
            if source != target:
                ends[source] += ends[target]
                wires[target] = source

        scan = []
        for i, obj in enumerate(dom):
            scan.append(fresh(obj))
            ends[scan[i]].append(i)
        boxes, loops, start = (), (), len(dom)
        for image, offset in images:
            arity, coarity = len(image.dom), len(image.cod)
            local, image_ports = {}, image.ports
            for source, target in enumerate(image.edges):
                if source <= target:
                    local[source] = local[target] = fresh(
                        image_ports[source].obj)
            for port in range(arity, image.n_ports - coarity):
                ends[find(local[port])].append(start + port - arity)
            for i in range(arity):
                union(scan[offset + i], local[i])
            scan[offset:offset + arity] = [
                local[image.n_ports - coarity + i] for i in range(coarity)]
            boxes, loops = boxes + image.boxes, loops + image.loops
            start += image.n_ports - arity - coarity
        for i, wire in enumerate(scan):
            ends[find(wire)].append(start + i)

        edges = list(range(start + len(cod)))
        for wire in {find(wire) for wire in range(len(wires))}:
            if not ends[wire]:
                loop = objects[wire]
                loop = loop if isinstance(loop, cls.category.ob)\
                    else cls.ob(loop)
                loops = loops + (
                    loop.r if getattr(loop, "z", 0) % 2 else loop, )
            else:
                source, target = ends[wire]
                edges[source], edges[target] = target, source
        return cls(dom, cod, boxes, edges, loops=loops)

    @classmethod
    def from_wiring(cls, boxes: tuple, wires, loops: tuple = ()) -> CMap:
        """
        A closed map given by boxes and wires between pairs ``(box_index,
        port_position)``, where the position counts the domain ports of the
        box followed by its codomain ports, in :meth:`logical_order`.

        Parameters:
            boxes : The boxes of the map.
            wires : Pairs of ``(box_index, port_position)`` pairs.
            loops : The types of the closed components with no ports.

        Raises:
            ValueError : If a wire names a box or a port that does not exist,
                or if a port is left unwired or wired twice.

        Example
        -------
        >>> from discopy.symmetric import Ty, Box, CMap
        >>> x = Ty('x')
        >>> f, g = Box('f', x, x @ x), Box('g', x @ x, x)
        >>> cm = CMap.from_wiring((f, g), [
        ...     ((0, 0), (1, 2)), ((0, 1), (1, 0)), ((0, 2), (1, 1))])
        >>> assert cm.edges.is_fixpoint_free_involution()
        >>> assert CMap.from_wiring((), [], loops=(x, )).loops == (x, )
        >>> CMap.from_wiring((f, ), [((0, 0), (0, 0))])
        Traceback (most recent call last):
            ...
        ValueError: Port (0, 0) is wired to itself.
        """
        boxes = tuple(boxes)
        starts, n_ports = [], 0
        for box in boxes:
            assert_isinstance(box, cls.category)
            starts.append(n_ports)
            n_ports += len(box.dom) + len(box.cod)

        def global_index(box_index: int, position: int) -> int:
            if not isinstance(box_index, int) \
                    or not 0 <= box_index < len(boxes):
                raise ValueError(f"There is no box {box_index}.")
            box = boxes[box_index]
            offsets = cls.logical_order(len(box.dom), len(box.cod))
            if not isinstance(position, int) \
                    or not 0 <= position < len(offsets):
                raise ValueError(f"Box {box_index} has no port {position}.")
            return starts[box_index] + offsets[position]

        pairs, seen = [], set()
        for (one, other) in wires:
            i, j = global_index(*one), global_index(*other)
            if i == j:
                raise ValueError(f"Port {one} is wired to itself.")
            for port, position in ((i, one), (j, other)):
                if port in seen:
                    raise ValueError(f"Port {position} is wired twice.")
            seen.update((i, j))
            pairs.append((i, j))
        if len(seen) != n_ports:
            missing = sorted(set(range(n_ports)) - seen)
            raise ValueError(f"Ports {missing} are left unwired.")
        edges = Permutation.from_transpositions(pairs, n_ports)
        return cls(cls.ob(), cls.ob(), boxes, edges, loops=loops)

    @classmethod
    def from_diagram(cls, old: Diagram) -> CMap:
        """
        Turn a :class:`Diagram` into a :class:`CMap`.

        Structure available at the map's categorical level becomes wiring;
        structure from the next level remains represented by boxes.

        The image of each box is computed by the functor into ``cls``, then
        the images are glued in a single pass with :meth:`from_glued`. The
        trace of a planar traced category is a box, as for
        :meth:`Hypergraph.from_diagram`.

        >>> from discopy.braided import Ty, Braid
        >>> from discopy.monoidal import CMap
        >>> x, y = map(Ty, "xy")
        >>> CMap.from_diagram(Braid(x, y)).boxes == (Braid(x, y),)
        True
        >>> from discopy.symmetric import Ty as STy, Swap
        >>> x, y = map(STy, "xy")
        >>> Swap(x, y).to_map().boxes
        ()
        """
        hypergraph.assert_transparent(old)
        category = type(old).ar
        factory = cls if cls.category is category else cls[category]
        if any(isinstance(box, Sum) for box in old.boxes):
            raise AxiomError(messages.NO_SUMS.format(factory_name(factory)))
        functor = (factory.functor if cls.category is None else cls.functor)(
            ob_map=lambda typ: typ, ar_map=factory.from_box,
            dom=category, cod=factory)
        trace = getattr(category, "trace_factory", ())
        if not isclass(trace) or issubclass(category, SymmetricCategory):
            trace = ()
        return factory.from_glued(old.dom, old.cod, [
            (factory.from_box(box) if isinstance(box, trace)
             else functor(box), offset)
            for layer in old.inside
            for box, offset in layer.boxes_and_offsets])

    @classmethod
    def swap(cls, left: Ty, right: Ty) -> CMap:
        """
        The swap of a symmetric category, absorbed into the edges.

        >>> from discopy import braided
        >>> x, y = braided.Ty('x'), braided.Ty('y')
        >>> CMap[braided.Diagram].swap(x, y)
        Traceback (most recent call last):
        ...
        discopy.utils.AxiomError: braided.Diagram is not symmetric, ...
        """
        if not issubclass(cls.category, SymmetricCategory):
            raise AxiomError(messages.NO_SWAP.format(
                factory_name(cls.category)))
        dom, cod = left @ right, right @ left
        left_len, right_len = len(left), len(right)
        output_start = len(dom)
        edge = Permutation.from_transpositions(
            [(i, output_start + right_len + i)
             for i in range(left_len)]
            + [(left_len + i, output_start + i)
               for i in range(right_len)],
            2 * len(dom))
        return cls(dom, cod, (), edge, check=False)

    @classmethod
    def braid(cls, left: Ty, right: Ty) -> CMap:
        """
        The braid of a braided category: the :meth:`swap` of a symmetric one,
        and boxes otherwise, a braid being no swap.

        >>> from discopy import braided
        >>> x, y = braided.Ty('x'), braided.Ty('y')
        >>> braid = CMap[braided.Diagram].braid(x, y)
        >>> assert braid.boxes == (braided.Braid(x, y), )
        """
        if not issubclass(cls.category, BraidedCategory):
            raise AxiomError(messages.NO_BRAID.format(
                factory_name(cls.category)))
        if issubclass(cls.category, SymmetricCategory):
            return cls.swap(left, right)
        if len(left) == len(right) == 1:
            return cls.from_box(cls.category.braid_factory(left, right))
        return cls.from_diagram(cls.category.braid(left, right))

    @classmethod
    def twist(cls, dom: Ty) -> CMap:
        """
        The twist of a balanced category: the identity of a symmetric one,
        and boxes otherwise, a twist being no identity.

        >>> from discopy import balanced
        >>> x = balanced.Ty('x')
        >>> twist = CMap[balanced.Diagram].twist(x)
        >>> assert twist.boxes == (balanced.Twist(x), )
        """
        if issubclass(cls.category, SymmetricCategory):
            return cls.id(dom)
        if not issubclass(cls.category, BalancedCategory):
            raise AxiomError(messages.NO_TWIST.format(
                factory_name(cls.category)))
        if len(dom) == 1:
            return cls.from_box(cls.category.twist_factory(dom))
        return cls.from_diagram(cls.category.twist(dom))

    @classmethod
    def permutation(cls, xs: Sequence[int], doms: Sequence[Ty]) -> CMap:
        """
        A permutation encoded as boundary wiring, i.e. with no boxes.

        The default of :class:`~discopy.abc.SymmetricCategory` composes
        one :meth:`swap` per object, moving it across the ones before it,
        so it builds and composes as many maps of the whole boundary,
        quadratic in the number of wires, where a permutation of the
        boundary of a map is one involution.

        Parameters:
            xs : The permutation, sending input ``xs[j]`` to output ``j``.
            doms : The objects to permute, tensored as the domain.

        Example
        -------
        >>> from discopy.symmetric import Ty, CMap
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> permutation = CMap.permutation([2, 0, 1], [x, y, z])
        >>> assert permutation.boxes == () and permutation.cod == z @ x @ y
        >>> assert permutation == CMap.swap(x @ y, z)
        """
        xs, doms = list(xs), list(doms)
        if sorted(xs) != list(range(len(doms))):
            raise ValueError(
                f"{xs} is not a permutation of {len(doms)} objects")
        dom = cls.ob().tensor(*doms)
        cod = cls.ob().tensor(*(doms[x] for x in xs))
        starts = [0, *accumulate(len(objects) for objects in doms)]
        inputs = [
            starts[x] + offset for x in xs for offset in range(len(doms[x]))]
        edge = Permutation.from_transpositions(
            ((i, len(dom) + j) for j, i in enumerate(inputs)), 2 * len(dom))
        return cls(dom, cod, (), edge, check=False)

    cup_factory = classmethod(lambda cls, left, right: cls.from_box(
        cls.category.cup_factory(left, right)))
    cap_factory = classmethod(lambda cls, left, right: cls.from_box(
        cls.category.cap_factory(left, right)))

    @classmethod
    def cups(cls, left: Ty, right: Ty) -> CMap:
        """
        A cup between a type and its right adjoint: a box over a rigid host
        that is not compact, whose cups are planar and oriented, see
        :meth:`Hypergraph.cups`, and boundary wiring otherwise, a map being
        compact whatever hosts it.

        >>> from discopy import rigid, compact
        >>> x = rigid.Ty('x')
        >>> assert rigid.Cup(x, x.r).to_map().boxes == (rigid.Cup(x, x.r), )
        >>> x = compact.Ty('x')
        >>> assert compact.Cup(x, x.r).to_map().boxes == ()
        """
        if cls.is_planar_rigid():
            return cls.cup_factory(left, right) if len(left) == 1\
                else cls.from_diagram(cls.category.cups(left, right))
        assert_isinstance(left, Pregroup)
        assert_isinstance(right, Pregroup)
        left.assert_isadjoint(right)
        size = len(left)
        edge = Permutation.from_transpositions(
            ((i, size + size - 1 - i) for i in range(size)),
            2 * size)
        return cls(left @ right, cls.ob(), (), edge, check=False)

    @classmethod
    def caps(cls, left: Ty, right: Ty) -> CMap:
        """ A cap between a type and its left adjoint, see :meth:`cups`. """
        if cls.is_planar_rigid():
            return cls.cap_factory(left, right) if len(left) == 1\
                else cls.from_diagram(cls.category.caps(left, right))
        assert_isinstance(left, Pregroup)
        assert_isinstance(right, Pregroup)
        right.assert_isadjoint(left)
        size = len(left)
        edge = Permutation.from_transpositions(
            ((i, size + size - 1 - i) for i in range(size)),
            2 * size)
        return cls(cls.ob(), left @ right, (), edge, check=False)

    @classmethod
    def is_planar_rigid(cls) -> bool:
        """ Whether the host category is rigid but not compact, so that its
        cups and caps are boxes of the map rather than its wiring. """
        return issubclass(cls.category, RigidCategory)\
            and not issubclass(cls.category, CompactCategory)

    @classmethod
    def copy(cls, typ: Ty, n: int = 2) -> CMap:
        """ Copy is kept as a box: one input cannot wire to many outputs. """
        return cls.from_box(cls.category.copy(typ, n))

    @classmethod
    def merge(cls, typ: Ty, n: int = 2) -> CMap:
        """ Merge is kept as a box: many inputs cannot wire to one output. """
        return cls.from_box(cls.category.merge(typ, n))

    @classmethod
    def discard(cls, typ: Ty) -> CMap:
        """ Discard is kept as a box. """
        return cls.copy(typ, 0)

    @classmethod
    def noise(cls, typ: Ty) -> CMap:
        """ Noise is kept as a box. """
        return cls.merge(typ, 0)

    @classmethod
    def ev(cls, base: Ty, exponent: Ty, left: bool = True) -> CMap:
        """
        Evaluation is kept as an explicit box by default, or comes from the
        wiring of cups when the host category is rigid.
        """
        if issubclass(cls.category, RigidCategory):
            return super().ev(base, exponent, left)
        return cls.from_box(cls.category.ev(base, exponent, left))

    def curry(self, n: int = 1, left: bool = True) -> CMap:
        """
        Currying is kept as an explicit curry box by default, the more
        rigorous representation, or comes from the wiring of caps when the
        host category is rigid.

        Parameters:
            n : The number of objects to curry.
            left : Whether to curry on the left or right.

        >>> from discopy.compact import Ty, Box
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box("f", x @ y, z).to_map()
        >>> assert f.curry(left=False).uncurry(left=False) == f
        >>> f.curry(left=False).draw(
        ...     doctest="docs/_static/cmap/compact-curry.dot", show=False)

        .. graphviz:: /_static/cmap/compact-curry.dot
            :align: center
        """
        if issubclass(self.category, RigidCategory):
            return super().curry(n, left)
        if n < 0 or n > len(self.dom):
            raise ValueError(messages.NOT_SPLITTABLE.format(n, self.dom))
        if not n:
            return self
        return self.from_box(self.category.curry_factory(
            self.to_diagram(), n, left))

    def base_and_exponent(self, n: int, left: bool) -> tuple[Ty, Ty]:
        """
        The base and exponent that :meth:`uncurry` evaluates, read off the
        codomain as in the host category.

        Parameters:
            n : The number of objects to uncurry.
            left : Whether to uncurry on the left or right.
        """
        return self.category.base_and_exponent(self, n, left)

    l = property(lambda self: self.transpose(left=True))
    r = property(lambda self: self.transpose(left=False))

    def dagger(self) -> CMap:
        """
        Reverse a combinatorial map: swap the boundary, dagger each box in
        reverse order and conjugate the edges by the port relabeling.

        Boundary ports keep their order while each box block is reversed:
        the clockwise port order of a daggered box is the reversed clockwise
        order of the original.

        >>> from discopy.compact import Ty, Box
        >>> x, y = map(Ty, "xy")
        >>> f, g = Box('f', x, y @ y), Box('g', y @ y, x)
        >>> assert (f >> g).dagger().to_map() == (f >> g).to_map().dagger()
        >>> assert (f >> g).to_map().dagger().dagger() == (f >> g).to_map()
        """
        n, n_dom, n_cod = self.n_ports, len(self.dom), len(self.cod)
        boxes = tuple(box.dagger() for box in reversed(self.boxes))
        dom_mapping = list(range(n - n_dom, n))
        box_mapping = list(reversed(range(n_cod, n - n_dom)))
        cod_mapping = list(range(n_cod))
        mapping = dom_mapping + box_mapping + cod_mapping
        edges = self.edges.conjugate(Permutation(mapping))
        return type(self)(
            self.cod, self.dom, boxes, edges,
            loops=self.loops, check=False)

    @classmethod
    def spiders(
            cls, n_legs_in: int, n_legs_out: int,
            typ: Ty, phases=None) -> CMap:
        """
        Spiders are kept as boxes, including their phase data.

        Example
        -------
        >>> from discopy.tensor import CMap, Dim, Tensor
        >>> assert CMap.spiders(1, 2, Dim(2, 3)).eval().is_close(
        ...     Tensor.spiders(1, 2, Dim(2, 3)))
        """
        return cls.from_box(cls.category.spiders(
            n_legs_in, n_legs_out, typ, phases))

    @unbiased
    def then(self, other: CMap) -> CMap:
        """
        Compose maps by gluing output ports to input ports.

        Closed components created by gluing are retained in :attr:`loops`.

        >>> from discopy.compact import Ty, CMap
        >>> x = Ty("x")
        >>> scalar = CMap.caps(x.r, x) >> CMap.cups(x.r, x)
        >>> scalar.boxes
        ()
        >>> scalar.loops == (x,)
        True
        """
        if not self.cod == other.dom:
            raise AxiomError(messages.TYPE_ERROR.format(other.dom, self.cod))
        dom, cod = self.dom, other.cod
        boxes = self.boxes + other.boxes
        edge = self.edges.tensor(other.edges)
        ports = self.ports + other.ports
        glue = Permutation.id(self.n_ports - len(self.cod)).tensor(
            Permutation.swap(len(self.cod), len(other.dom)),
            Permutation.id(other.n_ports - len(other.dom)))
        edge, new_scalars = self.splice(
            edge, glue, ports)
        loops = self.loops + other.loops + new_scalars
        return type(self)(
            dom, cod, boxes, edge, loops=loops, check=False)

    def trace(self, n: int = 1, left: bool = False) -> CMap:
        """
        Trace boundary wires by splicing the selected inputs and outputs, a
        map being compact whatever hosts it, except over a planar traced
        category, i.e. traced but not symmetric, whose trace is not wiring:
        there it is the trace of the category, one wire at a time, see
        :meth:`explicit_trace` and :meth:`Hypergraph.trace`.

        Parameters:
            n : The number of wires to trace.
            left : Whether to trace the leftmost rather than rightmost wires.

        >>> from discopy import feedback
        >>> f = feedback.Box('f', feedback.Ty('x'), feedback.Ty('x'))
        >>> assert f.to_map().trace().boxes == (f.trace(), )
        """
        if n < 0:
            raise ValueError(messages.NOT_SPLITTABLE.format(n, self.dom))
        if not n:
            return self
        assert_istraceable(self, n, left)
        if issubclass(self.category, TracedCategory)\
                and not issubclass(self.category, SymmetricCategory):
            result = self
            for _ in range(n):
                result = result.explicit_trace(left)
            return result

        if left:
            dom, cod = self.dom[n:], self.cod[n:]
            traced_inputs = range(n)
            traced_outputs = range(
                self.n_ports - len(self.cod),
                self.n_ports - len(self.cod) + n)
        else:
            dom, cod = self.dom[:-n], self.cod[:-n]
            traced_inputs = range(len(dom), len(self.dom))
            traced_outputs = range(self.n_ports - n, self.n_ports)

        glue = Permutation.from_transpositions(
            zip(traced_inputs, traced_outputs), self.n_ports)
        edge, new_scalars = self.splice(
            self.edges, glue, self.ports)
        loops = self.loops + new_scalars
        return type(self)(
            dom, cod, self.boxes, edge, loops=loops, check=False)

    def tensor(self, *others: CMap) -> CMap:
        """
        Tensor product given by disjoint union of ``n`` maps.

        The ports of a map come as ``dom``, then the ports of its boxes,
        then ``cod``, so the disjoint union is the concatenation of the
        factors read back in that order.

        >>> from discopy.compact import Ty, Box
        >>> x = Ty('x')
        >>> f, g, h = (Box(name, x, x).to_map() for name in "fgh")
        >>> assert f.tensor(g, h) == f @ g @ h
        """
        if not others:
            return self
        factors = (self, ) + others
        n_box_ports = [
            factor.n_ports - len(factor.dom) - len(factor.cod)
            for factor in factors]
        mapping, i, j = [], 0, sum(len(factor.dom) for factor in factors)
        k = j + sum(n_box_ports)
        boxes, loops = [], []
        for factor, n_box in zip(factors, n_box_ports):
            mapping += list(range(i, i + len(factor.dom)))
            mapping += list(range(j, j + n_box))
            mapping += list(range(k, k + len(factor.cod)))
            i, j, k = i + len(factor.dom), j + n_box, k + len(factor.cod)
            boxes += factor.boxes
            loops += factor.loops
        edges = self.edges.tensor(*(other.edges for other in others))
        return type(self)(
            self.dom.tensor(*(other.dom for other in others)),
            self.cod.tensor(*(other.cod for other in others)),
            tuple(boxes), edges.conjugate(Permutation(mapping)),
            loops=tuple(loops), check=False)

    def interchange(self, i: int, j: int) -> CMap:
        """
        Interchange boxes at indices ``i`` and ``j``.

        The edges permutation is relabeled so that ports follow the canonical
        order induced by the new box order. Unlike
        :meth:`Diagram.interchange`, the boxes need not commute: the result
        is the same map with its boxes out of order, which
        :meth:`topological_order` can put back.

        >>> from discopy.compact import Ty, Box
        >>> x, y = map(Ty, "xy")
        >>> f, g = Box("f", x, x), Box("g", y, y)
        >>> cmap = f.to_map() @ g.to_map()
        >>> cmap.interchange(0, 1).boxes == (g, f)
        True
        >>> assert not (f.to_map() >> f.to_map()).interchange(
        ...     0, 1).is_topologically_ordered
        """
        order = list(range(len(self.boxes)))
        order[i], order[j] = order[j], order[i]
        return self.reorder(order)

    def plug_input(
            self, input_index: int, box: Box,
            cod: C0, root_index: int = 0) -> CMap:
        """
        Plug an input boundary and the output root into a new box.

        If ``self : A @ x -> y`` and ``box : y -> z @ x``, then
        ``self.plug_input(i, box, z)`` removes the ``i``-th input, wires the
        old output to the domain of ``box``, wires the removed input to the
        non-root output of ``box``, and leaves ``root_index`` as the new root.

        The new box comes last but wires back to the domain of an earlier box,
        so this can introduce a cycle, which needs a traced category.

        Raises:
            ValueError : If the map or box does not have the required arity,
                or either index is out of range.
        """
        assert_isinstance(box, self.category)
        if len(self.cod) != 1 or len(box.dom) != 1 or len(box.cod) != 2:
            raise ValueError(messages.NOT_PLUGGABLE.format(box, self.cod))
        if root_index not in [0, 1]:
            raise ValueError(messages.NOT_AN_INDEX.format(2, root_index))
        if input_index < 0 or input_index >= len(self.dom):
            raise ValueError(messages.NOT_AN_INDEX.format(
                len(self.dom), input_index))

        old_input, old_output = input_index, self.n_ports - 1
        new_dom = self.ob()
        for i, obj in enumerate(self.dom):
            if i != input_index:
                new_dom = new_dom @ obj
        boxes = self.boxes + (box, )

        mapping, new_index = {}, 0
        for i in range(len(self.dom)):
            if i != old_input:
                mapping[i] = new_index
                new_index += 1
        for i in range(len(self.dom), self.n_ports - len(self.cod)):
            mapping[i] = new_index
            new_index += 1

        box_dom = new_index
        box_outputs = tuple(
            new_index + 1 + len(box.cod) - i - 1
            for i in range(len(box.cod)))
        box_root = box_outputs[root_index]
        box_parameter = box_outputs[1 - root_index]
        new_output = new_index + 3

        edge_pairs = []
        for i, j in enumerate(self.edges):
            if i < j and i not in [old_input, old_output]\
                    and j not in [old_input, old_output]:
                edge_pairs.append((mapping[i], mapping[j]))

        input_partner = self.edges[old_input]
        output_partner = self.edges[old_output]
        if input_partner == old_output:
            edge_pairs.append((box_parameter, box_dom))
        else:
            edge_pairs.append((mapping[input_partner], box_parameter))
            edge_pairs.append((mapping[output_partner], box_dom))
        edge_pairs.append((box_root, new_output))
        edge = Permutation.from_transpositions(edge_pairs, new_output + 1)

        return type(self)(
            new_dom, cod, boxes, edge,
            loops=self.loops)

    def explicit_trace(self, left: bool = False) -> CMap:
        """
        The trace of a map with explicit boxes (trace, cup or cap).

        Parameters:
            left : Whether to trace on the left or right.

        Note
        ----
        When ``category.trace_factory`` is a class, e.g. for symmetric
        diagrams, then the result is just one big trace box wrapped up as a
        map. Otherwise it is a class method, e.g. for compact diagrams, in
        which case we use it to introduce cup and cap boxes.
        """
        type(self).assert_istraced()
        factory = self.category.trace_factory
        if isclass(factory):
            return self.from_box(factory(self.to_diagram(), left))
        return factory.__func__(type(self), self, left)

    def make_monogamous(self) -> CMap:
        """
        Introduce cup and cap boxes to make self :attr:`is_monogamous`,
        i.e. so that every wire connects a positive and a negative port.

        The boxes come from ``category.cup_factory`` and
        ``category.cap_factory``, so this needs a rigid category.

        Example
        -------
        >>> from discopy.compact import Ty, Cup, Cap, CMap
        >>> x = Ty("x")
        >>> assert CMap.cups(x, x.r).make_monogamous()\\
        ...     == CMap.from_box(Cup(x, x.r))
        >>> assert CMap.caps(x.r, x).make_monogamous()\\
        ...     == CMap.from_box(Cap(x.r, x))
        """
        type(self).assert_isrigid()
        ports = self.ports
        for i, j in enumerate(self.edges):
            if i > j or ports[i].kind.is_positive\
                    != ports[j].kind.is_positive:
                continue
            source, target = ports[i].obj, ports[j].obj
            if ports[i].kind.is_positive:
                box = self.category.cup_factory(source, target)
                boxes = self.boxes + (box, )
                insert = self.n_ports - len(self.cod)
                box_wires = [(insert, i), (insert + 1, j)]
            else:
                box = self.category.cap_factory(source, target)
                boxes = (box, ) + self.boxes
                insert = len(self.dom)
                box_wires = [(insert + 1, i), (insert, j)]
            shift = lambda p: p if p < insert else p + 2
            edges = Permutation.from_transpositions(
                [(a, shift(b)) for a, b in box_wires]
                + [(shift(a), shift(b)) for a, b in enumerate(self.edges)
                   if a < b and (a, b) != (i, j)],
                self.n_ports + 2)
            return type(self)(
                self.dom, self.cod, boxes, edges,
                loops=self.loops, check=False).make_monogamous()
        return self

    def make_causal(self) -> CMap:
        """
        Make self :attr:`is_causal`, i.e. so that every wire points forward
        and there are no loops. Boxes that are merely out of order are put
        back in order with :meth:`topological_order`; only a wire that closes
        a cycle is cut into a trace, which needs a traced category.

        Example
        -------
        >>> from discopy.feedback import Ty, Box, Trace, CMap
        >>> f = Box("f", Ty("x"), Ty("x"))
        >>> assert f.to_map().trace().make_causal()\\
        ...     == CMap.from_box(Trace(f))
        """
        if not self.is_monogamous:
            return self.make_monogamous().make_causal()
        if self.is_acyclic:
            return self.topological_order()
        type(self).assert_istraced()

        ports, box_edges = self.ports, list(self.box_edges)

        def depth(port):
            return int(ports[port].depth + (
                .5 if ports[port].kind.is_output else -.5))

        successors = [[] for _ in self.boxes]
        for source, target in box_edges:
            successors[depth(source)].append(depth(target))

        def closes_a_cycle(source, target):
            seen, todo = set(), [depth(target)]
            while todo:
                seen.add(todo[-1])
                todo = todo[:-1] + [
                    box for box in successors[todo[-1]] if box not in seen]
            return depth(source) in seen

        cuts = [
            (i, j, ports[i].obj) for i, j in box_edges
            if depth(i) >= depth(j) and closes_a_cycle(i, j)]

        n_traces = len(self.loops) + len(cuts)
        shift = lambda p: p if p < len(self.dom) else p + n_traces
        typ = self.ob().tensor(*self.loops, *(obj for _, _, obj in cuts))
        boundary = [
            (len(self.dom) + i, self.n_ports + n_traces + i)
            for i in range(len(self.loops))]
        for i, (source_port, target_port, _) in enumerate(
                cuts, len(self.loops)):
            boundary += [
                (len(self.dom) + i, shift(target_port)),
                (shift(source_port), self.n_ports + n_traces + i)]
        cut_wires = {
            tuple(sorted(wire[:2])) for wire in cuts}
        edges = Permutation.from_transpositions(
            boundary + [
                (shift(a), shift(b)) for a, b in enumerate(self.edges)
                if a < b and (a, b) not in cut_wires],
            self.n_ports + 2 * n_traces)
        result = type(self)(
            self.dom @ typ, self.cod @ typ, self.boxes, edges,
            check=False)
        for _ in range(n_traces):
            result = result.explicit_trace()
        return result

    def to_compact(self) -> CMap:
        """
        Open every curry bubble into its argument followed by the dagger of
        :meth:`ev`, traced over the curried wires: the map is decoded and
        the image of each box is glued back, like :meth:`from_diagram`.

        Example
        -------
        >>> from discopy.closed import Ty, Box, CMap
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box("f", x @ y, z)
        >>> assert f.to_map().curry().to_compact()\\
        ...     == (f.to_map() >> CMap.ev(z, y).dagger()).trace()
        """
        curry_factory = self.category.curry_factory
        if not any(isinstance(box, curry_factory) for box in self.boxes):
            return self
        functor = self.functor(
            ob_map=lambda typ: typ, ar_map=type(self).from_box,
            dom=self.category, cod=type(self))

        def image(box):
            if not isinstance(box, curry_factory):
                return functor(box)
            exponent = box.cod.exponent
            return (type(self).from_diagram(box.arg).to_compact()
                    >> self.ev(box.cod.base, exponent, box.left).dagger()
                    ).trace(len(exponent), left=not box.left)

        diagram = self.to_diagram()
        return type(self).from_glued(diagram.dom, diagram.cod, [
            (image(box), offset)
            for layer in diagram.inside
            for box, offset in layer.boxes_and_offsets])

    def eval(self, *args, **params):
        """
        Evaluate the map directly with the ``eval`` of the host category,
        e.g. contract a tensor map in a single ``einsum`` call, see
        :meth:`discopy.tensor.Diagram.eval`.
        """
        return self.category.eval(self, *args, **params)

    def to_diagram(self) -> Diagram:
        """
        Downgrade to a diagram directly, preserving box orientation.

        This is where a map has to be a morphism of :attr:`category`: its
        cups, caps and traces are made explicit with :meth:`make_monogamous`
        and :meth:`make_causal`, which raise unless the category is rigid
        resp. traced, then its boxes are put in the order in which they are
        applied with :meth:`topological_order`.

        >>> from discopy.monoidal import Ty, Box, CMap
        >>> x = Ty("x")
        >>> f = Box("f", x, x)
        >>> CMap(Ty(), Ty(), (f, ), (1, 0)).to_diagram()
        Traceback (most recent call last):
        ...
        discopy.utils.AxiomError: monoidal.Diagram has no traces for the \
cycles of this map.

        What remains is decoded like in :meth:`Hypergraph.to_diagram`: we scan
        the currently open wires from left to right, for each box we gather
        its domain wires at the position of the first one with a single
        permutation, see :meth:`discopy.python.finset.Permutation.gather`,
        apply the box, and replace the consumed domain labels by the box
        codomain labels.

        >>> from discopy.compact import Ty, Box, CMap
        >>> x, y, z = map(Ty, "xyz")
        >>> cmap = Box("f", x, y).to_map()
        >>> cmap.to_diagram().to_map() == cmap
        True
        >>> print(CMap.permutation([2, 0, 1], x @ y @ z).to_diagram())
        Permutation(x @ y @ z, [2, 0, 1])

        A box with no domain has no wire to gather: it goes in the gap of the
        scan that its face reaches, i.e. the cycle of :attr:`faces` through
        the corner above it. Walking that face back to the wire on its left,
        or on to the wire on its right, the turns add up to half a turn when
        that wire is its neighbour in a planar drawing: a corner between two
        wires turns by half a turn, a corner around a state or an effect by
        minus half a turn and the side of a box not at all. When the two
        walks disagree, it goes next to the wire that its first reader reads
        beside it. When neither reaches the scan yet, the box waits for its
        first reader if it is a cap or the category has no swaps, so that
        the other boxes keep their order, and it goes on the right of the
        open wires when nothing says where, e.g. when the map is not planar.

        >>> from discopy.monoidal import Ty, Box
        >>> x, y = Ty('x'), Ty('y')
        >>> s, e = Box('s', Ty(), y), Box('e', y @ x, Ty())
        >>> print((s @ x).to_map().to_diagram())
        s @ x
        >>> print((s @ x >> e).to_map().to_diagram())
        s @ x >> e
        """
        if not self.is_causal:
            return self.make_causal().to_diagram()

        edge_wire = {}
        for i, j in enumerate(self.edges):
            if i <= j:
                edge_wire[i] = edge_wire[j] = len(edge_wire) // 2

        ports, walk = self.ports, self.faces
        back, first_output = walk.dagger(), self.n_ports - len(self.cod)
        right_side = None if not self.cod\
            else len(self.dom) - 1 if self.dom else first_output
        diagram = self.category.id(self.dom)
        scan = [edge_wire[i] for i in range(len(self.dom))]
        outputs = [edge_wire[i] for i in range(first_output, self.n_ports)]
        placed, waiting = set(), []
        symmetric = issubclass(self.category, SymmetricCategory)
        caps = getattr(self.category, "cap_factory", None)
        caps = caps if isclass(caps) else ()

        def permute(perm):
            if not symmetric:
                raise AxiomError(messages.NOT_SYMMETRIC.format(
                    factory_name(self.category)))
            return self.category.from_permutation(perm, diagram.cod)

        def turn(port):
            """ The turn at the corner before a port, in half turns, or
            ``None`` when that corner is above the scan. """
            kind, i = ports[port].kind, ports[port].i
            if kind.is_boundary:
                return 1 if kind == PortKind.OUTPUT and i else None
            depth = int(ports[port].depth + (-.5 if kind.is_input else .5))
            box = self.boxes[depth]
            this, that = (box.dom, box.cod) if kind.is_input\
                else (box.cod, box.dom)
            if depth in placed:
                return None
            if i > 0 if kind.is_input else i < len(this) - 1:
                return 1
            return 0 if that else -1

        def from_the_left(corner):
            total, port = 0, back[corner]
            while port != corner:
                if edge_wire[port] in scan:
                    return scan.index(edge_wire[port]) + 1 if total == 1\
                        and ports[port].kind.is_positive else None
                if port == first_output:
                    return 0 if total == 0 else None
                if turn(port) is None:
                    return None
                total, port = total + turn(port), back[port]
            return None

        def from_the_right(corner):
            total, port = 0, walk[corner]
            while port != corner:
                if port == right_side:
                    return len(scan) if total == 0 else None
                if turn(port) is None:
                    return None
                total += turn(port)
                if edge_wire[port] in scan:
                    return scan.index(edge_wire[port]) if total == 1\
                        and ports[port].kind.is_negative else None
                port = walk[port]
            return None

        def wires(depth):
            box, box_ports = self.boxes[depth], self._box_port_indices[depth]
            return [edge_wire[i] for i in box_ports[:len(box.dom)]], [
                edge_wire[i] for i in reversed(box_ports[len(box.dom):])]

        def gaps(depth):
            box_ports = self._box_port_indices[depth]
            return {from_the_left(box_ports[0]), from_the_right(
                box_ports[0])} - {None} if box_ports else {len(scan)}

        def beside(depth, reads):
            _, cod_wires = wires(depth)
            if cod_wires[0] in reads[1:]:
                before = reads[reads.index(cod_wires[0]) - 1]
                if before in scan:
                    return scan.index(before) + 1
            if cod_wires[-1] in reads[:-1]:
                after = reads[reads.index(cod_wires[-1]) + 1]
                if after in scan:
                    return scan.index(after)
            return None

        def reader(depth):
            _, cod_wires = wires(depth)
            return next((
                wires(later)[0] for later in range(depth + 1, len(self.boxes))
                if set(cod_wires).intersection(wires(later)[0])), outputs)

        def where(depth):
            found = gaps(depth)
            near = None if len(found) == 1 else beside(depth, reader(depth))
            return min(found, default=len(scan)) if near is None else near

        def force(depth):
            while not gaps(depth) and beside(depth, reader(depth)) is None\
                    and waiting[0] != depth:
                force(waiting[0])
            waiting.remove(depth)
            place(depth, where(depth))

        def place(depth, offset=None):
            nonlocal diagram, scan
            dom_wires, cod_wires = wires(depth)
            positions = [scan.index(wire_id) for wire_id in dom_wires]
            if offset is None:
                offset = positions[0] if positions else len(scan)
                offset -= sum(j < offset for j in positions)
            perm = Permutation.gather(positions, len(scan), offset)
            if not perm.is_identity:
                diagram >>= permute(perm)
                scan = [scan[j] for j in perm]
            scan = scan[:offset] + cod_wires + scan[offset + len(dom_wires):]
            diagram >>= diagram.cod[:offset] @ self.boxes[depth]\
                @ diagram.cod[offset + len(dom_wires):]
            placed.add(depth)

        def settle():
            ready = [depth for depth in waiting if gaps(depth)]
            while ready:
                waiting.remove(ready[0])
                place(ready[0], where(ready[0]))
                ready = [depth for depth in waiting if gaps(depth)]

        for depth, box in enumerate(self.boxes):
            reads, _ = wires(depth)
            for other in [other for other in waiting
                          if set(reads).intersection(wires(other)[1])]:
                if other in waiting:
                    force(other)
            if box.dom or symmetric and not isinstance(box, caps):
                place(depth, None if box.dom else where(depth))
            else:
                waiting.append(depth)
            settle()
        while waiting:
            force(waiting[0])

        perm = Permutation(
            [scan.index(wire_id) for wire_id in outputs], len(scan))
        if not perm.is_identity:
            diagram >>= permute(perm)
        return diagram

    def to_hypergraph(self):
        """
        Forget orientation and return the underlying bijective hypergraph
        given by the edge permutation. See documentation of
        :func:``Hypergraph.from_map`` for an example.
        """
        return hypergraph.Hypergraph[self.category].from_map(self)

    def to_dot(
            self, engine="dot", seed=None, graph_attr=None,
            port_indices=False) -> str:
        """
        Encode the combinatorial map as Graphviz DOT.

        The drawing has HTML-table nodes for the boundary interfaces and for
        each box, with one table port for each object in the signature, and
        one direct edge per 2-cycle of ``edges``.

        Parameters:
            engine : The Graphviz layout engine.
            seed : An optional Graphviz layout seed.
            graph_attr : Additional graph attributes.
            port_indices : Whether to display port indices.

        >>> from discopy.compact import Ty, CMap
        >>> CMap.id(Ty("x")).to_dot().startswith("graph cmap")
        True
        """
        attrs = {
            "layout": engine,
            "rankdir": "TB",
            "overlap": "false",
            "splines": "true",
            "outputorder": "edgesfirst",
            "bgcolor": "white",
            "fontname": "DejaVu Sans",
            "margin": "0.04",
        } | (graph_attr or {})
        if seed is not None:
            attrs["start"] = str(seed)

        class Html:
            def __init__(self, value):
                self.value = value

        def escape(value):
            return str(value).replace("\\", "\\\\").replace('"', r'\"')

        def escape_html(value):
            return str(value).replace("&", "&amp;").replace(
                "<", "&lt;").replace(">", "&gt;").replace(
                    '"', "&quot;")

        def attr_string(attributes):
            return ", ".join(
                f'{key}=<{value.value}>' if isinstance(value, Html)
                else f'{key}="{escape(value)}"'
                for key, value in sorted(attributes.items()))

        def boundary_label(port_index):
            return f"{port_index}" if port_indices else ""

        def boundary_cell(port_index, port):
            tooltip = escape_html(f"{port.kind} {port.i}: {port.obj}")
            return (
                f'<TD PORT="p{port_index}" TOOLTIP="{tooltip}" '
                f'BORDER="0" CELLPADDING="4">'
                f'{escape_html(boundary_label(port_index))}</TD>')

        def boundary_table(port_indices):
            return (
                '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR>'
                + "".join(
                    boundary_cell(port_index, self.ports[port_index])
                    for port_index in port_indices)
                + "</TR></TABLE>")

        def port_cell(port_index, port, colspan, width):
            tooltip = escape_html(
                f"{port.kind} {port.i}: {port.obj} ({port.side})")
            text = escape_html(port_index) if port_indices else ""
            cellpadding = 2 if port_indices else 0
            height = 18 if port_indices else 0
            fixedsize = ' FIXEDSIZE="TRUE"' if port_indices else ""
            return (
                f'<TD PORT="p{port_index}" TOOLTIP="{tooltip}" '
                f'BORDER="0" CELLPADDING="{cellpadding}" '
                f'COLSPAN="{colspan}" WIDTH="{width}" '
                f'HEIGHT="{height}"{fixedsize}>{text}</TD>')

        def port_row(port_indices, grid, box_width):
            colspan = grid // len(port_indices)
            width = round(box_width / len(port_indices))
            return "<TR>" + "".join(
                port_cell(
                    port_index, self.ports[port_index], colspan, width)
                for port_index in port_indices) + "</TR>"

        def box_table(vertex, box):
            box_ports = self._box_port_indices[vertex]
            dom_ports = box_ports[:len(box.dom)]
            cod_ports = tuple(reversed(box_ports[len(box.dom):]))
            dom_arity, cod_arity = len(dom_ports), len(cod_ports)
            grid = lcm(dom_arity or 1, cod_arity or 1)
            box_width = 18 * max(dom_arity, cod_arity, 1)
            rows = []
            if dom_ports:
                rows.append(port_row(dom_ports, grid, box_width))
            box_label = getattr(box, "drawing_name", box.name)
            rows.append(
                f'<TR><TD BORDER="1" CELLPADDING="6" '
                f'COLSPAN="{grid}" WIDTH="{box_width}">'
                f'{escape_html(box_label)}</TD></TR>')
            if cod_ports:
                rows.append(port_row(cod_ports, grid, box_width))
            return (
                '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                + "".join(rows) + "</TABLE>")

        node_attrs = {
            "color": "black", "fontname": "DejaVu Sans", "fontsize": "12",
            "margin": "0", "shape": "plain"}
        edge_attrs = {
            "color": "black", "fontname": "DejaVu Sans", "fontsize": "9",
            "headclip": "true", "penwidth": "1.4", "tailclip": "true"}
        lines = [
            "graph cmap {",
            f"  graph [{attr_string(attrs)}];",
            f"  node [{attr_string(node_attrs)}];",
            f"  edge [{attr_string(edge_attrs)}];",
        ]

        port_nodes = {}
        for vertex in range(len(self.boxes)):
            box = self.boxes[vertex]
            attributes = {"label": Html(box_table(vertex, box))}
            lines.append(
                f"  v{vertex} [{attr_string(attributes)}];")
            for port_index in self._box_port_indices[vertex]:
                compass = "n" if self.ports[
                    port_index].kind == "dom" else "s"
                port_nodes[port_index] = (
                    f"v{vertex}:p{port_index}:{compass}")
        input_ports = [
            i for i, port in enumerate(self.ports)
            if port.kind == PortKind.INPUT]
        output_ports = [
            i for i, port in enumerate(self.ports)
            if port.kind == PortKind.OUTPUT]
        for name, ports, compass in [
                (PortKind.INPUT, input_ports, "s"),
                (PortKind.OUTPUT, output_ports, "n")]:
            if not ports:
                continue
            attributes = {"label": Html(boundary_table(ports))}
            lines.append(f"  {name} [{attr_string(attributes)}];")
            for port_index in ports:
                port_nodes[port_index] = f"{name}:p{port_index}:{compass}"

        for rank, name, ports in [
                ("min", "input", input_ports),
                ("max", "output", output_ports)]:
            if ports:
                lines.append(f"  {{ rank={rank}; {name}; }}")

        for i, loop in enumerate(self.loops):
            attributes = {
                "label": "",
                "width": "0.08",
                "height": "0.08",
                "shape": "point",
                "tooltip": f"loop {i}: {loop}"}
            lines.append(f"  loop{i} [{attr_string(attributes)}];")
            attributes = {"len": "0.85", "label": loop}
            lines.append(
                f"  loop{i} -- loop{i} "
                f"[{attr_string(attributes)}];")

        def node_name(port_index):
            return port_nodes[port_index]

        def port_label(port_index):
            return self.ports[port_index].obj

        def edge_labels(left, right):
            left_label, right_label = port_label(left), port_label(right)
            if left_label == right_label:
                return {"label": left_label}
            return {"taillabel": left_label, "headlabel": right_label}

        for i, j in enumerate(self.edges):
            if i < j:
                attributes = {
                    "len": "0.85", "labeldistance": "1.6"} | edge_labels(i, j)
                lines.append(
                    f'  {node_name(i)} -- {node_name(j)} '
                    f'[{attr_string(attributes)}];')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def draw(
            self, path=None, doctest=None, engine="dot", format=None,
            seed=None, show=None, graph_attr=None, port_indices=False,
            block=True, tol=20):
        """
        Draw as a combinatorial map using Graphviz.

        This is intended for map-like pictures rather than the usual DisCoPy
        box-and-wire drawing.

        If ``path`` ends in ``.dot`` or ``.gv``, write DOT source. Otherwise,
        render with Graphviz. When ``show`` is true, display the rendered graph
        in a matplotlib window.

        Parameters:
            path : The output path, or ``None`` to display the map.
            engine : The Graphviz layout engine.
            format : The rendered format, inferred from ``path`` by default.
            seed : An optional Graphviz layout seed.
            show : Whether to display the rendered image.
            graph_attr : Additional Graphviz graph attributes.
            port_indices : Whether to display port indices.
            block : Whether displaying blocks execution.

        Scalar loops are drawn as dots with a loop, but the combinatorial map
        structure does not let us retain inclusion of such loops:

        >>> from discopy.compact import Ty, CMap
        >>> x, y, z = map(Ty, "xyz")
        >>> (CMap.caps((x @ y).r, x @ y) >> CMap.cups((x @ y).l, x @ y)).draw(
        ...     doctest="docs/_static/cmap/scalar-loop.dot", show=False)

        .. graphviz:: /_static/cmap/scalar-loop.dot
            :align: center
        """
        dot = self.to_dot(
            engine=engine, seed=seed, graph_attr=graph_attr,
            port_indices=port_indices)

        from discopy.drawing import backend
        path, compare = backend.doctest_or_path(path, doctest)
        show = show if show is not None else path is None
        if path is not None:
            path_str = str(path)
            suffix = path_str.rsplit(".", 1)[-1].lower()\
                if "." in path_str else ""
            if suffix in ["dot", "gv"]:
                def save(actual_path):
                    with open(
                            actual_path, "w", encoding="utf-8",
                            newline="\n") as stream:
                        stream.write(dot)
                if compare:
                    backend.save_and_compare(path, save, tol=tol)
                else:
                    save(path)
                return

        executable = shutil.which(engine) or shutil.which("dot")
        if executable is None:
            raise RuntimeError(
                f"Graphviz executable {engine!r} was not found.")

        if path is not None:
            output_format = format or suffix or "svg"

            def save(actual_path):
                subprocess.run(
                    [executable, f"-T{output_format}", "-o", actual_path],
                    input=dot.encode(), check=True)
            if compare:
                backend.save_and_compare(path, save, tol=tol)
            else:
                save(path)
        if not show:
            return

        png = subprocess.run(
            [executable, "-Tpng"], input=dot.encode(),
            capture_output=True, check=True).stdout
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        image = mpimg.imread(BytesIO(png), format="png")
        height, width = image.shape[:2]
        figsize = (max(width / 100, 1), max(height / 100, 1))
        figure, axis = plt.subplots(figsize=figsize, facecolor="white")
        axis.imshow(image)
        axis.axis("off")
        figure.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.show(block=block)
        return
