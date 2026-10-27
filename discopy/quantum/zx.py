# -*- coding: utf-8 -*-

"""
ZX-calculus diagrams.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Bubble
    Swap
    Spider
    Z
    Y
    X
    Scalar
    Number
    Hypergraph
    Rewriter
"""

from __future__ import annotations

import cmath
from collections import Counter
from collections.abc import Callable
from fractions import Fraction
from functools import reduce
from heapq import heappop, heappush
from itertools import combinations
from math import pi
from operator import index
from typing import NamedTuple

from discopy import (
    cat, frobenius, hypergraph, messages, rigid, tensor, quantum)
from discopy.cat import factory
from discopy.quantum.circuit import qubit, Circuit
from discopy.quantum.gates import (
    Bra, Ket, Rz, Rx, U1, CX, CZ, Controlled, format_number)
from discopy.quantum.gates import Scalar as GatesScalar
from discopy.rigid import Sum, Nat
from discopy.tensor import Dim, backend
from discopy.utils import assert_isatomic, factory_name, unbiased


@factory
class Diagram(tensor.Diagram[complex]):
    """
    A ZX diagram is a tensor diagram with qubits as wires: :meth:`eval` is
    the standard interpretation, where each wire is ``Dim(2)``, each spider
    the tensor of its phase in the eigenbasis of its colour, :code:`H` the
    Hadamard matrix and a scalar its number.

    Example
    -------
    >>> assert Z(1, 1, .5).eval().is_close(
    ...     tensor.Tensor[complex]([1, 0, 0, -1], Dim(2), Dim(2)))
    >>> assert (H >> H).eval().is_close(tensor.Tensor[complex].id(Dim(2)))
    >>> assert X(1, 1, .25).eval().is_close((H >> Z(1, 1, .25) >> H).eval())

    The copy and discard are the Z spiders, which :code:`H` commutes with
    neither:

    >>> assert not H.is_deterministic and not H.is_affine
    """
    ob = Nat

    @classmethod
    def dim(cls, typ: Nat) -> Dim:
        """ The wires of a ZX diagram are qubits, i.e. ``Dim(2)`` each. """
        return Dim(2) ** len(typ)

    @staticmethod
    def swap(left, right):
        left = left if isinstance(left, Nat) else Nat(left)
        right = right if isinstance(right, Nat) else Nat(right)
        return tensor.Diagram.swap.__func__(Diagram, left, right)

    @staticmethod
    def cup_factory(left, right):
        del left, right
        return Z(2, 0)

    @staticmethod
    def cap_factory(left, right):
        """ The cap of a ZX diagram, the dagger of its cup. """
        del left, right
        return Z(0, 2)

    @staticmethod
    def spider_factory(n_legs_in, n_legs_out, typ, phase=None):
        """
        The spider of a ZX diagram on one qubit, a :class:`Z` spider, so that
        a functor into ZX diagrams rebuilds a spider as the cups and caps.

        >>> assert Diagram.spiders(1, 2, Nat(2)) == Z(1, 2) @ Z(1, 2)\\
        ...     >> Nat(1) @ SWAP @ Nat(1)
        """
        assert_isatomic(typ, Nat)
        return Z(n_legs_in, n_legs_out, phase or 0)

    def grad(self, var, **params) -> rigid.Sum:
        """
        Gradient with respect to `var`, a formal sum by the product rule
        with the gradient of each spider given by :meth:`Spider.grad`.

        Parameters
        ----------
        var : sympy.Symbol
            Differentiated variable.

        Examples
        --------
        >>> from sympy.abc import phi  # doctest: +EXTRA
        >>> assert Z(1, 1, phi).grad(phi) == scalar(1j * pi) @ Z(1, 1, phi)\\
        ...     + scalar(-1j * pi) @ Z(1, 1, phi + .5)
        """
        return super().grad(var, **params)

    def split_scalar(self) -> tuple[Number, Diagram]:
        """
        The product of the :class:`Scalar` boxes, as an exact :class:`Number`,
        and the diagram without them.

        >>> diagram = Z(1, 1) @ scalar(2 ** .5) @ scalar(-1)
        >>> number, rest = diagram.split_scalar()
        >>> assert number == Number(1, Fraction(1, 2)) and rest == Z(1, 1)
        """
        numbers = [Number.from_complex(box.data)
                   for box in self.boxes if isinstance(box, Scalar)]
        if not numbers:
            return ONE, self
        return reduce(lambda x, y: x * y, numbers), frobenius.Functor(
            ob_map=lambda typ: typ, dom=Diagram, cod=Diagram,
            ar_map=lambda box: Id() if isinstance(box, Scalar) else box)(self)

    def to_hypergraph(self) -> Hypergraph:
        """ The :class:`Hypergraph` of a ZX diagram, with its scalar. """
        number, rest = self.split_scalar()
        return super(Diagram, rest).to_hypergraph().scaled(number)

    def to_pyzx(self):
        """
        Returns a :class:`pyzx.Graph`.

        >>> bialgebra = Z(1, 2, .25) @ Z(1, 2, .75)\\
        ...     >> Id(1) @ SWAP @ Id(1) >> X(2, 1, .5) @ X(2, 1, .5)
        >>> graph = bialgebra.to_pyzx()  # doctest: +EXTRA
        >>> assert len(graph.vertices()) == 8
        >>> assert (graph.inputs(), graph.outputs()) == ((0, 1), (6, 7))
        >>> from pyzx import VertexType
        >>> assert graph.type(2) == graph.type(3) == VertexType.Z
        >>> assert graph.phase(2) == 2 * .25 and graph.phase(3) == 2 * .75
        >>> assert graph.type(4) == graph.type(5) == VertexType.X
        >>> assert graph.phase(4) == graph.phase(5) == 2 * .5
        >>> assert graph.graph == {
        ...     0: {2: 1},
        ...     1: {3: 1},
        ...     2: {0: 1, 4: 1, 5: 1},
        ...     3: {1: 1, 4: 1, 5: 1},
        ...     4: {2: 1, 3: 1, 6: 1},
        ...     5: {2: 1, 3: 1, 7: 1},
        ...     6: {4: 1},
        ...     7: {5: 1}}

        A pyzx graph is simple, so the parallel wires between two spiders
        are merged with :meth:`pyzx.Graph.add_edge_table`, i.e. by the
        fusion and Hopf rules with their scalars in ``graph.scalar``, and
        pyzx has no Y vertex, so a :class:`Y` spider is encoded with
        :meth:`Y.to_x`.

        >>> hopf = X(1, 2, .75) >> Z(2, 1, 1 / 3)
        >>> graph = hopf.to_pyzx()  # doctest: +EXTRA
        >>> assert graph.num_edges() == 2 and graph.scalar.power2 == -2

        A spider left with no legs, which pyzx cannot contract, is folded
        into ``graph.scalar`` with :meth:`pyzx.Graph.remove_isolated_vertices`,
        as is a pair of spiders connected to nothing else.

        >>> graph = (X(0, 2) >> Z(2, 0, .25)).to_pyzx()  # doctest: +EXTRA
        >>> assert not graph.vertices() and graph.scalar.to_number() != 1
        """
        from pyzx import Graph, VertexType, EdgeType
        diagram = frobenius.Functor(
            ob_map=lambda typ: typ, dom=Diagram, cod=Diagram,
            ar_map=lambda box: box.to_x() if isinstance(box, Y) else box)(self)
        graph, scan = Graph(), []
        for i, _ in enumerate(diagram.dom):
            node, hadamard = graph.add_vertex(VertexType.BOUNDARY), False
            scan.append((node, hadamard))
            graph.set_inputs(graph.inputs() + (node,))
            graph.set_position(node, i, 0)
        boxes_and_offsets = zip(diagram.boxes, diagram.offsets)
        for row, (box, offset) in enumerate(boxes_and_offsets):
            if isinstance(box, Spider):
                node = graph.add_vertex(
                    VertexType.Z if isinstance(box, Z) else VertexType.X,
                    phase=Fraction(box.phase * 2) if box.phase else None)
                graph.set_position(node, offset, row + 1)
                edge_table = {}
                for i, _ in enumerate(box.dom):
                    source, hadamard = scan[offset + i]
                    edges = edge_table.setdefault((source, node), [0, 0])
                    edges[hadamard] += 1
                graph.add_edge_table(edge_table)
                scan = scan[:offset] + len(box.cod) * [(node, False)]\
                    + scan[offset + len(box.dom):]
            elif isinstance(box, Permutation):
                segment = scan[offset:offset + len(box.dom)]
                scan[offset:offset + len(box.dom)] = [
                    segment[i] for i in box.perm]
            elif isinstance(box, Scalar):
                graph.scalar.add_float(box.data)
            elif box == H:
                node, hadamard = scan[offset]
                scan[offset] = (node, not hadamard)
            else:
                raise NotImplementedError
        for i, _ in enumerate(diagram.cod):
            target = graph.add_vertex(VertexType.BOUNDARY)
            source, hadamard = scan[i]
            etype = EdgeType.HADAMARD if hadamard else EdgeType.SIMPLE
            graph.add_edge((source, target), etype)
            graph.set_position(target, i, len(diagram) + 1)
            graph.set_outputs(graph.outputs() + (target,))
        graph.remove_isolated_vertices()
        return graph

    @staticmethod
    def from_pyzx(graph):
        """
        Takes a :class:`pyzx.Graph` returns a :class:`zx.Diagram`.

        Examples
        --------

        >>> bialgebra = Z(1, 2, .25) @ Z(1, 2, .75)\\
        ...     >> Id(1) @ SWAP @ Id(1) >> X(2, 1, .5) @ X(2, 1, .5)
        >>> graph = bialgebra.to_pyzx()  # doctest: +EXTRA
        >>> assert Diagram.from_pyzx(graph) == bialgebra

        The scalar of the graph comes back as a :class:`Scalar`, so that the
        round trip is right on the nose rather than up to a scalar.

        >>> graph = (scalar(2) @ Nat(1)).to_pyzx()  # doctest: +EXTRA
        >>> assert Diagram.from_pyzx(graph) == scalar(2) @ Nat(1)

        Note
        ----

        Raises :code:`ValueError` if either:
        * a boundary node is not in :code:`graph.inputs() + graph.outputs()`,
        * or :code:`set(graph.inputs()).intersection(graph.outputs())`.
        """
        from pyzx import VertexType, EdgeType

        def node2box(node, n_legs_in, n_legs_out):
            if graph.type(node) not in {VertexType.Z, VertexType.X}:
                raise NotImplementedError  # pragma: no cover
            return \
                (Z if graph.type(node) == VertexType.Z else X)(  # noqa: E721
                    n_legs_in, n_legs_out, graph.phase(node) * .5)

        def move(scan, source, target):
            if target < source:
                swaps = Id(target)\
                    @ Diagram.swap(source - target, 1)\
                    @ Id(len(scan) - source - 1)
                scan = scan[:target] + (scan[source],)\
                    + scan[target:source] + scan[source + 1:]
            else:
                swaps = Id(len(scan))
            return scan, swaps

        def make_wires_adjacent(scan, diagram, inputs):
            if not inputs:
                return scan, diagram, len(scan)
            offset = scan.index(inputs[0])
            for i, _ in enumerate(inputs[1:]):
                source, target = scan.index(inputs[i + 1]), offset + i + 1
                scan, swaps = move(scan, source, target)
                diagram = diagram >> swaps
            return scan, diagram, offset

        missing_boundary = any(
            graph.type(node) == VertexType.BOUNDARY  # noqa: E721
            and node not in graph.inputs() + graph.outputs()
            for node in graph.vertices())
        if missing_boundary:
            raise ValueError
        duplicate_boundary = set(graph.inputs()).intersection(graph.outputs())
        if duplicate_boundary:
            raise ValueError
        diagram, scan = Id(len(graph.inputs())), graph.inputs()
        for node in [v for v in graph.vertices()
                     if v not in graph.inputs() + graph.outputs()]:
            inputs = [v for v in graph.neighbors(node) if v < node
                      and v not in graph.outputs() or v in graph.inputs()]
            inputs.sort(key=scan.index)
            outputs = [v for v in graph.neighbors(node) if v > node
                       and v not in graph.inputs() or v in graph.outputs()]
            scan, diagram, offset = make_wires_adjacent(scan, diagram, inputs)
            hadamards = Id().tensor(*[
                H if graph.edge_type((i, node)) == EdgeType.HADAMARD
                else Id(1) for i in scan[offset: offset + len(inputs)]])
            box = node2box(node, len(inputs), len(outputs))
            diagram = diagram >> Id(offset) @ (hadamards >> box)\
                @ Id(len(diagram.cod) - offset - len(inputs))
            scan = scan[:offset] + len(outputs) * (node,)\
                + scan[offset + len(inputs):]
        for target, output in enumerate(graph.outputs()):
            node, = graph.neighbors(output)
            etype = graph.edge_type((node, output))
            hadamard = H if etype == EdgeType.HADAMARD else Id(1)
            scan, swaps = move(scan, scan.index(node, target), target)
            diagram = diagram >> swaps\
                >> Id(target) @ hadamard @ Id(len(scan) - target - 1)
        number = graph.scalar.to_number()
        return diagram if number == 1 else scalar(number) @ diagram


class Box(tensor.Box[complex], Diagram):
    """
    A ZX box is a tensor box in a ZX diagram.

    Parameters:
        name (str) : The name of the box.
        dom (rigid.Nat) : The domain of the box, i.e. its input.
        cod (rigid.Nat) : The codomain of the box, i.e. its output.
    """


class Sum(tensor.Sum[complex], Box):
    """
    A formal sum of ZX diagrams with the same domain and codomain.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Dim) : The domain of the formal sum.
        cod (Dim) : The codomain of the formal sum.
    """


class Bubble(tensor.Bubble[complex], Box):
    """
    A ZX bubble is a tensor bubble in a ZX diagram, which differentiates as
    a ZX box does.

    >>> bubble = Z(1, 1).bubble()
    >>> assert (bubble >> Z(1, 1)).eval() == bubble.eval()
    """
    grad = Box.grad


class Permutation(tensor.Permutation[complex], Box):
    "A permutation in a ZX diagram."


class Swap(Permutation, tensor.Swap[complex], Box):
    """
    Swap in a ZX diagram.

    >>> assert repr(SWAP) == "quantum.zx.Swap(rigid.Nat(1), rigid.Nat(1))"
    >>> assert str(SWAP) == "SWAP"
    """
    def __str__(self):
        return "SWAP"


class Spider(tensor.Spider[complex], Box):
    """
    Abstract spider box, i.e. a phased spider in the eigenbasis of a Pauli.

    Its array has ``1`` on the all-zero entry and ``exp(2j * pi * phase)``
    on the all-one entry, conjugated by ``basis``, the unitary whose columns
    are the eigenvectors of the Pauli: the identity for :class:`Z`, the
    Hadamard for :class:`X` and ``[[1, 1], [1j, -1j]] / sqrt(2)`` for
    :class:`Y`, so that the spider of phase ``.5`` on one leg in and out is
    the Pauli itself. Only the :class:`Z` spiders, those of the
    computational basis that a tensor copies, have the flags of a total
    function.

    Parameters:
        n_legs_in (int) : The number of legs in.
        n_legs_out (int) : The number of legs out.
        phase : The phase, in units of ``2 * pi``.

    Example
    -------
    >>> assert Z(0, 0).eval().array == 2
    >>> for spider, pauli in [(Z, quantum.Z), (X, quantum.X), (Y, quantum.Y)]:
    ...     assert spider(1, 1, .5).eval().is_close(pauli.eval())
    >>> assert Z(1, 2).is_cartesian and not X(1, 2).is_deterministic
    """
    basis = ((1, 0), (0, 1))

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        n_legs_in, n_legs_out = index(n_legs_in), index(n_legs_out)
        super().__init__(n_legs_in, n_legs_out, Nat(1), phase)
        factory_str = type(self).__name__
        phase_str = f", {self.phase}" if self.phase else ""
        self.name = f"{factory_str}({n_legs_in}, {n_legs_out}{phase_str})"
        if not isinstance(self, Z):
            self.deterministic = self.affine = False

    def __getnewargs__(self):
        return len(self.dom), len(self.cod), self.phase

    def __setstate__(self, state):
        if "_name" in state and state["_name"] == type(self).__name__:
            phase = state.get("_data", None)
            phase_str = f', {phase}' if phase else ''
            state["_name"] = (
                type(self).__name__ +
                f"({state['_dom'].n}, {state['_cod'].n}{phase_str})"
            )
        super().__setstate__(state)
        if not isinstance(self, Z):
            self.deterministic = self.affine = False

    def __repr__(self):
        return str(self).replace(type(self).__name__, factory_name(type(self)))

    @classmethod
    def from_tree(cls, tree: dict) -> "Spider":
        """
        Decode the tree of :meth:`discopy.frobenius.Spider.to_tree`, whose
        type is a qubit that the constructor supplies.

        >>> from discopy.utils import dumps, loads
        >>> assert loads(dumps(Z(1, 2, .5))) == Z(1, 2, .5)
        """
        return cls(
            tree['n_legs_in'], tree['n_legs_out'], tree.get('phase', 0))

    def subs(self, *args):
        phase = cat.rsubs(self.phase, *args)
        return type(self)(len(self.dom), len(self.cod), phase=phase)

    def grad(self, var, **params):
        """
        The gradient of a spider is a formal sum, see arXiv:2103.07960: the
        all-zero entry of its array is constant and the derivative of the
        all-one entry is ``2j * pi * phase'`` times itself, i.e. ``1j * pi *
        phase'`` times the difference between the spider and the spider
        shifted by half a turn, which negates that entry and fixes the other.
        """
        if var not in self.free_symbols:
            return Sum((), self.dom, self.cod)
        gradient = self.phase.diff(var)
        gradient = complex(gradient) if not gradient.free_symbols else gradient
        n_legs_in, n_legs_out = len(self.dom), len(self.cod)
        return Scalar(1j * pi * gradient)\
            @ type(self)(n_legs_in, n_legs_out, self.phase)\
            + Scalar(-1j * pi * gradient)\
            @ type(self)(n_legs_in, n_legs_out, self.phase + .5)

    def dagger(self):
        return type(self)(len(self.cod), len(self.dom), -self.phase)

    def rotate(self, left=False):
        del left
        return type(self)(len(self.cod), len(self.dom), self.phase)

    @property
    def array(self):
        if self.free_symbols:
            import sympy
            phase = sympy.exp(2 * sympy.pi * sympy.I * self.phase)
        else:
            phase = cmath.exp(2j * pi * self.phase)
        with backend() as np:
            basis = np.array(self.basis)
            legs = [len(self.dom) * [np.conj(basis[:, b])]
                    + len(self.cod) * [basis[:, b]] for b in (0, 1)]
            zero, one = (
                reduce(np.kron, vectors, np.array(1)) for vectors in legs)
            return (zero + phase * one).reshape(
                len(self.dom @ self.cod) * (2, ))


class Z(Spider):
    """ Z spider, in the computational basis. """
    tikzstyle_name = 'Z'
    color = 'green'


class Y(Spider):
    """
    Y spider, in the eigenbasis of the Pauli Y, i.e. the Hadamard basis
    rotated by the phase gate ``S = Z(1, 1, .25)``, so that a Y spider is an
    X spider with ``S`` on each leg out and its dagger on each leg in.

    >>> S = Z(1, 1, .25)
    >>> assert Y(2, 1, .3).to_x()\\
    ...     == S.dagger() @ S.dagger() >> X(2, 1, .3) >> S
    >>> assert Y(2, 1, .3).to_x().eval().is_close(Y(2, 1, .3).eval())
    """
    tikzstyle_name = 'Y'
    color = "blue"
    basis = ((2 ** -.5, 2 ** -.5), (2 ** -.5 * 1j, -2 ** -.5 * 1j))

    def to_x(self) -> Diagram:
        """ The Y spider as an X spider between phase gates. """
        n_legs_in, n_legs_out = len(self.dom), len(self.cod)
        phase_gate = Z(1, 1, .25)
        return Id().tensor(*n_legs_in * [phase_gate.dagger()])\
            >> X(n_legs_in, n_legs_out, self.phase)\
            >> Id().tensor(*n_legs_out * [phase_gate])


class X(Spider):
    """ X spider, in the Hadamard basis. """
    tikzstyle_name = 'X'
    color = "red"
    basis = ((2 ** -.5, 2 ** -.5), (2 ** -.5, -2 ** -.5))


class Scalar(Box):
    """ Scalar in a ZX diagram. """
    serialised_attrs = ("data", )

    def __init__(self, data):
        super().__init__("scalar", Nat(0), Nat(0), data=data)
        self.drawing_name = format_number(data)

    def __str__(self):
        return f"scalar({format_number(self.data)})"

    def subs(self, *args):
        data = cat.rsubs(self.data, *args)
        return Scalar(data)

    def dagger(self):
        return Scalar(self.data.conjugate())

    def grad(self, var, **params):
        if var not in self.free_symbols:
            return Sum((), self.dom, self.cod)
        return Scalar(self.data.diff(var))


def scalar(data):
    """ Returns a scalar. """
    return Scalar(data)


root2 = scalar(2 ** 0.5)


def gate2zx(box):
    """ Turns gates into ZX diagrams. """
    if isinstance(box, (Bra, Ket)):
        dom, cod = (1, 0) if isinstance(box, Bra) else (0, 1)
        spiders = [X(dom, cod, phase=.5 * bit) for bit in box.bitstring]
        return Id().tensor(*spiders) @ scalar(pow(2, -len(box.bitstring) / 2))
    if isinstance(box, (Rz, Rx)):
        return (Z if isinstance(box, Rz) else X)(1, 1, box.phase)
    if isinstance(box, Controlled) and box.distance != 1:
        return circuit2zx(box._decompose())
    if isinstance(box, Controlled) and isinstance(box.controlled, (Rz, Rx)):
        gadget = Z(1, 2) @ Z(1, 2, box.phase / 2)\
            >> Id(1) @ (X(2, 1) >> Z(1, 0, -box.phase / 2)) @ Id(1) @ root2
        return gadget if isinstance(box.controlled, Rz)\
            else Id(1) @ H >> gadget >> Id(1) @ H
    if isinstance(box, Controlled) and isinstance(box.controlled, U1):
        return Z(1, 2, box.phase / 2) @ Z(1, 2, box.phase / 2)\
            >> Id(1) @ (X(2, 1) >> Z(1, 0, -box.phase / 2)) @ Id(1) @ root2
    if isinstance(box, GatesScalar):
        if box.is_mixed:
            raise NotImplementedError
        return scalar(box.data)
    standard_gates = {
        # A physical swap is wire crossing in the ZX calculus.
        quantum.SWAP: SWAP,
        quantum.H: H,
        quantum.Z: Z(1, 1, .5),
        quantum.X: X(1, 1, .5),
        quantum.Y: Z(1, 1, .5) >> X(1, 1, .5) @ scalar(1j),
        quantum.S: Z(1, 1, .25),
        quantum.T: Z(1, 1, .125),
        CZ: Z(1, 2) @ Id(1) >> Id(1) @ H @ Id(1) >> Id(1) @ Z(2, 1) @ root2,
        CX: Z(1, 2) @ Id(1) >> Id(1) @ X(2, 1) @ root2}
    return standard_gates[box]


circuit2zx = quantum.circuit.Functor(
    ob_map={qubit: Nat(1)}, ar_map=gate2zx,
    dom=Circuit, cod=Diagram)

H = Box('H', Nat(1), Nat(1), data=[x / 2 ** .5 for x in (1, 1, 1, -1)],
        is_deterministic=False, is_affine=False)
H.dagger = lambda: H
H.draw_as_spider = True
H.drawing_name, H.tikzstyle_name, = '', 'H'
H.color, H.shape = "yellow", "rectangle"

SWAP = Swap(Nat(1), Nat(1))
Diagram.box_factory = Box
Diagram.swap_factory, Diagram.sum_factory = Swap, Sum
Diagram.bubble_factory = Bubble
Diagram.permutation_factory = Permutation
Id = Diagram.id


def turn(phase) -> complex:
    """ The root of unity ``exp(2j * pi * phase)`` of a phase in turns. """
    return cmath.exp(2j * pi * phase)


class Number:
    """
    An exact complex number, the scalar of a :class:`Hypergraph`: the
    product of ``sqrt(2) ** power2``, of the root of unity ``turn(phase)``,
    of a sum ``c_1 turn(psi_1) + ... + c_n turn(psi_n)`` with integer
    coefficients for each of ``sums``, and of a complex ``factor``, one
    unless the number came in as one, e.g. the data of a :class:`Scalar`.
    These are the scalars of :cite:t:`KissingerWetering20`, so that the
    rules of :class:`Rewriter` are equations on the nose.

    A number is kept in normal form: phases are fractions of a turn in
    ``[0, 1)``, each sum has no two terms half a turn apart and starts with
    a positive coefficient at phase zero, a sum with one term is folded
    into the rest, and so is ``1 + turn(1/4) = sqrt(2) turn(1/8)``. Two
    numbers are equal when they are written the same in normal form.

    Parameters:
        power2 : The power of ``sqrt(2)``.
        phase : The phase, in turns.
        sums : The sums, each a tuple of ``(phase, coefficient)`` pairs.
        factor : A complex number, ``0`` for zero.

    Example
    -------
    >>> one, root2 = Number(), Number(power2=1)
    >>> assert root2 * root2 == Number(power2=2) and complex(one) == 1
    >>> assert Number.sum((0, 1), (Fraction(1, 2), 1)) == Number(factor=0)
    >>> assert Number.sum((0, 1), (Fraction(1, 4), 1))\\
    ...     == Number(power2=1, phase=Fraction(1, 8))
    >>> node = Number.sum((0, 1), (Fraction(1, 8), 1))
    >>> print(node * node.conjugate())
    (1 + e(1/8)) · (1 - e(3/8))
    >>> assert abs(complex(node * node.conjugate())
    ...            - abs(complex(node)) ** 2) < 1e-12
    """
    def __init__(self, power2: int = 0, phase=0, sums=(), factor=1):
        if not factor:
            power2, phase, sums = 0, 0, ()
        self.power2, self.phase = power2, Fraction(phase) % 1
        self.sums, self.factor = tuple(sorted(sums)), factor

    @classmethod
    def sum(cls, *terms: tuple[Fraction, int]) -> Number:
        """
        The number ``c_1 turn(psi_1) + ... + c_n turn(psi_n)`` in normal form,
        for terms ``(psi_i, c_i)`` with integer coefficients.
        """
        merged = {}
        for phase, coefficient in terms:
            phase = Fraction(phase) % 1
            if phase >= Fraction(1, 2):
                phase, coefficient = phase - Fraction(1, 2), -coefficient
            merged[phase] = merged.get(phase, 0) + coefficient
        terms = sorted((phase, c) for phase, c in merged.items() if c)
        if not terms:
            return cls(factor=0)
        (first, coefficient), *rest = terms
        phase = first + (Fraction(1, 2) if coefficient < 0 else 0)
        sign = 1 if coefficient > 0 else -1
        terms = ((0, abs(coefficient)), ) + tuple(
            (psi - first, sign * c) for psi, c in rest)
        if not rest:
            return cls(phase=phase) * cls.integer(abs(coefficient))
        quarter, sign = Fraction(1, 4), terms[1][1]
        if terms == ((0, 1), (quarter, sign)):
            return cls(power2=1, phase=phase + sign * quarter / 2)
        return cls(phase=phase, sums=(terms, ))

    @classmethod
    def integer(cls, n: int) -> Number:
        """ A natural number, a power of ``sqrt(2)`` when it is one of two. """
        if n > 0 and n & (n - 1) == 0:
            return cls(power2=2 * (n.bit_length() - 1))
        return cls(factor=n)

    @classmethod
    def from_complex(cls, number: complex) -> Number:
        """
        A complex number, as a power of ``sqrt(2)`` times a power of ``1j``
        when it is the float of one, e.g. ``2 ** .5`` or ``-1j / 2``.

        >>> assert Number.from_complex(2 ** -.5) == Number(power2=-1)
        >>> assert Number.from_complex(-1j) == Number(phase=Fraction(3, 4))
        >>> assert Number.from_complex(.3).factor == .3
        """
        for quarter, unit in enumerate((1, 1j, -1, -1j)):
            modulus = complex(number) / unit
            if modulus.imag == 0 and modulus.real > 0:
                for power2 in range(-64, 65):
                    if modulus.real == 2 ** (power2 / 2):
                        return cls(power2, Fraction(quarter, 4))
        return cls(factor=number)

    def __mul__(self, other: Number) -> Number:
        return Number(
            self.power2 + other.power2, self.phase + other.phase,
            self.sums + other.sums, self.factor * other.factor)

    def __truediv__(self, other: Number) -> Number:
        if other.sums or not other.factor:
            raise ZeroDivisionError(messages.NOT_INVERTIBLE.format(other))
        return Number(
            self.power2 - other.power2, self.phase - other.phase,
            self.sums, self.factor / other.factor)

    def conjugate(self) -> Number:
        """ The complex conjugate, with each phase negated. """
        result = Number(self.power2, -self.phase, factor=complex(
            self.factor).conjugate() if isinstance(self.factor, complex)
            else self.factor)
        for terms in self.sums:
            result *= Number.sum(*((-psi, c) for psi, c in terms))
        return result

    def __complex__(self):
        result = 2 ** (self.power2 / 2) * turn(self.phase) * self.factor
        for terms in self.sums:
            result *= sum(c * turn(psi) for psi, c in terms)
        return complex(result)

    def __eq__(self, other):
        return isinstance(other, Number) and (
            self.power2, self.phase, self.sums, self.factor) == (
            other.power2, other.phase, other.sums, other.factor)

    def __hash__(self):
        return hash((self.power2, self.phase, self.sums, self.factor))

    def __repr__(self):
        fields = [f"{name}={value!r}" for name, value, default in (
            ("power2", self.power2, 0), ("phase", self.phase, 0),
            ("sums", self.sums, ()), ("factor", self.factor, 1))
            if value != default]
        return f"{factory_name(Number)}({', '.join(fields)})"

    def __str__(self):
        if not self.factor:
            return "0"
        parts = [f"√2^{self.power2}"] * bool(self.power2)
        parts += [f"e({self.phase})"] * bool(self.phase)
        parts += ["(" + "".join(
            (" - " if c < 0 else " + ") * bool(i) + f"{abs(c)} " * (
                abs(c) != 1 and bool(psi)) + (f"e({psi})" if psi else str(c))
            for i, (psi, c) in enumerate(terms)) + ")" for terms in self.sums]
        parts += [format_number(self.factor)] * (self.factor != 1)
        return " · ".join(parts) or "1"

    def to_tree(self) -> dict:
        """
        Serialise a number by its fields.

        >>> from discopy.utils import dumps, loads
        >>> number = Number.sum((0, 1), (Fraction(1, 8), 1))\\
        ...     * Number(factor=-3)
        >>> assert loads(dumps(number)) == number
        """
        return {'factory': factory_name(Number), 'power2': self.power2,
                'phase': self.phase, 'factor': self.factor, 'sums': [
                    [list(term) for term in terms] for terms in self.sums]}

    @classmethod
    def from_tree(cls, tree: dict) -> Number:
        """ Decode a serialised number, see :meth:`to_tree`. """
        return cls(tree['power2'], tree['phase'], tuple(
            tuple(tuple(term) for term in terms) for terms in tree['sums']),
            tree['factor'])

    def to_pyzx(self):
        """
        The :class:`pyzx.graph.scalar.Scalar` of a number, whose phases are
        in half-turns: a sum ``1 + turn(psi)`` is one of its phase nodes, the
        first other sum is its sum of phases and the rest is its float
        factor.
        """
        from pyzx.graph.scalar import Scalar as PyZXScalar
        result = PyZXScalar()
        result.is_zero = not self.factor
        result.power2, result.phase = self.power2, 2 * self.phase
        result.floatfactor = complex(self.factor)
        for terms in self.sums:
            if len(terms) == 2 and terms[0] == (0, 1) and terms[1][1] == 1:
                result.phasenodes.append(2 * terms[1][0])
            elif not result.sum_of_phases:
                result.sum_of_phases = {2 * psi: c for psi, c in terms}
            else:
                result.floatfactor *= complex(Number(sums=(terms, )))
        return result

    @classmethod
    def from_pyzx(cls, number) -> Number:
        """ The number of a :class:`pyzx.graph.scalar.Scalar`. """
        if number.is_zero:
            return cls(factor=0)
        result = cls(number.power2, Fraction(number.phase) / 2) * (
            cls.from_complex(number.floatfactor))
        for node in number.phasenodes:
            result *= cls.sum((0, 1), (Fraction(node) / 2, 1))
        if number.sum_of_phases:
            result *= cls.sum(*((Fraction(psi) / 2, c)
                                for psi, c in number.sum_of_phases.items()))
        return result


ONE = Number()


class Hypergraph(hypergraph.Hypergraph):
    """
    A ZX diagram as a hypergraph, graph-like in the sense of
    :cite:t:`DuncanEtAl20`: its spiders are the green spiders, each with its
    phase in turns, its only boxes are the Hadamard :data:`H`, a red spider
    being a green one with a Hadamard on each leg, and its scalar is an
    exact :class:`Number`.

    Composition is the pushout of :class:`discopy.hypergraph.Hypergraph`,
    i.e. it fuses spiders and adds their phases, which this hypergraph keeps
    modulo a turn, and multiplies the scalars, as the tensor does, while the
    dagger conjugates the scalar. Green spiders are special, so that this is
    a faithful encoding: :meth:`to_diagram` is :meth:`from_diagram` undone,
    up to the axioms of ZX diagrams.

    Parameters:
        scalar : The scalar, one by default.

    Example
    -------
    >>> quarter = Z(1, 1, .25).to_hypergraph()
    >>> assert quarter >> quarter == Z(1, 1, .5).to_hypergraph()
    >>> assert quarter[::-1] == Z(1, 1, .75).to_hypergraph()
    >>> assert X(1, 1, .5).to_hypergraph() == (H >> Z(1, 1, .5) >> H
    ...                                        ).to_hypergraph()

    The phases of the hypergraph go back to spiders:

    >>> print((Z(1, 2, .25) >> Z(2, 1, .25)).to_hypergraph())
    Z(1, 1, 0.5)

    The :class:`Scalar` boxes of a diagram are its scalar:

    >>> graph = (Z(1, 1) @ scalar(2 ** .5) @ scalar(1j)).to_hypergraph()
    >>> assert not graph.boxes and graph.scalar == Number(1, Fraction(1, 4))
    >>> assert (graph >> graph[::-1]).scalar == Number(power2=2)
    """
    category = Diagram

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=redefined-outer-name
    def __init__(
            self, dom, cod, boxes, wires, spider_types=None, offsets=None,
            phases=None, scalar: Number = None, _scan: bool = True):
        super().__init__(
            dom, cod, boxes, wires, spider_types, offsets, phases, _scan=_scan)
        self.scalar = ONE if scalar is None else scalar
    # pylint: enable=too-many-arguments,too-many-positional-arguments
    # pylint: enable=redefined-outer-name

    def scaled(self, number: Number) -> Hypergraph:
        """ The same hypergraph with its scalar multiplied by a number. """
        if number == ONE:
            return self
        return type(self)(
            self.dom, self.cod, self.boxes, self.wires, self.spider_types,
            self.offsets, self.phases, self.scalar * number, _scan=False)

    def keeps_scalar(self, result: Hypergraph) -> Hypergraph:
        """
        A hypergraph built from this one by a method of
        :class:`discopy.hypergraph.Hypergraph`, which knows no scalar, with
        the scalar of this one.
        """
        return result if result is self else result.scaled(self.scalar)

    @unbiased
    def then(self, other: Hypergraph) -> Hypergraph:
        return super().then(other).scaled(
            self.scalar * getattr(other, "scalar", ONE))

    def tensor(self, *others: Hypergraph) -> Hypergraph:
        return super().tensor(*others).scaled(reduce(
            lambda x, y: x * y,
            (getattr(other, "scalar", ONE) for other in others), self.scalar))

    def dagger(self) -> Hypergraph:
        return super().dagger().scaled(self.scalar.conjugate())

    def trace(self, n=1, left=False) -> Hypergraph:
        return self.keeps_scalar(super().trace(n, left))

    def interchange(self, i: int, j: int) -> Hypergraph:
        return self.keeps_scalar(super().interchange(i, j))

    def topological_order(self) -> Hypergraph:
        return self.keeps_scalar(super().topological_order())

    def complements(self, pattern, match):
        for context in super().complements(pattern, match):
            yield self.keeps_scalar(context)

    def __eq__(self, other):
        return super().__eq__(other)\
            and self.scalar == getattr(other, "scalar", ONE)

    def __hash__(self):
        return hash((super().__hash__(), self.scalar))

    def __repr__(self):
        result = super().__repr__()
        return result if self.scalar == ONE\
            else result[:-1] + f", scalar={self.scalar!r})"

    def to_tree(self) -> dict:
        """
        Serialise a hypergraph with its scalar.

        >>> from discopy.utils import dumps, loads
        >>> graph = (Z(1, 2, .25) >> X(2, 1) @ scalar(3)).to_hypergraph()
        >>> assert loads(dumps(graph)) == graph
        """
        return super().to_tree() | (
            {} if self.scalar == ONE else {'scalar': self.scalar.to_tree()})

    @classmethod
    def from_tree(cls, tree: dict) -> Hypergraph:
        result = super().from_tree(tree)
        return result.scaled(Number.from_tree(tree['scalar']))\
            if 'scalar' in tree else result

    @classmethod
    def from_diagram(cls, old: Diagram) -> Hypergraph:
        """ A diagram as a hypergraph, see :meth:`Diagram.split_scalar`. """
        number, rest = old.split_scalar()
        return super().from_diagram(rest).scaled(number)

    def to_diagram(self) -> Diagram:
        """
        A diagram, with a :class:`Scalar` box for the scalar when it is not
        one.

        >>> print((H >> H @ scalar(2)).to_hypergraph().to_diagram())
        H >> H >> Nat(1) @ scalar(2)
        """
        result, number = super().to_diagram(), complex(self.scalar)
        return result if self.scalar == ONE\
            else result @ Scalar(number.real if not number.imag else number)

    @staticmethod
    def reduce_phase(phase):
        """ A phase modulo a turn, unless it is symbolic. """
        return phase if getattr(phase, "free_symbols", None) else phase % 1

    @classmethod
    def from_box(cls, box: Box) -> Hypergraph:
        """
        A :class:`Z` spider is a spider with its phase, an :class:`X` spider
        one behind a Hadamard on each leg, a :class:`Y` spider its
        :meth:`Y.to_x` and any other box is itself.

        >>> graph = Hypergraph.from_box(X(1, 2, .5))
        >>> assert graph.boxes == 3 * (H, ) and graph.phases == (0, .5, 0, 0)
        """
        if isinstance(box, Y):
            return cls.from_diagram(box.to_x())
        if not isinstance(box, (Z, X)):
            return super().from_box(box)
        n_legs_in, n_legs_out = len(box.dom), len(box.cod)
        if isinstance(box, Z):
            return cls.spiders(n_legs_in, n_legs_out, Nat(1), [box.phase])
        legs = tuple(range(1, n_legs_in + n_legs_out + 1))
        box_wires = tuple(((leg, ), (0, )) for leg in legs[:n_legs_in])\
            + tuple(((0, ), (leg, )) for leg in legs[n_legs_in:])
        return cls(
            box.dom, box.cod, len(legs) * (H, ),
            (legs[:n_legs_in], box_wires, legs[n_legs_in:]),
            (len(legs) + 1) * [Nat(1)], phases={0: box.phase})

    @classmethod
    def from_pyzx(cls, graph) -> Hypergraph:
        """
        A hypergraph from a PyZX graph: its green spiders joined by plain
        edges are one spider, with the sum of their phases, a red spider is
        green with its edges toggled between plain and Hadamard, and a
        boundary behind a Hadamard edge is a spider of its own, and the
        scalar of the graph is its scalar.

        Parameters:
            graph : A PyZX graph with boundaries, green and red spiders.

        >>> bialgebra = Z(1, 2, .25) @ Z(1, 2, .75)\\
        ...     >> Id(1) @ SWAP @ Id(1) >> X(2, 1, .5) @ X(2, 1, .5)
        >>> graph = bialgebra.to_pyzx()  # doctest: +EXTRA
        >>> assert Hypergraph.from_pyzx(graph) == bialgebra.to_hypergraph()
        """
        from pyzx import VertexType, EdgeType
        red = {v for v in graph.vertices()
               if graph.type(v) == VertexType.X}  # noqa: E721
        parent = {v: v for v in graph.vertices()}

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        hadamards = []
        for edge in graph.edges():
            a, b = graph.edge_st(edge)
            plain = (graph.edge_type(edge) == EdgeType.SIMPLE) == (
                (a in red) == (b in red))
            if plain:
                parent[find(a)] = find(b)
            else:
                hadamards.append((a, b))
        spiders = list(dict.fromkeys(find(v) for v in graph.vertices()))
        label = {s: i for i, s in enumerate(spiders)}
        phases = hypergraph.add_phases(
            (label[find(v)], Fraction(graph.phase(v)) / 2)
            for v in graph.vertices())
        dom_wires = tuple(label[find(v)] for v in graph.inputs())
        cod_wires = tuple(label[find(v)] for v in graph.outputs())
        box_wires = tuple(
            ((label[find(a)], ), (label[find(b)], )) for a, b in hadamards)
        return cls(
            Nat(len(dom_wires)), Nat(len(cod_wires)), len(hadamards) * (H, ),
            (dom_wires, box_wires, cod_wires), len(spiders) * [Nat(1)],
            phases=phases, scalar=Number.from_pyzx(graph.scalar))

    def to_pyzx(self):
        """
        The PyZX graph of a graph-like hypergraph, with a green spider for
        each spider and a boundary on a plain edge for each port, see
        :meth:`from_pyzx`, with the scalar of the hypergraph. PyZX graphs are
        simple, so parallel Hadamard edges cancel in pairs, and a Hadamard
        loop is a half-turn, each with its scalar.

        >>> graph = (Z(1, 1, .25) >> H >> X(1, 2, .5)).to_hypergraph()
        >>> pyzx_graph = graph.to_pyzx()  # doctest: +EXTRA
        >>> assert Hypergraph.from_pyzx(pyzx_graph) == graph
        """
        from pyzx import Graph, VertexType, EdgeType
        result = Graph()
        result.scalar = self.scalar.to_pyzx()
        spiders = [result.add_vertex(VertexType.Z, qubit=i, row=1)
                   for i, _ in enumerate(self.spider_types)]
        for spider, phase in zip(spiders, self.phases):
            if phase:
                result.add_to_phase(spider, Fraction(phase) * 2)
        table = {}
        for box, (dom, cod) in zip(self.boxes, self.box_wires):
            if is_hadamard(box) and dom[0] == cod[0]:
                result.add_to_phase(spiders[dom[0]], 1)
                result.scalar.add_power(-1)
            elif is_hadamard(box):
                key = result.edge(spiders[dom[0]], spiders[cod[0]])
                table.setdefault(key, [0, 0])[1] += 1
            else:
                raise ValueError(messages.NOT_GRAPHLIKE.format(box))
        result.add_edge_table(table)
        inputs, outputs = [], []
        for row, wires, ends in (
                (0, self.dom_wires, inputs), (2, self.cod_wires, outputs)):
            for i, spider in enumerate(wires):
                ends.append(result.add_vertex(
                    VertexType.BOUNDARY, qubit=i, row=row))
                result.add_edge(
                    result.edge(ends[-1], spiders[spider]), EdgeType.SIMPLE)
        result.set_inputs(inputs)
        result.set_outputs(outputs)
        return result

    def full_reduce(self) -> Hypergraph:
        """
        Simplify by the rules of the ``full_reduce`` of
        :cite:t:`KissingerWetering20`, see :class:`Rewriter`, each an
        equation on the nose, so that the result has the scalar it needs.

        >>> diagram = Z(1, 1, .125) >> H >> H >> Z(1, 1, .125)
        >>> print(diagram.to_hypergraph().full_reduce())
        Z(1, 1, 1/4)
        >>> print((Z(0, 1, .5) >> Z(1, 0)).to_hypergraph().full_reduce())
        scalar(0)
        """
        rewriter = Rewriter(self)
        rewriter.reduce()
        return rewriter.to_hypergraph()

    def size(self) -> tuple[int, int, int]:
        """
        The number of spiders, of Hadamard edges between them and of spiders
        whose phase is not Clifford, the T-count, counting a boundary behind
        a Hadamard edge as no spider, as PyZX does.

        Two spiders are joined by an edge when an odd number of Hadamards lie
        between them, again as PyZX has it: a Hadamard self-loop is a
        half-turn phase rather than an edge, and two parallel Hadamards
        cancel.

        >>> assert (Z(1, 1, .125) >> H >> Z(1, 1, .5)).to_hypergraph().size()\\
        ...     == (2, 1, 1)
        """
        rewriter = Rewriter(self)
        spiders = [s for s in range(self.n_spiders)
                   if not rewriter.is_boundary(s) and (
                       rewriter.ports[s] < 2 or rewriter.incidence[s]
                       or rewriter.phases[s])]
        inside = set(spiders)
        hadamards = Counter(
            frozenset({dom[0], cod[0]})
            for box, (dom, cod) in zip(self.boxes, self.box_wires)
            if is_hadamard(box) and {dom[0], cod[0]} <= inside)
        edges = sum(1 for pair, number in hadamards.items()
                    if len(pair) == 2 and number % 2)
        tcount = sum(
            1 for s in spiders if not is_clifford(rewriter.phases[s]))
        return len(spiders), edges, tcount


def is_hadamard(box) -> bool:
    """ Whether a box is :data:`H`, by identity first as equality is slow. """
    return box is H or box == H


Step = tuple[Hypergraph, Hypergraph, hypergraph.Match]
""" A rule for one match: its left and right sides, and the match. """


def is_pauli(phase: Fraction) -> bool:
    """ Whether a phase in turns, in ``[0, 1)``, is a half-turn or none. """
    return phase.denominator <= 2


def is_clifford(phase: Fraction) -> bool:
    """ Whether a phase in turns, in ``[0, 1)``, is a quarter-turn. """
    return phase.denominator in (1, 2, 4)


class Local(NamedTuple):
    """
    The neighbourhood of a spider: its phase, its Hadamard boxes to each
    other spider and to itself, its boundary ports and whether it has any
    other box, in which case no rule applies there.
    """
    phase: Fraction
    edges: dict[int, list[int]]
    loops: list[int]
    ports: int
    other: bool

    @property
    def degree(self) -> int:
        """ The number of neighbours, a boundary port being one. """
        return self.ports + len(self.edges)


def toggle(locals_: dict[int, Local],
           pairs: list[tuple[int, int]]) -> tuple[list[int], list[tuple]]:
    """
    The Hadamard edges between each pair of spiders to delete, and those to
    add where there was none: the edges are toggled.
    """
    old, new = [], []
    for a, b in pairs:
        edges = locals_[a].edges.get(b, [])
        old += edges
        if not edges:
            new.append((H, [a], [b]))
    return old, new


def classes(locals_: dict[int, Local], u: int, v: int,
            skip: set[int] = frozenset()) -> tuple[list[int], ...]:
    """
    The neighbours of ``u`` alone, of ``v`` alone and of both, other than
    ``u``, ``v`` and the spiders to skip, in a reproducible order.
    """
    near_u = [t for t in locals_[u].edges if t not in skip | {v}]
    near_v = [t for t in locals_[v].edges if t not in skip | {u}]
    both = set(near_u) & set(near_v)
    return ([t for t in near_u if t not in both],
            [t for t in near_v if t not in both],
            [t for t in near_u if t in both])


def across(*groups: list[int]) -> list[tuple[int, int]]:
    """ The pairs of spiders in different groups. """
    return [(a, b) for i, group in enumerate(groups)
            for other in groups[i + 1:] for a in group for b in other]


def spider_like(near: list[int]) -> list[tuple]:
    """
    The Hadamard edges to ``near`` of a new spider, labelled ``"kept"``,
    standing for a spider that PyZX keeps: the rule deletes the old one so
    that it is an equation on its own, the old one being inside the
    neighbourhood the equation rewrites.
    """
    return [(H, ["kept"], [t]) for t in near]


def edges_of(locals_: dict[int, Local], u: int, v: int) -> list[int]:
    """ The Hadamard edges of ``u`` and ``v``, each once. """
    return [j for t in (u, v) for target, js in locals_[t].edges.items()
            if t == u or target != u for j in js]


class Rewriter(hypergraph.Rewriter):  # pylint: disable=too-many-public-methods
    """
    A :class:`discopy.hypergraph.Rewriter` of a graph-like hypergraph with
    the rules of the ``full_reduce`` of :cite:t:`KissingerWetering20`, one
    method each: each step is a pair of hypergraphs built for its match, the
    left side the neighbourhood it rewrites and the right side what replaces
    it, applied in place. Their left sides are left-linear, so each match has
    one pushout complement, and each step is an equation of the ZX calculus
    on the nose: its right side carries the scalar, which the rewriter
    multiplies into its own.

    The rewriter keeps an index of its boxes by spider: the Hadamard edges to
    each other spider, the loops and the number of other boxes, updated with
    each box added, removed or moved by a merge, and the spiders merged so
    far. Its phases are fractions of a turn.

    Parameters:
        graph : The hypergraph to simplify, with rational phases.

    Example
    -------
    Two Hadamards between two spiders with a phase of an eighth of a turn
    cancel, and the spiders fuse into one with a quarter-turn:

    >>> rewriter = Rewriter(
    ...     (Z(1, 1, .125) >> H >> H >> Z(1, 1, .125)).to_hypergraph())
    >>> rewriter.reduce()
    1
    >>> assert rewriter.to_hypergraph() == Z(1, 1, .25).to_hypergraph()
    """
    def __init__(self, graph: Hypergraph):
        super().__init__(graph)
        self.scalar = graph.scalar
        self.phases = [Fraction(phase) % 1 for phase in self.phases]
        self.edges = [{} for _ in self.spider_types]
        self.loops = [[] for _ in self.spider_types]
        self.others = [0 for _ in self.spider_types]
        self.merged = set()
        for j, _ in enumerate(self.boxes):
            self.index(j)

    def substitute(self, left: Hypergraph, right: Hypergraph,
                   match: hypergraph.Match) -> list[int]:
        """
        Rewrite a match in place, see
        :meth:`discopy.hypergraph.Rewriter.substitute`, multiplying the scalar
        by that of ``right`` over that of ``left``.
        """
        self.scalar = self.scalar * right.scalar / left.scalar
        return super().substitute(left, right, match)

    def to_hypergraph(self) -> Hypergraph:
        """ The hypergraph rewritten so far, with its scalar. """
        return super().to_hypergraph().scaled(self.scalar)

    def index(self, j: int, remove: bool = False) -> None:
        """ Add a box to the index, or remove it from there. """
        box, (dom, cod) = self.boxes[j], self.box_wires[j]
        if not is_hadamard(box):
            for spider in set(dom + cod):
                self.others[spider] += -1 if remove else 1
            return
        groups = [self.loops[dom[0]]] if dom[0] == cod[0] else [
            self.edges[dom[0]].setdefault(cod[0], []),
            self.edges[cod[0]].setdefault(dom[0], [])]
        for group in groups:
            if remove:
                group.remove(j)
            else:
                group.append(j)
        if remove and dom[0] != cod[0] and not self.edges[dom[0]][cod[0]]:
            del self.edges[dom[0]][cod[0]], self.edges[cod[0]][dom[0]]

    def add_spider(self, typ, phase=0) -> int:
        for spiders, empty in ((self.edges, {}), (self.loops, []),
                               (self.others, 0)):
            spiders.append(empty)
        return super().add_spider(typ, phase)

    def add_box(self, box, dom, cod, offset=None) -> int:
        j = super().add_box(box, dom, cod, offset)
        self.index(j)
        return j

    def remove_box(self, j: int) -> None:
        self.index(j, remove=True)
        super().remove_box(j)

    def merge(self, spider: int, other: int) -> int:
        if spider == other:
            return spider
        moved = list(dict.fromkeys(j for j, *_ in self.incidence[spider]))
        moved += [j for j in dict.fromkeys(
            j for j, *_ in self.incidence[other]) if j not in moved]
        for j in moved:
            self.index(j, remove=True)
        kept = super().merge(spider, other)
        for j in moved:
            self.index(j)
        self.merged.add(kept)
        return kept

    def local(self, spider: int) -> Local:
        """
        The :class:`Local` neighbourhood of a spider, read off the index: it
        shares its lists, so it holds until the next step.
        """
        return Local(self.phases[spider], self.edges[spider],
                     self.loops[spider], self.ports[spider],
                     bool(self.others[spider]))

    def neighbourhood(self, *spiders: int, known: dict[int, Local] = None
                      ) -> dict[int, Local]:
        """
        The :class:`Local` neighbourhoods of spiders and their neighbours,
        adding to those already ``known``.
        """
        result = dict(known or {})
        result.update((spider, self.local(spider)) for spider in spiders
                      if spider not in result)
        for spider in spiders:
            for target in result[spider].edges:
                if target not in result:
                    result[target] = self.local(target)
        return result

    def is_boundary(self, spider: int) -> bool:
        """
        Whether a spider is a boundary behind a Hadamard edge, i.e. a
        phase-free spider with one port and one Hadamard box to another
        spider.
        """
        if self.phases[spider] or self.ports[spider] != 1\
                or len(self.incidence[spider]) != 1:
            return False
        (j, side, _), = self.incidence[spider]
        return is_hadamard(self.boxes[j])\
            and self.box_wires[j][1 - side][0] != spider

    def is_interior(self, spider: int, here: Local) -> bool:
        """ Whether a spider is a green spider away from the boundary. """
        return not here.ports and not here.other and not here.loops\
            and not self.is_boundary(spider)

    def boundaries(self, here: Local) -> list:
        """
        The boundaries of a spider: its ports, and the boundary spiders it
        has a Hadamard edge to.
        """
        return here.ports * [None] + [
            t for t in here.edges if self.is_boundary(t)]

    def near_boundary(self) -> list[int]:
        """
        The spiders with a boundary, i.e. those with a port that are not a
        boundary themselves, and those with a Hadamard edge to a boundary.
        """
        result = []
        for spider in self.dom_wires + self.cod_wires:
            if self.is_boundary(spider):
                (j, side, _), = self.incidence[spider]
                result.append(self.box_wires[j][1 - side][0])
            else:
                result.append(spider)
        return list(dict.fromkeys(result))

    def step(  # pylint: disable=too-many-arguments  # one per part of a rule
            self, interface: list[int], inner: list[int], boxes: list[int],
            new: list[tuple], *, glue: dict = None, phases: dict = None,
            core: Number = ONE) -> Step:
        """
        The rule that deletes ``boxes`` and the ``inner`` spiders, with their
        phases, keeps the ``interface`` spiders and adds the ``new`` boxes,
        each given with its domain and codomain spiders: host spiders of the
        interface, or labels of new spiders. Each spider in ``phases``, of
        either kind, gains that phase, and gluing an interface spider to
        another, as ``glue`` says, fuses them. Both sides are built with
        ``_scan=False``.

        The scalar of the right side is the ``core`` of the rule, i.e. its
        scalar were each Hadamard the matrix ``[[1, 1], [1, -1]]``, times
        ``sqrt(2)`` for each Hadamard it adds and over each it deletes.
        """
        glue = glue or {}
        phases = {s: phase for s, phase in (phases or {}).items() if phase}
        dom = Nat(len(interface))
        types = dict.fromkeys(interface + inner, Nat(1))
        left = Hypergraph(
            dom, Nat(0), tuple(self.boxes[j] for j in boxes),
            (tuple(interface), tuple(self.box_wires[j] for j in boxes), ()),
            types, phases={s: self.phases[s] for s in inner}, _scan=False)
        flat = interface + [s for j in boxes for wires in self.box_wires[j]
                            for s in wires]
        labels = dict(zip(left.flat_wires, flat))
        labels.update(enumerate(
            sorted(set(types) - set(flat)), len(set(flat))))
        boundary = tuple(glue.get(s, s) for s in interface)
        right_types = dict.fromkeys(boundary, Nat(1))
        right_types.update((s, Nat(1)) for _, *wires in new
                           for side in wires for s in side)
        right_types.update(dict.fromkeys(phases, Nat(1)))
        right = Hypergraph(
            dom, Nat(0), tuple(box for box, *_ in new),
            (boundary, tuple(tuple(map(tuple, wires)) for _, *wires in new),
             ()), right_types, phases=phases,
            scalar=core * Number(power2=len(new) - len(boxes)), _scan=False)
        return left, right, (
            tuple(labels[i] for i in range(left.n_spiders)), tuple(boxes))

    def tidy(self, spider: int) -> Step | None:
        """
        Keep the graph simple: turn a Hadamard loop into a half-turn, cancel
        parallel Hadamard edges in pairs by the Hopf law, and drop a spider,
        or a pair of them, cut off from the boundary.
        """
        here = self.local(spider)
        if here.other:
            return None
        alpha = here.phase + Fraction(len(here.loops), 2)
        if not here.ports and not here.edges:
            return self.step([], [spider], here.loops, [],
                             core=Number.sum((0, 1), (alpha, 1)))
        if not here.ports and len(here.edges) == 1:
            (target, hadamards), = here.edges.items()
            there = self.local(target)
            beta = there.phase + Fraction(len(there.loops), 2)
            if len(hadamards) == 1 and not there.ports and not there.other\
                    and list(there.edges) == [spider]:
                return self.step(
                    [], [spider, target],
                    here.loops + hadamards + there.loops, [],
                    core=Number.sum((0, 1), (alpha, 1), (beta, 1),
                                    (alpha + beta, -1)))
        parallel = {t: js for t, js in here.edges.items() if len(js) > 1}
        if not here.loops and not parallel:
            return None
        new = [(H, [spider], [t]) for t, js in parallel.items() if len(js) % 2]
        return self.step(
            [spider] + list(parallel), [],
            here.loops + [j for js in parallel.values() for j in js], new,
            phases={spider: Fraction(len(here.loops), 2)})

    def remove_identity(self, spider: int) -> Step | None:
        """
        Remove a phase-free spider between two Hadamard edges, fusing its two
        neighbours, as ``H >> H`` is the identity.
        """
        here = self.local(spider)
        if here.ports or here.phase or here.other or here.loops\
                or len(here.edges) != 2:
            return None
        (left, (first, )), (right, (second, )) = here.edges.items()
        return self.step([left, right], [spider], [first, second], [],
                         glue={right: left}, core=Number(power2=2))

    def complement(self, spider: int) -> Step | None:
        """
        Local complementation: remove an interior spider with a quarter-turn
        phase, toggling the edges between its neighbours and subtracting its
        phase from theirs.
        """
        here = self.local(spider)
        if not self.is_interior(spider, here)\
                or not is_clifford(here.phase) or is_pauli(here.phase):
            return None
        locals_ = self.neighbourhood(spider)
        near = list(here.edges)
        if any(locals_[t].other or self.is_boundary(t) for t in near):
            return None
        old_edges, new_edges = toggle(locals_, list(combinations(near, 2)))
        return self.step(
            near, [spider], [js[0] for js in here.edges.values()] + old_edges,
            new_edges, phases=dict.fromkeys(near, -here.phase),
            core=Number(1, Fraction(
                1 if here.phase == Fraction(1, 4) else -1, 8)))

    def pivot(self, spider: int) -> Step | None:
        """
        Pivot along a Hadamard edge between two spiders with Pauli phases and
        at most one boundary between them, which goes to the spider that
        stays.
        """
        here = self.local(spider)
        if here.other or here.loops or not is_pauli(here.phase)\
                or self.is_boundary(spider):
            return None
        bound = self.boundaries(here)
        if len(bound) > 1:
            return None
        for other in here.edges:
            if other in bound:
                continue
            there = self.local(other)
            if there.other or there.loops or not is_pauli(there.phase)\
                    or len(bound) + len(self.boundaries(there)) > 1:
                continue
            result = self.pivot_along(spider, other)
            if result is not None:
                return result
        return None

    def pivot_along(self, u: int, v: int) -> Step | None:
        """ The pivot along the edge from ``u`` to ``v``, see :meth:`pivot`.
        """
        locals_ = self.neighbourhood(u, v)
        there = locals_[v]
        if there.other or there.loops or not is_pauli(there.phase)\
                or self.is_boundary(v):
            return None
        bound_u = self.boundaries(locals_[u])
        bound_v = self.boundaries(there)
        if len(bound_u) + len(bound_v) > 1:
            return None
        if bound_v:
            u, v, bound_u, bound_v = v, u, bound_v, bound_u
        skip = {t for t in bound_u if t is not None}
        near_u, near_v, both = classes(locals_, u, v, skip)
        if any(locals_[t].other for t in near_u + near_v + both):
            return None
        a_u, a_v = locals_[u].phase, locals_[v].phase
        old_edges, new_edges = toggle(locals_, across(near_u, near_v, both))
        shift = dict.fromkeys(near_u, a_v)
        shift.update(dict.fromkeys(near_v, a_u))
        shift.update(dict.fromkeys(both, a_u + a_v + Fraction(1, 2)))
        interface = near_u + near_v + both
        deleted = edges_of(locals_, u, v) + old_edges
        core = Number(2, 2 * a_u * a_v)
        if not bound_u:
            return self.step(interface, [u, v], deleted, new_edges,
                             phases=shift, core=core)
        new = new_edges + spider_like(near_v + both)
        shift["kept"] = a_v
        if bound_u[0] is None:
            shift[u] = -a_u
            return self.step(interface + [u], [v], deleted,
                             new + [(H, [u], ["kept"])], phases=shift,
                             core=core / Number(power2=2))
        boundary, = bound_u
        return self.step(interface + [boundary], [u, v], deleted, new,
                         glue={boundary: "kept"}, phases=shift, core=core)

    def pivot_gadget(self, spider: int) -> Step | None:
        """
        Pivot along an edge from a Pauli spider ``u`` to a spider ``w`` with
        any other phase, which goes to a new leaf on ``u``: ``u`` becomes the
        hub of a phase gadget.
        """
        here = self.local(spider)
        if not self.is_interior(spider, here) or not is_pauli(here.phase):
            return None
        locals_ = self.neighbourhood(spider, known={spider: here})
        if any(locals_[t].other or locals_[t].loops or locals_[t].degree == 1
               or self.is_boundary(t) for t in here.edges):
            return None
        for w in here.edges:
            if not is_pauli(locals_[w].phase) and not locals_[w].ports:
                result = self.pivot_gadget_along(spider, w, locals_)
                if result is not None:
                    return result
        return None

    def pivot_gadget_along(self, u: int, w: int, locals_: dict[int, Local]
                           ) -> Step | None:
        """ The pivot of :meth:`pivot_gadget` from ``u`` to ``w``. """
        locals_ = self.neighbourhood(w, known=locals_)
        if any(locals_[t].other or self.is_boundary(t)
               for t in locals_[w].edges):
            return None
        near_u, near_w, both = classes(locals_, u, w)
        a_u, a_w = locals_[u].phase, locals_[w].phase
        old_edges, new_edges = toggle(locals_, across(near_u, near_w, both))
        shift = dict.fromkeys(near_w, a_u)
        shift.update(dict.fromkeys(both, a_u + Fraction(1, 2)))
        shift.update({"leaf": a_w, "kept": a_u})
        return self.step(
            near_u + near_w + both, [u, w],
            edges_of(locals_, w, u) + old_edges,
            new_edges + [(H, ["kept"], ["leaf"])]
            + spider_like(near_u + both), phases=shift)

    def pivot_boundary(self, spider: int) -> Step | None:
        """
        Pivot along an edge from an interior Pauli spider ``v`` to a spider
        ``w`` with one boundary, after unfusing the phase of ``w`` into a new
        phase gadget: the boundary goes to ``v``.
        """
        here = self.local(spider)
        if here.other or len(self.boundaries(here)) != 1:
            return None
        for v in here.edges:
            there = self.local(v)
            if self.is_interior(v, there) and is_pauli(there.phase):
                result = self.pivot_boundary_from(v, there)
                if result is not None:
                    return result
        return None

    def pivot_boundary_from(self, v: int, here: Local) -> Step | None:
        """ The :meth:`pivot_boundary` from an interior spider ``v``. """
        locals_ = self.neighbourhood(v, known={v: here})
        if not self.is_interior(v, here) or not is_pauli(here.phase) or any(
                locals_[t].other or self.is_boundary(t)
                or locals_[t].degree == 1 for t in here.edges):
            return None
        eligible = [t for t in here.edges if len(self.boundaries(
            locals_[t])) == 1 and not locals_[t].loops]
        if not eligible:
            return None
        quarter = [t for t in eligible if not is_pauli(locals_[t].phase)
                   and is_clifford(locals_[t].phase)]
        w = quarter[-1] if quarter else eligible[0]
        locals_.update(self.neighbourhood(w))
        bound, = self.boundaries(locals_[w])
        near_v, near_w, both = classes(
            locals_, v, w, set() if bound is None else {bound})
        if any(locals_[t].other for t in near_w):
            return None
        a_v, a_w = here.phase, locals_[w].phase
        old_edges, new_edges = toggle(locals_, across(near_v, near_w, both))
        shift = dict.fromkeys(near_w, a_v)
        shift.update(dict.fromkeys(both, a_v + Fraction(1, 2)))
        shift.update({"leaf": a_w, "hub": a_v, "kept": a_v})
        new = new_edges + [(H, ["hub"], ["leaf"])] + [
            (H, ["hub"], [t]) for t in near_v + both] + spider_like(
            near_v + both)
        interface = near_v + near_w + both
        deleted = edges_of(locals_, w, v) + old_edges
        if bound is None:
            shift[w] = -a_w
            return self.step(interface + [w], [v], deleted,
                             new + [(H, [w], ["kept"])], phases=shift,
                             core=Number(power2=-2))
        return self.step(interface + [bound], [v, w], deleted, new,
                         glue={bound: "kept"}, phases=shift)

    def gadget(self, hub: int) -> tuple[int, frozenset] | None:
        """
        The leaf of a phase gadget, i.e. the first neighbour of a Pauli
        spider with no other neighbour and a phase that is not Clifford, and
        the other neighbours of the hub.
        """
        here = self.local(hub)
        if here.other or here.loops or self.is_boundary(hub)\
                or not is_pauli(here.phase):
            return None
        for leaf in here.edges:
            if self.ports[leaf] or len(self.incidence[leaf]) > 1:
                continue
            if not is_clifford(self.phases[leaf]):
                return leaf, frozenset(here.edges) - {leaf}
        return None

    def fuse_gadgets(self, spider: int) -> Step | None:
        """
        Fuse the phase gadgets with the same neighbours into one, adding
        their phases, negated for a hub with a half-turn, and move the
        half-turn of a lone hub into its leaf.
        """
        here = self.local(spider)
        hubs = [spider] if self.gadget(spider) else []
        hubs += [t for t in here.edges
                 if len(here.edges) == 1 and not here.ports]
        for hub in hubs:
            found = self.gadget(hub)
            if found is None:
                continue
            leaf, key = found
            if not key or self.ports[hub]:
                group, leaves = [hub], [leaf]
            else:
                nearest = min(key, key=lambda t: len(self.incidence[t]))
                others = {t: self.gadget(t) for t in self.edges[nearest]
                          if t != hub and not self.ports[t]
                          and len(self.incidence[t]) - len(key) == 1}
                group = [hub] + [t for t, found in others.items()
                                 if found is not None and found[1] == key]
                leaves = [leaf] + [others[t][0] for t in group[1:]]
            phases = {t: self.phases[t] for t in group + leaves}
            if len(group) == 1 and not phases[hub]:
                continue
            total = sum(phases[l] if not phases[h] else -phases[l]
                        for h, l in zip(group, leaves)) % 1
            deleted = list(dict.fromkeys(
                j for t in group + leaves for j, *_ in self.incidence[t]))
            if self.ports[hub]:
                new = [(H, [hub], ["leaf"])]
                new += [(H, [hub], [t]) for t in sorted(key)]
                return self.step(sorted(key) + [hub], [leaf], deleted, new,
                                 phases={"leaf": total, hub: -phases[hub]},
                                 core=Number(phase=phases[leaf]))
            new = [(H, ["hub"], ["leaf"])] + [
                (H, ["hub"], [t]) for t in sorted(key)]
            return self.step(
                sorted(key), group + leaves, deleted, new,
                phases={"leaf": total}, core=Number(
                    2 * len(group) - 2, sum(phases[l] for h, l in zip(
                        group, leaves) if phases[h])))
        return None

    def copy(self, spider: int) -> Step | None:
        """
        Copy a Pauli state through its neighbour: a spider with no phase or a
        half-turn and one Hadamard edge, to an interior spider, goes with
        that neighbour, whose other neighbours each gain its phase.
        """
        here = self.local(spider)
        if here.ports or here.other or here.loops or len(here.edges) != 1\
                or not is_pauli(here.phase):
            return None
        (target, hadamards), = here.edges.items()
        there = self.local(target)
        if len(hadamards) != 1 or not self.is_interior(target, there):
            return None
        near = [t for t in there.edges if t != spider]
        return self.step(
            near, [spider, target],
            [j for js in there.edges.values() for j in js], [],
            phases=dict.fromkeys(near, here.phase),
            core=Number(2, 2 * here.phase * there.phase))

    def supplement(self, spider: int) -> Step | None:
        """
        Supplementarity: remove two interior spiders whose phases are not
        Clifford and add up to a half-turn or differ by one, with the same
        neighbours, the one aside, adding a half-turn to the neighbours when
        the phases are negated.
        """
        here = self.local(spider)
        if not self.is_interior(spider, here) or is_clifford(here.phase)\
                or not here.edges:
            return None
        near = set(here.edges)
        nearest = min(near, key=lambda t: len(self.incidence[t]))
        for other in dict.fromkeys(list(self.edges[nearest]) + list(near)):
            there = self.local(other)
            if other == spider or is_clifford(there.phase)\
                    or not self.is_interior(other, there):
                continue
            plus = (here.phase + there.phase) % 1
            minus = (here.phase - there.phase) % 1
            if other not in near and set(there.edges) == near\
                    and Fraction(1, 2) in (plus, minus):
                negate = plus == Fraction(1, 2)
            elif other in near\
                    and set(there.edges) - {spider} == near - {other}\
                    and (not plus or minus == Fraction(1, 2)):
                negate = not plus
            else:
                continue
            interface = sorted(near - {other})
            if negate and any(self.is_boundary(t) for t in interface):
                continue
            deleted = list(dict.fromkeys(
                j for t in (spider, other) for j, *_ in self.incidence[t]))
            alpha, sign = here.phase, -1 if other in near else 1
            core = Number.sum((alpha, 1), (-alpha, -sign)) if negate\
                else Number.sum((0, 1), (2 * alpha, -sign))
            return self.step(interface, [spider, other], deleted, [],
                             phases=dict.fromkeys(
                                 interface, Fraction(1, 2) * negate),
                             core=core)
        return None

    def reduce(self, rules=None) -> int:
        """
        Rewrite until no rule applies, returning the number of steps: each
        spider is tried once for each level of rules, the sparsest first, see
        :class:`Agenda`, and again whenever a step touches it or a
        neighbour.

        Parameters:
            rules : The rules by priority, :data:`RULES` by default.
        """
        rules = RULES if rules is None else rules
        alive = [s for s, typ in enumerate(self.spider_types) if typ]
        todo = [Agenda(self, alive) for _ in rules]
        stale = [anchors is not None for _, anchors in rules]
        steps = 0
        while True:
            level = next((i for i, agenda in enumerate(todo)
                          if agenda or stale[i]), None)
            if level is None:
                return steps
            if stale[level]:
                todo[level].update(rules[level][1](self))
                stale[level] = False
                continue
            spider = todo[level].pop()
            if self.spider_types[spider] is None:
                continue
            for rule in rules[level][0]:
                result = rule(self, spider)
                if result is None:
                    continue
                left, right, match = result
                touched = match[0][:len(left.dom)]
                new = self.substitute(left, right, match)
                region = dict.fromkeys(self.find(s) for s in touched)
                region.update(dict.fromkeys(
                    s for j in new for wires in self.box_wires[j]
                    for s in wires))
                region.update(dict.fromkeys(
                    self.find(s) for s in self.merged))
                self.merged.clear()
                todo[0].update(region)
                region.update(dict.fromkeys(
                    t for s in list(region) for t in self.edges[s]))
                for i, (_, anchors) in enumerate(rules[1:], 1):
                    if anchors is None:
                        todo[i].update(region)
                    else:
                        stale[i] = True
                steps += 1
                break


Rule = Callable[[Rewriter, int], Step | None]
""" A rule tried at a spider, giving the step to take or ``None``. """

RULES: list[tuple[list[Rule], Callable | None]] = [
    ([Rewriter.tidy], None),
    ([Rewriter.remove_identity, Rewriter.pivot, Rewriter.complement], None),
    ([Rewriter.pivot_boundary], Rewriter.near_boundary),
    ([Rewriter.fuse_gadgets], None),
    ([Rewriter.copy, Rewriter.supplement], None),
    ([Rewriter.pivot_gadget], None)]
"""
The rules of :meth:`Hypergraph.full_reduce` by priority, each level with the
spiders to try them at: those a step touched, or those that a function of the
rewriter gives, e.g. :meth:`Rewriter.near_boundary`. A rule is tried only
when no rule before it applies anywhere.
"""


class Agenda:
    """
    The spiders to try the rules of a level at, the one of least degree
    first, i.e. with fewest boxes, a phase counting as one: pivots and local
    complementations eliminate spiders the way Gaussian elimination does
    variables, and taking the sparsest first is what keeps the edges they
    add, the fill-in, low. A spider queued again moves to its current
    degree.

    Parameters:
        rewriter : The rewriter whose spiders these are.
        spiders : The spiders to try first.
    """
    def __init__(self, rewriter: Rewriter, spiders=()):
        self.rewriter, self.spiders, self.heap = rewriter, {}, []
        self.update(spiders)

    def update(self, spiders) -> None:
        """ Add spiders to try, or try them again, at their current degree. """
        for spider in spiders:
            degree = len(self.rewriter.incidence[spider])\
                + bool(self.rewriter.phases[spider])
            if self.spiders.get(spider) != degree:
                self.spiders[spider] = degree
                heappush(self.heap, (degree, spider))

    def pop(self) -> int | None:
        """ The next spider to try, ``None`` if there is none. """
        while self.heap:
            degree, spider = heappop(self.heap)
            if self.spiders.get(spider) == degree:
                del self.spiders[spider]
                return spider
        return None

    def __bool__(self):
        return bool(self.spiders)


Diagram.hypergraph_factory = Hypergraph
