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
    Swap
    Spider
    Z
    Y
    X
    Scalar
"""

import cmath
from fractions import Fraction
from functools import reduce
from math import pi

from discopy import cat, frobenius, rigid, tensor, quantum
from discopy.cat import factory
from discopy.quantum.circuit import qubit, Circuit
from discopy.quantum.gates import (
    Bra, Ket, Rz, Rx, U1, CX, CZ, Controlled, format_number)
from discopy.quantum.gates import Scalar as GatesScalar
from discopy.rigid import Sum, Nat
from discopy.tensor import Dim, backend
from discopy.utils import assert_isatomic, factory_name


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
            elif target > source:
                swaps = Id(source)\
                    @ Diagram.swap(1, target - source)\
                    @ Id(len(scan) - target - 1)
                scan = scan[:source] + scan[source + 1:target]\
                    + (scan[source],) + scan[target:]
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
            scan, swaps = move(scan, scan.index(node), target)
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

H = Box('H', Nat(1), Nat(1), data=[x / 2 ** .5 for x in (1, 1, 1, -1)])
H.dagger = lambda: H
H.draw_as_spider = True
H.drawing_name, H.tikzstyle_name, = '', 'H'
H.color, H.shape = "yellow", "rectangle"

SWAP = Swap(Nat(1), Nat(1))
Diagram.box_factory = Box
Diagram.swap_factory, Diagram.sum_factory = Swap, Sum
Diagram.permutation_factory = Permutation
Id = Diagram.id
