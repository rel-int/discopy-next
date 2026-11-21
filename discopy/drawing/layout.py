"""
Layout as optimisation: a monoidal functor from diagrams to open quadratic
programs, whose solution is a drawing.

An :class:`OpenQP` from a type ``x`` to a type ``y`` is a convex quadratic
program over real variables, an abscissa for each wire of ``x`` at its top
and of ``y`` at its bottom among them, with a plane graph whose nodes each
sit at a variable plus an offset, at a fixed height. Composition identifies
the variables of the shared boundary and adds the costs; tensor puts two
programs side by side, with a separation constraint between the right
extent of the first and the left extent of the second. A box has a
variable for each of its sides and each of its ports, at least as wide as
:meth:`Drawing.from_box` draws it, and its ports as far apart.

The isotopy class of the diagram is fixed, only the abscissas are free:
in each layer a wire is vertical in the band of the boxes and bends in the
margins of a quarter above and below it, which all the wires of the layer
share, so two wires in the same order at both ends of a margin never
cross. The cost weighs the straightness of the bends, their horizontal
displacement, against the width of the drawing and of its boxes, see
:meth:`OpenQP.solve`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    OpenQP

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    to_drawing

Example
-------
>>> from discopy.monoidal import Ty, Box
>>> x = Ty('x')
>>> f, g = Box('f', x, x ** 3), Box('g', x ** 2, x)
>>> u, v = Box('u', Ty(), x ** 3), Box('v', x ** 6, Ty())
>>> diagram = u >> g @ f >> x @ f @ x @ x >> v
>>> diagram.draw(doctest="docs/_static/drawing/layout-before.svg")

.. image:: /_static/drawing/layout-before.svg
    :align: center

Where the drawing of a box stretches the boxes around it until its wires
are vertical, its open program stretches them as much as it needs to:

>>> drawing = to_drawing(diagram)  # doctest: +EXTRA
>>> drawing.width, diagram.to_drawing().width
(6.0, 6.5)
>>> drawing.draw(doctest="docs/_static/drawing/layout-after.svg")

.. image:: /_static/drawing/layout-after.svg
    :align: center
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
import networkx as nx
import numpy as np

from discopy import messages, monoidal
from discopy.abc import MonoidalCategory
from discopy.drawing.drawing import (
    Drawing, Node, PlaneGraph, Point, trailing_margin)
from discopy.utils import assert_iscomposable, factory, unbiased

Linear = tuple[tuple[int, float], ...]
""" A linear form, the coefficient of each of its variables. """

MARGIN = 0.25
""" The height above and below the band of a layer where its wires bend. """


@factory
@dataclass
class OpenQP(MonoidalCategory[monoidal.Ty, "OpenQP"]):
    """
    An open quadratic program over the abscissas of a drawing.

    Parameters:
        dom : The type at the top, one variable for each of its wires.
        cod : The type at the bottom, one variable for each of its wires.
        size : The number of variables, numbered from zero.
        inputs : The variable of each wire of ``dom``.
        outputs : The variable of each wire of ``cod``.
        left : The variable that bounds every node on the left.
        right : The variable that bounds every node on the right.
        bends : The horizontal displacement ``form(x) + constant`` of each
            bend, for each ``(form, constant)``, which the cost penalises.
        constraints : A ``form(x) >= bound`` for each ``(form, bound)``.
        nodes : The abscissa of each node, a linear form of the variables,
            and its height.
        edges : The edges of the plane graph.
        widths : The width of each box, a linear form of the variables,
            which the cost weighs as it does the width of the drawing.
        boxes : The boxes of the drawing, as :attr:`Drawing.boxes`.
        height : The height of the drawing.

    A pivotal layout would add an angle to each box and a rotation system
    to the constraints, and a symmetric one the place of each crossing to
    the cost: the variables are only numbers here, the nodes say what they
    mean.

    >>> from discopy.monoidal import Ty, Box
    >>> qp = OpenQP.from_drawing(Box('f', Ty('x'), Ty('y')).to_drawing())
    >>> qp.size, qp.inputs, qp.outputs, qp.height
    (8, (0,), (1,), 1)
    """
    ob = monoidal.Ty

    dom: monoidal.Ty
    cod: monoidal.Ty
    size: int
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    left: int
    right: int
    bends: tuple[tuple[Linear, float], ...] = ()
    constraints: tuple[tuple[Linear, float], ...] = ()
    nodes: tuple[tuple[Node, tuple[Linear, float]], ...] = ()
    edges: tuple[tuple[Node, Node], ...] = ()
    widths: tuple[Linear, ...] = ()
    boxes: tuple[monoidal.Box, ...] = ()
    height: float = 0

    @property
    def wire_count(self) -> int:
        """ The number of waypoints, i.e. nodes of kind ``wire``. """
        return sum(node.kind == "wire" for node, _ in self.nodes)

    @staticmethod
    def id(dom: monoidal.Ty | None = None, height: float = 0) -> OpenQP:
        """
        The identity on a type, of height zero, or the wires of a type
        stretched to a height, vertical in the band and free to bend in the
        margins, i.e. three variables for each wire.

        Parameters:
            dom : The type of the wires.
            height : Their height.

        >>> from discopy.monoidal import Ty
        >>> OpenQP.id(Ty('x', 'y')).size
        4
        >>> OpenQP.id(Ty('x', 'y'), height=1).size
        8
        """
        dom = (monoidal.Ty() if dom is None else dom).to_drawing()
        rows = 3 if height else 1
        size, n = rows * len(dom) + 2, len(dom)
        left, right = size - 2, size - 1
        offsets = dom.wire_offsets()
        tops, bands, bottoms = (
            tuple(range(k * n, (k + 1) * n)) if rows == 3 else tuple(range(n))
            for k in range(3))
        margin = min(MARGIN, height / 2)
        constraints: list[tuple[Linear, float]] = []
        nodes: list[tuple[Node, tuple[Linear, float]]] = []
        bends, edges = [], []
        for row in dict.fromkeys([tops, bands, bottoms]):
            constraints += separate(row, offsets, dom, left, right)
        for i, x in enumerate(dom):
            dom_node, cod_node = Node("dom", i=i, x=x), Node("cod", i=i, x=x)
            if not height:
                nodes += [(dom_node, (((i, 1.), ), 0)),
                          (cod_node, (((i, 1.), ), 0))]
                edges.append((dom_node, cod_node))
                continue
            top, bottom = (Node("wire", i=2 * i + k, x=x) for k in range(2))
            nodes += [(dom_node, (((tops[i], 1.), ), height)),
                      (top, (((bands[i], 1.), ), height - margin)),
                      (bottom, (((bands[i], 1.), ), margin)),
                      (cod_node, (((bottoms[i], 1.), ), 0))]
            edges += [(dom_node, top), (top, bottom), (bottom, cod_node)]
            bends += [displacement(tops[i], bands[i]),
                      displacement(bands[i], bottoms[i])]
        if not dom:
            constraints.append((((right, 1.), (left, -1.)), 0))
        return OpenQP(dom, dom, size, tops, bottoms, left, right,
                      tuple(bends), tuple(constraints), tuple(nodes),
                      tuple(edges), (), (), height)

    @staticmethod
    def from_drawing(drawing: Drawing) -> OpenQP:
        """
        The open program of the drawing of one box: a variable for each
        wire at the boundary, for each port and for each side of the box,
        which is at least as wide as drawn, its ports at least as far apart
        and a quarter from its sides, its node at its centre. Each port has a
        waypoint, from which its wire bends to the boundary in a margin: a
        drawing with a port on the boundary, e.g. of a swap or a cup, is
        padded with a margin above and below. A drawing with no box is an
        identity.

        Parameters:
            drawing : The drawing of a box, e.g. :meth:`Drawing.from_box`.

        Raises:
            NotImplementedError : When the drawing has more than one box, or
                draws a bubble or a frame.
        """
        if not drawing.boxes:
            return OpenQP.id(drawing.dom)
        if len(drawing.boxes) > 1 or any(
                getattr(drawing.boxes[0], attr, False) for attr in (
                    "bubble_opening", "bubble_closing", "frame_boundary")):
            raise NotImplementedError(messages.LAYOUT_NOT_A_BOX)
        box = drawing.boxes[0]
        pad = MARGIN if any(
            drawing.positions[port].y in (0, drawing.height)
            for port in drawing.box_dom_nodes + drawing.box_cod_nodes) else 0
        height = drawing.height + 2 * pad
        framed = drawing.relabel_nodes()
        framed.add_box_corners()
        (low, high), = framed.rectangles()
        fresh = count()
        inputs = tuple(next(fresh) for _ in drawing.dom)
        outputs = tuple(next(fresh) for _ in drawing.cod)
        ports = {node: next(fresh) for node in sorted(
            drawing.box_dom_nodes + drawing.box_cod_nodes, key=repr)}
        side, other_side, left, right = (next(fresh) for _ in range(4))
        nodes: list[tuple[Node, tuple[Linear, float]]] = [
            (node, (((inputs[node.i], 1.), ), height))
            for node in drawing.dom_nodes] + [
            (node, (((outputs[node.i], 1.), ), 0))
            for node in drawing.cod_nodes]
        nodes.append((drawing.box_nodes[0], (
            ((side, .5), (other_side, .5)),
            drawing.positions[drawing.box_nodes[0]].y + pad)))
        nodes += [(port, (((variable, 1.), ), min(max(
            drawing.positions[port].y + pad, MARGIN), height - MARGIN)))
            for port, variable in ports.items()]
        positions, bends = dict(nodes), []
        edges: list[tuple[Node, Node]] = []
        for source, target in drawing.edges:
            if source.kind != "dom" and target.kind != "cod":
                edges.append((source, target))
                continue
            port = target if source.kind == "dom" else source
            waypoint = Node("wire", i=len(bends), x=port.x)
            nodes.append((waypoint, positions[port]))
            ends = (source, waypoint, port) if source.kind == "dom"\
                else (port, waypoint, target)
            edges += list(zip(ends, ends[1:]))
            bends.append(displacement(*(
                positions[node][0][0][0] for node in (ends[0], ends[2]))))
        constraints: list[tuple[Linear, float]] = [
            (((other_side, 1.), (side, -1.)), high.x - low.x),
            (((side, 1.), (left, -1.)), low.x),
            (((right, 1.), (other_side, -1.)), drawing.width - high.x)]
        for variable in ports.values():
            constraints += [(((variable, 1.), (side, -1.)), MARGIN),
                            (((other_side, 1.), (variable, -1.)), MARGIN)]
        for kind, ty, row in [("box_dom", drawing.dom, inputs),
                              ("box_cod", drawing.cod, outputs)]:
            offsets = ty.wire_offsets()
            constraints += separate(row, offsets, ty, left, right)
            column = [variable for port, variable in sorted(
                ports.items(), key=lambda item: item[0].i)
                if port.kind == kind]
            constraints += [
                (((b, 1.), (a, -1.)), offsets[i + 1] - offsets[i])
                for i, (a, b) in enumerate(zip(column, column[1:]))]
        return OpenQP(
            drawing.dom, drawing.cod, right + 1, inputs, outputs, left, right,
            tuple(bends), tuple(constraints), tuple(nodes),
            tuple(sorted(edges, key=repr)),
            (((other_side, 1.), (side, -1.)), ), (box, ), height)

    @unbiased
    def then(self, other: OpenQP) -> OpenQP:
        """
        Compose two open programs: the variables of the outputs of the first
        are those of the inputs of the second, the costs and the constraints
        add up and the extent bounds both.

        >>> from discopy.monoidal import Ty, Box
        >>> x = Ty('x')
        >>> f = OpenQP.from_drawing(Box('f', x, x).to_drawing())
        >>> (f >> f).size
        17
        """
        assert_iscomposable(self, other)
        if not self.height:
            return other
        if not other.height:
            return self
        fresh = count(self.size)
        rename = {
            v: self.outputs[other.inputs.index(v)] if v in other.inputs
            else next(fresh) for v in range(other.size)}
        left, right = next(fresh), next(fresh)
        nodes = {node: (form, y + other.height)
                 for node, (form, y) in self.nodes if node.kind != "cod"}
        relabel = relabelling(len(self.boxes), self.wire_count)
        nodes.update({
            relabel(node): (rename_form(form, rename), y)
            for node, (form, y) in other.nodes if node.kind != "dom"})
        above = {target: source for source, target in self.edges}
        below = dict(other.edges)
        edges = [edge for edge in self.edges if edge[1].kind != "cod"] + [
            (relabel(source), relabel(target))
            for source, target in other.edges if source.kind != "dom"] + [
            (above[Node("cod", i=i, x=x)],
             relabel(below[Node("dom", i=i, x=x)]))
            for i, x in enumerate(self.cod)]
        return OpenQP(
            self.dom, other.cod, right + 1, self.inputs,
            tuple(rename[v] for v in other.outputs), left, right,
            self.bends + rename_bends(other.bends, rename),
            self.constraints + rename_constraints(
                other.constraints, rename) + (
                (((self.left, 1.), (left, -1.)), 0),
                (((rename[other.left], 1.), (left, -1.)), 0),
                (((right, 1.), (self.right, -1.)), 0),
                (((right, 1.), (rename[other.right], -1.)), 0)),
            tuple(sorted(nodes.items(), key=lambda item: repr(item[0]))),
            tuple(sorted(edges, key=repr)),
            self.widths + rename_widths(other.widths, rename),
            self.boxes + other.boxes, self.height + other.height)

    @unbiased
    def tensor(self, other: OpenQP) -> OpenQP:
        """
        Put two open programs side by side, the shorter stretched to the
        height of the taller, with the left extent of the second at least
        the right extent of the first, plus a quarter when one has no box.

        >>> from discopy.monoidal import Ty, Box
        >>> x = Ty('x')
        >>> f = OpenQP.from_drawing(Box('f', x, x).to_drawing())
        >>> (f @ x).height, (f @ x).size
        (1, 13)
        """
        if not self.dom and not self.cod and not self.boxes:
            return other
        if not other.dom and not other.cod and not other.boxes:
            return self
        height = max(self.height, other.height)
        first, second = self.stretch(height), other.stretch(height)
        rename = {v: v + first.size for v in range(second.size)}
        relabel = relabelling(len(first.boxes), first.wire_count,
                              len(first.dom), len(first.cod))
        nodes = first.nodes + tuple(
            (relabel(node), (rename_form(form, rename), y))
            for node, (form, y) in second.nodes)
        edges = first.edges + tuple(
            (relabel(source), relabel(target))
            for source, target in second.edges)
        gap = 0 if first.boxes and second.boxes else MARGIN
        return OpenQP(
            first.dom @ second.dom, first.cod @ second.cod,
            first.size + second.size,
            first.inputs + tuple(rename[v] for v in second.inputs),
            first.outputs + tuple(rename[v] for v in second.outputs),
            first.left, rename[second.right],
            first.bends + rename_bends(second.bends, rename),
            first.constraints + rename_constraints(
                second.constraints, rename) + ((
                    ((rename[second.left], 1.), (first.right, -1.)), gap), ),
            tuple(sorted(nodes, key=lambda item: repr(item[0]))),
            tuple(sorted(edges, key=repr)),
            first.widths + rename_widths(second.widths, rename),
            first.boxes + second.boxes, height)

    def stretch(self, height: float) -> OpenQP:
        """
        The open program stretched to a height, by wires as tall as half
        the difference above and below it, see :meth:`id`.

        >>> from discopy.monoidal import Ty
        >>> OpenQP.id(Ty('x')).stretch(2).height
        2
        """
        if height <= self.height:
            return self
        if not self.height:
            return OpenQP.id(self.dom, height)
        half = (height - self.height) / 2
        return OpenQP.id(self.dom, half) >> self >> OpenQP.id(self.cod, half)

    def solve(self, width: float = 1., bend: float = 1.,
              square: float = .25) -> Drawing:
        """
        The drawing that minimises its cost, with its left extent at zero, by
        `OSQP <https://osqp.org>`_: the width of the drawing and of each of
        its boxes, the absolute value of the
        displacement of each bend, which keeps a wire straight unless
        bending it pays, and its square, which spreads the bends that do,
        each times a weight. The absolute values are auxiliary variables
        above both signs, so that the program stays quadratic.

        Parameters:
            width : The weight of the width of the drawing and of each box.
            bend : The weight of the absolute value of each bend.
            square : The weight of its square.

        Raises:
            ValueError : When a weight is negative, which would make the
                program non-convex or unbounded, or when the solver finds
                no solution.

        >>> from discopy.monoidal import Ty, Box
        >>> x = Ty('x')
        >>> f = Box('f', x, x)
        >>> drawing = OpenQP.functor()(f @ f).solve()  # doctest: +EXTRA
        >>> drawing.width
        2.0
        """
        if min(width, bend, square) < 0:
            raise ValueError(messages.LAYOUT_WEIGHTS)
        import osqp
        from scipy import sparse
        n, m = self.size, len(self.bends)
        quadratic, linear = sparse.dok_matrix((n + m, n + m)), np.zeros(n + m)
        rows, bounds = [], []
        for k, (form, constant) in enumerate(self.bends):
            for i, a in form:
                linear[i] += 2 * square * constant * a
                for j, b in form:
                    quadratic[i, j] += 2 * square * a * b
            linear[n + k] = bend
            for sign in (1, -1):
                rows.append(((n + k, 1.), ) + tuple(
                    (i, -sign * a) for i, a in form))
                bounds.append(sign * constant)
        linear[self.right] += width
        linear[self.left] -= width
        for form in self.widths:
            for i, a in form:
                linear[i] += width * a
        rows += [form for form, _ in self.constraints] + [((self.left, 1.), )]
        bounds += [bound for _, bound in self.constraints]
        matrix = sparse.dok_matrix((len(rows), n + m))
        for k, form in enumerate(rows):
            for i, a in form:
                matrix[k, i] += a
        lower = np.array(bounds + [0.])
        upper = np.array([np.inf] * len(bounds) + [0.])
        solver = osqp.OSQP()
        solver.setup(
            sparse.triu(quadratic.tocsc(), format="csc"), linear,
            matrix.tocsc(), lower, upper, verbose=False, polishing=True,
            eps_abs=1e-9, eps_rel=1e-9, max_iter=100000)
        result = solver.solve(raise_error=False)
        if result.info.status_val not in (1, 2):
            raise ValueError(messages.LAYOUT_INFEASIBLE.format(
                result.info.status))
        return self.to_drawing([round(float(v), 6) + 0. for v in result.x])

    def to_drawing(self, values) -> Drawing:
        """
        The drawing at a value of the variables, its left extent at zero.

        Parameters:
            values : A value for each variable.
        """
        positions = {
            node: Point(sum(a * values[i] for i, a in form) + 0., y)
            for node, (form, y) in self.nodes}
        graph = nx.DiGraph()
        graph.add_nodes_from(positions)
        graph.add_edges_from(self.edges)
        width = max([values[self.right]] + [p.x for p in positions.values()])
        return Drawing(PlaneGraph(graph, positions), self.dom, self.cod,
                       self.boxes, width, self.height)

    @staticmethod
    def functor(dom: type | None = None):
        """
        The monoidal functor from diagrams to open programs, which sends a
        box to the program of its drawing, see :meth:`from_drawing`.

        Parameters:
            dom : The class of diagrams, :class:`monoidal.Diagram` by default.
        """
        return monoidal.Functor(
            lambda x: x.to_drawing(),
            lambda box: OpenQP.from_drawing(box.to_drawing()),
            dom or monoidal.Diagram, OpenQP)


def to_drawing(diagram: monoidal.Diagram, **weights: float) -> Drawing:
    """
    The drawing of a diagram laid out by its open program, see
    :meth:`OpenQP.solve`.

    Parameters:
        diagram : The diagram to draw.
        weights : The weights of :meth:`OpenQP.solve`.
    """
    return OpenQP.functor(diagram.factory)(diagram).solve(**weights)


def displacement(top: int, bottom: int, constant: float = 0.) -> tuple:
    """ The horizontal displacement of a bend from one variable to another. """
    if top == bottom:
        return (((top, 1.), ), constant)
    return (((top, 1.), (bottom, -1.)), constant)


def separate(row, offsets, ty, left, right) -> list[tuple[Linear, float]]:
    """
    The constraints that keep the wires of a row in their order, as far
    apart as the offsets of their type, and between the left and the right
    extent, a quarter away and its trailing margin for the last.
    """
    if not row:
        return []
    return [
        (((b, 1.), (a, -1.)), offsets[i + 1] - offsets[i])
        for i, (a, b) in enumerate(zip(row, row[1:]))] + [
        (((row[0], 1.), (left, -1.)), MARGIN),
        (((right, 1.), (row[-1], -1.)),
         MARGIN + trailing_margin(ty.inside[-1]))]


def relabelling(boxes: int, wires: int, dom: int = 0, cod: int = 0):
    """ Shift the nodes of a program past those of another. """
    def relabel(node):
        if "box" in node.kind:
            return node.shift_j(boxes)
        if node.kind == "wire":
            return node.shift_i(wires)
        return node.shift_i(dom if node.kind == "dom" else cod)
    return relabel


def rename_form(form: Linear, rename) -> Linear:
    """ A linear form with its variables renamed. """
    return tuple((rename[i], a) for i, a in form)


def rename_widths(widths, rename) -> tuple:
    """ The widths with their variables renamed. """
    return tuple(rename_form(form, rename) for form in widths)


def rename_bends(bends, rename) -> tuple:
    """ The bends with their variables renamed. """
    return tuple((tuple((rename[i], a) for i, a in form), constant)
                 for form, constant in bends)


def rename_constraints(constraints, rename) -> tuple:
    """ The constraints with their variables renamed. """
    return tuple((tuple((rename[i], a) for i, a in form), bound)
                 for form, bound in constraints)
