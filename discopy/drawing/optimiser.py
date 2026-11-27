"""
Layout as optimisation: a monoidal functor from diagrams to open quadratic
programs, whose solution is a drawing.

.. warning::

    Work in progress: the interface may change without deprecation, and
    the pivotal layout of :func:`to_pivotal_drawing` can stop in a local
    minimum. :meth:`Diagram.draw` does not use this module.

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
    Optimised
    Waypoints
    Matplotlib
    TikZ
    Typst
    PlanarMap

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    to_drawing
    to_pivotal_drawing

Example
-------
>>> from discopy.monoidal import Ty, Box
>>> x = Ty('x')
>>> f, g = Box('f', x, x ** 3), Box('g', x ** 2, x)
>>> u, v = Box('u', Ty(), x ** 3), Box('v', x ** 6, Ty())
>>> diagram = u >> g @ f >> x @ f @ x @ x >> v
>>> diagram.draw(doctest="docs/_static/drawing/optimiser-before.svg")

.. image:: /_static/drawing/optimiser-before.svg
    :align: center

Where the drawing of a box stretches the boxes around it until its wires
are vertical, its open program stretches them as much as it needs to:

>>> drawing = to_drawing(diagram)  # doctest: +EXTRA
>>> drawing.width, diagram.to_drawing().width
(6.0, 6.5)
>>> drawing.draw(doctest="docs/_static/drawing/optimiser-after.svg")

.. image:: /_static/drawing/optimiser-after.svg
    :align: center

Pivotal layout
--------------

Planar layout: a pivotal diagram drawn as a planar map, its boxes free to
rotate and its wires to go anywhere, up to planar isotopy.

The rotation system of the map is fixed: the ports of a box keep their
cyclic order around it, and the ports of the boundary theirs around the
frame. A layout is a centre and a rotation for each box, a point for each
waypoint of each wire and an abscissa for each port of the boundary. Its
energy is that of Tutte's spring embedding: the sum of the squared lengths
of the segments of each wire, from the tip of a stub that leaves each box
along the normal of its side, so that a wire is short, straight and
perpendicular to the boxes it connects. Cups and caps are no boxes but
turning points of a wire: a snake is pulled straight.

A rotation is a point ``(a, b)`` on the unit circle, so that a port is at a
linear function of the variables: one step solves the quadratic program
with the circle replaced by its tangent at the current rotation, and the
overlap of boxes, and of a box and a waypoint, by the half-plane that
separates them now, then projects back to the circle. The step is halved
until the drawing is valid, i.e. no two wires cross or touch, no wire goes
through a box and no two boxes overlap, and its energy decreases: since
the cyclic order around each box is fixed, a valid drawing of a connected
diagram is isotopic to the start.

The start is the monoidal layout of :func:`to_drawing`, i.e. a
layout one level down is a feasible point one level up.

>>> from discopy.pivotal import Ty, Box, Cup, Cap
>>> x = Ty('x')
>>> snake = x @ Cap(x.r, x) >> Cup(x, x.r) @ x
>>> drawing = to_pivotal_drawing(snake)  # doctest: +EXTRA
>>> drawing.draw(doctest="docs/_static/drawing/optimiser-snake.svg")

.. image:: /_static/drawing/optimiser-snake.svg
    :align: center

The transpose of a box is the box turned upside down:

>>> f = Box('f', x, Ty('y'))
>>> drawing = to_pivotal_drawing(f.transpose())
>>> round(abs(drawing.boxes[1].angle), 2)
3.14
>>> drawing.draw(doctest="docs/_static/drawing/optimiser-transpose.svg")

.. image:: /_static/drawing/optimiser-transpose.svg
    :align: center
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import atan2, cos, hypot, sin
from itertools import count
import networkx as nx
import numpy as np

from discopy import messages, monoidal
from discopy.abc import MonoidalCategory
from discopy.drawing import backend
from discopy.drawing.planegraph import (
    Drawing, Node, PlaneGraph, Point, trailing_margin)
from discopy.utils import assert_iscomposable, factory, unbiased

Linear = tuple[tuple[int, float], ...]
""" A linear form, the coefficient of each of its variables. """

MARGIN = 0.25
""" The height above and below the band of a layer where its wires bend. """


class Waypoints(backend.Backend):  # pylint: disable=abstract-method
    """
    A backend mixin that draws the wires of an :class:`Optimised` drawing:
    each edge from or to a waypoint as a cubic Bezier, along the ``tangent``
    of a waypoint that has one, else vertical half way down, but for an
    edge of no length, which a waypoint on an end or a turning point has,
    and every other edge as its backend does.
    """
    def draw_wires(self, graph, **params):
        """ Draw the edges at a waypoint, then the others as before. """
        rest = graph.relabel_nodes()
        for source, target in list(graph.edges):
            if "wire" not in (source.kind, target.kind):
                continue
            rest.graph.remove_edge(source, target)
            start, end = graph.positions[source], graph.positions[target]
            if source.kind in ("dom", "box_cod"):
                self.draw_wire_label(source.x, *start, **params)
            if start == end:
                continue
            self.draw_bezier(
                [start, *controls(source, target, start, end), end],
                adaptive=self.on_neutral_canvas(
                    getattr(source, "x", None) or getattr(target, "x", None)))
        super().draw_wires(rest, **params)


def controls(source, target, start, end) -> tuple:
    """
    The two inner control points of the Bezier of a wire between two nodes
    at two points, see :class:`Waypoints`.
    """
    middle = (start[1] + end[1]) / 2
    first = (start[0] + source.tangent[0], start[1] + source.tangent[1])\
        if hasattr(source, "tangent") else (start[0], middle)
    second = (end[0] - target.tangent[0], end[1] - target.tangent[1])\
        if hasattr(target, "tangent") else (end[0], middle)
    return first, second


class Optimised(Drawing):
    """
    The drawing an optimiser lays out: a :class:`Drawing` whose wires go
    through waypoints, nodes of kind ``wire``, drawn by a backend with
    :class:`Waypoints`. It is a picture, not a morphism of drawings: it is
    not composed, and it has no dagger.
    """
    wire_nodes = property(lambda self: self.nodes_of_kind("wire"))

    def dagger(self):
        raise NotImplementedError(messages.OPTIMISED_DAGGER)

    def add_box_corners(self):
        """
        Draw the corners of each box, those of a box rotated by its
        ``angle`` around its node, as :meth:`Drawing.add_box_corners` draws
        those of an upright box in the frame of the box.
        """
        for j, box in enumerate(self.boxes):
            if not getattr(box, "angle", 0):
                continue
            centre = self.positions[Node("box", j=j, box=box)]
            c, s = cos(box.angle), sin(box.angle)
            ports = [self.positions[Node(kind, i=i, j=j, x=x)]
                     for kind, ty in [
                         ("box_dom", box.dom), ("box_cod", box.cod)]
                     for i, x in enumerate(ty)]
            xs = [c * (p.x - centre.x) + s * (p.y - centre.y) for p in ports]
            left, right = min(xs + [0]) - 0.25, max(xs + [0]) + 0.25
            if box.min_width and right - left < box.min_width:
                left, right = (left + right - box.min_width) / 2, (
                    left + right + box.min_width) / 2
            half = max(box.height / 2 - 0.25, 0.25)
            self.add_nodes({
                Node(f"box-corner-{a}{b}", j=j): Point(
                    centre.x + c * u - s * v, centre.y + s * u + c * v)
                for a, u in enumerate([left, right])
                for b, v in enumerate([-half, half])})
        super().add_box_corners()

    def framed(self) -> Drawing:
        result = Optimised(
            PlaneGraph(self.graph.copy(), dict(self.positions)), self.dom,
            self.cod, self.boxes, self.width, self.height, _check=False)
        result.add_box_corners()
        result.frame_dual_rail()
        return result

    @backend.matplotlib_context()
    def draw(self, **params):
        """
        Draw with the backend asked for, as :meth:`Drawing.draw` does, with
        :class:`Waypoints` mixed in.
        """
        params.setdefault("asymmetry", self.asymmetry())
        if params.get("format") == "typst":
            path, compare = backend.doctest_or_path(
                params.pop("path", None), params.pop("doctest", None))
            show = params.pop("show", True)
            compare = params.pop("compare", False) or compare
            return Typst(**params).draw(
                self.framed(), **params).output(
                    path=path, show=show, compare=compare, **params)
        if params.get("to_tikz", False):
            params.setdefault("backend", TikZ(
                use_tikzstyles=params.get("use_tikzstyles", None)))
        else:
            aspect = params.get(
                "aspect", "auto" if "figsize" in params else "equal")
            params.setdefault("backend", Matplotlib(
                figsize=params.get("figsize", None if aspect == "auto" else (
                    self.width or 1, self.height or 1)),
                linewidth=params.get("linewidth", 1),
                format=params.get("format", None)))
        return backend.draw(self.framed(), **params)


class Matplotlib(Waypoints, backend.Matplotlib):
    """ The matplotlib backend, with :class:`Waypoints`. """


class TikZ(Waypoints, backend.TikZ):
    """ The TikZ backend, with :class:`Waypoints`. """


class Typst(Waypoints, backend.Typst):
    """ The Typst backend, with :class:`Waypoints`. """


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
              square: float = .25) -> Optimised:
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

    def to_drawing(self, values) -> Optimised:
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
        return Optimised(PlaneGraph(graph, positions), self.dom, self.cod,
                         self.boxes, width, self.height, _check=False)

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


def to_drawing(diagram: monoidal.Diagram, **weights: float) -> Optimised:
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


STUB = .5
""" How far a wire leaves a box along the normal of its side. """

GAP = .25
""" The gap kept between boxes, and between a box and a waypoint. """

FRAME = .1
""" How far a waypoint keeps from the top and the bottom of the frame. """

SAMPLES = 8
""" The points sampled on each segment of a wire to check a drawing. """


@dataclass
class PlanarMap:
    """
    A pivotal diagram as a planar map, read from its monoidal drawing.

    Parameters:
        drawing : The monoidal drawing it is read from.
        boxes : The index of each box that rotates, i.e. that is not a cup
            or a cap.
        sizes : The half width and half height of each of those boxes.
        ports : The box, the offset from its centre and the outward normal
            of each port of a box that rotates, as it stands unrotated.
        wires : Each wire as a chain from one end to the other, an end
            being a port of a box or of the boundary: a waypoint is a
            ``Point``, a turning point, i.e. a cup or a cap, the triple of
            its box node and its two ports in the order of the chain.
        start : The variables of the drawing it is read from.
    """
    drawing: Drawing
    boxes: tuple[int, ...]
    sizes: tuple[tuple[float, float], ...]
    ports: dict
    wires: tuple[tuple, ...]
    start: np.ndarray

    @staticmethod
    def from_drawing(drawing: Drawing) -> PlanarMap:
        """
        The planar map of a monoidal drawing, its variables those of the
        drawing: no box is rotated yet.

        Raises:
            NotImplementedError : When a box drawn as wires is not a cup or
                a cap, or a wire is a closed loop.
        """
        wiring = {
            j for j, box in enumerate(drawing.boxes)
            if box.draw_as_wires and len(box.dom) + len(box.cod) == 2}
        if any((box.draw_as_wires or box.draw_as_spider) and j not in wiring
               for j, box in enumerate(drawing.boxes)):
            raise NotImplementedError(messages.PLANAR_NOT_PIVOTAL)
        framed = drawing.relabel_nodes()
        framed.add_box_corners()
        rectangles = framed.rectangles()
        boxes = tuple(j for j in range(len(drawing.boxes)) if j not in wiring)
        index = {j: k for k, j in enumerate(boxes)}
        sizes, centres, ports = [], [], {}
        for j in boxes:
            low, high = rectangles[j]
            centre = Point((low.x + high.x) / 2, (low.y + high.y) / 2)
            centres.append(centre)
            sizes.append(((high.x - low.x) / 2, (high.y - low.y) / 2))
            for node in drawing.nodes:
                if node.kind in ("box_dom", "box_cod") and node.j == j:
                    p = drawing.positions[node]
                    ports[node] = (index[j], (p.x - centre.x, p.y - centre.y),
                                   (0., 1. if node.kind == "box_dom" else -1.))
        ends = set(ports) | set(drawing.dom_nodes + drawing.cod_nodes)
        graph = drawing.inside.graph.to_undirected(as_view=True)
        wires, seen = [], set()
        for end in sorted(ends, key=repr):
            if end in seen:
                continue
            path, previous = [end], None
            while len(path) == 1 or path[-1] not in ends:
                step, = (n for n in graph.neighbors(path[-1]) if n != previous
                         and not (n.kind == "box" and n.j in index))
                previous = path[-1]
                path.append(step)
            seen.update((path[0], path[-1]))
            wires.append(chain(drawing, path, wiring))
        if any(j not in {item[0].j for wire in wires for item in wire[1:-1]
                         if is_turn(item)} for j in wiring):
            raise NotImplementedError(messages.PLANAR_LOOP)
        variables = [c for centre in centres for c in (*centre, 1., 0.)]
        variables += [c for wire in wires for item in wire[1:-1]
                      for c in position(drawing, item)]
        variables += [drawing.positions[n].x for n in drawing.dom_nodes]
        variables += [drawing.positions[n].x for n in drawing.cod_nodes]
        variables.append(drawing.height)
        return PlanarMap(drawing, boxes, tuple(sizes), ports, tuple(wires),
                         np.array(variables, dtype=float))

    @property
    def points(self) -> int:
        """ The number of waypoints and turning points. """
        return sum(len(wire) - 2 for wire in self.wires)

    def offset(self, kind: str) -> int:
        """ Where the variables of a kind start: points, top, bottom. """
        points = 4 * len(self.boxes)
        top = points + 2 * self.points
        bottom = top + len(self.drawing.dom)
        return {"points": points, "top": top, "bottom": bottom,
                "height": bottom + len(self.drawing.cod)}[kind]

    def chains(self) -> list[list]:
        """
        Each wire as the list of its ends and points, each end the pair of
        its node and its normal, each point its index among the points.
        """
        result, k = [], 0
        for wire in self.wires:
            middle = list(range(k, k + len(wire) - 2))
            k += len(wire) - 2
            result.append([wire[0], *middle, wire[-1]])
        return result

    def forms(self, end, tip: bool) -> tuple:
        """
        The linear forms of the abscissa and the ordinate of an end, or of
        the tip of its stub, or of a point given by its index: a dict of
        coefficients and a constant each.
        """
        if isinstance(end, int):
            i = self.offset("points") + 2 * end
            return ({i: 1.}, 0.), ({i + 1: 1.}, 0.)
        stub = STUB if tip else 0.
        if end.kind == "dom":
            return ({self.offset("top") + end.i: 1.}, 0.), (
                {self.offset("height"): 1.}, -stub)
        if end.kind == "cod":
            return ({self.offset("bottom") + end.i: 1.}, 0.), ({}, stub)
        k, (ox, oy), (mx, my) = self.ports[end]
        qx, qy = ox + stub * mx, oy + stub * my
        return ({4 * k: 1., 4 * k + 2: qx, 4 * k + 3: -qy}, 0.), (
            {4 * k + 1: 1., 4 * k + 2: qy, 4 * k + 3: qx}, 0.)

    def polygon(self, wire: list) -> list:
        """
        The control polygon of a wire as linear forms: each end, the tip of
        its stub, and its points in between.
        """
        first, *middle, last = wire
        return [self.forms(first, False), self.forms(first, True)] + [
            self.forms(p, False) for p in middle] + [
            self.forms(last, True), self.forms(last, False)]

    def terms(self) -> list[tuple[float, tuple]]:
        """
        The energy as a list of weighted squares of linear forms: the
        second differences of each control polygon, which vanish on evenly
        spaced straight wires, and its differences times a hundredth, which
        pull them short.
        """
        result = []
        for wire in self.chains():
            polygon = self.polygon(wire)
            for weight, coefficients in [
                    (1., (1., -2., 1.)), (.01, (1., -1.))]:
                result += [
                    (weight, combine(coefficients, [
                        p[axis] for p in polygon[i: i + len(coefficients)]]))
                    for i in range(len(polygon) - len(coefficients) + 1)
                    for axis in range(2)]
        return result

    def energy(self, z: np.ndarray) -> float:
        """ The value of the :meth:`terms` at the variables. """
        return sum(weight * (sum(a * z[i] for i, a in form.items())
                             + constant) ** 2
                   for weight, (form, constant) in self.terms())

    def point(self, z: np.ndarray, item, tip: bool = False) -> Point:
        """ The position of an end, the tip of its stub, or a point. """
        x, y = (sum(a * z[i] for i, a in f.items()) + c
                for f, c in self.forms(item, tip))
        return Point(x, y)

    def normal(self, z: np.ndarray, end) -> Point:
        """ The outward normal at an end, pointing into the drawing. """
        if end.kind in ("dom", "cod"):
            return Point(0., -1. if end.kind == "dom" else 1.)
        k, _, (mx, my) = self.ports[end]
        a, b = z[4 * k + 2], z[4 * k + 3]
        return Point(a * mx - b * my, b * mx + a * my)

    def corners(self, z: np.ndarray, k: int) -> np.ndarray:
        """ The four corners of a box, counterclockwise from bottom left. """
        cx, cy, a, b = z[4 * k: 4 * k + 4]
        w, h = self.sizes[k]
        return np.array([(cx + a * x - b * y, cy + b * x + a * y)
                         for x, y in [(-w, -h), (w, -h), (w, h), (-w, h)]])

    def support(self, z: np.ndarray, k: int, axis: np.ndarray) -> float:
        """ The half extent of a box along an axis. """
        _, _, a, b = z[4 * k: 4 * k + 4]
        w, h = self.sizes[k]
        return abs(w * (a * axis[0] + b * axis[1]))\
            + abs(h * (-b * axis[0] + a * axis[1]))

    def separation(self, z, k, centre, support):
        """
        The axis of a box, or of a convex shape at a centre with a support,
        along which they are furthest apart, and how far.
        """
        _, _, a, b = z[4 * k: 4 * k + 4]
        here = z[4 * k: 4 * k + 2]
        axes = [np.array([a, b]), np.array([-b, a])]
        best = (axes[0], -np.inf)
        for axis in axes + [np.array(u) for u in getattr(support, "axes", ())]:
            sign = 1. if np.dot(centre - here, axis) >= 0 else -1.
            gap = sign * np.dot(centre - here, axis) - self.support(
                z, k, axis) - support(sign * axis)
            if gap > best[1]:
                best = (sign * axis, gap)
        return best

    def curves(self, z: np.ndarray) -> list[list[tuple]]:
        """
        Each wire as cubic Bezier segments ``(p0, p1, p2, p3)`` through its
        points, in the order of its chain: a Catmull-Rom spline whose
        tangents are as long as a third of the mean of the segments around
        them, so that a wire turns round, leaving each end along its
        normal.
        """
        result = []
        for wire in self.chains():
            points = [np.array(self.point(z, item)) for item in wire]
            tangents = []
            for i, point in enumerate(points):
                if 0 < i < len(points) - 1:
                    tangents.append(tangent(*points[i - 1: i + 2]))
                    continue
                other = points[1] if i == 0 else points[-2]
                length = min(STUB, np.linalg.norm(other - point) / 2)
                normal = np.array(self.normal(z, wire[i]))
                tangents.append((1 if i == 0 else -1) * length * normal)
            result.append([
                (points[i], points[i] + tangents[i],
                 points[i + 1] - tangents[i + 1], points[i + 1])
                for i in range(len(points) - 1)])
        return result

    def is_valid(self, z: np.ndarray) -> bool:
        """
        Whether no two wires cross or come closer than a tenth, no wire
        goes through a box, no two boxes overlap and everything is between
        the top and the bottom of the frame, each wire sampled along its
        :meth:`curves`.
        """
        t = np.linspace(0, 1, SAMPLES + 1)[:, None]
        basis = [(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t),
                 t ** 3]
        lines = [np.concatenate([sum(w * p for w, p in zip(basis, segment))
                                 for segment in wire])
                 for wire in self.curves(z)]
        height = z[self.offset("height")]
        everything = np.concatenate(lines + [
            self.corners(z, k) for k in range(len(self.boxes))] + [
            np.zeros((0, 2))])
        if everything.size and ((everything[:, 1] < -1e-6).any()
                                or (everything[:, 1] > height + 1e-6).any()):
            return False
        for k in range(len(self.boxes)):
            for other in range(k + 1, len(self.boxes)):
                if self.box_gap(z, k, other) <= 1e-6:
                    return False
            local = self.to_local(z, k, np.concatenate(
                lines + [np.zeros((0, 2))]))
            w, h = self.sizes[k]
            if ((abs(local[:, 0]) < w - 1e-3)
                    & (abs(local[:, 1]) < h - 1e-3)).any():
                return False
        for i, first in enumerate(lines):
            if crosses(first, first):
                return False
            for second in lines[i + 1:]:
                if crosses(first, second) or np.min(np.linalg.norm(
                        first[:, None] - second[None], axis=2)) < .1:
                    return False
        return True

    def to_local(self, z, k, points):
        """ Points in the frame of a box, its centre at the origin. """
        cx, cy, a, b = z[4 * k: 4 * k + 4]
        d = points - np.array([cx, cy])
        return np.stack([a * d[:, 0] + b * d[:, 1],
                         -b * d[:, 0] + a * d[:, 1]], axis=1)

    def box_gap(self, z, k, other) -> float:
        """ How far apart two boxes are, by the separating axis theorem. """
        def support(axis):
            return self.support(z, other, axis)
        _, _, a, b = z[4 * other: 4 * other + 4]
        support.axes = ((a, b), (-b, a))
        return self.separation(z, k, z[4 * other: 4 * other + 2], support)[1]

    def step(self, z: np.ndarray, proximal: float = 1e-4) -> np.ndarray:
        """
        The solution of the quadratic program at the variables, its
        rotations projected back to the unit circle.
        """
        import osqp
        from scipy import sparse
        n = len(z)
        quadratic = sparse.dok_matrix((n, n))
        linear = -2 * proximal * z
        for i in range(n):
            quadratic[i, i] += 2 * proximal
        for weight, (form, constant) in self.terms():
            for i, a in form.items():
                linear[i] += 2 * weight * a * constant
                for j, b in form.items():
                    quadratic[i, j] += 2 * weight * a * b
        rows, lower, upper = [], [], []

        def at_least(form, bound, now):
            rows.append(form)
            lower.append(min(bound, now))
            upper.append(np.inf)
        for k in range(len(self.boxes)):
            a, b = z[4 * k + 2: 4 * k + 4]
            rows.append({4 * k + 2: a, 4 * k + 3: b})
            lower.append(1.)
            upper.append(1.)
            sy = self.support(z, k, np.array([0., 1.]))
            height = self.offset("height")
            at_least({4 * k + 1: 1.}, sy, z[4 * k + 1])
            at_least({height: 1., 4 * k + 1: -1.}, sy,
                     z[height] - z[4 * k + 1])
            for other in range(k + 1, len(self.boxes)):
                def support(axis, other=other):
                    return self.support(z, other, axis)
                c, d = z[4 * other + 2: 4 * other + 4]
                support.axes = ((c, d), (-d, c))
                axis, gap = self.separation(
                    z, k, z[4 * other: 4 * other + 2], support)
                at_least({4 * other: axis[0], 4 * other + 1: axis[1],
                          4 * k: -axis[0], 4 * k + 1: -axis[1]},
                         self.support(z, k, axis) + support(axis) + GAP,
                         self.support(z, k, axis) + support(axis) + gap)
            for p in range(self.points):
                i = self.offset("points") + 2 * p
                axis, gap = self.separation(z, k, z[i: i + 2], lambda _: 0.)
                at_least({i: axis[0], i + 1: axis[1],
                          4 * k: -axis[0], 4 * k + 1: -axis[1]},
                         self.support(z, k, axis) + GAP,
                         self.support(z, k, axis) + gap)
        for p in range(self.points):
            i = self.offset("points") + 2 * p
            at_least({i + 1: 1.}, FRAME, z[i + 1])
            at_least({self.offset("height"): 1., i + 1: -1.}, FRAME,
                     z[self.offset("height")] - z[i + 1])
        for kind, ty in [("top", self.drawing.dom),
                         ("bottom", self.drawing.cod)]:
            start, offsets = self.offset(kind), ty.wire_offsets()
            for i in range(len(ty)):
                if i:
                    at_least({start + i: 1., start + i - 1: -1.},
                             offsets[i] - offsets[i - 1],
                             z[start + i] - z[start + i - 1])
        matrix = sparse.dok_matrix((len(rows), n))
        for r, form in enumerate(rows):
            for i, a in form.items():
                matrix[r, i] += a
        solver = osqp.OSQP()
        solver.setup(sparse.triu(quadratic.tocsc(), format="csc"), linear,
                     matrix.tocsc(), np.array(lower), np.array(upper),
                     verbose=False, polishing=False, eps_abs=1e-7,
                     eps_rel=1e-7, max_iter=20000)
        result = solver.solve(raise_error=False)
        if result.info.status_val not in (1, 2):
            return z
        return self.project(np.array(result.x))

    def project(self, z: np.ndarray) -> np.ndarray:
        """ The variables with each rotation projected on the circle. """
        z = z.copy()
        for k in range(len(self.boxes)):
            a, b = z[4 * k + 2: 4 * k + 4]
            norm = hypot(a, b) or 1.
            z[4 * k + 2: 4 * k + 4] = a / norm, b / norm
        return z

    def solve(self, iterations: int = 100) -> np.ndarray:
        """
        Descend from the start: each :meth:`step`, halved until the
        drawing :meth:`is_valid` and its :meth:`energy` decreases, until
        no step does.

        Raises:
            ValueError : When the start itself is not valid.
        """
        z = self.start
        if not self.is_valid(z):
            raise ValueError(messages.PLANAR_START)
        energy = self.energy(z)
        for _ in range(iterations):
            target, scale = self.step(z), 1.
            while scale > 1e-3:
                candidate = self.project(z + scale * (target - z))
                value = self.energy(candidate)
                if value < energy - 1e-6 and self.is_valid(candidate):
                    z, energy = candidate, value
                    break
                scale /= 2
            else:
                break
        return z

    def to_drawing(self, z: np.ndarray) -> Optimised:
        """
        The drawing at the variables: each box rotated by its ``angle``,
        each cup and cap at its turning point, each wire through a waypoint
        node with the ``tangent`` of its spline.
        """
        drawing, curves = self.drawing, self.curves(z)
        lines = np.concatenate([np.array([s[0], s[3]]) for c in curves
                                for s in c] + [self.corners(z, k) for k in
                                               range(len(self.boxes))] + [
            np.zeros((0, 2))])
        shift = .25 - lines[:, 0].min(initial=.25)
        boxes, relabel = list(drawing.boxes), {}
        for k, j in enumerate(self.boxes):
            boxes[j] = copy.copy(drawing.boxes[j])
            boxes[j].angle = atan2(z[4 * k + 3], z[4 * k + 2])
            relabel[drawing.box_nodes[j]] = Node("box", box=boxes[j], j=j)
        positions: dict[Node, Point] = {}

        def put(node, point):
            positions[relabel.get(node, node)] = Point(
                round(float(point[0]) + shift, 6) + 0.,
                round(float(point[1]), 6) + 0.)
        for k, j in enumerate(self.boxes):
            put(drawing.box_nodes[j], z[4 * k: 4 * k + 2])
        graph = nx.DiGraph()
        graph.add_edges_from(
            (relabel.get(u, u), relabel.get(v, v)) for u, v in drawing.edges
            if "box" in (u.kind, v.kind))
        waypoints = 0
        for wire, chain_, segments in zip(self.wires, self.chains(), curves):
            for end in (wire[0], wire[-1]):
                put(end, self.point(z, end))
            pieces, current = [], [(wire[0], 0)]
            for i, item in enumerate(wire[1:-1], 1):
                if is_turn(item):
                    _, before, after = item
                    point = self.point(z, chain_[i])
                    for node in item:
                        put(node, point)
                    pieces.append(current + [(before, i)])
                    current = [(after, i)]
                else:
                    current.append((None, i))
            pieces.append(current + [(wire[-1], len(wire) - 1)])
            for piece in pieces:
                first, last = piece[0][0], piece[-1][0]
                forward = first.kind in ("dom", "box_cod")
                nodes = []
                for _, i in piece:
                    point, velocity = (segments[i][0], segments[i][1]
                                       - segments[i][0]) if i < len(segments)\
                        else (segments[-1][3], segments[-1][3]
                              - segments[-1][2])
                    velocity = velocity if forward else -velocity
                    nodes.append(Node("wire", i=waypoints, x=first.x, tangent=(
                        round(float(velocity[0]), 6) + 0.,
                        round(float(velocity[1]), 6) + 0.)))
                    waypoints += 1
                    put(nodes[-1], point)
                path = [first] + nodes + [last]
                if not forward:
                    path.reverse()
                graph.add_edges_from(
                    (relabel.get(u, u), relabel.get(v, v))
                    for u, v in zip(path, path[1:]))
        graph.add_nodes_from(positions)
        extent = np.concatenate([np.array(list(positions.values())).reshape(
            -1, 2)] + [self.corners(z, k) + [shift, 0]
                       for k in range(len(self.boxes))])
        width = extent[:, 0].max() + .25 if len(extent) else 0
        height = extent[:, 1].max(initial=float(z[self.offset("height")]))
        return Optimised(
            PlaneGraph(graph, positions), drawing.dom, drawing.cod,
            tuple(boxes), max(width, 1 if boxes else 0), height, _check=False)


def chain(drawing: Drawing, path: list, wiring: set) -> tuple:
    """
    The chain of a wire from the path of its nodes: its ends, a turning
    point for each cup or cap it goes through and a waypoint for each other
    node at a new position.
    """
    items = [path[0]]
    for i, node in enumerate(path[1:-1], 1):
        if node.kind == "box":
            items.append((node, path[i - 1], path[i + 1]))
        elif node.kind in ("box_dom", "box_cod") and node.j in wiring:
            continue
        else:
            items.append(drawing.positions[node])
    items.append(path[-1])
    result = [items[0]]
    for item in items[1:]:
        here = position(drawing, item)
        if isinstance(item, Point) and here == position(drawing, result[-1]):
            continue
        result.append(item)
    if len(result) > 2 and isinstance(result[-2], Point)\
            and position(drawing, result[-2]) == position(drawing, result[-1]):
        result.pop(-2)
    return tuple(result)


def combine(coefficients, forms) -> tuple[dict, float]:
    """ A linear combination of linear forms, each a dict and a constant. """
    result, constant = {}, 0.
    for c, (form, d) in zip(coefficients, forms):
        constant += c * d
        for i, a in form.items():
            result[i] = result.get(i, 0.) + c * a
    return result, constant


def tangent(before, here, after) -> np.ndarray:
    """
    The tangent of a spline at a point, in the direction from the point
    before to the point after, as long as a third of the mean length of the
    two segments, perpendicular to them at a turn back on itself.
    """
    direction = after - before
    if np.linalg.norm(direction) < 1e-9:
        direction = np.array([before[1] - here[1], here[0] - before[0]])
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.zeros(2)
    length = (np.linalg.norm(after - here) + np.linalg.norm(here - before)) / 6
    return direction / norm * length


def is_turn(item) -> bool:
    """ Whether an item of a chain is a turning point, i.e. a cup or a cap. """
    return isinstance(item, tuple) and isinstance(item[0], Node)


def position(drawing: Drawing, item) -> Point:
    """ The position of an item of a chain in the drawing. """
    if isinstance(item, Point):
        return item
    if is_turn(item):
        return drawing.positions[item[0]]
    return drawing.positions[item]


def crosses(first: np.ndarray, second: np.ndarray) -> bool:
    """ Whether two polylines cross, strictly. """
    a, b = first[:-1, None], first[1:, None]
    c, d = second[None, :-1], second[None, 1:]

    def side(p, q, r):
        return np.sign((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1])
                       - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))
    return bool(((side(a, b, c) * side(a, b, d) < 0)
                 & (side(c, d, a) * side(c, d, b) < 0)).any())


def to_pivotal_drawing(diagram, iterations: int = 100) -> Optimised:
    """
    The planar layout of a pivotal diagram, from its monoidal layout, see
    :func:`to_drawing`.

    Parameters:
        diagram : The diagram to draw.
        iterations : The number of steps at most.
    """
    planar = PlanarMap.from_drawing(to_drawing(diagram))
    return planar.to_drawing(planar.solve(iterations))
