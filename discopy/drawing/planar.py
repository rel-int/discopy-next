"""
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

The start is the monoidal layout of :mod:`discopy.drawing.layout`, i.e. a
layout one level down is a feasible point one level up.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    PlanarMap

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    to_drawing

Example
-------
>>> from discopy.pivotal import Ty, Box, Cup, Cap
>>> x = Ty('x')
>>> snake = x @ Cap(x.r, x) >> Cup(x, x.r) @ x
>>> drawing = to_drawing(snake)  # doctest: +EXTRA
>>> drawing.draw(doctest="docs/_static/drawing/planar-snake.svg")

.. image:: /_static/drawing/planar-snake.svg
    :align: center

The transpose of a box is the box turned upside down:

>>> f = Box('f', x, Ty('y'))
>>> drawing = to_drawing(f.transpose())
>>> round(abs(drawing.boxes[1].angle), 2)
3.14
>>> drawing.draw(doctest="docs/_static/drawing/planar-transpose.svg")

.. image:: /_static/drawing/planar-transpose.svg
    :align: center
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import atan2, hypot

import networkx as nx
import numpy as np

from discopy import messages
from discopy.drawing import layout
from discopy.drawing.drawing import Drawing, Node, PlaneGraph, Point

STUB = .5
""" How far a wire leaves a box along the normal of its side. """

GAP = .25
""" The gap kept between boxes, and between a box and a waypoint. """

MARGIN = .1
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
            at_least({i + 1: 1.}, MARGIN, z[i + 1])
            at_least({self.offset("height"): 1., i + 1: -1.}, MARGIN,
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

    def to_drawing(self, z: np.ndarray) -> Drawing:
        """
        The drawing at the variables: each box rotated by its ``angle``,
        each cup and cap at its turning point, each wire through a waypoint
        node with the ``tangent`` of its spline.
        """
        drawing, curves = self.drawing, self.curves(z)
        lines = np.concatenate([np.array([s[0], s[3]]) for c in curves
                                for s in c] + [self.corners(z, k) for k in
                                               range(len(self.boxes))])
        shift = .25 - min(lines[:, 0].min(), .25)
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
        count = 0
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
                    nodes.append(Node("wire", i=count, x=first.x, tangent=(
                        round(float(velocity[0]), 6) + 0.,
                        round(float(velocity[1]), 6) + 0.)))
                    count += 1
                    put(nodes[-1], point)
                path = [first] + nodes + [last]
                if not forward:
                    path.reverse()
                graph.add_edges_from(
                    (relabel.get(u, u), relabel.get(v, v))
                    for u, v in zip(path, path[1:]))
        graph.add_nodes_from(positions)
        width = max(p.x for p in positions.values()) + .25\
            if positions else 0
        height = max([float(z[self.offset("height")])]
                     + [p.y for p in positions.values()])
        return Drawing(PlaneGraph(graph, positions), drawing.dom, drawing.cod,
                       tuple(boxes), max(width, 1 if boxes else 0), height)


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


def to_drawing(diagram, iterations: int = 100) -> Drawing:
    """
    The planar layout of a pivotal diagram, from its monoidal layout, see
    :func:`discopy.drawing.layout.to_drawing`.

    Parameters:
        diagram : The diagram to draw.
        iterations : The number of steps at most.
    """
    planar = PlanarMap.from_drawing(layout.to_drawing(diagram))
    return planar.to_drawing(planar.solve(iterations))
