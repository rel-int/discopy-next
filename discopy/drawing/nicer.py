"""
How nice a drawing looks, as a score to maximise.

A layout of a diagram is a choice of foliation, see
:meth:`~discopy.monoidal.Diagram.foliations`, and of its geometry. The
:func:`hand_score` of its drawing is a linear map of hand features, each a
cost: the lower, the nicer. Its weights start equal, to be fitted by
Bradley-Terry on the team's votes between two layouts, i.e.
``P(a nicer than b) = sigmoid(score(a) - score(b))``.

Summary
-------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    wires
    rectangles
    features
    hand_score
    make_nicer
    stretch
    box_ports
    corners
    shape
    split
    clearance
    nicer_layout

Example
-------
>>> from discopy.monoidal import Ty, Box
>>> x = Ty('x')
>>> f, g, h = Box('f', x, x), Box('g', x, x), Box('h', x @ x, x)
>>> diagram = f @ x @ x >> x @ g @ x >> x @ x @ f >> h @ x >> h
>>> diagram.draw(doctest="docs/_static/drawing/nicer-before.svg")

.. image:: /_static/drawing/nicer-before.svg
    :align: center

>>> nicer = make_nicer(diagram)
>>> print(nicer)
f @ g @ f >> h @ x >> h
>>> assert hand_score(nicer.to_drawing()) >= hand_score(
...     diagram.to_drawing())
>>> nicer.draw(doctest="docs/_static/drawing/nicer-after.svg")

.. image:: /_static/drawing/nicer-after.svg
    :align: center
"""

from __future__ import annotations

from bisect import bisect_right
from itertools import accumulate
from math import hypot, log
from typing import TYPE_CHECKING, Callable, Sequence

from discopy.drawing.backend import Backend
from discopy.drawing.planegraph import Drawing, Node, Point
from discopy import messages
from discopy.utils import text_width

if TYPE_CHECKING:
    from discopy import monoidal

WEIGHTS = {
    "wire_length": 1.,
    "bends": 1.,
    "slope": 1.,
    "aspect": 1.,
    "asymmetry": 1.,
    "whitespace": 1.,
    "misalignment": 1.,
    "label_overlap": 1.,
    "area": 1.,
}
""" The weight of each feature in :func:`hand_score`, equal until fitted. """


def wires(drawing: Drawing) -> list[tuple[Point, Point]]:
    """
    The segments of a drawing that are wires, i.e. the edges a backend draws,
    see :meth:`Backend.visible_edges`, with each strand of a swap, a braid or
    a permutation from its input to its output, see
    :meth:`Backend.crossing_wires`, and each rail of a dual rail box, see
    :meth:`Backend.dual_rail_wires`.

    >>> from discopy.monoidal import Ty, Box
    >>> for wire in wires(Box('f', Ty('x'), Ty('x')).to_drawing()):
    ...     print(*wire)
    Point(x=0.5, y=1) Point(x=0.5, y=0.75)
    Point(x=0.5, y=0.25) Point(x=0.5, y=0)
    """
    crossings = [node for node in drawing.box_nodes
                 if node.box.draw_as_permutation or node.box.is_crossing]
    dual_rails = [node for node in drawing.box_nodes if any(
        getattr(node.box, f"draw_as_dual_rail_{kind}", False)
        for kind in ("braid", "twist", "cup", "cap"))]
    edges = [(u, v) for u, v in Backend.visible_edges(drawing)
             if u not in crossings and v not in crossings] + [
        edge for node in crossings for edge in Backend.crossing_wires(node)
    ] + [edge for node in dual_rails for edge in Backend.dual_rail_wires(node)]
    return [(drawing.positions[u], drawing.positions[v]) for u, v in edges
            if drawing.positions[u] != drawing.positions[v]]


def rectangles(drawing: Drawing) -> list[tuple[float, float, float, float]]:
    """
    The rectangle ``(left, bottom, right, top)`` of each box: the one
    :meth:`Drawing.add_box_corners` draws, see :meth:`Drawing.rectangles`,
    joined with the label it draws, centred at its box node: spiders draw
    their label too.

    >>> from discopy.monoidal import Ty, Box
    >>> rectangles(Box('f', Ty('x'), Ty('x')).to_drawing())
    [(0.25, 0.25, 0.75, 0.75)]
    """
    framed = drawing.relabel_nodes()
    framed.add_box_corners()
    result = []
    for node, (low, high) in zip(
            sorted(framed.box_nodes, key=lambda node: node.j),
            framed.rectangles()):
        box, x = node.box, framed.positions[node].x
        shown = getattr(box, "draw_as_spider", False)\
            or not getattr(box, "no_label", False)
        half = text_width(
            str(getattr(box, "drawing_name", box.name)) if shown else "") / 2
        result.append((min(low.x, x - half), low.y,
                       max(high.x, x + half), high.y))
    return result


def mean(xs: list[float]) -> float:
    """ The mean of a list, zero when it is empty. """
    return sum(xs) / len(xs) if xs else 0.


def features(drawing: Drawing) -> dict[str, float]:
    """
    The hand features of a drawing, each a cost, scale-free where it can be:

    * ``wire_length``: the mean length of a wire, over the drawing's height,
    * ``bends``: the fraction of wires that are not vertical,
    * ``slope``: the mean of ``|dx| / |dy|`` over wires,
    * ``aspect``: ``|log(width / height)|``,
    * ``asymmetry``: how far the nodes are from their mirror image, over the
      width,
    * ``whitespace``: the variance of the gaps between nodes at the same
      height, over the square of their mean,
    * ``misalignment``: the mean distance between the centres of the inputs
      and of the outputs of a box that has both,
    * ``label_overlap``: the fraction of pairs of boxes side by side, i.e.
      whose rectangles share a range of heights, that overlap, see
      :func:`rectangles`,
    * ``area``: the log of the area of the drawing per box, zero up to one
      unit square per box.

    >>> from discopy.monoidal import Ty, Box
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> features((f @ g).to_drawing())["bends"]
    0.0
    >>> features((f @ g).foliation().to_drawing())["area"]
    0.0
    """
    segments = wires(drawing)
    width, height = drawing.width or 1, drawing.height or 1
    positions = [p for node, p in drawing.positions.items()
                 if not node.kind.startswith("box-corner")]
    mirrored = [Point(width - p.x, p.y) for p in positions]
    rows: dict[float, list[float]] = {}
    for p in positions:
        rows.setdefault(p.y, []).append(p.x)
    gaps = [b - a for xs in rows.values()
            for a, b in zip(sorted(xs), sorted(xs)[1:]) if b > a]
    gap_mean = mean(gaps)
    ports: dict[int, dict[str, list[float]]] = {
        j: {"box_dom": [], "box_cod": []} for j in range(len(drawing.boxes))}
    for node in drawing.nodes:
        if node.kind in ("box_dom", "box_cod"):
            ports[node.j][node.kind].append(drawing.positions[node].x)
    boxes = rectangles(drawing)
    pairs = [(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:]
             if a[1] < b[3] and b[1] < a[3]]
    return {
        "wire_length": mean(
            [hypot(a.x - b.x, a.y - b.y) for a, b in segments]) / height,
        "bends": mean([float(a.x != b.x) for a, b in segments]),
        "slope": mean([abs(a.x - b.x) / abs(a.y - b.y)
                       for a, b in segments if a.y != b.y]),
        "aspect": abs(log(width / height)),
        "asymmetry": mean([min(
            hypot(p.x - q.x, p.y - q.y) for q in mirrored)
            for p in positions]) / width,
        "whitespace": mean([(gap - gap_mean) ** 2 for gap in gaps])
        / (gap_mean ** 2 or 1),
        "misalignment": mean([abs(mean(xs["box_dom"]) - mean(xs["box_cod"]))
                              for xs in ports.values() if all(xs.values())]),
        "label_overlap": mean([
            float(a[0] < b[2] and b[0] < a[2]) for a, b in pairs]),
        "area": log(max(1., width * height / max(1, len(drawing.boxes)))),
    }


def hand_score(
        drawing: Drawing, weights: dict[str, float] | None = None) -> float:
    """
    The score of a drawing, minus the weighted sum of its :func:`features`.

    Parameters:
        drawing : The drawing to score.
        weights : The weight of each feature, :data:`WEIGHTS` by default.

    >>> from discopy.monoidal import Ty, Box
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> assert hand_score((f @ g).foliation().to_drawing()) > hand_score(
    ...     (f @ g).to_drawing())
    """
    weights = WEIGHTS if weights is None else weights
    costs = features(drawing)
    return -sum(weight * costs[name] for name, weight in weights.items())


def make_nicer(
        diagram: "monoidal.Diagram",
        score: Callable[[Drawing], float] | None = None,
        limit: int = 64) -> "monoidal.Diagram":
    """
    The foliation of a diagram whose drawing scores best, the diagram itself
    when none scores better: the result is equal to it up to interchange.

    Parameters:
        diagram : The diagram to draw nicer.
        score : The score to maximise, :func:`hand_score` by default.
        limit : The number of foliations to try, see
            :meth:`~discopy.monoidal.Diagram.foliations`.

    >>> from discopy.monoidal import Ty, Box
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> print(f @ g)
    f @ x >> x @ g
    >>> print(make_nicer(f @ g))
    f @ g
    """
    score = hand_score if score is None else score
    candidates = [diagram] + diagram.foliations(limit)
    return max(candidates, key=lambda d: score(d.to_drawing()))


def piecewise(points: Sequence[float], scales: Sequence[float]):
    """
    The increasing map that stretches the gap between two consecutive points
    by its scale, fixing the first knot, with slope one outside the points.

    >>> f = piecewise([0, 1, 3], [2, .5])
    >>> [f(x) for x in (-1, 0, .5, 1, 2, 3, 4)]
    [-1.0, 0.0, 1.0, 2.0, 2.5, 3.0, 4.0]
    """
    images = [float(points[0])] if points else []
    for (a, b), scale in zip(zip(points, points[1:]), scales):
        images.append(images[-1] + scale * (b - a))

    def image(x: float) -> float:
        i = max(bisect_right(points, x) - 1, 0)
        slope = scales[i] if points[i] <= x and i < len(scales) else 1
        return images[i] + slope * (x - points[i])
    return image


def stretch(
        drawing: Drawing, columns: Sequence[float], rows: Sequence[float],
        widths: Sequence[float], heights: Sequence[float]) -> Drawing:
    """
    The drawing with the gaps between its columns and between its rows
    stretched by a scale each, and the width and the height of each box by a
    scale of their own, never narrower than its ``min_width``, its ports
    moving with its top and bottom, through :meth:`Drawing.relayout`.

    Parameters:
        drawing : The drawing to stretch.
        columns : A scale for each gap between consecutive distinct
            abscissas of the nodes that are not box corners.
        rows : A scale for each gap between their consecutive ordinates.
        widths : A scale for the width of each box.
        heights : A scale for the height of each box.

    Raises:
        ValueError : When the result does not draw the same diagram, see
            :meth:`Drawing.is_feasible`.

    >>> from discopy.monoidal import Ty, Box
    >>> x = Ty('x')
    >>> drawing = (Box('f', x, x) @ Box('g', x, x)).foliation().to_drawing()
    >>> xs, ys, n = knots(drawing)[0], knots(drawing)[1], len(drawing.boxes)
    >>> wide = stretch(drawing, [2] * (len(xs) - 1), [1] * (len(ys) - 1),
    ...                [1.5] * n, [1] * n)
    >>> wide.width > drawing.width
    True
    """
    framed = drawing.relabel_nodes()
    framed.add_box_corners()
    xs, ys = knots(framed)
    if [len(columns), len(rows), len(widths), len(heights)] != [
            len(xs) - 1, len(ys) - 1, len(framed.boxes), len(framed.boxes)]:
        raise ValueError(messages.STRETCH_LENGTHS)
    across, up = piecewise(xs, columns), piecewise(ys, rows)
    positions = {node: Point(across(p.x), up(p.y))
                 for node, p in framed.positions.items()
                 if not node.kind.startswith("box-corner")}
    positions.update(box_ports(framed, across, up, heights))
    for j, (low, high) in enumerate(framed.rectangles()):
        centre = Point(across((low.x + high.x) / 2), up((low.y + high.y) / 2))
        positions.update(corners(j, centre, Point(
            max(widths[j] * (high.x - low.x), framed.boxes[j].min_width) / 2,
            heights[j] * (high.y - low.y) / 2)))
    return drawing.relayout([coordinate for node in framed.ordered_nodes()
                             for coordinate in positions[node]])


def box_ports(drawing: Drawing, across: Callable[[float], float],
              up: Callable[[float], float],
              heights: Sequence[float]) -> dict[Node, Point]:
    """
    Where the ports of each box go when :func:`stretch` maps abscissas by
    ``across``, the centre of the box by ``up`` and scales its height: at
    the same offset from the centre, times the scale, so that they stay on
    its top and bottom. The drawing is framed by
    :meth:`Drawing.add_box_corners`.

    >>> from discopy.monoidal import Ty, Box
    >>> drawing = Box('f', Ty('x'), Ty('y')).to_drawing()
    >>> drawing.add_box_corners()
    >>> for node, point in box_ports(
    ...         drawing, lambda x: x, lambda y: 2 * y, [2]).items():
    ...     print(node.kind, point)
    box_cod Point(x=0.5, y=0.5)
    box_dom Point(x=0.5, y=1.5)
    """
    result: dict[Node, Point] = {}
    for j, ((low, high), box) in enumerate(
            zip(drawing.rectangles(), drawing.boxes)):
        middle = (low.y + high.y) / 2
        for kind, ty in [("box_cod", box.cod), ("box_dom", box.dom)]:
            for i, x in enumerate(ty):
                point = drawing.positions[Node(kind, i=i, j=j, x=x)]
                result[Node(kind, i=i, j=j, x=x)] = Point(
                    across(point.x),
                    up(middle) + heights[j] * (point.y - middle))
    return result


def corners(j: int, centre: Point, half: Point) -> dict[Node, Point]:
    """
    The four corners of the box ``j`` around a centre, at a half width and
    a half height.

    >>> corners(0, Point(1, 1), Point(.5, .25))[Node("box-corner-11", j=0)]
    Point(x=1.5, y=1.25)
    """
    return {Node(f"box-corner-{a}{b}", j=j): Point(
        centre.x + (2 * a - 1) * half.x, centre.y + (2 * b - 1) * half.y)
        for a in range(2) for b in range(2)}


def shape(drawing: Drawing) -> list[int]:
    """
    How many scales :func:`stretch` takes for each of its arguments: the
    gaps between columns, the gaps between rows, the widths and the heights.

    >>> from discopy.monoidal import Ty, Box
    >>> shape(Box('f', Ty('x'), Ty('x')).to_drawing())
    [0, 4, 1, 1]
    """
    xs, ys = knots(drawing)
    return [max(len(xs) - 1, 0), max(len(ys) - 1, 0)]\
        + 2 * [len(drawing.boxes)]


def split(drawing: Drawing, scales: Sequence[float]) -> list[list[float]]:
    """
    A flat list of the scales of :func:`stretch` cut by its :func:`shape`.

    >>> from discopy.monoidal import Ty, Box
    >>> split(Box('f', Ty('x'), Ty('x')).to_drawing(), [1, 2, 3, 4, 5, 6])
    [[], [1, 2, 3, 4], [5], [6]]
    """
    ends = list(accumulate(shape(drawing)))
    return [list(scales[a:b]) for a, b in zip([0] + ends, ends)]


def knots(drawing: Drawing) -> tuple[list[float], list[float]]:
    """
    The distinct abscissas and ordinates of the nodes of a drawing that are
    not box corners, once its boxes are centred by
    :meth:`Drawing.add_box_corners`, in increasing order: the columns and
    the rows that :func:`stretch` scales the gaps between.

    >>> from discopy.monoidal import Ty, Box
    >>> knots(Box('f', Ty('x'), Ty('x')).to_drawing())
    ([0.5], [0, 0.25, 0.5, 0.75, 1])
    """
    framed = drawing.relabel_nodes()
    framed.add_box_corners()
    points = [p for node, p in framed.positions.items()
              if not node.kind.startswith("box-corner")]
    return sorted({p.x for p in points}), sorted({p.y for p in points})


def clearance(drawing: Drawing) -> float:
    """
    How much room a drawing leaves: the least gap between two boxes, along
    the axis that separates them, and between a box and the row of the
    inputs or the outputs of the drawing, infinite with no box.

    >>> from discopy.monoidal import Ty, Box
    >>> x = Ty('x')
    >>> clearance((Box('f', x, x) >> Box('g', x, x)).to_drawing())
    0.25
    """
    framed = drawing.relabel_nodes()
    framed.add_box_corners()
    boxes = framed.rectangles()
    top = max((p.y for node, p in framed.positions.items()
               if node.kind == "dom"), default=framed.height)
    bottom = min((p.y for node, p in framed.positions.items()
                  if node.kind == "cod"), default=0)
    gaps = [max(c.x - b.x, a.x - d.x, c.y - b.y, a.y - d.y)
            for i, (a, b) in enumerate(boxes) for c, d in boxes[i + 1:]]
    gaps += [gap for a, b in boxes for gap in (top - b.y, a.y - bottom)]
    return min(gaps, default=float("inf"))


def nicer_layout(
        drawing: Drawing, score: Callable[[Drawing], float] | None = None,
        factors: Sequence[float] = (.5, .8, 1.25, 2.), rounds: int = 2
        ) -> Drawing:
    """
    The geometry of a drawing that scores best, by coordinate search over
    the scales of :func:`stretch`: each scale in turn is multiplied by each
    factor, and the change is kept when its drawing is feasible and scores
    better and leaves as much room as the drawing, see :func:`clearance`.
    The drawing itself is returned when nothing scores better.

    Parameters:
        drawing : The drawing to lay out.
        score : The score to maximise, :func:`hand_score` by default.
        factors : What a scale is multiplied by at each step.
        rounds : How many times every scale is tried.

    >>> from discopy.monoidal import Ty, Box
    >>> x = Ty('x')
    >>> drawing = (Box('f', x, x) @ Box('g', x, x)).foliation().to_drawing()
    >>> nicer = nicer_layout(drawing)
    >>> assert hand_score(nicer) >= hand_score(drawing)
    """
    score = hand_score if score is None else score
    dimension = sum(shape(drawing))
    scales = [1.] * dimension
    best, value, room = drawing, score(drawing), clearance(drawing)

    def attempt(values):
        columns, rows, widths, heights = split(drawing, values)
        try:
            candidate = stretch(drawing, columns, rows, widths, heights)
        except ValueError:
            return None
        return candidate if clearance(candidate) >= room else None

    for _ in range(rounds):
        for i in range(dimension):
            for factor in factors:
                values = scales[:i] + [scales[i] * factor] + scales[i + 1:]
                candidate = attempt(values)
                candidate_value = None if candidate is None\
                    else score(candidate)
                if candidate_value is not None and candidate_value > value:
                    best, value, scales = candidate, candidate_value, values
    return best
