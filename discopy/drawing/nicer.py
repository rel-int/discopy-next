"""
How nice a drawing looks, as a score to maximise.

A layout of a diagram is a choice of foliation, see
:meth:`discopy.monoidal.Diagram.foliations`, and of its geometry. The
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

from math import hypot, log
from typing import TYPE_CHECKING, Callable

from discopy.drawing.backend import Backend
from discopy.drawing.drawing import Drawing, Point
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
    see :meth:`Backend.visible_edges`.

    >>> from discopy.monoidal import Ty, Box
    >>> for wire in wires(Box('f', Ty('x'), Ty('x')).to_drawing()):
    ...     print(*wire)
    Point(x=0.5, y=1) Point(x=0.5, y=0.75)
    Point(x=0.5, y=0.25) Point(x=0.5, y=0)
    """
    return [(drawing.positions[u], drawing.positions[v])
            for u, v in Backend.visible_edges(drawing)]


def rectangles(drawing: Drawing) -> list[tuple[float, float, float, float]]:
    """
    The rectangle ``(left, bottom, right, top)`` of each box, wide enough for
    its wires and the label it draws, as high as its ``height``.

    >>> from discopy.monoidal import Ty, Box
    >>> rectangles(Box('f', Ty('x'), Ty('x')).to_drawing())
    [(0.25, 0.25, 0.75, 0.75)]
    """
    result = []
    for j, box in enumerate(drawing.boxes):
        x, y = drawing.positions[next(
            node for node in drawing.box_nodes if node.j == j)]
        xs = [drawing.positions[node].x for node in drawing.nodes
              if node.kind in ("box_dom", "box_cod") and node.j == j] + [x]
        label = "" if getattr(box, "no_label", False)\
            else str(getattr(box, "drawing_name", box.name))
        half = max(
            (max(xs) - min(xs)) / 2 + .25,
            text_width(label) / 2, box.min_width / 2)
        high = max(getattr(box, "height", 1) / 2 - .25, .25)
        centre = (max(xs) + min(xs)) / 2
        result.append((centre - half, y - high, centre + half, y + high))
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
    positions = list(drawing.positions.values())
    mirrored = [Point(width - p.x, p.y) for p in positions]
    rows: dict[float, list[float]] = {}
    for p in positions:
        rows.setdefault(p.y, []).append(p.x)
    gaps = [b - a for xs in rows.values()
            for a, b in zip(sorted(xs), sorted(xs)[1:]) if b > a]
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
        "whitespace": mean([(gap - mean(gaps)) ** 2 for gap in gaps])
        / (mean(gaps) ** 2 or 1),
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
            :meth:`discopy.monoidal.Diagram.foliations`.

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
