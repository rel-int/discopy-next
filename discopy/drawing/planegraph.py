"""
The category of labeled progressive plane graphs.

This was first defined in :cite:t:`JoyalStreet88`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Point
    PlaneGraph
    Drawing

Axioms
------

* Associativity and unit

>>> from discopy.monoidal import Ty, Box
>>> from discopy.monoidal import Equation

>>> x, y, z, w = map(Ty, "xyzw")
>>> f = Box('f', x, y).to_drawing()
>>> g = Box('g', y, z).to_drawing()
>>> h = Box('h', z, w).to_drawing()

>>> assert (f >> g) >> h == f >> (g >> h)
>>> assert (f @ g) @ h == f @ (g @ h)

>>> assert f >> Drawing.id(f.cod) == f == Drawing.id(f.dom) >> f
>>> assert f @ Drawing.id() == f == Drawing.id() @ f

* Interchanger

>>> f0, f1 = (Box(f'f{i}', f'x{i}', f'y{i}').to_drawing() for i in (0, 1))
>>> g0, g1 = (Box(f'g{i}', f'y{i}', f'z{i}').to_drawing() for i in (0, 1))

>>> Equation(f0 @ f1 >> g0 @ g1, (f0 >> g0) @ (f1 >> g1)).draw(
...     doctest="docs/_static/drawing/interchanger-1.svg")

.. image:: /_static/drawing/interchanger-1.svg
    :align: center

>>> Equation(f @ g.dom >> f.cod @ g, f @ g, f.dom @ g >> f @ g.cod).draw(
...     doctest="docs/_static/drawing/interchanger-2.svg")

.. image:: /_static/drawing/interchanger-2.svg
    :align: center

Gallery
-------

The drawing parameters below are exercised by the documentation itself.

>>> from discopy import monoidal
>>> Ty, Box, Id = monoidal.Ty, monoidal.Box, monoidal.Diagram.id
>>> x = Ty('x')

Boxes and wires reserve enough horizontal space for their labels:

>>> (Box('f', x, x @ x)
...  >> Box('a_box_with_a_very_long_name', x @ x, x)
...  >> Box('g', x, x)).draw(
...      aspect='equal', doctest="docs/_static/drawing/long-box-name.svg")

.. image:: /_static/drawing/long-box-name.svg
    :align: center

>>> (Box('$\\\\Lambda$', x, x, min_width=3) @ Box('f', x, x)).draw(
...     aspect='equal', doctest="docs/_static/drawing/box-min-width.svg")

.. image:: /_static/drawing/box-min-width.svg
    :align: center

>>> long_type = Ty('a_long_type_name')
>>> long_type.inside[0].min_right_margin = 1.5
>>> Id(x @ long_type @ x).draw(aspect='equal',
...     doctest="docs/_static/drawing/wire-min-right-margin.svg")

.. image:: /_static/drawing/wire-min-right-margin.svg
    :align: center

>>> custom = Ty('custom_margin_wire')
>>> custom.inside[0].right_margin = 3
>>> Id(x @ custom @ x).draw(
...     aspect='equal', doctest="docs/_static/drawing/wire-custom-margin.svg")

.. image:: /_static/drawing/wire-custom-margin.svg
    :align: center

>>> Box('f', x, x @ Ty('a_long_output_type')).draw(
...     aspect='equal', doctest="docs/_static/drawing/wire-auto-margin.svg")

.. image:: /_static/drawing/wire-auto-margin.svg
    :align: center

>>> (Box('$\\\\int_a^b f(x)\\\\,dx = \\\\sqrt{2}$', x, x)
...  @ Box('f', x, x)).draw(
...      aspect='equal', doctest="docs/_static/drawing/long-latex-name.svg")

.. image:: /_static/drawing/long-latex-name.svg
    :align: center

Bubbles, grammatical diagrams and quantum circuits use the same backend:

>>> (x @ Box('s', Ty(), Ty())).bubble().draw(
...     wire_labels=False,
...     doctest="docs/_static/drawing/bubble-straight-wire.svg")

.. image:: /_static/drawing/bubble-straight-wire.svg
    :align: center

A bubble whose inside and outside have a different number of wires keeps its
boundary, see issue #520:

>>> Box('f', x, x ** 3).bubble(dom=x ** 3, cod=x).draw(
...     doctest="docs/_static/drawing/bubble-uneven-wires.svg")

.. image:: /_static/drawing/bubble-uneven-wires.svg
    :align: center

>>> from discopy.compact import (
...     Cap, Ty as RTy, Box as RBox, Id as RId)
>>> n, s = map(RTy, 'ns')
>>> who = (Cap(n.r, n) >> RId(n.r) @ RBox('copy', n, n @ n)
...        >> RId(n.r @ n) @ Cap(s, s.l) @ RId(n)
...        >> RId(n.r) @ RBox('update', n @ s, s) @ RId(s.l @ n))
>>> who.draw(aspect='equal', doctest="docs/_static/drawing/who-ansatz.svg")

.. image:: /_static/drawing/who-ansatz.svg
    :align: center

>>> from discopy.lambek.categorial import Eval, Ty as CTy, Word
>>> s, n = map(CTy, 'sn')
>>> sentence = (Word('Alice', n) @ Word('loves', (n >> s) << n)
...             @ Word('Bob', n) >> n @ Eval((n >> s) << n) >> Eval(n >> s))
>>> sentence.draw(
...     aspect='equal', doctest="docs/_static/drawing/categorial-grammar.svg")

.. image:: /_static/drawing/categorial-grammar.svg
    :align: center

>>> from discopy.quantum.zx import Z, X, Id as ZId, SWAP
>>> bialgebra = (Z(1, 2) @ Z(1, 2) >> ZId(1) @ SWAP @ ZId(1)
...              >> X(2, 1) @ X(2, 1))
>>> (bialgebra + bialgebra).draw(
...     aspect='equal', doctest="docs/_static/drawing/bialgebra.svg")

.. image:: /_static/drawing/bialgebra.svg
    :align: center

>>> from discopy.quantum import qubit, H, sqrt, Bra, Ket, CX
>>> bell = sqrt(2) >> Ket(0, 0) >> H @ qubit >> CX >> Bra(0) @ qubit
>>> bell.draw(aspect='equal', doctest="docs/_static/drawing/bell-state.svg")

.. image:: /_static/drawing/bell-state.svg
    :align: center

A controlled gate over distinct wires, e.g. a classically-controlled gate,
picks the x-coordinate of its control from the wire it sits on:

>>> bit, qubit = Ty("bit"), Ty("qubit")
>>> gate = Box("F", qubit, qubit)
>>> controlled = Box(
...     "CF", bit @ qubit, bit @ qubit,
...     draw_as_controlled=True, controlled=gate, distance=1)
>>> left_controlled = Box(
...     "FC", qubit @ bit, qubit @ bit,
...     draw_as_controlled=True, controlled=gate, distance=-1)
>>> (controlled @ left_controlled).draw(
...     doctest="docs/_static/drawing/controlled-classical.svg")

Coloured regions are also checked as part of the gallery:

>>> Colour, Wire = monoidal.Colour, monoidal.Wire
>>> ol, o1, o2, o_r, o3, o4 = map(
...     Colour, ("red", "orange", "gold", "green", "blue", "purple"))
>>> il, i1, i_r, i2 = map(
...     Colour, ("cyan", "magenta", "brown", "pink"))
>>> outer_dom = Ty(
...     Wire("d", ol, o1), Wire("c", o1, o2), Wire("c", o2, o_r))
>>> outer_cod = Ty(
...     Wire("b", ol, o3), Wire("a", o3, o4), Wire("a", o4, o_r))
>>> inner_dom = Ty(Wire("a", il, i1), Wire("b", i1, i_r))
>>> inner_cod = Ty(Wire("c", il, i2), Wire("d", i2, i_r))
>>> Box("f", inner_dom, inner_cod).bubble(
...     dom=outer_dom, cod=outer_cod, name="g").draw(wire_labels=False,
...         doctest="docs/_static/drawing/coloured-bubble.svg")

.. image:: /_static/drawing/coloured-bubble.svg
    :align: center

The regions of a crossing follow its strands, see
:meth:`~discopy.drawing.backend.Backend.region_separators`:

>>> from discopy import symmetric
>>> red, blue = Colour("red"), Colour("blue")
>>> symmetric.Permutation(symmetric.Ty(*(
...     symmetric.Wire(name, dom, cod) for name, dom, cod in [
...         ("x", red, blue), ("y", blue, red), ("z", red, blue)])),
...     [2, 1, 0]).draw(
...     doctest="docs/_static/drawing/coloured-permutation.svg")

.. image:: /_static/drawing/coloured-permutation.svg
    :align: center
"""


from __future__ import annotations

import itertools
from math import isfinite
from typing import NamedTuple, TYPE_CHECKING, Sequence
from dataclasses import dataclass

import networkx as nx

from discopy import messages
from discopy.drawing import backend, Node, Point
from discopy.config import BOX_DRAWING_ATTRIBUTES, TRANSPARENT
from discopy.abc import DaggerMonoidalCategory, TracedCategory
from discopy.python import finset
from discopy.utils import (
    assert_isinstance, assert_iscomposable, unbiased, factory, RichDisplay)

if TYPE_CHECKING:
    from discopy import monoidal, symmetric


def trailing_margin(ob) -> float:
    """ The extra width needed to the right of the last wire of a type. """
    return max(
        0, getattr(ob, "right_margin", 0) - 0.5,
        getattr(ob, "min_right_margin", 0))


def sign(number: float) -> int:
    """ The sign of a number, -1, 0 or 1. """
    return (number > 0) - (number < 0)


def cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    """ Whether the segments ``ab`` and ``cd`` properly intersect. """
    def side(p, q, r):
        return sign((q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x))
    return side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(
        c, d, b) < 0


def through(a: Point, b: Point, low: Point, high: Point) -> bool:
    """
    Whether the segment ``ab`` passes through the inside of the rectangle
    with corners ``low, high``, by Liang-Barsky clipping: touching its sides
    does not count.
    """
    start, end = 0., 1.
    for p, q in [(a.x - b.x, a.x - low.x), (b.x - a.x, high.x - a.x),
                 (a.y - b.y, a.y - low.y), (b.y - a.y, high.y - a.y)]:
        if p == 0 and q <= 0:
            return False
        if p < 0:
            start = max(start, q / p)
        elif p > 0:
            end = min(end, q / p)
    return start < end


def touch(a: Point, b: Point, c: Point, d: Point) -> bool:
    """ Whether the closed segments ``ab`` and ``cd`` share a point. """
    def side(p, q, r):
        return sign((q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x))

    def within(p, q, r):
        return min(p.x, q.x) <= r.x <= max(p.x, q.x)\
            and min(p.y, q.y) <= r.y <= max(p.y, q.y)
    sides = side(a, b, c), side(a, b, d), side(c, d, a), side(c, d, b)
    if sides[0] * sides[1] < 0 and sides[2] * sides[3] < 0:
        return True
    return any(s == 0 and within(p, q, r) for s, (p, q, r) in zip(sides, [
        (a, b, c), (a, b, d), (c, d, a), (c, d, b)]))


def curve(a: Point, b: Point, steps: int = 8) -> list[tuple[Point, Point]]:
    """
    The quadratic Bézier curve from ``a`` to ``b`` with control point
    ``(a.x, b.y)``, the way a backend draws a wire, as ``steps`` segments.

    >>> curve(Point(0, 0), Point(2, 1), steps=2)
    [(Point(x=0.0, y=0.0), Point(x=0.5, y=0.75)), \
(Point(x=0.5, y=0.75), Point(x=2.0, y=1.0))]
    """
    points = [Point(a.x + t * t * (b.x - a.x), a.y + (2 - t) * t * (b.y - a.y))
              for t in (i / steps for i in range(steps + 1))]
    return list(zip(points, points[1:]))


def meet(test, path: list, other: list) -> bool:
    """ Whether ``test`` holds of a segment of ``path`` and of ``other``. """
    return any(test(*p, *q) for p in path for q in other)


def overlap(a: Point, b: Point, c: Point, d: Point) -> bool:
    """ Whether the rectangles with corners ``a, b`` and ``c, d`` overlap. """
    return a.x < d.x and c.x < b.x and a.y < d.y and c.y < b.y


class PlaneGraph(NamedTuple):
    """ A plane graph is a graph with a mapping from nodes to points. """
    graph: nx.DiGraph
    positions: dict[Node, Point]

    def __repr__(self):
        """
        The graph as the adjacency of each of its nodes sorted by
        :attr:`~discopy.utils.Node.index`, rather than its address: the order
        the graph happens to store its nodes in is no part of the plane
        graph, and an adjacency also names a node with no edge, which an
        edge list would lose.

        Example
        -------
        >>> from discopy import monoidal
        >>> from discopy.monoidal import Ty, Box
        >>> x = Ty('x')
        >>> f = Box('f', x, x)
        >>> assert eval(repr(f.to_drawing())) == f.to_drawing()
        """
        def key(node):
            return node.index, repr(node)

        nodes = sorted(self.graph.nodes, key=key)
        adjacency = {
            node: sorted(self.graph.successors(node), key=key)
            for node in nodes}
        positions = dict(
            sorted(self.positions.items(), key=lambda item: key(item[0])))
        return f"PlaneGraph(graph=nx.DiGraph({adjacency}), "\
               f"positions={positions})"


@factory
@dataclass
class Drawing(
        TracedCategory["monoidal.Ty", "Drawing"],
        DaggerMonoidalCategory["monoidal.Ty", "Drawing"], RichDisplay):
    """
    A drawing is a plane graph with designated input and output types.

    Its dagger is the reflection along the horizontal axis, which is
    involutive, while the layout is compared on the nose: reversing a
    composite or a tensor can lay the same diagram out with other positions,
    see :meth:`dagger`.

    Parameters:
        inside (PlaneGraph) : The plane graph underlying the drawing.
        dom (monoidal.Ty) : The domain of the drawing, i.e. its input type.
        cod (monoidal.Ty) : The codomain of the drawing, i.e. its output type.
        boxes (tuple[monoidal.Box, ...]) : The boxes inside the drawing.
        width (float) : The width of the drawing.
        height (float) : The height of the drawing.
        _check (bool) : Whether to call :meth:`validate_attributes`.

    .. admonition:: Summary

        .. autosummary::

            validate_attributes
            draw
            from_box
            id
            then
            tensor
            dagger
            bubble
            frame
    """
    inside: PlaneGraph
    dom: "monoidal.Ty"
    cod: "monoidal.Ty"
    boxes: tuple["monoidal.Box", ...] = ()
    width: float = 0
    height: float = 0

    graph = property(lambda self: self.inside.graph)
    nodes = property(lambda self: self.graph.nodes)
    edges = property(lambda self: self.graph.edges)
    positions = property(lambda self: self.inside.positions)

    yanking = TracedCategory.yanking.inapplicable(
        messages.PLANAR_YANKING)
    dagger_contravariance = DaggerMonoidalCategory.dagger_contravariance\
        .failing(messages.DRAWING_DAGGER_LAYOUT)
    dagger_monoidality = DaggerMonoidalCategory.dagger_monoidality.failing(
        messages.DRAWING_DAGGER_LAYOUT)

    def to_tree(self) -> dict:
        """
        A drawing has no tree, see :func:`~discopy.utils.dumps`: serialise the
        diagram it is the drawing of instead.

        >>> from discopy.monoidal import Box
        >>> from discopy.utils import dumps
        >>> dumps(Box('f', 'x', 'y').to_drawing())
        Traceback (most recent call last):
        ...
        TypeError: A drawing has no tree: it is the layout of a diagram, \
which to_drawing recomputes.
        """
        raise TypeError(messages.NO_TREE_DRAWING)

    def nodes_of_kind(self, kind):
        """
        The nodes of a kind, sorted by their :attr:`Node.index`: the order the
        graph happens to store them in is no part of the drawing.

        >>> from discopy.monoidal import Box
        >>> f, g = Box('f', 'x', 'y'), Box('g', 'y', 'z')
        >>> drawing = (f >> g).to_drawing()
        >>> assert drawing.dagger().dagger() == drawing
        """
        return sorted((node for node in self.nodes if node.kind == kind),
                      key=lambda node: node.index)

    box_nodes = property(lambda self: self.nodes_of_kind("box"))
    dom_nodes = property(lambda self: self.nodes_of_kind("dom"))
    cod_nodes = property(lambda self: self.nodes_of_kind("cod"))
    box_dom_nodes = property(lambda self: self.nodes_of_kind("box_dom"))
    box_cod_nodes = property(lambda self: self.nodes_of_kind("box_cod"))

    def __init__(
            self, inside, dom, cod, boxes=(), width=0, height=0, _check=True):
        self.inside, self.dom, self.cod = inside, dom, cod
        self.boxes, self.width, self.height = boxes, width, height
        if _check:
            self.validate_attributes()

    def validate_attributes(self):
        """
        Check that the attributes of a drawing are consistent.

        >>> from discopy.monoidal import Ty, Id
        >>> x = Ty('x')
        >>> drawing = Id(x).to_drawing()
        >>> drawing.add_edges([(Node("cod", i=0, x=x), Node("dom", i=0, x=x))])
        >>> drawing.validate_attributes()
        Traceback (most recent call last):
        ...
        ValueError: Wrong edge Node('cod', i=0, x=x) -> Node('dom', i=0, x=x)
        """
        from discopy.monoidal import Ty, Box
        assert_isinstance(self.dom, Ty)
        assert_isinstance(self.cod, Ty)
        for box in self.boxes:
            assert_isinstance(box, Box)
        assert self.dom_nodes == [
            Node("dom", i=i, x=x) for i, x in enumerate(self.dom)]
        assert self.cod_nodes == [
            Node("cod", i=i, x=x) for i, x in enumerate(self.cod)]
        assert self.box_nodes == [
            Node("box", j=j, box=box) for j, box in enumerate(self.boxes)]
        for j, box in enumerate(self.boxes):
            box_node = self.box_nodes[j]
            box_dom_nodes, box_cod_nodes = ([
                Node(f"box_{kind}", i=i, j=j, x=x)
                for i, x in enumerate(xs)] for kind, xs in [
                    ("dom", box.dom), ("cod", box.cod)])
            assert sorted(self.graph.predecessors(box_node),
                          key=lambda node: node.index) == box_dom_nodes
            assert sorted(self.graph.successors(box_node),
                          key=lambda node: node.index) == box_cod_nodes
        for source, target in self.edges:
            if source.kind == "box":
                assert target.kind == "box_cod"
            elif source.kind == "box_dom":
                assert target.kind == "box"
            elif source.kind in ("dom", "box_cod"):
                assert target.kind in ("cod", "box_dom")
            else:
                raise ValueError(f"Wrong edge {source} -> {target}")

        assert self.height >= (1 if self.boxes else 0)
        assert self.width >= (1 if self.boxes else 0)
        assert self.width >= max(
            (x for (x, _) in self.positions.values()), default=0)
        assert self.height >= max(
            (y for (_, y) in self.positions.values()), default=0)

        assert set(self.positions.keys()) == set(self.nodes) == set(
            self.dom_nodes + self.cod_nodes) | set(
                self.box_dom_nodes + self.box_nodes + self.box_cod_nodes)
        assert all(isinstance(x, Point) for x in self.positions.values())

    def __eq__(self, other):
        if not isinstance(other, Drawing):
            return False
        return self.is_parallel(other) and self.positions == other.positions\
            and set(self.edges) == set(other.edges)

    @backend.matplotlib_context()
    def draw(self, **params):
        """
        Call :meth:`add_box_corners` and :meth:`frame_dual_rail` on a copy,
        then :func:`backend.draw`, so that drawing never changes the drawing
        and draws it the same every time.

        >>> from discopy import ribbon
        >>> x = ribbon.Ty('x')
        >>> drawing = ribbon.Braid(x, x).trace().to_ribbons().to_drawing()
        >>> svg = drawing.to_svg()
        >>> assert drawing.to_svg() == svg
        """
        params.setdefault("asymmetry", self.asymmetry())
        return backend.draw(self.framed(), **params)

    def framed(self) -> Drawing:
        """
        A copy with its box corners added and its dual rails framed: what a
        backend draws, so that drawing never changes the drawing.
        """
        result = self.relabel_nodes()
        result.add_box_corners()
        result.frame_dual_rail()
        return result

    def asymmetry(self) -> float:
        """
        The shift of one corner of each box drawn as a polygon when nothing
        else is asked for: :code:`0.125` when some box is a conjugate, a
        transpose or a dagger not drawn as a braid, :code:`0` otherwise,
        counting the boxes of the drawings nested in its boxes.
        """
        return 0.125 * any(
            box.is_conjugate or box.is_transpose or (
                box.is_dagger and not box.draw_as_braid) or (
                any(nested.asymmetry()
                    for nested in getattr(box, "nested", None) or ()))
            for box in self.boxes)

    def to_typst(self, **params):
        """
        The :class:`~discopy.drawing.typst_ast.Document` of the drawing,
        whose ``render`` is its Typst source.

        >>> from discopy.monoidal import Ty, Box
        >>> drawing = Box('f', Ty('x'), Ty('y')).to_drawing()
        >>> nodes = len(drawing.positions)
        >>> assert 'canvas' in drawing.to_typst().render()
        >>> assert len(drawing.positions) == nodes
        """
        params.setdefault("asymmetry", self.asymmetry())
        return backend.Typst(**params).draw(
            self.framed(), **params).to_document(**params)

    def frame_dual_rail(self, margin=0.5):
        """
        Reframe a dual rail drawing so its boundary box contains the cup and
        cap arcs, which fold past the layout, with a uniform ``margin`` on each
        side. The input and output (identity) wires still run all the way to
        the top and bottom borders, only the folds are held a ``margin`` away.
        Drawings without dual rail cups or caps (found by the
        ``draw_as_dual_rail_cup`` and ``draw_as_dual_rail_cap`` attributes)
        are left unchanged.
        """
        from discopy.config import RIBBON_FOLD_DEPTH
        folds = [node for node in self.box_nodes
                 if getattr(node.box, "draw_as_dual_rail_cup", False)
                 or getattr(node.box, "draw_as_dual_rail_cap", False)]
        if not folds:
            return self
        xs = [p.x for p in self.positions.values()]
        # The folds may bulge below the bottom (cups) or above the top (caps).
        left, right = min(xs), max(xs)
        bottom, top = 0.0, self.height
        for node in folds:
            box = node.box
            kind, wires = ("box_dom", box.dom) if box.dom\
                else ("box_cod", box.cod)
            ends = [self.positions[Node(kind, i=i, j=node.j, x=wires[i])]
                    for i in (0, 3)]
            radius, wire_y = abs(ends[1].x - ends[0].x) / 2, ends[0].y
            depth = min(radius, RIBBON_FOLD_DEPTH)  # The fold is capped.
            if box.dom:  # A cup folds downwards.
                bottom = min(bottom, wire_y - depth)
            else:  # A cap folds upwards.
                top = max(top, wire_y + depth)
        self.relabel_nodes(copy=False, positions={
            n: p.shift(x=margin - left, y=margin - bottom)
            for n, p in self.positions.items()})
        self.width = right - left + 2 * margin
        self.height = top - bottom + 2 * margin
        # The identity wires run to the borders rather than stopping a margin
        # short, so that the diagram still composes along its inputs/outputs.
        for node in self.dom_nodes:
            self.positions[node] = Point(self.positions[node].x, self.height)
        for node in self.cod_nodes:
            self.positions[node] = Point(self.positions[node].x, 0)
        return self

    def add_box_corners(self):
        """
        Recenter boxes w.r.t their wires then draw the corners: a box is its
        ``height`` minus the half of a wire high, at least half a wire.

        >>> from discopy.monoidal import Ty, Box
        >>> x = Ty('x')
        >>> f, g = Box('f', x, x, height=2), Box('g', x, x)
        >>> (f @ g >> g @ f).draw(
        ...     doctest='docs/_static/drawing/box-height.svg')

        .. image:: /_static/drawing/box-height.svg
            :align: center
        """
        for j, box in enumerate(self.boxes):
            box_node = Node("box", j=j, box=box)
            if all(Node(f"box-corner-{a}{b}", j=j) in self.positions
                   for a in range(2) for b in range(2)):
                continue
            box_x, box_y = self.positions[box_node]
            box_dom_nodes, box_cod_nodes = ([
                    Node(kind, i=i, j=j, x=x) for i, x in enumerate(xs)]
                for kind, xs in [("box_dom", box.dom), ("box_cod", box.cod)])
            xs = [self.positions[n].x for n in box_dom_nodes + box_cod_nodes]
            left, right = min(xs + [box_x]) - 0.25, max(xs + [box_x]) + 0.25
            half = max(box.height / 2 - 0.25, 0.25)
            if box.min_width and right - left < box.min_width:
                center = (left + right) / 2
                left = center - box.min_width / 2
                right = center + box.min_width / 2
            self.add_nodes({
                Node(f"box-corner-{a}{b}", j=j): Point(x, box_y + y)
                for a, x in enumerate([left, right])
                for b, y in enumerate([-half, half])})
            if box.draw_as_wires or box.draw_as_spider:
                if len(box.dom) == 1 or len(box.cod) == 1:
                    continue
            self.positions[box_node] = Point(
                (right + left) / 2, self.positions[box_node].y)

    def ordered_nodes(self) -> list[Node]:
        """
        The nodes of a drawing in a fixed order: by kind, then by box and wire.

        >>> from discopy.monoidal import Box
        >>> for node in Box('f', 'x', 'y').to_drawing().ordered_nodes():
        ...     print(node)
        Node('box', box=f, j=0)
        Node('box_cod', i=0, j=0, x=y)
        Node('box_dom', i=0, j=0, x=x)
        Node('cod', i=0, x=y)
        Node('dom', i=0, x=x)
        """
        return sorted(self.nodes, key=lambda node: (
            node.kind, node.index, repr(node)))

    def layout(self) -> tuple[float, ...]:
        """
        The geometry of a drawing as a vector: the coordinates of each of
        :meth:`ordered_nodes` once its box corners are added, so that it holds
        the width and the height of each box, the gaps between wires and the
        heights of layers. :meth:`relayout` takes it back.

        >>> from discopy.monoidal import Box
        >>> Box('f', 'x', 'y').to_drawing().layout()
        (0.5, 0.5, 0.25, 0.25, 0.25, 0.75, 0.75, 0.25, 0.75, 0.75, 0.5, 0.25, \
0.5, 0.75, 0.5, 0, 0.5, 1)
        """
        drawing = self.relabel_nodes()
        drawing.add_box_corners()
        return tuple(coordinate for node in drawing.ordered_nodes()
                     for coordinate in drawing.positions[node])

    def relayout(self, layout: Sequence[float]) -> Drawing:
        """
        The drawing with the geometry of a vector given by :meth:`layout`.

        Parameters:
            layout : Two coordinates for each of the :meth:`ordered_nodes` of
                the drawing with its box corners.

        Raises:
            ValueError : When the layout has the wrong length or when it is
                not feasible, see :meth:`is_feasible`.

        >>> from discopy.monoidal import Ty, Box
        >>> x = Ty('x')
        >>> drawing = (Box('f', x, x @ x) >> Box('g', x @ x, x)).to_drawing()
        >>> svg = drawing.to_svg()
        >>> assert drawing.relayout(drawing.layout()).to_svg() == svg
        >>> wider = [2 * c for c in drawing.layout()]
        >>> drawing.relayout(wider).draw(
        ...     doctest='docs/_static/drawing/relayout.svg')

        .. image:: /_static/drawing/relayout.svg
            :align: center
        """
        reference = self.relabel_nodes()
        reference.add_box_corners()
        nodes = reference.ordered_nodes()
        if len(layout) != 2 * len(nodes):
            raise ValueError(messages.LAYOUT_LENGTH.format(
                len(layout), 2 * len(nodes)))
        positions = {
            node: Point(float(layout[2 * k]), float(layout[2 * k + 1]))
            for k, node in enumerate(nodes)}
        result = reference.relabel_nodes(positions=positions)
        result.width = reference.width + max(
            (p.x for p in positions.values()), default=0) - max(
                (p.x for p in reference.positions.values()), default=0)
        result.height = reference.height + max(
            (p.y for p in positions.values()), default=0) - max(
                (p.y for p in reference.positions.values()), default=0)
        if not result.is_feasible(reference):
            raise ValueError(messages.LAYOUT_NOT_FEASIBLE)
        return result

    def wires(self) -> list[tuple[Node, Node]]:
        """
        The edges of a drawing that are wires, i.e. not ending at a box node.

        >>> from discopy.monoidal import Box
        >>> len(Box('f', 'x', 'y').to_drawing().wires())
        2
        """
        return sorted(((u, v) for u, v in self.edges
                       if "box" not in (u.kind, v.kind)), key=repr)

    def crossings(self) -> set[tuple[int, int]]:
        """
        The pairs of :meth:`wires`, by index, that cross away from their ends,
        each wire drawn as its :func:`curve`.

        >>> from discopy.monoidal import Ty
        >>> drawing = Drawing.id(Ty('x', 'y'))
        >>> assert not drawing.crossings()
        >>> x, y = drawing.cod_nodes
        >>> drawing.relabel_nodes(positions={
        ...     x: drawing.positions[y], y: drawing.positions[x]}).crossings()
        {(0, 1)}
        """
        ends = [(self.positions[u], self.positions[v])
                for u, v in self.wires()]
        paths = [curve(*end) for end in ends]
        return {(i, j) for i, j in itertools.combinations(range(len(ends)), 2)
                if not set(ends[i]) & set(ends[j])
                and meet(cross, paths[i], paths[j])}

    def rectangles(self) -> list[tuple[Point, Point]]:
        """
        The bottom left and top right corners of each box, once added by
        :meth:`add_box_corners`.
        """
        return [(self.positions[Node("box-corner-00", j=j)],
                 self.positions[Node("box-corner-11", j=j)])
                for j, _ in enumerate(self.boxes)
                if Node("box-corner-00", j=j) in self.positions]

    def has_rectangles(self) -> bool:
        """
        Whether the four corners of each box are those of a rectangle, with
        its sides parallel to the axes.
        """
        return all(
            self.positions[Node("box-corner-01", j=j)]
            == Point(left.x, right.y)
            and self.positions[Node("box-corner-10", j=j)]
            == Point(right.x, left.y)
            and left.x < right.x and left.y < right.y
            for j, (left, right) in enumerate(self.rectangles()))

    def shape(self) -> dict[str, object]:
        """
        What a layout keeps of a drawing, see :meth:`is_feasible`: the
        direction of each edge, the :meth:`crossings` of wires, which wires
        with no end in common touch, which wires pass through which box, each
        wire drawn as its :func:`curve`, which boxes overlap, where each port
        and each box node sits with respect to the sides of its box, and the
        order from left to right of the wires at each boundary and at each
        side of each box.

        >>> from discopy.monoidal import Box
        >>> drawing = Box('f', 'x', 'y').to_drawing()
        >>> drawing.add_box_corners()
        >>> drawing.shape()["ports"]
        ((0, 1, -1, 1, 1), (0, 0, -1, 1, 1), (0, 1, 0, 1, 1))
        """
        positions, rectangles, wires = (
            self.positions, self.rectangles(), self.wires())
        paths = [curve(positions[u], positions[v]) for u, v in wires]
        ports = sorted((node for node in self.nodes
                        if node.kind in ("box_dom", "box_cod", "box")),
                       key=repr)
        rows: dict[tuple, list] = {}
        for node in sorted(self.nodes, key=repr):
            if node.kind in ("dom", "cod", "box_dom", "box_cod"):
                rows.setdefault((node.kind, node.index[0]), []).append(node)
        return {
            "edges": tuple(sign(positions[u].y - positions[v].y)
                           for u, v in sorted(
                               ((u, v) for u, v in self.edges), key=repr)),
            "crossings": self.crossings(),
            "touchings": {
                (i, j) for i, j in itertools.combinations(range(len(wires)), 2)
                if not set(wires[i]) & set(wires[j])
                and meet(touch, paths[i], paths[j])},
            "through": {(i, j) for i, path in enumerate(paths)
                        for j, rectangle in enumerate(rectangles)
                        if meet(through, path, [rectangle])},
            "overlaps": {(i, j) for i, j in itertools.combinations(
                range(len(rectangles)), 2)
                if overlap(*rectangles[i], *rectangles[j])},
            "ports": tuple(
                (node.j, *(sign(a - b) for a, b in [
                    (positions[node].y, rectangles[node.j][0].y),
                    (positions[node].y, rectangles[node.j][1].y),
                    (positions[node].x, rectangles[node.j][0].x),
                    (rectangles[node.j][1].x, positions[node].x)]))
                for node in ports if node.j < len(rectangles)),
            "order": tuple(
                (key, tuple(node.index[1] for node in sorted(
                    row, key=lambda node: positions[node].x)))
                for key, row in sorted(rows.items()))}

    def is_feasible(self, reference: Drawing) -> bool:
        """
        Whether a drawing with the nodes of a reference draws the same
        diagram: every coordinate is finite and not negative, the corners of
        each box make a rectangle and its :meth:`shape` is the reference's,
        once both have their box corners.

        Parameters:
            reference : The drawing it was laid out from.

        >>> from discopy.monoidal import Ty, Box
        >>> x = Ty('x')
        >>> drawing = (Box('f', x, x) @ Box('g', x, x)).to_drawing()
        >>> layout = list(drawing.layout())
        >>> assert drawing.relabel_nodes().is_feasible(drawing)
        >>> layout[1] += 10
        >>> drawing.relayout(layout)
        Traceback (most recent call last):
        ...
        ValueError: This layout does not draw the same diagram.
        """
        drawing, reference = self.relabel_nodes(), reference.relabel_nodes()
        drawing.add_box_corners()
        reference.add_box_corners()
        return all(isfinite(c) and c >= 0
                   for p in drawing.positions.values() for c in p)\
            and drawing.has_rectangles()\
            and drawing.shape() == reference.shape()

    def union(self, other, dom, cod, width=None, height=None, _check=True):
        """ Take the union of two drawings, assuming nodes are distinct. """
        graph = nx.union(self.inside.graph, other.inside.graph)
        inside = PlaneGraph(graph, self.positions | other.positions)
        boxes = self.boxes + other.boxes
        width = width or max(self.width, other.width)
        height = height or max(self.height, other.height)
        return Drawing(inside, dom, cod, boxes, width, height, _check)

    def add_nodes(self, positions: dict[Node, Point]):
        """ Add nodes to the graph given their positions. """
        if not positions:
            return
        self.graph.add_nodes_from(positions)
        self.positions.update(positions)
        self.width = max(self.width, max(i for (i, _) in positions.values()))
        self.height = max(self.height, max(j for (_, j) in positions.values()))

    def add_edges(self, edges: list[tuple[Node, Node]]):
        """ Add edges from a list. """
        self.graph.add_edges_from(edges)

    def relabel_nodes(
            self, mapping=dict(), positions=dict(), copy=True, _check=False):
        """ Relabel nodes and/or their positions. """
        graph = nx.relabel_nodes(self.graph, mapping, copy)
        if not copy:
            self.positions.update(positions)
            return self
        positions = {mapping.get(node, node): positions.get(node, pos)
                     for node, pos in self.positions.items()}
        inside = PlaneGraph(graph, positions)
        dom, cod, boxes = self.dom, self.cod, self.boxes
        x, y = self.width, self.height
        return Drawing(inside, dom, cod, boxes, x, y, _check=_check)

    def make_space(self, space, x,
                   y_min=None, y_max=None, exclusive=False, copy=False):
        """
        Make some horizontal space after position x
        for all nodes between y_min and y_max (inclusive).

        Example
        -------
        >>> from discopy.monoidal import Ty, Box
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Drawing.from_box(Box('f', x @ y, z))
        >>> f.make_space(2, 0.5, 0.75, 1.0, copy=True).draw(
        ...     aspect='equal', doctest="docs/_static/drawing/make-space.svg")

        .. image:: /_static/drawing/make-space.svg
            :align: center
        """
        y_min = 0 if y_min is None else y_min
        y_max = self.height if y_max is None else y_max
        result = self.relabel_nodes(copy=copy, positions={
            n: p.shift(x=((p.x > x if exclusive else p.x >= x) * space))
            for n, p in self.positions.items() if y_min <= p.y <= y_max})
        result.width += space
        return result

    def reposition_box_dom(self, j=0):
        """ Recenter dom nodes, used when drawing frames. """
        box_dom = self.boxes[j].dom
        xs = [self.positions[n].x for n in self.nodes
              if n.kind == "box_cod" and n.j == j]
        box_x = self.positions[self.box_nodes[j]].x
        left, right = min(xs + [box_x]), max(xs + [box_x])
        offsets = box_dom.wire_offsets()
        for i, x in enumerate(box_dom):
            target = Node("box_dom", i=i, j=j, x=x)
            source, = self.graph.predecessors(target)
            for n in (source, target):
                x = (right + left - offsets[-1]) / 2 + offsets[i]
                self.positions[n] = Point(x, self.positions[n].y)

    def reposition_box_cod(self, j=-1):
        """ Recenter cod nodes to recover legacy behaviour for layers. """
        j = j if j >= 0 else len(self.boxes) + j
        box = self.boxes[j]
        if box.bubble_closing and len(box.dom[1:-1]) == len(box.cod):
            return  # Otherwise the wires would bend when coming out.
        xs = [self.positions[n].x for n in self.nodes
              if n.kind == "box_dom" and n.j == j]
        box_x = self.positions[self.box_nodes[j]].x
        left, right = min(xs + [box_x]), max(xs + [box_x])
        offsets = box.cod.wire_offsets()
        for i, x in enumerate(box.cod):
            source = Node("box_cod", i=i, j=j, x=x)
            target, = self.graph.successors(source)
            if target.kind != "cod":
                return  # Otherwise we would have to reposition everything.
            for n in (source, target):
                x = (right + left - offsets[-1]) / 2 + offsets[i]
                self.positions[n] = Point(x, self.positions[n].y)
            if box.draw_as_spider and len(box.cod) == 1:
                box_node = Node("box", box=box, j=j)
                self.positions[box_node] = Point(x, self.positions[box_node].y)

    def align_box_cod(self, j=-1):
        """ Align outputs with inputs when they have equal number of wires. """
        j = j if j >= 0 else len(self.boxes) + j
        box = self.boxes[j]
        for i, (x_dom, x_cod) in enumerate(zip(box.dom, box.cod)):
            dom_node = Node("box_dom", i=i, j=j, x=x_dom)
            cod_node = Node("box_cod", i=i, j=j, x=x_cod)
            target, = self.graph.successors(cod_node)
            if target.kind != "cod":
                return  # Otherwise we would have to reposition everything.
            x, _ = self.positions[dom_node]
            for n in (cod_node, target):
                self.positions[n] = Point(x, self.positions[n].y)

    @property
    def is_identity(self):
        """ A drawing with no boxes is the identity. """
        return not self.boxes

    @property
    def is_empty(self):
        """ A drawing with no boxes and no wires is empty. """
        return self.is_identity and not self.dom

    @property
    def is_box(self):
        """ Whether the drawing is just one box. """
        return len(self.boxes) == 1 and self.is_parallel(self.boxes[0])

    @property
    def is_layer(self):
        """ Whether the drawing is just one box with wires on both sides. """
        return len(self.boxes) == 1

    @property
    def box(self):
        """ Syntactic sugar for self.boxes[0] when self.is_box """
        if not self.is_layer:
            raise ValueError
        return self.boxes[0]

    @property
    def left_is_whiskered(self):
        """ Whether `self = x @ f` for some non-empty type `x`. """
        if len(self.dom) == 0:
            return False
        left_dom = Node("dom", i=0, x=self.dom[0])
        target, = self.graph.successors(left_dom)
        return target.kind == "cod" and target.i == 0

    @property
    def right_is_whiskered(self):
        """ Whether `self = f @ x` for some non-empty type `x`. """
        if len(self.dom) == 0:
            return False
        right_dom = Node("dom", i=len(self.dom) - 1, x=self.dom[-1])
        target, = self.graph.successors(right_dom)
        return target.kind == "cod" and target.i == len(self.cod) - 1

    @classmethod
    def permutation(cls, xs: Sequence[int], doms) -> Drawing:
        """
        Draw a permutation of blocks of wires, the ``i``-th block of its
        codomain being the ``xs[i]``-th of ``doms``, as in
        :meth:`~discopy.abc.SymmetricCategory.permutation`.

        >>> from discopy.monoidal import Ty
        >>> x, y, z = map(Ty, "xyz")
        >>> drawing = Drawing.permutation([1, 0], [x @ y, z])
        >>> assert drawing.cod == (z @ x @ y).to_drawing()
        """
        doms = list(doms)
        dom = cls.ob().tensor(*doms)
        wires = finset.Permutation(finset.Function.permutation(
            xs, [len(block) for block in doms]).inside)
        if wires.is_identity:
            return Drawing.id(dom)
        return Drawing.from_box(Drawing.permutation_box(dom, wires))

    @staticmethod
    def permutation_box(dom: "monoidal.Ty", perm) -> "symmetric.Permutation":
        """
        The :class:`Permutation <discopy.symmetric.Permutation>` box of a
        drawing, built on the symmetric wires its labels name, then given the
        wires of the drawing ``dom`` and their permutation as boundaries.
        """
        from discopy.symmetric import Permutation, Ty, Wire
        box = Permutation(
            Ty(*(Wire(x.name, x.dom, x.cod) for x in dom.inside)), perm)
        box.dom, box.cod = dom, dom[:0].tensor(*(dom[i] for i in box.perm))
        return box

    @staticmethod
    def from_box(box: "monoidal.Box") -> Drawing:
        """
        Draw a diagram with just one box.

        A :class:`Permutation <discopy.symmetric.Permutation>` reindexes its
        domain as its codomain, so its drawing gets a copy of the codomain
        rather than the domain itself: the label attributes of one side stay
        on that side, and a wire the permutation keeps in place is drawn with
        one label rather than two.

        >>> from discopy.monoidal import Ty, Box
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box('f', x, y @ z)
        >>> assert f.to_drawing() == Drawing.from_box(f)
        >>> for ps in f.to_drawing().positions.items(): print(*ps)
        Node('box', box=f, j=0) Point(x=1.0, y=0.5)
        Node('dom', i=0, x=x) Point(x=1.0, y=1)
        Node('box_dom', i=0, j=0, x=x) Point(x=1.0, y=0.75)
        Node('box_cod', i=0, j=0, x=y) Point(x=0.5, y=0.25)
        Node('box_cod', i=1, j=0, x=z) Point(x=1.5, y=0.25)
        Node('cod', i=0, x=y) Point(x=0.5, y=0)
        Node('cod', i=1, x=z) Point(x=1.5, y=0)
        >>> f.draw(doctest="docs/_static/drawing/box.svg")

        .. image:: /_static/drawing/box.svg
            :align: center
        """
        from discopy.monoidal import Box
        from discopy.symmetric import Permutation
        box_dom, box_cod = box.dom.to_drawing(), box.cod.to_drawing()
        old_box = box
        box = Drawing.permutation_box(box_dom, old_box.perm)\
            if isinstance(old_box, Permutation)\
            else Box(
                old_box.name, box_dom, box_cod,
                is_dagger=old_box.is_dagger)
        if isinstance(old_box, Permutation):
            box.cod = box_cod
            for i, obj in enumerate(box.cod.inside):
                if old_box.perm[i] == i:
                    obj.skip_label = True

        for attr, default in BOX_DRAWING_ATTRIBUTES.items():
            setattr(box, attr, getattr(old_box, attr, default(box)))

        if box.draw_as_wires and not box.frame_boundary:
            for i, obj in enumerate(box.cod.inside):
                obj.reposition_label = 0.5 if (
                    box.bubble_closing or box.bubble_opening and i) else 0.25

        is_bubble = box.bubble_opening or box.bubble_closing
        offsets_dom = box.dom.wire_offsets()
        offsets_cod = box.cod.wire_offsets()
        span_dom = offsets_dom[-1] if offsets_dom else -1
        span_cod = offsets_cod[-1] if offsets_cod else -1

        if box.bubble_opening:
            content = max(1, len(box.dom), len(box.cod) - 2) + 0.5
        elif box.bubble_closing:
            content = max(1, len(box.dom) - 2, len(box.cod)) + 0.5
        else:
            content = max(1, span_dom + 1, span_cod + 1)

        content = max(content, box.min_width + 0.5)\
            if box.min_width else content

        trailing = 0 if is_bubble else max(
            (trailing_margin(row.inside[-1])
             for row in (box.dom, box.cod) if row.inside), default=0)
        width, height = content + trailing, box.height

        left, right = 0.25, content - 0.25

        inside = PlaneGraph(nx.DiGraph(), dict())
        result = Drawing(
            inside, box.dom, box.cod, (box, ), width, height, _check=False)

        box_node = Node("box", box=box, j=0)
        result.add_nodes({box_node: Point(content / 2, height / 2)})

        dom = [Node("dom", i=i, x=x) for i, x in enumerate(box.dom)]
        cod = [Node("cod", i=i, x=x) for i, x in enumerate(box.cod)]
        box_dom = [Node("box_dom", i=i, j=0, x=x.x) for i, x in enumerate(dom)]
        box_cod = [Node("box_cod", i=i, j=0, x=x.x) for i, x in enumerate(cod)]

        result.add_edges(list(zip(dom, box_dom)))
        result.add_edges([(x, box_node) for x in box_dom])
        result.add_edges([(box_node, x) for x in box_cod])
        result.add_edges(list(zip(box_cod, cod)))

        if box.bubble_opening:
            result.add_nodes({
                cod[0]: Point(left, 0),
                box_cod[0]: Point(left, 0),
                box_cod[-1]: Point(right, 0),
                cod[-1]: Point(right, 0)})
            cod, box_cod = cod[1:-1], box_cod[1:-1]
        elif box.bubble_closing:
            result.add_nodes({
                dom[0]: Point(left, height),
                box_dom[0]: Point(left, height),
                box_dom[-1]: Point(right, height),
                dom[-1]: Point(right, height)})
            dom, box_dom = dom[1:-1], box_dom[1:-1]

        if is_bubble:
            offsets_dom = list(range(len(dom)))
            offsets_cod = list(range(len(cod)))
            span_dom = offsets_dom[-1] if offsets_dom else -1
            span_cod = offsets_cod[-1] if offsets_cod else -1

        result.add_nodes({
            x: Point((content - span - 1) / 2 + 0.5 + offsets[i], y)
            for xs, y, offsets, span in [
                (dom, height, offsets_dom, span_dom),
                (box_dom, height if box.draw_as_wires else height - 0.25,
                 offsets_dom, span_dom),
                (box_cod, 0 if box.draw_as_wires else 0.25,
                 offsets_cod, span_cod),
                (cod, 0, offsets_cod, span_cod)]
            for i, x in enumerate(xs)})
        return result

    @staticmethod
    def id(dom: "monoidal.Ty" | None = None) -> Drawing:
        """
        Draw the identity diagram.

        >>> from discopy.monoidal import Ty
        >>> Drawing.id(Ty()).draw(doctest="docs/_static/drawing/empty.svg")

        .. image:: /_static/drawing/empty.svg
            :align: center

        >>> Drawing.id(Ty('x')).draw(doctest="docs/_static/drawing/idx.svg")

        .. image:: /_static/drawing/idx.svg
            :align: center

        >>> Drawing.id(Ty('x', 'y')).draw(
        ...     doctest="docs/_static/drawing/idxy.svg")

        .. image:: /_static/drawing/idxy.svg
            :align: center
        """
        from discopy.monoidal import Ty
        dom = (Ty() if dom is None else dom).to_drawing()
        inside = PlaneGraph(nx.DiGraph(), dict())
        offsets = dom.wire_offsets()
        height = 0.5
        width = 0.5 + offsets[-1] + trailing_margin(
            dom.inside[-1]) if dom else 0.5
        result = Drawing(inside, dom, dom, (), width, height, _check=False)
        dom_nodes = [Node("dom", i=i, x=x) for i, x in enumerate(dom)]
        cod_nodes = [Node("cod", i=i, x=x) for i, x in enumerate(dom)]
        result.add_nodes({
            x: Point(0.25 + offsets[i], 1) for i, x in enumerate(dom_nodes)})
        result.add_nodes({
            x: Point(0.25 + offsets[i], 0) for i, x in enumerate(cod_nodes)})
        result.add_edges(list(zip(dom_nodes, cod_nodes)))
        return result

    @unbiased
    def then(self, other: Drawing,
             draw_step_by_step=False) -> Drawing | list[Drawing]:
        """
        Draw one diagram composed with another.

        This is done by calling :meth:`make_space` to align the output wires of
        `self` and the input wires of `other` while keeping them straight.

        Parameters:
            other : The drawing to compose with.
            draw_step_by_step : Whether to return the list of drawings that
                connect one wire at a time, e.g. for
                :func:`~discopy.drawing.to_gif`, rather than the composite,
                for one ``other`` only.

        Example
        -------
        >>> from discopy.monoidal import Ty, Box, Diagram
        >>> x = Ty('x')
        >>> f = Drawing.from_box(Box('f', x, x @ x @ x))
        >>> g = Drawing.from_box(Box('g', x @ x, x))
        >>> u = Drawing.from_box(Box('u', Ty(), x ** 3))
        >>> v = Drawing.from_box(Box('v', x ** 7, Ty()))

        >>> top, bottom = u >> g @ f, g @ f @ f >> v
        >>> Diagram.to_gif(  # doctest: +EXTRA
        ...     *top.then(bottom, draw_step_by_step=True), loop=True,
        ...     wire_labels=False, draw_box_labels=False,
        ...     doctest="docs/_static/drawing/composition.gif")
        <IPython.core.display.HTML object>

        .. image:: /_static/drawing/composition.gif
            :align: center
        """
        assert_iscomposable(self, other)
        for side, rest in [(self, other), (other, self)]:
            if side.is_identity and side.height <= 1:
                result = rest.relabel_nodes()
                return [result] if draw_step_by_step else result
        dom, cod = self.dom, other.cod

        tmp_cod = [Node("tmp_cod", i=i) for i, n in enumerate(self.cod_nodes)]
        mapping = dict(zip(self.cod_nodes, tmp_cod))
        positions = {
            n: p.shift(y=other.height + 1) for n, p in self.positions.items()}

        tmp_dom = [Node("tmp_dom", i=i) for i, n in enumerate(other.dom_nodes)]
        other_mapping = dict(zip(other.dom_nodes, tmp_dom))
        other_mapping.update({
            n: n.shift_j(len(self.boxes))
            for n in other.nodes if "box" in n.kind})

        result = self.relabel_nodes(mapping, positions, _check=False).union(
                other.relabel_nodes(other_mapping), dom, cod, _check=False)
        result.height = self.height + other.height + 1

        cut, top_width, bot_width = other.height + 0.5, self.width, other.width
        if draw_step_by_step:
            steps = [result.relabel_nodes(copy=True)]
        for i, (u, v) in enumerate(zip(self.cod_nodes, other.dom_nodes)):
            top = result.positions[tmp_cod[i]].x
            bot = result.positions[tmp_dom[i]].x
            if top > bot:
                bot_width += top - bot
                result.make_space(top - bot, (i > 0) * bot, 0, cut)
            elif top < bot:
                top_width += bot - top
                result.make_space(bot - top, (i > 0) * top, cut, result.height)
            source, = self.graph.predecessors(u)
            target, = other.graph.successors(v)
            result.add_edges([(source, other_mapping.get(target, target))])
            if draw_step_by_step:
                steps.append(result.relabel_nodes(copy=True))

        result.graph.remove_nodes_from(tmp_dom + tmp_cod)
        [result.positions.pop(n) for n in tmp_dom + tmp_cod]
        result.relabel_nodes(copy=False, positions={
            n: p.shift(y=-1)
            for n, p in result.positions.items() if p.y > other.height})
        result.height = self.height + other.height
        result.width = max(top_width, bot_width)
        for j, box in enumerate(other.boxes):  # Recover legacy behaviour.
            if len(box.dom) == len(box.cod) and not box.frame_boundary:
                result.align_box_cod(len(self.boxes) + j)
            else:
                result.reposition_box_cod(len(self.boxes) + j)
        if draw_step_by_step:
            for step in steps:
                step.width = result.width
        return steps if draw_step_by_step else result

    def stretch(self, y, copy=True):
        """
        Stretch input and output wires to increase the height of a diagram
        by a given length.
        .
        Example
        -------
        >>> from discopy.monoidal import Box
        >>> f = Drawing.from_box(Box('f', 'x', 'x'))
        >>> f.stretch(2).draw(doctest="docs/_static/drawing/stretch.svg")

        .. image:: /_static/drawing/stretch.svg
            :align: center
        """
        if not y:
            # Nothing to stretch: avoid shifting middle nodes by y / 2 == 0.0,
            # which would turn integer coordinates into floats.
            return self.relabel_nodes(copy=copy)
        result = self.relabel_nodes(copy=copy, positions={n: p.shift(y=(
                y if n.kind == "dom" else 0 if n.kind == "cod" else y / 2))
            for n, p in self.positions.items()})
        result.height += y
        return result

    @unbiased
    def tensor(self, other: Drawing) -> Drawing:
        """
        Draw two diagrams side by side.

        Example
        -------
        >>> from discopy.monoidal import Box
        >>> f = Drawing.from_box(Box('f', 'x', 'x'))
        >>> d = (f >> f >> f) @ (f >> f)
        >>> d.draw(doctest="docs/_static/drawing/tensor.svg")

        .. image:: /_static/drawing/tensor.svg
            :align: center
        """
        if self.is_empty:
            return other.relabel_nodes()
        if other.is_empty:
            return self.relabel_nodes()
        mapping = {
            n: n.shift_j(len(self.boxes))
            for n in other.nodes if "box" in n.kind}
        mapping.update({n: n.shift_i(len(self.dom)) for n in other.dom_nodes})
        mapping.update({n: n.shift_i(len(self.cod)) for n in other.cod_nodes})
        if self.height < other.height:
            self = self.stretch(other.height - self.height)
        elif self.height > other.height:
            other = other.stretch(self.height - other.height)
        x_shift = self.width + (
            0.25 if self.right_is_whiskered or other.left_is_whiskered else 0)
        result = self.union(other.relabel_nodes(mapping, positions={
            n: p.shift(x=x_shift) for n, p in other.positions.items()}),
            dom=self.dom @ other.dom, cod=self.cod @ other.cod, _check=False)
        result.width = x_shift + other.width
        return result

    def trace(self, n=1, left=False) -> Drawing:
        from discopy.monoidal import Box, Ty
        if n == 0:
            return self
        if n > 1:
            return self.trace(left=left).trace(n=n - 1, left=left)
        dom = self.dom[1:] if left else self.dom[:-1]
        cod = self.cod[1:] if left else self.cod[:-1]
        traced_dom = self.dom[:1] if left else self.dom[-1:]
        traced_cod = self.cod[:1] if left else self.cod[-1:]
        cap_cod, cup_dom = traced_dom ** 2, (
            traced_dom @ traced_cod if left else traced_cod @ traced_dom)
        cup = Box('cup', cup_dom, Ty(), draw_as_wires=True).to_drawing()
        cap = Box('cap', Ty(), cap_cod, draw_as_wires=True).to_drawing()
        return (
            cap @ dom >> traced_dom @ self >> cup @ cod if left
            else dom @ cap >> self @ traced_dom >> cod @ cup)

    def dagger(self) -> Drawing:
        """ The reflection of a drawing along the the horizontal axis. """
        from discopy.symmetric import Permutation

        def box_dagger(box):
            result = Drawing.permutation_box(box.cod, box.perm.dagger())\
                if isinstance(box, Permutation) else box.dagger()
            for attr in BOX_DRAWING_ATTRIBUTES:
                if not hasattr(result, attr):
                    setattr(result, attr, getattr(box, attr))
            if getattr(box, "nested", None):
                # The reflection of a transpose is a conjugate, and back.
                vars(result).update(
                    mirror_nested=not box.mirror_nested,
                    is_transpose=box.is_conjugate,
                    is_conjugate=box.is_transpose)
            return result

        if self.is_box:
            return Drawing.from_box(box_dagger(self.box))

        mapping = {
            n: Node("box", box=box_dagger(n.box), j=len(self.boxes) - n.j - 1)
            for n in self.nodes if n.kind == "box"}
        mapping.update({
            n: Node(kd, i=n.i, j=len(self.boxes) - n.j - 1, x=n.x)
            for (k, kd) in [("box_dom", "box_cod"), ("box_cod", "box_dom")]
            for n in self.nodes if n.kind == k})
        mapping.update({
            n: Node(f"box-corner-{n.kind[-2]}{1 - int(n.kind[-1])}",
                    j=len(self.boxes) - n.j - 1)
            for n in self.nodes if n.kind.startswith("box-corner-")})
        mapping.update({
            n: Node("cod", i=i, x=n.x) for i, n in enumerate(self.dom_nodes)})
        mapping.update({
            n: Node("dom", i=i, x=n.x) for i, n in enumerate(self.cod_nodes)})
        graph = nx.relabel_nodes(self.graph, mapping).reverse()
        inside = PlaneGraph(graph, positions={
            mapping[n]: Point(x, self.height - y)
            for n, (x, y) in self.positions.items()})
        dom, cod = self.cod, self.dom
        boxes = tuple(map(box_dagger, self.boxes[::-1]))
        return Drawing(
            inside, dom, cod, boxes, self.width, self.height, _check=False)

    @staticmethod
    def bubble_opening(dom, arg_dom, left, right, frame_boundary=False):
        """
        Construct the opening of a bubble, i.e. a box drawn as wires.

        >>> from discopy.monoidal import Ty
        >>> x, y, z = map(Ty, "xyz")
        >>> Drawing.bubble_opening(x, y, z, Ty("")).draw(
        ...     doctest="docs/_static/drawing/bubble-opening.svg")

        .. image:: /_static/drawing/bubble-opening.svg
            :align: center
        """
        from discopy.monoidal import Box
        return Box(
            "top", dom, left @ arg_dom @ right,
            bubble_opening=True, frame_boundary=frame_boundary,
            height=(0.5 if frame_boundary else 1)).to_drawing()

    @staticmethod
    def bubble_closing(arg_cod, cod, left, right, frame_boundary=False):
        """
        Construct the closing of a bubble, i.e. a box drawn as wires.

        >>> from discopy.monoidal import Ty
        >>> x, y, z = map(Ty, "xyz")
        >>> Drawing.bubble_closing(x, y, z, Ty("")).draw(
        ...     doctest="docs/_static/drawing/bubble-closing.svg")

        .. image:: /_static/drawing/bubble-closing.svg
            :align: center
        """
        from discopy.monoidal import Box
        return Box(
            "bot", left @ arg_cod @ right, cod,
            bubble_closing=True, frame_boundary=frame_boundary,
            height=(0.5 if frame_boundary else 1)).to_drawing()

    @staticmethod
    def frame_opening(dom, arg_dom, left, right):
        """
        Construct the opening of a frame as the opening of a bubble squashed to
        zero height so that it looks like the upper half of a rectangle.

        >>> from discopy.monoidal import Ty
        >>> x, y, z = map(Ty, "xyz")
        >>> Drawing.frame_opening(x, y, z, Ty("")).draw(
        ...     doctest="docs/_static/drawing/frame-opening.svg")

        .. image:: /_static/drawing/frame-opening.svg
            :align: center
        """
        result = Drawing.bubble_opening(
            dom, arg_dom, left, right, frame_boundary=True)
        box_dom_nodes = result.box_dom_nodes
        box_cod_nodes = result.box_cod_nodes
        result.relabel_nodes(copy=False, positions={
            n: result.positions[n].shift(y=-0.25) for n in box_dom_nodes})
        result.relabel_nodes(copy=False, positions={
            n: result.positions[n].shift(y=0.25) for n in box_cod_nodes})
        result.graph.remove_edges_from([
            (u, v) for u in box_dom_nodes for v in result.box_nodes] + [
            (u, v) for u in result.box_nodes for v in box_cod_nodes[1:-1]])
        return result

    @staticmethod
    def frame_closing(arg_cod, cod, left, right):
        """
        Construct the closing of a frame as the closing of a bubble squashed to
        zero height so that it looks like the lower half of a rectangle.

        >>> from discopy.monoidal import Ty
        >>> x, y, z = map(Ty, "xyz")
        >>> Drawing.frame_closing(x, y, z, Ty("")).draw(
        ...     doctest="docs/_static/drawing/frame-closing.svg")

        .. image:: /_static/drawing/frame-closing.svg
            :align: center
        """
        result = Drawing.bubble_closing(
            arg_cod, cod, left, right, frame_boundary=True)
        box_dom_nodes = result.box_dom_nodes
        box_cod_nodes = result.box_cod_nodes
        result.relabel_nodes(copy=False, positions={
            n: result.positions[n].shift(y=-0.25) for n in box_dom_nodes})
        result.relabel_nodes(copy=False, positions={
            n: result.positions[n].shift(y=0.25) for n in box_cod_nodes})
        result.graph.remove_edges_from([
            (u, v) for u in box_dom_nodes[1:-1] for v in result.box_nodes] + [
            (u, v) for u in result.box_nodes for v in box_cod_nodes])
        return result

    def bubble(self, dom=None, cod=None, name=None, width=None, height=None,
               draw_as_square=False, frame_sides=False) -> Drawing:
        """
        Draw a closed line around a drawing, with some wires coming in and out.

        Parameters:
            dom (monoidal.Ty) : The wires coming into the bubble.
            cod (monoidal.Ty) : The wires coming out of the bubble.
            name (str) : The label of the bubble, drawn on the top left.
            width : The width of the bubble, its natural width by default.
            height : The height of the bubble, its natural height by default.
            draw_as_square : Whether to draw the closed line as a square
                rather than a bubble.
            frame_sides : Whether to draw the left and right sides with zero
                width, as :meth:`slot` and :meth:`frame` do: the colours of
                the regions they separate show the edge in their place. A
                bubble drawn as a square has no such colours, so it keeps its
                sides.

        >>> from discopy.symmetric import *
        >>> a, b, c, d = map(Ty, "abcd")
        >>> f = Box('f', a @ b, c @ d).to_drawing()
        >>> f.bubble(d @ c @ c, b @ a @ a, name="g").draw(
        ...     doctest="docs/_static/drawing/bubble-drawing.svg")

        .. image:: /_static/drawing/bubble-drawing.svg
            :align: center
        """
        dom = self.dom if dom is None else dom.to_drawing()
        cod = self.cod if cod is None else cod.to_drawing()
        arg_dom, arg_cod = self.dom, self.cod
        from discopy.monoidal import Wire, Ty
        left = Ty(Wire(name or "", dom.dom, arg_dom.dom))
        right = Ty(Wire("", arg_dom.cod, dom.cod))
        left.inside[0].always_draw_label = True
        wires_can_go_straight = (
            len(dom), len(cod)) == (len(arg_dom), len(arg_cod))
        if frame_sides:
            left.inside[0].frame_boundary = right.inside[0].frame_boundary \
                = True
        if draw_as_square or not wires_can_go_straight:
            top = Drawing.frame_opening(dom, arg_dom, left, right)
            bot = Drawing.frame_closing(arg_cod, cod, left, right)
        else:
            top = Drawing.bubble_opening(dom, arg_dom, left, right)
            bot = Drawing.bubble_closing(arg_cod, cod, left, right)
        middle = self if height is None else self.stretch(height - self.height)
        result = top >> left @ middle @ right >> bot
        result.make_space(-0.25, 0.25, exclusive=True)
        result.make_space(-0.25, result.width - 0.25)
        if width is not None and result.width != width:
            if result.width > width:
                raise ValueError
            space = (width - result.width) / 2
            result.make_space(space, 0.25, exclusive=True)
            result.make_space(space, result.width - 0.25)
        if len(dom) == len(arg_dom):
            dom_nodes, arg_dom_nodes = ([
                    Node(kind, x=x, i=i + off, j=0)
                    for i, x in enumerate(xs)]
                for (kind, xs, off) in [
                    ("box_dom", dom, 0), ("box_cod", arg_dom, 1)])
            node = Node("box", j=0, box=top.box)
            result.graph.add_edges_from(zip(dom_nodes, arg_dom_nodes))
            result.graph.remove_edges_from([(x, node) for x in dom_nodes])
            result.graph.remove_edges_from([(node, x) for x in arg_dom_nodes])
        if len(cod) == len(arg_cod):
            arg_cod_nodes, cod_nodes = ([
                    Node(kind, x=x, i=i + off, j=len(result.boxes) - 1)
                    for i, x in enumerate(xs)]
                for (kind, xs, off) in [
                    ("box_dom", arg_cod, 1), ("box_cod", cod, 0)])
            node = Node("box", j=len(result.boxes) - 1, box=bot.box)
            result.graph.add_edges_from(zip(arg_cod_nodes, cod_nodes))
            result.graph.remove_edges_from([(x, node) for x in arg_cod_nodes])
            result.graph.remove_edges_from([(node, x) for x in cod_nodes])
        return result

    def slot(self, colour, **params):
        """
        Wrap the drawing in a square slot filled with a background ``colour``,
        so that several terms can be tiled on the same background.

        This is shared by :meth:`frame`, for its arguments, and by
        :meth:`add`, for the terms of an :class:`Equation` between
        differently-coloured diagrams.

        Parameters:
            colour : The :class:`monoidal.Colour` of the slot's background,
                or ``None`` for the page.
            params : Passed to :meth:`bubble`, e.g. ``height`` or ``width``.
        """
        from discopy.monoidal import Ty
        frame_type = Ty.id(colour)
        return self.bubble(
            frame_type, frame_type,
            draw_as_square=True, frame_sides=True, **params)

    def frame(self, *others: Drawing,
              dom=None, cod=None, name=None, draw_vertically=False,
              frame_colour=None) -> Drawing:
        """
        >>> from discopy.monoidal import *
        >>> x, y = Ty('x'), Ty('y')
        >>> f, g, h = Box('f', x, y ** 3), Box('g', y, y @ y), Box('h', x, y)
        >>> f.bubble(dom=x @ x, cod=y @ y, name="b", draw_as_frame=True
        ...     ).draw(doctest="docs/_static/drawing/single-frame.svg")

        .. image:: /_static/drawing/single-frame.svg
            :align: center

        >>> Bubble(f, g, h >> h[::-1], dom=x, cod=y @ y
        ...     ).draw(doctest="docs/_static/drawing/horizontal-frame.svg")

        .. image:: /_static/drawing/horizontal-frame.svg
            :align: center

        >>> Bubble(f, g, h, dom=x, cod=y @ y, draw_vertically=True
        ...     ).draw(doctest="docs/_static/drawing/vertical-frame.svg")

        .. image:: /_static/drawing/vertical-frame.svg
            :align: center
        """
        from discopy.monoidal import Colour
        frame_colour = frame_colour or \
            BOX_DRAWING_ATTRIBUTES['frame_colour'](self)
        colour = Colour(frame_colour)
        args = (self, ) + others
        method = "then" if draw_vertically else "tensor"
        params = dict(
                width=max([arg.width for arg in args] + [0]) + 1
            ) if draw_vertically else dict(
                height=max([arg.height for arg in args] + [0]))
        slots = tuple(arg.slot(colour, **params) for arg in args)
        result = getattr(slots[0], method)(*slots[1:]).bubble(
            dom, cod, name, draw_as_square=True, frame_sides=True)
        result.reposition_box_dom()
        result.reposition_box_cod()
        return result

    @staticmethod
    def zero(dom, cod):
        from discopy.monoidal import Box
        result = Box("zero", dom, cod).to_drawing()
        result.zero_drawing = True
        return result

    def add(self, other: Drawing, symbol="+", space=1):
        """ Concatenate two drawings with a symbol in between. """
        from discopy.monoidal import Box, Ty
        if getattr(self, "zero_drawing", False):
            return other
        if getattr(other, "zero_drawing", False):
            return self
        height = max(self.height, other.height)
        self = self.stretch(height - self.height)
        other = other.stretch(height - other.height)
        scalar = Box(
            symbol, Ty(), Ty(), draw_as_spider=True, color=TRANSPARENT
        ).to_drawing()
        if self.cod.cod is not None or other.dom.dom is not None:
            # The boundary colours are painted, e.g. an Equation between
            # terms of different colours: give each term its own uncoloured
            # slot, as for Drawing.frame.
            self, other = (
                term.slot(None, height=height)
                for term in (self, other))
        result = self @ scalar @ other
        result.make_space(space - 1, self.width + 1)  # Right of the scalar.
        result.make_space(space - 1, self.width)  # Left of the scalar.
        return result

    __add__ = add

    def to_drawing(self):
        return self
