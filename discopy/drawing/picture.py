"""
Pictures of diagrams: sample a diagram, draw it as an image, label each of
its pixels, and decode the labels back to the diagram.

Drawing is a functor from diagrams to images, reading is a section of it:
:func:`decode` sends the labels of the picture of a diagram back to that
diagram, up to the axioms of its module, see :func:`equal`. Training data
for reading is free: :func:`sample` draws a diagram of a module, its
picture and the labels of its pixels, its mask.

A pixel is the background, a wire with a type, or a box with a name or a
structural generator, i.e. a swap, a cup, a cap or a spider, see
:data:`LABELS`. Where a wire meets a swap, a cup, a cap or a spider, its
port, is a disc of the wire's type, so that two generators that meet are
two components. A diagram is drawn one generator at a time, see
:func:`elementary`, so that its picture is planar
but for its swaps.

Summary
-------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    load
    kind
    substitute
    elementary
    dress
    diagrams
    render
    port_table
    to_dot
    layout
    graphs
    augment
    sample
    samples
    decode
    equal
    coarse
    coarsen
    regions
    attach
    refine
    scan
    read
    decode_least
    crop
"""

from __future__ import annotations

import importlib
from collections import defaultdict
from collections.abc import Callable, Iterator
import html
import json
import shutil
import subprocess
from contextlib import contextmanager, nullcontext
from io import BytesIO
from math import lcm

import networkx as nx
import numpy as np

from discopy.drawing.backend import Backend, Matplotlib
from discopy.testable import GENERATORS
from discopy.utils import AxiomError, Node

#: The modules a picture is read in, from the fewest generators to the most.
MODULES = ("monoidal", "symmetric", "compact", "frobenius")

#: The names of boxes, a closed vocabulary.
NAMES = tuple("fghkmnpq")

#: The types of wires: the generators of a free category and their adjoints.
TYPES = GENERATORS + tuple(f"{x}.r" for x in GENERATORS)

#: The structural generators, by module.
STRUCTURAL = {
    "monoidal": (),
    "symmetric": ("swap", ),
    "compact": ("swap", "cup", "cap"),
    "frobenius": ("swap", "cup", "cap", "spider")}

#: The label of each pixel, by its index in a mask: the text of a wire
#: label by the type it names, and ``"wire"`` a wire whose type is unknown.
LABELS = ("background", ) + TYPES + NAMES + STRUCTURAL["frobenius"]\
    + tuple(f"label {x}" for x in TYPES) + ("wire", )

#: The classes of the pixel net, coarser than :data:`LABELS`: no name, no
#: type, only what a box, a wire and a label are and which primitive.
COARSE = ("background", "wire", "box", "label") + STRUCTURAL["frobenius"]

#: The characters a box name or a wire type is written in.
CHARS = "".join(sorted(set("".join(NAMES + TYPES))))

#: The height and width of the crop a name or a type is read on, in pixels.
CROP = (24, 48)

#: The flags of :func:`augment`, applied in this order.
AUGMENTATIONS = (
    "warp", "rotate", "perspective", "chalk", "whiteboard", "blur", "jpeg")

#: The families of graphs :func:`graphs` draws: the graph of each name
#: with about ``n`` vertices, drawn by ``rng`` when it is random.
FAMILIES = {
    "complete": lambda n, rng: nx.complete_graph(n),
    "star": lambda n, rng: nx.star_graph(n - 1),
    "cycle": lambda n, rng: nx.cycle_graph(n),
    "cube": lambda n, rng: nx.hypercube_graph(3),
    "grid": lambda n, rng: nx.grid_2d_graph(2, max(n // 2, 2)),
    "tree": lambda n, rng: nx.random_labeled_tree(
        n, seed=int(rng.integers(1 << 16)))}

DPI, SPIDER, WIRE, TEXT = 72, .15, LABELS.index("wire"), 7

#: The units of a picture in an inch of a graphviz layout.
INCH = 3


def load(module: str):
    """ The module of a name in :data:`MODULES`, e.g. ``discopy.compact``. """
    if module not in MODULES:
        raise ValueError(f"Expected one of {MODULES}, got {module!r}.")
    return importlib.import_module(f"discopy.{module}")


def kind(box) -> str | None:
    """
    The structural generator a box is, if any, else ``None``.

    >>> from discopy import compact
    >>> x = compact.Ty('x')
    >>> kind(compact.Cup(x, x.r)), kind(compact.Box('f', x, x))
    ('cup', None)
    """
    from discopy import compact, frobenius, symmetric
    for name, cls in [("swap", symmetric.Swap), ("cup", compact.Cup),
                      ("cap", compact.Cap), ("spider", frobenius.Spider)]:
        if isinstance(box, cls):
            return name
    return None


def substitute(diagram, func: Callable):
    """
    The diagram with each box replaced by a diagram with the same boundary,
    ``func(box)``, one layer for each of its layers, whiskered by the rest
    of the layer of the box, so that a module that fuses what it composes,
    e.g. ``frobenius`` fusing whiskered swaps into a permutation, keeps the
    layers as they are.

    >>> from discopy.frobenius import Ty, Box, Swap
    >>> x = Ty('x')
    >>> f, g = Box('f', x @ x, x @ x), Box('g', x @ x, x @ x)
    >>> print(substitute(f @ f, lambda _: g))
    g @ x @ x >> x @ x @ g
    >>> print(substitute(x @ f, lambda _: Swap(x, x) >> g))
    x @ Swap(x, x) >> x @ g
    """
    typ, layers = type(diagram.dom), []
    for layer in diagram.inside:
        items = list(layer)
        for k, x in enumerate(items):
            if isinstance(x, typ):
                continue
            left = typ().tensor(*(
                y if isinstance(y, typ) else y.cod for y in items[:k]))
            right = typ().tensor(*(
                y if isinstance(y, typ) else y.dom for y in items[k + 1:]))
            layers += [whisker(left, inner, right)
                       for inner in func(x).inside]
    cls = importlib.import_module(type(diagram).__module__).Diagram
    return cls(tuple(layers), diagram.dom, diagram.cod)


def whisker(left, layer, right):
    """ A layer with types on its left and right, as it stands. """
    items, typ = [type(left)()], type(left)
    for x in layer:
        if isinstance(x, typ):
            items[-1] = items[-1] @ x
        else:
            items += [x, typ()]
    items[0], items[-1] = left @ items[0], items[-1] @ right
    return type(layer)(*(x for x in items if not (
        isinstance(x, typ) and not x)), normalise=False)


def elementary(diagram):
    """
    The diagram drawn one generator at a time: each permutation as swaps,
    each cospan as spiders, cups and caps, so that every crossing is a swap
    of two wires and every spider is one dot.

    >>> from discopy.frobenius import Ty, Cospan
    >>> x, y = Ty('x'), Ty('y')
    >>> print(elementary(Cospan(x @ y, [0, 1, 0], [1, 1])))
    x @ Swap(y, x) >> Cup(x, x) @ y >> Spider(1, 2, y)
    """
    from discopy import frobenius, symmetric

    def decompose(box):
        if isinstance(box, symmetric.Permutation):
            return box if kind(box) else substitute(
                box.to_swaps(), transposition)
        if isinstance(box, frobenius.Cospan):
            return elementary(box.to_spiders())
        return box

    def transposition(box):
        i = next(i for i, j in enumerate(box.perm) if i != j)
        mod = importlib.import_module(type(box).__module__)
        swap = mod.Swap(box.dom[i:i + 1], box.dom[i + 1:i + 2])
        layer = whisker(box.dom[:i], swap.inside[0], box.dom[i + 2:])
        return mod.Diagram((layer, ), box.dom, box.cod)
    return substitute(diagram, decompose)


def dress(diagram, rng: np.random.Generator,
          heights=True, widths=True, triangles=True):
    """
    The diagram with each box renamed from :data:`NAMES` and drawn at
    random: a height, an extra width, a triangle, each when its flag is set.
    A name is undaggered, so that the picture tells it.
    """
    def rename(box):
        if kind(box) is not None:
            return box
        params = {"height": rng.choice([1, 1.5, 2])} if heights else {}
        if widths:
            params["min_width"] = rng.uniform(0, 2)
        if triangles and rng.random() < .3:
            params["shape"] = "triangle"
        return type(box)(
            str(rng.choice(NAMES)), box.dom, box.cod, **params)
    return substitute(diagram, rename)


def diagrams(module: str, seed=0, max_boxes=6, batch=100) -> Iterator:
    """
    Yield the elementary diagrams of a module with at most ``max_boxes``
    boxes and no trace, from the ``strategy`` of its diagrams, in batches
    of ``batch`` examples, each batch shuffled by ``seed``.

    >>> from discopy import compact
    >>> first = next(diagrams("compact", seed=1))
    >>> assert isinstance(first, compact.Diagram)
    >>> assert all(kind(box) or type(box) is compact.Box
    ...            for box in first.boxes)
    """
    from hypothesis import HealthCheck, Phase, given, settings
    from hypothesis import seed as hypothesis_seed
    cls, rng = load(module).Diagram, np.random.default_rng(seed)
    allowed = (load(module).Box, ) + tuple(
        type(box) for box in structural_boxes(module))
    for round_ in range(1 << 30):
        found = []

        @hypothesis_seed(seed * (1 << 20) + round_)
        @settings(max_examples=batch, phases=[Phase.generate],
                  database=None, deadline=None,
                  suppress_health_check=list(HealthCheck))
        @given(cls.strategy())
        def collect(diagram):
            found.append(diagram)
        collect()
        for i in rng.permutation(len(found)):
            try:
                diagram = elementary(found[i])
            except NotImplementedError:
                continue
            if len(diagram.boxes) <= max_boxes and all(
                    type(box) in allowed for box in diagram.boxes)\
                    and not any(kind(box) == "spider" and not (
                        box.dom or box.cod) for box in diagram.boxes)\
                    and all(str(x[i:i + 1]) in TYPES for x in [
                        diagram.dom, diagram.cod] + [
                        y for box in diagram.boxes for y in (box.dom, box.cod)]
                        for i in range(len(x))):
                yield diagram


def structural_boxes(module: str) -> list:
    """ One structural generator of each kind in a module. """
    mod = load(module)
    x = mod.Ty('a')
    return [{"swap": lambda: mod.Swap(x, x),
             "cup": lambda: mod.Cup(x, x.r),
             "cap": lambda: mod.Cap(x, x.r),
             "spider": lambda: mod.Spider(1, 1, x)}[name]()
            for name in STRUCTURAL[module]]


class Mask(Matplotlib):
    """
    A backend that draws the label of each pixel as its red channel, the
    index of :data:`LABELS`, without antialiasing. In ``mode="shapes"`` it
    draws no text: a wire by its type, a box by its name, a structural
    generator by its kind, and a disc of radius ``port`` pixels of its type
    at each port of a generator drawn as wires, so that two of them that
    meet are apart. In ``mode="labels"`` it draws the labels of the wires
    alone, each in green with its index times ``TEXT`` in red, see
    :func:`render`.
    """
    def __init__(self, axis=None, linewidth=1, port=3, scale=1,
                 mode="shapes"):
        super().__init__(axis=axis, linewidth=linewidth)
        self.port, self.scale, self.mode = port, scale, mode

    @staticmethod
    def colour(label: str) -> tuple[float, float, float]:
        """ The colour of a label, its index as red. """
        assert label in LABELS, label
        return (LABELS.index(label) / 255, 0, 0)

    @staticmethod
    def label(node: Node) -> str:
        """ The label of a box node, or the type of a wire node. """
        if node.kind != "box":
            return str(node.x)
        box = node.box
        return "swap" if box.is_crossing else "cup" if box.draw_as_cup\
            else "cap" if box.draw_as_cap else "spider"\
            if box.draw_as_spider else box.name

    def stroke(self, label):
        """ The context in which strokes take the colour of a label. """
        import matplotlib as mpl
        return mpl.rc_context({
            "patch.edgecolor": self.colour(label),
            "lines.color": self.colour(label)})

    def draw_text(self, text, i, j, **params):
        if self.mode == "labels" and text in TYPES:
            params |= {"color": (TEXT * LABELS.index(f"label {text}") / 255,
                                 1, 0), "adaptive": False}
            return super().draw_text(text, i, j, **params)
        return Backend.draw_text(self, text, i, j, **params)

    def draw_boundary(self, graph, boundary_color="none", **params):
        """ The boundary is the background. """

    def draw_polygon(self, *points, facecolor=None, edgecolor=None):
        super().draw_polygon(
            *points, facecolor=self.current, edgecolor=self.current)

    def draw_box(self, positions, node, **params):
        self.current = self.colour(self.label(node))
        super().draw_box(positions, node, **params)

    def draw_wires(self, graph, **params):
        if self.mode == "labels":
            for source, _ in self.visible_edges(graph):
                if source.kind in ["dom", "box_cod"]:
                    self.draw_wire_label(
                        source.x, *graph.positions[source], **params)
            return
        for node in graph.nodes:
            if node.kind == "box" and node.box.is_crossing:
                with self.stroke(self.label(node)):
                    self.draw_braid(graph.positions, node)
        for source, target in self.visible_edges(graph):
            if any(n.kind == "box" and n.box.is_crossing
                   for n in (source, target)):
                continue
            box = source if source.kind == "box" else target
            label = self.label(box) if box.kind == "box"\
                else str(getattr(source, 'x', None) or target.x)
            with self.stroke(label):
                self.draw_wire(
                    graph.positions[source], graph.positions[target],
                    source.kind == "box", target.kind == "box")

    def draw_bezier(self, points, adaptive=True):
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path
        import matplotlib as mpl
        self.axis.add_patch(PathPatch(
            Path(list(points), [Path.MOVETO] + 3 * [Path.CURVE4]),
            facecolor='none', linewidth=self.linewidth,
            edgecolor=mpl.rcParams["patch.edgecolor"]))

    def draw_boxes(self, graph, **params):
        if self.mode == "labels":
            return
        from matplotlib.patches import Circle
        for node in graph.nodes:
            box = graph.boxes[node.j] if node.kind in (
                "box_dom", "box_cod") else None
            if box is not None and (box.draw_as_wires or box.draw_as_spider):
                self.axis.add_patch(Circle(
                    graph.positions[node], self.port / self.scale,
                    color=self.colour(str(node.x))))
        super().draw_boxes(graph, **params)

    def draw_spiders(self, graph, draw_box_labels=True, **params):
        if self.mode == "labels":
            return
        from matplotlib.patches import Circle
        for node in graph.nodes:
            if node.kind == "box" and node.box.draw_as_spider:
                self.axis.add_patch(Circle(
                    graph.positions[node], SPIDER,
                    color=self.colour("spider")))


def render(diagram, size=160, mask=False, **params) -> np.ndarray:
    """
    The picture of a diagram as an array of shape ``(size, size, 3)``, or
    with ``mask=True`` the label of each pixel, see :class:`Mask`, as an
    array of shape ``(size, size)``.

    Text cannot be drawn without antialiasing, so the mask draws the labels
    of the wires apart, on top of the rest, each in green with its index
    times ``TEXT`` in red, read back as their ratio where the green is at
    least half: on black, the ratio does not depend on the antialiasing. A
    label covers the background and boxes, never a wire or a primitive, so
    one drawn across a box can cut it in two, which :func:`decode` joins.

    The diagram is centred at ``scale`` pixels per unit, ``24`` by default,
    its text ``fontsize`` pixels high, its strokes ``linewidth`` wide.
    ``xkcd=True`` draws it by hand, see :func:`matplotlib.pyplot.xkcd`.
    ``graphviz=True`` draws it as laid out by :func:`layout` instead, with
    no wire labels, an inch of the layout ``INCH`` units wide, and
    ``graphviz=layout(diagram)`` draws a layout already computed.

    >>> from discopy.monoidal import Ty, Box
    >>> x = Ty('a')
    >>> f = Box('f', x, x)
    >>> image, mask = render(f, 64), render(f, 64, mask=True)
    >>> image.shape, mask.shape
    ((64, 64, 3), (64, 64))
    >>> sorted(LABELS[i] for i in set(mask.flat))
    ['a', 'background', 'f', 'label a']
    """
    scale = params.pop("scale", 24)
    graph = params.pop("graphviz", False)
    if graph:
        params.pop("wire_labels", None)
        graph = graph if isinstance(graph, dict) else layout(diagram)
        return paint_layout(graph, size, scale, mask, **params)
    xkcd, drawing = params.pop("xkcd", False), diagram.to_drawing()
    params = {"fontsize": .4 * scale, "wire_labels": True,
              "nodesize": np.pi * (SPIDER * scale) ** 2 / 300 * np.sqrt(
                  max(drawing.height, drawing.width, .01))} | params
    if not mask:
        return paint(drawing, size, scale, xkcd, Matplotlib, mask, **params)
    shapes, text = (paint(
        drawing, size, scale, xkcd, Mask, mask, mode=mode, **params)
        for mode in ("shapes", "labels"))
    labels = shapes[..., 0].astype(np.int64)
    labels[shapes[..., 1] > 0] = 0
    red, green = (text[..., i].astype(np.int64) for i in (0, 1))
    solid = (green >= 128) & ((labels == 0) | np.isin(
        labels, [LABELS.index(name) for name in NAMES]))
    labels[solid] = np.rint(red[solid] * 255 / (TEXT * green[solid]))
    labels[labels >= len(LABELS)] = 0
    return labels


@contextmanager
def canvas(size, xkcd, mask):
    """ A square figure of ``size`` pixels and its axis, black for a mask
    and drawn without antialiasing, by hand when ``xkcd`` is set. """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    with plt.xkcd(randomness=1, scale=.5, length=50) if xkcd\
            else nullcontext(), mpl.rc_context({
                "lines.antialiased": not mask, "patch.antialiased": not mask,
                "text.antialiased": not mask, "path.snap": False}):
        figure, axis = plt.subplots(
            figsize=(size / DPI, size / DPI), dpi=DPI,
            facecolor="black" if mask else "white")
        try:
            yield figure, axis
        finally:
            plt.close(figure)


def pixels(figure, axis, centre, size, scale) -> np.ndarray:
    """ The pixels of a figure, its axis centred on a point in units. """
    half = size / scale / 2
    axis.set_xlim(centre[0] - half, centre[0] + half)
    axis.set_ylim(centre[1] - half, centre[1] + half)
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()


def paint(drawing, size, scale, xkcd, backend, mask, mode=None, **params):
    """ The pixels of a drawing on a backend, see :func:`render`. """
    with canvas(size, xkcd, mask) as (figure, axis):
        painter = backend(axis, params.pop("linewidth", 1.5), scale=scale,
                          mode=mode) if mask else backend(
            axis, linewidth=params.pop("linewidth", 1.5))
        drawing.draw(backend=painter, show=False, aspect="equal", **params)
        return pixels(figure, axis, (drawing.width / 2, drawing.height / 2),
                      size, scale)


def fits(diagram, size=160, scale=24, margin=1, graphviz=False) -> bool:
    """ Whether the picture of a diagram fits in the image, with a margin
    in units around it. """
    if graphviz:
        graph = graphviz if isinstance(graphviz, dict) else layout(diagram)
        *_, width, height = map(float, graph["bb"].split(","))
        extent = INCH * max(width, height) / 72
    else:
        drawing = diagram.to_drawing()
        extent = max(drawing.width, drawing.height)
    return extent + 2 * margin <= size / scale


def port_table(name: str, dom: int, cod: int) -> str:
    """
    The HTML table of a box for :func:`to_dot`: a bordered cell of its
    name, with a row of ``dom`` port columns above and ``cod`` below, each
    port as wide as the others on its side.

    >>> print(port_table('f', 0, 1))  # doctest: +NORMALIZE_WHITESPACE
    <TABLE BORDER="0" CELLSPACING="0"><TR><TD BORDER="1" CELLPADDING="6"
    COLSPAN="1">f</TD></TR><TR><TD PORT="c0" BORDER="0" CELLPADDING="0"
    COLSPAN="1" WIDTH="18" HEIGHT="0"></TD></TR></TABLE>
    """
    grid, width, rows = lcm(dom or 1, cod or 1), 18 * max(dom, cod, 1), []
    for side, arity in (("d", dom), ("c", cod)):
        if arity:
            rows.append("<TR>" + "".join(
                f'<TD PORT="{side}{j}" BORDER="0" CELLPADDING="0" '
                f'COLSPAN="{grid // arity}" WIDTH="{width // arity}" '
                f'HEIGHT="0"></TD>' for j in range(arity)) + "</TR>")
    rows.insert(bool(dom), f'<TR><TD BORDER="1" CELLPADDING="6" '
                           f'COLSPAN="{grid}">{name}</TD></TR>')
    return '<TABLE BORDER="0" CELLSPACING="0">' + "".join(rows) + "</TABLE>"


def to_dot(diagram) -> str:
    """
    The DOT source of the hypergraph of a diagram, for :func:`layout`.

    A box is an HTML table with one column per port, its domain above and
    its codomain below, each from left to right, so that each wire leaves
    the box where its port is. A wire with two ends is an edge from its
    source, an input or a codomain port, to its target, and a spider with
    any other number of ends is a dot with an edge to each. The inputs
    are ranked first and the outputs last, each from left to right.

    >>> from discopy.frobenius import Ty, Box
    >>> x = Ty('a')
    >>> print(to_dot(Box('f', x, x @ x)))  # doctest: +ELLIPSIS
    graph diagram {
    ...
      b0 [tooltip="f", label=<...>];
    ...
      b0:c0:s -- out0 [label="a"];
      b0:c1:s -- out1 [label="a"];
    }
    """
    graph = diagram.to_hypergraph()
    lines = ["graph diagram {",
             "  graph [rankdir=TB, nodesep=.3, ranksep=.35];",
             '  node [shape=plain, fontname="DejaVu Sans", fontsize=12];']
    ends = defaultdict(list)

    for side, wires in (("in", graph.dom_wires), ("out", graph.cod_wires)):
        for i, spider in enumerate(wires):
            lines.append(f'  {side}{i} [shape=point, width=0, label=""];')
            ends[spider].append((side == "out", f"{side}{i}"))
        if len(wires) > 1:
            lines.append("  { rank=" + ("min" if side == "in" else "max")
                         + "; " + " -- ".join(
                             f"{side}{i}" for i in range(len(wires)))
                         + " [style=invis]; }")
    for i, (box, (dom, cod)) in enumerate(
            zip(graph.boxes, graph.box_wires)):
        label = html.escape(str(box.name))
        lines.append(f'  b{i} [tooltip="{label}", '
                     f'label=<{port_table(label, len(dom), len(cod))}>];')
        for j, spider in enumerate(dom):
            ends[spider].append((True, f"b{i}:d{j}:n"))
        for j, spider in enumerate(cod):
            ends[spider].append((False, f"b{i}:c{j}:s"))
    for spider, obj in enumerate(graph.spider_types):
        ends[spider].sort(key=lambda end: end[0])
        pairs = [[end for _, end in ends[spider]]]\
            if len(ends[spider]) == 2 else [
                (f"s{spider}", end) if is_target else (end, f"s{spider}")
                for is_target, end in ends[spider]]
        if len(ends[spider]) != 2:
            lines.append(
                f'  s{spider} [shape=point, width={2 * SPIDER / INCH}, '
                'label=""];')
        label = html.escape(str(obj))
        lines += [f'  {" -- ".join(pair)} [label="{label}"];'
                  for pair in pairs]
    return "\n".join(lines + ["}"])


def layout(diagram) -> dict:
    """
    The layout of a diagram by graphviz's ``dot``, as the JSON it writes:
    the bounding box ``bb``, each node of :func:`to_dot` with its ``pos``,
    ``width`` and ``height``, and each edge with the points of its spline.
    Raises :class:`RuntimeError` when ``dot`` is not installed.
    """
    if shutil.which("dot") is None:
        raise RuntimeError("Graphviz executable 'dot' was not found.")
    return json.loads(subprocess.run(
        ["dot", "-Tjson"], input=to_dot(diagram).encode(),
        capture_output=True, check=True).stdout)


def paint_layout(  # pylint: disable=too-many-locals  # one figure, as paint
        graph: dict, size, scale, mask, **params) -> np.ndarray:
    """
    The pixels of a :func:`layout`, see :func:`render`: a box is a
    rectangle of its name with the name inside, a spider a dot and a wire
    the spline of its edge, each in the colour of its label for a mask,
    by hand when ``xkcd`` is set, its strokes ``linewidth`` wide.
    """
    xkcd, linewidth = params.get("xkcd", False), params.get("linewidth", 1.5)
    from matplotlib.patches import Circle, PathPatch, Rectangle
    from matplotlib.path import Path

    def point(pos):
        return tuple(INCH * float(x) / 72 for x in pos.split(","))

    def colour(label):
        return (LABELS.index(label) / 255, 0, 0) if mask else "black"
    with canvas(size, xkcd, mask) as (figure, axis):
        axis.axis("off")
        for edge in graph.get("edges", []):
            if edge.get("style") == "invis":
                continue
            points = [point(p) for p in edge["pos"].split()
                      if not p.startswith(("s,", "e,"))]
            axis.add_patch(PathPatch(Path(points, [Path.MOVETO] + [
                Path.CURVE4] * (len(points) - 1)), facecolor="none",
                linewidth=linewidth,
                edgecolor=colour(html.unescape(edge["label"]))))
        for node in filter(lambda node: "pos" in node, graph["objects"]):
            (x, y), name = point(node["pos"]), node["name"]
            width, height = (INCH * float(node[k]) for k in (
                "width", "height"))
            if name[0] == "s":
                axis.add_patch(Circle((x, y), SPIDER, color=colour("spider")))
            elif name[0] == "b":
                label = html.unescape(node["tooltip"])
                axis.add_patch(Rectangle(
                    (x - width / 2, y - height / 2), width, height,
                    linewidth=linewidth, edgecolor=colour(label),
                    facecolor=colour(label) if mask else "white"))
                if not mask:
                    axis.text(x, y, label, ha="center", va="center",
                              fontsize=params.get("fontsize", .4 * scale))
        *_, right, top = map(float, graph["bb"].split(","))
        result = pixels(figure, axis, point(f"{right / 2},{top / 2}"),
                        size, scale)
    if not mask:
        return result
    labels = result[..., 0].astype(np.int64)
    labels[result[..., 1] > 0] = 0
    return labels


def graphs(module: str, seed=0, max_boxes=6) -> Iterator:
    """
    Yield the graphs of the :data:`FAMILIES` with three to ``max_boxes``
    vertices, the cube with eight, as closed diagrams of a module: a vertex
    is a box named ``f`` with an input for each smaller neighbour and an
    output for each larger, an edge a wire of type ``a`` between them.
    Raises :class:`ValueError` when ``max_boxes`` is less than three.

    >>> first = next(graphs("compact", seed=1))
    >>> assert not (first.dom or first.cod)
    """
    if max_boxes < 3:
        raise ValueError(f"Expected max_boxes >= 3, got {max_boxes}.")
    mod, rng = load(module), np.random.default_rng(seed)
    x = mod.Ty('a')
    while True:
        family, n = rng.choice(list(FAMILIES)), int(
            rng.integers(3, max_boxes + 1))
        graph = nx.convert_node_labels_to_integers(FAMILIES[family](n, rng))
        smaller = {v: sorted(u for u in graph[v] if u < v) for v in graph}
        larger = {v: sorted(u for u in graph[v] if u > v) for v in graph}
        boxes = [mod.Box('f', x ** len(smaller[v]), x ** len(larger[v]))
                 for v in graph]
        yield elementary(mod.CMap.from_wiring(boxes, [
            ((u, len(smaller[u]) + larger[u].index(v)),
             (v, smaller[v].index(u))) for u, v in map(sorted, graph.edges)
        ]).to_diagram())


def augment(image, mask, rng: np.random.Generator, **flags):
    """
    Move an image and its mask together, or the image alone, so that the
    labels stay exact, by each flag of :data:`AUGMENTATIONS` that is set:

    * ``warp``: a smooth random displacement field,
    * ``rotate``: a few degrees of rotation,
    * ``perspective``: a random homography close to the identity,
    * ``chalk``: light strokes with grain on a dark board, with smudges,
    * ``whiteboard``: marker on a white board, with glare and uneven light,
    * ``blur``: a Gaussian blur,
    * ``jpeg``: JPEG compression at a low quality.

    >>> from discopy.monoidal import Ty, Box
    >>> f = Box('f', Ty('a'), Ty('a'))
    >>> image, mask = render(f, 64), render(f, 64, mask=True)
    >>> rng = np.random.default_rng(0)
    >>> moved, labels = augment(image, mask, rng, rotate=True, jpeg=True)
    >>> moved.shape, labels.shape, set(labels.flat) <= set(mask.flat)
    ((64, 64, 3), (64, 64), True)
    """
    size = image.shape[0]
    rows, cols = np.mgrid[0:size, 0:size].astype(float)
    rows, cols = rows - size / 2, cols - size / 2
    if flags.get("warp"):
        coarse = rng.normal(0, size / 80, (2, 4, 4))
        field = [np.kron(c, np.ones((size // 4 + 1, size // 4 + 1)))
                 [:size, :size] for c in coarse]
        field = [blur(f, size / 8) for f in field]
        rows, cols = rows + field[0], cols + field[1]
    if flags.get("rotate"):
        theta = np.radians(rng.uniform(-4, 4))
        rows, cols = (np.cos(theta) * rows - np.sin(theta) * cols,
                      np.sin(theta) * rows + np.cos(theta) * cols)
    if flags.get("perspective"):
        a, b = rng.normal(0, .3 / size, 2)
        depth = 1 + a * rows + b * cols
        rows, cols = rows / depth, cols / depth
    rows, cols = rows + size / 2, cols + size / 2
    image = sample_at(image.astype(float), rows, cols, 255.)
    mask = sample_at(mask, rows, cols, 0, nearest=True)
    gray = image.mean(axis=2)
    ink = 1 - gray / 255
    if flags.get("chalk"):
        grain = rng.random(gray.shape) < .7
        board = np.array([40, 60, 50]) + rng.normal(0, 6, 3)
        smudge = blur(rng.random(gray.shape) < .002, 6) * 600
        light = (ink * grain)[..., None] * 230 + smudge[..., None]
        image = np.clip(board + light, 0, 255)
    elif flags.get("whiteboard"):
        marker = np.array(rng.choice(
            [[20, 20, 30], [20, 40, 150], [160, 30, 30]]))
        light = 1 - .25 * np.linspace(0, 1, gray.shape[1])[None, :]\
            * rng.uniform(.5, 1.5)
        cy, cx = rng.uniform(0, size, 2)
        glare = np.exp(-((rows - cy) ** 2 + (cols - cx) ** 2)
                       / (2 * (size / 4) ** 2)) * 60
        board = (240 * light + glare)[..., None]
        image = np.clip(board * (1 - ink[..., None]) + marker * ink[
            ..., None], 0, 255)
    if flags.get("blur"):
        image = np.stack([blur(image[..., c], rng.uniform(.5, 1.2))
                          for c in range(3)], axis=-1)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if flags.get("jpeg"):
        from PIL import Image
        buffer = BytesIO()
        Image.fromarray(image).save(
            buffer, "JPEG", quality=int(rng.integers(20, 60)))
        image = np.asarray(Image.open(buffer).convert("RGB"))
    return image, mask


def blur(array, sigma: float) -> np.ndarray:
    """ A separable Gaussian blur of a two-dimensional array. """
    radius = max(1, int(3 * sigma))
    kernel = np.exp(-np.arange(-radius, radius + 1) ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()
    array = np.asarray(array, dtype=float)
    for axis in (0, 1):
        array = np.apply_along_axis(
            np.convolve, axis, array, kernel, mode="same")
    return array


def sample_at(array, rows, cols, fill, nearest=False) -> np.ndarray:
    """ The values of an array at real coordinates, bilinear or nearest,
    ``fill`` outside it.

    >>> grey = np.full((2, 2, 1), 9.)
    >>> sample_at(grey, np.array([-5., .5, 3.]), np.array([0., .5, 0.]), 0.)
    array([[0.],
           [9.],
           [0.]])
    """
    size = array.shape[0]
    if nearest:
        r, c = np.rint(rows).astype(int), np.rint(cols).astype(int)
        inside = (r >= 0) & (r < size) & (c >= 0) & (c < size)
        result = np.full(rows.shape + array.shape[2:], fill, array.dtype)
        result[inside] = array[r[inside], c[inside]]
        return result
    r0, c0 = np.floor(rows).astype(int), np.floor(cols).astype(int)
    dr, dc = (rows - r0)[..., None], (cols - c0)[..., None]
    padded = np.pad(array, ((2, 2), (2, 2), (0, 0)), constant_values=fill)
    r0, c0 = np.clip(r0 + 2, 0, size + 2), np.clip(c0 + 2, 0, size + 2)
    return (padded[r0, c0] * (1 - dr) * (1 - dc)
            + padded[r0 + 1, c0] * dr * (1 - dc)
            + padded[r0, c0 + 1] * (1 - dr) * dc
            + padded[r0 + 1, c0 + 1] * dr * dc)


def samples(module: str, seed=0, size=160, max_boxes=6, scales=(20, 28),
            text=None, **flags) -> Iterator:
    """
    Yield triples ``(diagram, image, mask)``: a diagram of a module from
    :func:`diagrams`, :func:`dress`-ed by the flags ``heights``, ``widths``
    and ``triangles`` (on by default), its picture of ``size`` pixels, by
    :func:`render` with ``wire_labels`` (on by default) and ``xkcd``, and
    the labels of its pixels, both moved by the flags of :func:`augment`.
    A picture is drawn at a scale drawn from ``scales``, in pixels per unit,
    its text a fraction of it drawn from ``text`` if given, else ``.4``.

    ``graphviz`` lays the diagram out by graphviz, see :func:`layout`, and
    ``graphs`` draws it from :func:`graphs` instead of :func:`diagrams`.

    Every flag is drawn at random when set to ``"random"``, e.g.
    ``samples("compact", blur="random")`` blurs half the images.
    """
    rng = np.random.default_rng(seed)
    defaults = {"heights": True, "widths": True, "triangles": True,
                "wire_labels": True, "graphviz": False, "graphs": False}
    sources = diagrams(module, seed, max_boxes), graphs(
        module, seed, max_boxes)
    while True:
        drawn = {key: rng.random() < .5 if value == "random" else value
                 for key, value in (defaults | flags).items()}
        diagram = next(sources[drawn.pop("graphs")], None)
        if diagram is None:
            return
        diagram = dress(diagram, rng, **{
            key: drawn.pop(key) for key in ("heights", "widths", "triangles")})
        params = {"xkcd": drawn.pop("xkcd", False),
                  "wire_labels": drawn.pop("wire_labels"),
                  "graphviz": drawn.pop("graphviz") and layout(diagram),
                  "scale": rng.uniform(*scales),
                  "linewidth": rng.uniform(1, 2.5)}
        if text is not None:
            params["fontsize"] = params["scale"] * rng.uniform(*text)
        if not fits(diagram, size, params["scale"],
                    graphviz=params["graphviz"]):
            continue
        image = render(diagram, size, **params)
        mask = render(diagram, size, mask=True, **params)
        yield (diagram, ) + augment(image, mask, rng, **drawn)


def sample(module: str, seed=0, **flags):
    """
    A diagram of a module, its picture and the labels of its pixels, the
    first of :func:`samples`.

    >>> diagram, image, mask = sample("monoidal", seed=0, size=96)
    >>> image.shape, mask.shape
    ((96, 96, 3), (96, 96))
    >>> assert equal(decode(mask, "monoidal"), diagram, "monoidal")
    """
    return next(samples(module, seed, **flags))


def equal(diagram, other, module: str) -> bool:
    """
    Whether two diagrams of a module are equal up to its axioms: their
    normal forms for a monoidal diagram, their hypergraphs otherwise or
    when one has a component with no boundary, a scalar, which commutes
    with every box by interchange.

    >>> from discopy.symmetric import Ty, Box, Swap
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> assert equal(f @ g >> Swap(x, x), Swap(x, x) >> g @ f, "symmetric")
    """
    if module == "monoidal":
        try:
            return diagram.normal_form() == other.normal_form()
        except NotImplementedError:
            pass  # A closed component is a scalar, which commutes.
    return diagram.to_hypergraph() == other.to_hypergraph()


SHIFTS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def shift(array, dr: int, dc: int, fill):
    """ The array moved by ``(dr, dc)``: ``shift(a, 1, 0)[r] == a[r - 1]``,
    ``fill`` where nothing moves in. """
    result = np.full_like(array, fill)
    rows, cols = array.shape
    result[max(dr, 0):rows + min(dr, 0), max(dc, 0):cols + min(dc, 0)]\
        = array[max(-dr, 0):rows + min(-dr, 0), max(-dc, 0):cols + min(-dc, 0)]
    return result


def components(classes: np.ndarray) -> np.ndarray:
    """
    The 8-connected components of the pixels of equal class, ``-1`` where
    the class is negative, numbered from ``0`` in reading order.

    >>> components(np.array([[1, 1, -1, 2], [-1, -1, -1, 2], [1, -1, 2, 2]]))
    array([[ 0,  0, -1,  1],
           [-1, -1, -1,  1],
           [ 2, -1,  1,  1]])
    """
    size = classes.size
    ids = np.where(classes >= 0, np.arange(size).reshape(classes.shape), size)
    while True:
        previous = ids
        for dr, dc in SHIFTS:
            same = shift(classes, dr, dc, -1) == classes
            ids = np.where(same & (classes >= 0), np.minimum(
                ids, shift(ids, dr, dc, size)), ids)
        if np.array_equal(ids, previous):
            break
    _, numbered = np.unique(ids, return_inverse=True)
    numbered = numbered.reshape(classes.shape)
    return np.where(classes >= 0, numbered, -1)


def decode(mask: np.ndarray, module: str, min_pixels=3):
    """
    The diagram of a module whose picture has these labels, see
    :data:`LABELS`. Raises :class:`ValueError` when the labels are not those
    of a picture.

    The generators are the components of box pixels of each label, the
    links those of wire and port pixels together. A link meets a generator
    on its top or bottom, at a port, ordered from left to right. A sweep
    from top to bottom keeps the list of open links: a generator closes the
    adjacent links on its top and opens those on its bottom. A link with no
    type is given one by the generators it meets, e.g. the two sides of a
    swap, else the first of :data:`TYPES`.

    >>> from discopy.symmetric import Ty, Box, Swap
    >>> x, y = Ty('a'), Ty('b')
    >>> diagram = Box('f', x, x @ y) >> Swap(x, y)
    >>> mask = render(diagram, 96, mask=True)
    >>> print(decode(mask, "symmetric"))
    f >> Swap(a, b)
    >>> assert equal(decode(mask, "symmetric"), diagram, "symmetric")
    """
    mask, mod = np.asarray(mask, dtype=np.intp), load(module)
    if mask.size and not 0 <= mask.min() <= mask.max() < len(LABELS):
        raise ValueError(f"Expected labels in range({len(LABELS)}).")
    first = LABELS.index(NAMES[0])
    allowed = [LABELS.index(name) for name in NAMES + STRUCTURAL[module]]
    for name in set(STRUCTURAL["frobenius"]) - set(STRUCTURAL[module]):
        ids = components(np.where(mask == LABELS.index(name), 0, -1))
        if (np.bincount(ids[ids >= 0], minlength=1) >= min_pixels).any():
            raise ValueError(f"A {name} is not in {module}.")
    is_box = np.isin(mask, allowed)
    boxes = join(components(np.where(is_box, mask, -1)), mask,
                 (mask >= LABELS.index("label " + TYPES[0])) & (mask != WIRE))
    links = components(np.where(
        (mask > 0) & (mask < first) | (mask == WIRE), 0, -1))
    small = np.bincount(links[links >= 0], minlength=1) < min_pixels
    links = components(np.where(
        (links >= 0) & ~small[np.maximum(links, 0)], 0, -1))
    sizes = np.bincount(boxes[boxes >= 0])
    rows, cols = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    generators = [g for g in range(len(sizes)) if sizes[g] >= min_pixels]
    centre = {g: rows[boxes == g].mean() for g in generators}
    contacts = defaultdict(list)
    for dr, dc in SHIFTS:
        moved = shift(boxes, dr, dc, -1)
        touch = (links >= 0) & (moved >= 0)
        for link, g, r, c in zip(
                links[touch], moved[touch], rows[touch], cols[touch]):
            contacts[link, g].append((r, c))
    tops, bottoms, ends = defaultdict(list), defaultdict(list), defaultdict(
        list)
    for (link, g), points in contacts.items():
        if g not in centre:
            continue
        r, c = np.mean(points, axis=0)
        side = tops if r < centre[g] else bottoms
        side[g].append((c, link))
        ends[link].append((g, side is tops, r))
    types = link_types(mask, links, first)
    inputs, outputs = [], []
    for link in range(links.max() + 1):
        at = [top for _, top, _ in ends[link]]
        rs, cs = rows[links == link], cols[links == link]
        if not any(not top for top in at):
            inputs.append((cs[rs == rs.min()].mean(), link))
        if not any(at):
            outputs.append((cs[rs == rs.max()].mean(), link))
    label = {g: LABELS[np.bincount(mask[boxes == g]).argmax()]
             for g in generators}
    tops = {g: [link for _, link in sorted(tops[g])] for g in generators}
    bottoms = {g: [link for _, link in sorted(bottoms[g])] for g in generators}
    infer_types(types, label, tops, bottoms, module)
    for link in range(links.max() + 1):
        if link not in types:
            types[link] = TYPES[0]
            infer_types(types, label, tops, bottoms, module)

    def typ(link):
        name = types.get(link, TYPES[0])
        result = mod.Ty(name.removesuffix(".r"))
        if name.endswith(".r") and not hasattr(result, "r"):
            raise ValueError(f"No adjoint {name} in {module}.")
        return result.r if name.endswith(".r") else result

    def top_row(g):
        return min((r for link in tops[g] for h, top, r in ends[link]
                    if h == g and top), default=rows[boxes == g].min())

    def column(link, row):
        rs, cs = rows[links == link], cols[links == link]
        nearest = rs[np.abs(rs - row).argmin()]
        return cs[rs == nearest].mean()

    open_ = [link for _, link in sorted(inputs)]
    result = mod.Diagram.id(mod.Ty().tensor(*map(typ, open_)))
    for g in sorted(generators, key=lambda g: (
            top_row(g), cols[boxes == g].mean())):
        dom = tops[g]
        if dom:
            if dom[0] not in open_:
                raise ValueError(f"Generator {label[g]} has no open input.")
            i = open_.index(dom[0])
            if open_[i:i + len(dom)] != dom:
                raise ValueError(f"Inputs of {label[g]} are not adjacent.")
        else:
            c, row = cols[boxes == g].mean(), rows[boxes == g].min()
            i = sum(column(link, row) < c for link in open_)
        box = generator(mod, label[g], list(map(typ, dom)),
                        list(map(typ, bottoms[g])))
        left = mod.Ty().tensor(*map(typ, open_[:i]))
        right = mod.Ty().tensor(*map(typ, open_[i + len(dom):]))
        result = result >> left @ box @ right
        open_[i:i + len(dom)] = bottoms[g]
    if open_ != [link for _, link in sorted(outputs)]:
        raise ValueError("The outputs do not match the open links.")
    return result


def join(boxes: np.ndarray, mask: np.ndarray, text: np.ndarray):
    """
    The components of the boxes of a mask, two of one name joined where a
    label drawn across a box cuts it in two: labels cover boxes, so that
    one under a box keeps its type, see :func:`render`.
    """
    joined = components(np.where((boxes >= 0) | text, 0, -1))
    for j in np.unique(joined[boxes >= 0]):
        inside = (joined == j) & (boxes >= 0)
        if len(np.unique(mask[inside])) == 1:
            boxes[inside] = boxes[inside].min()
    return boxes


def link_types(mask, links, first) -> dict[int, str]:
    """ The type of each link with wire pixels, by majority. """
    types = {}
    wire = (mask > 0) & (mask < first)  # a wire of unknown type is not
    for link in np.unique(links[wire]):
        types[int(link)] = LABELS[np.bincount(
            mask[wire & (links == link)]).argmax()]
    return types


def adjoint(name: str, module: str) -> str:
    """ The name of the adjoint of a type, itself unless compact. """
    if module != "compact":
        return name
    return name.removesuffix(".r") if name.endswith(".r") else name + ".r"


def infer_types(types, label, tops, bottoms, module):
    """ Give a type to each link that has none, from the generators it
    meets, until none is given. """
    def rules(g):
        top, bottom = tops[g], bottoms[g]
        if label[g] == "swap" and len(top) == 2 == len(bottom):
            yield from [(top[0], bottom[1], str), (top[1], bottom[0], str)]
        if label[g] in ("cup", "cap"):
            legs = top if label[g] == "cup" else bottom
            if len(legs) == 2:
                yield legs[0], legs[1], lambda x: adjoint(x, module)
        if label[g] == "spider":
            legs = top + bottom
            yield from [(legs[0], leg, str) for leg in legs[1:]]
    changed = True
    while changed:
        changed = False
        for g in label:
            for a, b, func in rules(g):
                if a in types and b not in types:
                    types[b], changed = func(types[a]), True
                elif b in types and a not in types:
                    types[a], changed = func(types[b]), True


def generator(mod, label: str, dom: list, cod: list):
    """ The box of a module for a label and the types of its ports. """
    if label in NAMES:
        return mod.Box(label, mod.Ty().tensor(*dom), mod.Ty().tensor(*cod))
    arity = {"swap": (2, 2), "cup": (2, 0), "cap": (0, 2)}.get(label)
    if arity is not None and arity != (len(dom), len(cod)):
        raise ValueError(f"A {label} with {len(dom)} inputs, {len(cod)} "
                         "outputs.")
    if label == "swap":
        return mod.Swap(*dom)
    if label == "cup":
        return mod.Cup(*dom)
    if label == "cap":
        return mod.Cap(*cod)
    legs = dom + cod
    if not legs or any(x != legs[0] for x in legs):
        raise ValueError("A spider with no legs or legs of many types.")
    return mod.Spider(len(dom), len(cod), legs[0])


def coarse(label: str) -> str:
    """
    The class of :data:`COARSE` a label of :data:`LABELS` falls in.

    >>> [coarse(x) for x in ("a.r", "f", "label a", "swap")]
    ['wire', 'box', 'label', 'swap']
    """
    if label in TYPES or label == "wire":
        return "wire"
    if label in NAMES:
        return "box"
    return "label" if label.startswith("label ") else label


def coarsen(mask: np.ndarray) -> np.ndarray:
    """ The classes of :data:`COARSE` of the labels of a mask. """
    table = np.array([COARSE.index(coarse(label)) for label in LABELS])
    return table[mask]


def regions(classes: np.ndarray, min_pixels=3):
    """
    The boxes, the wires and the labels of a picture from the classes of
    its pixels, see :data:`COARSE`: three arrays of component ids, ``-1``
    elsewhere. The characters of one label are one component, the label
    pixels grown by two first; a label drawn over a box is part of the box
    too, which joins it across the label; a component of fewer than
    ``min_pixels`` pixels is dropped.
    """
    def drop_small(ids):
        sizes = np.bincount(ids[ids >= 0], minlength=1)
        return np.where((ids >= 0) & (sizes[np.maximum(ids, 0)] >= min_pixels),
                        ids, -1)
    text = classes == COARSE.index("label")
    box = classes == COARSE.index("box")
    joined = components(np.where(box | text, 0, -1))
    boxes = np.where(np.isin(joined, joined[box]), joined, -1)
    wires = components(np.where(classes == COARSE.index("wire"), 0, -1))
    near = text
    for _ in range(2):
        near = near | np.any([shift(near, dr, dc, False)
                              for dr, dc in SHIFTS], axis=0)
    labels = np.where(text, components(np.where(near, 0, -1)), -1)
    return drop_small(boxes), drop_small(wires), drop_small(labels)


def attach(labels: np.ndarray, wires: np.ndarray) -> dict[int, int]:
    """ The wire each label names: the one nearest its top left pixel,
    as :meth:`Backend.draw_wire_label` draws a label right of its wire. """
    wire_rows, wire_cols = np.nonzero(wires >= 0)
    result = {}
    for label in np.unique(labels[labels >= 0]):
        rows, cols = np.nonzero(labels == label)
        left = cols.min()
        top = rows[cols == left].min()
        if len(wire_rows):
            nearest = ((wire_rows - top) ** 2
                       + (wire_cols - left) ** 2).argmin()
            result[int(label)] = int(
                wires[wire_rows[nearest], wire_cols[nearest]])
    return result


def refine(classes, boxes, wires, names: dict, types: dict) -> np.ndarray:
    """
    The mask of :data:`LABELS` from the classes of the pixels, see
    :data:`COARSE`, the components of :func:`regions`, the name of each
    box and the type of each wire: a wire with no type is ``"wire"``, a
    label is background. Raises :class:`ValueError` on a name or a type
    outside :data:`NAMES` and :data:`TYPES`.

    >>> from discopy.monoidal import Ty, Box
    >>> f = Box('f', Ty('a'), Ty('b'))
    >>> mask = render(f, 64, mask=True)
    >>> classes = coarsen(mask)
    >>> boxes, wires, labels = regions(classes)
    >>> names = {b: "f" for b in np.unique(boxes[boxes >= 0])}
    >>> mask = refine(classes, boxes, wires, names, {})
    >>> assert equal(decode(mask, "monoidal"),
    ...              Box('f', Ty('a'), Ty('a')), "monoidal")
    """
    for name in names.values():
        if name not in NAMES:
            raise ValueError(f"Unknown box name {name!r}.")
    for typ in types.values():
        if typ not in TYPES:
            raise ValueError(f"Unknown wire type {typ!r}.")
    fine = np.zeros(classes.shape, dtype=np.intp)
    for name in STRUCTURAL["frobenius"]:
        fine[classes == COARSE.index(name)] = LABELS.index(name)
    fine[wires >= 0] = WIRE
    for wire, typ in types.items():
        fine[wires == wire] = LABELS.index(typ)
    for box, name in names.items():
        fine[boxes == box] = LABELS.index(name)
    return fine


def scan(classes: np.ndarray, types=(), min_pixels=3) -> str:
    """
    The smallest module whose structural primitives appear in a picture,
    a component of at least ``min_pixels`` each, from the classes of its
    pixels: ``monoidal`` with none, ``symmetric``
    with swaps, ``compact`` with cups or caps, ``frobenius`` with spiders,
    and ``compact`` at least if one of the ``types`` read is an adjoint.

    >>> from discopy.symmetric import Ty, Box, Swap
    >>> x = Ty('a')
    >>> scan(coarsen(render(Box('f', x, x), 64, mask=True)))
    'monoidal'
    >>> scan(coarsen(render(Swap(x, x), 64, mask=True)))
    'symmetric'
    >>> scan(coarsen(render(Swap(x, x), 64, mask=True)), ['a.r'])
    'compact'
    """
    def appears(name):
        ids = components(np.where(classes == COARSE.index(name), 0, -1))
        return bool((np.bincount(ids[ids >= 0], minlength=1)
                     >= min_pixels).any())
    seen = set(filter(appears, STRUCTURAL["frobenius"]))
    least = MODULES.index("compact") if any(
        typ.endswith(".r") for typ in types) else 0
    return next(module for module in MODULES[least:]
                if seen <= set(STRUCTURAL[module]))


def read(classes: np.ndarray, name: Callable, typ: Callable, module=None):
    """
    The diagram a picture shows from the classes of its pixels: each box is
    named by ``name(rows, cols)`` and each labelled wire typed by
    ``typ(rows, cols)``, given the pixels of the box or the label, the
    module is the smallest from the :func:`scan` of the classes and the
    types read in which it decodes, e.g. ``frobenius`` for a cup on ``a``
    and ``a`` that is not compact, unless given, then :func:`decode`.
    """
    boxes, wires, labels = regions(classes)
    names = {int(b): name(*np.nonzero(boxes == b))
             for b in np.unique(boxes[boxes >= 0])}
    types = {wires_id: typ(*np.nonzero(labels == label))
             for label, wires_id in attach(labels, wires).items()}
    mask = refine(classes, boxes, wires, names, types)
    if module is not None:
        return decode(mask, module)
    return decode_least(mask, scan(classes, types.values()))


def decode_least(mask: np.ndarray, module: str, most=MODULES[-1]):
    """
    :func:`decode` in the least module from ``module`` up to ``most`` in
    which the mask decodes, e.g. ``frobenius`` for a cup on ``a`` and ``a``,
    which is not compact: a :func:`scan` cannot tell them apart.
    """
    least, last = MODULES.index(module), MODULES.index(most)
    if last < least:
        raise ValueError(f"Expected {most!r} at or after {module!r}.")
    for i, candidate in enumerate(MODULES[least:last + 1], least):
        try:
            return decode(mask, candidate)
        except (AxiomError, ValueError):
            if i == last:
                raise


def crop(image: np.ndarray, rows, cols, label=False) -> np.ndarray:
    """
    The ink of an image, from zero for white to one for black, in a window
    of shape :data:`CROP`: centred on the box with these pixels, where its
    name is drawn, or from the top left of the label with these pixels,
    with no ink outside their bounding box. Keeping only the label's own
    pixels, or erasing those classed wire, box or primitive, reads a box
    edge through a label better but loses the text the pixel net misses:
    recovery fell by 7 to 15 points (#559).
    """
    height, width = CROP
    if label:
        top = rows[cols == cols.min()].min() - height // 4
        left = cols.min() - 4
    else:
        top = (rows.min() + rows.max()) // 2 - height // 2
        left = (cols.min() + cols.max()) // 2 - width // 2
    ink = 1 - image.mean(axis=2) / 255
    if label:
        inside = np.zeros(ink.shape, dtype=bool)
        inside[max(rows.min() - 1, 0):rows.max() + 2,
               max(cols.min() - 1, 0):cols.max() + 2] = True
        ink = ink * inside
    ink = np.pad(ink, ((height, height), (width, width)))
    return ink[top + height:top + 2 * height,
               left + width:left + 2 * width].astype(np.float32)
