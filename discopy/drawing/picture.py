"""
Pictures of diagrams: sample a diagram, draw it as an image and label each
of its pixels.

Drawing is a functor from diagrams to images, reading one back is a
section of it, up to the axioms of its module, see :func:`equal`. Training
data for reading is free: :func:`sample` draws a diagram of a module, its
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
    augment
    sample
    samples
    equal
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from io import BytesIO

import numpy as np

from discopy.drawing.backend import Backend, Matplotlib
from discopy.testable import GENERATORS
from discopy.utils import Node

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

#: The flags of :func:`augment`, applied in this order.
AUGMENTATIONS = (
    "warp", "rotate", "perspective", "chalk", "whiteboard", "blur", "jpeg")

DPI, SPIDER, WIRE, TEXT = 72, .15, LABELS.index("wire"), 7


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
    label covers the background and boxes, never a wire or a primitive.

    The diagram is centred at ``scale`` pixels per unit, ``24`` by default,
    its text ``fontsize`` pixels high, its strokes ``linewidth`` wide.
    ``xkcd=True`` draws it by hand, see :func:`matplotlib.pyplot.xkcd`.

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


def paint(drawing, size, scale, xkcd, backend, mask, mode=None, **params):
    """ The pixels of a drawing on a backend, see :func:`render`. """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    with plt.xkcd(randomness=1, scale=.5, length=50) if xkcd\
            else nullcontext(), mpl.rc_context({
                "lines.antialiased": not mask, "patch.antialiased": not mask,
                "text.antialiased": not mask, "path.snap": False}):
        figure, axis = plt.subplots(
            figsize=(size / DPI, size / DPI), dpi=DPI,
            facecolor="black" if mask else "white")
        painter = backend(axis, params.pop("linewidth", 1.5), scale=scale,
                          mode=mode) if mask else backend(
            axis, linewidth=params.pop("linewidth", 1.5))
        drawing.draw(backend=painter, show=False, aspect="equal", **params)
        half = size / scale / 2
        axis.set_xlim(drawing.width / 2 - half, drawing.width / 2 + half)
        axis.set_ylim(drawing.height / 2 - half, drawing.height / 2 + half)
        figure.canvas.draw()
        pixels = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        plt.close(figure)
    return pixels


def fits(diagram, size=160, scale=24, margin=1) -> bool:
    """ Whether the picture of a diagram fits in the image, with a margin
    in units around it. """
    drawing = diagram.to_drawing()
    return max(drawing.width, drawing.height) + 2 * margin <= size / scale


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

    Every flag is drawn at random when set to ``"random"``, e.g.
    ``samples("compact", blur="random")`` blurs half the images.
    """
    rng = np.random.default_rng(seed)
    defaults = {"heights": True, "widths": True, "triangles": True,
                "wire_labels": True}
    for diagram in diagrams(module, seed, max_boxes):
        drawn = {key: rng.random() < .5 if value == "random" else value
                 for key, value in (defaults | flags).items()}
        diagram = dress(diagram, rng, **{
            key: drawn.pop(key) for key in ("heights", "widths", "triangles")})
        params = {"xkcd": drawn.pop("xkcd", False),
                  "wire_labels": drawn.pop("wire_labels"),
                  "scale": rng.uniform(*scales),
                  "linewidth": rng.uniform(1, 2.5)}
        if text is not None:
            params["fontsize"] = params["scale"] * rng.uniform(*text)
        if not fits(diagram, size, params["scale"]):
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
