# -*- coding: utf-8 -*-
"""
DisCopy's drawing backends: Matplotlib, TikZ and Typst.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Backend
    TikZ
    Matplotlib
    Typst

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        draw
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from io import StringIO, TextIOBase
from itertools import combinations
from math import sqrt
import os
import pathlib
import re
from types import SimpleNamespace
from xml.etree import ElementTree

from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cbook
from matplotlib.colors import is_color_like, to_hex, to_rgb
from matplotlib.testing.compare import compare_images
from matplotlib.testing.exceptions import ImageComparisonFailure
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path
from PIL import Image, ImageSequence

from discopy import config
from discopy.drawing import Node, Point
from discopy.utils import text_width
from discopy.drawing.typst_ast import (
    Bool, Call, Canvas, ContentExpr, Coord, Document, Float, Ident,
    Import, RawText, Str, TypstNode, latex_math)

from discopy.config import (  # noqa: F401  pylint: disable=unused-import
    BOX_DRAWING_ATTRIBUTES as ATTRIBUTES,
    DRAWING_DEFAULT as DEFAULT, COLORS, SHAPES, RIBBON_FOLD_DEPTH,
    TRANSPARENT)

if TYPE_CHECKING:
    from discopy.drawing import Drawing


MATPLOTLIB_RC = {
    "agg.path.chunksize": 0,
    "figure.dpi": 100.,
    "font.family": ["sans-serif"],
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10.,
    "mathtext.fontset": "dejavusans",
    "path.simplify": True,
    "path.simplify_threshold": 0.111111111111,
    "savefig.bbox": None,
    "savefig.dpi": "figure",
    "savefig.pad_inches": 0.1,
    "savefig.transparent": False,
    "svg.fonttype": "path",
    "svg.hashsalt": "discopy",
    "svg.image_inline": True,
    "text.antialiased": True,
    "text.hinting": "force_autohint",
    "text.hinting_factor": 8,
    # Keep ``$...$`` mathtext, but never invoke a machine-local TeX install.
    "text.usetex": False,
}


@contextmanager
def matplotlib_context():
    """ Isolate DisCoPy drawings from the caller's Matplotlib settings. """
    with matplotlib.rc_context(MATPLOTLIB_RC):
        yield


def draw(graph: Drawing, **params):
    """ Load a :class:`Backend` and draw a :class:`Drawing` on it. """
    if params.get("format") == "typst":
        path, compare = doctest_or_path(
            params.pop("path", None), params.pop("doctest", None))
        show = params.pop("show", True)
        compare = params.pop("compare", False) or compare
        return Typst(**params).draw(graph, **params).output(
            path=path, show=show, compare=compare, **params)
    aspect = params.get('aspect', 'auto' if 'figsize' in params else 'equal')
    left_to_right = params.get('left_to_right', False)
    width, height = (graph.height, graph.width) if left_to_right\
        else (graph.width, graph.height)
    if params.get('legend', False) and not params.get('to_tikz', False):
        colours = Backend.region_colours(graph)
        if colours:
            # Widen the figure by the legend width rather than squeeze diagram.
            longest = max(len(c.legend_label) for c in colours.values())
            legend_inches = DEFAULT['legend_base_width']\
                + DEFAULT['legend_char_width'] * longest
            margin_inches = DEFAULT['legend_margin']
            extra = legend_inches + margin_inches
            fig_width = params['figsize'][0] if 'figsize' in params\
                else (width or 1)
            if 'figsize' in params:
                params['figsize'] = (fig_width + extra, params['figsize'][1])
            space = params.get('legend_space', extra * width / fig_width)
            if left_to_right:  # The physical right of a transposed drawing.
                width += space
            else:
                graph = graph.make_space(
                    space, graph.width, exclusive=True, copy=True)
                width = graph.width
    figsize = params.get('figsize', None if aspect == 'auto' else (
        width or 1, height or 1))
    backend = params.get('backend', None) or (
        TikZ(use_tikzstyles=params.get('use_tikzstyles', None))
        if params.get('to_tikz', False)
        else Matplotlib(figsize=figsize,
                        linewidth=params.get('linewidth', 1),
                        format=params.get('format', None)))

    max_v = max(graph.height, graph.width, 0.01)
    params['nodesize'] = round(params.get('nodesize', 1.) / sqrt(max_v), 3)

    if left_to_right:
        backend.draw_boundary(
            SimpleNamespace(width=width, height=height), **params)
        canvas = Transposed(backend, Point(0, 0), graph)
    else:
        canvas = backend
        canvas.draw_boundary(graph, **params)
    canvas.draw_drawing(graph, **params)
    if params.get('legend', False):
        backend.draw_legend(graph, **params)

    path, compare = doctest_or_path(
        params.get('path', None), params.get('doctest', None))
    return backend.output(
        path=path,
        baseline=height / 2 or .5,
        tikz_options=params.get('tikz_options', None),
        show=params.get('show', True), aspect=aspect,
        margins=params.get('margins', DEFAULT['margins']),
        compare=compare,
        tol=params.get('tol', DEFAULT['plt_tol']))


def doctest_or_path(path=None, doctest=None):
    """ A doctest path is a baseline that drawing checks against,
    a plain path is just where the drawing gets saved. """
    return (doctest, True) if doctest is not None else (path, False)


XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
NUMBER = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
CLIP_REFERENCE = re.compile(r"url\(#([^)]+)\)")
GEOMETRY_ATTRIBUTES = {
    "d", "points", "transform", "viewBox",
    "x", "y", "dx", "dy", "x1", "y1", "x2", "y2",
    "cx", "cy", "r", "rx", "ry", "fx", "fy", "fr",
    "refX", "refY", "width", "height", "markerWidth", "markerHeight",
    "pathLength",
}


def local_name(name):
    """ Drop an XML namespace from a tag or attribute name. """
    return name.rsplit("}", 1)[-1]


def close_enough(expected: str, actual: str, tol: float) -> bool:
    """ Whether two geometry strings differ by at most `tol`. """
    if NUMBER.sub("#", expected) != NUMBER.sub("#", actual):
        return False
    return all(
        x == y or abs(float(x) - float(y)) <= tol for x, y in
        zip(NUMBER.findall(expected), NUMBER.findall(actual)))


def normalize_svg(path):
    """ Remove metadata and normalize geometry-derived clip ids. """
    root = ElementTree.parse(path).getroot()
    clip_ids = {}

    def clip_id(identifier):
        return f"clip-{clip_ids.setdefault(identifier, len(clip_ids))}"

    for parent in root.iter():
        for child in list(parent):
            if local_name(child.tag) == "metadata":
                parent.remove(child)
    for element in root.iter():
        element.attrib.pop(XML_SPACE, None)
        if element.text is not None and not element.text.strip():
            element.text = None
        element.tail = None
        if local_name(element.tag) == "clipPath"\
                and "id" in element.attrib:
            element.set("id", clip_id(element.attrib["id"]))
        reference = CLIP_REFERENCE.fullmatch(
            element.attrib.get("clip-path", ""))
        if reference:
            element.set(
                "clip-path", f"url(#{clip_id(reference[1])})")
    return root


def svg_equal(path, actual_path, tol=DEFAULT['svg_tol']) -> bool:
    """ Compare exact SVG structure with geometry tolerance. """
    def equal(expected, actual):
        return expected.tag == actual.tag\
            and expected.text == actual.text\
            and expected.attrib.keys() == actual.attrib.keys()\
            and all(
                close_enough(value, actual.attrib[key], tol)
                if local_name(key) in GEOMETRY_ATTRIBUTES
                else value == actual.attrib[key]
                for key, value in expected.attrib.items())\
            and len(expected) == len(actual)\
            and all(map(equal, expected, actual))

    return equal(normalize_svg(path), normalize_svg(actual_path))


def temporary_path(path):
    """ Prefix the basename of a path with an underscore. """
    folder, name = os.path.split(os.fspath(path))
    return os.path.join(folder, "_" + name)


def compare_drawing(path, actual_path, tol=DEFAULT['plt_tol']):
    """ Compare a drawing against its baseline and remove it when equal. """
    extension = os.path.splitext(os.fspath(path))[1].lower()
    if extension == ".svg":
        equal = svg_equal(path, actual_path)
        difference = None
    elif extension in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        try:
            difference = compare_images(path, actual_path, tol)
        except ImageComparisonFailure as error:
            difference = str(error)
        equal = difference is None
    elif extension == ".gif":
        def gif_data(gif_path):
            with Image.open(gif_path) as image:
                default_duration = image.info.get("duration")
                loop = image.info.get("loop")
                frames = [
                    (frame.convert("RGBA"), frame.size,
                     frame.info.get("duration", default_duration))
                    for frame in ImageSequence.Iterator(image)]
            return frames, loop

        expected_frames, expected_loop = gif_data(path)
        actual_frames, actual_loop = gif_data(actual_path)
        same_animation = expected_loop == actual_loop\
            and len(expected_frames) == len(actual_frames)\
            and all(
                expected[1:] == actual[1:]
                for expected, actual
                in zip(expected_frames, actual_frames))
        rms = [
            np.sqrt(np.mean((
                np.asarray(expected[0], dtype=float)
                - np.asarray(actual[0], dtype=float)) ** 2))
            for expected, actual in zip(expected_frames, actual_frames)
            if expected[1] == actual[1]]
        equal = same_animation and len(rms) == len(expected_frames)\
            and all(value <= tol for value in rms)
        difference = None
    else:
        with open(path, "rb") as expected, open(actual_path, "rb") as actual:
            equal = expected.read() == actual.read()
        difference = None
    if not equal:
        message = f"Drawing differs from {path!r}; see {actual_path!r}."
        if difference is not None:
            message += f"\n{difference}"
        raise ValueError(message)
    os.remove(actual_path)


def save_and_compare(path, save, tol=DEFAULT['plt_tol']):
    """ Save a drawing baseline when missing, compare against it otherwise. """
    if config.OVERRIDE_DOCTEST_IMAGES or not os.path.exists(path):
        save(path)
        return
    actual_path = temporary_path(path)
    save(actual_path)
    compare_drawing(path, actual_path, tol)


DARK_MODE_STYLE = (
    '<style type="text/css">'
    '@media (prefers-color-scheme: dark) {'
    '[id^="dark-stroke"] path { stroke: #ffffff !important; } '
    '[id^="dark-fill"] g, [id^="dark-fill"] use '
    '{ fill: #ffffff !important; stroke: #ffffff !important; }'
    '}</style>')


def add_dark_mode_style(text):
    """
    Insert :data:`DARK_MODE_STYLE` after the opening tag of the SVG ``text``:
    a media query turning the elements tagged by :meth:`Matplotlib.dark_gid`
    white on a dark page, every other renderer keeping the static black on a
    transparent canvas.
    """
    opening = text.index(">", text.index("<svg")) + 1
    return text[:opening] + DARK_MODE_STYLE + text[opening:]


def savefig(path, format=None, compare=False, tol=DEFAULT['plt_tol']):
    """
    Save the current Matplotlib figure, as a baseline when ``compare``.

    The format is taken from the extension of ``path`` when it is a file name,
    it has to be given explicitly when ``path`` is an in-memory buffer. PNGs
    embed the Matplotlib version as ``"Software"``, SVGs embed the current date
    and randomise the ids of their clip paths, so we drop the former and fix
    the salt used for the latter.

    SVGs are saved on a transparent canvas with :func:`add_dark_mode_style`,
    so a single file reads on both light and dark pages; raster formats
    cannot adapt to the page behind them, so they keep their white background.
    """
    path_str = str(path)
    is_svg = format == "svg" or (format is None and path_str.endswith(".svg"))
    is_png = format == "png" or (format is None and path_str.endswith(".png"))
    metadata = {
        "Date": None, "Creator": None, "Format": None, "Type": None
    } if is_svg else\
        {"Software": None} if is_png else None

    def save(actual_path):
        target = StringIO() if is_svg else actual_path
        with matplotlib_context():
            plt.savefig(target, format="svg" if is_svg else format,
                        metadata=metadata, transparent=is_svg)
        if is_svg:
            text = add_dark_mode_style(target.getvalue())
            with cbook.open_file_cm(
                    actual_path, "w", encoding="utf-8") as file:
                file.write(text if cbook.file_requires_unicode(file)
                           else text.encode("utf-8"))

    save_and_compare(path, save, tol) if compare else save(path)


def blossom(points, u, v):
    """
    Evaluate the `blossom`_ of a quadratic Bezier (3 control points) on a
    pair of parameters, i.e. the symmetric multi-affine map that gives back
    the curve on the diagonal: ``blossom(points, t, t)`` is the point of
    the curve at ``t``.

    .. _blossom: https://en.wikipedia.org/wiki/Blossom_(functional)
    """
    (x0, y0), (xc, yc), (x1, y1) = points
    return Point(
        (1 - u) * (1 - v) * x0 + ((1 - u) * v + u * (1 - v)) * xc
        + u * v * x1,
        (1 - u) * (1 - v) * y0 + ((1 - u) * v + u * (1 - v)) * yc
        + u * v * y1)


def quadratic_subcurve(points, t0, t1):
    """
    Restrict a quadratic Bezier (3 control points) to the range [t0, t1],
    i.e. evaluate its :func:`blossom` on the pairs (t0, t0), (t0, t1) and
    (t1, t1).
    """
    return (blossom(points, t0, t0), blossom(points, t0, t1),
            blossom(points, t1, t1))


def lerp(a, b, t):
    """ The point at parameter ``t`` on the segment from ``a`` to ``b``. """
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def bezier_subcurve(points, t0, t1):
    """ Restrict a cubic Bezier (4 control points) to the range [t0, t1]. """
    def split(p, t):  # The two halves of a cubic Bezier split at ``t``.
        a, b, c = lerp(p[0], p[1], t), lerp(p[1], p[2], t), lerp(p[2], p[3], t)
        d, e = lerp(a, b, t), lerp(b, c, t)
        f = lerp(d, e, t)
        return [p[0], a, d, f], [f, e, c, p[3]]

    right = split(points, t0)[1]
    return split(right, (t1 - t0) / (1 - t0))[0]


def bezier(points, t):
    """
    The point at parameter ``t`` of a Bezier curve of any degree, by de
    Casteljau's algorithm: :func:`lerp` between consecutive control points
    until one remains.
    """
    while len(points) > 1:
        points = [lerp(a, b, t) for a, b in zip(points, points[1:])]
    return Point(*points[0])


def subcurve(points, t0, t1):
    """
    Restrict a quadratic or a cubic Bezier to the range [t0, t1], see
    :func:`quadratic_subcurve` and :func:`bezier_subcurve`.
    """
    restrict = quadratic_subcurve if len(points) == 3 else bezier_subcurve
    return tuple(Point(*point) for point in restrict(points, t0, t1))


def monotone_root(function, low=0., high=1.):
    """
    The root in ``[low, high]`` of a monotone ``function`` that changes sign
    on it, by bisection to double precision.
    """
    increasing = function(low) < function(high)
    for _ in range(64):
        middle = (low + high) / 2
        value = function(middle)
        if value == 0:
            return middle
        if (value < 0) == increasing:
            low = middle
        else:
            high = middle
    return (low + high) / 2


class Backend(ABC):
    """ Abstract drawing backend. """
    def __init__(self):
        self.max_width = 0

    def draw_text(self, text, i, j, **params):
        """ Draws a piece of text at a given position. """
        # pylint: disable=unused-argument  # the base only measures the width
        self.max_width = max(self.max_width, i)

    def draw_node(self, i, j, **params):
        """ Draws a node for a given position, color and shape. """
        # pylint: disable=unused-argument  # the base only measures the width
        self.max_width = max(self.max_width, i)

    def draw_polygon(self, *points, facecolor=None, edgecolor=None):
        """ Draws a polygon given a list of points. """
        # pylint: disable=unused-argument  # the base only measures the width
        self.max_width = max(self.max_width, max(i for i, _ in points))

    @staticmethod
    def is_transparent(colour) -> bool:
        """
        Whether a colour is :data:`discopy.config.TRANSPARENT`, comparing
        only a name, since an array compares elementwise.

        >>> import numpy as np
        >>> assert Backend.is_transparent("none")
        >>> assert not Backend.is_transparent(np.array([.1, .2, .3]))
        """
        return isinstance(colour, str) and colour == TRANSPARENT

    @staticmethod
    def resolve(colour):
        """
        The colour matplotlib reads for a name in :data:`COLORS`, any other
        colour as it is, e.g. an RGB list or array, which is unhashable.

        >>> Backend.resolve("red"), Backend.resolve([.1, .2, .3])
        ('#e8a5a5', [0.1, 0.2, 0.3])
        """
        return COLORS.get(colour, colour) if isinstance(colour, str)\
            else colour

    @staticmethod
    def readable_foreground(colour, threshold=.5):
        """
        Pick a readable foreground text colour, i.e. ``"white"`` or
        ``"black"``, for text drawn on top of a given background
        ``colour``, using the standard luma formula for perceived
        brightness.

        >>> Backend.readable_foreground("white")
        'black'
        >>> Backend.readable_foreground("black")
        'white'
        """
        from matplotlib.colors import to_rgb
        try:
            red, green, blue = to_rgb(Backend.resolve(colour))
        except (ValueError, TypeError):
            return "black"
        luma = 0.299 * red + 0.587 * green + 0.114 * blue
        return "white" if luma < threshold else "black"

    def draw_wire(self, source, target, bend_out=False, bend_in=False,
                  style=None, linewidth=None, adaptive=True):
        """ Draws a wire from source to target, possibly with a Bezier.
        An ``adaptive`` wire lies on the neutral canvas, so its stroke may
        adapt to a dark page, see :meth:`Matplotlib.dark_gid`. """
        # pylint: disable=unused-argument  # the base only measures the width
        self.max_width = max(self.max_width, source[0], target[0])

    def draw_bezier(self, points, adaptive=True):
        """ Draws a cubic Bezier curve from a list of four control points. """
        # pylint: disable=unused-argument  # the base only measures the width
        self.max_width = max(self.max_width, max(x for x, _ in points))

    @staticmethod
    def on_neutral_canvas(*types):
        """
        Whether wires of the given types only border regions with no colour,
        i.e. the neutral canvas, so that their strokes and labels may adapt
        to a dark page rather than keep a static colour readable over their
        region.
        """
        for typ in types:
            colours = [getattr(typ, "dom", None), getattr(typ, "cod", None)]
            colours += [
                colour for obj in getattr(typ, "inside", ())
                for colour in (
                    getattr(obj, "dom", None), getattr(obj, "cod", None))]
            if any(colour is not None for colour in colours):
                return False
        return True

    def draw_filled_shape(self, start, steps, color):
        """
        Fills the closed region whose boundary starts at ``start`` and follows
        ``steps``, a list of either ``("line", end)`` for a straight segment or
        ``("curve", control1, control2, end)`` for a cubic Bezier. The region
        is filled with ``color`` and drawn without an outline, e.g. behind the
        wires to colour the inside of a ribbon.
        """
        # pylint: disable=unused-argument  # the base only measures the width
        points = [start] + [step[-1] for step in steps]
        self.max_width = max([self.max_width] + [x for x, _ in points])

    @staticmethod
    def braid_strand(source, target, middle):
        """
        The four control points of the cubic Bezier drawn by
        :meth:`draw_braid_strand`, i.e. a strand from ``source`` to ``target``
        crossing the horizontal line at height ``middle``.
        """
        return [
            source, Point(source[0], middle), Point(target[0], middle), target]

    @staticmethod
    def half_circle_beziers(left, right, centre, sign, depth=None):
        """
        The two Bezier control groups of the arc of a half circle (or ellipse
        if ``depth`` differs from the radius) from ``(left, centre)`` to
        ``(right, centre)``, bulging to ``centre + sign * depth`` at the apex,
        as drawn by :meth:`draw_half_circle`.
        """
        middle, radius = (left + right) / 2, (right - left) / 2
        depth = radius if depth is None else depth
        kx, ky = (d * 4 * (sqrt(2) - 1) / 3 for d in (radius, depth))
        apex = centre + sign * depth
        return ([(left, centre), (left, centre + sign * ky),
                 (middle - kx, apex), (middle, apex)],
                [(middle, apex), (middle + kx, apex),
                 (right, centre + sign * ky), (right, centre)])

    def draw_half_circle(self, left, right, end, centre, sign, depth=None,
                         adaptive=True):
        """
        Draws a half circle (or ellipse if ``depth`` differs from the radius)
        from ``(left, centre)`` to ``(right, centre)`` with vertical sides up
        to ``end``, e.g. the fold of a :class:`discopy.ribbon.DualRailCup`,
        see :meth:`half_circle_beziers`.
        """
        if end != centre:
            self.draw_wire((left, end), (left, centre), adaptive=adaptive)
            self.draw_wire((right, centre), (right, end), adaptive=adaptive)
        for points in self.half_circle_beziers(
                left, right, centre, sign, depth):
            self.draw_bezier(points, adaptive=adaptive)

    @staticmethod
    def fold_depths(xs):
        """
        The vertical depth of the outer and inner fold of a dual rail cup or
        cap with rails at the four positions ``xs``, capping the half circle
        at :data:`~discopy.config.RIBBON_FOLD_DEPTH` so a wide cup flattens
        into an ellipse. The inner fold is one ribbon width shallower,
        keeping the width of the band constant.
        """
        radius, ribbon = (xs[3][0] - xs[0][0]) / 2, xs[1][0] - xs[0][0]
        outer = min(radius, RIBBON_FOLD_DEPTH)
        return outer, outer - ribbon

    def fill_strand_band(
            self, first: tuple[Point, Point], second: tuple[Point, Point],
            middle, color, gap=0):
        """
        Fills the band between the two parallel braid strands of a ribbon,
        see :meth:`draw_dual_rail_braid`. A non-zero ``gap`` breaks the band
        around the crossing, matching the broken strands, so the ribbon going
        under is shadowed by the one going over.
        """
        a, b = (self.braid_strand(*first, middle),
                self.braid_strand(*second, middle))
        spans = [(0, 1)] if not gap else [(0, 0.5 - gap), (0.5 + gap, 1)]
        for t0, t1 in spans:
            a_sub, b_sub = (
                bezier_subcurve(a, t0, t1), bezier_subcurve(b, t0, t1))
            self.draw_filled_shape(a_sub[0], [
                ("curve", a_sub[1], a_sub[2], a_sub[3]), ("line", b_sub[3]),
                ("curve", b_sub[2], b_sub[1], b_sub[0])], color)

    def draw_braid_strand(self, source, target, middle, gap=0, adaptive=True):
        """
        Draws a single strand of a braid crossing the horizontal line at
        height ``middle``. The strand is vertical at both ends and diagonal in
        between, so that two strands cross at a right angle rather than meeting
        flat. If ``gap`` is non-zero the strand is broken around the crossing,
        i.e. it goes under the other strand.
        """
        control = self.braid_strand(source, target, middle)
        if not gap:
            return self.draw_bezier(control, adaptive=adaptive)
        self.draw_bezier(bezier_subcurve(control, 0, 0.5 - gap),
                         adaptive=adaptive)
        self.draw_bezier(bezier_subcurve(control, 0.5 + gap, 1),
                         adaptive=adaptive)

    def draw_spiders(self, graph, draw_box_labels=True, **params):
        """ Draws a list of boxes depicted as spiders. """
        # pylint: disable=unused-argument  # the base only measures the width
        spider_widths = [
            p.x for n, p in graph.positions.items()
            if n.kind == 'box' and n.box.draw_as_spider]
        if spider_widths:
            self.max_width = max(self.max_width, max(spider_widths))

    @abstractmethod
    def output(self, path=None, show=True, **params):
        """ Output the drawing. """

    def draw_boundary(self, graph, boundary_color="white", **params):
        # pylint: disable=unused-argument  # params are a backend's
        x, y = graph.width, graph.height
        self.draw_polygon(
            (0, 0), (x, 0), (x, y), (0, y),
            facecolor=boundary_color, edgecolor=boundary_color)

    def draw_drawing(self, graph, **params):
        """ Draw the regions, wires, boxes and spiders of a drawing. """
        self.draw_regions(graph, **params)
        self.draw_wires(graph, **params)
        self.draw_boxes(graph, **params)
        self.draw_spiders(graph, **params)

    def draw_regions(self, graph, **params):
        """
        Draw coloured 0-cell regions, e.g. the background of a
        :meth:`Drawing.frame`, the wire-typed regions of a diagram or the
        inside of a ribbon, one :meth:`draw_region_cell` for each cell of
        :meth:`region_cells`.
        """
        # pylint: disable=unused-argument  # draw passes params to all
        for left, right, colour in self.region_cells(graph):
            self.draw_region_cell(left, right, colour)

    def draw_legend(self, graph, **params):
        """Draw a legend of region colours when supported by the backend."""

    @staticmethod
    def region_colours(graph):
        """
        The distinct painted region colours of a diagram, keyed by colour.

        Returns an order-preserving mapping from each colour's name to its
        :class:`monoidal.Colour`, suitable for a drawing legend. A region
        with no colour is omitted as it is the neutral background.
        """
        colours = {}
        types = [graph.dom, graph.cod]
        for box in graph.boxes:
            types += [box.dom, box.cod]
            for nested in getattr(box, "nested", None) or ():
                for name, colour in Backend.region_colours(nested).items():
                    colours.setdefault(name, colour)
        for typ in types:
            candidates = [getattr(typ, "dom", None), getattr(typ, "cod", None)]
            for obj in getattr(typ, "inside", ()):
                candidates += [
                    getattr(obj, "dom", None), getattr(obj, "cod", None)]
            for colour in candidates:
                if colour is not None:
                    colours.setdefault(colour.name, colour)
        return colours

    @staticmethod
    def region_separators(graph):
        """
        The boundaries between the regions of a drawing: a quadratic Bezier
        ``(top, control, bottom)`` for each non-horizontal visible edge and
        for both sides of each box, and a cubic one for each strand of a
        crossing, paired with the name of the colour of the region to its
        right. The control points are those of :meth:`draw_wire` and of
        :meth:`braid_strand`, so a separator hugs the wire drawn over it:
        the edges of a crossing bend at its box node where its strands go
        straight from each input to the output it is permuted to, so the
        strands stand in for them, see :meth:`strand_separators`.
        Horizontal edges bound no region on either side, so they are left
        out, and the left side of a box carries ``None`` since a box is a
        2-cell, i.e. there is no region underneath it.
        """
        separators = []
        for source, target in Backend.visible_edges(graph):
            top, bottom = graph.positions[source], graph.positions[target]
            if top.y == bottom.y or any(
                    n.kind == "box" and n.box.is_crossing
                    for n in (source, target)):
                continue
            control = Point(bottom.x, top.y) if source.kind == "box"\
                else Point(top.x, bottom.y)
            if top.y < bottom.y:
                top, bottom = bottom, top
            typ = getattr(source, 'x', None) or getattr(target, 'x', None)
            separators.append(
                ((top, control, bottom), getattr(typ.cod, "name", None)))
        for node in graph.box_nodes:
            if node.box.is_crossing:
                separators += Backend.strand_separators(graph, node)
            if node.box.draw_as_wires or node.box.draw_as_spider:
                continue
            for side, colour in [
                    (0, None), (1, getattr(node.box.dom.cod, "name", None))]:
                top = graph.positions[Node(f"box-corner-{side}1", j=node.j)]
                bottom = graph.positions[Node(f"box-corner-{side}0", j=node.j)]
                separators.append(
                    ((top, Point(top.x, bottom.y), bottom), colour))
        return separators

    @staticmethod
    def strand_separators(graph, node):
        """
        The separators of a crossing: the cubic Bezier of each of its
        strands, see :meth:`braid_strand` and :meth:`crossing_wires`, paired
        with the colour to the right of its wire.
        """
        middle = graph.positions[node].y
        return [
            (tuple(Backend.braid_strand(
                graph.positions[dom], graph.positions[cod], middle)),
             getattr(dom.x.cod, "name", None))
            for dom, cod in Backend.crossing_wires(node)]

    @staticmethod
    def separator_param(curve, y):
        """
        The Bezier parameter at which a separator ``curve`` from top to
        bottom crosses the horizontal line at height ``y``, see
        :meth:`region_separators`. The control point of a quadratic
        separator is level with one of its endpoints, so its height is
        monotone and the parameter has a closed form. The two control
        points of a strand are level with each other in between its
        endpoints, so its height is monotone too, a cubic inverted with
        :func:`monotone_root`.
        """
        if len(curve) == 4:
            return monotone_root(lambda t: bezier(curve, t).y - y)
        top, control, bottom = (point.y for point in curve)
        if control == top:
            return sqrt(max(0., min(1., (top - y) / (top - bottom))))
        return 1 - sqrt(max(0., min(1., (y - bottom) / (top - bottom))))

    @staticmethod
    def region_band(separators, bottom, top):
        """
        The separators that span the height band from ``bottom`` to ``top``,
        restricted to it and ordered from left to right, see
        :meth:`region_cells`. Two separators may touch at a shared endpoint
        on the boundary of the band but never cross inside it, so they are
        ordered by their horizontal position at its middle height.
        """
        band = []
        for curve, colour in separators:
            if curve[0].y < top or curve[-1].y > bottom:
                continue
            first, last, middle = (
                Backend.separator_param(curve, y)
                for y in (top, bottom, (top + bottom) / 2))
            restricted = subcurve(curve, first, last)
            band.append((
                subcurve(curve, middle, middle)[0].x,
                (Point(restricted[0].x, top), *restricted[1:-1],
                 Point(restricted[-1].x, bottom)), colour))
        return [(curve, colour)
                for _, curve, colour in sorted(band, key=lambda x: x[0])]

    @staticmethod
    def region_cells(graph):
        """
        The exact extents of the coloured regions of a drawing, computed by
        trapezoidal decomposition: the canvas is subdivided at every height
        where a separator starts or ends or two strands of a crossing meet,
        see :meth:`crossing_height`, and, inside each of the resulting
        bands, each consecutive pair of separators bounds one cell, filled
        with the colour that its left boundary carries -- ``graph.dom.dom``
        for the leftmost cell, with the sides of the canvas as outermost
        boundaries. Cells with no colour are left out as they are the neutral
        background, see :meth:`region_colours`, and so are the cells
        underneath a box, whose left side carries no colour at all.

        Returns the list of cells ``(left, right, colour)`` where ``left``
        and ``right`` are quadratic Beziers ``(top, control, bottom)`` or
        cubic strands restricted to the band, ordered bottom to top then
        left to right.
        """
        separators = Backend.region_separators(graph)
        strands = [curve for curve, _ in separators if len(curve) == 4]
        heights = sorted(set(
            [0, graph.height]
            + [point.y for curve, _ in separators
               for point in (curve[0], curve[-1])]
            + [Backend.crossing_height(first, second)
               for first, second in combinations(strands, 2)
               if Backend.strands_cross(first, second)]))
        cells = []
        for bottom, top in zip(heights, heights[1:]):
            left, colour = (
                Point(0, top), Point(0, bottom), Point(0, bottom)
            ), getattr(graph.dom.dom, "name", None)
            for right, next_colour in Backend.region_band(
                    separators, bottom, top):
                if right != left:
                    cells.append((left, right, colour))
                left, colour = right, next_colour
            cells.append((left, (
                Point(graph.width, top), Point(graph.width, bottom),
                Point(graph.width, bottom)), colour))
        return [cell for cell in cells if cell[-1] is not None]

    @staticmethod
    def strands_cross(first, second):
        """
        Whether two strands meet, i.e. they belong to one crossing, so that
        they are level with each other at every parameter, and swap sides
        between their top and their bottom.
        """
        return [point.y for point in first] == [point.y for point in second]\
            and (first[0].x - second[0].x) * (first[-1].x - second[-1].x) < 0

    @staticmethod
    def crossing_height(first, second):
        """
        The height at which two strands that :meth:`strands_cross` meet:
        their horizontal distance is monotone in the parameter they share,
        see :meth:`braid_strand`, so it vanishes at its :func:`monotone_root`.
        """
        param = monotone_root(
            lambda t: bezier(first, t).x - bezier(second, t).x)
        return bezier(first, param).y

    def draw_region_cell(self, left, right, facecolor):
        """
        Fill the cell between two Beziers ``left`` and ``right``, quadratic
        ``(top, control, bottom)`` triples or cubic strands, spanning the
        same height band, see :meth:`region_cells`.
        """
        # pylint: disable=unused-argument  # the base only measures the width
        self.max_width = max(
            self.max_width, max(x for x, _ in left + right))

    @staticmethod
    def cubic(curve):
        """
        The four control points of a Bezier given by three or four, i.e. a
        quadratic Bezier raised to the cubic one that draws the same curve.

        >>> start, control, end = Point(0, 3), Point(0, 0), Point(3, 0)
        >>> [tuple(p) for p in Backend.cubic([start, control, end])]
        [(0, 3), (0.0, 1.0), (1.0, 0.0), (3, 0)]
        """
        if len(curve) == 4:
            return list(curve)
        start, control, end = curve
        return [start] + [
            Point(point.x + 2 / 3 * (control.x - point.x),
                  point.y + 2 / 3 * (control.y - point.y))
            for point in (start, end)] + [end]

    @classmethod
    def region_cell_steps(cls, left, right):
        """
        The boundary of a cell of :meth:`region_cells` as the ``steps`` of
        :meth:`draw_filled_shape` from the top of ``left``: down ``left``,
        across the bottom, up ``right`` and back across the top.
        """
        left, right = cls.cubic(left), cls.cubic(right)
        return [("curve", *left[1:]), ("line", right[3]),
                ("curve", right[2], right[1], right[0]), ("line", left[0])]

    @staticmethod
    def visible_edges(graph):
        """ Yield the edges of a graph that are not inside a box. """
        def inside_a_box(node):
            return node.kind == "box"\
                and not node.box.draw_as_wires\
                and not node.box.draw_as_spider
        for source, target in graph.edges():
            if inside_a_box(source) or inside_a_box(target):
                continue  # no need to draw wires inside a box
            yield source, target

    def draw_wire_label(self, x, i, j, **params):
        """
        Draw the label of a wire, in a colour readable over the region to
        its right, which :meth:`draw_regions` paints underneath it. A
        region with no colour is the page, taken to be white and adapted to
        when it is dark, see :meth:`Matplotlib.dark_gid`.
        """
        draw_label_anyway = params.get('draw_box_labels', True) and getattr(
            x, "always_draw_label", False)
        if not params.get('wire_labels', True) and not draw_label_anyway:
            return
        if getattr(x.inside[0], "skip_label", False):
            return  # A wire at a fixed point of a permutation.
        if hasattr(x.inside[0], "reposition_label"):
            j += 0.25  # The label of e.g. cups, caps and swaps.
        label = str(x.inside[0])
        pad_i, pad_j = params.get('textpad', DEFAULT['textpad'])
        i += pad_i
        j -= pad_j
        fontsize = params.get('fontsize_types', params.get('fontsize', None))
        background = self.label_background(x)
        adaptive = background is None
        color = self.readable_foreground(
            "white" if adaptive else background.name)
        self.draw_text(
            label, i, j, verticalalignment='top', fontsize=fontsize,
            color=color, adaptive=adaptive)

    @staticmethod
    def label_background(typ):
        """ The region a wire label is drawn on, the one to its right. """
        return getattr(typ, "cod", None)

    @staticmethod
    def has_boundary_sides(typ):
        """ Whether the first or last object of a type is a frame side, i.e.
        it carries the ``frame_boundary`` flag set by :meth:`Drawing.bubble`
        with ``draw_as_square``. """
        return any(getattr(obj, "frame_boundary", False)
                   for obj in typ.inside[:1] + typ.inside[-1:])

    @staticmethod
    def is_frame_boundary(node):
        """ Whether a node belongs to the sides of a frame, i.e. the box drawn
        around the terms of an :class:`Equation` with coloured boundaries.

        A box carrying the ``frame_boundary`` flag is a bubble opening or
        closing squashed to half height, whose horizontal boundary is drawn
        as a wire through the box node. That boundary is hidden only for a
        square frame, i.e. when the left and right side types carry the flag
        too; a plain bubble keeps its top and bottom boundaries visible. """
        box = getattr(node, "box", None)
        if box is not None and getattr(box, "frame_boundary", False):
            return Backend.has_boundary_sides(box.dom)\
                or Backend.has_boundary_sides(box.cod)
        typ = getattr(node, "x", None)
        return typ is not None and getattr(
            typ.inside[0], "frame_boundary", False)

    def draw_wires(self, graph, **params):
        # Braids, swaps and permutations are drawn as their own smooth curves.
        for node in graph.nodes:
            if node.kind != "box":
                continue
            if node.box.draw_as_permutation:
                self.draw_permutation(graph.positions, node)
            elif node.box.is_crossing:
                self.draw_braid(graph.positions, node)
        for source, target in self.visible_edges(graph):
            source_position = graph.positions[source]
            target_position = graph.positions[target]
            # The sides of a frame are drawn with zero width.
            is_frame_boundary = self.is_frame_boundary(source)\
                or self.is_frame_boundary(target)
            if source.kind in ["dom", "box_cod"]:
                self.draw_wire_label(source.x, *source_position, **params)
            if source_position == target_position:
                continue
            if any(n.kind == "box" and n.box.is_crossing
                   for n in (source, target)):
                continue  # crossings are drawn on their own
            bend_out, bend_in = source.kind == "box", target.kind == "box"
            typ = getattr(source, 'x', None) or getattr(target, 'x', None)
            self.draw_wire(
                source_position, target_position, bend_out, bend_in,
                linewidth=(0 if is_frame_boundary else None),
                adaptive=self.on_neutral_canvas(typ))

    def fill_fold(self, outer, inner, color):
        """
        Fills the fold of a ribbon between two concentric arcs given as pairs
        of Bezier control groups, see :meth:`half_circle_beziers` and e.g.
        :meth:`draw_dual_rail_cup`.
        """
        self.draw_filled_shape(outer[0][0], [
            ("curve", *outer[0][1:]), ("curve", *outer[1][1:]),
            ("line", inner[1][3]),
            ("curve", inner[1][2], inner[1][1], inner[1][0]),
            ("curve", inner[0][2], inner[0][1], inner[0][0])], color)

    def draw_dual_rail_cup(self, positions, node, **params):
        """
        Draws a :class:`discopy.ribbon.DualRailCup` as a single constant-width
        fold, i.e. two concentric half circles joining the outer and inner
        rails of two ribbons, filled with the colour of their region. A wide
        cup is flattened into a half ellipse, see :meth:`fold_depths`.
        """
        # pylint: disable=unused-argument  # draw_boxes passes params to all
        box, j = node.box, node.j
        xs = [positions[Node("box_dom", i=i, j=j, x=box.dom[i])]
              for i in range(4)]
        end, ribbon = xs[0][1], box.dom.inside[0].ribbon
        outer_depth, inner_depth = self.fold_depths(xs)
        if ribbon is not None:  # Fill between the outer and the inner arc.
            self.fill_fold(
                self.half_circle_beziers(
                    xs[0][0], xs[3][0], end, -1, outer_depth),
                self.half_circle_beziers(
                    xs[1][0], xs[2][0], end, -1, inner_depth),
                ribbon.name)
        for (a, b), depth in [((0, 3), outer_depth), ((1, 2), inner_depth)]:
            self.draw_half_circle(xs[a][0], xs[b][0], end, end, -1, depth,
                                  adaptive=ribbon is None)

    def draw_dual_rail_cap(self, positions, node, **params):
        """
        Draws a :class:`discopy.ribbon.DualRailCap` as a single constant-width
        fold, i.e. two concentric half circles joining the outer and inner
        rails of two ribbons, filled with the colour of their region. A wide
        cap is flattened into a half ellipse, see :meth:`fold_depths`.
        """
        # pylint: disable=unused-argument  # draw_boxes passes params to all
        box, j = node.box, node.j
        xs = [positions[Node("box_cod", i=i, j=j, x=box.cod[i])]
              for i in range(4)]
        end, ribbon = xs[0][1], box.cod.inside[0].ribbon
        outer_depth, inner_depth = self.fold_depths(xs)
        if ribbon is not None:  # Fill between the outer and the inner arc.
            self.fill_fold(
                self.half_circle_beziers(
                    xs[0][0], xs[3][0], end, 1, outer_depth),
                self.half_circle_beziers(
                    xs[1][0], xs[2][0], end, 1, inner_depth),
                ribbon.name)
        for (a, b), depth in [((0, 3), outer_depth), ((1, 2), inner_depth)]:
            self.draw_half_circle(xs[a][0], xs[b][0], end, end, 1, depth,
                                  adaptive=ribbon is None)

    @staticmethod
    def crossing_wires(node):
        """
        The ``box_dom`` and ``box_cod`` nodes joined by each strand of a
        crossing, in the order of its outputs: a swap or a braid exchanges
        its two wires and a permutation sends input
        ``permutation_indices[i]`` to output ``i``, see :meth:`draw_braid`
        and :meth:`draw_permutation`.
        """
        box, j = node.box, node.j
        indices = box.permutation_indices\
            if box.draw_as_permutation else (1, 0)
        return [(Node("box_dom", i=source, j=j, x=box.dom[source]),
                 Node("box_cod", i=i, j=j, x=box.cod[i]))
                for i, source in enumerate(indices)]

    @staticmethod
    def dual_rail_wires(node):
        """
        The nodes joined by each rail of a dual rail box, as
        :meth:`crossing_wires` gives the strands of a crossing: a
        :class:`discopy.balanced.DualRailBraid` sends the ribbon of inputs
        ``(0, 1)`` to outputs ``(2, 3)`` and vice-versa, a
        :class:`discopy.balanced.DualRailTwist` sends each rail back to its
        own position, a :class:`discopy.ribbon.DualRailCup` (resp.
        :class:`discopy.ribbon.DualRailCap`) folds input (resp. output) ``0``
        onto ``3`` and ``1`` onto ``2``, see :meth:`draw_dual_rail_braid`,
        :meth:`draw_dual_rail_twist`, :meth:`draw_dual_rail_cup` and
        :meth:`draw_dual_rail_cap`.
        """
        box, j = node.box, node.j
        dom, cod = ([Node(kind, i=i, j=j, x=x) for i, x in enumerate(typ)]
                    for kind, typ in [("box_dom", box.dom),
                                      ("box_cod", box.cod)])
        if getattr(box, "draw_as_dual_rail_braid", False):
            return [(dom[i], cod[(i + 2) % 4]) for i in range(4)]
        if getattr(box, "draw_as_dual_rail_twist", False):
            return list(zip(dom, cod))
        side = dom if getattr(box, "draw_as_dual_rail_cup", False) else cod
        return [(side[0], side[3]), (side[1], side[2])]

    def draw_braid(self, positions, node):
        """
        Draws a braid or a swap as its two wires crossing diagonally, so that
        they meet at a right angle. A braid (over/under) breaks the wire that
        goes under; a symmetric swap simply crosses both wires.
        """
        box = node.box
        right, left = [(positions[dom], positions[cod])
                       for dom, cod in self.crossing_wires(node)]
        _, middle = positions[node]
        adaptive = self.on_neutral_canvas(box.dom, box.cod)
        if not getattr(box, "draw_as_braid", False):
            self.draw_braid_strand(*left, middle, adaptive=adaptive)
            self.draw_braid_strand(*right, middle, adaptive=adaptive)
            return
        # Keep the shadow roughly the same height, e.g. for a double braid,
        # by widening the (relative) gap when the braid is short.
        gap = min(0.3, 0.1 / (left[0][1] - left[1][1]))
        # The left wire goes under the right one unless the box is dagger.
        over, under = (left, right) if box.is_dagger else (right, left)
        self.draw_braid_strand(*under, middle, gap=gap, adaptive=adaptive)
        self.draw_braid_strand(*over, middle, adaptive=adaptive)

    def draw_permutation(self, positions, node):
        """ Draw a permutation as a band of crossing wires. """
        middle = positions[node][1]
        adaptive = self.on_neutral_canvas(node.box.dom, node.box.cod)
        for dom, cod in self.crossing_wires(node):
            self.draw_braid_strand(
                positions[dom], positions[cod], middle, adaptive=adaptive)

    def draw_boxes(self, graph, **params):
        drawing_methods = [
            ("draw_as_brakets", "draw_brakets"),
            ("draw_as_controlled", "draw_controlled_gate"),
            ("draw_as_discards", "draw_discard"),
            ("draw_as_measures", "draw_measure"),
            ("draw_as_dual_rail_braid", "draw_dual_rail_braid"),
            ("draw_as_dual_rail_twist", "draw_dual_rail_twist"),
            ("draw_as_dual_rail_cup", "draw_dual_rail_cup"),
            ("draw_as_dual_rail_cap", "draw_dual_rail_cap"),
            (None, "draw_box")]
        box_nodes = [node for node in graph.nodes if node.kind == "box"]
        for node in box_nodes:
            if node.box.draw_as_spider or node.box.draw_as_wires:
                continue
            for attribute, method in drawing_methods:
                if attribute is None or getattr(node.box, attribute, False):
                    getattr(self, method)(graph.positions, node, **params)
                    break

    def draw_box(self, positions, node, **params):
        """ Draws a box node on a given backend. """
        box, j = node.box, node.j
        asymmetry = params.get('asymmetry', 0)
        points = [positions[Node(f"box-corner-{c}", j=j)]
                  for c in ["00", "01", "11", "10"]]
        i = (0 if box.is_conjugate else
             1 if box.is_transpose else
             2 if box.is_dagger else 3)
        if box.is_conjugate or box.is_transpose:
            asymmetry *= -1
        points[i] = points[i].shift(x=asymmetry)
        if box.shape == "triangle":
            points = self.triangle_corners(positions, node)
        self.draw_polygon(*points, facecolor=box.color)
        label = positions[node]
        if box.nested is not None:
            label = self.draw_nested(box, points, label, **params)
        if params.get('draw_box_labels', True):
            adaptive = self.is_transparent(box.color)
            self.draw_text(box.drawing_name, *label,
                           ha='center', va='center',
                           color=self.readable_foreground(
                               "white" if adaptive else box.color),
                           fontsize=params.get('fontsize', None),
                           adaptive=adaptive)

    def draw_nested(self, box, corners, centre, **params):
        """
        Draw the drawings nested in a box left to right, one under the other,
        on a background of the box's ``frame_colour``, see
        :class:`Transposed`, centred in the space right of the box's label,
        and return the centre of that label, on the left of the box. Their
        wires plug into the edges of the background, a line separates each
        from the next, so that no two regions meet without one, and a
        daggered box reflects them, the last on top.
        """
        nested = box.nested
        width = sum(drawing.width for drawing in nested)
        height = max(drawing.height for drawing in nested)
        left, right = min(x for x, _ in corners), max(x for x, _ in corners)
        start = left + (
            text_width(box.drawing_name) + .25 if box.drawing_name else 0)
        x, y = (start + right - height) / 2, centre[1] - width / 2
        self.draw_polygon((x, y), (x + height, y), (x + height, y + width),
                          (x, y + width), facecolor=box.frame_colour)
        offset = 0
        for drawing in nested:
            offset += drawing.width
            origin = Point(x, y + offset - drawing.width if box.is_dagger
                           else y + width - offset)
            Transposed(self, origin, drawing, flip=box.is_dagger)\
                .draw_drawing(drawing, **params)
            if offset < width:
                edge = y + offset if box.is_dagger else y + width - offset
                self.draw_bezier([Point(x + t * height / 3, edge)
                                  for t in range(4)], adaptive=False)
        return Point((left + start) / 2, centre[1])

    @staticmethod
    def triangle_corners(positions, node):
        """
        The corners of a box drawn as a triangle, i.e. with ``shape``
        ``"triangle"``: its base the width of the side with more wires, its
        apex on the other side, at the wire when there is one. With two or
        more wires on both sides, the apex is cut to the span of its wires.

        The apex points down when the codomain has fewer wires, or as many
        and the box is flipped upside down by neither a dagger nor a half
        turn, i.e. a transpose, so that both flip it and a conjugate, their
        composite, does not.

        Example
        -------
        >>> from discopy.monoidal import Ty, Box
        >>> x = Ty('x')
        >>> merge = Box('t', x ** 2, x, shape="triangle").to_drawing()
        >>> merge.add_box_corners()
        >>> node, = (n for n in merge.positions if n.kind == "box")
        >>> Backend.triangle_corners(merge.positions, node)
        [Point(x=0.25, y=0.75), Point(x=1.75, y=0.75), Point(x=1.0, y=0.25)]

        >>> t = Box('t', x ** 3, x, shape="triangle")
        >>> (t >> t[::-1]).draw(doctest='docs/_static/drawing/triangle.svg')

        .. image:: /_static/drawing/triangle.svg
            :align: center
        """
        box, j = node.box, node.j
        (left, bottom), (right, top) = (
            positions[Node(f"box-corner-{c}", j=j)] for c in ["00", "11"])
        flipped = (
            box.is_dagger + bool(box.is_transpose) + box.is_conjugate) % 2
        down = len(box.cod) < len(box.dom) or (
            len(box.cod) == len(box.dom) and not flipped)
        side, base, apex = ("box_cod", top, bottom) if down\
            else ("box_dom", bottom, top)
        xs = [positions[Node(side, i=i, j=j, x=x)][0]
              for i, x in enumerate(box.cod if down else box.dom)]
        tips = [min(xs), max(xs)] if xs else [positions[node][0]]
        tips = tips[:1] if tips[0] == tips[-1] else tips
        return [Point(left, base), Point(right, base)] + [
            Point(x, apex) for x in reversed(tips)]

    def draw_discard(self, positions, node, **params):
        """ Draws a :class:`discopy.quantum.circuit.Discard` box. """
        # pylint: disable=unused-argument  # draw_boxes passes params to all
        box, j = node.box, node.j
        for i in range(len(box.dom)):
            x = box.dom[i]
            wire = Node("box_dom", x=x, j=j, i=i)
            middle = positions[wire]
            left, right = middle[0] - .25, middle[0] + .25
            height = positions[node][1] + .25
            for k in range(3):
                source = (left + .1 * k, height - .1 * k)
                target = (right - .1 * k, height - .1 * k)
                self.draw_wire(source, target)

    def draw_measure(self, positions, node, **params):
        """ Draws a :class:`discopy.quantum.circuit.Measure` box. """
        self.draw_box(positions, node, **dict(params, draw_box_labels=False))
        i, j = positions[node]
        self.draw_wire(
            (i - .15, j - .1), (i, j + .1), bend_in=True, adaptive=False)
        self.draw_wire(
            (i, j + .1), (i + .15, j - .1), bend_out=True, adaptive=False)
        self.draw_wire(
            (i, j - .1), (i + .05, j + .15), style='->', adaptive=False)

    def draw_dual_rail_braid(self, positions, node, **params):
        """
        Draws a :class:`discopy.balanced.DualRailBraid`, i.e. the two ribbons
        ``(0, 1)`` and ``(2, 3)`` crossing as a whole rather than wire by wire.
        """
        # pylint: disable=unused-argument  # draw_boxes passes params to all
        box = node.box
        rails = [(positions[dom], positions[cod])
                 for dom, cod in self.dual_rail_wires(node)]
        _, y_middle = positions[node]
        # The left ribbon goes to the right and vice-versa. As with a braid,
        # the left ribbon goes under the right one unless the box is dagger.
        left = (rails[:2], getattr(box.dom.inside[0].ribbon, "name", None))
        right = (rails[2:], getattr(box.dom.inside[2].ribbon, "name", None))
        over, under = (left, right) if box.is_dagger else (right, left)
        # Fill (and stroke) under first, then over: the band going under is
        # broken around the crossing so the one going over is clearly on top.
        for (ribbon, color), gap in [(under, 0.2), (over, 0)]:
            if color is not None:  # Fill the band between the two strands.
                self.fill_strand_band(
                    ribbon[0], ribbon[1], y_middle, color, gap)
        for (ribbon, color), gap in [(under, 0.2), (over, 0)]:
            for source, target in ribbon:
                self.draw_braid_strand(source, target, y_middle, gap,
                                       adaptive=color is None)

    def draw_dual_rail_twist(self, positions, node, **params):
        """
        Draws a :class:`discopy.balanced.DualRailTwist`, i.e. the two rails of
        a ribbon crossing each other twice in quick succession.
        """
        # pylint: disable=unused-argument  # draw_boxes passes params to all
        box, j = node.box, node.j
        dom = [positions[Node("box_dom", i=i, j=j, x=box.dom[i])]
               for i in range(2)]
        cod = [positions[Node("box_cod", i=i, j=j, x=box.cod[i])]
               for i in range(2)]
        _, middle = positions[node]
        # The rails swap at the middle then swap back, i.e. they twist.
        swap = [(dom[1][0], middle), (dom[0][0], middle)]
        upper, lower = (dom[0][1] + middle) / 2, (middle + cod[0][1]) / 2
        color = getattr(box.dom.inside[0].ribbon, "name", None)
        if color is not None:
            top0, bottom0 = (
                self.braid_strand(dom[0], swap[0], upper),
                self.braid_strand(swap[0], cod[0], lower))
            top1, bottom1 = (
                self.braid_strand(dom[1], swap[1], upper),
                self.braid_strand(swap[1], cod[1], lower))
            # Each rail crosses the other at the midpoint (t=0.5) of `top`
            # and of `bottom`. The two crossings pinch off a lens where the
            # ribbon has turned over: fill it with a darker shade of the
            # front colour, for a nicer visual of the twist. A colour
            # matplotlib reads is darkened from matplotlib's RGB, in TikZ
            # too, where the front of a name like purple is xcolor's RGB.
            # Any other is an xcolor expression, mixed 60% with black, a
            # trailing `!<pct>` being a mix with white.
            value = self.resolve(color)
            mix = f"{color}!white" if str(color).count("!") % 2 else color
            back = color if self.is_transparent(color)\
                else f"{mix}!60!black" if not is_color_like(value)\
                else config.darken(to_hex(value, keep_alpha=True))
            front_top0, back_top0 = (
                bezier_subcurve(top0, 0, .5), bezier_subcurve(top0, .5, 1))
            back_bottom0, front_bottom0 = (
                bezier_subcurve(bottom0, 0, .5),
                bezier_subcurve(bottom0, .5, 1))
            front_top1, back_top1 = (
                bezier_subcurve(top1, 0, .5), bezier_subcurve(top1, .5, 1))
            back_bottom1, front_bottom1 = (
                bezier_subcurve(bottom1, 0, .5),
                bezier_subcurve(bottom1, .5, 1))
            self.draw_filled_shape(front_top0[0], [
                ("curve", *front_top0[1:]), ("line", front_top1[3]),
                ("curve", front_top1[2], front_top1[1], front_top1[0])],
                color)
            self.draw_filled_shape(back_top0[0], [
                ("curve", *back_top0[1:]), ("curve", *back_bottom0[1:]),
                ("line", back_bottom1[3]),
                ("curve", back_bottom1[2], back_bottom1[1], back_bottom1[0]),
                ("curve", back_top1[2], back_top1[1], back_top1[0])], back)
            self.draw_filled_shape(front_bottom0[0], [
                ("curve", *front_bottom0[1:]), ("line", front_bottom1[3]),
                ("curve", front_bottom1[2], front_bottom1[1],
                 front_bottom1[0])], color)
        crossings = [
            [(dom[0], swap[0], upper), (dom[1], swap[1], upper)],
            [(swap[0], cod[0], lower), (swap[1], cod[1], lower)]]
        for k, (first_rail, second_rail) in enumerate(crossings):
            # A twist is a braid followed by another of the same handedness
            # (not by its inverse), so the same diagonal stays on top: the
            # rails swap which one goes under between the two crossings.
            first_under = (k == 0) != box.is_dagger
            under, over = (first_rail, second_rail) if first_under\
                else (second_rail, first_rail)
            self.draw_braid_strand(under[0], under[1], under[2], gap=0.15,
                                   adaptive=color is None)
            self.draw_braid_strand(over[0], over[1], over[2],
                                   adaptive=color is None)

    def draw_brakets(self, positions, node, **params):
        """ Draws a :class:`discopy.quantum.gates.Ket` box. """
        box, j = node.box, node.j
        is_bra, adaptive = len(box.dom) > 0, self.is_transparent(box.color)
        for i, bit in enumerate(box._digits):
            kind = "box_dom" if is_bra else "box_cod"
            x = box.dom[i] if is_bra else box.cod[i]
            wire = Node(kind, x=x, j=j, i=i)
            middle = positions[wire]
            left = middle[0] - .25, middle[1]
            right = middle[0] + .25, middle[1]
            top = middle[0], middle[1] + .5
            bottom = middle[0], middle[1] - .5
            self.draw_polygon(
                left, right, bottom if is_bra else top, facecolor=box.color)
            self.draw_text(
                bit, middle[0], middle[1] + (-.25 if is_bra else .2),
                ha='center', va='center',
                color=self.readable_foreground(
                    "white" if adaptive else box.color),
                fontsize=params.get('fontsize', None), adaptive=adaptive)

    def draw_controlled_gate(self, positions, node, **params):
        """ Draws a :class:`discopy.quantum.gates.Controlled` gate. """
        box, j = node.box, node.j
        distance = box.distance
        c_size = len(box.controlled.dom)

        index = (0, distance) if distance > 0 else (c_size - distance - 1, 0)
        dom = Node("box_dom", x=box.dom[index[0]], i=index[0], j=j)
        cod = Node("box_cod", x=box.cod[index[0]], i=index[0], j=j)
        middle = positions[dom][0], (positions[dom][1] + positions[cod][1]) / 2
        controlled_box = box.controlled.to_drawing().box
        controlled = Node("box", box=controlled_box, j=j)
        c_dom = Node("box_dom", x=box.dom[index[1]], i=index[1], j=j)
        c_cod = Node("box_cod", x=box.cod[index[1]], i=index[1], j=j)
        c_middle = Point(
            positions[c_dom][0],
            (positions[c_dom][1] + positions[c_cod][1]) / 2)
        target = Point(
            positions[c_dom][0] + (c_size - 1) / 2,
            (positions[c_dom][1] + positions[c_cod][1]) / 2)
        target_boundary = target
        if controlled_box.name == "X":  # CX gets drawn as a circled plus sign.
            self.draw_wire(positions[c_dom], positions[c_cod])
            self.draw_node(
                *target,
                shape="circle", color="white", edgecolor="black",
                nodesize=2 * params.get("nodesize", 1))
            self.draw_node(
                *target, shape="plus",
                nodesize=2 * params.get("nodesize", 1))
        else:
            fake_positions = {controlled: target} | {
                Node(f"box-corner-{a}{b}", j=j): target.shift(x=x, y=y)
                for a, x in enumerate([-0.25, 0.25])
                for b, y in enumerate([-0.25, 0.25])}

            for i in range(c_size):
                dom_node = Node("box_dom", x=controlled_box.dom[i], i=i, j=j)
                x, y = positions[c_dom][0] + i, positions[c_dom][1]
                fake_positions[dom_node] = x, y

                cod_node = Node("box_cod", x=controlled_box.cod[i], i=i, j=j)
                x, y = positions[c_cod][0] + i, positions[c_cod][1]
                fake_positions[cod_node] = x, y

            shift_boundary = True
            if hasattr(box.controlled, "draw_as_controlled"):
                self.draw_controlled_gate(fake_positions, controlled, **params)

                next_box = box.controlled
                while hasattr(next_box, "controlled"):
                    if controlled_box.distance * next_box.distance < 0:
                        shift_boundary = False
                        break
                    next_box = next_box.controlled
                if next_box.name == "X":
                    shift_boundary = False
            else:
                self.draw_box(fake_positions, controlled, **params)

            if shift_boundary:
                if box.distance > 0:
                    target_boundary = c_middle[0] - .25, c_middle[1]
                else:
                    target_boundary = (
                        c_middle[0] + c_size - 1 + .25, c_middle[1])
            else:
                if box.distance > 0:
                    target_boundary = c_middle[0], c_middle[1]
                else:
                    target_boundary = c_middle[0] + c_size - 1, c_middle[1]
        self.draw_wire(positions[dom], positions[cod])

        # draw all the other vertical wires
        extra_offset = 1 if distance > 0 else len(box.controlled.dom)
        for i in range(extra_offset, extra_offset + abs(distance) - 1):
            node1 = Node("box_dom", x=box.dom[i], i=i, j=j)
            node2 = Node("box_cod", x=box.cod[i], i=i, j=j)
            self.draw_wire(positions[node1], positions[node2])

        self.draw_wire(middle, target_boundary, bend_in=True, bend_out=True)

        self.draw_node(
            *middle, color="black", shape="circle",
            nodesize=params.get("nodesize", 1), adaptive=True)


class Transposed(Backend):
    """
    A backend that draws left to right on another one, through the
    reflection along the anti-diagonal of a drawing :math:`w \\times h`

    .. math::

        (x, y) \\mapsto (h - y, w - x)

    which takes its domain to the left, its codomain to the right and its
    first wire to the top, then moves it to ``origin``. Wires, regions and
    boxes are Beziers and polygons, whose image under an affine map is the
    curve of the images of their control points, so the layout of the
    drawing is kept as it is and only its picture is transposed. Labels stay
    upright, and drawing on a transposed backend transposes back.

    Parameters:
        parent : The backend to draw on.
        origin : The bottom left corner of the transposed drawing.
        graph : The drawing, for its width and height.
        flip : Whether to reflect the transposed drawing along the
            horizontal axis too, e.g. inside a daggered box.

    Example
    -------
    >>> from discopy.monoidal import Ty, Box
    >>> x = Ty('x')
    >>> (Box('f', Ty(), x) >> Box('g', x, x @ x)).draw(
    ...     left_to_right=True,
    ...     doctest="docs/_static/drawing/left-to-right.svg")

    .. image:: /_static/drawing/left-to-right.svg
        :align: center
    """
    def __init__(self, parent: Backend, origin: Point, graph: Drawing,
                 flip=False):
        self.parent, self.origin, self.flip = parent, origin, flip
        self.textpad = DEFAULT["textpad"][0]
        self.width, self.height = graph.width, graph.height
        super().__init__()

    def map(self, point) -> Point:
        """ The image of a point of the drawing on the parent backend. """
        x, y = point
        return Point(self.origin.x + self.height - y,
                     self.origin.y + (x if self.flip else self.width - x))

    def draw_boundary(self, graph, *args, **params):
        self.parent.draw_boundary(
            SimpleNamespace(width=self.height, height=self.width),
            *args, **params)

    def label_background(self, typ):
        """ A label above its wire lies on the region right of the wire
        when the drawing is reflected, on the one left of it otherwise. """
        return getattr(typ, "cod" if self.flip else "dom", None)

    def draw_wire_label(self, x, i, j, **params):
        self.textpad = params.get("textpad", DEFAULT["textpad"])[0]
        super().draw_wire_label(x, i, j, **params)

    def draw_text(self, text, i, j, **params):
        point = self.map((i, j))
        if params.get("verticalalignment") == "top":
            # A wire label hangs below its point, a textpad off its wire:
            # lift it above the wire, the textpad off it the other way.
            point = point.shift(y=(
                params.get("fontsize") or DEFAULT["fontsize"]) / 72 + (
                    0 if self.flip else 2 * self.textpad))
        self.parent.draw_text(text, *point, **params)

    def draw_node(self, i, j, **params):
        self.parent.draw_node(*self.map((i, j)), **params)

    def draw_polygon(self, *points, **params):
        self.parent.draw_polygon(*map(self.map, points), **params)

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def draw_wire(self, source, target, bend_out=False, bend_in=False,
                  style=None, linewidth=None, adaptive=True):
        """ Draw the curve :class:`Matplotlib` draws, as a cubic Bezier,
        since other backends bend wires vertically. """
        if linewidth == 0:
            return
        if style is not None:  # e.g. the arrow of a measure, straight
            self.parent.draw_wire(
                self.map(source), self.map(target), style=style,
                linewidth=linewidth, adaptive=adaptive)
            return
        source, target = Point(*source), Point(*target)
        middle = Point(target.x, source.y) if bend_out\
            else Point(source.x, target.y)
        self.draw_bezier(self.cubic([source, middle, target]), adaptive)

    def draw_bezier(self, points, adaptive=True):
        self.parent.draw_bezier(list(map(self.map, points)), adaptive)

    def draw_filled_shape(self, start, steps, color):
        self.parent.draw_filled_shape(self.map(start), [
            (kind, *map(self.map, points)) for kind, *points in steps], color)

    def draw_region_cell(self, left, right, facecolor):
        self.parent.draw_region_cell(
            list(map(self.map, left)), list(map(self.map, right)), facecolor)

    def draw_spiders(self, graph, draw_box_labels=True, **params):
        self.parent.draw_spiders(graph.relabel_nodes(positions={
            node: self.map(point) for node, point in graph.positions.items()
        }), draw_box_labels, **params)

    def output(self, path=None, show=True, **params):
        return self.parent.output(path, show, **params)


class TikZ(Backend):
    """ Tikz drawing backend. """
    #: The size in points of the label of a box, and of a wire one size down.
    LABEL_SIZE, WIRE_LABEL_SIZE = 8, 7

    @classmethod
    def unit(cls) -> str:
        """
        The length of one unit of a drawing, at which the label of a box fits
        it: the layout measures a label at twelve points in inches, see
        :func:`discopy.utils.text_width`, and TikZ sets it at
        :attr:`LABEL_SIZE` points.

        >>> TikZ.unit()
        '0.6667in'
        """
        return f"{cls.LABEL_SIZE / 12:.4f}in"

    @staticmethod
    def font(size) -> str:
        """ The font option of a label of ``size`` points. """
        return f"font=\\fontsize{{{size}}}{{{1.2 * size:g}}}\\selectfont"

    def __init__(self, use_tikzstyles=None):
        self.use_tikzstyles = DEFAULT["use_tikzstyles"]\
            if use_tikzstyles is None else use_tikzstyles
        self.node_styles, self.edge_styles = [], []
        self.nodes, self.nodelayer, self.edgelayer = {}, [], []
        super().__init__()

    #: The colour names every xcolor knows, i.e. without package options.
    XCOLOR_NAMES = (
        "red", "green", "blue", "cyan", "magenta", "yellow", "black", "gray",
        "white", "darkgray", "lightgray", "brown", "lime", "olive", "orange",
        "pink", "purple", "teal", "violet")

    @classmethod
    def needs_rgb(cls, color):
        """
        Whether a color is one matplotlib reads and xcolor does not, e.g.
        ``"tab:blue"``, ``"C0"``, ``"0.5"``, ``"navy"`` or an RGB tuple, so
        that TikZ must spell it by its RGB values, or a hexcode.

        >>> assert TikZ.needs_rgb("tab:blue") and TikZ.needs_rgb((0, 0, 1))
        >>> assert not TikZ.needs_rgb("purple") and not TikZ.needs_rgb("none")
        """
        if cls.is_transparent(color) or isinstance(color, str) and (
                color in COLORS or color in cls.XCOLOR_NAMES):
            return False
        return is_color_like(color)

    @classmethod
    def format_color(cls, color):
        """
        Formats a color: a name in :data:`COLORS` or one that matplotlib
        reads and xcolor does not, e.g. a hexcode, as its RGB values, any
        other color as it is, e.g. transparency, which TikZ spells like
        matplotlib, or an xcolor name or expression.
        A TikZ color has no alpha channel, so that of a hexcode is dropped.

        >>> TikZ.format_color("red")
        '{rgb,255: red,232; green,165; blue,165}'
        >>> assert TikZ.format_color("#e8a5a5") == TikZ.format_color("red")
        >>> assert TikZ.format_color("#abc") == TikZ.format_color("#aabbcc80")
        >>> TikZ.format_color("tab:blue")
        '{rgb,255: red,31; green,119; blue,180}'
        >>> TikZ.format_color("lightgray")
        'lightgray'
        """
        is_named = isinstance(color, str) and color in COLORS
        if not is_named and not cls.needs_rgb(color):
            return color
        rgb = to_rgb(COLORS[color] if is_named else color)
        red, green, blue = (round(255 * x) for x in rgb)
        return f"{{rgb,255: red,{red}; green,{green}; blue,{blue}}}"

    @classmethod
    def style_color(cls, color):
        """
        The color in the name of a tikzstyle: as it is written, without the
        ``#`` of a hexcode, or the hexcode of a color xcolor does not read.

        >>> TikZ.style_color("#336699"), TikZ.style_color("tab:blue")
        ('336699', '1f77b4')
        """
        return to_hex(color)[1:] if cls.needs_rgb(color)\
            else str(color).removeprefix("#")

    #: What a label keeps as it is, an escaped character or the mathematics
    #: between two dollars, then what LaTeX reads as markup outside them.
    MARKUP = re.compile(
        r"(?P<kept>\\.|\$[^$]*\$)|(?P<char>[_&%#])"
        r"|(?P<pair>[<>])(?=(?P=pair))")

    @classmethod
    def escape(cls, text) -> str:
        """
        A label as LaTeX reads it: outside the mathematics between dollars,
        an underscore, an ampersand, a percent and a hash are escaped, and
        angle brackets are kept from joining into guillemets. A backslash
        with the character after it and braces are left, so that a label can
        still be LaTeX, with its own escapes.

        >>> print(TikZ.escape("feed_forward $x_1$ (n >> s) 50%"))
        feed\\_forward $x_1$ (n >{}> s) 50\\%
        >>> print(TikZ.escape(r"\\$5 for x_y\\$, a\\_b, <<<"))
        \\$5 for x\\_y\\$, a\\_b, <{}<{}<
        """
        return cls.MARKUP.sub(
            lambda match: match["kept"] or (
                "\\" + match["char"] if match["char"]
                else match["pair"] + "{}"), str(text))

    def add_node(self, i, j, text=None, options=None, rounded=4):
        """ Add a node to the tikz picture, return its unique id. """
        node = len(self.nodelayer) + 1
        text = "" if text is None else self.escape(text)
        self.nodelayer.append(
            f"\\node [{options or ''}] ({node}) at "
            f"({round(i, rounded)}, {round(j, rounded)}) {{{text}}};\n")
        self.nodes.update({(i, j): node})
        return node

    def draw_node(self, i, j, text=None, **params):
        options = []
        if 'shape' in params:
            options.append(params['shape'])
        if 'color' in params:
            options.append(params['color'])
        self.add_node(i, j, text, options=", ".join(options))
        super().draw_node(i, j, **params)

    def draw_text(self, text, i, j, **params):
        """
        A label as a node, in a font of :attr:`LABEL_SIZE` points and its
        ``color`` when it is not black. A wire label, aligned to its top,
        hangs below and to the right of its point in a font of
        :attr:`WIRE_LABEL_SIZE` points, filled white only over a region with
        no colour, see :meth:`draw_wire_label`.

        >>> tikz = TikZ()
        >>> tikz.draw_text("f", 0, 0)
        >>> tikz.draw_text("x", 0, 1, verticalalignment="top")
        >>> tikz.draw_text("y", 1, 1, verticalalignment="top", adaptive=False)
        >>> for node in tikz.nodelayer:
        ...     print(node.split("] ")[0].split("style=none, ")[1])
        fill=white, font=\\fontsize{8}{9.6}\\selectfont
        fill=white, anchor=north west, font=\\fontsize{7}{8.4}\\selectfont
        anchor=north west, font=\\fontsize{7}{8.4}\\selectfont
        """
        is_wire_label = params.get("verticalalignment", "center") == "top"
        options = "style=none" + (
            ", fill=white" if params.get("adaptive", True) else "")
        if params.get('horizontalalignment', 'center') == 'left':
            options += ", anchor=west"
        if is_wire_label:
            options += ", anchor=north west"
        options += ", " + self.font(
            self.WIRE_LABEL_SIZE if is_wire_label else self.LABEL_SIZE)
        color = params.get("color")
        if color is not None and not (
                isinstance(color, str) and color == "black"):
            options += f", text={self.color_option(color)}"
        if 'fontsize' in params and params['fontsize'] is not None:
            options += f", scale={params['fontsize']}"
        self.add_node(i, j, text, options)
        super().draw_text(text, i, j, **params)

    @classmethod
    def color_option(cls, color):
        """
        A color as the value of an option, in braces.

        >>> TikZ.color_option("white"), TikZ.color_option("#ff0000")
        ('{white}', '{rgb,255: red,255; green,0; blue,0}')
        """
        return cls.format_color(color) if cls.needs_rgb(color)\
            else f"{{{color}}}"

    def draw_polygon(
            self, *points,
            facecolor=DEFAULT["facecolor"], edgecolor=DEFAULT["edgecolor"]):
        """
        A polygon filled with ``facecolor`` and stroked with ``edgecolor``,
        which is written when it is not the default: the background of a
        drawing has the edge of its face, not a black frame.

        >>> tikz = TikZ(use_tikzstyles=False)
        >>> tikz.draw_polygon((0, 0), (1, 0), (0, 1), edgecolor="white")
        >>> print(tikz.edgelayer[0].split("]")[0] + "]")
        \\draw [-, fill={white}, draw={white}]
        """
        nodes = [self.add_node(*point) for point in points]
        nodes.append(nodes[0])
        colors = {"-, fill": facecolor} | ({} if isinstance(
            edgecolor, str) and edgecolor == DEFAULT["edgecolor"]
            else {"draw": edgecolor})
        options = ", ".join(
            f"{key}={self.color_option(color)}"
            for key, color in colors.items())
        if self.use_tikzstyles:
            name = "box" if isinstance(facecolor, str) and list(
                colors.values()) == [DEFAULT["facecolor"]] else "_".join(
                    list(map(self.style_color, colors.values())) + ["box"])
            style = f"\\tikzstyle{{{name}}}=[" + ", ".join(
                f"{key}={self.format_color(color)}"
                for key, color in colors.items()) + "]\n"
            if style not in self.edge_styles:
                self.edge_styles.append(style)
            options = f"style={name}"
        str_connections = " to ".join(f"({node}.center)" for node in nodes)
        self.edgelayer.append(f"\\draw [{options}] {str_connections};\n")
        super().draw_polygon(*points)

    def draw_region_cell(self, left, right, facecolor):
        """
        Fill the cell between two Beziers as one closed path, see
        :meth:`Backend.region_cell_steps`, stroked in its own colour so that
        two cells meeting edge to edge leave no seam where a viewer
        antialiases each of them.
        """
        self.draw_filled_shape(
            left[0], self.region_cell_steps(left, right), facecolor,
            stroke=True)
        super().draw_region_cell(left, right, facecolor)

    def draw_wire(self, source, target, bend_out=False, bend_in=False,
                  style=None, linewidth=None, adaptive=True):
        out = -90 if not bend_out or source[0] == target[0]\
            else (180 if source[0] > target[0] else 0)
        inp = 90 if not bend_in or source[0] == target[0]\
            else (180 if source[0] < target[0] else 0)
        looseness = 1
        if not (source[0] == target[0] or source[1] == target[1]):
            dx, dy = abs(source[0] - target[0]), abs(source[1] - target[1])
            length = sqrt(dx * dx + dy * dy)
            distance = min(dx, dy)
            looseness = round(distance / length * 2.1, 4)
        if looseness != 1:
            style = ', '.join(  # ty: ignore[no-matching-overload]
                [style] * bool(style) + [f'looseness={looseness}'])

        cmd = (
            "\\draw [in={}, out={}{}] "
            "({}.center) to ({}.center);\n")
        if source not in self.nodes:
            self.add_node(*source)
        if target not in self.nodes:
            self.add_node(*target)
        self.edgelayer.append(cmd.format(
            inp, out,
            f", {style}" if style is not None else "",
            self.nodes[source], self.nodes[target]))
        super().draw_wire(source, target, bend_out=bend_out, bend_in=bend_in)

    def draw_bezier(self, points, adaptive=True):
        for point in points:
            if tuple(point) not in self.nodes:
                self.add_node(*point)
        self.edgelayer.append(
            "\\draw ({}.center) .. controls ({}.center) and ({}.center) .. "
            "({}.center);\n".format(*(self.nodes[tuple(p)] for p in points)))
        super().draw_bezier(points)

    def draw_filled_shape(self, start, steps, color, stroke=False):
        """ The shape of :meth:`Backend.draw_filled_shape`, with a hairline
        in its own colour if ``stroke``, else none. """
        def node(point):
            if tuple(point) not in self.nodes:
                self.add_node(*point)
            return self.nodes[tuple(point)]
        path = f"({node(start)}.center)"
        for step in steps:
            if step[0] == "line":
                path += f" to ({node(step[1])}.center)"
            else:
                path += " .. controls ({}.center) and ({}.center) .. "\
                    "({}.center)".format(*(node(p) for p in step[1:]))
        outline = f"{self.format_color(color)}, line width=0.25pt"\
            if stroke else "none"
        self.edgelayer.append(
            f"\\draw [fill={self.format_color(color)}, draw={outline}] "
            f"{path} -- cycle;\n")
        super().draw_filled_shape(start, steps, color)

    @staticmethod
    def inner_sep(nodesize) -> str:
        """
        The padding of a spider of a given ``nodesize``, a share of the
        default ``0.3333em`` of a node, i.e. half the width of its disc.

        >>> TikZ.inner_sep(0.5)
        '0.1667em'
        """
        return f"{round(nodesize / 3, 4)}em"

    def draw_spiders(self, graph, draw_box_labels=True, **params):
        """
        Draws each spider as an empty node of its shape and colour, its size
        set by ``nodesize``, with its label over its center at the size of
        the text, as matplotlib does: a ``scale`` would shrink the label with
        the node, e.g. the symbol of an equation in a large drawing.

        >>> from discopy.monoidal import Ty, Box, Equation
        >>> x = Ty('x')
        >>> tikz = TikZ(use_tikzstyles=False)
        >>> drawing = Equation(Box('f', x, x), Box('g', x, x)).to_drawing()
        >>> tikz.draw_spiders(drawing, nodesize=0.5)
        >>> print(tikz.nodelayer[0].split("]")[0] + "]")
        \\node [circle, fill=none, inner sep=0.1667em, label={center:{=}}]
        """
        spiders = [(node, node.box.color, node.box.shape)
                   for node in graph.nodes
                   if node.kind == "box" and node.box.draw_as_spider]
        for node, color, shape in spiders:
            i, j = graph.positions[node]
            text = node.box.drawing_name if draw_box_labels else ""
            if self.use_tikzstyles:
                style_name = f"{node.box.tikzstyle_name}_"\
                    f"{self.style_color(color)}_{shape}"
                style = f"\\tikzstyle{{{style_name}}}=" \
                        f"[fill={self.format_color(color)}, shape={shape}]\n"
                if style not in self.node_styles:
                    self.node_styles.append(style)
                options = f"style={style_name}"
            else:
                options = f"{shape}, fill=" + (
                    self.format_color(color) if self.needs_rgb(color)
                    else color)
            if params.get("nodesize", 1) != 1:
                options += f", inner sep={self.inner_sep(params['nodesize'])}"
            if text:
                options += f", label={{center:{{{self.escape(text)}}}}}"
            self.add_node(i, j, None, options)
        super().draw_spiders(graph, draw_box_labels)

    def output(self, path=None, show=True, **params):
        baseline = params.get("baseline", 0)
        tikz_options = params.get("tikz_options", None)
        output_tikzstyle = self.use_tikzstyles\
            and params.get("output_tikzstyle", True)
        options = f"baseline=(0.base), x={self.unit()}, y={self.unit()}"\
            + ("" if tikz_options is None else ", " + tikz_options)
        begin = [f"\\begin{{tikzpicture}}[{options}]\n"]
        nodes = ["\\begin{pgfonlayer}{nodelayer}\n",
                 f"\\node (0) at (0, {baseline}) {{}};\n"]\
            + self.nodelayer + ["\\end{pgfonlayer}\n"]
        edges = ["\\begin{pgfonlayer}{edgelayer}\n"] + self.edgelayer\
            + ["\\end{pgfonlayer}\n"]
        end = ["\\end{tikzpicture}\n"]

        def write(lines, actual_path):
            with open(actual_path, 'w+') as file:
                file.writelines(lines)

        def save(file_path, lines):
            if params.get("compare", False):
                save_and_compare(file_path, partial(write, lines))
            else:
                write(lines, file_path)

        if path is not None:
            if output_tikzstyle:
                style_path = os.path.splitext(os.fspath(path))[0]\
                    + '.tikzstyles'
                save(style_path, ["% Node styles\n"] + self.node_styles
                     + ["% Edge styles\n"] + self.edge_styles)
            save(path, begin + nodes + edges + end)
        elif show:  # pragma: no cover
            if output_tikzstyle:
                print(''.join(self.node_styles + self.edge_styles))
            print(''.join(begin + nodes + edges + end))


def wire_bezier_points(source, target, bend_out):
    """
    The two control points of the cubic bezier of a wire from ``source`` to
    ``target``, or ``None`` when it is straight: the quadratic curve of the
    Matplotlib backend, which bends out of the source or else into the
    target.
    """
    if source[0] == target[0] or source[1] == target[1]:
        return None
    mid = (target[0], source[1]) if bend_out else (source[0], target[1])
    c1 = ((source[0] + 2 * mid[0]) / 3, (source[1] + 2 * mid[1]) / 3)
    c2 = ((2 * mid[0] + target[0]) / 3, (2 * mid[1] + target[1]) / 3)
    return c1, c2


class Typst(Backend):
    """Typst/CeTZ drawing backend."""

    def __init__(self, **params):
        self.body: list = []
        self.labels: list = []
        self.nodesize = params.get("nodesize", 1)
        self.fontsize = params.get("fontsize", DEFAULT["fontsize"])
        # Match Matplotlib's default stroke weight so wires/boxes read at
        # paper scale once the page is auto-cropped (see ``to_document``).
        self.linewidth = params.get("linewidth", 1.0)
        super().__init__()

    def draw(self, graph, **params):
        """
        Draw a graph on the canvas, as :func:`draw` does, and return the
        backend: :meth:`to_document` is its source, :meth:`output` compiles
        it.
        """
        max_v = max(graph.height, graph.width, 0.01)
        self.nodesize = round(params.get("nodesize", 1.0) / sqrt(max_v), 3)
        canvas = Transposed(self, Point(0, 0), graph)\
            if params.get("left_to_right", False) else self
        canvas.draw_boundary(graph, **params)
        canvas.draw_drawing(graph, **params)
        return self

    @staticmethod
    def format_color(color):
        """Format a colour name or hexcode to a Typst-friendly string."""
        if color is None or Backend.is_transparent(color):
            return "none"
        color = Backend.resolve(color)
        return to_hex(color) if is_color_like(color) else color

    @staticmethod
    def color_expr(hexcode):
        """Build a Typst ``rgb(\"#hex\")`` expression string."""
        if hexcode == "none":
            return "none"
        return f'rgb("{hexcode}")' if hexcode.startswith("#") else hexcode

    @staticmethod
    def math_label(text):
        """
        Return ``(inner, math)`` for a diagram label.

        A label between dollars, with no other dollar but escaped ones, is
        LaTeX mathematics, as matplotlib and TikZ read it, written in Typst
        mathematics by :func:`~discopy.drawing.typst_ast.latex_math`. Any
        other label is plain text, e.g. ``Alice``, which Typst would read as
        a variable, or ``$a$ and $b$``.

        >>> Typst.math_label("$\\\\mapsto$"), Typst.math_label("Alice")
        (('arrow.r.bar', True), ('Alice', False))
        """
        text = str(text)
        if len(text) >= 2 and text[0] == text[-1] == "$"\
                and not re.search(r"(?<!\\)\$", text[1:-1]):
            return latex_math(text[1:-1]), True
        return text, False

    def coord(self, x, y):
        """ A point of the canvas. """
        return Coord(x, y)

    def stroke(self, color="black", linewidth=None):
        """ A stroke, i.e. a colour and a width in points. """
        lw = self.linewidth if linewidth is None else linewidth
        return Ident(
            f"{self.color_expr(self.format_color(color))} + {lw:g}pt")

    def draw_text(self, text, i, j, **params):
        colour = params.get("color", None)
        extra: dict[str, TypstNode] = {}
        if colour and colour != "black":
            extra["fill"] = Ident(self.color_expr(self.format_color(colour)))
        # Wire labels use va='top' (see draw_wire_label); park them to the
        # right of the wire like TikZ's ``anchor=west``, so the stroke does
        # not strike through the glyph.
        if params.get("verticalalignment", "center") == "top":
            extra["anchor"] = Str("west")
        inner, math = self.math_label(text)
        content = ContentExpr(inner, math=math)
        fs = params.get("fontsize", None)
        if fs is not None and fs != DEFAULT["fontsize"]:
            scale = Float(fs / DEFAULT["fontsize"]).render()
            text_call = Call(
                Ident("text"), [content], {"size": Ident(f"{scale}em")})
        else:
            text_call = content
        self.labels.append(Call(
            Ident("content"), [self.coord(i, j), text_call], extra))
        super().draw_text(text, i, j, **params)

    def draw_node(self, i, j, **params):
        """
        A node of a controlled gate: a disc of the given colour, its edge
        drawn when it has an ``edgecolor``, or a plus sign.
        """
        radius = 0.06 * self.nodesize * sqrt(params.get("nodesize", 1))
        if params.get("shape", "circle") == "plus":
            for x, y in ((radius, 0), (0, radius)):
                self.body.append(Call(
                    Ident("line"),
                    [self.coord(i - x, j - y), self.coord(i + x, j + y)],
                    {"stroke": self.stroke(linewidth=0.5)}))
        else:
            edge = params.get("edgecolor", None)
            self.body.append(Call(Ident("circle"), [self.coord(i, j)], {
                "radius": Float(radius),
                "fill": Ident(self.color_expr(
                    self.format_color(params.get("color", "black")))),
                "stroke": Ident("none") if edge is None
                else self.stroke(edge, linewidth=0.5)}))
        super().draw_node(i, j, **params)

    def draw_polygon(self, *points,
                     facecolor=DEFAULT["facecolor"],
                     edgecolor=DEFAULT["edgecolor"]):
        coords = [self.coord(*p) for p in points]
        kwargs = {"close": Bool(True)}
        if facecolor is None or self.is_transparent(facecolor):
            kwargs["fill"] = Ident("none")
        else:
            kwargs["fill"] = Ident(
                self.color_expr(self.format_color(facecolor)))
        if edgecolor is None or self.is_transparent(edgecolor):
            kwargs["stroke"] = Ident("none")
        else:
            kwargs["stroke"] = self.stroke(edgecolor)
        self.body.append(Call(Ident("line"), coords, kwargs))
        super().draw_polygon(*points)

    def draw_wire(self, source, target, bend_out=False, bend_in=False,
                  style=None, linewidth=None, adaptive=True):
        options: dict[str, TypstNode] = {
            "stroke": self.stroke("black", linewidth)}
        cps = wire_bezier_points(source, target, bend_out)
        if style == "->":
            cps, options["mark"] = None, Ident('(end: ">", fill: black)')
        if cps is not None:
            c1, c2 = cps
            self.body.append(
                Call(
                    Ident("bezier"),
                    [
                        self.coord(*source),
                        self.coord(*target),
                        self.coord(*c1),
                        self.coord(*c2),
                    ],
                    options,
                )
            )
        else:
            self.body.append(
                Call(
                    Ident("line"),
                    [self.coord(*source), self.coord(*target)],
                    options,
                )
            )
        super().draw_wire(source, target, bend_out=bend_out, bend_in=bend_in)

    def draw_bezier(self, points, adaptive=True):
        start, first, second, end = points
        self.body.append(
            Call(
                Ident("bezier"),
                [self.coord(*start), self.coord(*end),
                 self.coord(*first), self.coord(*second)],
                {"stroke": self.stroke()},
            )
        )
        super().draw_bezier(points)

    def draw_filled_shape(self, start, steps, color, stroke=False):
        """
        The shape of :meth:`Backend.draw_filled_shape` as one CeTZ
        ``merge-path`` of its lines and curves, with a hairline in its own
        colour if ``stroke``, else none, as :meth:`TikZ.draw_filled_shape`.

        >>> typst = Typst()
        >>> typst.draw_filled_shape((0, 0), [
        ...     ("line", (1, 0)), ("curve", (1, 1), (0, 1), (0, 0))], "red")
        >>> print(typst.body[0].render())  # doctest: +NORMALIZE_WHITESPACE
        merge-path({line((0, 0), (1, 0));
            bezier((1, 0), (0, 0), (1, 1), (0, 1))},
            close: true, fill: rgb("#e8a5a5"), stroke: none)
        """
        segments, point = [], start
        for step in steps:
            if step[0] == "line":
                segments.append(Call(Ident("line"), [
                    self.coord(*point), self.coord(*step[1])]))
            else:
                segments.append(Call(Ident("bezier"), [
                    self.coord(*point), self.coord(*step[3]),
                    self.coord(*step[1]), self.coord(*step[2])]))
            point = step[-1]
        self.body.append(Call(Ident("merge-path"), [RawText(
            "{" + "; ".join(segment.render() for segment in segments) + "}")
        ], {"close": Bool(True),
            "fill": Ident(self.color_expr(self.format_color(color))),
            "stroke": self.stroke(color, linewidth=0.25) if stroke
            else Ident("none")}))
        super().draw_filled_shape(start, steps, color)

    def draw_region_cell(self, left, right, facecolor):
        """
        Fill the cell between two Beziers as one closed path, see
        :meth:`Backend.region_cell_steps`, stroked in its own colour so that
        two cells meeting edge to edge leave no seam, as
        :meth:`TikZ.draw_region_cell` does.
        """
        self.draw_filled_shape(
            left[0], self.region_cell_steps(left, right), facecolor,
            stroke=True)
        super().draw_region_cell(left, right, facecolor)

    def spider_shape(self, box, i, j) -> list[Call]:
        """
        The shape of a spider at ``(i, j)``, one of :data:`SHAPES` as
        matplotlib draws it: a square, a triangle pointing up or down, a
        plus sign in its colour or else a disc. Nothing when its colour is
        transparent: such a spider is just its label, e.g. the symbol of an
        :class:`discopy.monoidal.Equation`.

        >>> from discopy.frobenius import Ty, Spider
        >>> spider = Spider(1, 1, Ty('x'))
        >>> for shape in SHAPES:
        ...     spider.shape = shape
        ...     call, *_ = Typst().spider_shape(spider, 0, 0)
        ...     print(shape, call.func.render(), len(call.args))
        rectangle rect 2
        triangle_up line 3
        triangle_down line 3
        circle circle 1
        plus line 2
        """
        if self.is_transparent(box.color):
            return []
        colour = self.color_expr(self.format_color(box.color))
        options = {"fill": Ident(colour), "stroke": self.stroke(linewidth=0.5)}
        size = self.nodesize * 0.15
        if box.shape == "rectangle":
            return [Call(Ident("rect"), [
                self.coord(i - size, j - size),
                self.coord(i + size, j + size)], options)]
        if box.shape in ("triangle_up", "triangle_down"):
            apex = size if box.shape == "triangle_up" else -size
            return [Call(Ident("line"), [
                self.coord(i - size, j - apex), self.coord(i + size, j - apex),
                self.coord(i, j + apex)], {"close": Bool(True), **options})]
        if box.shape == "plus":
            return [Call(Ident("line"), [
                self.coord(i - x, j - y), self.coord(i + x, j + y)],
                {"stroke": self.stroke(box.color)})
                for x, y in ((size, 0), (0, size))]
        return [Call(Ident("circle"), [self.coord(i, j)], {
            "radius": Float(self.nodesize * 0.12), **options})]

    def draw_spiders(self, graph, draw_box_labels=True, **params):
        """
        Draws each spider as its shape and its label, at
        ``fontsize_spiders`` whatever the ``fontsize``.
        """
        for node in graph.nodes:
            if node.kind != "box" or not node.box.draw_as_spider:
                continue
            i, j = graph.positions[node]
            hexcolour = self.format_color(node.box.color)
            self.body.extend(self.spider_shape(node.box, i, j))
            if draw_box_labels and node.box.drawing_name:
                label = node.box.drawing_name
                inner, math = self.math_label(label)
                fc = 'rgb("#ffffff")' if hexcolour == "#000000"\
                    else 'rgb("#000000")'
                scale = Float(params.get(
                    "fontsize_spiders", DEFAULT["fontsize_spiders"])
                    / DEFAULT["fontsize"]).render()
                lbl = Call(
                    Ident("text"), [ContentExpr(inner, math=math)],
                    {"fill": Ident(fc), "size": Ident(f"{scale}em")})
                self.labels.append(Call(
                    Ident("content"), [self.coord(i, j), lbl]))
        super().draw_spiders(graph, draw_box_labels)

    def to_document(self, **params):
        """
        The document of what was drawn: a CeTZ canvas on a page cropped to
        it, with ``page_margin`` around and ``unit_length`` to a unit. Its
        :attr:`labels` come after the rest of its :attr:`body`, so that no
        box or wire drawn after a label hides it, as TikZ puts its labels on
        a layer of their own.
        """
        margin = params.get("page_margin", "0.2cm")
        doc = Document(
            imports=[
                Import("@preview/cetz:0.5.2", members=["canvas", "draw"]),
            ],
            preamble=[
                RawText("#set page("
                        f"width: auto, height: auto, margin: {margin})")
            ],
        )
        unit_len = params.get("unit_length", "1in")
        canvas = Canvas(
            canvas_kwargs={"length": Ident(unit_len)},
            body=[RawText("import draw: *")] + self.body + self.labels,
        )
        doc.content = canvas
        return doc

    @staticmethod
    def compile(source: str) -> bytes:
        """ The SVG of a Typst ``source``, with the optional ``typst``. """
        try:
            import typst
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "Typst rendering requires the 'typst' package. "
                "Install it with: pip install typst") from error
        return typst.compile(source.encode(), format="svg", pretty=True)

    @classmethod
    def save(cls, source: str, path) -> str | bytes:
        """ Write a Typst ``source`` to a path ending in ``.typ``, its
        SVG to any other path, and return what was written. """
        if str(path).endswith(".typ"):
            pathlib.Path(path).write_bytes(source.encode("utf-8"))
            return source
        svg = cls.compile(source)
        pathlib.Path(path).write_bytes(svg)
        return svg

    def output(self, path=None, show=True, **params):
        """
        The Typst source of the drawing when ``path`` ends in ``.typ``,
        written there, else its SVG, written to ``path`` when there is one.
        With ``compare``, i.e. ``draw(doctest=path)``, the file at ``path``
        is a baseline, its source or its SVG, so ``path`` ends in ``.typ``
        or ``.svg``: written when missing, compared with otherwise, as
        :func:`save_and_compare` does for any other backend.
        """
        source = self.to_document(**params).render()
        is_source = isinstance(path, (str, os.PathLike))\
            and str(path).endswith(".typ")
        if params.get("compare", False):
            if not str(path).endswith((".typ", ".svg")):
                raise ValueError(
                    f"A Typst baseline ends in .typ or .svg, got {path!r}.")
            written = []
            save_and_compare(path, lambda file: written.append(
                self.save(source, file)))
            return written[0]
        if is_source:
            return self.save(source, path)
        svg = self.compile(source)
        if hasattr(path, "write"):
            path.write(svg.decode() if isinstance(path, TextIOBase) else svg)
            return svg
        if path is not None:
            pathlib.Path(path).write_bytes(svg)
            return svg
        if show:
            try:
                from IPython.display import SVG

                return SVG(svg)
            except ImportError:
                pass
        return svg


class Matplotlib(Backend):
    """ Matplotlib drawing backend. """
    def __init__(self, axis=None, figsize=None, linewidth=1, format=None):
        self.axis = axis or plt.subplots(figsize=figsize, facecolor='white')[1]
        self.linewidth = linewidth
        self.format = format
        self.dark_count = 0
        super().__init__()

    def dark_gid(self, kind):
        """
        A fresh id prefixed ``dark-stroke`` or ``dark-fill``, tagging an
        element drawn black on the neutral canvas so that
        :data:`DARK_MODE_STYLE` turns it white on a dark page.
        """
        self.dark_count += 1
        return f"dark-{kind}-{self.dark_count}"

    def draw_boundary(self, graph, boundary_color="none", **params):
        super().draw_boundary(graph, boundary_color=boundary_color, **params)

    def draw_text(self, text, i, j, **params):
        if params.get('fontsize') is None:
            params['fontsize'] = DEFAULT['fontsize']
        if params.pop('adaptive', False):
            params['gid'] = self.dark_gid("fill")
        self.axis.text(i, j, text, **params)
        super().draw_text(text, i, j, **params)

    def draw_node(self, i, j, **params):
        color = params.get("color", "black")
        self.axis.scatter(
            [i], [j],
            c=self.resolve(color),
            marker=SHAPES[params.get("shape", "circle")],
            s=300 * params.get("nodesize", 1),
            edgecolors=params.get("edgecolor", None),
            gid=self.dark_gid("fill")
            if params.get("adaptive", False) else None)
        super().draw_node(i, j, **params)

    def draw_polygon(
            self,
            *points,
            facecolor=DEFAULT["facecolor"],
            edgecolor=DEFAULT["edgecolor"]):
        codes = [Path.MOVETO]
        codes += len(points[1:]) * [Path.LINETO] + [Path.CLOSEPOLY]
        path = Path(points + points[:1], codes)
        self.axis.add_patch(PathPatch(
            path,
            linewidth=self.linewidth,
            facecolor=self.resolve(facecolor),
            edgecolor=self.resolve(edgecolor)))
        super().draw_polygon(*points)

    def draw_region_cell(self, left, right, facecolor):
        """
        Fill the cell between two Beziers with a single patch: down the
        ``left`` boundary, straight across the bottom, up the ``right``
        boundary and straight back across the top. Antialiasing is
        disabled so that abutting cells of the same colour do not leave a
        hairline seam where the background shows through.
        """
        def codes(curve):
            return (len(curve) - 1) * [
                Path.CURVE3 if len(curve) == 3 else Path.CURVE4]
        vertices = list(left) + list(reversed(right)) + [left[0]]
        self.axis.add_patch(PathPatch(
            Path(vertices, [Path.MOVETO] + codes(left) + [Path.LINETO]
                 + codes(right) + [Path.CLOSEPOLY]),
            linewidth=0, antialiased=False,
            facecolor=self.resolve(facecolor), edgecolor='none'))
        super().draw_region_cell(left, right, facecolor)

    def draw_legend(self, graph, **params):
        """Add a legend mapping each region colour to its label."""
        colours = self.region_colours(graph)
        if not colours:
            return
        handles = [
            Patch(facecolor=self.resolve(colour.name),
                  edgecolor="none", label=colour.legend_label)
            for colour in colours.values()]
        self.axis.legend(
            handles=handles, loc=params.get("legend_loc", "upper right"),
            fontsize=params.get("fontsize_types", params.get("fontsize")))

    def draw_wire(self, source, target, bend_out=False, bend_in=False,
                  style=None, linewidth=None, adaptive=True):
        linewidth = self.linewidth if linewidth is None else linewidth
        if style == '->':  # pragma: no cover
            self.axis.arrow(
                *(source + (target[0] - source[0], target[1] - source[1])),
                head_width=.02, color="black")
        else:
            mid = (target[0], source[1])\
                if bend_out else (source[0], target[1])
            path = Path([source, mid, target],
                        [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            self.axis.add_patch(PathPatch(
                path, facecolor='none', linewidth=linewidth,
                gid=self.dark_gid("stroke")
                if adaptive and linewidth else None))
        super().draw_wire(source, target, bend_out=bend_out, bend_in=bend_in)

    def draw_bezier(self, points, adaptive=True):
        path = Path(
            list(points),
            [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        self.axis.add_patch(PathPatch(
            path, facecolor='none', linewidth=self.linewidth,
            gid=self.dark_gid("stroke")
            if adaptive and self.linewidth else None))
        super().draw_bezier(points)

    def draw_filled_shape(self, start, steps, color):
        vertices, codes = [start], [Path.MOVETO]
        for step in steps:
            if step[0] == "line":
                vertices.append(step[1])
                codes.append(Path.LINETO)
            else:
                vertices += [step[1], step[2], step[3]]
                codes += 3 * [Path.CURVE4]
        vertices.append(start)
        codes.append(Path.CLOSEPOLY)
        self.axis.add_patch(PathPatch(
            Path(vertices, codes), facecolor=self.resolve(color),
            edgecolor='none', linewidth=0))
        super().draw_filled_shape(start, steps, color)

    def draw_spiders(self, graph, draw_box_labels=True, **params):
        """
        Draws the spiders, grouped by shape and by whether they adapt to a
        dark page: black spiders lie on the neutral canvas so they turn
        white, coloured ones keep their colour. A transparent spider is just
        its label, e.g. the symbol of an :class:`discopy.monoidal.Equation`,
        which leaves no patch on a coloured page. The labels are drawn at
        ``DRAWING_DEFAULT["fontsize_spiders"]``, whatever the ``fontsize``.
        """
        import networkx as nx
        nodes = [node for node in graph.nodes
                 if node.kind == "box" and node.box.draw_as_spider]
        groups = {}
        for node in nodes:
            groups.setdefault(
                (node.box.shape, isinstance(node.box.color, str)
                 and node.box.color == "black"), []).append(node)
        for (shape, adaptive), group in groups.items():
            nx.draw_networkx_nodes(
                *graph.inside, nodelist=group,
                node_color=[self.resolve(node.box.color) for node in group],
                node_shape=SHAPES[shape], ax=self.axis,
                node_size=300 * params.get("nodesize", 1)
            ).set_gid(self.dark_gid("fill") if adaptive else None)
            if draw_box_labels:
                for node in group:
                    self.draw_text(
                        node.box.drawing_name, *graph.positions[node],
                        ha='center', va='center',
                        fontsize=DEFAULT['fontsize_spiders'],
                        adaptive=self.is_transparent(node.box.color))
        super().draw_spiders(graph, draw_box_labels)

    def output(self, path=None, show=True, **params):
        xlim, ylim = params.get("xlim", None), params.get("ylim", None)
        margins = params.get("margins", DEFAULT['margins'])
        plt.margins(*margins)
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        self.axis.set_aspect(params.get("aspect", "auto"))
        plt.axis('off')
        if xlim is not None:
            self.axis.set_xlim(*xlim)
        if ylim is not None:
            self.axis.set_ylim(*ylim)
        if path is not None:
            try:
                savefig(
                    path, format=self.format,
                    compare=params.get("compare", False),
                    tol=params.get("tol", DEFAULT['plt_tol']))
            finally:
                plt.close()
        elif not show:
            plt.close(self.axis.figure)  # ty: ignore[invalid-argument-type]
        if show:
            plt.show()
