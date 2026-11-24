"""
DisCoPy's drawing modules: :mod:`discopy.drawing.drawing` is the category of
drawings, i.e. labeled graphs embedded in the plane, onto which every diagram
is sent by a functor; :mod:`discopy.drawing.backend` renders them with
matplotlib, TikZ or Typst; :mod:`discopy.drawing.widget` shows them as
interactive anywidget widgets in notebooks.
"""

import os
from base64 import b64encode
from PIL import Image
from tempfile import TemporaryDirectory

from discopy.utils import Node, Point
from discopy.drawing import backend, drawing
from discopy.drawing.backend import (
    Backend,
    TikZ,
    Typst,
    Matplotlib,
    ATTRIBUTES,
    DEFAULT,
    COLORS,
    SHAPES,
)
from discopy.drawing.drawing import (
    Point,
    PlaneGraph,
    Drawing,
)


def __getattr__(name):
    if name == "Equation":
        import warnings
        from discopy.monoidal import Equation
        warnings.warn(
            "discopy.drawing.Equation is deprecated, use the Equation of the "
            "relevant module instead, e.g. discopy.symmetric.Equation or "
            "discopy.monoidal.Equation.", DeprecationWarning, stacklevel=2)
        return Equation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def draw(diagram, **params):
    """
    Draws a diagram with matplotlib, or as TikZ code with ``to_tikz=True``,
    see :func:`discopy.drawing.backend.draw`.

    Parameters
    ----------
    textpad : pair of floats, optional
        Padding between text and wires, default is :code:`(2**-4, 2**-4)`.
    wire_labels : bool, optional
        Whether to draw type labels, default is :code:`False`.
    draw_box_labels : bool, optional
        Whether to draw box labels, default is :code:`True`.
    smooth_wires : bool, optional
        Whether to draw a wire between two ports vertical at both ends, as a
        cubic Bezier, rather than bending into the lower one, default is
        :code:`False`.
    aspect : string, optional
        Aspect ratio, one of :code:`['auto', 'equal']`.
    margins : tuple, optional
        Margins, default is :code:`(0, 0)`.
    nodesize : float, optional
        Node size for spiders and controlled gates.
    fontsize : int, optional
        Font size for the boxes, default is :code:`10`.
    fontsize_types : int, optional
        Font size for the types, default is :code:`10`.
    figsize : tuple, optional
        Figure size.
    path : str, optional
        Where to save the image, if `None` we call :code:`plt.show()`.
    format : str, optional
        Format of the saved image, taken from the extension of :code:`path`
        when it is a file name, required when it is an in-memory buffer;
        :code:`"typst"` draws with the :class:`Typst` backend.
    doctest : str, optional
        Path to a documentation image used as a drawing baseline: the
        image is created if missing and compared against otherwise, or
        overwritten when :code:`discopy.config.OVERRIDE_DOCTEST_IMAGES`
        is :code:`True`.
    tol : float, optional
        Comparison tolerance for raster images, default is :code:`20`.
    to_tikz : bool, optional
        Whether to output tikz code instead of matplotlib.
    asymmetry : float, optional
        Shift one corner of each box drawn as a plain polygon (not spiders,
        wires, braids or other special shapes) so that a box, its dagger, its
        transpose and its conjugate are told apart, default is :code:`0.125`
        when some box is a conjugate, a transpose or a dagger not drawn as a
        braid, :code:`0` otherwise.
    """
    return diagram.to_drawing().draw(**params)


def from_picture(jpeg, config: dict):
    """
    The diagram a picture of it shows, read back by
    :mod:`discopy.drawing.vision`: a pixel net says what each pixel is, a
    small OCR model reads the name of each box and the type of each
    labelled wire, the module is the smallest in which it decodes from the
    one whose structural primitives and types appear, and
    :func:`discopy.drawing.picture.decode` builds the diagram. It needs
    PyTorch and trained weights, see
    :func:`discopy.drawing.vision.from_picture` for the ``config``.
    """
    from discopy.drawing import vision
    return vision.from_picture(jpeg, config)


def to_typst(diagram, **params):
    """
    Convert a diagram to a Typst Document AST.

    Parameters
    ----------
    diagram : Diagram
        The diagram to convert.
    params : Passed to :class:`Typst`.

    Returns
    -------
    doc : Document
        A Typst Document AST that can be rendered to source text or SVG.

    Example
    -------
    >>> from discopy.monoidal import Ty, Box
    >>> f = Box('f', Ty('x'), Ty('y'))
    >>> doc = to_typst(f)
    >>> source = doc.render()
    >>> assert 'import "@preview/cetz' in source
    >>> assert 'bezier' in source or 'line' in source
    """
    return diagram.to_drawing().to_typst(**params)


def to_gif(diagram, *diagrams, **params):  # pragma: no cover
    """
    Builds a gif with the normalisation steps.

    The gif is embedded in the :code:`<img>` tag that gets returned, so that
    it displays wherever the notebook gets rendered rather than only
    relative to the working directory.

    Parameters
    ----------
    diagrams : :class:`Diagram`, optional
        Sequence of diagrams to draw.
    path : str
        Where to save the image, if :code:`None` the gif is only embedded.
    timestep : int, optional
        Time step in milliseconds, default is :code:`500`.
    tween : int, optional
        Frames per step, default is :code:`1`: from each diagram to the next
        the drawing moves by :meth:`Drawing.interpolate`, so that the gif is
        an isotopy, e.g. along :meth:`discopy.monoidal.Equation.path`, and
        raises :code:`ValueError` when two consecutive diagrams are not one
        move apart. The wires between two ports are then drawn vertical at
        both ends, i.e. with :code:`smooth_wires=True`.
    loop : bool, optional
        Whether to loop, default is :code:`False`, going back to the first
        diagram by playing the gif backwards unless it ends where it starts,
        e.g. the loop of an Eckmann-Hilton argument.
    params : any, optional
        Passed to :meth:`Diagram.draw`.
    """
    path, compare = backend.doctest_or_path(
        params.pop("path", None), params.pop("doctest", None))
    timestep = params.get("timestep", 500)
    loop = params.get("loop", False)
    tween = params.pop("tween", 1)
    params.setdefault("smooth_wires", tween > 1)
    steps, frames = [d.to_drawing() for d in (diagram, ) + diagrams], []
    steps = tweened(steps, tween)
    if 'figsize' not in params:
        params['figsize'] = tuple(
            max(getattr(step, attr) for step in steps)
            for attr in ("width", "height"))
    if tween > 1:
        params.setdefault("xlim", (0, max(step.width for step in steps)))
        params.setdefault("ylim", (0, max(step.height for step in steps)))
    with TemporaryDirectory() as directory:
        for i, _diagram in enumerate(steps):
            tmp_path = os.path.join(directory, f'{i}.png')
            _diagram.draw(path=tmp_path, **dict(params, show=False))
            frames.append(Image.open(tmp_path))
        if loop and diagrams and diagrams[-1] == diagram:
            frames = frames[:-1]
        elif loop:
            frames = frames + frames[::-1]

        def save(actual_path):
            frames[0].save(
                actual_path, format='GIF', append_images=frames[1:],
                save_all=True, duration=timestep,
                **{'loop': 0} if loop else {})

        path = path or os.path.join(directory, 'tmp.gif')
        if compare:
            backend.save_and_compare(
                path, save, tol=params.get("tol", backend.DEFAULT['plt_tol']))
        else:
            save(path)
        with open(path, 'rb') as file:
            source = b64encode(file.read()).decode()
    image = ('<img alt="animated diagram" '
             f'src="data:image/gif;base64,{source}">')
    try:
        from IPython.display import HTML
        return HTML(image)
    except ImportError:
        return image


def tweened(steps: list[Drawing], tween: int) -> list[Drawing]:
    """
    The drawings with ``tween - 1`` more between each one and the next, see
    :meth:`Drawing.interpolate`.

    Example
    -------
    >>> from discopy.monoidal import Ty, Box
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> steps = [d.to_drawing() for d in (f @ x >> x @ g, x @ g >> f @ x)]
    >>> assert len(tweened(steps, 4)) == 1 + 4
    >>> assert len(tweened(steps + [f.to_drawing()], 1)) == 3
    """
    result = steps[:1]
    for before, after in zip(steps, steps[1:]):
        result += [before.interpolate(after, k / tween)
                   for k in range(1, tween)] + [after]
    return result


def spiral(n_cups):
    """
    Implements the asymptotic worst-case for normal_form, see arXiv:1804.07832.
    """
    from discopy.monoidal import Ty, Box, Id
    x = Ty('x')
    unit, counit = Box('unit', Ty(), x), Box('counit', x, Ty())
    cup, cap = Box('cup', x @ x, Ty()), Box('cap', Ty(), x @ x)
    for box in [unit, counit, cup, cap]:
        box.draw_as_spider, box.drawing_name = True, ""
        box.shape, box.color = "circle", "black"
    result = unit
    for i in range(n_cups):
        result = result >> Id(x ** i) @ cap @ Id(x ** (i + 1))
    result = result >> Id(x ** n_cups) @ counit @ Id(x ** n_cups)
    for i in range(n_cups):
        result = result >>\
            Id(x ** (n_cups - i - 1)) @ cup @ Id(x ** (n_cups - i - 1))
    return result
