# -*- coding: utf-8 -*-

"""
Rewrites, i.e. the 2-cells between diagrams, composed from steps.

The objects of the category :class:`Rewrite` are diagrams and its arrows are
sequences of :class:`Layer`, each a diagram with holes, its ``context``,
where every hole is filled by a :class:`Step`, a generating 2-cell, e.g. an
:class:`Interchange` or a :class:`Slide` of a box through a braid. Each step
of a layer happens at once, so that a rewrite can rewrite in parallel.

Rewrites form a sesquicategory :cite:p:`Street96`: they compose vertically
with ``>>``, as in :mod:`discopy.cat`, and they are whiskered by diagrams
(:meth:`Rewrite.whisker`) and tensored (:meth:`Rewrite.tensor`), but they do
not satisfy the interchange law of a 2-category. Their equality is strict:
two rewrites are equal when their layers are, so the two orders in which two
steps can happen one after the other differ, as do both and doing them at
once. The axioms of the free monoidal 2-category, e.g. the naturality of the
interchanger, are not checked.

Drawing a rewrite is a pseudofunctor into animations, see
:meth:`Rewrite.to_gif`: the frame of a layer at time ``t`` is the drawing of
its context with each hole drawn by the frame of its step at time ``t``, see
:meth:`Step.frame`. The last frame of a layer and the first of the next are
both drawings of the same diagram, laid out around different holes, so they
need not be equal: the animation plays the laxator from one to the other,
see :meth:`discopy.drawing.Drawing.interpolate`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Rewrite
    Layer
    Step
    Interchange
    Slide

Example
-------
Two interchangers take one scalar round the other: the side each one goes
is an argument of the step, since both interchangers of two scalars have the
same domain and codomain.

>>> from discopy.monoidal import Box, Ty
>>> s0, s1 = Box('s0', Ty(), Ty()), Box('s1', Ty(), Ty())
>>> loop = Interchange(s0, s1, left=True) >> Interchange(s1, s0, left=True)
>>> print(loop)
Interchange(s0, s1, left=True) >> Interchange(s1, s0, left=True)
>>> assert loop.dom == s0 >> s1 == loop.cod
>>> assert Interchange(s0, s1).dom == Interchange(s0, s1, left=True).dom
>>> _ = loop.to_gif(tween=12, timestep=60, loop=True,
...     doctest="docs/_static/rewrite/eckmann-hilton.gif")

.. image:: /_static/rewrite/eckmann-hilton.gif
    :align: center

The steps that normalise a diagram are a rewrite, and so is the way from one
diagram to another through their normal form:

>>> x = Ty('x')
>>> f, g, h = Box('f', x, x), Box('g', x @ x, x), Box('h', x, x @ x)
>>> source = f @ x @ x >> x @ g >> x @ h
>>> target = x @ g >> x @ h >> f @ x @ x
>>> rewrite = source.normalization() >> target.normalization().dagger()
>>> for layer in rewrite.inside: print(layer)
Interchange(f, g, left=True) >> x @ h
x @ g >> Interchange(f, h, left=True)
>>> _ = rewrite.to_gif(tween=8, timestep=80, loop=True,
...     doctest="docs/_static/rewrite/planar-isotopy.gif")

.. image:: /_static/rewrite/planar-isotopy.gif
    :align: center
"""

from __future__ import annotations

from discopy import cat, messages, monoidal
from discopy.drawing import Drawing, tweened, to_gif
from discopy.utils import AxiomError, factory


class Layer(cat.Box):
    """
    A layer of a rewrite: a diagram, its ``context``, in which each box with
    a :class:`Step` as ``data`` is a hole, the :attr:`Step.hole` of that
    step. Its domain is the context with each hole replaced by the domain of
    its step and its codomain the same with their codomains.

    Parameters:
        context : The diagram with holes.

    Example
    -------
    >>> from discopy.monoidal import Box, Ty
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> step = Interchange(f, g)
    >>> layer = Layer(f @ x >> step.hole)
    >>> print(layer, "|", layer.dom, "|", layer.cod)
    f @ x >> Interchange(f, g) | f @ x >> x @ f >> g @ x \
| f @ x >> g @ x >> x @ f
    >>> assert layer.steps == (step, )
    """
    ob = monoidal.Diagram
    serialised_attrs = ("context", )

    def __init__(self, context: monoidal.Diagram):
        self.context = context
        cat.Box.__init__(
            self, str(context), self.fill(lambda step: step.dom),
            self.fill(lambda step: step.cod))

    @property
    def steps(self) -> tuple[Step, ...]:
        """ The steps of the holes, in the order of the boxes. """
        return tuple(box.data for box in self.context.boxes
                     if isinstance(box.data, Step))

    def fill(self, side) -> monoidal.Diagram:
        """
        The context with each hole replaced by ``side`` of its step.

        Parameters:
            side : The function from a step to the diagram in its hole.
        """
        factory_ = self.context.factory
        return self.context.functor_factory(
            ob_map=lambda x: x, ar_map=lambda box: side(box.data)
            if isinstance(box.data, Step) else box,
            dom=factory_, cod=factory_)(self.context)

    def frame(self, t: float) -> Drawing:
        """
        The drawing of the context with each hole drawn by the frame of its
        step at time ``t``, see :meth:`Step.frame`.

        Parameters:
            t : The fraction of the way, from ``0`` to ``1``.
        """
        return self.context.functor_factory(
            ob_map=lambda x: x.to_drawing(),
            ar_map=lambda box: box.data.frame(t)
            if isinstance(box.data, Step) else box.to_drawing(),
            dom=self.context.factory, cod=Drawing)(self.context)

    def dagger(self) -> Layer:
        """ The layer in the same context, with each step reversed. """
        return Layer(self.fill(lambda step: step.dagger().hole))

    def __eq__(self, other):
        return isinstance(other, Layer) and self.context == other.context

    def __hash__(self):
        return hash(self.context)

    def __repr__(self):
        return f"rewrite.Layer({self.context!r})"


@factory
class Rewrite(cat.Arrow):
    """
    A rewrite is a sequence of composable :class:`Layer` from a diagram
    ``dom`` to a diagram ``cod``.

    Parameters:
        inside : The layers of the rewrite.
        dom : The diagram it rewrites.
        cod : The diagram it rewrites into.
    """
    ob = monoidal.Diagram
    generator_factory = Layer

    def whisker(self, above=None, below=None, left=None, right=None
                ) -> Rewrite:
        """
        The rewrite in the context ``above >> left @ - @ right >> below``.

        Parameters:
            above : The diagram above, if any.
            below : The diagram below, if any.
            left : The type or diagram on the left, if any.
            right : The type or diagram on the right, if any.

        Example
        -------
        >>> from discopy.monoidal import Box, Ty
        >>> x = Ty('x')
        >>> f, g = Box('f', x, x), Box('g', x, x)
        >>> step = Interchange(f, g).whisker(above=f @ x @ x, right=x)
        >>> print(step)
        f @ x @ x >> Interchange(f, g) @ x
        >>> print(step.cod)
        f @ x @ x >> g @ x @ x >> x @ f @ x
        """
        def around(diagram):
            diagram = diagram if left is None else left @ diagram
            diagram = diagram if right is None else diagram @ right
            diagram = diagram if above is None else above >> diagram
            return diagram if below is None else diagram >> below
        return self.ar(
            tuple(Layer(around(layer.context)) for layer in self.inside),
            around(self.dom), around(self.cod))

    def tensor(self, other: Rewrite | monoidal.Diagram) -> Rewrite:
        """
        The rewrites side by side, at once: the layers of both in turn, each
        tensored with the one of the other at the same position, the shorter
        padded with layers with no holes. A diagram is its identity rewrite.
        Called with ``@``.

        Parameters:
            other : The rewrite on the right.

        Example
        -------
        >>> from discopy.monoidal import Box, Ty
        >>> x = Ty('x')
        >>> f, g = Box('f', x, x), Box('g', x, x)
        >>> both = Interchange(f, g, left=True) @ Interchange(g, f, left=True)
        >>> layer, = both.inside
        >>> assert len(layer.steps) == 2
        >>> print(both.cod)
        x @ g @ x @ x >> f @ x @ x @ x >> x @ x @ x @ f >> x @ x @ g @ x
        """
        if not isinstance(other, Rewrite):
            other = self.id(other)
        length = max(len(self.inside), len(other.inside))

        def padded(rewrite):
            return rewrite.inside\
                + (Layer(rewrite.cod), ) * (length - len(rewrite.inside))
        return self.ar(tuple(
            Layer(this.context @ that.context)
            for this, that in zip(padded(self), padded(other))),
            self.dom @ other.dom, self.cod @ other.cod)

    __matmul__ = tensor

    def frames(self, tween: int = 1) -> list[Drawing]:
        """
        The drawings of each layer at times ``k / tween`` for ``k`` from
        ``0`` to ``tween``, see :meth:`Layer.frame`, from the drawing of the
        domain to that of the codomain. Two drawings of one diagram need not
        be equal, e.g. the frame of a layer and the drawing of its domain:
        between them the laxator moves from one to the other in ``tween``
        frames, see :meth:`discopy.drawing.Drawing.interpolate`.

        Parameters:
            tween : The number of frames for each layer and each laxator.

        Example
        -------
        >>> from discopy.monoidal import Box, Ty
        >>> x = Ty('x')
        >>> e, s = Box('e', x, Ty()), Box('s', Ty(), x @ x @ x)
        >>> m = Box('m', x @ x, x)
        >>> rewrite = Interchange(e, s, left=True).whisker(above=m)
        >>> layer, = rewrite.inside
        >>> assert layer.frame(0) != rewrite.dom.to_drawing()
        >>> assert len(rewrite.frames(4)) == 1 + 4 + 4
        >>> assert len(Rewrite.id(m).frames(4)) == 1
        """
        result = [self.dom.to_drawing()]
        for layer in self.inside:
            first, *others = (layer.frame(k / tween) for k in range(tween + 1))
            if result[-1] != first:
                result += tweened([result[-1], first], tween)[1:]
            result += others
        last = self.cod.to_drawing()
        if result[-1] != last:
            result += tweened([result[-1], last], tween)[1:]
        return result

    def to_gif(self, tween: int = 1, **params):
        """
        Animate the rewrite, its :meth:`frames`, see
        :func:`discopy.drawing.to_gif`, which loops round a rewrite that ends
        where it starts.

        Parameters:
            tween : The number of frames for each layer and each laxator.
            params : Passed to :func:`discopy.drawing.to_gif`.
        """
        frames = self.frames(tween)
        params.setdefault("smooth_wires", tween > 1)
        params.setdefault("xlim", (0, max(frame.width for frame in frames)))
        params.setdefault("ylim", (0, max(frame.height for frame in frames)))
        return to_gif(*frames, **params)


class Step(cat.Box, Rewrite):
    """
    A step is a generating rewrite from a diagram ``dom`` to a diagram
    ``cod`` with the same domain and codomain. It is the rewrite of one layer
    with one hole, its :attr:`hole`, a box named after the step with the step
    as ``data``.

    A step is drawn by :meth:`frame`. Its subclasses are the steps the axioms
    give, :class:`Interchange` and :class:`Slide`, each with its own frames.

    Parameters:
        name : The name of the step.
        dom : The diagram it rewrites.
        cod : The diagram it rewrites into.
        data : Extra data in the step, default is ``None``.
        is_dagger : Whether the step is reversed.

    Example
    -------
    >>> from discopy.monoidal import Box, Ty
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> step = Step('commute', f >> g, g >> f)
    >>> print(step.dagger(), "|", step.dagger().dom)
    commute[::-1] | g >> f
    >>> Step('wrong', f, Box('h', x, x @ x))
    Traceback (most recent call last):
    ...
    discopy.utils.AxiomError: Expected parallel arrows, got f: x -> x and \
h: x -> x @ x instead.
    """
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def __init__(self, name: str, dom: monoidal.Diagram,  # cat.Box's
                 cod: monoidal.Diagram, data=None, is_dagger=False):
        cat.Box.__init__(self, name, dom, cod, data, is_dagger)
        if (self.dom.dom, self.dom.cod) != (self.cod.dom, self.cod.cod):
            raise AxiomError(messages.NOT_PARALLEL.format(*(
                f"{d}: {d.dom} -> {d.cod}" for d in (self.dom, self.cod))))
        self.hole = self.dom.factory.box_factory(
            str(self), self.dom.dom, self.dom.cod, data=self)
        Rewrite.__init__(
            self, (Layer(self.hole), ), self.dom, self.cod, _scan=False)

    def frame(self, t: float) -> Drawing:
        """
        The drawing a fraction ``t`` of the way from ``dom`` to ``cod``, by
        :meth:`discopy.drawing.Drawing.interpolate`, which raises unless
        they are one interchanger or one slide apart.

        Parameters:
            t : The fraction of the way, from ``0`` to ``1``.
        """
        return self.dom.to_drawing().interpolate(self.cod.to_drawing(), t)


class Interchange(Step):
    """
    The interchanger of a box ``top`` with a box ``bottom`` under it, which
    moves ``bottom`` above ``top``: the instance of ``bifunctoriality`` with
    identities that :class:`discopy.monoidal.Diagram` breaks.

    Parameters:
        top : The box above, i.e. before.
        bottom : The box below, i.e. after, with the wires between them.
        left : Whether ``top`` is on the left of ``bottom``, else right.

    When ``top`` has no output and ``bottom`` no input, e.g. two scalars,
    both sides give the same domain and codomain and ``left`` is the side
    ``top`` passes on: the two wind in opposite directions.

    Example
    -------
    >>> from discopy.monoidal import Box, Ty
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> step = Interchange(f, g, left=True)
    >>> print(step.dom, "|", step.cod)
    f @ x >> x @ g | x @ g >> f @ x
    >>> assert step.dagger() == Interchange(g, f)
    """
    serialised_attrs = ("top", "bottom", "left")

    def __init__(self, top: monoidal.Diagram, bottom: monoidal.Diagram,
                 left=False):
        self.top, self.bottom, self.left = top, bottom, left
        if left:
            dom = top @ bottom.dom >> top.cod @ bottom
            cod = top.dom @ bottom >> top @ bottom.cod
        else:
            dom = bottom.dom @ top >> bottom @ top.cod
            cod = bottom @ top.dom >> bottom.cod @ top
        Step.__init__(self, f"Interchange({top}, {bottom}"
                      + (", left=True)" if left else ")"), dom, cod)

    def __repr__(self):
        return f"rewrite.Interchange({self.top!r}, {self.bottom!r}"\
            + (", left=True)" if self.left else ")")

    def dagger(self) -> Interchange:
        return Interchange(self.bottom, self.top, not self.left)

    def frame(self, t: float) -> Drawing:
        """
        The two boxes move apart, swap their rows and move back, ``top`` on
        its side, see :meth:`discopy.drawing.Drawing.interpolate`.

        Parameters:
            t : The fraction of the way, from ``0`` to ``1``.
        """
        return self.dom.to_drawing().interpolate(
            self.cod.to_drawing(), t, left=self.left)

    @classmethod
    def at(cls, diagram: monoidal.Diagram, i: int, left=False) -> Rewrite:
        """
        The interchanger of layers ``i`` and ``i + 1`` of a diagram with one
        box in each layer, whiskered by the rest: its codomain is
        ``diagram.interchange(i, i + 1, left)``, see
        :meth:`discopy.monoidal.Diagram.interchange`.

        Parameters:
            diagram : The diagram to rewrite.
            i : The layer of the box above.
            left : Whether the box above is on the left when it can be
                either, as for two scalars.

        Raises:
            AxiomError : When the two boxes are connected.

        Example
        -------
        >>> from discopy.monoidal import Box, Ty
        >>> x = Ty('x')
        >>> f, g = Box('f', x, x), Box('g', x, x)
        >>> diagram = f @ x @ x >> x @ x @ g >> f @ x @ x
        >>> step = Interchange.at(diagram, 0)
        >>> print(step)
        Interchange(f, x @ g, left=True) >> f @ x @ x
        >>> assert step.cod == diagram.interchange(0, 1)
        >>> Interchange.at(f >> g, 0)
        Traceback (most recent call last):
        ...
        discopy.utils.AxiomError: Boxes f and g do not commute.
        """
        top, bottom = diagram.boxes[i], diagram.boxes[i + 1]
        off0, off1 = diagram.offsets[i], diagram.offsets[i + 1]
        wires = diagram.inside[i + 1].dom
        is_left = off1 >= off0 + len(top.cod)
        is_right = off0 >= off1 + len(bottom.dom)
        if is_left and (left or not is_right):
            start, end = off0, off1 + len(bottom.dom)
            step = cls(top, wires[off0 + len(top.cod):off1] @ bottom, True)
        elif is_right:
            start, end = off1, off0 + len(top.cod)
            step = cls(top, bottom @ wires[off1 + len(bottom.dom):off0])
        else:
            raise AxiomError(messages.INTERCHANGER_ERROR.format(top, bottom))
        return step.whisker(
            diagram[:i], diagram[i + 2:], wires[:start], wires[end:])


class Slide(Step):
    """
    A box slides through the braid with a wire next to it: the instance of
    ``braid_naturality`` with an identity, see
    :meth:`discopy.braided.Diagram.naturality`.

    Parameters:
        box : The box that slides.
        wire : The type of one wire it slides through.
        left : Whether the wire is on the left of the box, else right.
        down : Whether the box slides down, else up.
        braid : The braid, the :meth:`discopy.braided.Diagram.braid` of the
            box's diagram by default.

    Example
    -------
    >>> from discopy.braided import Box, Ty
    >>> x, y, z = map(Ty, "xyz")
    >>> f = Box('f', x, y)
    >>> step = Slide(f, z)
    >>> print(step.dom, "|", step.cod)
    z @ f >> Braid(z, y) | Braid(z, x) >> f @ z
    >>> assert step.dagger() == Slide(f, z, left=False, down=False)
    """
    serialised_attrs = ("box", "wire", "*", "left", "down", "braid")

    # pylint: disable-next=too-many-arguments  # naturality's arguments
    def __init__(self, box: monoidal.Box, wire: monoidal.Ty, *, left=True,
                 down=True, braid=None):
        self.box, self.wire, self.left, self.down = box, wire, left, down
        self.braid = braid
        braid = braid or box.braid
        if left and down:
            dom = wire @ box >> braid(wire, box.cod)
            cod = braid(wire, box.dom) >> box @ wire
        elif left:
            dom = braid(box.dom, wire) >> wire @ box
            cod = box @ wire >> braid(box.cod, wire)
        elif down:
            dom = box @ wire >> braid(box.cod, wire)
            cod = braid(box.dom, wire) >> wire @ box
        else:
            dom = braid(wire, box.dom) >> box @ wire
            cod = wire @ box >> braid(wire, box.cod)
        Step.__init__(self, f"Slide({box}, {wire}{self.flags})", dom, cod)

    @property
    def flags(self) -> str:
        """ The arguments ``left`` and ``down`` that are not the default. """
        return (", left=False" if not self.left else "")\
            + (", down=False" if not self.down else "")

    def __repr__(self):
        return f"rewrite.Slide({self.box!r}, {self.wire!r}{self.flags})"

    def dagger(self) -> Slide:
        return Slide(self.box, self.wire, left=not self.left,
                     down=not self.down, braid=self.braid)

    def frame(self, t: float) -> Drawing:
        """
        The crossings shrink into the box and grow out of its other side,
        see :meth:`discopy.drawing.Drawing.slide`.

        Parameters:
            t : The fraction of the way, from ``0`` to ``1``.
        """
        return self.dom.to_drawing().slide(self.cod.to_drawing(), t)

    @classmethod
    # pylint: disable-next=too-many-arguments  # naturality's arguments
    def at(cls, diagram: monoidal.Diagram, i: int, *, left=True, down=True,
           braid=None) -> Rewrite:
        """
        The slide of box ``i`` of a diagram with one box in each layer,
        whiskered by the rest, see
        :meth:`discopy.braided.Diagram.naturality`, which is its codomain.

        Parameters:
            diagram : The diagram to rewrite.
            i : The index of the box to slide.
            left : Whether to slide left or right.
            down : Whether to slide down or up.
            braid : The braiding method to be used.

        Raises:
            AxiomError : When there is no braid to slide through.
        """
        box, offset = diagram.boxes[i], diagram.offsets[i]
        dom = diagram.inside[i].dom
        left_wires, right_wires = dom[:offset], dom[offset + len(box.dom):]
        wire = left_wires[-1] if left else right_wires[0]
        step = cls(box, wire, left=left, down=down, braid=braid)
        start = i if down else i - len(step.dom) + 1
        outside = (left_wires[:-1], right_wires) if left\
            else (left_wires, right_wires[1:])
        if start < 0 or diagram[start:start + len(step.dom)]\
                != outside[0] @ step.dom @ outside[1]:
            raise AxiomError(messages.NO_BRAID_TO_SLIDE.format(box, i))
        return step.whisker(
            diagram[:start], diagram[start + len(step.dom):], *outside)


Rewrite.box_factory = Step
