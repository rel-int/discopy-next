# -*- coding: utf-8 -*-

"""
Rewrites, i.e. the 2-cells between diagrams, composed from steps.

A :class:`Rewrite` is parametrised by the class of its diagrams, as
:class:`discopy.cmap.CMap` is, and each level of the hierarchy has its own,
e.g. ``monoidal.Rewrite = Rewrite[monoidal.Diagram]``, with the steps of its
axioms defined in its module.

The objects of the category :class:`Rewrite` are diagrams and its arrows are
sequences of :class:`Layer`, each a diagram with holes, its ``context``,
where every hole is filled by a :class:`Step`, a generating 2-cell, e.g. an
:class:`discopy.monoidal.Interchange` or a :class:`discopy.braided.Slide` of
a box through a braid. Each step
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

Example
-------
Two interchangers take one scalar round the other: the side each one goes
is an argument of the step, since both interchangers of two scalars have the
same domain and codomain.

>>> from discopy.monoidal import Box, Interchange, Ty
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

from typing import TYPE_CHECKING

from discopy import cat, messages
from discopy.drawing import Drawing, tweened, to_gif
from discopy.testable import TestableGeneric
from discopy.utils import (
    AxiomError, assert_isinstance, classproperty, factory)

if TYPE_CHECKING:
    from discopy import monoidal


class Layer[category](cat.Box, TestableGeneric):
    """
    A layer of a rewrite of diagrams of the ``category``, taken from the
    ``context`` when not given: a diagram in which each box with
    a :class:`Step` as ``data`` is a hole, the :attr:`Step.hole` of that
    step. Its domain is the context with each hole replaced by the domain of
    its step and its codomain the same with their codomains.

    Parameters:
        context : The diagram with holes.

    Example
    -------
    >>> from discopy.monoidal import Box, Diagram, Interchange, Ty
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> step = Interchange(f, g)
    >>> layer = Layer(f @ x >> step.hole)
    >>> assert type(layer) is Layer[Diagram]
    >>> print(layer, "|", layer.dom, "|", layer.cod)
    f @ x >> Interchange(f, g) | f @ x >> x @ f >> g @ x \
| f @ x >> g @ x >> x @ f
    >>> assert layer.steps == (step, )
    """
    ob = classproperty(lambda cls: cls.category)
    serialised_attrs = ("context", )

    def __new__(cls, context: monoidal.Diagram):
        return object.__new__(
            cls if cls.category is not None else cls[context.factory])

    @classmethod
    def checked_params(cls) -> tuple:
        """ None, a layer is checked with its rewrites, see
        :meth:`Rewrite.checked_params`. """
        return ()

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
        # pylint: disable-next=import-outside-toplevel  # it imports this
        from discopy.monoidal import Functor
        return Functor(
            ob_map=lambda x: x.to_drawing(),
            ar_map=lambda box: box.data.frame(t)
            if isinstance(box.data, Step) else box.to_drawing(),
            dom=self.context.factory, cod=Drawing)(self.context)

    def dagger(self) -> Layer:
        """ The layer in the same context, with each step reversed. """
        return type(self)(self.fill(lambda step: step.dagger().hole))

    def __eq__(self, other):
        return isinstance(other, Layer) and self.context == other.context

    def __hash__(self):
        return hash(self.context)

    def __repr__(self):
        return f"rewrite.Layer({self.context!r})"


@factory
class Rewrite[category](cat.Arrow, TestableGeneric):
    """
    A rewrite of diagrams of the ``category`` is a sequence of composable
    :class:`Layer` from a diagram ``dom`` to a diagram ``cod``. Each level
    of the hierarchy has its own, its :attr:`Diagram.rewrite_factory
    <discopy.monoidal.Diagram.rewrite_factory>`, with the steps of its
    axioms, e.g. :class:`discopy.monoidal.Interchange` from
    :mod:`discopy.monoidal` up and :class:`discopy.braided.Slide` from
    :mod:`discopy.braided` up.

    Parameters:
        inside : The layers of the rewrite.
        dom : The diagram it rewrites.
        cod : The diagram it rewrites into.

    Example
    -------
    >>> from discopy import braided, monoidal
    >>> x = monoidal.Ty('x')
    >>> f = monoidal.Box('f', x, x)
    >>> assert monoidal.Rewrite is Rewrite[monoidal.Diagram]
    >>> assert type(monoidal.Interchange(f, f)).ar is monoidal.Rewrite
    >>> braided.Rewrite.id(f)
    Traceback (most recent call last):
    ...
    TypeError: Expected braided.Diagram, got monoidal.Box instead.
    """
    ob = classproperty(lambda cls: cls.category)
    generator_factory = classproperty(lambda cls: Layer[cls.category])
    box_factory = classproperty(lambda cls: Step[cls.category])

    @classmethod
    def cast_ob(cls, x):
        """ A diagram of the category, which is not cast but checked. """
        assert_isinstance(x, cls.ob)
        return x

    @classmethod
    def checked_params(cls) -> tuple:
        """
        None: the property matrix does not check rewrites yet, since a step
        goes between parallel diagrams only and the matrix draws each object
        of a law with its own boundary, not from one hom.

        >>> from discopy.testable import Testable
        >>> from discopy.monoidal import Diagram
        >>> assert Rewrite[Diagram] not in Testable.enrolled()
        """
        return ()

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
        >>> from discopy.monoidal import Box, Interchange, Ty
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
            tuple(self.generator_factory(around(layer.context))
                  for layer in self.inside),
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
        >>> from discopy.monoidal import Box, Interchange, Ty
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
            return rewrite.inside + (self.generator_factory(rewrite.cod), )\
                * (length - len(rewrite.inside))
        return self.ar(tuple(
            self.generator_factory(this.context @ that.context)
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
        >>> from discopy.monoidal import Box, Interchange, Rewrite, Ty
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


class Step[category](cat.Box, Rewrite):
    """
    A step is a generating rewrite from a diagram ``dom`` to a diagram
    ``cod`` with the same domain and codomain, of the ``category`` of
    ``dom`` when not given. It is the rewrite of one layer with one hole,
    its :attr:`hole`, a box named after the step with the step as ``data``.

    A step is drawn by :meth:`frame`. Its subclasses are the steps the axioms
    give, each in the module of its axiom with its own frames, e.g.
    :class:`discopy.monoidal.Interchange` and
    :class:`discopy.braided.Slide`, and a user's own 2-cells. Such a subclass
    takes the category from its arguments, see :meth:`category_of`.

    Parameters:
        name : The name of the step.
        dom : The diagram it rewrites.
        cod : The diagram it rewrites into.
        data : Extra data in the step, default is ``None``.
        is_dagger : Whether the step is reversed.

    Example
    -------
    >>> from discopy.monoidal import Box, Diagram, Rewrite, Ty
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> step = Step('commute', f >> g, g >> f)
    >>> assert isinstance(step, Rewrite) and type(step) is Step[Diagram]
    >>> print(step.dagger(), "|", step.dagger().dom)
    commute[::-1] | g >> f
    >>> Step('wrong', f, Box('h', x, x @ x))
    Traceback (most recent call last):
    ...
    discopy.utils.AxiomError: Expected parallel arrows, got f: x -> x and \
h: x -> x @ x instead.
    """
    @classmethod
    def subscript_bases(cls, category) -> tuple[type, ...]:
        """ The rewrites of the category, of which a step is one. """
        return (Rewrite[category], )

    @staticmethod
    def category_of(*args, **kwargs) -> type:
        """ The category of the diagram ``dom``, the second argument. """
        return (args[1] if len(args) > 1 else kwargs["dom"]).factory

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls if cls.category is not None
                              else cls[cls.category_of(*args, **kwargs)])

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def __init__(self, name: str, dom: monoidal.Diagram,  # cat.Box's
                 cod: monoidal.Diagram, data=None, is_dagger=False):
        cat.Box.__init__(self, name, dom, cod, data, is_dagger)
        if (self.dom.dom, self.dom.cod) != (self.cod.dom, self.cod.cod):
            raise AxiomError(messages.NOT_PARALLEL.format(*(
                f"{d}: {d.dom} -> {d.cod}" for d in (self.dom, self.cod))))
        self.hole = self.dom.factory.box_factory(
            str(self), self.dom.dom, self.dom.cod, data=self)
        Rewrite.__init__(self, (Layer[self.category](self.hole), ),
                         self.dom, self.cod, _scan=False)

    def frame(self, t: float) -> Drawing:
        """
        The drawing a fraction ``t`` of the way from ``dom`` to ``cod``, by
        :meth:`discopy.drawing.Drawing.interpolate`, which raises unless
        they are one interchanger or one slide apart.

        Parameters:
            t : The fraction of the way, from ``0`` to ``1``.
        """
        return self.dom.to_drawing().interpolate(self.cod.to_drawing(), t)
