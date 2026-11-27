# -*- coding: utf-8 -*-

"""
The free feedback category, i.e. diagrams with delayed feedback loops, and
the free traced category as the case where the loop is not delayed.

We follow the definition of :cite:t:`DiLavoreEtAl22`, whose head and tail of
streams live in :mod:`~discopy.dataflow`.
A traced category is a feedback category whose delay is the identity: the
wires of a feedback diagram carry a time step, and a :class:`Trace` is a
:class:`Feedback` whose memory is not delayed.

This package is the monoidal level, :mod:`~discopy.feedback.cat` the free
category with a delay and no tensor, and its other submodules the levels of
the hierarchy with feedback, each subclassing the ordinary module of the same
name: :mod:`~discopy.feedback.balanced`, :mod:`~discopy.feedback.symmetric`
for the usual notion of trace, :mod:`~discopy.feedback.markov` for the syntax
of streams, :mod:`~discopy.feedback.comarkov` and the package
:mod:`~discopy.feedback.closed`. The ordinary modules are not feedback
categories, except :mod:`~discopy.pivotal` and the levels below it, whose
wires still carry a time step. Note that these diagrams are planar, so that
e.g. pivotal diagrams are traced in this sense. See :mod:`~discopy.stream` for
the interpretation of streams, the main example of a feedback category.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    cat
    balanced
    symmetric
    markov
    comarkov
    closed

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Wire
    Ty
    Nat
    Layer
    Diagram
    Box
    Feedback
    Feedforward
    Trace
    Sum
    Functor
    TermBase
    Variable
    Constant
    Application
    Kappa

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        structural_boxes

Axioms
------
A feedback category is a monoidal category with a monoidal endofunctor
:meth:`Diagram.delay`, shortened to `.d` and a method :meth:`Diagram.feedback`
of the following shape:

>>> x, y, m = map(Ty, "xym")
>>> f = Box('f', x @ m.delay(), y @ m)
>>> Equation(f, f.feedback(), symbol="$\\\\mapsto$").draw(
...     doctest="docs/_static/feedback/feedback-operator.svg")

.. image:: /_static/feedback/feedback-operator.svg
    :align: center

The :class:`Trace` is a right trace by default:

>>> z = Ty('z')
>>> g = Box("g", x @ z, y @ z)
>>> assert g.trace() == g.feedback() == g.feedback(mem=z) == Trace(g)
>>> assert str(g.trace()) == "Trace(g)"
>>> Equation(g, g.trace(), symbol="$\\\\mapsto$").draw(
...     doctest='docs/_static/traced/right-trace.svg')

.. image:: /_static/traced/right-trace.svg
    :align: center

and a left trace with ``left=True``:

>>> h = Box("h", z @ x, z @ y)
>>> Equation(h, h.trace(left=True), symbol="$\\\\mapsto$").draw(
...     doctest='docs/_static/traced/left-trace.svg')

.. image:: /_static/traced/left-trace.svg
    :align: center

The laws are stated once on :class:`~discopy.abc.FeedbackCategory`, those of
a traced category being the same laws with the delay the identity, plus
yanking from :class:`~discopy.abc.TracedCategory`. Under each law, feedback
comes first, checked on the nose where it can be and otherwise drawn and
checked up to a functor into streams, then the trace on symmetric diagrams,
checked on the nose for vanishing and joining and otherwise with
:class:`~discopy.feedback.symmetric.Equation`, i.e. up to hypergraph
isomorphism: equality of planar traced diagrams is not implemented.

>>> from discopy import stream
>>> from discopy.feedback import symmetric
>>> F0 = Functor(
...     lambda x: stream.Ty.sequence(x.generator.name), cod=stream.Stream)
>>> F = Functor(
...     F0, lambda f: stream.Stream.sequence(f.name, F0(f.dom), F0(f.cod)),
...     cod=stream.Stream)
>>> all_eq = lambda xs: len(set(xs)) == 1
>>> eq_up_to_F = lambda *fs, n=2: all_eq(F(f).unroll(n).now for f in fs)

>>> s = symmetric.Ty('s')
>>> t, k = symmetric.Box('t', s @ s, s @ s), symmetric.Box('k', s, s)

Vanishing
=========
Feeding back the unit does nothing.

>>> assert Box('f', x, y).feedback(mem=Ty()) == Box('f', x, y)
>>> assert t.trace(n=0) == t == t.trace(n=0, left=True)

Joining
=======
Feeding back a tensor is feeding back its factors in turn.

>>> f = Box('f', x @ (m @ m).delay(), y @ m @ m)
>>> assert f.feedback(mem=m @ m) == f.feedback().feedback()
>>> assert t.trace(n=2) == t.trace().trace()
>>> assert t.trace(n=2, left=True) == t.trace(left=True).trace(left=True)

Strength
========
Feedback commutes with tensoring a morphism on the side, also called
superposing.

>>> f, g = Box('f', x @ m.delay(), y @ m), Box('g', x, y)
>>> strength = Equation(g @ f.feedback(), (g @ f).feedback())
>>> assert eq_up_to_F(*strength.terms)
>>> strength.draw(
...     doctest='docs/_static/feedback/strength.svg', wire_labels=False)

.. image:: /_static/feedback/strength.svg
    :align: center

>>> assert symmetric.Equation((s @ t).trace(), s @ t.trace())
>>> assert symmetric.Equation(
...     (t @ s).trace(left=True), t.trace(left=True) @ s)

Tightening
==========
Feedback is natural in its domain and its codomain.

>>> tightening_left = symmetric.Equation(
...     (s @ k >> t >> s @ k).trace(left=True),
...     k >> t.trace(left=True) >> k)
>>> tightening_left.draw(
...     doctest='docs/_static/traced/tightening-left.svg', wire_labels=False)

.. image:: /_static/traced/tightening-left.svg
    :align: center

>>> tightening_right = symmetric.Equation(
...     (k @ s >> t >> k @ s).trace(),
...     k >> t.trace() >> k)
>>> tightening_right.draw(
...     doctest='docs/_static/traced/tightening-right.svg',
...     wire_labels=False)

.. image:: /_static/traced/tightening-right.svg
    :align: center

>>> assert tightening_left and tightening_right

Sliding
=======
A morphism on the memory slides along the feedback, delayed. For feedback,
this can only be checked up to extensional equivalence of streams.

>>> n = Ty("n")
>>> h = Box('h', m, n)  # assume h is an isomorphism
>>> f = Box('f', x @ n.d, y @ m)
>>> sliding = Equation((f >> y @ h).feedback(), (x @ h.d >> f).feedback())
>>> sliding.draw(
...     doctest='docs/_static/feedback/sliding.svg', wire_labels=False)

.. image:: /_static/feedback/sliding.svg
    :align: center

>>> LHS, RHS = sliding.terms
>>> assert list(map(str, F(LHS).unroll(2).now.dom)) == ["x0", "x1", "x2"]
>>> eq = Equation(*map(lambda f: F(f).unroll(2).now, sliding.terms),
...     symbol="$\\\\sim$").draw(
...         doctest='docs/_static/feedback/slide-unroll.svg')
>>> assert symmetric.Equation(
...     F(LHS).unroll(2).now,
...     F(RHS).unroll(2).now
...         >> F(y).unroll(2).now @ F(h).later.later.now)

.. image:: /_static/feedback/slide-unroll.svg
    :align: center

For a trace, the delay is the identity and the morphism slides without
delay, up to the hypergraph isomorphism of
:class:`~discopy.feedback.symmetric.Equation`:

>>> sliding_left = symmetric.Equation(
...     (t >> k @ s).trace(left=True),
...     (k @ s >> t).trace(left=True))
>>> sliding_left.draw(
...     doctest='docs/_static/traced/sliding-left.svg', wire_labels=False)

.. image:: /_static/traced/sliding-left.svg
    :align: center

>>> sliding_right = symmetric.Equation(
...     (t >> s @ k).trace(),
...     (s @ k >> t).trace())
>>> sliding_right.draw(
...     doctest='docs/_static/traced/sliding-right.svg', wire_labels=False)

.. image:: /_static/traced/sliding-right.svg
    :align: center

>>> assert sliding_left and sliding_right

Yanking
=======
The one law of a traced category on top of feedback: the trace of the swap
of an object with itself is its identity.

>>> yanking = symmetric.Equation(
...     symmetric.Swap(s, s).trace(left=True),
...     symmetric.Id(s),
...     symmetric.Swap(s, s).trace())
>>> yanking.draw(
...     doctest='docs/_static/traced/yanking.svg',
...     wire_labels=False, figsize=(4, 1))

.. image:: /_static/traced/yanking.svg
    :align: center

>>> assert yanking
"""

from __future__ import annotations

from copy import copy
from collections.abc import Mapping
from typing import ClassVar, Self

from discopy import monoidal, cmap, hypergraph, messages
from discopy.abc import (
    DaggerCategory, FeedbackCategory, MonoidalCategory, TracedCategory)
from discopy.config import BOX_DRAWING_ATTRIBUTES
from discopy.monoidal import (  # noqa: F401  pylint: disable=unused-import
    Equation)
from discopy.cat import kappa_calculus
from discopy.utils import (
    deprecated_alias,
    factory, factory_name, assert_isinstance, AxiomError, from_tree,
    subclass,
)


def str_delayed(time_step: int | None):
    if not time_step:
        return ""
    return time_step * ".d" if time_step <= 3 else f".delay({time_step})"


class Wire(monoidal.Wire):
    """
    A feedback object is an object with a `time_step`.

    The time step is a keyword, so that a wire of a level below, e.g. a
    :class:`~discopy.pivotal.Wire`, passes its own arguments through.

    Parameters:
        name : The name of the object.
        time_step : The number of times the object has been delayed.

    Example
    -------
    >>> from discopy import feedback
    >>> x = Wire('x')
    >>> assert x.delay().time_step == 1 and x.delay().reset() == x
    >>> assert eval(repr(x.delay(2))) == x.delay(2)
    """
    time_step = 0

    def __init__(self, name: str, *args, time_step: int = 0, **params):
        assert_isinstance(time_step, int)
        if time_step < 0:
            raise NotImplementedError
        super().__init__(name, *args, **params)
        self.time_step = time_step

    def __setstate__(self, state):
        state.setdefault('time_step', 0)
        super().__setstate__(state)

    def delay(self, n_steps=1) -> Wire:
        """ The delay of a feedback object. """
        if self.time_step + n_steps < 0:
            raise NotImplementedError
        result = copy(self)
        result.time_step = self.time_step + n_steps
        return result

    def reset(self) -> Wire:
        """ Reset an object to time step zero, used in :class:`Functor`. """
        return self.delay(-self.time_step)

    def dagger(self) -> Wire:
        result = super().dagger()
        result.time_step = self.time_step
        return result

    def __eq__(self, other):
        return super().__eq__(other) and self.time_step == other.time_step

    def __hash__(self):
        return hash((super().__hash__(), self.time_step))

    def repr_time_step(self) -> str:
        """ The keyword arguments of the time step, as :meth:`__repr__` reads
        them. """
        return f", time_step={self.time_step}" if self.time_step else ""

    def __repr__(self):
        return monoidal.Wire.__repr__(self)[:-1] + self.repr_time_step() + ")"

    def __str__(self):
        return super().__str__() + str_delayed(self.time_step)

    def to_tree(self):
        tree = super().to_tree()
        if self.time_step:
            tree['time_step'] = self.time_step
        return tree

    @classmethod
    def from_tree(cls, tree):
        result = super().from_tree(tree)
        result.time_step = tree.get('time_step', 0)
        return result

    @property
    def d(self):
        """ Syntactic sugar for meth:`delay`. """
        return self.delay()


@factory
class Ty(monoidal.Ty):
    """ A feedback type is a monoidal type with a `delay`. """
    generator_factory = Wire

    def delay(self, n_steps=1):
        """ The delay of a feedback type by `n_steps`. """
        return self.factory(*(x.delay(n_steps) for x in self.inside))

    d = Wire.d


@factory
class Nat(monoidal.Nat, Ty):
    """
    A feedback ``Nat`` is a natural number ``n`` seen as a feedback type of
    length ``n``, i.e. the objects of a PRO below feedback. Its generator
    has no time step, so it has no delay.

    Parameters
    ----------
    n : int
        The natural number.
    """


class Layer(monoidal.Layer):
    """ A feedback layer is a monoidal layer with a `delay` method. """
    def delay(self, n_steps=1):
        """ The delay of each box and type in the layer. """
        return type(self)(*(x.delay(n_steps) for x in self.boxes_or_types),
                          normalise=False)


@factory
class Diagram(monoidal.Diagram, TracedCategory):
    """
    A feedback diagram is a monoidal diagram with a :meth:`delay` endofunctor
    and a :meth:`feedback` operator, building :class:`Feedback` boxes over a
    delayed memory and :class:`Trace` boxes over an undelayed one.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty
    layer_factory = Layer

    box_factory: ClassVar[type["Box"]]
    functor_factory: ClassVar[type["Functor"]]
    feedback_factory: ClassVar[type["Feedback"]]
    trace_factory: ClassVar[type["Trace"]]

    yanking = TracedCategory.yanking.inapplicable(
        messages.PLANAR_YANKING)

    def delay(self, n_steps=1):
        """ The delay of a feedback diagram. """
        dom, cod = self.dom.delay(n_steps), self.cod.delay(n_steps)
        inside = tuple(box.delay(n_steps) for box in self.inside)
        return type(self)(inside, dom, cod, _scan=False)

    def feedback(self, n=None, left=False, *, dom=None, cod=None, mem=None):
        """
        Feed a memory ``mem`` back, one wire at a time, the last first or the
        first when ``left``: through :attr:`trace_factory` when the wire fed
        back is the memory itself and :attr:`feedback_factory` when it is its
        delay. The keyword form ``dom``, ``cod``, ``mem`` is primary, the
        positional ``n`` its fallback, see
        :meth:`~discopy.abc.FeedbackCategory.split_memory`.

        >>> x, y = Ty('x'), Ty('y')
        >>> f = Box('f', x @ (x @ y).delay(), x @ x @ y)
        >>> fb = f.feedback(dom=x, cod=x, mem=x @ y)
        >>> assert fb == f.feedback(mem=y).feedback(mem=x) == f.feedback(2)
        >>> assert (fb.dom, fb.cod) == (x, x)
        """
        dom, cod, mem = self.split_memory(
            n, left, dom=dom, cod=cod, mem=mem)
        size = len(mem)
        fed = self.dom[:size] if left else self.dom[len(self.dom) - size:]
        if size == 0:
            return self
        if size > 1 and left:
            return self.feedback(
                dom=fed[1:] @ dom, cod=mem[1:] @ cod, mem=mem[:1], left=True
            ).feedback(dom=dom, cod=cod, mem=mem[1:], left=True)
        if size > 1:
            return self.feedback(
                dom=dom @ fed[:-1], cod=cod @ mem[:-1], mem=mem[-1:]
            ).feedback(dom=dom, cod=cod, mem=mem[:-1])
        if fed == mem or not isinstance(mem, Ty):
            if (self.dom, self.cod) != ((fed @ dom, mem @ cod) if left
                                        else (dom @ fed, cod @ mem)):
                raise AxiomError(messages.NOT_TRACEABLE.format(self, mem))
            return self.trace_factory(self, left)
        return self.feedback_factory(
            self, dom=dom, cod=cod, mem=mem, left=left)

    def trace(self, n=None, left=False, *, dom=None, cod=None, mem=None):
        """
        The :meth:`feedback` of a memory that is not delayed, in either form.

        Parameters:
            n : The number of output wires to feedback into inputs.
            left : Whether to trace the wires on the left or right.
            dom : The domain of the trace.
            cod : The codomain of the trace.
            mem : The memory to trace over.

        Example
        -------
        >>> from discopy.monoidal import Equation as Eq
        >>> x = Ty('x')
        >>> f = Box('f', x @ x, x @ x)
        >>> LHS, RHS = f.trace(left=True), f.trace(left=False)
        >>> Eq(Eq(LHS, f, symbol="$\\\\mapsfrom$"),
        ...     RHS, symbol="$\\\\mapsto$").draw(
        ...         doctest="docs/_static/traced/trace.svg")

        .. image:: /_static/traced/trace.svg
        """
        dom, cod, mem = self.split_memory(
            n, left, dom=dom, cod=cod, mem=mem)
        fed = self.dom[:len(mem)] if left\
            else self.dom[len(self.dom) - len(mem):]
        if fed != mem:
            raise AxiomError(messages.NOT_TRACEABLE.format(fed, mem))
        return self.feedback(dom=dom, cod=cod, mem=mem, left=left)

    @property
    def time_step(self) -> int | None:
        """
        The time step of a diagram is defined only if it is in fact a box.

        This is used for checking equality between boxes and diagrams.

        Example
        -------
        >>> f = Box('f', 'x', 'y')
        >>> assert f.delay(42).time_step == 42
        """
        if len(self) != 1 or self != self.boxes[0]:
            raise ValueError
        return self.boxes[0].time_step

    def to_drawing(self):
        return monoidal.Diagram.to_drawing(self, functor_factory=Functor)

    d = Wire.d

    trace_naturality_left = TracedCategory.trace_naturality_left.failing(
        messages.NO_PLANAR_NORMAL_FORM)
    trace_naturality_right = TracedCategory.trace_naturality_right.failing(
        messages.NO_PLANAR_NORMAL_FORM)
    trace_dinaturality_left = TracedCategory.trace_dinaturality_left.failing(
        messages.NO_PLANAR_NORMAL_FORM)
    trace_dinaturality_right = TracedCategory.trace_dinaturality_right.failing(
        messages.NO_PLANAR_NORMAL_FORM)
    trace_superposing_left = TracedCategory.trace_superposing_left.failing(
        messages.NO_PLANAR_NORMAL_FORM)
    trace_superposing_right = TracedCategory.trace_superposing_right.failing(
        messages.NO_PLANAR_NORMAL_FORM)
    dagger_involution_traced = DaggerCategory.dagger_involution.weaken(
        within=MonoidalCategory)
    dagger_contravariance_traced = DaggerCategory.dagger_contravariance\
        .weaken(within=MonoidalCategory)
    tightening = FeedbackCategory.tightening.failing(
        messages.FREE_FEEDBACK)
    sliding = FeedbackCategory.sliding.failing(
        messages.FREE_FEEDBACK)
    strength = FeedbackCategory.strength.failing(
        messages.FREE_FEEDBACK)


class Box(monoidal.Box, Diagram):
    """
    A feedback box is a monoidal box in a feedback diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
        time_step (int) : The number of times the box has been delayed, as
            :attr:`time_step` reads it, ``None`` for a structural box.
    """
    serialised_attrs = monoidal.Box.serialised_attrs + ("time_step", )

    _time_step = 0

    @property
    def time_step(self) -> int | None:
        """
        The number of times a box has been delayed, ``None`` for a structural
        box, e.g. a copy, a swap, a merge or a permutation: it has no time
        step of its own, its wires carry it, see :meth:`delay`.

        Example
        -------
        >>> from discopy.feedback.markov import Copy
        >>> x = Ty('x')
        >>> assert Box('f', x, x).delay(42).time_step == 42
        >>> copy = Copy(x).delay(42)
        >>> assert copy.time_step is None and copy.dom == x.delay(42)
        """
        return None if hasattr(self, "__getnewargs__") else self._time_step

    def __init__(self, *args, time_step: int = 0, **params):
        self._time_step = time_step
        super().__init__(*args, **params)

    @time_step.setter
    def time_step(self, value: int):
        self._time_step = value

    def to_drawing(self):
        result = super().to_drawing()
        if self.args:
            return result
        if result.box.drawing_name:
            result.box.drawing_name += str_delayed(self.time_step)
        return result

    def delay(self, n_steps=1):
        """
        The delay of a box: a box of a category whose objects are not made of
        wires, e.g. a tensor box over dimensions, is its own delay, as its
        objects are; a structural box, one whose ``__getnewargs__`` gives the
        arguments it is built from, e.g. a swap, a copy or a controlled gate,
        is built from their delays, those of its types and its diagrams; any
        other box from its delayed boundary, one more time step on, drawn as
        it is, e.g. a box with args, which stay as they are inside. A
        :class:`Sum` is the sum of the delays of its terms.

        Parameters:
            n_steps : The number of time steps to delay.
        """
        if not issubclass(self.ob.generator_factory, monoidal.Wire):
            return self
        arguments = getattr(self, "__getnewargs__", None)
        if arguments is not None:
            return type(self)(*(
                x.delay(n_steps) if isinstance(x, (Ty, Diagram)) else x
                for x in arguments()))
        box = self.rebuild(
            dom=self.dom.delay(n_steps), cod=self.cod.delay(n_steps),
            time_step=self.time_step + n_steps)
        vars(box).update(
            (attr, value) for attr, value in vars(self).items()
            if attr in BOX_DRAWING_ATTRIBUTES)
        return box

    def reset(self):
        """
        Reset a box to time step zero, used in :class:`Functor`: a structural
        box has no time step of its own to reset, it stays as it is.

        Example
        -------
        >>> from discopy.feedback.markov import Copy
        >>> x = Ty('x')
        >>> assert Box('f', x, x).delay(3).reset() == Box('f', x, x)
        >>> assert Copy(x).delay(3).reset() == Copy(x.delay(3))
        """
        if self.time_step is None:
            return self
        return self.delay(-self.time_step)

    def atoms(self) -> list:
        """
        The atoms of a box, each at time step zero, which the string of a
        delayed box delays, e.g. ``f.d``.

        >>> x = Ty('x')
        >>> assert Box('f', x, x).delay().atoms() == [Box('f', x, x), x, x]
        """
        return [atom.reset() if isinstance(atom, Box) and atom.time_step
                else atom for atom in super().atoms()]

    @classmethod
    def from_args(cls, *args: Diagram, dom: Ty | None = None,
                  cod: Ty | None = None, time_step: int = 0,
                  **params) -> Self:
        """
        The box with some ``args``, its boundary that of its args delayed
        ``time_step`` times unless given: a box with args delays as a box
        does, its args stay as they are inside.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> f = Box('f', x, y)
        >>> bubble = f.bubble().delay()
        >>> assert bubble.args == (f, )
        >>> assert (bubble.dom, bubble.cod) == (x.d, y.d)
        >>> assert bubble.time_step == 1 and bubble.reset() == f.bubble()
        >>> assert bubble == f.bubble(time_step=1) != f.delay().bubble()
        >>> print(bubble)
        (f).bubble().d
        """
        boundary = [
            given if given is not None
            else cls.common(args, attr).delay(time_step) if time_step
            else cls.common(args, attr)
            for attr, given in [("dom", dom), ("cod", cod)]]
        return super().from_args(
            *args, dom=boundary[0], cod=boundary[1], time_step=time_step,
            **params)

    def __str__(self):
        if self.args and self.time_step:
            return str(self.reset()) + str_delayed(self.time_step)
        return super().__str__() + str_delayed(self.time_step)

    def __repr__(self):
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return super().__repr__()[:-1] + time_step + ")"

    def to_tree(self) -> dict:
        """
        Serialise a feedback box, with the number of times it was delayed.

        >>> from discopy.utils import dumps, loads
        >>> f = Box('f', Ty('x'), Ty('y')).delay()
        >>> assert loads(dumps(f)) == f

        A delayed box with args is serialised as at time step zero, with its
        time step, since its args are not delayed.

        >>> bubble = Box('f', Ty('x'), Ty('y')).bubble().delay(2)
        >>> assert loads(dumps(bubble)) == bubble
        """
        if self.args and self.time_step:
            return self.reset().to_tree() | {'time_step': self.time_step}
        tree = super().to_tree()
        if self.time_step:
            tree['time_step'] = self.time_step
        return tree

    @classmethod
    def from_tree(cls, tree: dict) -> Box:
        if 'args' in tree and tree.get('time_step'):
            reset = {k: v for k, v in tree.items() if k != 'time_step'}
            return cls.from_tree(reset).delay(tree['time_step'])
        result = super().from_tree(tree)
        result._time_step = tree.get(  # pylint: disable=protected-access
            'time_step', 0)
        return result  # ty: ignore[invalid-return-type]

    def setoid(self):
        return super().setoid() + (self.time_step, )


class Feedback(Box):
    """
    Feedback is a bubble that takes a diagram from `dom @ mem.delay()` to
    `cod @ mem`, or from `mem.delay() @ dom` to `mem @ cod` when `left`, and
    returns a box from `dom` to `cod`.

    Parameters:
        arg : The diagram to feed back.
        dom : The domain of the feedback.
        cod : The codomain of the feedback.
        mem : The memory to feed back, the last wire of `arg.cod` by default.
        left : Whether to feed back on the left rather than the right.

    Examples
    --------
    >>> x, y, z = map(Ty, "xyz")
    >>> f = Box('f', x @ y.delay(), z @ y)
    >>> fb = f.feedback()
    >>> Equation(f, fb, symbol="$\\\\mapsto$").draw(
    ...     doctest="docs/_static/feedback/feedback-bubble.svg")

    .. image:: /_static/feedback/feedback-bubble.svg
        :align: center

    >>> from discopy import feedback
    >>> assert str(fb) == "Feedback(f)" and eval(repr(fb)) == fb
    """
    serialised_attrs = ("*args", "dom", "cod", "mem", "left")

    def __init__(self, arg: Diagram, dom=None, cod=None, mem=None, left=False):
        assert_isinstance(arg, self.ar)
        dom, cod, mem = arg.split_memory(
            left=left, dom=dom, cod=cod, mem=mem)
        expected = self.arg_boundary(dom, cod, mem, left)
        if (arg.dom, arg.cod) != expected:
            raise AxiomError(messages.NOT_FEEDBACK.format(mem, arg, expected))
        self.mem, self.left = mem, left
        name = f"{type(self).__name__}({arg}"\
            + ("" if len(mem) == 1 else f", mem={mem}")\
            + (", left=True" if left else "") + ")"
        monoidal.Box.__init__(self, name, dom, cod, args=(arg, ))

    @staticmethod
    def fed_back(mem: Ty) -> Ty:
        """ The input fed back for a memory ``mem``, its delay. """
        return mem.delay()

    @classmethod
    def arg_boundary(cls, dom: Ty, cod: Ty, mem: Ty, left: bool
                     ) -> tuple[Ty, Ty]:
        """
        The domain and codomain of the argument of a feedback from ``dom`` to
        ``cod`` over the memory ``mem``, on the left when ``left``.
        """
        fed = cls.fed_back(mem)
        return (fed @ dom, mem @ cod) if left else (dom @ fed, cod @ mem)

    @classmethod
    # pylint: disable-next=arguments-differ  # drawn with no boundary
    def strategy(cls, *, types=None):
        """
        Generate the feedback of a free box over an atomic memory, on
        either side, the trace of one for a :class:`Trace`.

        Parameters:
            types : The strategy to draw the types from, the memory one of
                them with one wire, see :meth:`atomic_strategy`.
        """
        from hypothesis import strategies as st

        types = cls.ob.strategy() if types is None else types

        def feedbacks(args):
            mem, dom, cod, left = args
            arg_dom, arg_cod = cls.arg_boundary(dom, cod, mem, left)
            return cls.ar.box_factory.strategy(
                types=types, dom=arg_dom, cod=arg_cod).map(
                    lambda arg: cls(arg, mem=mem, left=left))

        return st.tuples(
            cls.atomic_strategy(types), types, types, st.booleans()
        ).flatmap(feedbacks)

    def delay(self, n_steps=1):
        return type(self)(self.arg.delay(n_steps), mem=self.mem.delay(n_steps),
                          left=self.left)

    def __str__(self):
        return self.name

    def __repr__(self):
        mem = "" if len(self.mem) == 1 else f", mem={self.mem!r}"
        left = ", left=True" if self.left else ""
        return factory_name(type(self)) + f"({self.arg!r}{mem}{left})"

    def to_tree(self) -> dict:
        """
        Serialise a feedback by its argument, its memory and its side.

        >>> from discopy.utils import dumps, loads
        >>> x, y = Ty('x'), Ty('y')
        >>> fb = Box('f', x @ (x @ y).delay(), x @ x @ y).feedback(mem=x @ y)
        >>> assert loads(dumps(fb)) == fb
        """
        tree = {'factory': factory_name(type(self)),
                'arg': self.arg.to_tree(), 'dom': self.dom.to_tree(),
                'cod': self.cod.to_tree(), 'mem': self.mem.to_tree()}
        if self.left:
            tree['left'] = True
        return tree

    @classmethod
    def from_tree(cls, tree: dict) -> Feedback:
        arg, dom, cod, mem = (
            from_tree(tree[key]) if key in tree else None
            for key in ('arg', 'dom', 'cod', 'mem'))
        return cls(
            arg,  # ty: ignore[invalid-argument-type]
            dom=dom, cod=cod, mem=mem, left='left' in tree)

    def dagger(self):
        """
        The dagger of a feedback is the :class:`Feedforward` of the dagger of
        its argument, which goes from ``cod @ mem`` to ``dom @ mem.delay()``.

        >>> x = Ty('x')
        >>> fb = Box('f', x @ x.delay(), x @ x).feedback()
        >>> assert isinstance(fb.dagger(), Feedforward)
        >>> assert fb.dagger().dagger() == fb
        """
        return self.ar.feedforward_factory(
            self.arg.dagger(), mem=self.mem, left=self.left)

    def rotate(self, left=False):
        """
        A feedback has no rotation, only a :class:`Trace` does: the half-turn
        of a delayed feedback would feed its memory forward in time.
        """
        raise AxiomError(messages.FEEDBACK_HAS_NO_ROTATION)

    def to_drawing(self):
        return self.arg.to_drawing().trace(len(self.mem), left=self.left)


class Feedforward(Box):
    """
    Feedforward is the dual of :class:`Feedback`, the bubble a feedback's
    dagger draws: it takes a diagram from `dom @ mem` to `cod @ mem.delay()`,
    or from `mem @ dom` to `mem.delay() @ cod` when `left`, and returns a box
    from `dom` to `cod`.

    Parameters:
        arg : The diagram to feed forward.
        dom : The domain of the feedforward.
        cod : The codomain of the feedforward.
        mem : The memory to feed forward, the last wire of `arg.dom` by
            default.
        left : Whether to feed forward on the left rather than the right.

    Example
    -------
    >>> from discopy import feedback
    >>> x, y, z = map(Ty, "xyz")
    >>> f = Box('f', z @ y, x @ y.delay())
    >>> ff = Feedforward(f)
    >>> assert (ff.dom, ff.cod, ff.mem) == (z, x, y)
    >>> assert str(ff) == "Feedforward(f)" and eval(repr(ff)) == ff
    >>> assert ff.dagger() == f.dagger().feedback()
    >>> assert ff.dagger().dagger() == ff
    """
    serialised_attrs = Feedback.serialised_attrs
    mem: Ty
    left: bool

    # pylint: disable-next=super-init-not-called  # Feedback's calls them
    def __init__(self, arg: Diagram, dom=None, cod=None, mem=None, left=False):
        assert_isinstance(arg, self.ar)
        if mem is None:
            mem = arg.dom[:1] if left else arg.dom[-1:]
        # pylint: disable-next=non-parent-init-called  # the dual's boundary
        Feedback.__init__(
            self,  # ty: ignore[invalid-argument-type]
            arg, dom, cod, mem, left)

    fed_back = staticmethod(Feedback.fed_back)

    @classmethod
    def arg_boundary(cls, dom: Ty, cod: Ty, mem: Ty, left: bool
                     ) -> tuple[Ty, Ty]:
        """
        The domain and codomain of the argument of a feedforward, those of
        the argument of the feedback from ``cod`` to ``dom`` swapped.
        """
        return Feedback.arg_boundary.__func__(
            cls,  # ty: ignore[invalid-argument-type]
            cod, dom, mem, left)[::-1]

    strategy = classmethod(Feedback.strategy.__func__)
    delay = Feedback.delay
    __str__, __repr__ = Feedback.__str__, Feedback.__repr__
    to_tree = Feedback.to_tree
    from_tree = classmethod(Feedback.from_tree.__func__)
    to_drawing = Feedback.to_drawing

    def dagger(self):
        """ The dagger of a feedforward, the :class:`Feedback` it dualises. """
        return self.ar.feedback_factory(
            self.arg.dagger(), mem=self.mem, left=self.left)

    def rotate(self, left=False):
        """ A feedforward has no rotation, as a :class:`Feedback` has none. """
        raise AxiomError(messages.FEEDBACK_HAS_NO_ROTATION)


class Trace(Feedback, Feedforward):
    """
    A trace is a :class:`Feedback` whose memory is not delayed, i.e. a diagram
    ``arg`` with an output wire fed back into an input.

    Parameters:
        arg : The diagram to trace.
        left : Whether to trace the wires on the left or right.

    See also
    --------
    :meth:`Diagram.trace`

    Example
    -------
    >>> from discopy import feedback
    >>> x = Ty('x')
    >>> trace = Box('f', x @ x, x @ x).trace()
    >>> assert eval(repr(trace)) == trace
    >>> assert trace.delay() == Box('f', x @ x, x @ x).delay().trace()
    """
    def __init__(self, arg: Diagram, left=False, *, dom=None, cod=None,
                 mem=None):
        super().__init__(arg, dom=dom, cod=cod, mem=mem, left=left)

    @staticmethod
    def fed_back(mem: Ty) -> Ty:
        """ The input fed back for a memory ``mem``, itself. """
        return mem

    def delay(self, n_steps=1):
        return self.arg.delay(n_steps).trace(len(self.mem), left=self.left)

    def dagger(self):
        return self.arg.dagger().trace(len(self.mem), left=self.left)

    def rotate(self, left=False):
        """
        The rotation of a trace, the trace of the rotation of its argument on
        the other side: a left trace rotates to a right trace and back.

        >>> from discopy import pivotal
        >>> x = pivotal.Ty('x')
        >>> f = pivotal.Box('f', x @ x, x @ x)
        >>> assert pivotal.Trace(f).rotate() == f.rotate().trace(left=True)

        A trace in a category with no rotation has none either:

        >>> x = Ty('x')
        >>> Box('f', x @ x, x @ x).trace().rotate()
        Traceback (most recent call last):
        ...
        discopy.utils.AxiomError: A trace rotates only in a pivotal category...
        """
        if not hasattr(self.arg, "rotate"):
            raise AxiomError(messages.TRACE_HAS_NO_ROTATION.format(
                factory_name(self.ar)))
        return self.arg.rotate(left=left).trace(
            len(self.mem), left=not self.left)

    def to_drawing(self):
        return self.ar.to_drawing(self)


class Sum(monoidal.Sum, Box):
    """
    A feedback sum is a sum of feedback diagrams, over which delay
    distributes.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, g, h = Box('f', x, y), Box('g', x, y), Box('h', y, y)
    >>> assert (f + g).delay() == f.delay() + g.delay()
    >>> assert ((f + g) >> h).delay() == (f + g).delay() >> h.delay()
    """
    def delay(self, n_steps=1):
        """ The delay of a sum, the sum of the delays of its terms. """
        return self.sum_factory(
            tuple(term.delay(n_steps) for term in self.terms),
            self.dom.delay(n_steps), self.cod.delay(n_steps))


class Functor(monoidal.Functor):
    """
    A feedback functor is a monoidal functor that preserves delay, feedback
    and traces: a delayed box that is not a key of a mapping
    :code:`ar_map` goes to the delay of the image of the box, which is the
    image itself in a codomain that overrides :code:`delay` with the
    identity, e.g. :class:`~discopy.tensor.Tensor`; a callable
    :code:`ar_map` gets the delayed box.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.

    Example
    -------
    >>> x, y, m = map(Ty, "xym")
    >>> f = Box('f', x @ m.d, y @ m)
    >>> g = Box('g', y.d @ m.d.d, x.d @ m.d)
    >>> F = Functor({x: y.d, y: x.d, m: m.d}, {f: g})

    >>> assert F(f.delay()) == F(f).delay()
    >>> assert F(f.feedback()) == F(f).feedback()
    >>> assert F(f.feedback().dagger()) == F(f).feedback().dagger()

    Let's compute the golden ratio as the trace of a Python function, i.e.
    the nested radical iterated from ``float()``, see
    :meth:`python.Function.trace <discopy.python.Function.trace>`.

    >>> from math import sqrt
    >>> from discopy import python
    >>> x = Ty('$\\\\mathbb{R}$')
    >>> f = Box('$\\\\lambda x . (x, \\\\sqrt{1 + x})$', x, x @ x)
    >>> g = Box('$\\\\frac{1 + \\\\sqrt{5}}{2}$', Ty(), x)
    >>> F = Functor(
    ...     ob_map={x: float},
    ...     ar_map={
    ...         f: lambda x: (x, sqrt(1 + x)),
    ...         g: lambda: (1 + sqrt(5)) / 2},
    ...     cod=python.Function)
    >>> assert F(f.trace())() == F(g)()

    >>> Equation(f.trace(), g).draw(doctest="docs/_static/traced/golden.svg")

    .. image:: /_static/traced/golden.svg
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Trace):
            arg = other.arg
            while len(arg.boxes) == 1 and isinstance(arg.boxes[0], Trace)\
                    and (arg.dom, arg.cod) == (
                        arg.boxes[0].dom, arg.boxes[0].cod)\
                    and arg.boxes[0].left == other.left:
                arg = arg.boxes[0].arg
            n = len(self(arg.dom)) - len(self(other.dom))
            return self.cod.trace(self(arg), n, left=other.left)
        if isinstance(other, (Wire, Box)) and other.time_step:
            cod, mapping = (self.cod.ob, self.ob_map.mapping)\
                if isinstance(other, Wire) else (
                    self.cod.ar, self.ar_map.mapping)
            if getattr(cod, "delay", FeedbackCategory.delay)\
                    is not FeedbackCategory.delay\
                    and not (isinstance(mapping, Mapping)
                             and other in mapping):
                result = self(other.reset())
                for _ in range(other.time_step):
                    result = result.delay()
                return result
        if isinstance(other, Feedforward) and not isinstance(other, Feedback)\
                and issubclass(self.cod, DaggerCategory)\
                and self.cod.has_dagger():
            return self(other.dagger()).dagger()
        if isinstance(other, Feedback)\
                and issubclass(self.cod, FeedbackCategory)\
                and self.cod.delay is not FeedbackCategory.delay:
            left = {"left": True} if other.left else {}
            return self(other.arg).feedback(
                dom=self(other.dom), cod=self(other.cod),
                mem=self(other.mem), **left)
        if isinstance(other, Box) and other.time_step\
                and hasattr(self.cod.ar, "delay")\
                and isinstance(self.ar_map.mapping, Mapping)\
                and other not in self.ar_map.mapping:
            return self(other.reset()).delay(other.time_step)
        return super().__call__(other)


CMap = cmap.CMap[Diagram]

Diagram.box_factory = Box
Diagram.functor_factory = Functor
Diagram.feedback_factory = Feedback
Diagram.feedforward_factory = Feedforward
Diagram.trace_factory = Trace
Diagram.sum_factory = Sum
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id


STRUCTURAL_BOXES = {
    "Feedback": ("feedback_factory", "A feedback of {} diagrams."),
    "Feedforward": (
        "feedforward_factory", "A feedforward of {} diagrams."),
    "Trace": ("trace_factory", "A trace of {} diagrams."),
}


def structural_boxes(
        diagram: type, box: type
        ) -> tuple[type, type, type]:
    """
    The :class:`Feedback`, :class:`Feedforward` and :class:`Trace` of a level
    of the hierarchy, generated rather than written by hand: each subclasses
    the one of every level ``diagram`` subclasses and ``box``, lives in their
    module and is bound to the factory of ``diagram`` that builds it. A level
    whose trace is not a bubble, e.g. made of cups and caps in
    :mod:`~discopy.pivotal`, keeps its own :attr:`Diagram.trace_factory`.

    Parameters:
        diagram : The diagram class of the level.
        box : The box class of the level.

    Example
    -------
    >>> from discopy.feedback import markov, symmetric
    >>> assert issubclass(markov.Feedback, symmetric.Feedback)
    >>> assert issubclass(markov.Feedback, markov.Box)
    >>> assert markov.Diagram.feedback_factory is markov.Feedback
    >>> assert repr(markov.Trace).endswith("'discopy.feedback.markov.Trace'>")
    >>> assert issubclass(markov.Trace, markov.Feedforward)
    """
    parents = [base for base in diagram.__bases__ if issubclass(base, Diagram)]
    module, level = diagram.__module__, diagram.__module__.split('.')[-1]
    result = {}
    for name, (attr, doc) in STRUCTURAL_BOXES.items():
        own = [result["Feedback"], result["Feedforward"]]\
            if name == "Trace" else [box]
        bases = [getattr(parent, attr, None) for parent in parents]
        bases = [base for base in bases if isinstance(base, type)] or [next(
            vars(base)[attr] for base in diagram.__mro__
            if isinstance(vars(base).get(attr), type))]
        bases += own
        result[name] = subclass(name, bases, module, doc.format(level))
    for name, (attr, _) in STRUCTURAL_BOXES.items():
        if name != "Trace" or isinstance(diagram.trace_factory, type):
            setattr(diagram, attr, result[name])
    return result["Feedback"], result["Feedforward"], result["Trace"]


__getattr__ = deprecated_alias(__name__, {
    "Ob": "Wire",
    "Swap": "discopy.feedback.markov.Swap",
    "Permutation": "discopy.feedback.markov.Permutation",
    "Copy": "discopy.feedback.markov.Copy",
    "Discard": "discopy.feedback.markov.Discard",
})

TermBase, Variable, Constant, Application, Kappa = kappa_calculus(
    Diagram)
Term = Variable | Constant | Application | Kappa


Bubble = Box.from_args


# The levels with feedback subclass this one, so they are imported last.
# pylint: disable-next=wrong-import-position
from discopy.feedback import (  # noqa: E402,F401
    cat, balanced, symmetric, markov, comarkov, closed)
