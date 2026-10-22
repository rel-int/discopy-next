# -*- coding: utf-8 -*-

"""
The free feedback category, i.e. diagrams with delayed feedback loops, and
the free traced category as the case where the loop is not delayed.

We follow the definition of :cite:t:`DiLavoreEtAl22` with some extra structure
for the head and tail of streams with the :class:`FollowedBy` generator.
A traced category is a feedback category whose delay is the identity, so
every level of the hierarchy below this one, from :mod:`balanced` and
:mod:`pivotal` down, is a feedback category: its wires carry a time step,
a :class:`Trace` is a :class:`Feedback` whose memory is not delayed.

Note that these diagrams are planar, so that e.g. :mod:`pivotal` diagrams are
traced in this sense. See :mod:`symmetric` for the usual notion of trace,
:mod:`markov` for the syntax of streams and :mod:`discopy.stream` for its
interpretation, the main example of a feedback category.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Wire
    HeadOb
    TailOb
    Ty
    Nat
    Layer
    Diagram
    Box
    Feedback
    Trace
    FollowedBy
    Head
    Tail
    Sum
    Functor

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        generators
        wire_generators

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

The laws are stated once on :class:`discopy.abc.FeedbackCategory`:

Vanishing
=========

>>> assert Box('f', x, y).feedback(mem=Ty()) == Box('f', x, y)

Joining
=======

>>> f = Box('f', x @ (m @ m).delay(), y @ m @ m)
>>> assert f.feedback(mem=m @ m) == f.feedback().feedback()

Strength
========
This can only be checked up to a functor into streams.

>>> from discopy import stream
>>> F0 = Functor(
...     lambda x: stream.Ty.sequence(x.generator.name), cod=stream.Stream)
>>> F = Functor(
...     F0, lambda f: stream.Stream.sequence(f.name, F0(f.dom), F0(f.cod)),
...     cod=stream.Stream)
>>> all_eq = lambda xs: len(set(xs)) == 1
>>> eq_up_to_F = lambda *fs, n=2: all_eq(F(f).unroll(2).now for f in fs)

>>> f, g = Box('f', x @ m.delay(), y @ m), Box('g', x, y)
>>> strength = Equation(g @ f.feedback(), (g @ f).feedback())
>>> assert eq_up_to_F(*strength.terms)
>>> strength.draw(
...     doctest='docs/_static/feedback/strength.svg', wire_labels=False)

.. image:: /_static/feedback/strength.svg
    :align: center

Sliding
=======
This can only be checked up to extensional equivalence of streams.

>>> from discopy import symmetric
>>> n = Ty("n")
>>> h = Box('h', m, n)  # assume h is an isomorphism
>>> f = Box('f', x @ n.d, y @ m)
>>> sliding = Equation((f >> y @ h).feedback(), (x @ h.d >> f).feedback())
>>> sliding.draw(
...     doctest='docs/_static/feedback/sliding.svg', wire_labels=False)

.. image:: /_static/feedback/sliding.svg
    :align: center

>>> LHS, RHS = sliding.terms
>>> assert F(LHS).unroll(2).now.dom == symmetric.Ty("x0", "x1", "x2")
>>> eq = Equation(*map(lambda f: F(f).unroll(2).now, sliding.terms),
...     symbol="$\\\\sim$").draw(
...         doctest='docs/_static/feedback/slide-unroll.svg')
>>> assert symmetric.Equation(
...     F(LHS).unroll(2).now,
...     F(RHS).unroll(2).now
...         >> F(y).unroll(2).now @ F(h).later.later.now)

.. image:: /_static/feedback/slide-unroll.svg
    :align: center

Traces
------

A monoidal category is right-traced when it comes with an operator of shape:

>>> x, y, z = map(Ty, "xyz")
>>> f = Box("f", x @ z, y @ z)
>>> Equation(f, f.trace(), symbol="$\\\\mapsto$").draw(
...     doctest='docs/_static/traced/right-trace.svg')

It is left-traced when it comes with an operator of the following shape:

>>> g = Box("g", z @ x, z @ y)
>>> Equation(g, g.trace(left=True), symbol="$\\\\mapsto$").draw(
...     doctest='docs/_static/traced/left-trace.svg')

A trace is a feedback whose memory is not delayed, i.e. a feedback in a
category whose delay is the identity:

>>> assert f.trace() == f.feedback() == f.feedback(mem=z) == Trace(f)
>>> assert str(f.trace()) == "Trace(f)"

These are subjects to the axioms of feedback with yanking on top, the laws of
a traced category when it is braided. Note however that at the moment
equality of planar traced diagrams is not implemented, only symmetric traced.

>>> from discopy.symmetric import Ty, Box, Swap, Id, Equation
>>> x = Ty('x')
>>> f, g = Box('f', x @ x, x @ x), Box('g', x, x)

Vanishing
=========

>>> assert f.trace(n=0) == f == f.trace(n=0, left=True)
>>> assert f.trace(n=2) == f.trace().trace()
>>> assert f.trace(n=2, left=True) == f.trace(left=True).trace(left=True)

Superposing
===========

>>> assert Equation((x @ f).trace(), x @ f.trace())
>>> assert Equation((f @ x).trace(left=True), f.trace(left=True) @ x)

Yanking
=======

>>> yanking = Equation(
...     Swap(x, x).trace(left=True), Id(x), Swap(x, x).trace())
>>> yanking.draw(
...     doctest='docs/_static/traced/yanking.svg',
...     wire_labels=False, figsize=(4, 1))

.. image:: /_static/traced/yanking.svg
    :align: center

>>> assert yanking

Naturality
==========

>>> tightening_left = Equation(
...     (x @ g >> f >> x @ g).trace(left=True),
...     g >> f.trace(left=True) >> g)
>>> tightening_left.draw(
...     doctest='docs/_static/traced/tightening-left.svg', wire_labels=False)

.. image:: /_static/traced/tightening-left.svg
    :align: center

>>> tightening_right = Equation(
...     (g @ x >> f >> g @ x).trace(),
...     g >> f.trace() >> g)
>>> tightening_right.draw(
...     doctest='docs/_static/traced/tightening-right.svg',
...     wire_labels=False)

.. image:: /_static/traced/tightening-right.svg
    :align: center

>>> assert tightening_left and tightening_right

Dinaturality
============

>>> sliding_left = Equation(
...     (f >> g @ x).trace(left=True),
...     (g @ x >> f).trace(left=True))
>>> sliding_left.draw(
...     doctest='docs/_static/traced/sliding-left.svg', wire_labels=False)

.. image:: /_static/traced/sliding-left.svg
    :align: center

>>> sliding_right = Equation(
...     (f >> x @ g).trace(),
...     (x @ g >> f).trace())
>>> sliding_right.draw(
...     doctest='docs/_static/traced/sliding-right.svg', wire_labels=False)

.. image:: /_static/traced/sliding-right.svg
    :align: center

>>> assert sliding_left and sliding_right

Note
----
We also implement endofunctors :class:`Head` and :class:`Tail` together with an
isomorphism :class:`FollowedBy` between `x` and `x.head @ x.tail.delay()`.

This satisfies the following equations:

>>> from discopy.feedback import Ty
>>> x = Ty('x')
>>> assert x.head.head == x.head
>>> assert x.head.tail == Ty()
>>> assert x.delay().head == Ty()
>>> assert x.delay().tail == x

In the category of streams, this is just the identity.
"""

from __future__ import annotations

from copy import copy
from typing import ClassVar

from discopy import monoidal, cmap, hypergraph, messages
from discopy.abc import FeedbackCategory, TracedCategory
from discopy.monoidal import (  # noqa: F401  pylint: disable=unused-import
    Equation)
from discopy.utils import (
    deprecated_alias,
    factory, factory_name, assert_isinstance, AxiomError, from_tree,
)


def str_delayed(time_step: int):
    return time_step * ".d" if time_step <= 3 else f".delay({time_step})"


class Wire(monoidal.Wire):
    """
    A feedback object is an object with a `time_step` and an optional argument
    `is_constant` for whether the object is interpreted as a constant stream.

    The time step and constancy are keywords, so that a wire of a level below,
    e.g. a :class:`discopy.pivotal.Wire`, passes its own arguments through.

    Parameters:
        name : The name of the object.
        time_step : The number of times the object has been delayed.
        is_constant : Whether the object is a constant stream.

    Example
    -------
    >>> from discopy import feedback
    >>> x = Wire('x')
    >>> assert x.delay().time_step == 1 and x.delay().reset() == x
    >>> assert eval(repr(x.delay(2))) == x.delay(2)
    """
    time_step, is_constant = 0, True

    def __init__(self, name: str, *args, time_step: int = 0,
                 is_constant: bool = True, **params):
        assert_isinstance(time_step, int)
        assert_isinstance(is_constant, bool)
        if time_step < 0:
            raise NotImplementedError
        super().__init__(name, *args, **params)
        self.time_step, self.is_constant = time_step, is_constant

    def __setstate__(self, state):
        state.setdefault('time_step', 0)
        state.setdefault('is_constant', True)
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
        result.time_step, result.is_constant = self.time_step, self.is_constant
        return result

    @property
    def head(self) -> HeadOb | None:
        """ The :attr:`head_factory` of the object, or `None` if delayed. """
        return None if self.time_step else self.head_factory(self)

    @property
    def tail(self) -> Wire | None:
        """
        The :attr:`tail_factory` of the object, or `self` if `is_constant`.
        """
        return self.delay(-1) if self.time_step > 0 else (
            self if self.is_constant else self.tail_factory(self))

    def __eq__(self, other):
        return super().__eq__(other) and (
            self.time_step, self.is_constant) == (
                other.time_step, other.is_constant)

    def __hash__(self):
        return hash((super().__hash__(), self.time_step, self.is_constant))

    def repr_time_step(self) -> str:
        """ The keyword arguments of the time step, as :meth:`__repr__` reads
        them. """
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        is_constant = "" if self.is_constant else ", is_constant=False"
        return time_step + is_constant

    def __repr__(self):
        colours = "" if self.dom == self.cod == monoidal.transparent\
            else f", dom={self.dom!r}, cod={self.cod!r}"
        return factory_name(type(self))\
            + f"({self.name!r}{colours}{self.repr_time_step()})"

    def __str__(self):
        return super().__str__() + str_delayed(self.time_step)

    def to_tree(self):
        tree = super().to_tree()
        if self.time_step:
            tree['time_step'] = self.time_step
        if not self.is_constant:
            tree['is_constant'] = False
        return tree

    @classmethod
    def from_tree(cls, tree):
        result = super().from_tree(tree)
        result.time_step = tree.get('time_step', 0)
        result.is_constant = tree.get('is_constant', True)
        return result

    @property
    def d(self):
        """ Syntactic sugar for meth:`delay`. """
        return self.delay()


class HeadOb(Wire):
    """
    The head of a feedback object, interpreted as the first element of a stream
    followed by the constant stream on the empty type.

    Note the object `arg: Wire` cannot be itself a `HeadOb` or be delayed.
    """
    def __init__(self, arg: Wire, time_step: int = 0):
        assert_isinstance(arg, Wire)
        if isinstance(arg, HeadOb) or arg.time_step:
            raise ValueError
        self.arg = arg
        super().__init__(f"{arg}.head", time_step=time_step, is_constant=False)

    def __repr__(self):
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return factory_name(type(self)) + f"({repr(self.arg)}{time_step})"

    def delay(self, n_steps=1):
        return type(self)(self.arg, self.time_step + n_steps)

    def reset(self) -> HeadOb:
        return type(self)(self.arg)

    @property
    def l(self) -> HeadOb:
        """ The head of the left adjoint, for wires that have adjoints. """
        return type(self)(self.arg.l, self.time_step)

    @property
    def r(self) -> HeadOb:
        """ The head of the right adjoint, for wires that have adjoints. """
        return type(self)(self.arg.r, self.time_step)

    @property
    def head(self):
        return None if self.time_step else self

    @property
    def tail(self):
        return self.delay(-1) if self.time_step else None

    def to_tree(self) -> dict:
        """
        Serialise the head of an object by its argument and its time step.

        >>> from discopy.utils import dumps, loads
        >>> x = Ty('x')
        >>> assert loads(dumps(x.head.delay())) == x.head.delay()
        """
        tree = {'factory': factory_name(type(self)), 'arg': self.arg.to_tree()}
        if self.time_step:
            tree['time_step'] = self.time_step
        return tree

    @classmethod
    def from_tree(cls, tree: dict) -> HeadOb:
        return cls(from_tree(tree['arg']), tree.get('time_step', 0))


class TailOb(Wire):
    """
    The tail of a non-constant feedback object, interpreted as the stream
    starting from the second time step.

    Example
    -------
    >>> x = Wire('x', is_constant=False)
    >>> assert x.tail == TailOb(x)
    """
    def __init__(self, arg: Wire, time_step: int = 0):
        assert_isinstance(arg, Wire)
        if isinstance(arg, HeadOb) or arg.is_constant or arg.time_step > 0:
            raise ValueError
        self.arg = arg
        super().__init__(f"{arg}.tail", time_step=time_step, is_constant=False)

    delay, reset, __repr__ = HeadOb.delay, HeadOb.reset, HeadOb.__repr__
    to_tree, from_tree = HeadOb.to_tree, classmethod(HeadOb.from_tree.__func__)
    l, r = HeadOb.l, HeadOb.r


Wire.head_factory, Wire.tail_factory = HeadOb, TailOb


@factory
class Ty(monoidal.Ty):
    """ A feedback type is a monoidal type with `delay`, `head` and `tail`. """
    generator_factory = Wire

    def delay(self, n_steps=1):
        """ The delay of a feedback type by `n_steps`. """
        return type(self)(*(x.delay(n_steps) for x in self.inside))

    @property
    def head(self):
        """ The head of a feedback type, see :class:`HeadOb`. """
        return type(self)(*(x.head for x in self.inside if x.head))

    @property
    def tail(self):
        """ The tail of a feedback type, see :class:`TailOb`. """
        return type(self)(*(x.tail for x in self.inside if x.tail))

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

    feedback_factory: ClassVar[type["Feedback"]]
    trace_factory: ClassVar[type["Trace"]]
    head_factory: ClassVar[type["Head"]]
    tail_factory: ClassVar[type["Tail"]]
    followed_by_factory: ClassVar[type["FollowedBy"]]

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
        :meth:`discopy.abc.MonoidalCategory.split_memory`.

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
    def time_step(self) -> int:
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

    @property
    def head(self):
        """ Syntactic sugar for :class:`Head`. """
        return self.head_factory(self)

    @property
    def tail(self):
        """ Syntactic sugar for :class:`Tail`. """
        return self.tail_factory(self)

    @classmethod
    def followed_by(cls, dom: Ty) -> Diagram:
        """ Syntactic sugar for :class:`FollowedBy`. """
        return cls.followed_by_factory(dom)

    def to_drawing(self):
        return monoidal.Diagram.to_drawing(self, functor_factory=Functor)

    d = Wire.d


class Box(monoidal.Box, Diagram):
    """
    A feedback box is a monoidal box in a feedback diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
        time_step (int) : The number of times the box has been delayed.
    """

    _time_step, _args, _params = 0, (), {}
    time_step = property(lambda self: self._time_step)

    def __init__(self, *args, time_step: int = 0, **params):
        self._time_step, self._args, self._params = time_step, args[3:], params
        super().__init__(*args, **params)

    def to_drawing(self):
        result = super().to_drawing()
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
        other box from its delayed boundary, one more time step on.

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
        dom, cod = self.dom.delay(n_steps), self.cod.delay(n_steps)
        time_step = self.time_step + n_steps
        return type(self)(
            self.name, dom, cod, *self._args, time_step=time_step,
            **self._params)

    def reset(self):
        """ Reset a box to time step zero, used in :class:`Functor`. """
        return self.delay(-self.time_step)

    def dagger(self):
        result = super().dagger()
        result._time_step = self.time_step  # pylint: disable=protected-access
        return result

    def __str__(self):
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
        """
        tree = super().to_tree()
        if self.time_step:
            tree['time_step'] = self.time_step
        return tree

    @classmethod
    def from_tree(cls, tree: dict) -> Box:
        result = super().from_tree(tree)
        result._time_step = tree.get(  # pylint: disable=protected-access
            'time_step', 0)
        return result

    def setoid(self):
        return super().setoid() + (self.time_step, )


class Feedback(Box, monoidal.Bubble):
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
    def __init__(self, arg: Diagram, dom=None, cod=None, mem=None, left=False):
        assert_isinstance(arg, self.ar)
        dom, cod, mem = arg.split_memory(
            left=left, dom=dom, cod=cod, mem=mem)
        fed = self.fed_back(mem)
        expected = (fed @ dom, mem @ cod) if left else (dom @ fed, cod @ mem)
        if (arg.dom, arg.cod) != expected:
            raise AxiomError(messages.NOT_FEEDBACK.format(mem, arg, fed))
        self.mem, self.left = mem, left
        name = f"{type(self).__name__}({arg}"\
            + ("" if len(mem) == 1 else f", mem={mem}")\
            + (", left=True" if left else "") + ")"
        monoidal.Bubble.__init__(self, arg, dom=dom, cod=cod)
        monoidal.Box.__init__(self, name, dom, cod)

    @staticmethod
    def fed_back(mem: Ty) -> Ty:
        """ The input fed back for a memory ``mem``, its delay. """
        return mem.delay()

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
        return cls(arg, dom=dom, cod=cod, mem=mem, left='left' in tree)

    def to_drawing(self):
        return self.arg.to_drawing().trace(len(self.mem), left=self.left)


class Trace(Feedback):
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

    def to_drawing(self):
        return self.ar.to_drawing(self)


class Head(monoidal.Bubble, Box):
    """
    The head of a feedback diagram, interpreted as the first element followed
    by the identity stream on the empty type.

    Parameters:
        arg : The diagram to take the head of.
        time_step : The number of times the head has been delayed.
    """
    part = "head"

    def __init__(self, arg: Diagram, time_step=0):
        dom, cod = (
            getattr(x, self.part).delay(time_step) for x in [arg.dom, arg.cod])
        monoidal.Bubble.__init__(self, arg, dom=dom, cod=cod)
        Box.__init__(self, f"({arg}).{self.part}", self.dom, self.cod,
                     time_step=time_step)

    delay, reset, __repr__ = HeadOb.delay, HeadOb.reset, HeadOb.__repr__
    to_tree, from_tree = HeadOb.to_tree, classmethod(HeadOb.from_tree.__func__)
    __str__ = Box.__str__


class Tail(monoidal.Bubble, Box):
    """
    The tail of a feedback diagram, interpreted as the stream starting from the
    second time step with the identity on the empty type at the first step.

    Parameters:
        arg : The diagram to take the tail of.
        time_step : The number of times the tail has been delayed.
    """
    part = "tail"

    __init__ = Head.__init__
    delay, reset, __repr__ = HeadOb.delay, HeadOb.reset, HeadOb.__repr__
    to_tree, from_tree = HeadOb.to_tree, classmethod(HeadOb.from_tree.__func__)
    __str__ = Box.__str__


class FollowedBy(Box):
    """
    The isomorphism between `x.head @ x.tail.delay()` and `x`.

    In the category of streams, this is just the identity.

    Example
    -------
    >>> from discopy import stream
    >>> x = Ty(Wire('x', is_constant=False))
    >>> FollowedBy(x).draw(doctest="docs/_static/feedback/followed-by.svg")

    .. image:: /_static/feedback/followed-by.svg
        :align: center

    >>> F = Functor({x: stream.Ty.sequence('x')}, cod=stream.Stream)
    >>> X, Xh, Xtd = map(F, (x, x.head, x.tail.delay()))
    >>> for xh, xtd in [(Xh.now, Xtd.now),
    ...                 (Xh.later.now, Xtd.later.now),
    ...                 (Xh.later.later.now, Xtd.later.later.now)]:
    ...     print(f"({xh}, {xtd})")
    (x0, Ty())
    (Ty(), x1)
    (Ty(), x2)
    >>> eq_up_to_F = lambda f, g: F(f).unroll(2).now == F(g).unroll(2).now
    >>> assert eq_up_to_F(FollowedBy(x), Id(x))
    """
    def __init__(self, arg: Ty, is_dagger=False, time_step=0):
        self.arg = arg
        dagger_name = ", is_dagger=True" if is_dagger else ""
        name = f"FollowedBy({arg}{dagger_name})"
        dom, cod = arg.head @ arg.tail.delay(), arg
        dom, cod = (cod, dom) if is_dagger else (dom, cod)
        dom, cod = [x.delay(time_step) for x in (dom, cod)]
        super().__init__(
            name, dom, cod, time_step=time_step, is_dagger=is_dagger)

    def __repr__(self):
        is_dagger = ", is_dagger=True" if self.is_dagger else ""
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return factory_name(type(self))\
            + f"({repr(self.arg)}{is_dagger}{time_step})"

    def delay(self, n_steps=1):
        return type(self)(self.arg, self.is_dagger, self.time_step + n_steps)

    def reset(self):
        return type(self)(self.arg, self.is_dagger)

    def dagger(self):
        return type(self)(self.arg, not self.is_dagger, self.time_step)

    def to_tree(self) -> dict:
        """
        Serialise by the type, the direction and the time step.

        >>> from discopy.utils import dumps, loads
        >>> x = Ty(Wire('x', is_constant=False))
        >>> fby = FollowedBy(x, is_dagger=True).delay()
        >>> assert loads(dumps(fby)) == fby
        """
        tree = {'factory': factory_name(type(self)), 'arg': self.arg.to_tree()}
        if self.is_dagger:
            tree['is_dagger'] = True
        if self.time_step:
            tree['time_step'] = self.time_step
        return tree

    @classmethod
    def from_tree(cls, tree: dict) -> FollowedBy:
        return cls(from_tree(tree['arg']), tree.get('is_dagger', False),
                   tree.get('time_step', 0))


class Sum(monoidal.Sum, Box):
    """
    A feedback sum is a sum of feedback diagrams.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(monoidal.Functor):
    """
    A feedback functor is a monoidal functor that preserves delay, feedback
    and traces.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.

    Example
    -------
    >>> x, y, m = [Ty(Wire(n, is_constant=False)) for n in "xym"]
    >>> f = Box('f', x @ m.d, y @ m)
    >>> g = Box('g', y.d @ m.d.d, x.d @ m.d)
    >>> F = Functor({x: y.d, y: x.d, m: m.d}, {f: g})

    >>> assert F(f.delay()) == F(f).delay()
    >>> assert F(f.feedback()) == F(f).feedback()
    >>> assert F(x.head) == F(x).head and F(x.tail) == F(x).tail
    >>> assert F(FollowedBy(x)) == FollowedBy(F(x))
    >>> assert F(f.head) == F(f).head and F(f.tail) == F(f).tail

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
            n = len(self(other.arg.dom)) - len(self(other.dom))
            return self.cod.trace(self(other.arg), n, left=other.left)
        if isinstance(other, (Wire, Box)) and other.time_step:
            cod = self.cod.ob if isinstance(other, Wire) else self.cod.ar
            if getattr(cod, "delay", FeedbackCategory.delay)\
                    is not FeedbackCategory.delay:
                result = self(other.reset())
                for _ in range(other.time_step):
                    result = result.delay()
                return result
        if isinstance(other, (HeadOb, TailOb, Head, Tail)):
            cod = self.cod if isinstance(
                other, (Head, Tail)) else self.cod.ob
            attr = "head" if isinstance(other, (HeadOb, Head)) else "tail"
            if hasattr(cod, attr):
                return getattr(self(other.arg), attr)
        if isinstance(
                other, FollowedBy) and hasattr(self.cod, "followed_by"):
            arg = other.dom if other.is_dagger else other.cod
            return self.cod.followed_by(self(arg))
        if isinstance(other, Feedback) and hasattr(self.cod, "feedback"):
            left = {"left": True} if other.left else {}
            return self(other.arg).feedback(
                dom=self(other.dom), cod=self(other.cod),
                mem=self(other.mem), **left)
        return super().__call__(other)


CMap = cmap.CMap[Diagram]

Diagram.box_factory = Box
Diagram.functor_factory = Functor
Diagram.feedback_factory = Feedback
Diagram.trace_factory = Trace
Diagram.head_factory, Diagram.tail_factory = Head, Tail
Diagram.followed_by_factory = FollowedBy
Diagram.sum_factory = Sum
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id


GENERATORS = {
    "Feedback": ("feedback_factory", "A feedback of {} diagrams."),
    "Trace": ("trace_factory", "A trace of {} diagrams."),
    "FollowedBy": ("followed_by_factory", "A followed-by in {} diagrams."),
    "Head": ("head_factory", "The head of a {} diagram."),
    "Tail": ("tail_factory", "The tail of a {} diagram."),
}


def subclass(name: str, bases: list[type], module: str, doc: str) -> type:
    """
    The class ``name`` of ``module`` with the given bases, less those another
    base subclasses, so that it prints, pickles and reads back from its tree
    as a class written in that module would.
    """
    bases = tuple(dict.fromkeys(
        base for base in bases if not any(
            other is not base and issubclass(other, base) for other in bases)))
    return type(name, bases, {
        "__module__": module, "__qualname__": name, "__doc__": doc})


def generators(
        diagram: type, box: type) -> tuple[type, type, type, type, type]:
    """
    The :class:`Feedback`, :class:`Trace`, :class:`FollowedBy`, :class:`Head`
    and :class:`Tail` of a level of the hierarchy, generated rather than
    written by hand: each subclasses the one of every level ``diagram``
    subclasses and ``box``, lives in their module and is bound to the
    factory of ``diagram`` that builds it. A level whose trace is not a
    bubble, e.g. made of cups and caps in :mod:`discopy.pivotal`, keeps its
    own :attr:`Diagram.trace_factory`.

    Parameters:
        diagram : The diagram class of the level.
        box : The box class of the level.

    Example
    -------
    >>> from discopy import markov, symmetric
    >>> assert issubclass(markov.Head, symmetric.Head)
    >>> assert issubclass(markov.Head, markov.Box)
    >>> assert markov.Diagram.head_factory is markov.Head
    >>> assert repr(markov.Trace).endswith("'discopy.markov.Trace'>")
    """
    parents = [base for base in diagram.__bases__ if issubclass(base, Diagram)]
    module, level = diagram.__module__, diagram.__module__.split('.')[-1]
    result = {}
    for name, (attr, doc) in GENERATORS.items():
        own = result["Feedback"] if name == "Trace" else box
        bases = [getattr(parent, attr) for parent in parents]
        bases = [base for base in bases if isinstance(base, type)] + [own]
        result[name] = subclass(name, bases, module, doc.format(level))
    for name, (attr, _) in GENERATORS.items():
        if name != "Trace" or isinstance(diagram.trace_factory, type):
            setattr(diagram, attr, result[name])
    return (result["Feedback"], result["Trace"], result["FollowedBy"],
            result["Head"], result["Tail"])


def wire_generators(wire: type) -> tuple[type, type]:
    """
    The :class:`HeadOb` and :class:`TailOb` of a wire class below feedback,
    generated as :func:`generators` does for boxes, so that the head of a
    wire is a wire of its own level, and bound to its factories.

    Parameters:
        wire : The wire class of the level.

    Example
    -------
    >>> from discopy import pivotal
    >>> x = pivotal.Ty('x')
    >>> assert x.head.inside[0] == pivotal.HeadOb(pivotal.Wire('x'))
    >>> assert x.head.l == x.l.head and x.head.r.r == x.head
    """
    parents = [base for base in wire.__bases__ if issubclass(base, Wire)]
    module, level = wire.__module__, wire.__module__.split('.')[-1]
    head, tail = (
        subclass(name, [getattr(parent, attr) for parent in parents] + [wire],
                 module, f"The {part} of a {level} wire.")
        for name, attr, part in (
            ("HeadOb", "head_factory", "head"),
            ("TailOb", "tail_factory", "tail")))
    wire.head_factory, wire.tail_factory = head, tail
    return head, tail


__getattr__ = deprecated_alias(__name__, {
    "Ob": "Wire",
    "Swap": "discopy.markov.Swap",
    "Permutation": "discopy.markov.Permutation",
    "Copy": "discopy.markov.Copy",
    "Discard": "discopy.markov.Discard",
})
