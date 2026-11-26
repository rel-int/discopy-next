# -*- coding: utf-8 -*-

"""
The free closed monoidal feedback category, i.e. closed monoidal diagrams
with feedback, whose wires carry a time step, see :mod:`discopy.feedback`.

Its delay is a monoidal endofunctor but not a closed one: an exponential is
a wire of its own, delayed as a whole, leaving its base and its exponent
alone, and the delay of an evaluation evaluates the delayed exponential.

>>> x, y = Ty('x'), Ty('y')
>>> assert (x << y).delay() != x.delay() << y.delay()
>>> ev = Eval(x << y, left=True).delay()
>>> assert ev.dom == (x << y).delay() @ y.delay() and ev.cod == x.d

With the delay the identity, it is a traced closed monoidal category: a
curry is the trace of a coevaluation, see :meth:`Diagram.to_compact`, and a
:mod:`discopy.pivotal` category is one where the exponential of ``x`` is
its adjoint. Its planar diagrams have no swap, so there is no Int here:
:cite:t:`Hasegawa09` shows that a balanced traced category is closed if and
only if its embedding into :class:`discopy.interaction.Int` has a right
adjoint, a theorem that starts at :mod:`discopy.feedback.closed.balanced`.

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
    Exp
    Over
    Under
    Diagram
    Box
    Timed
    Eval
    Coeval
    Curry
    Feedback
    Feedforward
    Trace
    FollowedBy
    Head
    Tail
    Sum
    Functor
"""

from __future__ import annotations

from functools import cached_property
from typing import ClassVar

from discopy import closed, cmap, feedback, hypergraph, messages, monoidal
from discopy.abc import DaggerMonoidalCategory
from discopy.cat import factory
# pylint: disable-next=unused-import
from discopy.closed.monoidal import Equation  # noqa: F401
from discopy.drawing import Drawing
from discopy.utils import factory_name


class Wire(feedback.Wire, closed.monoidal.Wire):
    """ A closed feedback object is a closed object with a time step. """


HeadOb, TailOb = feedback.structural_wires(Wire)


@factory
class Ty(closed.monoidal.Ty, feedback.Ty):
    """ A closed feedback type is a closed type and a feedback type. """
    generator_factory = Wire


class Exp(Wire, closed.monoidal.Exp):
    """ An exponential object with a time step, delayed as a whole. """
    ob = Ty

    @cached_property
    def name(self):  # pylint: disable=method-hidden  # deleted by __init__
        "The name of an exponential is its undelayed print, read when needed."
        return str(self.reset())

    def __repr__(self):
        return factory_name(type(self))\
            + f"({self.base!r}, {self.exponent!r}{self.repr_time_step()})"


class Over(Exp, closed.monoidal.Over):
    """
    An exponent over a base with a time step, called with ``<<``.

    >>> x, y = Ty('x'), Ty('y')
    >>> assert str((x << y).delay()) == "(x << y).d"
    >>> assert eval(repr((x << y).delay())) == (x << y).delay()
    """


class Under(Exp, closed.monoidal.Under):
    """ A base under an exponent with a time step, called with ``>>``. """


@factory
class Diagram(closed.monoidal.Diagram, feedback.Diagram):
    """
    A closed feedback diagram is a closed diagram and a feedback diagram.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    trace_factory: ClassVar[type["Trace"]]  # ty: ignore[invalid-type-form]
    ob = Ty
    layer_factory = feedback.Layer

    def to_compact(self) -> Diagram:
        """
        Open the curry bubbles into coevaluation and feedback, which stays
        a :class:`Diagram` as a closed feedback category is traced: each curry
        becomes its argument followed by :class:`Coeval`, traced over the
        curried wires.

        Example
        -------
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box("f", x @ y, z)
        >>> assert f.curry().to_compact() == (
        ...     f >> Coeval(z << y, left=True)).trace()
        """
        def image(box):
            if box.time_step:
                return image(box.reset()).delay(box.time_step)
            if isinstance(box, Curry) and not box.n:
                return box.arg.to_compact()
            if isinstance(box, Curry):
                return (box.arg.to_compact() >> self.coeval_factory(
                    box.cod, left=box.left)).trace(
                        len(box.cod.exponent), left=not box.left)
            return box
        result = self.id(self.dom)
        for layer in self.inside:
            for box, offset in layer.boxes_and_offsets:
                cod = result.cod
                result >>= cod[:offset] @ image(box)\
                    @ cod[offset + len(box.dom):]
        return result

    def to_drawing(self):
        return monoidal.Diagram.to_drawing(self, functor_factory=Functor)


class Box(closed.monoidal.Box, feedback.Box, Diagram):
    """ A closed feedback box is a closed box and a feedback box. """
    serialised_attrs = feedback.Box.serialised_attrs


class Timed(Box):
    """
    A closed feedback box built from arguments rather than a name and a
    boundary, with a time step of its own: delayed, it keeps its arguments
    and delays its boundary, as the delay is not a closed functor. E.g. the
    delay of the evaluation of ``x << y`` evaluates ``(x << y).delay()`` at
    ``y.delay()`` into ``x.delay()``, where ``x.delay() << y.delay()`` is
    another exponential.

    Parameters:
        args : The arguments of the box at time step zero.
        time_step : The number of times the box has been delayed.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> ev = Eval(x << y, left=True).delay()
    >>> assert ev.x == x << y and ev == Eval(x << y, True, time_step=1)
    """
    def __init__(self, *args, time_step: int = 0, **params):
        super().__init__(*args, **params)
        if time_step:
            self._time_step = time_step
            monoidal.Box.__init__(self, self.name, self.dom.delay(time_step),
                                  self.cod.delay(time_step), args=self.args,
                                  method=self.method)

    time_step = property(lambda self: self._time_step)

    def delay(self, n_steps=1):
        return type(self)(
            *self.__getnewargs__(), time_step=self.time_step + n_steps)

    def dagger(self):
        return super().dagger().delay(self.time_step)

    def repr_time_step(self) -> str:
        """ The keyword of the time step, as :meth:`__repr__` reads it. """
        return f", time_step={self.time_step}" if self.time_step else ""

    def __repr__(self):
        return super().__repr__()[:-1] + self.repr_time_step() + ")"

    def __str__(self):
        if not self.time_step:
            return super().__str__()
        return str(self.reset()) + feedback.str_delayed(self.time_step)

    def to_tree(self):
        tree = super().to_tree()
        if self.time_step:
            tree['time_step'] = self.time_step
        return tree

    @classmethod
    def from_tree(cls, tree):
        return super().from_tree(tree).delay(tree.get('time_step', 0))


class Eval(Timed, closed.monoidal.Eval):
    """
    The evaluation of an exponential type, with a time step.

    >>> x, y = Ty('x'), Ty('y')
    >>> print(Eval(y >> x, left=False).delay())
    Eval(y >> x).d
    """
    serialised_attrs = closed.monoidal.Eval.serialised_attrs + ("time_step", )

    def __getnewargs__(self):
        """ The arguments of the constructor, which a delay reads. """
        return self.x, self.left


class Coeval(Timed, closed.monoidal.Coeval):
    """ The coevaluation of an exponential type, with a time step. """
    serialised_attrs = closed.monoidal.Coeval.serialised_attrs\
        + ("time_step", )
    __getnewargs__ = Eval.__getnewargs__


class Curry(Timed, closed.monoidal.Curry):
    """ The currying of a closed feedback diagram, with a time step. """
    serialised_attrs = closed.monoidal.Curry.serialised_attrs\
        + ("time_step", )

    def __getnewargs__(self):
        """ The arguments of the constructor, which a delay reads. """
        return self.arg, self.n, self.left

    def to_drawing(self):
        """
        A delayed curry is drawn as the delay of its argument followed by the
        delay of the coevaluation, traced over the delayed wires it curries.
        """
        if not self.time_step:
            return super().to_drawing()
        if not self.n:
            return self.arg.delay(self.time_step).to_drawing()
        coeval = self.coeval_factory(self.reset().cod, left=self.left)
        return (self.arg >> coeval).delay(self.time_step).to_drawing()\
            .trace(self.n, left=not self.left)


class Sum(closed.monoidal.Sum, feedback.Sum, Box):
    """ A closed feedback sum is a closed and a feedback sum. """


class Functor(closed.monoidal.Functor, feedback.Functor):
    """
    A closed feedback functor is a closed and a feedback functor: a delayed
    wire or box goes through the delay of the codomain.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to
            :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if self.cod is not Drawing and isinstance(other, (Wire, Box))\
                and other.time_step:
            return feedback.Functor.__call__(self, other)
        return super().__call__(other)


Diagram.functor_factory = Functor
CMap = cmap.CMap[Diagram]
Hypergraph = hypergraph.Hypergraph[Diagram]
for law in (
        "dagger_involution", "dagger_contravariance", "dagger_monoidality"):
    setattr(Hypergraph, law, getattr(DaggerMonoidalCategory, law).failing(
        messages.CURRY_HAS_NO_DAGGER))
Feedback, Feedforward, Trace, FollowedBy, Head, Tail = \
    feedback.structural_boxes(Diagram, Box)
Diagram.box_factory = Box
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
Diagram.coeval_factory = Coeval
Diagram.sum_factory = Sum
Ty.over_factory, Ty.under_factory = Over, Under

Id = Diagram.id


Bubble = Box.from_args
