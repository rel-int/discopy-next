# -*- coding: utf-8 -*-

"""
The free symmetric closed feedback category, i.e. symmetric closed diagrams
with feedback, whose wires carry a time step, see
:mod:`~discopy.feedback.closed.monoidal`.

With the delay the identity, it is a traced symmetric monoidal closed
category, of which :mod:`~discopy.compact` is the case where the exponential
of ``x`` is its dual: :cite:t:`HajgatoHasegawa13` show that a traced
:math:`*`-autonomous category is compact closed. It is the setting of the
geometry of interaction, :cite:t:`AbramskyEtAl02`, where
:class:`~discopy.interaction.Int` takes a traced category to a compact one.

>>> x, y = Ty('x'), Ty('y')
>>> f = Box('f', x @ y, x)
>>> assert Swap(x, x).trace().dom == x
>>> assert (Swap(y, x) >> f).curry().to_compact() == (
...     Swap(y, x) >> f >> Coeval(x << x, left=True)).trace()

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
    Diagram
    Box
    Eval
    Coeval
    Curry
    Swap
    Permutation
    Feedback
    Feedforward
    Trace
    FollowedBy
    Head
    Tail
    Wait
    Sum
    Functor
"""

from __future__ import annotations

from functools import cached_property

from discopy import closed, cmap, feedback, hypergraph, messages, symmetric
from discopy.abc import (
    BalancedTracedCategory, DaggerMonoidalCategory, TracedCategory)
from discopy.cat import factory
from discopy.feedback.closed import balanced
from discopy.feedback.closed import monoidal as planar
# pylint: disable-next=unused-import
from discopy.symmetric import Equation  # noqa: F401
from discopy.utils import factory_name


class Wire(planar.Wire, closed.symmetric.Wire):
    """ A symmetric closed feedback object, with a time step. """


HeadOb, TailOb = feedback.structural_wires(Wire)


@factory
class Ty(closed.symmetric.Ty, planar.Ty):
    """
    A symmetric closed feedback type is a symmetric closed type and a
    feedback type.
    """
    generator_factory = Wire


class Exp(Wire, closed.symmetric.Exp):
    """
    An exponential object with a time step, delayed as a whole.

    >>> x, y = Ty('x'), Ty('y')
    >>> assert eval(repr((x ** y).delay())) == (x ** y).delay()
    >>> assert str((x ** y).delay()) == "(y >> x).d"
    """
    ob = Ty

    @cached_property
    def name(self):  # pylint: disable=method-hidden  # deleted by __init__
        "The name of an exponential is its undelayed print, read when needed."
        return str(self.reset())

    def __str__(self):
        return closed.symmetric.Exp.__str__(self)\
            + feedback.str_delayed(self.time_step)

    @property
    def drawing_name(self) -> str:
        return closed.symmetric.Exp.drawing_name.fget(self)\
            + feedback.str_delayed(self.time_step)

    def __repr__(self):
        return factory_name(type(self))\
            + f"({self.base!r}, {self.exponent!r}{self.repr_time_step()})"


@factory
# pylint: disable-next=inconsistent-mro  # astroid's, not Python's
class Diagram(
        closed.symmetric.Diagram, balanced.Diagram,
        feedback.symmetric.Diagram):
    """
    A symmetric closed feedback diagram is a symmetric closed diagram and a
    balanced closed feedback diagram.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty
    layer_factory = feedback.symmetric.Layer
    yanking = BalancedTracedCategory.yanking.modulo(
        symmetric.Diagram.to_hypergraph)
    trace_naturality_left = TracedCategory.trace_naturality_left
    trace_naturality_right = TracedCategory.trace_naturality_right
    trace_dinaturality_left = TracedCategory.trace_dinaturality_left
    trace_dinaturality_right = TracedCategory.trace_dinaturality_right
    trace_superposing_left = TracedCategory.trace_superposing_left
    trace_superposing_right = TracedCategory.trace_superposing_right
    currying_symmetry = closed.symmetric.Diagram.currying_symmetry
    twist_factory = vars(symmetric.Diagram)["twist_factory"]
    braid_factory = vars(symmetric.Diagram)["braid_factory"]

    to_compact = planar.Diagram.to_compact
    to_drawing = planar.Diagram.to_drawing


class Box(balanced.Box, closed.symmetric.Box, feedback.symmetric.Box,
          Diagram):
    """ A symmetric closed feedback box. """
    serialised_attrs = feedback.Box.serialised_attrs


class Eval(balanced.Eval, closed.symmetric.Eval, Box):
    """ The evaluation of an exponential type, with a time step. """
    serialised_attrs = closed.symmetric.Eval.serialised_attrs\
        + ("time_step", )
    __str__ = balanced.Eval.__str__


class Coeval(balanced.Coeval, closed.symmetric.Coeval, Box):
    """ The coevaluation of an exponential type, with a time step. """
    serialised_attrs = closed.symmetric.Coeval.serialised_attrs\
        + ("time_step", )


class Curry(balanced.Curry, closed.symmetric.Curry, Box):
    """ The currying of a symmetric closed feedback diagram. """


class Permutation(closed.symmetric.Permutation,
                  feedback.symmetric.Permutation, Box):
    """ A permutation in a symmetric closed feedback diagram. """


class Swap(Permutation, closed.symmetric.Swap, feedback.symmetric.Swap,
           Box):
    """ A swap in a symmetric closed feedback diagram. """


class Sum(closed.symmetric.Sum, balanced.Sum, feedback.symmetric.Sum, Box):
    """ A symmetric closed feedback sum. """


class Functor(closed.symmetric.Functor, balanced.Functor,
              feedback.symmetric.Functor):
    """
    A symmetric closed feedback functor.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to
            :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram
    __call__ = planar.Functor.__call__


Diagram.functor_factory = Functor
CMap = cmap.CMap[Diagram]
Hypergraph = hypergraph.Hypergraph[Diagram]
for law in (
        "dagger_involution", "dagger_contravariance", "dagger_monoidality"):
    setattr(Hypergraph, law, getattr(
        DaggerMonoidalCategory, law).failing(messages.CURRY_HAS_NO_DAGGER))
Feedback, Feedforward, Trace, FollowedBy, Head, Tail, Wait = \
    feedback.structural_boxes(Diagram, Box)
Diagram.box_factory = Box
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
Diagram.coeval_factory = Coeval
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.sum_factory = Sum
Ty.under_factory = Ty.over_factory = staticmethod(Exp)

Id = Diagram.id


Bubble = Box.from_args
