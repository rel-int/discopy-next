# -*- coding: utf-8 -*-

"""
The free balanced closed feedback category, i.e. closed monoidal diagrams
with braids, twists and feedback, whose wires carry a time step, see
:mod:`~discopy.feedback.closed.monoidal`.

With the delay the identity, it is a balanced traced closed category, the
setting of :cite:t:`Hasegawa09`: a balanced traced category is closed if and
only if its embedding into :class:`~discopy.interaction.Int` has a right
adjoint (his Theorem 4.1). It is the balanced, and not merely braided,
notion of trace: :cite:t:`Selinger10` (Section 5.5) calls braided traced
categories somewhat unnatural, coherent only up to regular isotopy, and
balanced ones (Section 5.6) those coherent up to framed isotopy. A
:mod:`~discopy.ribbon` category is one where the exponential of ``x`` is its
dual: the braided linear lambda calculus of :cite:t:`Hasegawa21` and the
framed tangles are both interpreted in the ribbon combinatory algebras of
:cite:t:`HasegawaLechenne24`, which are the reflexive objects of a ribbon
category, and balanced combinatory algebras with a trace combinator.
There is no plain ``closed.balanced`` level: its braids and twists come from
:mod:`~discopy.feedback.balanced`.

>>> x, y = Ty('x'), Ty('y')
>>> f = Box('f', x @ y, x @ y)
>>> assert Braid(x, x).trace().dom == x and Twist(x << y).dom == x << y
>>> assert f.curry(2).to_compact() == (f >> Coeval(
...     (x @ y) << (x @ y), left=True)).trace(2)

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Eval
    Coeval
    Curry
    Braid
    Twist
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

from discopy import cmap, feedback, hypergraph, messages
from discopy.abc import BalancedTracedCategory, DaggerMonoidalCategory
from discopy.cat import factory
from discopy.feedback import balanced
from discopy.feedback.closed import monoidal as planar
# pylint: disable-next=unused-import
from discopy.feedback.closed.monoidal import (  # noqa: F401
    Wire, HeadOb, TailOb, Ty, Exp, Over, Under, Equation)


@factory
# pylint: disable-next=inconsistent-mro  # astroid's, not Python's
class Diagram(planar.Diagram, balanced.Diagram, BalancedTracedCategory):
    """
    A balanced closed feedback diagram is a closed feedback diagram and a
    balanced feedback diagram.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty
    yanking = BalancedTracedCategory.yanking.failing(messages.FREE_YANKING)


class Box(planar.Box, balanced.Box, Diagram):
    """ A balanced closed feedback box is a closed and a balanced box. """


class Eval(planar.Eval, Box):
    """ The evaluation of an exponential type, with a time step. """


class Coeval(planar.Coeval, Box):
    """ The coevaluation of an exponential type, with a time step. """


class Curry(planar.Curry, Box):
    """ The currying of a balanced closed feedback diagram. """


class Braid(balanced.Braid, Box):
    """ A braid in a balanced closed feedback diagram. """


class Twist(balanced.Twist, Box):
    """ A twist in a balanced closed feedback diagram. """


class Sum(planar.Sum, balanced.Sum, Box):
    """ A balanced closed feedback sum. """


class Functor(planar.Functor, balanced.Functor):
    """
    A balanced closed feedback functor is a closed feedback functor and a
    balanced feedback functor.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to
            :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram


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
Diagram.braid_factory = Braid
Diagram.twist_factory = Twist
Diagram.sum_factory = Sum

Id = Diagram.id


Bubble = Box.from_args
