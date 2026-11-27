# -*- coding: utf-8 -*-

"""
The free balanced feedback category, i.e. balanced diagrams with feedback,
whose wires carry a time step, see :mod:`discopy.feedback`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Braid
    Twist
    DualRailBraid
    DualRailTwist
    Feedback
    Feedforward
    Trace
    Sum
    Functor
    DualRail

Note
----
Not every balanced category embeds faithfully into a traced one (see the
`nLab`_), but the free balanced category does have the cancellation property
and embeds faithfully into the free balanced traced category.

.. _nLab: https://ncatlab.org/nlab/show/traced+monoidal+category

The trace of a braid is not its twist on the nose, and a braid delays as
its wires do.

>>> x = Ty('x')
>>> assert Braid(x, x).trace() != Twist(x) and Braid(x, x).trace().dom == x
>>> assert Braid(x, x).delay() == Braid(x.d, x.d)
"""

from __future__ import annotations

from discopy import balanced, cmap, feedback, hypergraph, messages
from discopy.abc import BalancedTracedCategory
from discopy.balanced import (  # noqa: F401  pylint: disable=unused-import
    Equation)
from discopy.cat import factory, kappa_calculus
from discopy.feedback import (  # noqa: F401  pylint: disable=unused-import
    Wire, Ty, Nat, Layer)


@factory
class Diagram(balanced.Diagram, feedback.Diagram, BalancedTracedCategory):
    """
    A balanced feedback diagram is a balanced diagram and a feedback diagram.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty
    layer_factory = Layer
    yanking = BalancedTracedCategory.yanking.failing(messages.FREE_YANKING)


class Box(balanced.Box, feedback.Box, Diagram):
    """
    A balanced feedback box is a balanced box and a feedback box.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
        time_step (int) : The number of times the box has been delayed.
    """


class Braid(balanced.Braid, Box):
    """ A braid in a balanced feedback diagram. """


class Twist(balanced.Twist, Box):
    """ A twist in a balanced feedback diagram. """


class DualRailBraid(balanced.DualRailBraid, Box):
    """ A dual rail braid in a balanced feedback diagram. """


class DualRailTwist(balanced.DualRailTwist, Box):
    """ A dual rail twist in a balanced feedback diagram. """


class Sum(balanced.Sum, feedback.Sum, Box):
    """ A balanced feedback sum is a balanced and a feedback sum. """


class Functor(balanced.Functor, feedback.Functor):
    """
    A balanced feedback functor is a balanced functor and a feedback functor.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to
            :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain, :code:`Diagram` by default.
    """
    dom = cod = Diagram


class DualRail(balanced.DualRail, Functor):
    """
    The dual rail encoding of a balanced feedback diagram, which lands in
    balanced feedback diagrams, so that a trace has somewhere to go.

    >>> z = Ty('z')
    >>> g = Box('g', z @ z, z @ z)
    >>> rails = balanced.double_rail(z @ z)
    >>> assert g.trace().to_braided() == Box('g', rails, rails).trace(n=2)
    """
    cod = Diagram
    dual_rail_twist_factory = DualRailTwist
    dual_rail_braid_factory = DualRailBraid


Feedback, Feedforward, Trace = feedback.structural_boxes(Diagram, Box)
Diagram.box_factory = Box
Diagram.functor_factory = Functor
Diagram.braid_factory = Braid
Diagram.twist_factory = Twist
Diagram.sum_factory = Sum
Diagram.dual_rail_factory = DualRail
CMap = cmap.CMap[Diagram]
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id

TermBase, Variable, Constant, Application, Kappa = kappa_calculus(Diagram)
Term = Variable | Constant | Application | Kappa


Bubble = Box.from_args
