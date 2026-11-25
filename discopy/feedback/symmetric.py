# -*- coding: utf-8 -*-

"""
The free symmetric feedback category, i.e. symmetric diagrams with feedback,
whose wires carry a time step, see :mod:`discopy.feedback`. Their traces
satisfy the axioms of a symmetric traced category, up to the hypergraph of
each side: see :class:`Equation`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Layer
    Diagram
    Box
    Permutation
    Swap
    Feedback
    Feedforward
    Trace
    FollowedBy
    Head
    Tail
    Sum
    Functor
    DualRail

Example
-------
>>> x = Ty('x')
>>> assert Equation(Swap(x, x).trace(), Id(x))
>>> assert Diagram.wait(x) == Swap(x, x.delay()).feedback()
"""

from __future__ import annotations

from discopy import cmap, feedback, hypergraph, symmetric
from discopy.abc import BalancedTracedCategory, TracedCategory
from discopy.cat import factory, kappa_calculus
from discopy.feedback import balanced
from discopy.feedback import (  # noqa: F401  pylint: disable=unused-import
    Wire, Ty, Nat)
from discopy.symmetric import (  # noqa: F401  pylint: disable=unused-import
    Equation)


class Layer(symmetric.Layer, feedback.Layer):
    """ A symmetric feedback layer is a symmetric layer that delays. """


@factory
class Diagram(symmetric.Diagram, balanced.Diagram):
    """
    A symmetric feedback diagram is a symmetric diagram and a balanced
    feedback diagram.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty
    layer_factory = Layer
    yanking = BalancedTracedCategory.yanking.modulo(
        symmetric.Diagram.to_hypergraph)
    trace_naturality_left = TracedCategory.trace_naturality_left
    trace_naturality_right = TracedCategory.trace_naturality_right
    trace_dinaturality_left = TracedCategory.trace_dinaturality_left
    trace_dinaturality_right = TracedCategory.trace_dinaturality_right
    trace_superposing_left = TracedCategory.trace_superposing_left
    trace_superposing_right = TracedCategory.trace_superposing_right

    @classmethod
    def wait(cls, dom: Ty) -> Diagram:
        """
        Wait one time step, i.e. `Swap(x, x.delay()).feedback()`.

        Example
        -------
        >>> x = Ty('x')
        >>> Diagram.wait(x).draw(doctest="docs/_static/feedback/wait.svg")

        .. image:: /_static/feedback/wait.svg
            :align: center
        """
        return cls.swap(dom, dom.delay()).feedback()


class Box(symmetric.Box, balanced.Box, Diagram):
    """
    A symmetric feedback box is a symmetric box and a balanced feedback box.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
        time_step (int) : The number of times the box has been delayed.
    """


class Permutation(symmetric.Permutation, Box):
    """ A permutation in a symmetric feedback diagram. """


class Swap(Permutation, symmetric.Swap, balanced.Braid, Box):
    """ A swap in a symmetric feedback diagram. """


class Sum(symmetric.Sum, balanced.Sum, Box):
    """ A symmetric feedback sum is a symmetric and a feedback sum. """


class Functor(symmetric.Functor, balanced.Functor):
    """
    A symmetric feedback functor is a symmetric and a feedback functor.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to
            :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain, :code:`Diagram` by default.
    """
    dom = cod = Diagram


class DualRail(symmetric.DualRail, balanced.DualRail):
    """ The dual rail encoding of a symmetric feedback diagram. """
    dom = Diagram


Feedback, Feedforward, Trace, FollowedBy, Head, Tail = \
    feedback.structural_boxes(Diagram, Box)
Diagram.box_factory = Box
Diagram.functor_factory = Functor
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.sum_factory = Sum
Diagram.dual_rail_factory = DualRail
CMap = cmap.CMap[Diagram]
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id

TermBase, Variable, Constant, Application, Kappa = kappa_calculus(Diagram)
Term = Variable | Constant | Application | Kappa


Bubble = Box.from_args
