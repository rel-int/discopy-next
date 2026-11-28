# -*- coding: utf-8 -*-

"""
The free Markov feedback category, i.e. Markov diagrams with feedback, whose
wires carry a time step: the syntax of streams, see :mod:`~discopy.stream`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Layer
    Diagram
    Box
    Function
    Permutation
    Swap
    Copy
    Discard
    Feedback
    Feedforward
    Trace
    FollowedBy
    Head
    Tail
    Sum
    Functor

Example
-------
>>> x = Ty('x')
>>> zero = Box('0', Ty(), x.head)
>>> rand = Box('rand', Ty(), x)
>>> plus = Box('+', x @ x, x)
>>> walk = (rand.delay() @ x.delay() >> zero @ plus.delay()
...         >> FollowedBy(x) >> Copy(x)).feedback()
>>> walk.draw(doctest="docs/_static/feedback/feedback-random-walk.svg")

.. image:: /_static/feedback/feedback-random-walk.svg
    :align: center
"""

from __future__ import annotations

from discopy import cmap, feedback, hypergraph, markov, monoidal
from discopy.cat import factory, kappa_calculus
from discopy.feedback import symmetric
from discopy.feedback import (  # noqa: F401  pylint: disable=unused-import
    Wire, Ty)
from discopy.markov import (  # noqa: F401  pylint: disable=unused-import
    Equation)


class Layer(markov.Layer, symmetric.Layer):
    """ A Markov feedback layer is a Markov layer that delays. """


@factory
class Diagram(markov.Diagram, symmetric.Diagram):
    """
    A Markov feedback diagram is a Markov diagram and a symmetric feedback
    diagram.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty
    layer_factory = Layer

    @classmethod
    def spider_factory(cls, n_legs_in, n_legs_out, typ, phase=None):
        """
        The spiders of a Markov feedback category: the copies, with one leg
        in, and the traces of copies, with none, i.e. what tracing a copy back
        into its own input encodes to as a hypergraph.

        >>> x = Ty('x')
        >>> assert Diagram.spider_factory(0, 1, x) == Copy(x).trace(left=True)
        """
        if not n_legs_in and phase is None:
            return cls.copy(typ, n_legs_out + 1).trace(left=True)
        return super().spider_factory(n_legs_in, n_legs_out, typ, phase)


class Box(markov.Box, symmetric.Box, Diagram):
    """
    A Markov feedback box is a Markov box and a symmetric feedback box.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
        is_deterministic : Whether copying commutes with the box.
        is_affine : Whether discarding commutes with the box.
        time_step (int) : The number of times the box has been delayed.
    """
    serialised_attrs = markov.Box.serialised_attrs + ("time_step", )


class Function(markov.Function, Box):
    """ A function in a Markov feedback diagram. """


Layer.plumbing = (monoidal.Ty, Function)


class Permutation(markov.Permutation, symmetric.Permutation, Box):
    """ A permutation in a Markov feedback diagram. """


Function.register(Permutation)


class Swap(Permutation, markov.Swap, symmetric.Swap, Box):
    """ A swap in a Markov feedback diagram. """


class Copy(markov.Copy, Function):
    """ A copy in a Markov feedback diagram. """


class Discard(markov.Discard, Copy):
    """ A discard in a Markov feedback diagram. """


class Sum(markov.Sum, symmetric.Sum, Box):
    """ A Markov feedback sum is a Markov and a feedback sum. """


class Functor(markov.Functor, symmetric.Functor):
    """
    A Markov feedback functor is a Markov and a feedback functor.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to
            :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain, :code:`Diagram` by default.
    """
    dom = cod = Diagram


Feedback, Feedforward, Trace, FollowedBy, Head, Tail, Wait = \
    feedback.structural_boxes(Diagram, Box)
Diagram.box_factory = Box
Diagram.functor_factory = Functor
Diagram.copy_factory = Copy
Diagram.function_factory = Function
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.discard_factory = Discard
Diagram.sum_factory = Sum
CMap = cmap.CMap[Diagram]
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id

TermBase, Variable, Constant, Application, Kappa = kappa_calculus(Diagram)
Term = Variable | Constant | Application | Kappa


Bubble = Box.from_args
