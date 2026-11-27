# -*- coding: utf-8 -*-

"""
The free comarkov feedback category, i.e. comarkov diagrams with feedback,
whose wires carry a time step, the dual of :mod:`~discopy.feedback.markov`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Layer
    Diagram
    Box
    Cofunction
    Permutation
    Swap
    Merge
    Noise
    Feedback
    Feedforward
    Trace
    Sum
    Functor

Example
-------
>>> x = Ty('x')
>>> assert Merge(x).delay() == Merge(x.d)
>>> assert Swap(x, x).trace().dom == x
"""

from __future__ import annotations

from discopy import cmap, comarkov, feedback, hypergraph, monoidal
from discopy.cat import factory, kappa_calculus
from discopy.comarkov import (  # noqa: F401  pylint: disable=unused-import
    Equation)
from discopy.feedback import symmetric
from discopy.feedback import (  # noqa: F401  pylint: disable=unused-import
    Wire, Ty)


class Layer(comarkov.Layer, symmetric.Layer):
    """ A comarkov feedback layer is a comarkov layer that delays. """


@factory
class Diagram(comarkov.Diagram, symmetric.Diagram):
    """
    A comarkov feedback diagram is a comarkov diagram and a symmetric
    feedback diagram.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty
    layer_factory = Layer


class Box(comarkov.Box, symmetric.Box, Diagram):
    """
    A comarkov feedback box is a comarkov box and a symmetric feedback box.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
        time_step (int) : The number of times the box has been delayed.
    """


class Cofunction(comarkov.Cofunction, Box):
    """ A cofunction in a comarkov feedback diagram. """


Layer.plumbing = (monoidal.Ty, Cofunction)


class Permutation(comarkov.Permutation, symmetric.Permutation, Box):
    """ A permutation in a comarkov feedback diagram. """


Cofunction.register(Permutation)


class Swap(Permutation, comarkov.Swap, symmetric.Swap, Box):
    """ A swap in a comarkov feedback diagram. """


class Merge(comarkov.Merge, Cofunction):
    """ A merge in a comarkov feedback diagram. """


class Noise(comarkov.Noise, Merge):
    """ A noise in a comarkov feedback diagram. """


class Sum(comarkov.Sum, symmetric.Sum, Box):
    """ A comarkov feedback sum is a comarkov and a feedback sum. """


class Functor(comarkov.Functor, symmetric.Functor):
    """
    A comarkov feedback functor is a comarkov and a feedback functor.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to
            :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain, :code:`Diagram` by default.
    """
    dom = cod = Diagram


Feedback, Feedforward, Trace = feedback.structural_boxes(Diagram, Box)
Diagram.box_factory = Box
Diagram.functor_factory = Functor
Diagram.merge_factory = Merge
Diagram.cofunction_factory = Cofunction
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.noise_factory = Noise
Diagram.sum_factory = Sum
CMap = cmap.CMap[Diagram]
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id

TermBase, Variable, Constant, Application, Kappa = kappa_calculus(Diagram)
Term = Variable | Constant | Application | Kappa


Bubble = Box.from_args
