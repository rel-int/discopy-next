# -*- coding: utf-8 -*-

"""
The free spacial category, i.e. planar diagrams where scalars slide.

A monoidal category is spacial when a scalar ``s : Ty() -> Ty()`` slides
past every morphism, ``s @ f == f @ s``, see :cite:t:`Selinger10`. Since a
scalar can cross any wire, a component of a diagram that touches nothing
else lies in no face in particular: the map of a spacial diagram is the
diagram up to the axioms, see :class:`~discopy.cmap.CMap`, so that rewriting
it with :meth:`~discopy.cmap.CMap.rewrite` is sound whether or not it is
connected. Over :mod:`~discopy.monoidal`, a host must be connected.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Sum
    Functor
    TermBase
    Variable
    Constant
    Application
    Kappa
    Equation

Axioms
------
A scalar slides on the map of a diagram, not on the nose.

>>> x = Ty('x')
>>> f, s = Box('f', x, x), Box('s', Ty(), Ty())
>>> assert s @ f != f @ s
>>> assert len(list((f @ s).to_map().match((s @ f).to_map()))) == 1

A scalar inside a circle is a scalar beside it, so that removing the circle
leaves the scalar.

>>> cap, cup = Box('cap', Ty(), x @ x), Box('cup', x @ x, Ty())
>>> circle = (cap >> cup).to_map(), Id().to_map()
>>> rewrite, = (cap >> x @ s @ x >> cup).to_map().rewrite(circle)
>>> print(rewrite.to_diagram())
s
"""

from __future__ import annotations

from discopy import monoidal, cmap, hypergraph, messages
from discopy.abc import SpacialCategory
from discopy.cat import factory, kappa_calculus
from discopy.monoidal import Ty
from discopy.utils import export


@factory
class Diagram(monoidal.Diagram, SpacialCategory):
    """
    A spacial diagram is a monoidal diagram whose scalars slide.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty
    scalar_sliding = SpacialCategory.scalar_sliding.failing(
        messages.FREE_SCALAR)


class Equation(monoidal.Equation):
    """ The :class:`monoidal.Equation` of spacial diagrams. """


CMap = cmap.CMap[Diagram]
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id

TermBase = kappa_calculus(Diagram)


Bubble = Diagram.Box.from_args

export(__name__)
