# -*- coding: utf-8 -*-

"""
The free compact category, i.e. diagrams with swaps, cups and caps.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Layer
    Diagram
    Box
    Cup
    Cap
    Swap
    Permutation
    Feedback
    Feedforward
    Trace
    FollowedBy
    Head
    Tail
    Sum
    Functor
    DualRail
    Equation

Axioms
------

>>> x, y = Ty('x'), Ty('y')

Snake equations
===============

>>> snake = Equation(Id(x.l).transpose(left=True), Id(x), Id(x.r).transpose())
>>> assert snake
>>> snake.draw(doctest="docs/_static/compact/snake.svg")

.. image:: /_static/compact/snake.svg
    :align: center

Yanking
=======
a.k.a. Reidemeister move 1

>>> cap_yanking = Equation(Cap(x, x.r) >> Swap(x, x.r), Cap(x.r, x))
>>> cup_yanking = Equation(Swap(x, x.r) >> Cup(x.r, x), Cup(x, x.r))
>>> assert cap_yanking and cup_yanking
>>> Equation(cap_yanking, cup_yanking, symbol='', space=1).draw(
...     doctest="docs/_static/compact/yanking_cup_and_cap.svg")

.. image:: /_static/compact/yanking_cup_and_cap.svg
    :align: center

Coherence
=========

>>> assert Equation(Diagram.caps(x @ y, y.r @ x.r),
...     Cap(x, x.r) @ Cap(y, y.r) >> x @ Diagram.swap(x.r, y @ y.r))
"""

from typing import ClassVar

# pylint: disable=inconsistent-mro  # astroid misreads the MRO, C3 does not

from discopy import (
    symmetric, ribbon, pivotal, cmap, hypergraph, interaction)
from discopy.abc import (
    CompactCategory, DaggerMonoidalCategory, RigidCategory, SymmetricCategory)
from discopy.cat import factory
from discopy.utils import export, deprecated_alias
from discopy.pivotal import (  # noqa: F401  pylint: disable=unused-import
    Wire, Ty)


class Layer(symmetric.Layer, pivotal.Layer):
    """ A compact layer with permutation plumbing and rigid rotation. """


@factory
class Diagram(symmetric.Diagram, ribbon.Diagram, CompactCategory):
    """
    A compact diagram is a symmetric diagram and a ribbon diagram.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (pivotal.Ty) : The domain of the diagram, i.e. its input.
        cod (pivotal.Ty) : The codomain of the diagram, i.e. its output.
    """
    # A hypergraph is faithful from symmetric categories on.
    snake_equations = CompactCategory.snake_equations
    transpose_contravariance = CompactCategory.transpose_contravariance
    pivotality = CompactCategory.pivotality
    currying_left = CompactCategory.currying_left
    currying_right = CompactCategory.currying_right
    currying_eta_left = CompactCategory.currying_eta_left
    currying_eta_right = CompactCategory.currying_eta_right
    currying_naturality_left = CompactCategory.currying_naturality_left
    currying_naturality_right = CompactCategory.currying_naturality_right

    Sum: ClassVar[type["symmetric.Sum"]]
    Cup: ClassVar[type["ribbon.Cup"]]
    Cap: ClassVar[type["ribbon.Cap"]]
    Box: ClassVar[type["symmetric.Box"]]
    Swap: ClassVar[type["symmetric.Swap"]]
    Functor: ClassVar[type["Functor"]]
    Permutation: ClassVar[type["Permutation"]]
    ob = Ty
    Layer = Layer
    Trace = classmethod(pivotal.Diagram.Trace.__func__)


@Diagram.extend
class Permutation(symmetric.Permutation, Diagram.Box):
    """
    A compact permutation is a symmetric permutation in a compact category.

    Parameters:
        dom (pivotal.Ty) : The domain, i.e. the wires to permute.
        perm : The permutation as a :class:`finset.Permutation` or a list.
    """
    def rotate(self, left=False):
        dom = self.cod.l if left else self.cod.r
        return type(self)(dom, self.perm.rotate())

    l = property(lambda self: self.rotate(left=True))
    r = property(lambda self: self.rotate(left=False))


@Diagram.extend
class Functor(symmetric.Functor, ribbon.Functor):
    """
    A compact functor is both a symmetric functor and a ribbon functor.

    Parameters:
        ob_map (Mapping[pivotal.Ty, pivotal.Ty]) :
            Map from atomic :class:`pivotal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, (symmetric.Swap, symmetric.Permutation)):
            return symmetric.Functor.__call__(self, other)
        return ribbon.Functor.__call__(self, other)


CMap = cmap.CMap[Diagram]

Id = Diagram.id

Hypergraph = hypergraph.Hypergraph[Diagram]
interaction.Diagram[Diagram].dagger_involution_rigid\
    = DaggerMonoidalCategory.dagger_involution.weaken(within=RigidCategory)
interaction.Diagram[Diagram].dagger_contravariance_rigid\
    = DaggerMonoidalCategory.dagger_contravariance.weaken(within=RigidCategory)
interaction.Diagram[Diagram].dagger_monoidality_symmetric\
    = DaggerMonoidalCategory.dagger_monoidality.weaken(
        within=SymmetricCategory)


class Equation(symmetric.Equation):
    """ The :class:`symmetric.Equation` of compact diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)


__getattr__ = deprecated_alias(__name__, {"Ob": "Wire"})


Bubble = Diagram.Box.from_args

export(__name__)
