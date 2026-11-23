# -*- coding: utf-8 -*-

"""
The free compact category, i.e. diagrams with swaps, cups and caps.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Nat
    Layer
    Diagram
    Box
    Cup
    Cap
    Swap
    Permutation
    Feedback
    Trace
    FollowedBy
    Head
    Tail
    Sum
    Bubble
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
    symmetric, ribbon, pivotal, feedback, cmap, hypergraph, interaction,
    messages)
from discopy.abc import (
    CompactCategory, DaggerMonoidalCategory, RigidCategory, SymmetricCategory)
from discopy.cat import factory
from discopy.utils import deprecated_alias
from discopy.pivotal import (  # noqa: F401  pylint: disable=unused-import
    Wire)


@factory
class Ty(ribbon.Ty, symmetric.Ty):
    """ A compact type, i.e. a ribbon type and a symmetric one. """


@factory
class Nat(ribbon.Nat, symmetric.Nat, Ty):
    """ A natural number ``n`` seen as a compact type of length ``n``. """


Ty.nat_factory = Nat


class Layer(symmetric.Layer, pivotal.Layer):
    """ A compact layer with permutation plumbing and rigid rotation. """


@factory
class Diagram(symmetric.Diagram, ribbon.Diagram, CompactCategory):
    """
    A compact diagram is a symmetric diagram and a ribbon diagram.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
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

    sum_factory: ClassVar[type["Sum"]]
    cup_factory: ClassVar[type["Cup"]]
    cap_factory: ClassVar[type["Cap"]]
    box_factory: ClassVar[type["Box"]]
    swap_factory: ClassVar[type["Swap"]]
    functor_factory: ClassVar[type["Functor"]]
    permutation_factory: ClassVar[type["Permutation"]]
    ob = Ty
    layer_factory = Layer
    trace_factory = classmethod(pivotal.Diagram.trace_factory.__func__)


class Box(symmetric.Box, ribbon.Box, Diagram):
    """
    A compact box is a symmetric and ribbon box in a compact diagram.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
    """


class Cup(ribbon.Cup, Box):
    """
    A compact cup is a ribbon cup in a compact diagram.

    Parameters:
        left (Ty) : The atomic type.
        right (Ty) : Its adjoint.
    """


class Cap(ribbon.Cap, Box):
    """
    A compact cap is a ribbon cap in a compact diagram.

    Parameters:
        left (Ty) : The atomic type.
        right (Ty) : Its adjoint.
    """


class Permutation(symmetric.Permutation, Box):
    """
    A compact permutation is a symmetric permutation in a compact category.

    Parameters:
        dom (Ty) : The domain, i.e. the wires to permute.
        perm : The permutation as a :class:`finset.Permutation` or a list.
    """
    def rotate(self, left=False):
        dom = self.cod.l if left else self.cod.r
        return type(self)(dom, self.perm.rotate())

    l = property(lambda self: self.rotate(left=True))
    r = property(lambda self: self.rotate(left=False))


class Swap(Permutation, symmetric.Swap, ribbon.Braid, Box):
    """
    A compact swap is a symmetric swap and a ribbon braid.

    Parameters:
        left (Ty) : The type on the top left and bottom right.
        right (Ty) : The type on the top right and bottom left.
    """


class Sum(symmetric.Sum, ribbon.Sum, Box):
    """
    A compact sum is a sum of compact diagrams.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Bubble(symmetric.Bubble, ribbon.Bubble, Box):
    """
    A compact bubble is a symmetric and ribbon bubble and a compact box.
    """


class Functor(symmetric.Functor, ribbon.Functor):
    """
    A compact functor is both a symmetric functor and a ribbon functor.

    Parameters:
        ob_map (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, (symmetric.Swap, symmetric.Permutation)):
            return symmetric.Functor.__call__(self, other)
        return ribbon.Functor.__call__(self, other)


class DualRail(symmetric.DualRail, ribbon.DualRail):
    """
    The functor sending a compact diagram to its dual rail encoding, i.e.
    :class:`discopy.symmetric.DualRail` sending a :class:`Permutation` to the
    dual rail braids of its swaps, and :class:`discopy.ribbon.DualRail`
    otherwise.

    >>> x, y, z = map(Ty, "xyz")
    >>> diagram = (Swap(x, y) @ z).to_braided()
    >>> assert [type(box) for box in diagram.boxes] == [ribbon.DualRailBraid]
    """
    dom = Diagram


CMap = cmap.CMap[Diagram]

Id = Diagram.id

Feedback, Trace, FollowedBy, Head, Tail = feedback.structural_boxes(
    Diagram, Box)
Diagram.box_factory = Box
Diagram.swap_factory = Swap
Diagram.functor_factory = Functor
Diagram.dual_rail_factory = DualRail
Diagram.permutation_factory = Permutation
Hypergraph = hypergraph.Hypergraph[Diagram]
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
Diagram.sum_factory = Sum
Diagram.bubble_factory = Bubble
for name in (
        "dagger_involution", "dagger_contravariance", "dagger_monoidality"):
    setattr(interaction.Diagram[Diagram], name, getattr(
        DaggerMonoidalCategory, name).failing(messages.FEEDBACK_HAS_NO_DAGGER))
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
