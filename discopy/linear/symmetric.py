# -*- coding: utf-8 -*-

"""
The free symmetric linear category, i.e. a symmetric monoidal category with
a bang, see :mod:`discopy.linear.monoidal`: the :class:`Exchange` that makes
a banged object central is the swap by default, as a symmetric category is
balanced with the trivial twist.

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
    Sum
    Bang
    Structural
    Dereliction
    Digging
    Contraction
    Weakening
    Exchange
    Laxator
    Functor

Example
-------
The translation of ``lambda x, y: (y, x)`` with ``x`` banged swaps it.

>>> x, y = Ty('x'), Ty('y')
>>> assert Diagram.wiring([1, 0], x.bang() @ y) == Swap(x.bang(), y)
>>> assert Diagram.exchange(x, y, is_dagger=True) == Swap(y, x.bang())

A category overriding :meth:`Diagram.exchange` keeps its own, e.g. the free
one with an :class:`Exchange` box of its own, which a functor sends to the
swap of its codomain.

>>> f = Box('f', x, y)
>>> F = Functor({x: x, y: y}, {f: f})
>>> assert F(Exchange(x, y)) == Swap(x.bang(), y)
"""

from __future__ import annotations

from discopy import symmetric
from discopy.cat import factory
from discopy.linear import monoidal as planar
from discopy.linear.monoidal import (  # noqa: F401
    Ty, BangOb)  # pylint: disable=unused-import


class Layer(symmetric.Layer):
    """ A symmetric linear layer, with permutations as plumbing. """


@factory
class Diagram(planar.Diagram, symmetric.Diagram):
    """
    A symmetric linear diagram is a linear diagram and a symmetric diagram,
    whose :meth:`exchange` is the swap.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty
    layer_factory = Layer

    @classmethod
    def exchange(cls, x: Ty, y: Ty, is_dagger: bool = False) -> Diagram:
        """
        The exchange ``!x @ y -> y @ !x`` is the swap, and its dagger the
        swap back.

        Parameters:
            x : The type that is banged.
            y : The type it is exchanged with.
            is_dagger : Whether to exchange ``y @ !x -> !x @ y`` instead.
        """
        return cls.swap(y, x.bang()) if is_dagger else cls.swap(x.bang(), y)


class Box(planar.Box, symmetric.Box, Diagram):
    """
    A symmetric linear box is a linear box and a symmetric box.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
    """


class Permutation(symmetric.Permutation, Box):
    """ A permutation in a symmetric linear diagram. """


class Swap(Permutation, symmetric.Swap, Box):
    """ A swap in a symmetric linear diagram. """


class Sum(planar.Sum, symmetric.Sum, Box):
    """ A sum of symmetric linear diagrams. """


class Bang(planar.Bang, Box):
    """
    The bang of a symmetric linear diagram, see
    :class:`~discopy.linear.monoidal.Bang`.
    """


class Structural(planar.Structural, Box):
    """ A structural box of a symmetric linear category. """


class Dereliction(planar.Dereliction, Structural):
    """
    The dereliction ``!x -> x``, see
    :class:`~discopy.linear.monoidal.Dereliction`.
    """


class Digging(planar.Digging, Structural):
    """
    The digging ``!x -> !!x``, see
    :class:`~discopy.linear.monoidal.Digging`.
    """


class Contraction(planar.Contraction, Structural):
    """
    The contraction ``!x -> !x @ !x``, see
    :class:`~discopy.linear.monoidal.Contraction`.
    """


class Weakening(planar.Weakening, Structural):
    """
    The weakening ``!x -> 1``, see
    :class:`~discopy.linear.monoidal.Weakening`.
    """


class Exchange(planar.Exchange, Structural):
    """
    The exchange ``!x @ y -> y @ !x`` as a box, see
    :class:`~discopy.linear.monoidal.Exchange`: :meth:`Diagram.exchange` is
    the swap rather than this box.
    """


class Laxator(planar.Laxator, Structural):
    """
    The laxator ``!x @ !y -> !(x @ y)``, see
    :class:`~discopy.linear.monoidal.Laxator`.
    """


class Functor(planar.Functor, symmetric.Functor):
    """
    A symmetric linear functor is a linear functor and a symmetric functor.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to
            :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain, :code:`Diagram` by default.
    """
    dom = cod = Diagram


Id = Diagram.id
Diagram.box_factory = Box
Diagram.sum_factory = Sum
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.bang_factory = Bang
Diagram.dereliction_factory = Dereliction
Diagram.digging_factory = Digging
Diagram.contraction_factory = Contraction
Diagram.weakening_factory = Weakening
Diagram.exchange_factory = Exchange
Diagram.laxator_factory = Laxator
Diagram.functor_factory = Functor

Bubble = Box.from_args
