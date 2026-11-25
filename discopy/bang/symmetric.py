# -*- coding: utf-8 -*-

"""
The free symmetric linear category, i.e. symmetric diagrams with a bang, the
exponential of intuitionistic linear logic :cite:t:`Bierman95`: the exchange
of a banged wire is its swap, so its laws hold up to hypergraph isomorphism,
see :class:`discopy.symmetric.Equation`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Permutation
    Swap
    Sum
    Bubble
    Bang
    Dereliction
    Digging
    Contraction
    Weakening
    Laxator
    Functor

Example
-------
>>> x, y = Ty('x'), Ty('y')
>>> assert Diagram.exchange(x, y) == Swap(x.bang(), y)
>>> assert Diagram.wiring([1, 0], x.bang() @ y) == Swap(x.bang(), y)
>>> assert Equation(Diagram.exchange(x, y) >> Diagram.exchange(x, y, True),
...                 Id(x.bang() @ y))
"""

from __future__ import annotations

from discopy import bang, cmap, hypergraph, symmetric
from discopy.abc import LinearCategory
from discopy.bang import (  # noqa: F401  pylint: disable=unused-import
    Ty, BangOb)
from discopy.cat import factory
from discopy.symmetric import (  # noqa: F401  pylint: disable=unused-import
    Equation)


@factory
class Diagram(symmetric.Diagram, bang.Diagram):
    """
    A symmetric linear diagram is a symmetric diagram and a linear diagram,
    whose exchange is the swap.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty
    exchange_hexagon = LinearCategory.exchange_hexagon
    exchange_inverse = LinearCategory.exchange_inverse
    exchange_inverse_dagger = LinearCategory.exchange_inverse_dagger
    exchange_naturality = LinearCategory.exchange_naturality
    exchange_naturality_bang = LinearCategory.exchange_naturality_bang
    exchange_unit = LinearCategory.exchange_unit

    @classmethod
    def exchange(cls, x: Ty, y: Ty, is_dagger: bool = False) -> Diagram:
        """
        The exchange of ``!x`` with ``y`` is the swap, or its inverse when
        ``is_dagger``.
        """
        return cls.swap(y, x.bang()) if is_dagger else cls.swap(x.bang(), y)


class Box(symmetric.Box, bang.Box, Diagram):
    """ A symmetric linear box is a symmetric box and a linear box. """


class Permutation(symmetric.Permutation, Box):
    """ A permutation in a symmetric linear diagram. """


class Swap(Permutation, symmetric.Swap, Box):
    """ A swap in a symmetric linear diagram. """


class Sum(symmetric.Sum, bang.Sum, Box):
    """ A sum in a symmetric linear diagram. """


class Bubble(symmetric.Bubble, bang.Bubble, Box):
    """ A bubble in a symmetric linear diagram. """


class Bang(bang.Bang, Box):
    """ The bang of a symmetric linear diagram. """


class Dereliction(bang.Dereliction, Box):
    """ The dereliction in a symmetric linear diagram. """


class Digging(bang.Digging, Box):
    """ The digging in a symmetric linear diagram. """


class Contraction(bang.Contraction, Box):
    """ The contraction in a symmetric linear diagram. """


class Weakening(bang.Weakening, Box):
    """ The weakening in a symmetric linear diagram. """


class Laxator(bang.Laxator, Box):
    """ The laxator in a symmetric linear diagram. """


class Functor(symmetric.Functor, bang.Functor):
    """
    A symmetric linear functor is a symmetric functor and a linear functor.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to
            :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain, :code:`Diagram` by default.
    """
    dom = cod = Diagram


Id = Diagram.id
Diagram.box_factory = Box
Diagram.functor_factory = Functor
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.sum_factory = Sum
Diagram.bubble_factory = Bubble
Diagram.bang_factory = Bang
Diagram.dereliction_factory = Dereliction
Diagram.digging_factory = Digging
Diagram.contraction_factory = Contraction
Diagram.weakening_factory = Weakening
Diagram.laxator_factory = Laxator
CMap = cmap.CMap[Diagram]
Hypergraph = hypergraph.Hypergraph[Diagram]
