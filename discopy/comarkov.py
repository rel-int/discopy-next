# -*- coding: utf-8 -*-

"""
The free comarkov category, i.e. a symmetric category with a supply of
commutative monoids, the dual of :mod:`markov`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Swap
    Permutation
    Merge
    Noise
    Feedback
    Trace
    FollowedBy
    Head
    Tail
    Functor

Axioms
------

>>> x = Ty('x')

>>> merge, noise = Merge(x), Merge(x, n=0)

Commutative monoid
==================

>>> unitality = Equation(noise @ x >> merge, Id(x), x @ noise >> merge)
>>> associativity = Equation(merge @ x >> merge, x @ merge >> merge)
>>> commutativity = Equation(Swap(x, x) >> merge, merge)
>>> assert unitality and associativity and commutativity
>>> Equation(unitality, associativity, commutativity, symbol='').draw(
...     doctest="docs/_static/frobenius/monoid.svg")

.. image:: /_static/frobenius/monoid.svg
    :align: center

Coherence
=========

>>> assert Equation(Diagram.merge(x @ x, n=0), noise @ noise)
>>> assert Equation(Diagram.merge(x @ x),
...     x @ Swap(x, x) @ x >> merge @ merge)

Note
----
Equality of comarkov diagrams is computed by translation to hypergraph.
Merge boxes are translated to spiders, thus when they appear in the same
diagram as the copy boxes of :mod:`markov` they automatically satisfy the
:mod:`frobenius` axioms.
"""

from __future__ import annotations

from typing import ClassVar

from discopy import symmetric, monoidal, feedback, cmap, hypergraph, messages
from discopy.abc import ComarkovCategory, MarkovCategory
from discopy.cat import factory
from discopy.feedback import (  # noqa: F401  pylint: disable=unused-import
    Wire, Ty)
from discopy.utils import (
    AxiomError, assert_isatomic, factory_name, from_tree)


Layer = symmetric.Layer


@factory
class Diagram(symmetric.Diagram, ComarkovCategory):
    """
    A comarkov diagram is a symmetric diagram with :class:`Merge` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.
    """
    merge_factory: ClassVar[type["Merge"]]
    noise_factory: ClassVar[type["Noise"]]

    @classmethod
    def spider_factory(cls, n_legs_in, n_legs_out, typ, phase=None):
        if phase is not None or n_legs_out != 1:
            raise ValueError(messages.NO_SUCH_SPIDER.format(
                factory_name(cls), "merges"))
        return cls.merge_factory(typ, n_legs_in)

    @classmethod
    def merge(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        Merge :code:`n` copies of a given type :code:`x` into one.

        Parameters:
            x : The type to merge.
            n : The number of copies.
        """
        from discopy import frobenius
        return frobenius.Diagram.spiders.__func__(cls, n, 1, x)


class Box(symmetric.Box, Diagram):
    """
    A comarkov box is a symmetric box in a comarkov diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """


class Permutation(symmetric.Permutation, Box):
    """
    A permutation in a comarkov category.

    Parameters:
        dom (monoidal.Ty) : The domain, i.e. the wires to permute.
        perm : The permutation as a :class:`finset.Permutation` or a list.
    """


class Swap(Permutation, symmetric.Swap, Box):
    """
    Symmetric swap in a comarkov diagram.

    Parameters:
        left (monoidal.Ty) : The type on the top left and bottom right.
        right (monoidal.Ty) : The type on the top right and bottom left.
    """


class Merge(Box):
    """
    The merge of an atomic type :code:`x` some :code:`n` number of times.

    Parameters:
        x : The type of wires to merge.
        n : The number of wires to merge.
    """
    def __init__(self, x: monoidal.Ty, n: int = 2):
        assert_isatomic(x, monoidal.Ty)
        name = f"Merge({x}" + ("" if n == 2 else f", {n}") + ")"
        Box.__init__(self, name, dom=x ** n, cod=x,
                     draw_as_spider=True, color="black", drawing_name="")

    def __new__(cls, x: monoidal.Ty, n: int = 2):
        return super().__new__(cls) if n else\
            cls.noise_factory.__new__(cls.noise_factory, x)

    def __getnewargs__(self):
        """
        The arguments of :meth:`__new__`, which a pickle needs.

        >>> from pickle import dumps, loads
        >>> x = Ty('x')
        >>> assert loads(dumps(Merge(x))) == Merge(x)
        >>> assert loads(dumps(Noise(x))) == Noise(x)
        """
        return self.cod, len(self.dom)

    def dagger(self) -> Diagram:
        """
        The copy of the same type, when the category is also Markov.

        >>> Merge(Ty('x')).dagger()
        Traceback (most recent call last):
            ...
        discopy.utils.AxiomError: Merge(x) has no dagger: comarkov.Diagram \
is not a dagger category.
        """
        if not issubclass(self.ar, MarkovCategory):
            raise AxiomError(messages.NOT_A_DAGGER.format(
                self, factory_name(self.ar)))
        return self.ar.copy_factory(self.cod, len(self.dom))

    def __repr__(self):
        return (
            factory_name(type(self)) + f"({repr(self.cod)}, {len(self.dom)})")

    @classmethod
    def from_tree(cls, tree: dict) -> Merge:
        return cls(from_tree(tree['cod']), len(from_tree(tree['dom'])))


class Noise(Merge):
    """
    The noise on an atomic type :code:`x`, i.e. the merge of no copy: the
    state every object of a comarkov category carries, dual to the effect
    :class:`discopy.markov.Discard` every object of a Markov category
    carries. "The time reversal of a causal theory is eternal noise"
    (toumix, 2026-10-17).

    Parameters:
        x : The type of the noise.
    """
    def __init__(self, x: monoidal.Ty, n: int = 0):
        if n:
            raise ValueError(f"A noise has no copies, got n={n}.")
        super().__init__(x, 0)


class Sum(symmetric.Sum, Box):
    """
    A comarkov sum is a symmetric sum and a comarkov box.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(symmetric.Functor):
    """
    A comarkov functor is a symmetric functor that preserves merges.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.

    Example
    -------

    We build a functor into Python functions with the disjoint union as
    tensor, where the merge of a type is the codiagonal: whichever side its
    input comes from, it comes out the same.

    >>> from discopy.python import additive
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> F = Functor({x: int}, {f: lambda n: n + 1, g: lambda n: 2 * n},
    ...             cod=additive.Function)
    >>> assert F(f @ g >> Merge(x))(41, 0) == 42
    >>> assert F(f @ g >> Merge(x))(21, 1) == 42
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Merge) and hasattr(self.cod, "merge"):
            return self.cod.merge(self(other.cod), len(other.dom))
        return super().__call__(other)


CMap = cmap.CMap[Diagram]

Feedback, Trace, FollowedBy, Head, Tail = feedback.generators(Diagram, Box)
Diagram.box_factory = Box
Diagram.functor_factory = Functor
Hypergraph = hypergraph.Hypergraph[Diagram]
Diagram.merge_factory = Merge
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.noise_factory = Noise
Diagram.sum_factory = Sum
Id = Diagram.id


class Equation(symmetric.Equation):
    """ The :class:`symmetric.Equation` of comarkov diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)
