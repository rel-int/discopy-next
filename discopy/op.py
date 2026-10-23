# -*- coding: utf-8 -*-

"""
The opposite of a category, i.e. the same objects with every arrow the other
way round: an arrow ``f : x -> y`` of the opposite is an arrow
``f.inside : y -> x`` of the underlying ``category``.

Composition is reversed and the tensor is not, so the opposite of a monoidal
or symmetric category is monoidal or symmetric, with the swap the other way
round. The opposite of a Markov category is comarkov, its copies read
backwards as merges, and vice versa: :class:`Markov` and :class:`Comarkov`
swap the two.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Category
    Monoidal
    Symmetric
    Markov
    Comarkov

Example
-------
A :class:`discopy.python.finset.Function` is a function between finite sets
read backwards, from the outputs it reads to the inputs, so that the copy is
an arrow from ``1`` to ``2``. Its opposite is the category of finite sets and
functions, where the merge of two elements is the function from ``2`` to
``1``:

>>> from discopy.abc import Nat
>>> from discopy.python.finset import Function
>>> FinSet = Comarkov[Function]
>>> merge = FinSet.merge(Nat(1))
>>> assert merge == FinSet(Function.copy(Nat(1)))
>>> assert (merge.dom, merge.cod) == (Nat(2), Nat(1))

Composition is that of the underlying category in the other order:

>>> noise = FinSet.noise(Nat(1))
>>> assert noise @ Nat(1) >> merge == FinSet.id(Nat(1))
>>> assert (noise @ Nat(1) >> merge).inside\\
...     == merge.inside >> (noise @ Nat(1)).inside
"""

from __future__ import annotations

from dataclasses import dataclass

from discopy import abc, cat, monoidal, symmetric
from discopy.abc import (
    ComarkovCategory, MarkovCategory, MonoidalCategory, NamedGeneric,
    SymmetricCategory)
from discopy.axioms import Testable
from discopy.python import finset
from discopy.utils import (
    assert_isinstance, classproperty, factory_name, get_origin, unbiased)

NO_TREE = "The opposite of a category has no tree: it reads back from the "\
    "class of its arrow, which a tree does not name."


@dataclass
class Category(abc.Category, Testable["Category"], NamedGeneric['category']):
    """
    An arrow of the opposite of ``category``, i.e. an arrow ``inside`` of the
    category read from its codomain to its domain.

    Parameters:
        inside (category) : The arrow of the category, the other way round.

    .. admonition:: Summary

        .. autosummary::

            id
            then
    """
    category = cat.Arrow
    ob = classproperty(lambda cls: cls.category.ob)

    inside: category

    def __post_init__(self):
        assert_isinstance(self.inside, self.category)

    @property
    def dom(self):
        """ The domain of the opposite arrow, the codomain of ``inside``. """
        return self.inside.cod

    @property
    def cod(self):
        """ The codomain of the opposite arrow, the domain of ``inside``. """
        return self.inside.dom

    def __repr__(self):
        factory, category = map(
            factory_name, (get_origin(type(self)), self.category))
        return f"{factory}[{category}]({self.inside!r})"

    @classmethod
    def id(cls, dom=None) -> Category:
        """
        The identity, i.e. the identity of the category.

        Parameters:
            dom : The object on which to take the identity.
        """
        return cls(cls.category.id(cls.ob() if dom is None else dom))

    @unbiased
    def then(self, other: Category) -> Category:
        """
        Composition, i.e. the composition of the category in the other order.

        Parameters:
            other : The arrow to compose with.
        """
        assert_isinstance(other, type(self))
        return type(self)(other.inside >> self.inside)

    @classmethod
    def strategy(cls, *, dom=None, cod=None, **params):
        """
        Generate the arrows of the category the other way round, i.e. those
        of its strategy from ``cod`` to ``dom``.
        """
        return cls.category.strategy(dom=cod, cod=dom, **params).map(cls)

    @classmethod
    def environment(cls) -> dict:
        """
        The namespace of :class:`Testable`, with that of the category on top
        so that the representation of ``inside`` reads back too.
        """
        return dict(super().environment(), **getattr(
            cls.category, "environment", dict)())

    serialisation = Testable.serialisation.inapplicable(NO_TREE)


class Monoidal(Category, MonoidalCategory):
    """
    An arrow of the opposite of a monoidal ``category``, with the tensor of
    the category, i.e. the opposite of a monoidal category is monoidal.

    Parameters:
        inside (category) : The arrow of the category, the other way round.
    """
    category = monoidal.Diagram

    @unbiased
    def tensor(self, other: Monoidal) -> Monoidal:
        """
        The tensor, i.e. the tensor of the category in the same order.

        Parameters:
            other : The arrow to tensor with.
        """
        assert_isinstance(other, type(self))
        return type(self)(self.inside @ other.inside)


class Symmetric(Monoidal, SymmetricCategory):
    """
    An arrow of the opposite of a symmetric ``category``, with the swap of
    the category the other way round.

    Parameters:
        inside (category) : The arrow of the category, the other way round.
    """
    category = symmetric.Diagram

    @classmethod
    def swap(cls, left, right) -> Symmetric:
        """
        The swap of ``left`` and ``right``, i.e. that of ``right`` and
        ``left`` in the category.

        Parameters:
            left : The object on the left of the swap.
            right : The object on the right of the swap.
        """
        return cls(cls.category.swap(right, left))


class Comarkov(Symmetric, ComarkovCategory):
    """
    An arrow of the opposite of a Markov ``category``, whose merges are the
    copies of the category: the opposite of :class:`finset.Function
    <discopy.python.finset.Function>` by default, i.e. finite sets and
    functions.

    Parameters:
        inside (category) : The arrow of the category, the other way round.
    """
    category = finset.Function

    @classmethod
    def merge(cls, x, n: int = 2) -> Comarkov:
        """
        The merge of ``n`` copies of ``x``, i.e. its copy in the category.

        Parameters:
            x : The object to merge.
            n : The number of copies.
        """
        return cls(cls.category.copy(x, n))


class Markov(Symmetric, MarkovCategory):
    """
    An arrow of the opposite of a comarkov ``category``, whose copies are the
    merges of the category: the opposite of :class:`Comarkov` by default,
    i.e. :class:`finset.Function <discopy.python.finset.Function>` again.

    Parameters:
        inside (category) : The arrow of the category, the other way round.
    """
    category = Comarkov[finset.Function]

    @classmethod
    def copy(cls, x, n: int = 2) -> Markov:
        """
        The ``n`` copies of ``x``, i.e. their merge in the category.

        Parameters:
            x : The object to copy.
            n : The number of copies.
        """
        return cls(cls.category.merge(x, n))
