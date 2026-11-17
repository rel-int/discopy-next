# -*- coding: utf-8 -*-

"""
The enrichment of an underlying `category` in commutative monoids.

A sum from `dom` to `cod` is a formal finite sum, i.e. a multiset, of
parallel arrows `dom -> cod` of the underlying category, its `terms`.
Composition and tensor distribute over the sum: the terms of `f >> g` are
the composites of a term of `f` with a term of `g`, those of `f @ g` the
tensors of a term of each, and the unit is the empty sum of a given type,
the zero arrow. A sum of one term is equal to that term.

A sum is an arrow of its category: :class:`Sum[C] <Sum>` is a box of `C`,
the one its ``box_factory`` names, so that a sum composes with the arrows
of `C` and a functor sends it to the sum of the images of its terms.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Sum

Axioms
------

>>> from discopy.cat import Arrow, Ob, Box
>>> x, y, z = Ob('x'), Ob('y'), Ob('z')
>>> f, g = Box('f', x, y), Box('g', y, z)
>>> zero = Sum[Arrow]((), x, y)
>>> assert f + zero == f == zero + f
>>> assert f >> (g + g) == (f >> g) + (f >> g) == (f + f) >> g

The tensor distributes too, over a monoidal category:

>>> from discopy import monoidal
>>> h = monoidal.Box('h', 'x', 'x')
>>> assert (h + h) @ (h + h) == sum(4 * [h @ h])
>>> assert type(h + h) is Sum[monoidal.Diagram]
>>> assert isinstance(h + h, monoidal.Box)
"""

from __future__ import annotations

import itertools
from functools import cached_property

from discopy import cat, messages
from discopy.utils import (
    NamedGeneric, assert_iscomposable, assert_isparallel, factory_key,
    factory_name, from_tree, unbiased)


class Sum[category](cat.Box):
    """
    A sum is a tuple of arrows :code:`terms` with the same domain and codomain,
    in an underlying `category`, a box of it.

    Parameters:
        terms (tuple[category, ...]) : The terms of the formal sum.
        dom (category.ob) : The domain of the formal sum.
        cod (category.ob) : The codomain of the formal sum.

    Example
    -------
    >>> from discopy.cat import Ob, Box
    >>> x, y = Ob('x'), Ob('y')
    >>> f = Box('f', x, y)
    >>> print(f + f)
    (f) + (f)
    >>> f + f
    sum.Sum(terms=(cat.Box('f', cat.Ob('x'), cat.Ob('y')), \
cat.Box('f', cat.Ob('x'), cat.Ob('y'))))

    Important
    ---------
    Domain and codomain are optional only if the terms are non-empty.

    Note
    ----
    The sum is non-commutative, i.e. :code:`Sum([f, g]) != Sum([g, f])`.
    """
    category = cat.Arrow
    serialised_attrs = ("terms", "dom", "cod")

    # A NamedGeneric subscript, not a base, which would come before
    # the box of the category in the method resolution order.
    __class_getitem__ = vars(NamedGeneric)["__class_getitem__"]

    @classmethod
    def subscript_bases(cls, category) -> tuple[type, ...]:
        """
        The box of the underlying category, which ``Sum[category]``
        subclasses beside :class:`Sum`, so that a sum is a box of it.

        >>> from discopy import monoidal
        >>> assert Sum.subscript_bases(monoidal.Diagram) == (monoidal.Box, )
        """
        return (category.box_factory, )

    def __init__(self, terms: tuple[cat.Arrow, ...], dom=None, cod=None):
        terms = tuple(terms)
        if not terms and (dom is None or cod is None):
            raise ValueError(messages.MISSING_TYPES_FOR_EMPTY_SUM)
        dom = terms[0].dom if dom is None else dom
        cod = terms[0].cod if cod is None else cod
        unit = Sum((), dom, cod) if terms else None
        for arrow in terms:
            assert_isparallel(unit, arrow)  # ty: ignore[invalid-argument-type]
        self.terms = terms
        super().__init__("", dom, cod)
        del self.name

    @cached_property
    def name(self):  # pylint: disable=method-hidden  # deleted by __init__
        """ The name of a sum is its :code:`repr`, computed when needed. """
        str_args = "" if self.terms\
            else f", dom={repr(self.dom)}, cod={repr(self.cod)}"
        return factory_name(type(self))\
            + f"(terms={repr(self.terms)}{str_args})"

    @property
    def is_generator(self):
        return len(self.terms) == 1 and self.terms[0].is_generator

    @property
    def generator(self):
        return self.terms[0].generator if self.is_generator else None

    def setoid(self):
        """ Ensure that a singleton sum is in fact equal to its only term. """
        return self.terms[0].setoid() if len(self.terms) == 1 else (
            factory_key(type(self)), self.terms, self.dom, self.cod)

    def __repr__(self):
        return self.name

    def __str__(self):
        return " + ".join(f"({arrow})" for arrow in self.terms)\
            if self.terms else\
            f"{factory_name(type(self))}((), {self.dom}, {self.cod})"

    def atoms(self) -> list:
        """ The atoms of a sum: those of its boundary and of its terms. """
        return self.dom.atoms() + self.cod.atoms() + [
            atom for term in self.terms for atom in term.atoms()]

    def __add__(self, other):
        assert_isparallel(self, other)
        other = other if isinstance(other, Sum)\
            else self.sum_factory((other, ))
        return self.sum_factory(
            (*self.terms, *other.terms), self.dom, self.cod)

    def __iter__(self):
        yield from self.terms

    def __len__(self):
        return len(self.terms)

    @unbiased
    def then(self, other):
        assert_iscomposable(self, other)
        other = other if isinstance(other, Sum)\
            else self.sum_factory((other, ))
        terms = tuple(f.then(g) for f in self.terms for g in other.terms)
        return self.sum_factory(terms, self.dom, other.cod)

    def tensor(self, *others):
        """
        The tensor of sums distributes over the sum, i.e. its terms are the
        ``n``-ary tensors of one term from each factor.

        >>> from discopy.monoidal import Box
        >>> f = Box('f', 'x', 'x')
        >>> print(f @ (f + f))
        (f @ x >> x @ f) + (f @ x >> x @ f)
        """
        if not others:
            return self
        sums = [
            other if isinstance(other, Sum) else self.sum_factory((other, ))
            for other in others]
        dom = self.dom.tensor(*(other.dom for other in sums))
        cod = self.cod.tensor(*(other.cod for other in sums))
        terms = tuple(
            term.tensor(*rest) for term in self.terms
            for rest in itertools.product(
                *(other.terms for other in sums)))
        return self.sum_factory(terms, dom, cod)

    def dagger(self):
        terms = tuple(f.dagger() for f in self.terms)
        return self.sum_factory(terms, self.cod, self.dom)

    def to_drawing(self):
        """ A sum is drawn as a diagram, not as a box. """
        return self.factory.to_drawing(self)

    @property
    def free_symbols(self):
        return {x for box in self.terms for x in box.free_symbols}

    def subs(self, *args):
        terms = tuple(f.subs(*args) for f in self.terms)
        return self.sum_factory(terms, self.dom, self.cod)

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: self.sum_factory(
            tuple(box.lambdify(*symbols, **kwargs)(*xs) for box in self.terms),
            dom=self.dom, cod=self.cod)

    def to_tree(self):
        return {
            'factory': factory_name(type(self)),
            'terms': [t.to_tree() for t in self.terms],
            'dom': self.dom.to_tree(),
            'cod': self.cod.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        dom, cod = map(from_tree, (tree['dom'], tree['cod']))
        terms = tuple(map(from_tree, tree['terms']))
        return cls(terms=terms, dom=dom, cod=cod)


cat.Arrow.sum_factory = Sum
