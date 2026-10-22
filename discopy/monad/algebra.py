# -*- coding: utf-8 -*-

"""
The Eilenberg-Moore category of a monad, i.e. its algebras and their
homomorphisms.

An algebra of a :class:`~discopy.kleisli.monad.Monad` ``M`` is a type ``A``,
its carrier, together with a function :math:`a : M(A) \\to A`, its
structure, satisfying

.. math::
    a \\circ \\eta_A = \\mathrm{id}_A
    \\qquad\\qquad
    a \\circ \\mu_A = a \\circ M(a)

i.e. evaluating a pure value gives it back, and flattening before
evaluating is evaluating twice. A homomorphism from ``(A, a)`` to
``(B, b)`` is a function :math:`h : A \\to B` preserving the structure

.. math::
    h \\circ a = b \\circ M(h)

Every monad has the free algebra ``(M(X), mu_X)`` on any type ``X``, see
:meth:`Algebra.free`, and the free algebras span a full subcategory
isomorphic to the Kleisli category, see :mod:`discopy.monad.free`. The
other algebras depend on the monad:

* an algebra of :attr:`~discopy.kleisli.monad.Maybe` is a pointed type,
  the structure sending ``None`` to the point, and a homomorphism
  preserves the point,
* an algebra of :attr:`~discopy.kleisli.monad.Powerset` is a join
  semilattice with a least element, the structure sending a finite set to
  its join, e.g. the booleans with :func:`any`, and a homomorphism
  preserves joins: de Morgan's law says that ``not`` is a homomorphism from
  :func:`any` to :func:`all`,
* an algebra of :attr:`~discopy.kleisli.monad.Subdistribution` is e.g. the
  expectation of numbers, the missing mass counting as zero, and a
  homomorphism between expectations is linear.

Python functions cannot be compared, so the laws are checked at a point:
:meth:`Algebra.unitality`, :meth:`Algebra.associativity` and
:meth:`Homomorphism.preservation` return an
:class:`~discopy.axioms.Equation` between two values, which holds when the
law does at that point.

Example
-------
The expectation of a subdistribution of numbers is an algebra, its
associativity saying that the expectation of expectations is the
expectation of the average:

>>> from discopy.kleisli.monad import Subdistribution
>>> mean = Algebra[Subdistribution](
...     "mean", float, lambda d: sum((p * x for x, p in d), 0.))
>>> assert mean.unitality(4.)
>>> coin = frozenset({(0., .5), (4., .5)})
>>> assert mean.associativity(frozenset({(coin, .5), (frozenset(), .5)}))

Doubling is a homomorphism of expectations, while incrementing is not: the
missing mass of a subdistribution counts as zero, and zero is not fixed.

>>> double = Homomorphism[Subdistribution](lambda x: 2 * x, mean, mean)
>>> increment = Homomorphism[Subdistribution](lambda x: x + 1, mean, mean)
>>> half = frozenset({(4., .5)})
>>> assert double.preservation(half) and not increment.preservation(half)

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Algebra
    Homomorphism
"""
from __future__ import annotations

from collections.abc import Callable

from discopy.abc import Category, NamedGeneric
from discopy.axioms import Equation
from discopy.kleisli.monad import Monad
from discopy.monad import free
from discopy.python.function import Function
from discopy.utils import (
    assert_iscomposable, assert_ismember, factory, factory_name)


class Algebra(NamedGeneric['monad']):
    """
    An algebra of a monad ``M``, i.e. a carrier type ``A`` with a structure
    function from ``M(A)`` to ``A``.

    Parameters:
        name : The name of the algebra.
        carrier : The carrier type ``A``.
        structure : The structure function, from ``M(A)`` to ``A``.

    Note
    ----
    Two algebras are equal when they have the same monad, name and carrier,
    as two monads are equal when they have the same name, see
    :class:`~discopy.kleisli.monad.Monad`.

    Example
    -------
    The booleans with :func:`any` are an algebra of the powerset monad, and
    so are the booleans with :func:`all`:

    >>> from discopy.kleisli.monad import Powerset
    >>> disjunction = Algebra[Powerset]("any", bool, any)
    >>> conjunction = Algebra[Powerset]("all", bool, all)
    >>> disjunction(frozenset({True, False}))
    True
    >>> assert disjunction.unitality(True) and conjunction.unitality(False)
    >>> assert disjunction.associativity(frozenset({
    ...     frozenset({False}), frozenset({True, False})}))
    """
    monad: Monad

    def __init__(self, name: str, carrier: type, structure: Callable):
        monad = type(self).monad
        self.name, self.carrier = name, carrier
        self.structure = structure if isinstance(structure, Function)\
            else Function(structure, monad(carrier), carrier)

    @classmethod
    def free(cls, generators: type) -> Algebra:
        """
        The free algebra on a type of generators ``X``, i.e. ``M(X)`` with
        the multiplication of the monad as structure.

        Parameters:
            generators : The type of generators ``X``.

        Example
        -------
        >>> from discopy.kleisli.monad import Maybe
        >>> Algebra[Maybe].free(int)
        monad.algebra.Algebra[Maybe]('free', int | None)
        >>> assert Algebra[Maybe].free(int).unitality(None)
        """
        monad = cls.monad
        return cls(
            "free", monad(generators).inside[0], monad.mult(generators))

    def unitality(self, value) -> Equation:
        """
        The unit law at a value ``x`` of the carrier, i.e.
        :math:`a(\\eta(x)) = x`.

        Parameters:
            value : The value ``x`` of the carrier.
        """
        unit = type(self).monad.unit(self.carrier)
        return Equation((unit >> self.structure)(value), value)

    def associativity(self, value) -> Equation:
        """
        The associativity law at a value ``m`` of ``M(M(A))``, i.e.
        :math:`a(\\mu(m)) = a(M(a)(m))`.

        Parameters:
            value : The value ``m`` of ``M(M(A))``.
        """
        monad = type(self).monad
        return Equation(
            (monad.mult(self.carrier) >> self.structure)(value),
            (monad(self.structure) >> self.structure)(value))

    def __call__(self, value):
        return self.structure(value)

    def __eq__(self, other):
        return isinstance(other, Algebra) and (
            self.monad, self.name, self.carrier) == (
                other.monad, other.name, other.carrier)

    def __hash__(self):
        return hash((self.monad, self.name, self.carrier))

    def __repr__(self):
        return factory_name(type(self))\
            + f"({self.name!r}, {factory_name(self.carrier)})"

    def __str__(self):
        return self.name


@factory
class Homomorphism(Category, NamedGeneric['monad']):
    """
    A homomorphism between two algebras of a monad ``M``, i.e. a function
    between their carriers preserving their structure.

    Parameters:
        inside : The underlying function, between the carriers.
        dom : The domain algebra.
        cod : The codomain algebra.

    Note
    ----
    That ``inside`` preserves the structure is not checked: a function
    cannot be compared with another, but the law can be checked at a point
    with :meth:`preservation`.

    Example
    -------
    The negation is a homomorphism from the booleans with :func:`any` to the
    booleans with :func:`all`, i.e. de Morgan's law:

    >>> from discopy.kleisli.monad import Powerset
    >>> disjunction = Algebra[Powerset]("any", bool, any)
    >>> conjunction = Algebra[Powerset]("all", bool, all)
    >>> negation = Homomorphism[Powerset](
    ...     lambda x: not x, disjunction, conjunction)
    >>> assert all(negation.preservation(xs) for xs in map(frozenset, [
    ...     (), (False, ), (True, ), (False, True)]))

    The identity on the booleans with :func:`any` is not a homomorphism to
    the booleans with :func:`all`, since a join is not a meet:

    >>> identity = Homomorphism[Powerset](
    ...     lambda x: x, disjunction, conjunction)
    >>> assert not identity.preservation(frozenset({False, True}))
    """
    ob = Algebra
    monad: Monad

    def __init__(self, inside: Callable, dom: Algebra, cod: Algebra):
        monad = type(self).monad
        assert_ismember(dom, Algebra[monad])
        assert_ismember(cod, Algebra[monad])
        self.inside = inside if isinstance(inside, Function)\
            else Function(inside, dom.carrier, cod.carrier)
        self.dom, self.cod = dom, cod

    @classmethod
    def id(cls, dom: Algebra) -> Homomorphism:
        """
        The identity homomorphism on an algebra, i.e. the identity function
        on its carrier.

        Parameters:
            dom : The algebra on which to take the identity.
        """
        return cls(Function.id(dom.carrier), dom, dom)

    def then(self, other: Homomorphism) -> Homomorphism:
        """
        The composition of two homomorphisms, i.e. of their functions,
        called with :code:`>>`.

        Parameters:
            other : The other homomorphism to compose in sequence.
        """
        assert_ismember(other, type(self))
        assert_iscomposable(self, other)
        return type(self)(self.inside >> other.inside, self.dom, other.cod)

    @classmethod
    def from_free(cls, homomorphism: free.Homomorphism) -> Homomorphism:
        """
        The inclusion of a homomorphism between free algebras, i.e. the same
        function between the free algebras on its domain and codomain.

        Parameters:
            homomorphism : The homomorphism between free algebras.

        Example
        -------
        >>> from discopy.kleisli.monad import Powerset
        >>> from discopy.kleisli.channel import Channel
        >>> neighbours = Channel[Powerset](
        ...     lambda x: frozenset({x - 1, x + 1}), int, int)
        >>> step = Homomorphism[Powerset].from_free(
        ...     free.Homomorphism[Powerset].from_channel(neighbours))
        >>> step.dom
        monad.algebra.Algebra[Powerset]('free', frozenset[int])
        >>> assert step.preservation(frozenset({
        ...     frozenset({0}), frozenset({0, 10})}))
        """
        monad = cls.monad
        assert_ismember(homomorphism, free.Homomorphism[monad])
        return cls(
            homomorphism.inside,
            Algebra[monad].free(homomorphism.dom),
            Algebra[monad].free(homomorphism.cod))

    def preservation(self, value) -> Equation:
        """
        The homomorphism law at a value ``m`` of ``M(A)``, i.e.
        :math:`h(a(m)) = b(M(h)(m))`.

        Parameters:
            value : The value ``m`` of ``M(A)``.
        """
        monad = type(self).monad
        return Equation(
            (self.dom.structure >> self.inside)(value),
            (monad(self.inside) >> self.cod.structure)(value))

    def __call__(self, value):
        return self.inside(value)

    def __repr__(self):
        return factory_name(type(self))\
            + f"({self.inside!r}, dom={self.dom!r}, cod={self.cod!r})"
