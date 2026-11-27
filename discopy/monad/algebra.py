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
isomorphic to the Kleisli category, see :mod:`~discopy.monad.free`. The
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

The product of algebras :math:`(A_1, a_1), \\dots, (A_n, a_n)` is the
product of their carriers with the structure computed pointwise

.. math::
    M(A_1 \\times \\dots \\times A_n) \\to A_1 \\times \\dots \\times A_n
    \\qquad
    m \\mapsto (a_1(M(\\pi_1)(m)), \\dots, a_n(M(\\pi_n)(m)))

see :meth:`Algebra.product`: the projections :math:`\\pi_i` are
homomorphisms and a tuple of homomorphisms into the factors is one into the
product, i.e. the forgetful functor to types creates products, for every
monad. The power :math:`A^n` is the product of ``n`` copies of ``A``, see
:meth:`Algebra.power`.

A value :math:`t \\in M(n)` of the monad over the outcomes
:math:`n = \\{0, \\dots, n - 1\\}` is an ``n``-ary :class:`Operation`: it
acts on every algebra ``(A, a)`` as the function

.. math::
    t_A : A^n \\to A
    \\qquad
    (x_0, \\dots, x_{n - 1}) \\mapsto a(M(i \\mapsto x_i)(t))

and every homomorphism ``h`` from ``(A, a)`` to ``(B, b)`` commutes with it,
:math:`h \\circ t_A = t_B \\circ h^n`, see :meth:`Operation.naturality`.
E.g. ``frozenset({0, 1})`` is binary choice, the join of every algebra of
:attr:`~discopy.kleisli.monad.Powerset`, and ``{(0, .5), (1, .5)}`` is a
fair coin, the midpoint of every algebra of
:attr:`~discopy.kleisli.monad.Subdistribution`.

The writer monad of a group ``G``, see :class:`Writer`, sends a type ``X``
to :math:`G \\times X`, with unit :math:`x \\mapsto (e, x)` and
multiplication :math:`(g, (h, x)) \\mapsto (gh, x)`. An algebra
:math:`a : G \\times A \\to A` is then unital, :math:`a(e, x) = x`, and
associative, :math:`a(gh, x) = a(g, a(h, x))`: it is exactly an action of
``G`` on ``A``, writing :math:`g \\cdot x = a(g, x)`, and a homomorphism is
exactly an equivariant function, :math:`h(g \\cdot x) = g \\cdot h(x)`.
On ``range(n)`` the actions are the :class:`~discopy.group.Action` of
``G``, see :meth:`Writer.from_action` and :meth:`Writer.to_action`. Since
``>>`` multiplies in diagrammatic order, ``gh`` is ``h >> g``.

Python functions cannot be compared, so the laws are checked at a point:
:meth:`Algebra.unitality`, :meth:`Algebra.associativity` and
:meth:`Homomorphism.preservation` return an
:class:`~discopy.testable.Equation` between two values, which holds when the
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
    Operation
    Writer
"""
from __future__ import annotations

from collections.abc import Callable

from discopy.abc import Category, NamedGeneric
from discopy.testable import Equation, TestableGeneric
from discopy.group import Action, FiniteGroup
from discopy.kleisli.monad import Monad
from discopy.monad import free
from discopy.python.finset import Permutation
from discopy.python.function import Function
from discopy.utils import (
    assert_iscomposable, assert_ismember, factory, factory_name)


class Algebra[monad](NamedGeneric):
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
            "free", monad(generators).inside[0],
            monad.mult(generators))  # ty: ignore[invalid-argument-type]

    @classmethod
    def product(cls, *algebras: Algebra) -> Algebra:
        """
        The product of algebras, i.e. the product of their carriers with the
        structure computed pointwise, through the projections.

        Parameters:
            algebras : The algebras to take the product of, none giving the
                terminal algebra on the empty tuple.

        Example
        -------
        >>> from discopy.kleisli.monad import Powerset
        >>> disjunction = Algebra[Powerset]("any", bool, any)
        >>> conjunction = Algebra[Powerset]("all", bool, all)
        >>> both = Algebra[Powerset].product(disjunction, conjunction)
        >>> both
        monad.algebra.Algebra[Powerset]('any × all', tuple[bool, bool])
        >>> both(frozenset({(True, True), (False, True)}))
        (True, True)
        """
        monad = cls.monad
        for algebra in algebras:
            assert_ismember(algebra, cls)
        carriers = tuple(algebra.carrier for algebra in algebras)
        carrier = tuple[carriers]  # ty: ignore[invalid-type-form]
        projections = tuple(
            monad(Function(lambda xs, i=i: xs[i], carrier, algebra.carrier))
            for i, algebra in enumerate(algebras))

        def structure(value):
            return tuple(
                algebra(projection(value))
                for algebra, projection in zip(algebras, projections))
        name = " × ".join(map(str, algebras)) or "1"
        return cls(name, carrier, structure)

    def power(self, n: int) -> Algebra:
        """
        The power :math:`A^n` of an algebra, i.e. the product of ``n`` copies.

        Parameters:
            n : The exponent, a non-negative integer.

        Raises:
            ValueError : When ``n`` is negative.

        Example
        -------
        >>> from discopy.kleisli.monad import Maybe
        >>> zero = Algebra[Maybe]("zero", int, lambda x: 0 if x is None else x)
        >>> zero.power(3)(None)
        (0, 0, 0)
        """
        if n < 0:
            raise ValueError(f"Expected a non-negative exponent, got {n}.")
        return type(self).product(*n * (self, ))

    def unitality(self, value) -> Equation:
        """
        The unit law at a value ``x`` of the carrier, i.e.
        :math:`a(\\eta(x)) = x`.

        Parameters:
            value : The value ``x`` of the carrier.
        """
        unit = type(self).monad.unit(
            self.carrier)  # ty: ignore[invalid-argument-type]
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
            (monad.mult(
                self.carrier)  # ty: ignore[invalid-argument-type]
             >> self.structure)(value),
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
class Homomorphism[monad](Category, TestableGeneric):
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
    @classmethod
    def checked_params(cls) -> tuple:
        """ None, a homomorphism of algebras having no strategy yet. """
        return ()

    ob = Algebra
    monad: Monad

    def __init__(self, inside: Callable, dom: Algebra, cod: Algebra):
        monad = type(self).monad
        assert_ismember(dom, Algebra[monad])  # ty: ignore[invalid-type-form]
        assert_ismember(cod, Algebra[monad])  # ty: ignore[invalid-type-form]
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
        homomorphisms = free.Homomorphism[
            monad]  # ty: ignore[invalid-type-form]
        algebras = Algebra[monad]  # ty: ignore[invalid-type-form]
        assert_ismember(homomorphism, homomorphisms)
        return cls(
            homomorphism.inside,
            algebras.free(homomorphism.dom),
            algebras.free(homomorphism.cod))

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


class Operation[monad](NamedGeneric):
    """
    An ``n``-ary operation of a monad ``M``, i.e. a value ``t`` of ``M(n)``
    over the outcomes ``range(n)``, acting on every algebra ``(A, a)`` as
    the function :math:`t_A : A^n \\to A` sending ``xs`` to
    :math:`a(M(i \\mapsto x_i)(t))`.

    Parameters:
        inside : The value ``t`` of ``M(n)``.
        arity : The number ``n`` of arguments.

    Example
    -------
    A fair coin is the midpoint of every expectation:

    >>> from discopy.kleisli.monad import Subdistribution
    >>> coin = Operation[Subdistribution](frozenset({(0, .5), (1, .5)}), 2)
    >>> mean = Algebra[Subdistribution](
    ...     "mean", float, lambda d: sum((p * x for x, p in d), 0.))
    >>> coin(mean)((2., 4.))
    3.0

    Doubling commutes with it, being a homomorphism:

    >>> double = Homomorphism[Subdistribution](lambda x: 2 * x, mean, mean)
    >>> assert coin.naturality(double, (2., 4.))
    """
    monad: Monad

    def __init__(self, inside, arity: int):
        self.inside, self.arity = inside, arity

    def __call__(self, algebra: Algebra) -> Function:
        monad = type(self).monad
        assert_ismember(
            algebra, Algebra[monad])  # ty: ignore[invalid-type-form]
        return Function(lambda xs: algebra(monad(Function(
            xs.__getitem__, int, algebra.carrier))(self.inside)),
            algebra.power(self.arity).carrier, algebra.carrier)

    def naturality(self, homomorphism: Homomorphism, value) -> Equation:
        """
        The commutation of a homomorphism ``h`` with the operation at a value
        ``xs`` of :math:`A^n`, i.e. :math:`h(t_A(xs)) = t_B(h^n(xs))`.

        Parameters:
            homomorphism : The homomorphism ``h`` from ``A`` to ``B``.
            value : The tuple ``xs`` of arguments.
        """
        assert_ismember(homomorphism, Homomorphism[
            type(self).monad])  # ty: ignore[invalid-type-form]
        h = homomorphism
        return Equation(
            h(self(h.dom)(value)), self(h.cod)(tuple(map(h, value))))

    def __eq__(self, other):
        return isinstance(other, Operation) and (
            self.monad, self.inside, self.arity) == (
                other.monad, other.inside, other.arity)

    def __hash__(self):
        return hash((self.monad, self.inside, self.arity))

    def __repr__(self):
        return factory_name(type(self)) + f"({self.inside!r}, {self.arity})"


class Writer(Monad):
    """
    The writer monad of a group ``G``, sending a type ``X`` to the type
    ``tuple[G, X]``, with unit :math:`x \\mapsto (e, x)` and multiplication
    :math:`(g, (h, x)) \\mapsto (gh, x)`, where ``gh`` is ``h >> g``.

    Its algebras are the actions of ``G`` and its homomorphisms the
    equivariant functions, see :meth:`from_action` and :meth:`to_action`.

    Parameters:
        group : The group ``G``.

    Example
    -------
    >>> from discopy.group import Cyclic
    >>> C4 = Writer(Cyclic[4])
    >>> C4
    Writer(group.Cyclic[4])
    >>> C4.unit(str)("egg")
    (Cyclic[4](0), 'egg')
    >>> C4.mult(str)((Cyclic[4](1), (Cyclic[4](2), "egg")))
    (Cyclic[4](3), 'egg')
    """
    def __init__(self, group: type[FiniteGroup]):
        def ob_map(X):
            return tuple[group, X]  # ty: ignore[invalid-type-form]
        super().__init__(
            f"Writer[{factory_name(group)}]",
            ob_map=ob_map,
            lift=lambda f: Function(
                lambda gx: (gx[0], f(gx[1])),
                ob_map(f.dom.inside[0]), ob_map(f.cod.inside[0])),
            unit_map=lambda X: Function(
                lambda x: (
                    group.id(), x),  # ty: ignore[missing-argument]
                X, ob_map(X)),
            mult_map=lambda X: Function(
                lambda ghx: (ghx[1][0] >> ghx[0], ghx[1][1]),
                ob_map(ob_map(X)), ob_map(X)))
        self.group = group

    def from_action(self, action: Action) -> Algebra:
        """
        The algebra on ``int`` of an action on ``range(action.size)``,
        sending ``(g, x)`` to ``action(g)[x]``, named by the action.

        Parameters:
            action : The action of the group.

        Example
        -------
        >>> from discopy.group import Cyclic
        >>> rotation = Writer(Cyclic[4]).from_action(Cyclic[4].action())
        >>> rotation((Cyclic[4](3), 2))
        1
        >>> assert rotation.associativity((Cyclic[4](3), (Cyclic[4](2), 1)))
        """
        if action.group is not self.group:
            raise TypeError(f"Expected an action of {self.group}.")
        return Algebra[self](
            repr(action), int, lambda gx: action(gx[0])[gx[1]])

    def to_action(self, algebra: Algebra, size: int) -> Action:
        """
        The action on ``range(size)`` of an algebra on ``int``, each
        generator ``g`` acting by the permutation ``x ↦ a(g, x)``, inverse
        to :meth:`from_action`: the algebra laws make it a bijection, with
        inverse ``x ↦ a(g.dagger(), x)``.

        Parameters:
            algebra : The algebra, whose carrier contains ``range(size)``.
            size : The number of points acted on.

        Example
        -------
        >>> from discopy.group import Cyclic
        >>> C4 = Writer(Cyclic[4])
        >>> flip = Algebra[C4]("flip", int, lambda gx: (gx[1] + gx[0].k) % 2)
        >>> C4.to_action(flip, 2)
        Action(group.Cyclic[4], 2, ((1, 0),))
        """
        assert_ismember(
            algebra, Algebra[self])
        return Action(self.group, size, tuple(
            Permutation(tuple(algebra((g, x)) for x in range(size)), size)
            for g in self.group.generators()))

    def __repr__(self):
        return f"Writer({factory_name(self.group)})"
