# -*- coding: utf-8 -*-

"""
The category of finite sets implemented as Python lists.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Function
    Cycles
    Permutation
    Cospan
"""

from __future__ import annotations
from discopy.utils import assert_isinstance
from typing import Any, Iterable, Self, overload
from itertools import accumulate
from collections.abc import Sequence
from operator import index

from dataclasses import dataclass

from discopy import messages
from discopy.abc import (
    Groupoid, HypergraphCategory, MarkovCategory, PROP, Nat)
from discopy.testable import Testable
from discopy.utils import AxiomError, factory_name, unbiased


@dataclass
class Function(MarkovCategory[Nat, "Function"], Sequence):
    """
    A function between finite sets encoded as a Python list, read backwards
    as an arrow from ``dom`` to ``cod``, i.e. an arrow of the free Markov
    category on one object: output ``i`` is a copy of input ``inside[i]``.

    Functions implement the standard Python sequence protocol.

    Parameters:
        inside : The list from ``range(cod)`` to ``range(dom)``.
        dom : The size of domain of the function.
        cod : The size of codomain of the function.

    .. admonition:: Summary

        .. autosummary::

            id
            then
            tensor
            swap
            is_swap
            copy
            factorise
    """
    inside: list[int]
    dom: Nat
    cod: Nat

    ob = Nat

    def __post_init__(self):
        self.dom = self.dom if isinstance(self.dom, Nat) else Nat(self.dom)
        self.cod = self.cod if isinstance(self.cod, Nat) else Nat(self.cod)
        if isinstance(self.inside, dict):
            self.inside = [self.inside[i] for i in range(len(self.cod))]
        else:
            self.inside = list(self.inside)
        if len(self.inside) != len(self.cod):
            raise ValueError
        if any(not 0 <= i < len(self.dom) for i in self.inside):
            raise ValueError

    def __getitem__(self, key):
        return self.inside[key]

    def __len__(self) -> int:
        return len(self.cod)

    @staticmethod
    def id(x: int | Nat = 0):
        return Function(list(range(x)), x, x)

    def then(self, *others: Function) -> Function:
        """
        The composite ``self ; others``, read off backwards, i.e. each
        index of the last codomain walked back through every function.

        Raises:
            AxiomError : When two adjacent functions do not compose.
        """
        if not others:
            return self
        factors = (self, ) + others
        for left, right in zip(factors, factors[1:]):
            if left.cod.n != right.dom.n:
                raise AxiomError(messages.NOT_COMPOSABLE.format(
                    left, right, left.cod, right.dom))
        inside = factors[-1].inside
        for factor in reversed(factors[:-1]):
            image = factor.inside
            inside = [image[i] for i in inside]
        return Function(inside, self.dom, factors[-1].cod)

    def tensor(self, *others: Function) -> Function:
        """
        The disjoint union of ``n`` functions, i.e. their lists concatenated
        with each shifted by the domains before it.

        >>> unit = Function([0], 1, 1)
        >>> assert unit.tensor(unit) == Function([0, 1], 2, 2)
        """
        if not others:
            return self
        inside, shift = [], 0
        for factor in (self, ) + others:
            inside += [shift + i for i in factor.inside]
            shift += int(factor.dom)
        return Function(
            inside,
            self.dom.tensor(*(other.dom for other in others)),
            self.cod.tensor(*(other.cod for other in others)))

    @staticmethod
    def swap(x: int | Nat, y: int | Nat) -> Function:
        """
        The symmetry of ``x @ y``, which sends each output of ``y @ x`` to
        the input it reads: the ``y`` that come after the ``m = x`` inputs,
        then the ``x``.

        >>> assert Function.swap(2, 1) == Function([2, 0, 1], 3, 3)
        >>> assert Function.swap(2, 1) >> Function.swap(1, 2) == Function.id(3)
        """
        m, n = int(x), int(y)
        return Function(
            list(range(m, m + n)) + list(range(m)), Nat(m + n), Nat(m + n))

    def is_swap(self: Function | Sequence[int]) -> bool:
        """
        Whether this is the permutation ``(1, 0)``, callable on a raw
        sequence as well as on a :class:`Function` with a two-wire domain.
        """
        dom = getattr(self, "dom", None)
        return (dom is None or len(dom) == 2)\
            and len(self) == 2 and self[0] == 1 and self[1] == 0

    @classmethod
    def permutation(
            cls, xs: Sequence[int], doms: Sequence[int | Nat]) -> Function:
        """
        The function that permutes blocks of elements, the ``i``-th block of
        its codomain being the ``xs[i]``-th block of its domain, of size
        ``doms[xs[i]]``.

        Parameters:
            xs : The permutation of the blocks.
            doms : The size of each block.

        >>> assert Function.permutation([1, 0], [2, 3]) == Function.swap(2, 3)
        """
        sizes = [int(d) for d in doms]
        xs = Permutation(xs, len(sizes))
        offsets = [0, *accumulate(sizes)]
        inside = [offsets[i] + j for i in xs for j in range(sizes[i])]
        return Function(inside, Nat(offsets[-1]), Nat(offsets[-1]))

    @staticmethod
    def copy(x: int | Nat, n=2) -> Function:
        k, n = int(x), index(n)
        return Function([i % k for i in range(n * k)], Nat(k), Nat(n * k))

    def factorise(self) -> tuple[list[int], Permutation]:
        """
        The copies then the permutation that the function factors as, i.e.
        how many outputs read each input and the permutation that puts each
        of its copies in their place, the inputs copied in order.

        >>> f = Function([1, 0, 1], 2, 3)
        >>> counts, perm = f.factorise()
        >>> counts, perm
        ([1, 2], python.finset.Permutation((1, 0, 2)))
        >>> copies = [Function.copy(1, n) for n in counts]
        >>> assert copies[0].tensor(*copies[1:]) >> perm == f
        """
        counts = [self.inside.count(j) for j in range(len(self.dom))]
        offsets = [sum(counts[:j]) for j in range(len(self.dom))]
        perm = []
        for i in self.inside:
            perm.append(offsets[i])
            offsets[i] += 1
        return counts, Permutation(perm, len(self))


type Cycle = Iterable[int]
type Cycles = Iterable[Cycle]


class Permutation(Function, PROP, Groupoid):
    """
    A permutation of a finite set, seen as a bijective finite-set function,
    i.e. an arrow of the groupoid of finite sets and bijections.

    A permutation is represented by its action on ``range(n)``.

    Examples
    --------
    >>> Permutation((1, 0, 3, 2)).cycles()
    ((0, 1), (2, 3))
    >>> Permutation.from_cycles([(0, 1), (2, 3)], 4)
    python.finset.Permutation((1, 0, 3, 2))
    >>> Permutation((1, 0)).is_fixpoint_free_involution()
    True
    """
    def __init__(self, inside=(), size: int | None = None):
        inside = tuple(inside)
        if size is None:
            size = len(inside)
        else:
            assert_isinstance(size, int)
        if len(inside) != size:
            raise ValueError(
                messages.WRONG_PERMUTATION.format(size, len(inside))
            )
        if sorted(inside) != list(range(size)):
            raise ValueError(messages.NOT_A_PERMUTATION.format(size, inside))
        super().__init__(list(inside), Nat(size), Nat(size))

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    @overload
    def __getitem__(self, key: int) -> int: ...

    @overload
    def __getitem__(self, key: slice) -> tuple[int, ...]: ...

    def __getitem__(self, key: int | slice) -> int | tuple[int, ...]:
        if isinstance(key, slice):
            return tuple(self)[key]
        return super().__getitem__(key)

    def __repr__(self) -> str:
        return factory_name(type(self)) + f"({tuple(self)!r})"

    def __str__(self) -> str:
        return str(tuple(self))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Permutation):
            return tuple(self) == tuple(other)
        if isinstance(other, tuple):
            return tuple(self) == other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(tuple(self))

    @classmethod
    def id(cls, dom: int | Nat = 0) -> Self:
        """ The identity permutation on ``range(size)``. """
        n = int(dom)
        return cls(range(n), n)

    identity = id

    @property
    def is_identity(self) -> bool:
        """ Whether this is the identity permutation. """
        return list(self) == list(range(len(self)))

    @classmethod
    def from_cycles(cls, cycles: Cycles, size: int) -> Self:
        """ Build a permutation from cycles. """
        result = list(range(size))
        seen = set()
        for cycle in map(tuple, cycles):
            if len(set(cycle)) != len(cycle):
                raise ValueError
            for i in cycle:
                if i < 0 or i >= size or i in seen:
                    raise ValueError
                seen.add(i)
            for source, target in zip(cycle, cycle[1:] + cycle[:1]):
                result[source] = target
        return cls(result, size)

    @classmethod
    def from_transpositions(cls, transpositions: Cycles, size: int) -> Self:
        """ Build a permutation from disjoint 2-cycles. """
        result = list(range(size))
        seen = set()
        for left, right in transpositions:
            if left == right:
                raise ValueError
            if left < 0 or right < 0 or left >= size or right >= size:
                raise ValueError
            if left in seen or right in seen:
                raise ValueError
            seen.update([left, right])
            result[left], result[right] = right, left
        return cls(result, size)

    @classmethod
    def gather(cls, xs: Sequence[int], size: int, offset: int = 0) -> Self:
        """
        Gather the inputs ``xs`` at the consecutive outputs from ``offset``,
        in the order given, the other inputs keeping their order, i.e.
        ``result[offset + i] == xs[i]``.

        This is the wire routing of
        :meth:`discopy.hypergraph.Hypergraph.to_diagram` and
        :meth:`discopy.cmap.CMap.to_diagram`, which gather the domain wires
        of each box at its offset.

        >>> Permutation.gather([3, 1], 5, offset=1)
        python.finset.Permutation((0, 3, 1, 2, 4))
        >>> assert Permutation.gather([], 3) == Permutation.id(3)
        """
        if not 0 <= offset <= size - len(xs):
            raise ValueError
        rest = [i for i in range(size) if i not in xs]
        return cls(rest[:offset] + list(xs) + rest[offset:], size)

    def cycles(self) -> Cycles:
        """ Return the cycles of the permutation. """
        result, seen = [], set()
        for i in range(len(self)):
            if i in seen:
                continue
            result.append(self.cycle(i, seen))
        return tuple(result)

    def cycle(
            self, start: int,
            seen: set[int] | None = None) -> Cycle:
        """ Return the cycle reached from ``start``. """
        if start < -len(self) or start >= len(self):
            raise ValueError
        start %= len(self)
        cycle, local_seen, i = [], set() if seen is None else seen, start
        while i not in local_seen:
            local_seen.add(i)
            cycle.append(i)
            i = self[i]
        return tuple(cycle)

    def then(self, *others: Self) -> Self:
        """ Return ``self ; others``, i.e. each index pushed through them
        in turn, ``result[i] == other[self[i]]`` for a single ``other``. """
        if not others:
            return self
        inside = self.inside
        for other in others:
            image = type(self)(other, len(self)).inside
            inside = [image[i] for i in inside]
        return type(self)(inside, len(self))

    def dagger(self) -> Self:
        """ Return the inverse permutation. """
        result = list(range(len(self)))
        for source, target in enumerate(self):
            result[target] = source
        return type(self)(result, len(self))

    def rotate(self) -> Self:
        """ Rotate by reversing and inverting the permutation. """
        reverse = type(self)(reversed(range(len(self))))
        return self.dagger().conjugate(reverse)

    def conjugate(self, other: Self) -> Self:
        """ Return ``other^-1 ; self ; other``. """
        other = type(self)(other, len(self))
        return other.dagger().then(self).then(other)

    def tensor(self, *others) -> Self:
        """
        The disjoint union of ``n`` permutations.

        >>> swap = Permutation([1, 0])
        >>> assert swap.tensor(swap) == Permutation([1, 0, 3, 2])
        """
        if not others:
            return self
        inside, shift = list(self), len(self)
        for other in others:
            other = type(self)(other)
            inside += [shift + i for i in other]
            shift += len(other)
        return type(self)(inside, shift)

    def embed(self, injection: Iterable[int], size: int) -> Self:
        """ Embed into ``range(size)`` along ``injection``. """
        injection = tuple(injection)
        complement = tuple(i for i in range(size) if i not in injection)
        relabeling = type(self)(injection + complement, size)
        embedded = self.tensor(type(self).id(size - len(self)))
        return embedded.conjugate(relabeling)

    def coequalizer(self, *others: Self) -> dict[int, int]:
        """
        Coequalize permutations by quotienting generated orbits.

        This computes the quotient of ``range(len(self))`` by the equivalence
        relation generated by ``i ~ self[i]`` and ``i ~ other[i]`` for each
        of the ``others``.

        Equivalently, it returns the orbits of the subgroup generated by the
        permutations. The returned quotient map ``q`` satisfies
        ``q[i] == q[self[i]]`` and ``q[i] == q[other[i]]`` for every element
        ``i``, and is minimal with this property. Components are numbered in
        the order of their least element.

        Parameters:
            others : The other permutations, with the same size as ``self``.

        Returns:
            A dictionary mapping each element to its quotient component.

        Example
        -------
        >>> Permutation((1, 0, 2, 3)).coequalizer((0, 1, 3, 2))
        {0: 0, 1: 0, 2: 1, 3: 1}
        >>> Permutation((1, 2, 0, 3)).coequalizer()
        {0: 0, 1: 0, 2: 0, 3: 1}
        """
        others = tuple(type(self)(other, len(self)) for other in others)
        component_of = {}
        component = 0
        for start in range(len(self)):
            if start in component_of:
                continue
            stack = [start]
            while stack:
                i = stack.pop()
                if i in component_of:
                    continue
                component_of[i] = component
                for j in [self[i]] + [other[i] for other in others]:
                    if j not in component_of:
                        stack.append(j)
            component += 1
        return component_of

    @classmethod
    def swap(cls, left: int | Nat, right: int | Nat) -> Self:
        m, n = int(left), int(right)
        inside = tuple(
            i + n if i < m else i - m
            for i in range(m + n))
        return cls(inside, m + n)

    def trace(self, n: int = 1, left: bool = False) -> Self:
        raise NotImplementedError

    def is_fixpoint_free_involution(self) -> bool:
        """ Whether this is a product of disjoint 2-cycles. """
        return all(self[i] != i and self[self[i]] == i
                   for i in range(len(self)))


@dataclass
class Cospan(HypergraphCategory, Testable["Cospan"]):
    """
    A cospan of finite sets ``dom -> apex <- cod``, i.e. an arrow of the free
    hypergraph category on one object, the PROP of special commutative
    Frobenius algebras, see :cite:t:`Lack04`. Its composition is the pushout,
    its tensor the disjoint union.

    The lists ``left`` and ``right`` send each element of the domain and of
    the codomain to a point of the apex, i.e. to a spider with these legs. The
    points are numbered in the order the legs first reach them, the domain
    then the codomain, and those no leg reaches, the spiders with no legs,
    come last: two cospans are equal exactly when they are isomorphic.

    Parameters:
        left : The point of each element of the domain.
        right : The point of each element of the codomain.
        apex : The number of points, by default those the legs reach.

    Example
    -------
    >>> Cospan.copy(1)
    Cospan(left=[0], right=[0, 0], apex=1)
    >>> assert Cospan.copy(1) >> Cospan.merge(1) == Cospan.id(1)
    >>> circle = Cospan.caps(1, 1) >> Cospan.cups(1, 1)
    >>> assert circle == Cospan([], [], 1) != Cospan.id(0)
    """
    left: list[int]
    right: list[int]
    apex: int | None = None

    ob = Nat

    def __post_init__(self):
        left, right = list(self.left), list(self.right)
        apex = max(left + right, default=-1) + 1 if self.apex is None\
            else int(self.apex)
        if apex < 0 or any(not 0 <= i < apex for i in left + right):
            raise ValueError(f"Expected points in range({apex}).")
        order = {k: n for n, k in enumerate(self.order(left, right, apex))}
        self.left, self.right = [order[i] for i in left], [
            order[j] for j in right]
        self.apex = apex

    @staticmethod
    def order(left: list[int], right: list[int], apex: int) -> list[int]:
        """
        The points of an apex in the order the legs first reach them, the
        domain then the codomain, those no leg reaches last.

        >>> Cospan.order([2, 2], [0], 4)
        [2, 0, 1, 3]
        """
        return list(dict.fromkeys([*left, *right, *range(apex)]))

    @property
    def dom(self) -> Nat:
        """ The size of the domain, i.e. of the left leg. """
        return Nat(len(self.left))

    @property
    def cod(self) -> Nat:
        """ The size of the codomain, i.e. of the right leg. """
        return Nat(len(self.right))

    @classmethod
    def id(cls, dom: int | Nat = 0) -> Cospan:
        return cls(list(range(int(dom))), list(range(int(dom))), int(dom))

    def pushout(self, other: Cospan) -> tuple[Cospan, list[int]]:
        """
        The composite with ``other`` and the point of the composite that each
        point of the two apexes goes to, the points of ``self`` first: the
        points that the codomain of ``self`` and the domain of ``other`` share
        are glued together.

        Parameters:
            other : The cospan to compose with.

        >>> merge, copy = Cospan.merge(1), Cospan.copy(1)
        >>> merge.pushout(copy)
        (Cospan(left=[0, 0], right=[0, 0], apex=1), [0, 0])
        """
        if self.cod != other.dom:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                self, other, self.cod, other.dom))
        parent = list(
            range(self.apex + other.apex))

        def find(i):
            while parent[i] != i:
                parent[i] = i = parent[parent[i]]
            return i
        for i, j in zip(self.right, other.left):
            parent[find(i)] = find(self.apex + j)
        roots = list(map(find, range(len(parent))))
        left = [roots[i] for i in self.left]
        right = [roots[self.apex + j] for j in other.right]
        order = {k: n for n, k in enumerate(
            dict.fromkeys(left + right + roots))}
        return type(self)(
            [order[k] for k in left], [order[k] for k in right], len(order)
        ), [order[k] for k in roots]

    @unbiased
    def then(self, other: Cospan) -> Cospan:
        return self.pushout(other)[0]

    def tensor(self, *others: Cospan) -> Cospan:
        """
        The disjoint union of cospans.

        >>> assert Cospan.copy(1).tensor(Cospan.id(1)) == Cospan.copy(1) @ 1
        """
        left, right, apex = list(self.left), list(self.right), self.apex
        for other in others:
            left += [apex + i for i in other.left]
            right += [apex + j for j in other.right]
            apex += other.apex
        return type(self)(left, right, apex)

    @classmethod
    def swap(cls, left: int | Nat, right: int | Nat) -> Cospan:
        m, n = int(left), int(right)
        return cls(
            list(range(m + n)), list(range(m, m + n)) + list(range(m)), m + n)

    @classmethod
    def spiders(cls, n_legs_in: int, n_legs_out: int, typ: int | Nat
                ) -> Cospan:
        """
        One spider on each element of ``typ``, with ``n_legs_in`` legs in and
        ``n_legs_out`` out, interleaved as ``typ ** n``.

        >>> Cospan.spiders(2, 0, 2)
        Cospan(left=[0, 1, 0, 1], right=[], apex=2)
        """
        k = int(typ)
        n_legs_in, n_legs_out = index(n_legs_in), index(n_legs_out)
        return cls([i % k for i in range(n_legs_in * k)],
                   [j % k for j in range(n_legs_out * k)], k)

    @classmethod
    def copy(cls, x: int | Nat, n: int = 2) -> Cospan:
        """ The spiders with one leg in and ``n`` out on ``x``. """
        return cls.spiders(1, n, x)

    @classmethod
    def merge(cls, x: int | Nat, n: int = 2) -> Cospan:
        """ The spiders with ``n`` legs in and one out on ``x``. """
        return cls.spiders(n, 1, x)

    @classmethod
    def cups(cls, left: int | Nat, right: int | Nat) -> Cospan:
        """
        The nested cups of ``left`` and ``right``, of the same size.

        >>> Cospan.cups(2, 2)
        Cospan(left=[0, 1, 1, 0], right=[], apex=2)
        """
        if int(left) != int(right):
            raise AxiomError(messages.NOT_ADJOINT.format(left, right))
        k = int(left)
        return cls(list(range(k)) + list(reversed(range(k))), [], k)

    @classmethod
    def caps(cls, left: int | Nat, right: int | Nat) -> Cospan:
        return cls.cups(left, right).dagger()

    def dagger(self) -> Cospan:
        """ The cospan the other way round, i.e. its legs swapped. """
        return type(self)(self.right, self.left, self.apex)

    def trace(self, n: int = 1, left: bool = False) -> Cospan:
        """
        Glue the last ``n`` elements of the domain to the last ``n`` of the
        codomain, the first ``n`` if ``left``, with caps and cups.

        >>> assert Cospan.swap(1, 1).trace() == Cospan.id(1)
        >>> assert Cospan.id(1).trace() == Cospan([], [], 1)
        """
        x, dom, cod = Nat(n), self.dom[:len(self.dom) - n], self.cod[
            :len(self.cod) - n]
        if left:
            return self.caps(x, x) @ dom >> x @ self >> self.cups(x, x) @ cod
        return dom @ self.caps(x, x) >> self @ x >> cod @ self.cups(x, x)

    @classmethod
    def strategy(cls, *, dom=None, cod=None, max_size=3):
        """
        Generate cospans between the given sizes, or sizes up to
        ``max_size``, with an apex of up to ``max_size`` points.
        """
        from hypothesis import strategies as st
        sizes = Nat.strategy(max_size=max_size)

        @st.composite
        def cospans(draw):
            m, n = (int(draw(sizes) if x is None else x) for x in (dom, cod))
            least = 1 if m or n else 0
            apex = draw(st.integers(least, max(max_size, least)))
            points = st.integers(0, max(apex - 1, 0))
            return cls(draw(st.lists(points, min_size=m, max_size=m)),
                       draw(st.lists(points, min_size=n, max_size=n)), apex)
        return cospans()

    serialisation = Testable.serialisation.inapplicable(
        "A cospan of finite sets has no tree.")
