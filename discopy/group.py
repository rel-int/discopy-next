# -*- coding: utf-8 -*-

"""
Finite groups as normal forms, built from cyclic and symmetric groups by
direct and wreath products, and their actions on finite sets.

A group is a :class:`discopy.abc.Group`, i.e. a groupoid with one object:
its elements are its arrows, ``g >> h`` multiplies them in diagrammatic
order, first ``g`` then ``h``, and ``g.dagger()`` is the inverse. A
:class:`Group` comes with its :meth:`Group.generators`, the
:meth:`Group.word` of each element in them, its :meth:`Group.order`, its
:meth:`Group.elements` and a uniform :meth:`Group.sample`, each computed
from the normal form of its elements, a pair for a product and a tuple and
an element of the top group for a wreath product, so that a group with a
trillion elements is as cheap as its factors.

An :class:`Action` of a group on ``range(size)`` is a functor into
:class:`discopy.python.finset.Permutation`, given by the images of the
generators and extended along words. Each group acts on a set of its own,
:meth:`Group.action`, and a product or a wreath product acts through
actions of its factors. The :meth:`Action.orbits` are computed from the
generators alone: those of the :meth:`Action.tensor` of two actions, on
pairs, are what an equivariant linear map between them is constant on.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Group
    Cyclic
    Symmetric
    Product
    Wreath
    Action

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        kronecker

Example
-------
The symmetries of a sudoku grid permute the rows within each band of three
and the bands themselves, likewise the columns, transpose the grid and
relabel the digits independently. The rows form the wreath product
:math:`S_3 \\wr S_3` acting on the nine lines, the cells its wreath
product with :math:`C_2` acting on pairs of lines, which the transposition
exchanges.

>>> lines = Wreath[Symmetric[3], Symmetric[3]]
>>> cells = Wreath[lines, Cyclic[2]]
>>> sudoku = Product[cells, Symmetric[9]]
>>> sudoku.order()
1218998108160
>>> action = Product[cells, Symmetric[9]].action(
...     cells.action(power=True), Symmetric[9].action())
>>> action.size, len(action.orbits())
(729, 1)
"""

from __future__ import annotations

import itertools
import math
import random
from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar, Self

from discopy import abc
from discopy.axioms import Testable
from discopy.python.finset import Permutation
from discopy.utils import NamedGeneric, assert_ismember, factory_name


NO_TREE = "A group has no tree: utils.resolve_factory reads one class "\
    "as the parameter of a class, not an integer nor several groups."


def kronecker(left: Permutation, right: Permutation) -> Permutation:
    """
    The permutation of the pairs ``(i, j)``, numbered ``i * len(right) +
    j``, sending each to ``(left[i], right[j])``, i.e. the Kronecker product
    of permutation matrices.

    Parameters:
        left : The permutation of the first coordinate.
        right : The permutation of the second coordinate.

    Example
    -------
    >>> kronecker(Permutation((1, 0)), Permutation((1, 2, 0)))
    (4, 5, 3, 1, 2, 0)
    """
    m = len(right)
    return Permutation(
        (left[i] * m + right[j] for i in range(len(left)) for j in range(m)),
        len(left) * m)


class Group(Testable, abc.Group):
    """
    A finite group whose elements are normal forms.

    A concrete group implements :meth:`multiply`, :meth:`dagger`,
    :meth:`generators`, :meth:`word`, :meth:`order`, :meth:`elements`,
    :meth:`sample` and :meth:`action`; :meth:`tensor`, i.e. ``>>`` and
    ``@``, folds :meth:`multiply` and the elements Hypothesis generates are
    samples.
    """
    inside: tuple

    @abstractmethod
    def multiply(self, other: Self) -> Self:
        """ The product of two elements, first ``self`` then ``other``. """

    def tensor(self, *others: Self) -> Self:
        result = self
        for other in others:
            assert_ismember(other, type(self))
            result = result.multiply(other)
        return result

    @classmethod
    @abstractmethod
    def generators(cls) -> tuple[Self, ...]:
        """ The generators of the group, in a fixed order. """

    @abstractmethod
    def word(self) -> tuple[int, ...]:
        """
        The indices of :meth:`generators` whose product, in order, is the
        element: an :class:`Action` maps it along its word.
        """

    @classmethod
    @abstractmethod
    def order(cls) -> int:
        """ The number of elements of the group. """

    @classmethod
    @abstractmethod
    def elements(cls) -> Iterator[Self]:
        """ The elements of the group, each one once. """

    @classmethod
    @abstractmethod
    def sample(cls, rng: random.Random) -> Self:
        """ An element drawn uniformly at random. """

    @classmethod
    @abstractmethod
    def action(cls) -> Action:
        """ The action of the group on a set of its own. """

    @classmethod
    def strategy(cls, dom=None, cod=None):
        """
        Generate elements by :meth:`sample`, from a random generator under
        the control of Hypothesis. The only object, :obj:`None`, is the
        domain and codomain of each.
        """
        from hypothesis import strategies as st

        if (dom, cod) != (None, None):
            raise ValueError("A group has a single object, None.")
        return st.randoms(use_true_random=False).map(cls.sample)

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(map(repr, self.inside))})"

    def __eq__(self, other):
        return type(self) is type(other) and self.inside == other.inside

    def __hash__(self):
        return hash(self.inside)

    serialisation = Testable.serialisation.inapplicable(NO_TREE)


class Cyclic(NamedGeneric["n"], Group):
    """
    The cyclic group of order ``n``, i.e. the integers modulo ``n`` under
    addition, generated by one and acting on ``range(n)`` by rotation.

    Parameters:
        k : The integer, reduced modulo ``n``.

    Example
    -------
    >>> Cyclic[4](3) >> Cyclic[4](2)
    Cyclic[4](1)
    >>> Cyclic[4](3).dagger(), Cyclic[4](3).word()
    (Cyclic[4](1), (0, 0, 0))
    """
    n: ClassVar[int]

    def __init__(self, k: int = 0):
        if self.n is None:
            raise TypeError(
                f"Expected a subscript, e.g. {type(self).__name__}[2].")
        if self.n < 1:
            raise ValueError(f"Expected a positive order, got {self.n}.")
        self.inside = (k % self.n, )

    @property
    def k(self) -> int:
        """ The integer modulo ``n``. """
        return self.inside[0]

    def multiply(self, other):
        return type(self)(self.k + other.k)

    def dagger(self):
        return type(self)(-self.k)

    @classmethod
    def generators(cls):
        return (cls(1), )

    def word(self):
        return self.k * (0, )

    @classmethod
    def order(cls):
        return cls.n

    @classmethod
    def elements(cls):
        return map(cls, range(cls.n))

    @classmethod
    def sample(cls, rng):
        return cls(rng.randrange(cls.n))

    @classmethod
    def action(cls):
        rotation = Permutation(((i + 1) % cls.n for i in range(cls.n)), cls.n)
        return Action(cls, cls.n, (rotation, ))


class Symmetric(NamedGeneric["n"], Group):
    """
    The symmetric group on ``range(n)``, i.e. its permutations, generated by
    the transpositions of neighbours and acting on ``range(n)`` itself.

    Parameters:
        inside : The permutation, the identity by default.

    Example
    -------
    >>> g = Symmetric[3]((1, 2, 0))
    >>> g.word()
    (1, 0)
    >>> s0, s1 = Symmetric[3].generators()
    >>> assert g == s1 >> s0 and s0 == Symmetric[3]((1, 0, 2))
    """
    n: ClassVar[int]

    def __init__(self, inside: tuple[int, ...] = None):
        if self.n is None:
            raise TypeError(
                f"Expected a subscript, e.g. {type(self).__name__}[2].")
        inside = range(self.n) if inside is None else inside
        self.inside = Permutation(inside, self.n)

    def __repr__(self):
        return f"{type(self).__name__}({self.inside!r})"

    def multiply(self, other):
        return type(self)(self.inside.then(other.inside))

    def dagger(self):
        return type(self)(self.inside.dagger())

    @classmethod
    def generators(cls):
        return tuple(
            cls(Permutation.from_transpositions([(i, i + 1)], cls.n))
            for i in range(cls.n - 1))

    def word(self):
        """
        Bubble sort the permutation into the identity: each swap of
        neighbours ``i, i + 1`` is a transposition composed on the left,
        and the swaps in the order they are made multiply back to it.
        """
        inside, word = list(self.inside), []
        for end in range(self.n - 1, 0, -1):
            for i in range(end):
                if inside[i] > inside[i + 1]:
                    inside[i], inside[i + 1] = inside[i + 1], inside[i]
                    word.append(i)
        return tuple(word)

    @classmethod
    def order(cls):
        return math.factorial(cls.n)

    @classmethod
    def elements(cls):
        return map(cls, itertools.permutations(range(cls.n)))

    @classmethod
    def sample(cls, rng):
        return cls(rng.sample(range(cls.n), cls.n))

    @classmethod
    def action(cls):
        return Action(cls, cls.n, tuple(g.inside for g in cls.generators()))


class Product(NamedGeneric["left", "right"], Group):
    """
    The direct product of two groups, i.e. their pairs multiplied
    componentwise, generated by the generators of each factor and acting
    on the pairs of points.

    Parameters:
        inside : An element of each factor, the identities by default.

    Example
    -------
    >>> G = Product[Cyclic[2], Symmetric[3]]
    >>> g = G(Cyclic[2](1), Symmetric[3]((1, 2, 0)))
    >>> g.word(), g >> g == G(Cyclic[2](0), Symmetric[3]((2, 0, 1)))
    ((0, 2, 1), True)
    >>> G.order(), G.action().size
    (12, 6)
    """
    left: ClassVar[type[Group]]
    right: ClassVar[type[Group]]

    def __init__(self, *inside: Group):
        if self.left is None:
            raise TypeError(
                f"Expected subscripts, e.g. {type(self).__name__}[G, H].")
        self.inside = inside or (self.left(), self.right())
        if len(self.inside) != 2:
            raise ValueError("Expected an element of each factor.")
        for element, group in zip(self.inside, (self.left, self.right)):
            assert_ismember(element, group)

    def multiply(self, other):
        return type(self)(*(
            x >> y for x, y in zip(self.inside, other.inside)))

    def dagger(self):
        return type(self)(*(x.dagger() for x in self.inside))

    @classmethod
    def generators(cls):
        return tuple(cls(g, cls.right()) for g in cls.left.generators())\
            + tuple(cls(cls.left(), h) for h in cls.right.generators())

    def word(self):
        g, h = self.inside
        shift = len(self.left.generators())
        return g.word() + tuple(shift + i for i in h.word())

    @classmethod
    def order(cls):
        return cls.left.order() * cls.right.order()

    @classmethod
    def elements(cls):
        return itertools.starmap(cls, itertools.product(
            cls.left.elements(), cls.right.elements()))

    @classmethod
    def sample(cls, rng):
        return cls(cls.left.sample(rng), cls.right.sample(rng))

    @classmethod
    def action(cls, left: Action = None, right: Action = None) -> Action:
        """
        The action on the pairs of points of an action of each factor, the
        action of the factor on a set of its own by default.

        Parameters:
            left : An action of the left factor.
            right : An action of the right factor.
        """
        left = cls.left.action() if left is None else left
        right = cls.right.action() if right is None else right
        if (left.group, right.group) != (cls.left, cls.right):
            raise TypeError(f"Expected actions of {cls.left}, {cls.right}.")
        return Action(cls, left.size * right.size, tuple(
            kronecker(p, Permutation.id(right.size)) for p in left.images
        ) + tuple(
            kronecker(Permutation.id(left.size), p) for p in right.images))


class Wreath(NamedGeneric["base", "top"], Group):
    """
    The wreath product of a group ``base`` by a group ``top`` acting on
    ``range(k)`` by its own :meth:`Group.action`, i.e. a ``k``-tuple of
    elements of ``base`` followed by an element of ``top`` permuting them:
    the semidirect product of the ``k``-th power of ``base`` by ``top``.

    The pair ``(f, h)`` followed by ``(f', h')`` is ``(f'', h >> h')``
    where ``f''[j] == f[j] >> f'[p[j]]`` and ``p`` is the permutation
    ``h`` acts by. It is generated by the generators of ``base`` at each
    position and those of ``top``.

    Parameters:
        inside : A ``k``-tuple of elements of ``base`` and an element of
            ``top``, the identities by default.

    Example
    -------
    The hyperoctahedral group, i.e. the signed permutations or the
    symmetries of a cube, is the wreath product of the cyclic group of
    order two by a symmetric group.

    >>> G = Wreath[Cyclic[2], Symmetric[3]]
    >>> G.order()
    48
    >>> one, zero = Cyclic[2](1), Cyclic[2](0)
    >>> g = G((one, zero, zero), Symmetric[3]((1, 2, 0)))
    >>> (g >> g).inside
    ((Cyclic[2](1), Cyclic[2](0), Cyclic[2](1)), Symmetric[3]((2, 0, 1)))
    >>> g >> g.dagger() == G()
    True
    """
    base: ClassVar[type[Group]]
    top: ClassVar[type[Group]]

    def __init__(self, base: tuple[Group, ...] = None, top: Group = None):
        if self.base is None:
            raise TypeError(
                f"Expected subscripts, e.g. {type(self).__name__}[G, H].")
        base = self.degree() * (self.base(), ) if base is None else base
        top = self.top() if top is None else top
        if len(base) != self.degree():
            raise ValueError(f"Expected {self.degree()} elements of base.")
        for element in base:
            assert_ismember(element, self.base)
        assert_ismember(top, self.top)
        self.inside = (tuple(base), top)

    def __repr__(self):
        base, top = self.inside
        return f"{type(self).__name__}({base!r}, {top!r})"

    @classmethod
    def degree(cls) -> int:
        """ The number of copies of ``base``, which ``top`` permutes. """
        return cls.top.action().size

    def permutation(self) -> Permutation:
        """ The permutation of the copies of ``base`` the element makes. """
        return self.top.action()(self.inside[1])

    def multiply(self, other):
        (f, h), (g, k), p = self.inside, other.inside, self.permutation()
        return type(self)(
            tuple(f[j] >> g[p[j]] for j in range(len(f))), h >> k)

    def dagger(self):
        (f, h), p = self.inside, self.permutation().dagger()
        return type(self)(tuple(f[p[i]].dagger() for i in range(len(f))),
                          h.dagger())

    @classmethod
    def generators(cls):
        unit = cls.degree() * (cls.base(), )
        return tuple(
            cls(unit[:j] + (g, ) + unit[j + 1:])
            for j in range(cls.degree()) for g in cls.base.generators()
        ) + tuple(cls(unit, h) for h in cls.top.generators())

    def word(self):
        (f, h), shift = self.inside, len(self.base.generators())
        return sum((tuple(j * shift + i for i in x.word())
                    for j, x in enumerate(f)), ())\
            + tuple(len(f) * shift + i for i in h.word())

    @classmethod
    def order(cls):
        return cls.base.order() ** cls.degree() * cls.top.order()

    @classmethod
    def elements(cls):
        bases = itertools.product(*cls.degree() * [list(cls.base.elements())])
        return itertools.starmap(
            cls, itertools.product(bases, cls.top.elements()))

    @classmethod
    def sample(cls, rng):
        return cls(tuple(cls.base.sample(rng) for _ in range(cls.degree())),
                   cls.top.sample(rng))

    @classmethod
    def action(cls, base: Action = None, power: bool = False) -> Action:
        """
        An action of the wreath product given an action of ``base`` on
        ``range(n)``, the action of ``base`` on a set of its own by default.

        It is imprimitive by default, i.e. on the ``k`` blocks of ``n``
        points, numbered ``j * n + x``: the element ``(f, h)`` acts on the
        ``j``-th block by ``f[j]`` then moves it to the block ``h`` sends
        ``j`` to. The power action, on the ``k``-tuples of points numbered
        in base ``n``, acts on the ``j``-th coordinate by ``f[j]`` then
        moves it to the coordinate ``h`` sends ``j`` to.

        Parameters:
            base : An action of ``base``.
            power : Whether to take the power action.

        Example
        -------
        >>> G = Wreath[Symmetric[2], Cyclic[2]]
        >>> G.action().size, G.action(power=True).size
        (4, 4)
        >>> pairs = G.action().tensor(G.action())
        >>> len(G.action().orbits()), len(pairs.orbits())
        (1, 3)
        """
        base = cls.base.action() if base is None else base
        if base.group is not cls.base:
            raise TypeError(f"Expected an action of {cls.base}.")
        n, k = base.size, cls.degree()

        def on_blocks(element):
            (f, _), p = element.inside, element.permutation()
            images = [base(x) for x in f]
            return Permutation(
                (p[j] * n + images[j][x] for j in range(k) for x in range(n)),
                n * k)

        def on_tuples(element):
            (f, _), p = element.inside, element.permutation()
            images, result = [base(x) for x in f], []
            for point in itertools.product(range(n), repeat=k):
                target = k * [0]
                for j, x in enumerate(point):
                    target[p[j]] = images[j][x]
                result.append(sum(x * n ** (k - 1 - j)
                                  for j, x in enumerate(target)))
            return Permutation(result, n ** k)

        image = on_tuples if power else on_blocks
        return Action(cls, n ** k if power else n * k,
                      tuple(map(image, cls.generators())))


@dataclass(unsafe_hash=True)
class Action:
    """
    An action of a group on ``range(size)``, i.e. a functor from the group
    to :class:`discopy.python.finset.Permutation`, given by the images of
    its generators: an element acts by the product of the images along its
    :meth:`Group.word`. It is a functor exactly when the images satisfy the
    relations between the generators.

    Parameters:
        group : The group acting.
        size : The number of points acted on.
        images : The permutation each generator acts by.

    Example
    -------
    >>> action = Cyclic[4].action()
    >>> action
    Action(group.Cyclic[4], 4, ((1, 2, 3, 0),))
    >>> action(Cyclic[4](3))
    (3, 0, 1, 2)
    >>> action.orbits(), action.tensor(action).orbits()[0]
    (((0, 1, 2, 3),), (0, 5, 10, 15))
    """
    group: type[Group]
    size: int
    images: tuple[Permutation, ...]

    def __post_init__(self):
        self.images = tuple(Permutation(p, self.size) for p in self.images)
        if len(self.images) != len(self.group.generators()):
            raise ValueError(
                f"Expected an image for each generator of {self.group}.")

    def __repr__(self):
        return f"Action({factory_name(self.group)}, {self.size}, "\
            f"{self.images!r})"

    def __call__(self, element: Group) -> Permutation:
        assert_ismember(element, self.group)
        result = Permutation.id(self.size)
        for i in element.word():
            result = result.then(self.images[i])
        return result

    def tensor(self, *others: Action) -> Action:
        """
        The action on the tuples of points, acting on each coordinate, i.e.
        the tensor of permutation representations, numbered as in
        :func:`kronecker`.

        Parameters:
            others : Other actions of the same group.
        """
        result = self
        for other in others:
            if other.group is not self.group:
                raise TypeError(f"Expected an action of {self.group}.")
            result = Action(self.group, result.size * other.size, tuple(
                map(kronecker, result.images, other.images)))
        return result

    def orbits(self) -> tuple[tuple[int, ...], ...]:
        """
        The orbits of the points, each in increasing order and the orbits
        in the order of their least point, computed from the images of the
        generators by :meth:`Permutation.coequalizer`.
        """
        quotient = Permutation.id(self.size).coequalizer(*self.images)
        orbits = {}
        for point in range(self.size):
            orbits.setdefault(quotient[point], []).append(point)
        return tuple(map(tuple, orbits.values()))
