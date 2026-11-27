# -*- coding: utf-8 -*-

"""
The change of base of a category along a monad.

Given a category ``C`` and a :class:`~discopy.kleisli.monad.Monad` ``M``,
the category :math:`M_* C` has the objects of ``C`` and an arrow from ``A``
to ``B`` is an element of :math:`M(C(A, B))`, e.g. a list of arrows for the
:attr:`~discopy.kleisli.monad.List` monad. The identity is the unit on the
identity, :math:`\\eta(\\mathrm{id}_A)`, and the composite of ``m`` and
``n`` pairs their arrows then composes each pair

.. math::
    M(C(A, B)) \\times M(C(B, C))
    \\xrightarrow{\\varphi} M(C(A, B) \\times C(B, C))
    \\xrightarrow{M(\\mathbin{;})} M(C(A, C))

where :math:`\\varphi` takes the arrows of ``m`` first, see
:meth:`Arrow.pairs`. This is a category for every monad, and a monoidal one
when ``C`` is monoidal and ``M`` is commutative, see :meth:`Arrow.tensor`.
For the list monad the composite of two lists is the list of composites,
which is how :class:`~discopy.cat.Sum` composes: a sum is a list of arrows.

A functor ``F: C -> D`` extends to :math:`M_* C \\to D` when each hom of
``D`` is an algebra of ``M`` and composition in ``D`` is a homomorphism in
each variable: an arrow ``m`` goes to the structure of
:math:`D(F(A), F(B))` applied to :math:`M(F)(m)`, see :class:`Functor`.
``D`` gives this algebra as a classmethod, ``D.algebra(M, dom, cod)``, e.g.
:meth:`Arrow.algebra <discopy.cat.Arrow.algebra>`, the formal sum, and
:meth:`Matrix.algebra <discopy.matrix.Matrix.algebra>`, the sum of matrices,
both for the list monad.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Arrow
    Functor

Example
-------
A list of arrows composes as a sum does:

>>> from discopy import cat, messages
>>> from discopy.kleisli.monad import List
>>> x, y, z = map(cat.Ob, "xyz")
>>> f, g, h = cat.Box('f', x, y), cat.Box('g', x, y), cat.Box('h', y, z)
>>> m = Arrow[cat.Arrow, List]([f, g], x, y)
>>> n = Arrow[cat.Arrow, List]([h], y, z)
>>> assert (m >> n).inside == [f >> h, g >> h]
>>> assert (Arrow[cat.Arrow, List].id(x) >> m) == m == (m >> n.id(y))

A functor into arrows extends along their sum, and the extension is a
functor: it sends a list to a :class:`~discopy.cat.Sum` and preserves
composition.

>>> F = Functor(ob_map=lambda x: x, ar_map=lambda f: f,
...             dom=Arrow[cat.Arrow, List], cod=cat.Arrow)
>>> assert F(m) == f + g and F(n) == h
>>> assert F(m >> n) == F(m) >> F(n) == (f + g) >> h
"""
from __future__ import annotations

from collections.abc import Callable

from discopy import cat, messages
from discopy.abc import Category
from discopy.testable import TestableGeneric
from discopy.kleisli.channel import Channel
from discopy.kleisli.monad import Monad
from discopy.python.function import Function
from discopy.utils import (
    assert_iscomposable, assert_ismember, classproperty, factory,
    factory_name, unbiased)


@factory
class Arrow[category, monad](Category, TestableGeneric):
    """
    An arrow of the change of base of a category ``C`` along a monad ``M``,
    i.e. an element of ``M(C(A, B))``.

    Parameters:
        inside : The element of ``M(C)``, e.g. a list of arrows of ``C``.
        dom : The domain ``A``, an object of ``C``.
        cod : The codomain ``B``, an object of ``C``.

    Note
    ----
    That the arrows inside go from ``dom`` to ``cod`` is not checked. Two
    arrows are equal when their monadic values are, as observed by the
    monad, see :class:`~discopy.kleisli.monad.Monad`.

    Example
    -------
    A fair coin between two functions:

    >>> from discopy.python import Function
    >>> from discopy.kleisli.monad import Subdistribution
    >>> Coin = Arrow[Function, Subdistribution]
    >>> succ = Function(lambda n: n + 1, int, int)
    >>> double = Function(lambda n: 2 * n, int, int)
    >>> coin = Coin(frozenset({(succ, .5), (double, .5)}), int, int)
    >>> sorted(p for _, p in (coin >> coin).inside)
    [0.25, 0.25, 0.25, 0.25]
    """
    @classmethod
    def checked_params(cls) -> tuple:
        """ None, the arrows of a change of base having no strategy yet. """
        return ()

    # pylint: disable=not-callable  # the monad, set by subscripting
    ob = classproperty(lambda cls: cls.category.ob)
    monad: Monad

    def __init__(self, inside, dom, cod):
        self.inside = inside
        self.dom, self.cod = map(self.cast_ob, (dom, cod))

    @classmethod
    def cast_ob(cls, x):
        """
        Turn a boundary argument into an object of the category, as the
        category casts it: its ``cast_ob`` for a free category, the
        ``cast`` of its objects otherwise, e.g. a Python type into a list of
        types for :class:`~discopy.python.function.Function`.

        >>> from discopy.python import Function
        >>> from discopy.kleisli.monad import List
        >>> Arrow[Function, List].cast_ob(int)
        monoidal.List[type](int)
        """
        if hasattr(cls.category, "cast_ob"):
            return cls.category.cast_ob(x)
        return cls.ob.cast(x)

    @classmethod
    def id(cls, dom) -> Arrow:
        """
        The identity on an object ``dom``, i.e. the unit of the monad on the
        identity of the category.

        Parameters:
            dom : The object on which to take the identity.
        """
        unit = cls.monad.unit(cls.category)
        return cls(
            unit(cls.category.id(dom)),  # ty: ignore[call-non-callable]
            dom, dom)

    def pairs(self, other: Arrow):
        """
        The pairs of an arrow of ``self`` and one of ``other``, i.e. the
        map :math:`\\varphi : M(X) \\times M(Y) \\to M(X \\times Y)` of the
        monad, taking the arrows of ``self`` first.

        Parameters:
            other : The other arrow.

        Example
        -------
        >>> from discopy import cat
        >>> from discopy.kleisli.monad import List
        >>> x = cat.Ob('x')
        >>> f, g, h = (cat.Box(name, x, x) for name in "fgh")
        >>> Arrow[cat.Arrow, List]([f, g], x, x).pairs(
        ...     Arrow[cat.Arrow, List]([h], x, x))
        [(cat.Box('f', cat.Ob('x'), cat.Ob('x')), \
cat.Box('h', cat.Ob('x'), cat.Ob('x'))), \
(cat.Box('g', cat.Ob('x'), cat.Ob('x')), \
cat.Box('h', cat.Ob('x'), cat.Ob('x')))]
        """
        monad, C = type(self).monad, type(self).category
        pair = tuple[C, C]

        def with_other(f):
            return monad(Function(lambda g: (f, g), C, pair))(other.inside)
        mult = monad.mult(pair)  # ty: ignore[invalid-argument-type]
        return (monad(Function(with_other, C, monad(pair))) >> mult)(
            self.inside)

    def combine(self, other: Arrow, operation: Callable, dom, cod) -> Arrow:
        """
        Apply a binary operation of the category to each pair of arrows, see
        :meth:`pairs`.

        Parameters:
            other : The other arrow.
            operation : The binary operation on arrows, e.g. composition.
            dom : The domain of the result.
            cod : The codomain of the result.
        """
        assert_ismember(other, type(self))
        monad, C = type(self).monad, type(self).category
        pair = tuple[C, C]
        inside = monad(Function(lambda fg: operation(*fg), pair, C))(
            self.pairs(other))
        return type(self)(inside, dom, cod)

    @unbiased  # pylint: disable-next=arguments-differ  # made variadic
    def then(self, other: Arrow) -> Arrow:
        """
        The composition of two arrows, i.e. the monadic value of the
        composites of their pairs, called with :code:`>>`.

        Parameters:
            other : The other arrow to compose in sequence.
        """
        assert_iscomposable(self, other)
        return self.combine(other, lambda f, g: f >> g, self.dom, other.cod)

    def tensor(self, other: Arrow) -> Arrow:
        """
        The tensor of two arrows of a monoidal category, i.e. the monadic
        value of the tensors of their pairs, called with :code:`@`. It is
        a monoidal product when the monad is commutative, e.g.
        :attr:`~discopy.kleisli.monad.Subdistribution`: otherwise the
        interchange law can fail, since its two sides pair the arrows in
        different orders, e.g. for :attr:`~discopy.kleisli.monad.List`.

        Parameters:
            other : The other arrow to tensor in parallel.

        Example
        -------
        >>> from discopy.monoidal import Ty, Box, Diagram
        >>> from discopy.kleisli.monad import List
        >>> x, y = Ty('x'), Ty('y')
        >>> f, g, h = Box('f', x, y), Box('g', x, y), Box('h', y, x)
        >>> m = Arrow[Diagram, List]([f, g], x, y)
        >>> assert (m @ Arrow[Diagram, List]([h], y, x)).inside\\
        ...     == [f @ h, g @ h]
        """
        return self.combine(
            other, lambda f, g: f @ g, self.dom @ other.dom,
            self.cod @ other.cod)

    __matmul__ = tensor

    def to_channel(self) -> Channel:
        """
        The channel ``X -> M(Y)`` of an arrow of the change of base of
        :class:`~discopy.python.function.Function`, from ``X`` to ``Y``,
        i.e. each function evaluated at the input, for one type ``X`` and
        one type ``Y``.

        This is a functor into the Kleisli category, not a faithful one: a
        fair coin between the identity and ``not`` and one between the
        constants ``True`` and ``False`` give the same channel, though the
        function drawn by the first never gives the same output to both
        inputs and the one drawn by the second always does.

        Raises:
            TypeError : When the category is not that of Python functions.
            ValueError : When the domain or codomain is not one type.

        Example
        -------
        >>> from discopy.python import Function
        >>> from discopy.kleisli.monad import Subdistribution
        >>> Coin = Arrow[Function, Subdistribution]
        >>> id_bool, not_ = Function.id(bool), Function(
        ...     lambda x: not x, bool, bool)
        >>> true, false = (
        ...     Function(lambda _, b=b: b, bool, bool) for b in (True, False))
        >>> left = Coin(frozenset({(id_bool, .5), (not_, .5)}), bool, bool)
        >>> right = Coin(frozenset({(true, .5), (false, .5)}), bool, bool)
        >>> assert left != right and all(
        ...     left.to_channel()(b) == right.to_channel()(b)
        ...     == frozenset({(True, .5), (False, .5)}) for b in (True, False))
        """
        monad, C = type(self).monad, type(self).category
        if not issubclass(C, Function):
            raise TypeError(messages.TYPE_ERROR.format(
                factory_name(Function), factory_name(C)))
        dom, cod = self.dom, self.cod
        if len(dom) != 1 or len(cod) != 1:
            raise ValueError(messages.NOT_ONE_TYPE.format(dom, cod))
        X, Y = dom.inside[0], cod.inside[0]

        def inside(x):
            return monad(Function(lambda f: f(x), C, Y))(self.inside)
        return Channel[monad](  # ty: ignore[invalid-type-form]
            inside, X, Y)

    def __eq__(self, other):
        observe = type(self).monad.observe
        return type(self) is type(other)\
            and (self.dom, self.cod) == (other.dom, other.cod)\
            and observe(self.inside) == observe(other.inside)

    def __repr__(self):
        return factory_name(type(self))\
            + f"({self.inside!r}, dom={self.dom!r}, cod={self.cod!r})"


class Functor(cat.Functor):
    """
    A functor ``F: C -> D`` extended to the change of base of ``C`` along a
    monad ``M``, when the homs of ``D`` are algebras of ``M``: an arrow
    ``m`` from ``A`` to ``B`` goes to the structure of
    ``D.algebra(M, F(A), F(B))`` applied to ``M(F)(m)``, where ``F`` is the
    ``Functor`` of ``C``, e.g. :class:`~discopy.monoidal.Functor`
    for :class:`~discopy.monoidal.Diagram`.

    Parameters:
        ob_map : Mapping from the objects of ``C`` to those of ``D``.
        ar_map : Mapping from the boxes of ``C`` to the arrows of ``D``.
        dom : The change of base, e.g. ``Arrow[cat.Arrow, List]``.
        cod : The category ``D``, with a classmethod ``algebra``, see
            :meth:`~discopy.cat.Arrow.algebra`.

    Example
    -------
    A list of arrows goes to the sum of the images of its arrows, here
    matrices:

    >>> from discopy import cat
    >>> from discopy.matrix import Matrix
    >>> from discopy.kleisli.monad import List
    >>> x = cat.Ob('x')
    >>> f, g = cat.Box('f', x, x), cat.Box('g', x, x)
    >>> F = Functor(
    ...     ob_map={x: 2}, ar_map={f: [0, 1, 1, 0], g: [1, 1, 0, 1]},
    ...     dom=Arrow[cat.Arrow, List], cod=Matrix[int])
    >>> F(Arrow[cat.Arrow, List]([f, g, f >> g], x, x))
    Matrix[int]([1, 3, 2, 2], dom=Nat(2), cod=Nat(2))
    """
    @property
    def base(self) -> cat.Functor:
        """
        The functor ``F: C -> D`` that is extended, i.e. the
        ``Functor`` of ``C`` with the same maps, which evaluates the
        objects and the arrows inside.
        """
        C = self.dom.category
        return C.Functor(self.ob_map, self.ar_map, dom=C, cod=self.cod)

    def __call__(self, other):
        base = self.base
        if not isinstance(other, Arrow):
            return base(other)
        monad, C = type(other).monad, type(other).category
        algebra = self.cod.algebra(monad, base(other.dom), base(other.cod))
        return algebra(monad(Function(base, C, self.cod))(other.inside))
