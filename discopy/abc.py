"""
The abstract base classes for categories.

These mirror the concrete hierarchy of :mod:`discopy` modules: each class adds
the characteristic generator of its categorical structure as an
:func:`abc.abstractmethod`, e.g. :class:`BraidedCategory` is a
:class:`MonoidalCategory` with an abstract :meth:`BraidedCategory.braid`.

.. raw:: html
    :file: api/architecture.html

Software dependencies between modules go top-to-bottom, left-to-right and
forgetful functors between categories go the other way.

Each class also declares its :func:`discopy.axioms.axiom` equations, which
every free category inherits along with the structure they axiomatise:
:class:`Category` states the unitality and associativity of composition,
the typing of its identities and composites, and a
:class:`DaggerCategory` the involution and contravariance of its dagger;
a :class:`ColouredMonoid` inherits them as
the unitality and associativity of its product, its composition. The
structural methods carry their own :func:`discopy.search.rule`, from
which
:meth:`discopy.monoidal.Diagram.strategy` searches for the diagrams the
laws quantify over: a level of the hierarchy declares its structure here
and inherits the search as is.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Category
    DaggerCategory
    ColouredMonoid
    Monoid
    Nat
    MonoidalCategory
    PRO
    TracedCategory
    ResiduatedMonoid
    BiclosedCategory
    Pregroup
    RigidCategory
    PivotalCategory
    BraidedCategory
    PROB
    SymmetricCategory
    PROP
    MarkovCategory
    ClosedCategory
    DelayedMonoid
    FeedbackCategory
    BalancedCategory
    RibbonCategory
    CompactCategory
    HypergraphCategory
    NamedGeneric
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from types import NoneType
from typing import (
    Annotated, ClassVar, Literal, Self, TYPE_CHECKING, overload)

from discopy.axioms import (  # noqa: F401  pylint: disable=unused-import
    Atom, Axiom, axiom, Count, D, declarations, Equation, L, Over, R,
    Repeat, Rule, rule, Serialisable, Tensor, Testable, Under, Unit)
from discopy.utils import (  # noqa: F401  pylint: disable=unused-import
    NamedGeneric, classproperty, factory_name)

type Obj[T, X = None] = Annotated[T, X]
""" The premise ``x: Obj[T, p]`` of a pattern ``p`` beside its coarse
type, ``Obj[C0]`` in a bound the sort of an object variable, expanded
by :func:`discopy.pattern.expand` and read as ``Annotated`` by a
typechecker. """

type Hom[T, A, B] = Annotated[T, A, B]
""" The premise or conclusion ``Hom[C1, dom, cod]`` of a morphism
between two sides, expanded by :func:`discopy.pattern.expand` and
read as ``Annotated`` by a typechecker. """


class Category[C0, C1: Category](Testable, ABC):
    """
    A category is a class with two class variables ``ob, ar``, two attributes
    ``dom, cod`` and two methods ``id, then``.

    This base class also implements syntactic sugar :code:`>>` and :code:`<<`
    for forward and backward composition with the method :code:`then`.

    Example
    -------
    >>> class List(list, Category):
    ...     ob, dom, cod = type(None), None, None
    ...     def then(self, other):
    ...         return self + other
    >>> assert List([1, 2]) >> List([3]) == List([1, 2, 3])
    >>> assert List([3]) << List([1, 2]) == List([1, 2, 3])
    """
    ob: ClassVar[type]
    factory: ClassVar[type]
    dom: C0
    cod: C0

    #: Backward-compatible alias for :attr:`factory`, since types are
    #: themselves the objects of diagrams.
    ar = classproperty(lambda cls: getattr(cls, "factory", cls))

    #: The equation up to which the axioms of the category compare, with
    #: strict equality by default. A category that quotients its equations
    #: binds its own, e.g. by hypergraph isomorphism from symmetric
    #: categories on, and :meth:`discopy.axioms.Axiom.modulo` weakens it
    #: further.
    Equation: ClassVar[type[Equation]] = Equation

    @classproperty
    def rules(cls: type) -> dict[str, Rule]:
        """
        The inference rules inherited by ``cls``, by name: the rule each
        structural method carries, see :func:`discopy.search.rule`, bound
        to ``cls`` and owned by the class declaring it, the latest in the
        method resolution order winning like ordinary attribute lookup. A
        method implementing a rule is decorated
        :func:`discopy.search.rule` itself, restating its sequent, and
        the search calls it by name; one declared
        :meth:`discopy.search.Rule.inapplicable` or
        :meth:`discopy.search.Rule.admissible` is dropped.
        """
        return declarations(cls, Rule)

    @classproperty
    def generators(cls: type) -> dict[str, Rule]:
        """
        The logical constants inherited by ``cls``, by name: the rules
        with no hom premise, which the search builds in one step, see
        :meth:`discopy.search.Rule.recursive`. A class adjusts the set by
        assigning a dictionary of rules instead,
        :meth:`discopy.search.Rule.constant` giving the rule of one given
        box, so that its strategy draws from a fixed vocabulary, e.g. the
        words of a pregroup grammar or the gates of a circuit.

        >>> from hypothesis import find
        >>> from discopy.axioms import Rule
        >>> from discopy.grammar import pregroup
        >>> n, s = pregroup.Ty('n'), pregroup.Ty('s')
        >>> Alice, sleeps = pregroup.Word('Alice', n), pregroup.Word(
        ...     'sleeps', n.r @ s)
        >>> class Sentence(pregroup.Diagram):
        ...     generators = {
        ...         "cups": pregroup.Diagram.generators["cups"],
        ...         **{w.name: Rule.constant(w) for w in (Alice, sleeps)}}
        >>> print(find(Sentence.strategy(), bool).foliation())
        Alice @ sleeps >> Cup(n, n.r) @ s
        """
        return {name: rule for name, rule in cls.rules.items()
                if not rule.recursive}

    @classmethod
    @rule
    @abstractmethod
    def id[A: Obj[C0]](cls, dom: Obj[C0, A]) -> Hom[C1, A, A]:
        """
        Identity morphism on an object :code:`dom: C0`, to be instantiated:
        as a rule, ``x ⊢ x`` with no box is the identity.

        Parameters:
            dom (C0) : The domain of an identity is also its codomain.
        """

    @rule
    @abstractmethod
    def then[A: Obj[C0], B: Obj[C0], C: Obj[C0]](
            self: Hom[C1, A, B], other: Hom[C1, B, C]) -> Hom[C1, A, C]:
        """
        Sequential composition, to be instantiated: the rule composes two
        morphisms, an implementation may take ``n >= 1`` of them.

        Parameters:
            other : The other morphism to compose sequentially.
        """

    def is_composable(self, other: C1) -> bool:
        """
        Whether two morphisms are composable, i.e. the codomain of the first is
        the domain of the second.

        Parameters:
            other : The other morphism.
        """
        return self.cod == other.dom

    def is_parallel(self, other: Category) -> bool:
        """
        Whether two morphisms are parallel, i.e. they have the same
        domain and codomain.

        Parameters:
            other : The other morphism.
        """
        return (self.dom, self.cod) == (other.dom, other.cod)

    @axiom
    def unitality[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ Left and right unitality of composition. """
        return cls.Equation(cls.id(f.dom).then(f), f, f.then(cls.id(f.cod)))

    @axiom
    def associativity[A: Obj[C0], B: Obj[C0], C: Obj[C0], D: Obj[C0]](
            cls, f: Hom[C1, A, B],
            g: Hom[C1, B, C],
            h: Hom[C1, C, D]) -> Equation[Hom[C1, A, D]]:
        """ Associativity of composition. """
        return cls.Equation(f.then(g).then(h), f.then(g.then(h)))




    __rshift__ = __llshift__ = lambda self, other: self.then(other)
    __lshift__ = __lrshift__ = lambda self, other: other.then(self)


class DaggerCategory[C0, C1: DaggerCategory](Category[C0, C1]):
    """
    A `dagger category <https://ncatlab.org/nlab/show/dagger+category>`_ is a
    :class:`Category` with a method :code:`dagger` for the identity-on-objects
    contravariant involution, i.e. such that
    ``(f >> g).dagger() == g.dagger() >> f.dagger()``
    and ``f.dagger().dagger() == f``.

    Its two laws are stated here rather than on :class:`Category`, so that a
    category with no dagger does not have to declare them inapplicable.
    """
    @abstractmethod
    def dagger[A: Obj[C0], B: Obj[C0]](
            self: Hom[C1, A, B]) -> Hom[C1, B, A]:
        """ The dagger of a morphism, to be instantiated. """

    @axiom
    def dagger_involution[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ The dagger is involutive. """
        return cls.Equation(f.dagger().dagger(), f)

    @axiom
    def dagger_contravariance[A: Obj[C0], B: Obj[C0], C: Obj[C0]](
            cls, f: Hom[C1, A, B],
            g: Hom[C1, B, C]) -> Equation[Hom[C1, C, A]]:
        """ The dagger reverses composition. """
        return cls.Equation(f.then(g).dagger(), g.dagger().then(f.dagger()))


class ColouredMonoid[C0, C1: ColouredMonoid](Category[C0, C1]):
    """
    A coloured monoid is a category whose sequential composition ``then`` is
    given by a monoidal ``tensor``, with the objects ``C0`` (its colours) as
    the boundaries of its morphisms.

    An ordinary :obj:`Monoid` is the special case with a single, trivial
    colour, i.e. :class:`type(None)`. We do not enforce this so
    that e.g. :class:`monoidal.Ty` can take colours as objects.
    """
    if TYPE_CHECKING:
        def __len__(self) -> int:
            """ The number of generating objects inside, assumed free. """

        def __getitem__(self, key) -> Self:
            """ The slices of a free monoid, assumed to stay inside it. """

    @classmethod
    @rule
    def id[A: Obj[C0]](
            cls, dom: Obj[C0 | None, A] = None) -> Hom[C1, A, A]:
        """The monoidal unit, i.e. the empty tensor ``cls()``."""
        return cls()  # ty: ignore[invalid-return-type]

    @classmethod
    def unit[A: Obj[C0]](
            cls, colour: Obj[C0 | None, A] = None) -> Obj[C0 | C1, Unit[C0]]:
        """
        The unit at a colour, i.e. the identity on it.

        It need not be an element of the monoid, which is why it may land in
        ``C0``: the layers of :class:`monoidal.Layer` are closed under
        ``tensor`` but the empty one is a type rather than a layer.
        """
        return cls.id(colour)

    @abstractmethod
    def tensor(self, *objects: Self) -> Self:
        """ The n-ary product of a monoid for ``n > 0``. """

    def then(self, *others: Self) -> Self:
        """Sequential composition, given by the monoid product."""
        return self.tensor(*others)

    @classmethod
    def cast(cls, atoms) -> Self:
        """
        The element of a tuple of atoms, or of a single atom; an element of
        the monoid is unchanged.

        Parameters:
            atoms : An element, a tuple of atoms or a single atom.

        >>> assert Nat.cast(2) == Nat(2) == Nat.cast(Nat(2))
        """
        if isinstance(atoms, cls):
            return atoms
        if isinstance(atoms, tuple):
            return cls(*atoms)
        return cls(atoms)  # ty: ignore[too-many-positional-arguments]

    @classmethod
    def whisker[A: Obj[C0], B: Obj[C0]](
            cls, other: Obj[C0, A] | Hom[C1, A, B]) -> Hom[C1, A, B]:
        """
        Do nothing if ``other`` is already a morphism else apply :meth:`id`.

        Parameters:
            other : The object or morphism to be tensored on the left or right.
        """
        return (
            other if isinstance(other, cls)  # ty: ignore[invalid-return-type]
            else cls.id(other))  # ty: ignore[invalid-argument-type]

    def __matmul__(self, other):
        return self.tensor(other)

    def __rmatmul__(self, other):
        return self.whisker(other).tensor(self)


class Monoid[C1: Monoid](ColouredMonoid[NoneType, C1]):
    """ A monoid is a coloured monoid with a single, trivial colour. """


@dataclass
class Nat(Monoid["Nat"]):
    """
    ``Nat`` is the free monoid on one generator, i.e. the natural numbers
    with addition as tensor. It is also a sequence over its unary encoding:
    :meth:`__len__` gives back the natural number itself and slicing reads
    it off as a sequence of ``1``'s, e.g. ``Nat(3)[:1] == Nat(1)``.

    Parameters:
        n : The natural number.
    """
    n: int = 0

    def tensor(self, *others: Nat) -> Nat:
        if any(not isinstance(other, Nat) for other in others):
            return NotImplemented  # This allows whiskering on the left.
        return type(self)(self.n + sum(other.n for other in others))

    def __len__(self) -> int:
        return self.n

    def __index__(self) -> int:
        return self.n

    def __str__(self) -> str:
        return str(self.n)

    def __getitem__(self, key: int | slice) -> Nat:
        """
        Slicing a natural number reads it off as a sequence of ``1``'s.

        Parameters:
            key : An integer or a slice.
        """
        if isinstance(key, slice):
            return type(self)(len(range(self.n)[key]))
        if key >= self.n or key < -self.n:
            raise IndexError
        return type(self)(1)


class MonoidalCategory[C0: ColouredMonoid, C1: MonoidalCategory](
        Category[C0, C1]):
    """
    A monoidal category is a :class:`Category` with a method :code:`tensor`
    for both its objects and its morphisms.

    This base class also implements syntactic sugar :code:`@` for whiskering.
    """
    @rule
    @abstractmethod
    def tensor[A: Obj[C0], B: Obj[C0], C: Obj[C0], D: Obj[C0]](
            self: Hom[C1, A, B], other: Hom[C1, C, D]
    ) -> Hom[C1, Tensor[A, C], Tensor[B, D]]:
        """
        Parallel composition, to be instantiated: the rule tensors two
        morphisms, an implementation may take ``n >= 0`` of them.

        Parameters:
            other : The other morphism to compose in parallel.
        """

    @classmethod
    def whisker[A: Obj[C0], B: Obj[C0]](
            cls, other: Obj[C0, A] | Hom[C1, A, B]) -> Hom[C1, A, B]:
        """
        Do nothing if ``other`` is already a morphism else apply :meth:`id`.

        Parameters:
            other : The object or morphism to be tensored on the left or right.
        """
        return (other if isinstance(  # ty: ignore[invalid-return-type]
            other, MonoidalCategory) else cls.id(other))

    def __matmul__(self, other):
        return self.tensor(self.whisker(other))

    def __rmatmul__(self, other):
        return self.whisker(other).tensor(self)

    @rule
    def cut[A: Obj[C0], B: Obj[C0], C: Obj[C0], X: Obj[C0], Y: Obj[C0]](
            self: Hom[C1, B, A], other: Hom[C1, Tensor[X, A, Y], C],
            left: Obj[C0, X], right: Obj[C0, Y]
    ) -> Hom[C1, Tensor[X, B, Y], C]:
        """
        Composition in context, the `cut rule
        <https://en.wikipedia.org/wiki/Cut_rule>`_ of the Lambek calculus:
        plug a morphism into the middle of the domain of ``other``, i.e.
        one layer of a diagram. The rule derives from
        :meth:`Category.then` and :meth:`tensor`, and replaces them as
        the recursive rule of the search from monoidal categories on:
        every diagram is a sequence of layers and every layer is one
        cut, while the conclusion anchors both premises on the goal
        where the fresh middle of :meth:`Category.then` anchors neither.

        Parameters:
            other : The morphism consuming the codomain of ``self``.
            left : The context on the left of ``self``.
            right : The context on the right of ``self``.
        """
        return left @ self @ right >> other

    @axiom
    def bifunctoriality[A: Obj[C0], B: Obj[C0], C: Obj[C0], D: Obj[C0],
                        U: Obj[C0], V: Obj[C0]](
            cls, f: Hom[C1, A, B],
            g: Hom[C1, C, D],
            h: Hom[C1, B, U],
            k: Hom[C1, D, V]
    ) -> Equation[Hom[C1, Tensor[A, C], Tensor[U, V]]]:
        """ Bifunctoriality of the tensor. """
        return cls.Equation(
            f @ g >> h @ k, (f >> h) @ (g >> k))

    @axiom
    def tensor_unitality[X: Obj[C0], Y: Obj[C0]](
            cls, x: Obj[C0, X], y: Obj[C0, Y]
    ) -> Equation[Hom[C1, Tensor[X, Y], Tensor[X, Y]]]:
        """ Preservation of identities by tensor. """
        return cls.Equation(
            cls.id(x) @ cls.id(y), cls.id(x @ y))



    @axiom
    def dagger_monoidality[A: Obj[C0], B: Obj[C0], C: Obj[C0], D: Obj[C0]](
            cls, f: Hom[C1, A, B], g: Hom[C1, C, D]
    ) -> Equation[Hom[C1, Tensor[B, D], Tensor[A, C]]]:
        """ The dagger distributes over the tensor. """
        return cls.Equation((f @ g).dagger(), f.dagger() @ g.dagger())


class PRO[C1: PRO](MonoidalCategory[Nat, C1]):
    """
    A PRO is a :class:`MonoidalCategory` whose objects are the natural
    numbers :class:`Nat`, i.e. the free monoidal category on one generator.
    """


class TracedCategory[C0: ColouredMonoid, C1: TracedCategory](
        MonoidalCategory[C0, C1]):
    """
    A traced category is a :class:`MonoidalCategory` with methods
    :code:`trace_left` and :code:`trace_right` for the partial trace of a
    morphism over some objects on either side.
    """
    @rule
    def trace_left[A: Obj[C0], B: Obj[C0], M: Atom[C0]](
            self: Hom[C1, Tensor[M, A], Tensor[M, B]]) -> Hom[C1, A, B]:
        """ The trace of one wire on the left, :meth:`trace` takes ``n``. """
        return self.trace(1, left=True)

    @rule
    def trace_right[A: Obj[C0], B: Obj[C0], M: Atom[C0]](
            self: Hom[C1, Tensor[A, M], Tensor[B, M]]) -> Hom[C1, A, B]:
        """ The trace of one wire on the right, :meth:`trace` takes ``n``. """
        return self.trace(1)

    @overload
    def trace[A: Obj[C0], B: Obj[C0], M: Atom[C0]](
            self: Hom[C1, Tensor[A, M], Tensor[B, M]], n: int = ...,
            left: Literal[False] = ...) -> Hom[C1, A, B]: ...

    @overload
    def trace[A: Obj[C0], B: Obj[C0], M: Atom[C0]](
            self: Hom[C1, Tensor[M, A], Tensor[M, B]], n: int = ...,
            left: Literal[True] = ...) -> Hom[C1, A, B]: ...

    @abstractmethod
    def trace(self, n=1, left=False):
        """
        The trace of ``n`` wires on either side, to be instantiated: the
        rules :meth:`trace_left` and :meth:`trace_right`, whose sequents
        the two overloads restate, are its one-wire instances. Tracing no
        object at all is the identity, i.e. the vanishing axiom
        ``f.trace(0) == f``, see `nLab
        <https://ncatlab.org/nlab/show/traced+monoidal+category>`_.

        Parameters:
            n : The number of objects to trace over.
            left : Whether to trace the wires on the left or right.
        """

    @axiom
    def trace_vanishing[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ Vanishing of a trace over the unit. """
        return cls.Equation(
            f.trace(0), f, f.trace(0, left=True))

    @axiom
    def trace_superposing_left[
            A: Obj[C0], B: Obj[C0], X: Obj[C0], M: Atom[C0]](
            cls, f: Hom[C1, Tensor[M, A], Tensor[M, B]], x: Obj[C0, X]
    ) -> Equation[Hom[C1, Tensor[A, X], Tensor[B, X]]]:
        """ Left-oriented superposing. """
        return cls.Equation(
            (f @ x).trace(left=True), f.trace(left=True) @ x)

    @axiom
    def trace_superposing_right[
            A: Obj[C0], B: Obj[C0], X: Obj[C0], M: Atom[C0]](
            cls, f: Hom[C1, Tensor[A, M], Tensor[B, M]], x: Obj[C0, X]
    ) -> Equation[Hom[C1, Tensor[X, A], Tensor[X, B]]]:
        """ Right-oriented superposing. """
        return cls.Equation(
            (x @ f).trace(), x @ f.trace())

    @axiom
    def trace_naturality_left[M: Atom[C0], X: Obj[C0], A: Obj[C0], B: Obj[C0]](
            cls, x: Obj[C0, Tensor[M, X]],
            f: Hom[C1, Tensor[M, X, B], Tensor[M, X, A]],
            g: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ Left-oriented trace naturality. """
        return cls.Equation(
            (x @ g).then(f).then(x @ g).trace(len(x), left=True),
            g.then(f.trace(len(x), left=True)).then(g))

    @axiom
    def trace_naturality_right[
            M: Atom[C0], X: Obj[C0], A: Obj[C0], B: Obj[C0]](
            cls, x: Obj[C0, Tensor[M, X]],
            f: Hom[C1, Tensor[B, M, X], Tensor[A, M, X]],
            g: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ Right-oriented trace naturality. """
        return cls.Equation(
            (g @ x).then(f).then(g @ x).trace(len(x)),
            g.then(f.trace(len(x))).then(g))

    @axiom
    def trace_dinaturality_left[M: Atom[C0], N: Atom[C0], S: Obj[C0],
                                T: Obj[C0], A: Obj[C0], B: Obj[C0]](
            cls,
            f: Hom[C1, Tensor[M, S, A], Tensor[N, T, B]],
            g: Hom[C1, Tensor[N, T], Tensor[M, S]]
    ) -> Equation[Hom[C1, A, B]]:
        """ Left-oriented trace dinaturality. """
        source, target = g.cod, g.dom
        base, cobase = f.dom[len(source):], f.cod[len(target):]
        return cls.Equation(
            f.then(g @ cobase).trace(len(source), left=True),
            (g @ base).then(f).trace(len(target), left=True))

    @axiom
    def trace_dinaturality_right[M: Atom[C0], N: Atom[C0], S: Obj[C0],
                                 T: Obj[C0], A: Obj[C0], B: Obj[C0]](
            cls,
            f: Hom[C1, Tensor[A, M, S], Tensor[B, N, T]],
            g: Hom[C1, Tensor[N, T], Tensor[M, S]]
    ) -> Equation[Hom[C1, A, B]]:
        """ Right-oriented trace dinaturality. """
        source, target = g.cod, g.dom
        base = f.dom[:-len(source)] if len(source) else f.dom
        cobase = f.cod[:-len(target)] if len(target) else f.cod
        return cls.Equation(
            f.then(cobase @ g).trace(len(source)),
            (base @ g).then(f).trace(len(target)))


class ResiduatedMonoid[C0, C1: ResiduatedMonoid](ColouredMonoid[C0, C1]):
    """
    A monoid is residuated when it comes with methods ``over`` and ``under``
    with syntactic sugar ``<<`` and ``>>``.

    We also assume the exponential objects can be recognised and taken apart,
    see :class:`biclosed.Exp`.
    """
    if TYPE_CHECKING:
        @property
        def is_exp(self) -> bool:
            """ Whether this is an exponential object. """

        @property
        def base(self) -> Self:
            """ The base of an exponential object. """

        @property
        def exponent(self) -> Self:
            """ The exponent of an exponential object. """

    @abstractmethod
    def over[X: Obj[C1], Y: Obj[C1]](
            self: Obj[C1, X], other: Obj[C1, Y]) -> Obj[C1, Over[X, Y]]:
        """ The right-to-left exponential object ``self`` to the ``other``. """

    @abstractmethod
    def under[X: Obj[C1], Y: Obj[C1]](
            self: Obj[C1, X], other: Obj[C1, Y]) -> Obj[C1, Under[Y, X]]:
        """ The left-to-right exponential object ``self`` to the ``other``. """

    def __lshift__[X: Obj[C1], Y: Obj[C1]](
            self: Obj[C1, X], other: Obj[C1, Y]) -> Obj[C1, Over[X, Y]]:
        return self.over(other)

    def __rshift__[X: Obj[C1], Y: Obj[C1]](
            self: Obj[C1, X], other: Obj[C1, Y]) -> Obj[C1, Under[X, Y]]:
        return other.under(self)


class BiclosedCategory[C0: ResiduatedMonoid, C1: BiclosedCategory](
        MonoidalCategory[C0, C1]):
    """
    A biclosed category is a :class:`MonoidalCategory` with methods for the
    evaluation and currying of morphisms on either side.

    We also assume the type for objects comes with methods for left and right
    exponentials :code`x << y` and :code`x >> y`.
    """
    @classmethod
    @rule
    @abstractmethod
    def ev_left[Y: Atom[C0], E: Atom[C0]](
            cls, base: Obj[C0, Y], exponent: Obj[C0, E]
    ) -> Hom[C1, Tensor[Over[Y, E], E], Y]:
        """
        The left evaluation of an exponential type, to be instantiated:
        as a rule, ``(y << e) @ e ⊢ y``.

        Parameters:
            base : The base of the exponential type.
            exponent : The exponent of the exponential type.
        """

    @classmethod
    @rule
    @abstractmethod
    def ev_right[Y: Atom[C0], E: Atom[C0]](
            cls, base: Obj[C0, Y], exponent: Obj[C0, E]
    ) -> Hom[C1, Tensor[E, Under[E, Y]], Y]:
        """
        The right evaluation of an exponential type, to be instantiated:
        as a rule, ``e @ (e >> y) ⊢ y``.

        Parameters:
            base : The base of the exponential type.
            exponent : The exponent of the exponential type.
        """

    @overload
    @classmethod
    def ev[Y: Atom[C0], E: Atom[C0]](
            cls, base: Obj[C0, Y], exponent: Obj[C0, E],
            left: Literal[True] = ...
    ) -> Hom[C1, Tensor[Over[Y, E], E], Y]: ...

    @overload
    @classmethod
    def ev[Y: Atom[C0], E: Atom[C0]](
            cls, base: Obj[C0, Y], exponent: Obj[C0, E],
            left: Literal[False] = ...
    ) -> Hom[C1, Tensor[E, Under[E, Y]], Y]: ...

    @classmethod
    def ev(cls, base, exponent, left=True):
        """
        The evaluation of an exponential type on either side,
        :meth:`ev_left` or :meth:`ev_right`, whose sequents the two
        overloads restate.

        Parameters:
            base : The base of the exponential type.
            exponent : The exponent of the exponential type.
            left : Whether to take the left or right evaluation.
        """
        return (cls.ev_left if left else cls.ev_right)(base, exponent)

    @rule
    def curry_left[X: Obj[C0], Y: Atom[C0], Z: Obj[C0]](
            self: Hom[C1, Tensor[X, Y], Z]) -> Hom[C1, X, Over[Z, Y]]:
        """ The currying of one object on the left, :meth:`curry` takes
        ``n``. """
        return self.curry(1, left=True)

    @rule
    def curry_right[Y: Atom[C0], X: Obj[C0], Z: Obj[C0]](
            self: Hom[C1, Tensor[Y, X], Z]) -> Hom[C1, X, Under[Y, Z]]:
        """ The currying of one object on the right, :meth:`curry` takes
        ``n``. """
        return self.curry(1, left=False)

    @overload
    def curry[X: Obj[C0], Y: Atom[C0], Z: Obj[C0]](
            self: Hom[C1, Tensor[X, Y], Z], n: int = ...,
            left: Literal[True] = ...) -> Hom[C1, X, Over[Z, Y]]: ...

    @overload
    def curry[Y: Atom[C0], X: Obj[C0], Z: Obj[C0]](
            self: Hom[C1, Tensor[Y, X], Z], n: int = ...,
            left: Literal[False] = ...) -> Hom[C1, X, Under[Y, Z]]: ...

    @abstractmethod
    def curry(self, n=1, left=True):
        """
        The currying of ``n`` objects on either side, to be instantiated:
        the rules :meth:`curry_left` and :meth:`curry_right`, whose
        sequents the two overloads restate, are its one-object instances.

        Parameters:
            n : The number of objects to curry.
            left : Whether to curry on the left or right.
        """

    @overload
    def base_and_exponent[X: Obj[C0], Y: Atom[C0], Z: Obj[C0]](
            self: Hom[C1, X, Over[Z, Y]], n: int,
            left: Literal[True]) -> tuple[Obj[C0, Z], Obj[C0, Y]]: ...

    @overload
    def base_and_exponent[X: Obj[C0], Y: Atom[C0], Z: Obj[C0]](
            self: Hom[C1, X, Under[Y, Z]], n: int,
            left: Literal[False]) -> tuple[Obj[C0, Z], Obj[C0, Y]]: ...

    def base_and_exponent(self, n, left):
        """
        The base and exponent that :meth:`uncurry` evaluates, read off the
        exponential object in the codomain.

        Parameters:
            n : The number of objects to uncurry.
            left : Whether to uncurry on the left or right.
        """
        # pylint: disable=unused-argument  # the exponential says the side
        if not self.cod.is_exp:
            raise ValueError
        base, exponent = self.cod.base, self.cod.exponent
        if n < len(exponent):
            raise ValueError
        return base, exponent

    @overload
    def uncurry[X: Obj[C0], Y: Atom[C0], Z: Obj[C0]](
            self: Hom[C1, X, Over[Z, Y]], n: int = ...,
            left: Literal[True] = ...) -> Hom[C1, Tensor[X, Y], Z]: ...

    @overload
    def uncurry[Y: Atom[C0], X: Obj[C0], Z: Obj[C0]](
            self: Hom[C1, X, Under[Y, Z]], n: int = ...,
            left: Literal[False] = ...) -> Hom[C1, Tensor[Y, X], Z]: ...

    def uncurry(self, n: int = 1, left: bool = True):
        """
        Uncurry a morphism by composing it with :meth:`ev`, assuming its
        codomain is an exponential object, i.e. undo :meth:`curry`, whose
        sequents the two overloads state upside down. If the exponent has
        less than ``n`` objects, we uncurry the remaining ones in turn.

        Parameters:
            n : The number of objects to uncurry.
            left : Whether to uncurry on the left or right.
        """
        if n < 0:
            raise ValueError
        if not n:
            return self
        base, exponent = self.base_and_exponent(
            n, left)  # ty: ignore[no-matching-overload]
        result = self @ exponent >> self.ev(base, exponent, True) if left\
            else exponent @ self >> self.ev(base, exponent, False)
        return result.uncurry(n - len(exponent), left)

    @overload
    @classmethod
    def uncurry_composition[A: Obj[C0], X: Atom[C0], E: Atom[C0]](
            cls, f: Hom[C1, Tensor[A, E], X], base: Obj[C0, X],
            exponent: Obj[C0, E], left: Literal[True]
    ) -> Hom[C1, Tensor[A, E], X]: ...

    @overload
    @classmethod
    def uncurry_composition[A: Obj[C0], X: Atom[C0], E: Atom[C0]](
            cls, f: Hom[C1, Tensor[E, A], X], base: Obj[C0, X],
            exponent: Obj[C0, E], left: Literal[False]
    ) -> Hom[C1, Tensor[E, A], X]: ...

    @classmethod
    def uncurry_composition(cls, f, base, exponent, left: bool):
        """
        Curry ``f`` then evaluate it back, i.e. whisker the currying with
        ``exponent`` and compose with the evaluation, the roundtrip that
        :meth:`currying_left` and :meth:`currying_right` state equal to
        ``f``, through the one-wire rules :meth:`curry_left`,
        :meth:`curry_right`, :meth:`ev_left` and :meth:`ev_right`.

        Parameters:
            f : The morphism to curry and evaluate back.
            base : The base of the exponential, i.e. the codomain of ``f``.
            exponent : The objects curried out of the domain of ``f``.
            left : Whether to curry on the left or right.
        """
        if left:
            curried, ev = f.curry_left(), cls.ev_left(base, exponent)
            return (curried @ exponent).then(ev)
        curried, ev = f.curry_right(), cls.ev_right(base, exponent)
        return (exponent @ curried).then(ev)

    @axiom
    def currying_left[A: Obj[C0], X: Atom[C0], E: Atom[C0]](
            cls, f: Hom[C1, Tensor[A, E], X],
            base: Obj[C0, X],
            exponent: Obj[C0, E]) -> Equation[Hom[C1, Tensor[A, E], X]]:
        """ Left currying followed by evaluation. """
        return cls.Equation(
            cls.uncurry_composition(f, base, exponent, left=True), f)

    @axiom
    def currying_right[A: Obj[C0], X: Atom[C0], E: Atom[C0]](
            cls, f: Hom[C1, Tensor[E, A], X],
            base: Obj[C0, X],
            exponent: Obj[C0, E]) -> Equation[Hom[C1, Tensor[E, A], X]]:
        """ Right currying followed by evaluation. """
        return cls.Equation(
            cls.uncurry_composition(f, base, exponent, left=False), f)


class Pregroup[C0, C1: Pregroup](ResiduatedMonoid[C0, C1]):
    """
    A pregroup is a residuated monoid where the left and right exponentials are
    given by tensoring with the chosen left and right duals for each object.
    """
    @property
    @abstractmethod
    def l[X: Obj[C1]](self: Obj[C1, X]) -> Obj[C1, L[X]]:
        """ The left adjoint, to be instantiated. """

    @property
    @abstractmethod
    def r[X: Obj[C1]](self: Obj[C1, X]) -> Obj[C1, R[X]]:
        """ The right adjoint, to be instantiated. """

    def over[X: Obj[C1], Y: Obj[C1]](
            self: Obj[C1, X], other: Obj[C1, Y]) -> Obj[C1, Over[X, Y]]:
        return self @ other.l

    def under[X: Obj[C1], Y: Obj[C1]](
            self: Obj[C1, X], other: Obj[C1, Y]) -> Obj[C1, Under[Y, X]]:
        return other.r @ self

    @axiom
    def adjunction[A: Obj[C0], B: Obj[C0]](
            cls, x: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ The left and right adjoints are mutually inverse. """
        return cls.Equation(x.l.r, x, x.r.l)


class RigidCategory[C0: Pregroup, C1: RigidCategory](BiclosedCategory[C0, C1]):
    """
    A rigid category is a :class:`BiclosedCategory` with a :class:`Pregroup` as
    object type and methods for :code:`cups` and :code:`caps`.
    """
    @classmethod
    @rule
    @abstractmethod
    def cups[X: Atom[C0]](
            cls, left: Obj[C0, X], right: Obj[C0, R[X]]
    ) -> Hom[C1, Tensor[X, R[X]], Unit[C0]]:
        """
        The cups witnessing :code:`right` as the adjoint of :code:`left`:
        as a rule, ``x @ x.r ⊢ 1`` is a cup, ``x.l @ x`` included.

        Parameters:
            left : The left-hand side of the cups.
            right : Its adjoint, i.e. the right-hand side of the cups.
        """

    @classmethod
    @rule
    @abstractmethod
    def caps[X: Atom[C0]](
            cls, left: Obj[C0, X], right: Obj[C0, L[X]]
    ) -> Hom[C1, Unit[C0], Tensor[X, L[X]]]:
        """
        The caps witnessing :code:`right` as the adjoint of :code:`left`:
        as a rule, ``1 ⊢ x @ x.l`` is a cap, ``x.r @ x`` included.

        Parameters:
            left : The left-hand side of the caps.
            right : Its adjoint, i.e. the right-hand side of the caps.
        """

    @classmethod
    @rule
    def ev_left[Y: Atom[C0], E: Atom[C0]](
            cls, base: Obj[C0, Y], exponent: Obj[C0, E]
    ) -> Hom[C1, Tensor[Y, L[E], E], Y]:
        """ The left evaluation of a rigid morphism is obtained using cups. """
        return base @ cls.cups(exponent.l, exponent)

    @classmethod
    @rule
    def ev_right[Y: Atom[C0], E: Atom[C0]](
            cls, base: Obj[C0, Y], exponent: Obj[C0, E]
    ) -> Hom[C1, Tensor[E, R[E], Y], Y]:
        """ The right evaluation of a rigid morphism, using cups. """
        return cls.cups(exponent, exponent.r) @ base

    @rule
    def curry_left[X: Obj[C0], Y: Atom[C0], Z: Obj[C0]](
            self: Hom[C1, Tensor[X, Y], Z]) -> Hom[C1, X, Tensor[Z, L[Y]]]:
        """ The left curry of one object of a rigid morphism. """
        return self.curry(1, left=True)

    @rule
    def curry_right[Y: Atom[C0], X: Obj[C0], Z: Obj[C0]](
            self: Hom[C1, Tensor[Y, X], Z]) -> Hom[C1, X, Tensor[R[Y], Z]]:
        """ The right curry of one object of a rigid morphism. """
        return self.curry(1, left=False)

    def curry(self, n=1, left=True):
        """
        The curry of a rigid morphism is obtained using caps.

        Parameters:
            n : The number of objects to curry.
            left : Whether to curry on the left or right.
        """
        if n < 0 or n > len(self.dom):
            raise ValueError
        if not n:
            return self
        if left:
            base, exponent = self.dom[:-n], self.dom[-n:]
            return base @ self.caps(exponent, exponent.l)\
                >> self @ exponent.l
        base, exponent = self.dom[n:], self.dom[:n]
        return self.caps(exponent.r, exponent) @ base >> exponent.r @ self

    @overload
    def base_and_exponent[X: Obj[C0], Y: Atom[C0], Z: Obj[C0]](
            self: Hom[C1, X, Tensor[Z, L[Y]]], n: int,
            left: Literal[True]) -> tuple[Obj[C0, Z], Obj[C0, Y]]: ...

    @overload
    def base_and_exponent[X: Obj[C0], Y: Atom[C0], Z: Obj[C0]](
            self: Hom[C1, X, Tensor[R[Y], Z]], n: int,
            left: Literal[False]) -> tuple[Obj[C0, Z], Obj[C0, Y]]: ...

    def base_and_exponent(self, n, left):
        """
        Contrary to :meth:`BiclosedCategory.base_and_exponent`, a pregroup has
        no exponential object to read the exponent off the codomain: it is the
        ``n`` objects at the end resp. the start of the codomain, dualised.

        Parameters:
            n : The number of objects to uncurry.
            left : Whether to uncurry on the left or right.
        """
        if n > len(self.cod):
            raise ValueError
        if left:
            return self.cod[:-n], self.cod[-n:].r
        return self.cod[n:], self.cod[:n].l

    @overload
    def transpose[A: Obj[C0], B: Obj[C0]](
            self: Hom[C1, A, B], left: Literal[False] = ...
    ) -> Hom[C1, R[B], R[A]]: ...

    @overload
    def transpose[A: Obj[C0], B: Obj[C0]](
            self: Hom[C1, A, B], left: Literal[True] = ...
    ) -> Hom[C1, L[B], L[A]]: ...

    def transpose(self, left=False):
        """
        The transpose of a morphism, i.e. its composition with cups and caps.

        Parameters:
            left : Whether to transpose left or right.

        Example
        -------
        >>> from discopy.monoidal import Equation
        >>> from discopy.rigid import Ty, Box
        >>> x, y = map(Ty, "xy")
        >>> f = Box('f', x, y)
        >>> Equation(f.transpose(left=True), f, f.transpose(),
        ...     symbols=("$\\\\mapsfrom$", "$\\\\mapsto$")).draw(
        ...         figsize=(8, 3), doctest="docs/_static/rigid/transpose.svg")

        .. image:: /_static/rigid/transpose.svg
        """
        if left:
            return self.cod.l @ self.caps(self.dom, self.dom.l)\
                >> self.cod.l @ self @ self.dom.l\
                >> self.cups(self.cod.l, self.cod) @ self.dom.l
        return self.caps(self.dom.r, self.dom) @ self.cod.r\
            >> self.dom.r @ self @ self.cod.r\
            >> self.dom.r @ self.cups(self.cod, self.cod.r)

    @axiom
    def snake_equations[X: Obj[C0]](
            cls, x: Obj[C0, X]) -> Equation[Hom[C1, X, X]]:
        """ The two snake equations. """
        snake_r = (cls.id(x) @ cls.caps(x.r, x)).then(
            cls.cups(x, x.r) @ cls.id(x))
        snake_l = (cls.caps(x, x.l) @ cls.id(x)).then(
            cls.id(x) @ cls.cups(x.l, x))
        return cls.Equation(snake_r, cls.id(x), snake_l)

    @axiom
    def caps_coherence[M: Atom[C0], N: Atom[C0], X: Obj[C0], Y: Obj[C0]](
            cls, x: Obj[C0, Tensor[M, X]],
            y: Obj[C0, Tensor[N, Y]]) -> Equation[Hom[
                C1, Unit[C0], Tensor[M, X, N, Y, L[Tensor[M, X, N, Y]]]]]:
        """ Monoidal coherence of caps. """
        return cls.Equation(
            cls.caps(x @ y, (x @ y).l),
            cls.caps(x, x.l).then(x @ cls.caps(y, y.l) @ x.l))

    @axiom
    def rotate_contravariance[A: Obj[C0], B: Obj[C0], C: Obj[C0]](
            cls, f: Hom[C1, A, B],
            g: Hom[C1, B, C]) -> Equation[Hom[C1, R[C], R[A]]]:
        """ Rotation reverses composition. """
        return cls.Equation(
            f.then(g).rotate(), g.rotate().then(f.rotate()))


class PivotalCategory[C0: Pregroup, C1: PivotalCategory](
        RigidCategory[C0, C1], TracedCategory[C0, C1]):
    """
    A pivotal category is a :class:`RigidCategory` where the left and right
    adjoints coincide, hence it is also a :class:`TracedCategory`.
    """

    @axiom
    def self_dual[X: Obj[C0]](cls, x: Obj[C0, X]) -> Equation[Obj[C0, R[X]]]:
        """ Equality of left and right adjoints. """
        return cls.ob.Equation(x.r, x.l)

    @axiom
    def pivotality[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]) -> Equation[Hom[C1, L[B], L[A]]]:
        """ Equality of left and right transposes. """
        dom, cod = f.dom, f.cod
        left_transpose = (cod.l @ cls.caps(dom, dom.l)).then(
            cod.l @ f @ dom.l).then(cls.cups(cod.l, cod) @ dom.l)
        right_transpose = (cls.caps(dom.r, dom) @ cod.r).then(
            dom.r @ f @ cod.r).then(dom.r @ cls.cups(cod, cod.r))
        return cls.Equation(left_transpose, right_transpose)


class BraidedCategory[C0: ColouredMonoid, C1: BraidedCategory](
        MonoidalCategory[C0, C1]):
    """
    A braided category is a :class:`MonoidalCategory` with a method
    :code:`braid` for the natural isomorphism :code:`x @ y -> y @ x`.
    """
    @classmethod
    @rule
    @abstractmethod
    def braid[X: Atom[C0], Y: Atom[C0]](
            cls, left: Obj[C0, X], right: Obj[C0, Y]
    ) -> Hom[C1, Tensor[X, Y], Tensor[Y, X]]:
        """
        The braid of two objects, to be instantiated: as a rule, ``x @ y
        ⊢ y @ x`` is a braid over.

        Parameters:
            left : The object on the left of the braid.
            right : The object on the right of the braid.
        """

    @classmethod
    @rule
    def braid_inverse[X: Atom[C0], Y: Atom[C0]](
            cls, left: Obj[C0, X], right: Obj[C0, Y]
    ) -> Hom[C1, Tensor[Y, X], Tensor[X, Y]]:
        """
        The inverse of the braid of two objects, crossing the other way.

        Parameters:
            left : The object on the left of the braid.
            right : The object on the right of the braid.
        """
        return cls.braid(left, right).dagger()

    @axiom
    def hexagon_left[X: Atom[C0], Y: Atom[C0], Z: Atom[C0]](
            cls, x: Obj[C0, X], y: Obj[C0, Y], z: Obj[C0, Z]
    ) -> Equation[Hom[C1, Tensor[X, Y, Z], Tensor[Y, Z, X]]]:
        """ The left hexagon equation. """
        return cls.Equation(
            cls.braid(x, y @ z),
            (cls.braid(x, y) @ z).then(y @ cls.braid(x, z)))

    @axiom
    def hexagon_right[X: Atom[C0], Y: Atom[C0], Z: Atom[C0]](
            cls, x: Obj[C0, X], y: Obj[C0, Y], z: Obj[C0, Z]
    ) -> Equation[Hom[C1, Tensor[X, Y, Z], Tensor[Z, X, Y]]]:
        """ The right hexagon equation. """
        return cls.Equation(
            cls.braid(x @ y, z),
            (x @ cls.braid(y, z)).then(cls.braid(x, z) @ y))

    @axiom
    def braid_naturality[A: Obj[C0], B: Obj[C0], C: Obj[C0], D: Obj[C0]](
            cls, f: Hom[C1, A, B], g: Hom[C1, C, D]
    ) -> Equation[Hom[C1, Tensor[A, C], Tensor[D, B]]]:
        """ Naturality of the braid. """
        return cls.Equation(
            f @ g >> cls.braid(f.cod, g.cod),
            cls.braid(f.dom, g.dom) >> g @ f)


class PROB[C1: PROB](PRO[C1], BraidedCategory[Nat, C1]):
    """
    A PROB is a :class:`BraidedCategory` whose objects are the natural
    numbers :class:`Nat`, i.e. the free braided category on one generator.
    """


class SymmetricCategory[C0: ColouredMonoid, C1: SymmetricCategory](
        BraidedCategory[C0, C1]):
    """
    A symmetric category is a :class:`BraidedCategory` where the braid is its
    own inverse called :code:`swap` for the symmetry :code:`x @ y -> y @ x`.
    """
    @classmethod
    @rule
    @abstractmethod
    def swap[X: Atom[C0], Y: Atom[C0]](
            cls, left: Obj[C0, X], right: Obj[C0, Y]
    ) -> Hom[C1, Tensor[X, Y], Tensor[Y, X]]:
        """
        The swap of two objects, to be instantiated: as a rule, ``x @ y ⊢
        y @ x`` is a swap.

        Parameters:
            left : The object on the left of the swap.
            right : The object on the right of the swap.
        """

    @classmethod
    def permutation(cls, xs: Sequence[int], doms: Sequence[C0]) -> Self:
        """ Compose swaps to permute the atomic objects in ``dom``. """
        xs, doms = list(xs), list(doms)
        if list(range(len(doms))) != sorted(xs):
            raise ValueError
        tensor = lambda objects: cls.ob().tensor(*objects)
        result, done = cls.id(tensor(doms)), cls.ob()
        while xs != list(range(len(xs))):
            i = xs[0]
            left, head = tensor(doms[:i]), tensor(doms[i:i + 1])
            result >>= done @ cls.swap(left, head) @ tensor(doms[i + 1:])
            done, doms = done @ head, doms[:i] + doms[i + 1:]
            xs = [x - 1 if x > i else x for x in xs[1:]]
        return result  # ty: ignore[invalid-return-type]

    @classmethod
    @rule
    def braid[X: Atom[C0], Y: Atom[C0]](
            cls, left: Obj[C0, X], right: Obj[C0, Y]
    ) -> Hom[C1, Tensor[X, Y], Tensor[Y, X]]:
        """ The braid of a symmetric category is its swap. """
        return cls.swap(left, right)

    @axiom
    def swap_inverse[X: Obj[C0], Y: Obj[C0]](
            cls, x: Obj[C0, X], y: Obj[C0, Y]
    ) -> Equation[Hom[C1, Tensor[X, Y], Tensor[X, Y]]]:
        """ Involutivity of the swap. """
        return cls.Equation(
            cls.swap(x, y).then(cls.swap(y, x)), cls.id(x @ y))


class PROP[C1: PROP](PROB[C1], SymmetricCategory[Nat, C1]):
    """
    A PROP is a :class:`SymmetricCategory` whose objects are the natural
    numbers :class:`Nat`, i.e. the free symmetric category on one generator.
    """


class MarkovCategory[C0: ColouredMonoid, C1: MarkovCategory](
        SymmetricCategory[C0, C1]):
    """
    A Markov category is a :class:`SymmetricCategory` with methods
    :code:`copy` and :code:`merge` for the supply of commutative comonoids.
    """
    @classmethod
    @rule
    @abstractmethod
    def copy[X: Atom[C0], N: Count](
            cls, x: Obj[C0, X], n: Obj[int, N]
    ) -> Hom[C1, X, Repeat[X, N]]:
        """
        Make :code:`n` copies of a given object :code:`x`: as a rule,
        ``x ⊢ x @ .. @ x`` is a copy, none or up to three drawn.

        Parameters:
            x : The object to copy.
            n : The number of copies, two by default in implementations.
        """

    @classmethod
    @rule
    def merge[X: Atom[C0], N: Count](
            cls, x: Obj[C0, X], n: Obj[int, N]
    ) -> Hom[C1, Repeat[X, N], X]:
        """
        Merge :code:`n` copies of a given object :code:`x`, the dagger of
        :meth:`copy`.

        Parameters:
            x : The object to merge.
            n : The number of copies, two by default in implementations.
        """
        return cls.copy(x, n).dagger()

    @axiom
    def merge_dagger[X: Atom[C0], N: Count](
            cls, x: Obj[C0, X], n: Obj[int, N]
    ) -> Equation[Hom[C1, Repeat[X, N], X]]:
        """ Merging is the dagger of copying. """
        return cls.Equation(cls.merge(x, n), cls.copy(x, n).dagger())

    @axiom
    def copy_counitality[X: Obj[C0]](
            cls, x: Obj[C0, X]) -> Equation[Hom[C1, X, X]]:
        """ Counitality of copying. """
        copy, discard = cls.copy(x, n=2), cls.copy(x, n=0)
        return cls.Equation(
            copy.then(discard @ x), cls.id(x),
            copy.then(x @ discard))

    @axiom
    def copy_coassociativity[X: Obj[C0]](
            cls, x: Obj[C0, X]) -> Equation[Hom[C1, X, Tensor[X, X, X]]]:
        """ Coassociativity of copying. """
        copy = cls.copy(x, n=2)
        return cls.Equation(
            copy.then(copy @ x), copy.then(x @ copy))

    @axiom
    def copy_cocommutativity[X: Obj[C0]](
            cls, x: Obj[C0, X]) -> Equation[Hom[C1, X, Tensor[X, X]]]:
        """ Cocommutativity of copying. """
        copy = cls.copy(x, n=2)
        return cls.Equation(copy.then(cls.swap(x, x)), copy)

    @axiom
    def discard_coherence[X: Obj[C0]](
            cls, x: Obj[C0, X]
    ) -> Equation[Hom[C1, Tensor[X, X], Unit[C0]]]:
        """ Monoidal coherence of discarding. """
        return cls.Equation(
            cls.copy(x @ x, n=0),
            cls.copy(x, n=0) @ cls.copy(x, n=0))

    @axiom
    def copy_monoidal_coherence[X: Obj[C0]](
            cls, x: Obj[C0, X]
    ) -> Equation[Hom[C1, Tensor[X, X], Tensor[X, X, X, X]]]:
        """ Monoidal coherence of copying. """
        return cls.Equation(
            cls.copy(x @ x, n=2),
            (cls.copy(x, n=2) @ cls.copy(x, n=2)).then(
                x @ cls.swap(x, x) @ x))


class ClosedCategory[C0: ResiduatedMonoid, C1: ClosedCategory](
        BiclosedCategory[C0, C1], MarkovCategory[C0, C1]):
    """
    A closed category is a symmetric :class:`BiclosedCategory`. We also assume
    it comes with copy and discard so it is also a :class:`MarkovCategory`.
    """


class DelayedMonoid[C0, C1: DelayedMonoid](ColouredMonoid[C0, C1]):
    """
    A delayed monoid is a coloured monoid with a :meth:`delay` endomorphism,
    the objects of a :class:`FeedbackCategory`: the memory it feeds back
    is one time step later on the way in, shortened to :attr:`d`.
    """
    @abstractmethod
    def delay(self, n_steps: int = 1) -> Self:
        """
        The delay of an object by some time steps, to be instantiated.

        Parameters:
            n_steps : The number of time steps to delay.
        """

    @property
    def d[X: Obj[C1]](self: Obj[C1, X]) -> Obj[C1, D[X]]:
        """ Syntactic sugar for :meth:`delay` by one time step. """
        return self.delay()


class FeedbackCategory[C0: DelayedMonoid, C1: FeedbackCategory](
        MarkovCategory[C0, C1]):
    """
    A feedback category is a :class:`MarkovCategory` whose objects are a
    :class:`DelayedMonoid`, with a :code:`delay` endofunctor and a
    :code:`feedback` operator.
    """
    @abstractmethod
    def delay(self, n_steps: int = 1) -> Self:
        """
        The delay endofunctor applied to a morphism.

        Parameters:
            n_steps : The number of time steps to delay.
        """

    @rule
    def feedback_left[A: Obj[C0], B: Obj[C0], M: Atom[C0]](
            self: Hom[C1, Tensor[D[M], A], Tensor[M, B]],
            dom: C0 | None = None, cod: C0 | None = None,
            mem: C0 | None = None) -> Hom[C1, A, B]:
        """
        The feedback of one wire of memory on the left, :meth:`feedback`
        takes a compound memory.

        Parameters:
            dom : The domain of the feedback.
            cod : The codomain of the feedback.
            mem : The memory type to feed back.
        """
        return self.feedback(dom, cod, mem, left=True)

    @rule
    def feedback_right[A: Obj[C0], B: Obj[C0], M: Atom[C0]](
            self: Hom[C1, Tensor[A, D[M]], Tensor[B, M]],
            dom: C0 | None = None, cod: C0 | None = None,
            mem: C0 | None = None) -> Hom[C1, A, B]:
        """
        The feedback of one wire of memory on the right, :meth:`feedback`
        takes a compound memory.

        Parameters:
            dom : The domain of the feedback.
            cod : The codomain of the feedback.
            mem : The memory type to feed back.
        """
        return self.feedback(dom, cod, mem)

    @overload
    def feedback[A: Obj[C0], B: Obj[C0], M: Atom[C0]](
            self: Hom[C1, Tensor[A, D[M]], Tensor[B, M]],
            dom: C0 | None = ..., cod: C0 | None = ...,
            mem: C0 | None = ...,
            left: Literal[False] = ...) -> Hom[C1, A, B]: ...

    @overload
    def feedback[A: Obj[C0], B: Obj[C0], M: Atom[C0]](
            self: Hom[C1, Tensor[D[M], A], Tensor[M, B]],
            dom: C0 | None = ..., cod: C0 | None = ...,
            mem: C0 | None = ...,
            left: Literal[True] = ...) -> Hom[C1, A, B]: ...

    @abstractmethod
    def feedback(self, dom=None, cod=None, mem=None, left=False):
        """
        The feedback operator on either side, to be instantiated: the
        rules :meth:`feedback_left` and :meth:`feedback_right`, whose
        sequents the two overloads restate, are its one-wire instances.

        Parameters:
            dom : The domain of the feedback.
            cod : The codomain of the feedback.
            mem : The memory type to feed back.
            left : Whether the memory is on the left or right.
        """

    @axiom
    def feedback_vanishing[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ Vanishing of feedback over the unit. """
        return cls.Equation(f.feedback(mem=cls.ob()), f)

    @axiom
    def feedback_joining[X: Obj[C0], M: Atom[C0], N: Atom[C0]](
            cls, f: Hom[C1, Tensor[X, D[Tensor[M, N]]], Tensor[X, M, N]]
    ) -> Equation[Hom[C1, X, X]]:
        """ Joining nested feedback loops. """
        return cls.Equation(
            f.feedback(mem=f.cod[-2:]), f.feedback().feedback())

    dagger_involution = DaggerCategory.dagger_involution.inapplicable(
        "The delay of a feedback category is not reversible.")

    dagger_contravariance = DaggerCategory.dagger_contravariance\
        .inapplicable("The delay of a feedback category is not reversible.")

    dagger_monoidality = MonoidalCategory.dagger_monoidality.inapplicable(
        "The delay of a feedback category is not reversible.")


class BalancedCategory[C0: ColouredMonoid, C1: BalancedCategory](
        BraidedCategory[C0, C1], TracedCategory[C0, C1]):
    """
    A balanced category is a :class:`BraidedCategory` and a
    :class:`TracedCategory` with a method :code:`twist` for the natural
    automorphism :code:`x -> x`.
    """
    @classmethod
    @rule
    @abstractmethod
    def twist[X: Atom[C0]](cls, dom: Obj[C0, X]) -> Hom[C1, X, X]:
        """
        The twist on an object, to be instantiated. As a rule, ``x ⊢ x``
        is a twist.

        Parameters:
            dom : The object on which to take the twist.
        """

    @axiom
    def balanced_twist[X: Atom[C0], Y: Atom[C0]](
            cls, x: Obj[C0, X], y: Obj[C0, Y]
    ) -> Equation[Hom[C1, Tensor[X, Y], Tensor[X, Y]]]:
        """ Compatibility of the twist and braid. """
        return cls.Equation(
            cls.twist(x @ y),
            cls.braid(x, y).then(
                cls.twist(y) @ cls.twist(x)).then(
                    cls.braid(y, x)))


class RibbonCategory[C0: Pregroup, C1: RibbonCategory](
        PivotalCategory[C0, C1], BalancedCategory[C0, C1]):
    """
    A ribbon category is a :class:`PivotalCategory` which is also a
    :class:`BalancedCategory`, i.e. where diagrams can draw knots and links.
    """

    @axiom
    def twist_as_trace[X: Atom[C0]](
            cls, x: Obj[C0, X]) -> Equation[Hom[C1, X, X]]:
        """ The twist as both orientations of a traced braid. """
        braid = cls.braid(x, x)
        return cls.Equation(
            braid.trace(left=True), cls.twist(x), braid.trace())


class CompactCategory[C0: Pregroup, C1: CompactCategory](
        RibbonCategory[C0, C1], SymmetricCategory[C0, C1]):
    """
    A compact category is a :class:`RibbonCategory` which is also a
    :class:`SymmetricCategory`, i.e. with cups, caps and swaps and where
    the twist is the identity.
    """
    def twist[X: Atom[C0]](cls, dom: Obj[C0, X]) -> Hom[C1, X, X]:
        """ The twist of a compact category is the identity. """
        return cls.id(dom)

    twist = classmethod(  # ty: ignore[invalid-assignment]
        rule(twist).admissible("The twist is the identity."))

    @axiom
    def reidemeister_1_cap[X: Obj[C0]](
            cls, x: Obj[C0, X]
    ) -> Equation[Hom[C1, Unit[C0], Tensor[R[X], X]]]:
        """ Reidemeister move 1 for caps. """
        return cls.Equation(
            cls.caps(x, x.r).then(cls.swap(x, x.r)),
            cls.caps(x.r, x))

    @axiom
    def reidemeister_1_cup[X: Obj[C0]](
            cls, x: Obj[C0, X]
    ) -> Equation[Hom[C1, Tensor[X, R[X]], Unit[C0]]]:
        """ Reidemeister move 1 for cups. """
        return cls.Equation(
            cls.swap(x, x.r).then(cls.cups(x.r, x)),
            cls.cups(x, x.r))


class HypergraphCategory[C0: Pregroup, C1: HypergraphCategory](
        CompactCategory[C0, C1], MarkovCategory[C0, C1]):
    """
    A hypergraph category is a symmetric category with a supply of spiders,
    i.e. special commutative Frobenius algebras on each objects.

    This makes it both a :class:`CompactCategory` and a :class:`MarkovCategory`
    """
    @classmethod
    @rule
    @abstractmethod
    def spiders[X: Atom[C0], M: Count, N: Count](
            cls, n_legs_in: Obj[int, M],
            n_legs_out: Obj[int, N],
            typ: Obj[C0, X]) -> Hom[C1, Repeat[X, M], Repeat[X, N]]:
        """
        The spiders on a given type with ``n_legs_in`` and ``n_legs_out``:
        as a rule, ``x @ .. @ x ⊢ x @ .. @ x`` is a spider, on up to three
        legs a side drawn.

        Parameters:
            n_legs_in : The number of legs in for each spider.
            n_legs_out : The number of legs out for each spider.
            typ : The type of the spiders.
        """
