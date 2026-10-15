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

Each class also declares the :func:`discopy.axioms.axiom` equations of its
structure, which every category below inherits, e.g. :class:`Category`
states the unitality and associativity of composition, the typing of its
identities and composites, and the involution and contravariance of its
dagger, and :class:`BraidedCategory` the hexagons and the naturality of the
braid. A law quantifies over the type parameters of its signature, objects
of the category, and over morphisms between them, see
:class:`discopy.axioms.Axiom`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Category
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
from typing import TYPE_CHECKING, ClassVar

from discopy.axioms import (
    Atom, Count, Delay, Equation, Hom, L, Obj, Over, R, Repeat, Tensor,
    Testable, Under, Unit, Var, axiom, rule)
from discopy import messages
from discopy.utils import (  # noqa: F401  pylint: disable=unused-import
    AxiomError, NamedGeneric, classproperty)


class Category[C0, C1: Category](Testable[C1], ABC):
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

    The class variable ``ar`` is a backward-compatible alias for
    ``factory``, since types are themselves the objects of diagrams.
    """
    ob: ClassVar[type]
    factory: ClassVar[type]
    dom: C0
    cod: C0

    ar = classproperty(lambda cls: getattr(cls, "factory", cls))

    @classmethod
    def parameters(cls) -> tuple[type, type]:
        """
        What the type parameters of the category stand for, in order: its
        objects ``C0`` and its arrows ``C1``, so that a sequent stated with
        them reads as one of the category it is bound to, see
        :func:`discopy.axioms.position`.

        >>> from discopy.monoidal import Ty, Diagram
        >>> assert Diagram.parameters() == (Ty, Diagram)
        """
        return getattr(cls, "ob", cls), cls

    @classmethod
    def equation_factory(cls, *terms) -> Equation:
        """
        Construct an equation, using strict equality by default.

        A class that quotients its equations overrides this, e.g. by
        hypergraph isomorphism from symmetric categories on, so an axiom
        built with it is checked up to whatever quotient the category
        defines — and :meth:`discopy.axioms.Axiom.modulo` weakens it
        further.
        """
        return Equation(*terms)

    @classmethod
    @abstractmethod
    def id(cls, dom: C0) -> C1:
        """
        Identity morphism on an object :code:`dom: C0`, to be instantiated.

        Parameters:
            dom (C0) : The domain of an identity is also its codomain.
        """

    @abstractmethod
    def then(self, *others: C1) -> C1:
        """
        Sequential composition of `n >= 1` morphisms, to be instantiated.

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

    def is_parallel(self, other: "Category") -> bool:
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
        return cls.equation_factory(
            cls.id(f.dom).then(f), f, f.then(cls.id(f.cod)))

    @axiom
    def associativity[A: Obj[C0], B: Obj[C0], C: Obj[C0], D: Obj[C0]](
            cls, f: Hom[C1, A, B], g: Hom[C1, B, C], h: Hom[C1, C, D]
    ) -> Equation[Hom[C1, A, D]]:
        """ Associativity of composition. """
        return cls.equation_factory(f.then(g).then(h), f.then(g.then(h)))

    @axiom
    def dagger_involution[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ The dagger is involutive. """
        return cls.equation_factory(f.dagger().dagger(), f)

    @axiom
    def dagger_contravariance[A: Obj[C0], B: Obj[C0], C: Obj[C0]](
            cls, f: Hom[C1, A, B], g: Hom[C1, B, C]
    ) -> Equation[Hom[C1, C, A]]:
        """ The dagger reverses composition. """
        return cls.equation_factory(
            f.then(g).dagger(), g.dagger().then(f.dagger()))

    __rshift__ = __llshift__ = lambda self, other: self.then(other)
    __lshift__ = __lrshift__ = lambda self, other: other.then(self)

    @rule
    def ax[X: Obj[C0]](cls, x: Var[C0, X]) -> Hom[C1, X, X]:
        """ The axiom rule: the identity on an object. """
        return cls.id(x)

    @rule
    def cut[A: Obj[C0], B: Obj[C0], C: Obj[C0]](
            cls, f: Hom[C1, A, B], g: Hom[C1, B, C]) -> Hom[C1, A, C]:
        """ The cut rule: the composition of two morphisms. """
        return f.then(g)


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

        def __getitem__(self, key) -> C1:
            """ The slices of a free monoid, assumed to stay inside it. """

    @classmethod
    def id(cls: type[C1], dom: C0 | None = None) -> C1:
        """The monoidal unit, i.e. the empty tensor ``cls()``."""
        return cls()

    @classmethod
    def unit(cls: type[C1], colour: C0 | None = None) -> C0 | C1:
        """
        The unit at a colour, i.e. the identity on it.

        It need not be an element of the monoid, which is why it may land in
        ``C0``: the layers of :class:`monoidal.Layer` are closed under
        ``tensor`` but the empty one is a type rather than a layer.
        """
        return cls.id(colour)

    @abstractmethod
    def tensor(self, *objects: C1) -> C1:
        """ The n-ary product of a monoid for ``n > 0``. """

    def then(self, *others: C1) -> C1:
        """Sequential composition, given by the monoid product."""
        return self.tensor(*others)

    @classmethod
    def cast(cls: type[C1], atoms) -> C1:
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
    def whisker(cls: type[C1], other: C0 | C1) -> C1:
        """
        Do nothing if ``other`` is already a morphism else apply :meth:`id`.

        Parameters:
            other : The object or morphism to be tensored on the left or right.
        """
        return other if isinstance(other, cls) else cls.id(other)

    def __matmul__(self, other):
        return self.tensor(other)

    def __rmatmul__(self: C1, other):
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

    def tensor(self, *others: "Nat") -> "Nat":
        if any(not isinstance(other, Nat) for other in others):
            return NotImplemented  # This allows whiskering on the left.
        return type(self)(self.n + sum(other.n for other in others))

    def __len__(self) -> int:
        return self.n

    def __index__(self) -> int:
        return self.n

    def __str__(self) -> str:
        return str(self.n)

    def __getitem__(self, key: int | slice) -> "Nat":
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
    A monoidal category is a :class:`Category` with a method :code:`tensor` for
    both its objects and its morphisms.

    This base class also implements syntactic sugar :code:`@` for whiskering.
    """

    @classmethod
    @abstractmethod
    def tensor(cls, *morphisms: C1) -> C1:
        """
        Parallel composition of ``n >= 0`` morphisms, to be instantiated.

        Parameters:
            other : The other morphism to compose in parallel.
        """

    @classmethod
    def whisker(cls, other: C0 | C1) -> C1:
        """
        Do nothing if ``other`` is already a morphism else apply :meth:`id`.

        Parameters:
            other : The object or morphism to be tensored on the left or right.
        """
        return (
            other  # ty: ignore[invalid-return-type]
            if isinstance(other, MonoidalCategory) else cls.id(other))

    def __matmul__(self, other):
        return self.tensor(self.whisker(other))

    def __rmatmul__(self: C1, other):
        return self.whisker(other).tensor(self)

    @axiom
    def bifunctoriality[
            A: Obj[C0], B: Obj[C0], C: Obj[C0], D: Obj[C0], U: Obj[C0],
            V: Obj[C0]](
            cls, f: Hom[C1, A, B], g: Hom[C1, C, D], h: Hom[C1, B, U],
            k: Hom[C1, D, V]
    ) -> Equation[Hom[C1, Tensor[A, C], Tensor[U, V]]]:
        """ Bifunctoriality of the tensor, i.e. the interchange law. """
        return cls.equation_factory(
            f @ g >> h @ k, (f >> h) @ (g >> k))

    @axiom
    def tensor_unitality[X: Obj[C0], Y: Obj[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y]
    ) -> Equation[Hom[C1, Tensor[X, Y], Tensor[X, Y]]]:
        """ Preservation of identities by tensor. """
        return cls.equation_factory(cls.id(x) @ cls.id(y), cls.id(x @ y))

    @axiom
    def dagger_monoidality[A: Obj[C0], B: Obj[C0], C: Obj[C0], D: Obj[C0]](
            cls, f: Hom[C1, A, B], g: Hom[C1, C, D]
    ) -> Equation[Hom[C1, Tensor[B, D], Tensor[A, C]]]:
        """ The dagger distributes over the tensor. """
        return cls.equation_factory(
            (f @ g).dagger(), f.dagger() @ g.dagger())

    @rule
    def mix[A: Obj[C0], B: Obj[C0], C: Obj[C0], D: Obj[C0]](
            cls, f: Hom[C1, A, B], g: Hom[C1, C, D]
    ) -> Hom[C1, Tensor[A, C], Tensor[B, D]]:
        """ The mix rule: the tensor of two morphisms. """
        return f.tensor(g)


class PRO[C1: PRO](MonoidalCategory[Nat, C1]):
    """
    A PRO is a :class:`MonoidalCategory` whose objects are the natural
    numbers :class:`Nat`, i.e. the free monoidal category on one generator.
    """


class TracedCategory[
        C0: ColouredMonoid, C1: TracedCategory](MonoidalCategory[C0, C1]):
    """
    A traced category is a :class:`MonoidalCategory` with a method
    :code:`trace` for the partial trace of a morphism over some objects.
    """
    @abstractmethod
    def trace(self, n: int = 1, left: bool = False) -> C1:
        """
        The trace of a morphism, to be instantiated.

        Tracing no object at all is the identity, i.e. the vanishing axiom
        ``f.trace(0) == f``, see `nLab
        <https://ncatlab.org/nlab/show/traced+monoidal+category>`_.

        Parameters:
            n : The number of objects to trace over.
            left : Whether to trace the wires on the left or right.
        """

    @axiom
    def trace_vanishing[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ Vanishing of a trace over the unit, on either side. """
        return cls.equation_factory(f.trace(0, left=True), f, f.trace(0))

    @axiom
    def trace_iteration_left[Γ: Obj[C0], Δ: Obj[C0], M: Obj[C0]](
            cls, f: Hom[C1, Tensor[M, Γ], Tensor[M, Δ]], mem: Var[C0, M]
    ) -> Equation[Hom[C1, Γ, Δ]]:
        """ The left trace of many wires is that of each, the first first. """
        traced = f
        for _ in range(len(mem)):
            traced = traced.trace(left=True)
        return cls.equation_factory(f.trace(len(mem), left=True), traced)

    @axiom
    def trace_iteration_right[Γ: Obj[C0], Δ: Obj[C0], M: Obj[C0]](
            cls, f: Hom[C1, Tensor[Γ, M], Tensor[Δ, M]], mem: Var[C0, M]
    ) -> Equation[Hom[C1, Γ, Δ]]:
        """ The right trace of many wires is that of each, the last first. """
        traced = f
        for _ in range(len(mem)):
            traced = traced.trace()
        return cls.equation_factory(f.trace(len(mem)), traced)

    @axiom
    def trace_superposing_left[
            Γ: Obj[C0], Δ: Obj[C0], X: Obj[C0], M: Atom[C0]](
            cls, f: Hom[C1, Tensor[M, Γ], Tensor[M, Δ]], x: Var[C0, X]
    ) -> Equation[Hom[C1, Tensor[Γ, X], Tensor[Δ, X]]]:
        """ The left trace is natural with respect to the tensor. """
        return cls.equation_factory(
            (f @ x).trace(left=True), f.trace(left=True) @ x)

    @axiom
    def trace_superposing_right[
            Γ: Obj[C0], Δ: Obj[C0], X: Obj[C0], M: Atom[C0]](
            cls, f: Hom[C1, Tensor[Γ, M], Tensor[Δ, M]], x: Var[C0, X]
    ) -> Equation[Hom[C1, Tensor[X, Γ], Tensor[X, Δ]]]:
        """ The right trace is natural with respect to the tensor. """
        return cls.equation_factory((x @ f).trace(), x @ f.trace())

    @axiom
    def trace_naturality_left[
            M: Atom[C0], X: Obj[C0], Γ: Obj[C0], Δ: Obj[C0]](
            cls, mem: Var[C0, Tensor[M, X]],
            f: Hom[C1, Tensor[M, X, Δ], Tensor[M, X, Γ]], g: Hom[C1, Γ, Δ]
    ) -> Equation[Hom[C1, Γ, Δ]]:
        """ The left trace is natural in its domain and codomain. """
        n = len(mem)
        return cls.equation_factory(
            (mem @ g >> f >> mem @ g).trace(n, left=True),
            g >> f.trace(n, left=True) >> g)

    @axiom
    def trace_naturality_right[
            M: Atom[C0], X: Obj[C0], Γ: Obj[C0], Δ: Obj[C0]](
            cls, mem: Var[C0, Tensor[M, X]],
            f: Hom[C1, Tensor[Δ, M, X], Tensor[Γ, M, X]], g: Hom[C1, Γ, Δ]
    ) -> Equation[Hom[C1, Γ, Δ]]:
        """ The right trace is natural in its domain and codomain. """
        n = len(mem)
        return cls.equation_factory(
            (g @ mem >> f >> g @ mem).trace(n), g >> f.trace(n) >> g)

    @axiom
    def trace_dinaturality_left[
            M: Atom[C0], N: Atom[C0], S: Obj[C0], T: Obj[C0], Γ: Obj[C0],
            Δ: Obj[C0]](
            cls, f: Hom[C1, Tensor[M, S, Γ], Tensor[N, T, Δ]],
            g: Hom[C1, Tensor[N, T], Tensor[M, S]]
    ) -> Equation[Hom[C1, Γ, Δ]]:
        """ A morphism slides around a left trace. """
        base, cobase = f.dom[len(g.cod):], f.cod[len(g.dom):]
        return cls.equation_factory(
            (f >> g @ cobase).trace(len(g.cod), left=True),
            (g @ base >> f).trace(len(g.dom), left=True))

    @axiom
    def trace_dinaturality_right[
            M: Atom[C0], N: Atom[C0], S: Obj[C0], T: Obj[C0], Γ: Obj[C0],
            Δ: Obj[C0]](
            cls, f: Hom[C1, Tensor[Γ, M, S], Tensor[Δ, N, T]],
            g: Hom[C1, Tensor[N, T], Tensor[M, S]]
    ) -> Equation[Hom[C1, Γ, Δ]]:
        """ A morphism slides around a right trace. """
        base = f.dom[:len(f.dom) - len(g.cod)]
        cobase = f.cod[:len(f.cod) - len(g.dom)]
        return cls.equation_factory(
            (f >> cobase @ g).trace(len(g.cod)),
            (base @ g >> f).trace(len(g.dom)))

    @rule
    def trace_left[Γ: Obj[C0], Δ: Obj[C0], M: Obj[C0]](
            cls, f: Hom[C1, Tensor[M, Γ], Tensor[M, Δ]], mem: Var[C0, M]
    ) -> Hom[C1, Γ, Δ]:
        """ The trace of a memory on the left. """
        return f.trace(len(mem), left=True)

    @rule
    def trace_right[Γ: Obj[C0], Δ: Obj[C0], M: Obj[C0]](
            cls, f: Hom[C1, Tensor[Γ, M], Tensor[Δ, M]], mem: Var[C0, M]
    ) -> Hom[C1, Γ, Δ]:
        """ The trace of a memory on the right. """
        return f.trace(len(mem))


class ResiduatedMonoid[C0, C1: ResiduatedMonoid](ColouredMonoid[C0, C1]):
    """
    A monoid is residuated when it comes with methods ``over`` and ``under``
    with syntactic sugar ``<<`` and ``>>``.
    """
    @abstractmethod
    def over(self, other: C1) -> C1:
        """ The right-to-left exponential object ``self`` to the ``other``. """

    @abstractmethod
    def under(self, other: C1) -> C1:
        """ The left-to-right exponential object ``self`` to the ``other``. """

    def __lshift__(self, other):
        return self.over(other)

    def __rshift__(self, other):
        return other.under(self)


class BiclosedCategory[
        C0: ResiduatedMonoid, C1: BiclosedCategory](MonoidalCategory[C0, C1]):
    """
    A biclosed category is a :class:`MonoidalCategory` with methods :code:`ev`
    and :code:`curry` for the evaluation and currying of morphisms.

    We also assume the type for objects comes with methods for left and right
    exponentials :code`x << y` and :code`x >> y`.
    """
    @classmethod
    @abstractmethod
    def ev(cls, base: C0, exponent: C0, left: bool = True) -> C1:
        """
        The evaluation of an exponential type, to be instantiated.

        Parameters:
            base : The base of the exponential type.
            exponent : The exponent of the exponential type.
            left : Whether to take the left or right evaluation.
        """

    @abstractmethod
    def curry(self: C1, n: int = 1, left: bool = True) -> C1:
        """
        The currying of a morphism, to be instantiated.

        Parameters:
            n : The number of objects to curry.
            left : Whether to curry on the left or right.
        """

    def base_and_exponent(self, n: int, left: bool) -> tuple[C0, C0]:
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

    def uncurry(self: C1, n: int = 1, left: bool = True) -> C1:
        """
        Uncurry a morphism by composing it with :meth:`ev`, assuming its
        codomain is an exponential object. If the exponent has less than
        ``n`` objects, we uncurry the remaining ones in turn.

        Parameters:
            n : The number of objects to uncurry.
            left : Whether to uncurry on the left or right.
        """
        if n < 0:
            raise ValueError
        if not n:
            return self
        base, exponent = self.base_and_exponent(n, left)
        result = self @ exponent >> self.ev(base, exponent, True) if left\
            else exponent @ self >> self.ev(base, exponent, False)
        return result.uncurry(n - len(exponent), left)

    @axiom
    def currying_left[Γ: Obj[C0], X: Atom[C0], E: Atom[C0]](
            cls, f: Hom[C1, Tensor[Γ, E], X]
    ) -> Equation[Hom[C1, Tensor[Γ, E], X]]:
        """ Left currying followed by evaluation. """
        return cls.equation_factory(f.curry().uncurry(), f)

    @axiom
    def currying_right[Γ: Obj[C0], X: Atom[C0], E: Atom[C0]](
            cls, f: Hom[C1, Tensor[E, Γ], X]
    ) -> Equation[Hom[C1, Tensor[E, Γ], X]]:
        """ Right currying followed by evaluation. """
        return cls.equation_factory(
            f.curry(left=False).uncurry(left=False), f)

    @axiom
    def currying_eta_left[Y: Obj[C0], E: Obj[C0]](
            cls, base: Var[C0, Y], exponent: Var[C0, E]
    ) -> Equation[Hom[C1, Over[Y, E], Over[Y, E]]]:
        """ The left currying of the left evaluation is the identity. """
        return cls.equation_factory(
            cls.ev(base, exponent).curry(len(exponent)),
            cls.id(base << exponent))

    @axiom
    def currying_eta_right[Y: Obj[C0], E: Obj[C0]](
            cls, base: Var[C0, Y], exponent: Var[C0, E]
    ) -> Equation[Hom[C1, Under[E, Y], Under[E, Y]]]:
        """ The right currying of the right evaluation is the identity. """
        return cls.equation_factory(
            cls.ev(base, exponent, left=False).curry(
                len(exponent), left=False),
            cls.id(exponent >> base))

    @axiom
    def currying_naturality_left[
            Γ: Obj[C0], Δ: Obj[C0], E: Obj[C0], Z: Obj[C0]](
            cls, f: Hom[C1, Tensor[Γ, E], Z], g: Hom[C1, Δ, Γ],
            exponent: Var[C0, E]) -> Equation[Hom[C1, Δ, Over[Z, E]]]:
        """ Left currying is natural in the context. """
        n = len(exponent)
        return cls.equation_factory(
            (g @ exponent >> f).curry(n), g >> f.curry(n))

    @axiom
    def currying_naturality_right[
            Γ: Obj[C0], Δ: Obj[C0], E: Obj[C0], Z: Obj[C0]](
            cls, f: Hom[C1, Tensor[E, Γ], Z], g: Hom[C1, Δ, Γ],
            exponent: Var[C0, E]) -> Equation[Hom[C1, Δ, Under[E, Z]]]:
        """ Right currying is natural in the context. """
        n = len(exponent)
        return cls.equation_factory(
            (exponent @ g >> f).curry(n, left=False),
            g >> f.curry(n, left=False))

    @rule
    def curry_left[Γ: Obj[C0], E: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, Tensor[Γ, E], B], exponent: Var[C0, E]
    ) -> Hom[C1, Γ, Over[B, E]]:
        """ The currying of an exponent on the left, ``Γ -> B << E``. """
        return f.curry(len(exponent))

    @rule
    def curry_right[Γ: Obj[C0], E: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, Tensor[E, Γ], B], exponent: Var[C0, E]
    ) -> Hom[C1, Γ, Under[E, B]]:
        """ The currying of an exponent on the right, ``Γ -> E >> B``. """
        return f.curry(len(exponent), left=False)

    @rule
    def ev_left[B: Obj[C0], E: Obj[C0]](
            cls, base: Var[C0, B], exponent: Var[C0, E]
    ) -> Hom[C1, Tensor[Over[B, E], E], B]:
        """ The evaluation on the left, ``(B << E) @ E -> B``. """
        return cls.ev(base, exponent)

    @rule
    def ev_right[B: Obj[C0], E: Obj[C0]](
            cls, base: Var[C0, B], exponent: Var[C0, E]
    ) -> Hom[C1, Tensor[E, Under[E, B]], B]:
        """ The evaluation on the right, ``E @ (E >> B) -> B``. """
        return cls.ev(base, exponent, left=False)


class Pregroup[C0, C1: Pregroup](ResiduatedMonoid[C0, C1]):
    """
    A pregroup is a residuated monoid where the left and right exponentials are
    given by tensoring with the chosen left and right duals for each object.
    """
    l: C1
    r: C1

    def is_adjoint(self: C1, other: C1) -> bool:
        """
        Whether ``other`` is an adjoint of ``self``, on either side.

        Parameters:
            other : The alleged left or right adjoint.

        >>> from discopy.rigid import Ty
        >>> x = Ty('x')
        >>> assert x.is_adjoint(x.l) and x.is_adjoint(x.r)
        >>> assert not x.is_adjoint(x)
        """
        return self.r == other or self == other.r

    def assert_isadjoint(self: C1, other: C1):
        """
        Raise ``AxiomError`` unless ``other`` is the right adjoint of
        ``self``, with a hint when it is the left one.

        Parameters:
            other : The alleged right adjoint.
        """
        if not self.is_adjoint(other):
            raise AxiomError(messages.NOT_ADJOINT.format(self, other))
        if self.r != other:
            raise AxiomError(messages.NOT_RIGID_ADJOINT.format(self, other))

    def over(self, other: C1) -> C1:
        return self @ other.l

    def under(self, other: C1) -> C1:
        return other.r @ self


class RigidCategory[C0: Pregroup, C1: RigidCategory](BiclosedCategory[C0, C1]):
    """
    A rigid category is a :class:`BiclosedCategory` with a :class:`Pregroup` as
    object type and methods for :code:`cups` and :code:`caps`.
    """
    @classmethod
    @abstractmethod
    def cups(cls, left: C0, right: C0) -> C1:
        """
        The cups witnessing :code:`right` as the adjoint of :code:`left`.

        Parameters:
            left : The left-hand side of the cups.
            right : Its adjoint, i.e. the right-hand side of the cups.
        """

    @classmethod
    @abstractmethod
    def caps(cls, left: C0, right: C0) -> C1:
        """
        The caps witnessing :code:`right` as the adjoint of :code:`left`.

        Parameters:
            left : The left-hand side of the caps.
            right : Its adjoint, i.e. the right-hand side of the caps.
        """

    @classmethod
    def ev(cls, base: C0, exponent: C0, left: bool = True) -> C1:
        """
        The evaluation of a rigid morphism is obtained using cups.

        Parameters:
            base : The base of the exponential type.
            exponent : The exponent of the exponential type.
            left : Whether to take the left or right evaluation.
        """
        return base @ cls.cups(exponent.l, exponent) if left\
            else cls.cups(exponent, exponent.r) @ base

    def curry(self: C1, n: int = 1, left: bool = True) -> C1:
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
            return base @ self.caps(exponent, exponent.l) >> self @ exponent.l
        base, exponent = self.dom[n:], self.dom[:n]
        return self.caps(exponent.r, exponent) @ base >> exponent.r @ self

    def base_and_exponent(self, n: int, left: bool) -> tuple[C0, C0]:
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
        return (self.cod[:-n], self.cod[-n:].r) if left\
            else (self.cod[n:], self.cod[:n].l)

    def transpose(self, left: bool = False) -> C1:
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
            cls, x: Var[C0, X]) -> Equation[Hom[C1, X, X]]:
        """ The two snake equations. """
        snake_r = x @ cls.caps(x.r, x) >> cls.cups(x, x.r) @ x
        snake_l = cls.caps(x, x.l) @ x >> x @ cls.cups(x.l, x)
        return cls.equation_factory(snake_r, cls.id(x), snake_l)

    @axiom
    def caps_coherence[M: Atom[C0], N: Atom[C0], X: Obj[C0], Y: Obj[C0]](
            cls, x: Var[C0, Tensor[M, X]], y: Var[C0, Tensor[N, Y]]
    ) -> Equation[Hom[
            C1, Unit[C0], Tensor[M, X, N, Y, L[Tensor[M, X, N, Y]]]]]:
        """ Monoidal coherence of caps. """
        return cls.equation_factory(
            cls.caps(x @ y, (x @ y).l),
            cls.caps(x, x.l) >> x @ cls.caps(y, y.l) @ x.l)

    @axiom
    def cups_coherence[M: Atom[C0], N: Atom[C0], X: Obj[C0], Y: Obj[C0]](
            cls, x: Var[C0, Tensor[M, X]], y: Var[C0, Tensor[N, Y]]
    ) -> Equation[Hom[
            C1, Tensor[M, X, N, Y, R[Tensor[M, X, N, Y]]], Unit[C0]]]:
        """ Monoidal coherence of cups. """
        return cls.equation_factory(
            cls.cups(x @ y, (x @ y).r),
            x @ cls.cups(y, y.r) @ x.r >> cls.cups(x, x.r))

    @axiom
    def transpose_contravariance[A: Obj[C0], B: Obj[C0], C: Obj[C0]](
            cls, f: Hom[C1, A, B], g: Hom[C1, B, C]
    ) -> Equation[Hom[C1, R[C], R[A]]]:
        """ Transposition reverses composition. """
        return cls.equation_factory(
            (f >> g).transpose(), g.transpose() >> f.transpose())

    @axiom
    def rotate_contravariance[A: Obj[C0], B: Obj[C0], C: Obj[C0]](
            cls, f: Hom[C1, A, B], g: Hom[C1, B, C]
    ) -> Equation[Hom[C1, R[C], R[A]]]:
        """ Rotation, i.e. the right adjoint, reverses composition. """
        return cls.equation_factory((f >> g).r, g.r >> f.r)

    @axiom
    def adjunction[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ The left and right adjoints are mutually inverse. """
        return cls.equation_factory(f.l.r, f, f.r.l)

    @rule
    def counit[X: Obj[C0]](
            cls, x: Var[C0, X]) -> Hom[C1, Tensor[X, R[X]], Unit[C0]]:
        """ The counit of a duality, i.e. a cup ``X @ X.r -> X[:0]``. """
        return cls.cups(x, x.r)

    @rule
    def unit[X: Obj[C0]](
            cls, x: Var[C0, X]) -> Hom[C1, Unit[C0], Tensor[X, L[X]]]:
        """ The unit of a duality, i.e. a cap ``X[:0] -> X @ X.l``. """
        return cls.caps(x, x.l)


class PivotalCategory[
        C0: Pregroup, C1: PivotalCategory](
            RigidCategory[C0, C1], TracedCategory[C0, C1]):
    """
    A pivotal category is a :class:`RigidCategory` where the left and right
    adjoints coincide, hence it is also a :class:`TracedCategory`.
    """

    @axiom
    def self_dual[X: Obj[C0]](
            cls, x: Var[C0, X]) -> Equation[Var[C0, R[X]]]:
        """ The left and right adjoints of an object are equal. """
        return Equation(x.r, x.l)

    @axiom
    def pivotality[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]) -> Equation[Hom[C1, L[B], L[A]]]:
        """ The left and right transposes are equal. """
        return cls.equation_factory(f.transpose(left=True), f.transpose())


class BraidedCategory[
        C0: ColouredMonoid, C1: BraidedCategory](MonoidalCategory[C0, C1]):
    """
    A braided category is a :class:`MonoidalCategory` with a method
    :code:`braid` for the natural isomorphism :code:`x @ y -> y @ x`.
    """
    @classmethod
    @abstractmethod
    def braid(cls, left: C0, right: C0) -> C1:
        """
        The braid of two objects, to be instantiated.

        Parameters:
            left : The object on the left of the braid.
            right : The object on the right of the braid.
        """

    @axiom
    def braid_then_inverse[X: Atom[C0], Y: Atom[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y]
    ) -> Equation[Hom[C1, Tensor[X, Y], Tensor[X, Y]]]:
        """ The braid followed by its inverse is the identity. """
        braid = cls.braid(x, y)
        return cls.equation_factory(braid >> braid.dagger(), cls.id(x @ y))

    @axiom
    def inverse_then_braid[X: Atom[C0], Y: Atom[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y]
    ) -> Equation[Hom[C1, Tensor[Y, X], Tensor[Y, X]]]:
        """ The inverse of the braid followed by it is the identity. """
        braid = cls.braid(x, y)
        return cls.equation_factory(braid.dagger() >> braid, cls.id(y @ x))

    @axiom
    def hexagon_left[X: Atom[C0], Y: Atom[C0], Z: Atom[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y], z: Var[C0, Z]
    ) -> Equation[Hom[C1, Tensor[X, Y, Z], Tensor[Y, Z, X]]]:
        """ The left hexagon equation. """
        return cls.equation_factory(
            cls.braid(x, y @ z), cls.braid(x, y) @ z >> y @ cls.braid(x, z))

    @axiom
    def hexagon_right[X: Atom[C0], Y: Atom[C0], Z: Atom[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y], z: Var[C0, Z]
    ) -> Equation[Hom[C1, Tensor[X, Y, Z], Tensor[Z, X, Y]]]:
        """ The right hexagon equation. """
        return cls.equation_factory(
            cls.braid(x @ y, z), x @ cls.braid(y, z) >> cls.braid(x, z) @ y)

    @axiom
    def braid_naturality[A: Obj[C0], B: Obj[C0], C: Obj[C0], D: Obj[C0]](
            cls, f: Hom[C1, A, B], g: Hom[C1, C, D]
    ) -> Equation[Hom[C1, Tensor[A, C], Tensor[D, B]]]:
        """ Naturality of the braid. """
        return cls.equation_factory(
            f @ g >> cls.braid(f.cod, g.cod),
            cls.braid(f.dom, g.dom) >> g @ f)

    @rule
    def braiding[X: Atom[C0], Y: Atom[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y]
    ) -> Hom[C1, Tensor[X, Y], Tensor[Y, X]]:
        """ The braid of two atoms. """
        return cls.braid(x, y)

    @rule
    def inverse_braiding[X: Atom[C0], Y: Atom[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y]
    ) -> Hom[C1, Tensor[Y, X], Tensor[X, Y]]:
        """ The inverse of the braid of two atoms. """
        return cls.braid(x, y).dagger()


class PROB[C1: PROB](PRO[C1], BraidedCategory[Nat, C1]):
    """
    A PROB is a :class:`BraidedCategory` whose objects are the natural
    numbers :class:`Nat`, i.e. the free braided category on one generator.
    """


class SymmetricCategory[
        C0: ColouredMonoid, C1: SymmetricCategory](BraidedCategory[C0, C1]):
    """
    A symmetric category is a :class:`BraidedCategory` where the braid is its
    own inverse called :code:`swap` for the symmetry :code:`x @ y -> y @ x`.
    """
    @classmethod
    @abstractmethod
    def swap(cls, left: C0, right: C0) -> C1:
        """
        The swap of two objects, to be instantiated.

        Parameters:
            left : The object on the left of the swap.
            right : The object on the right of the swap.
        """

    @classmethod
    def permutation(cls, xs: Sequence[int], doms: Sequence[C0]) -> C1:
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
        return result

    @classmethod
    def braid(cls, left: C0, right: C0) -> C1:
        return cls.swap(left, right)

    @axiom
    def swap_inverse[X: Obj[C0], Y: Obj[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y]
    ) -> Equation[Hom[C1, Tensor[X, Y], Tensor[X, Y]]]:
        """ Involutivity of the swap. """
        return cls.equation_factory(
            cls.swap(x, y) >> cls.swap(y, x), cls.id(x @ y))


class PROP[C1: PROP](PROB[C1], SymmetricCategory[Nat, C1]):
    """
    A PROP is a :class:`SymmetricCategory` whose objects are the natural
    numbers :class:`Nat`, i.e. the free symmetric category on one generator.
    """


class MarkovCategory[
        C0: ColouredMonoid, C1: MarkovCategory](SymmetricCategory[C0, C1]):
    """
    A Markov category is a :class:`SymmetricCategory` with methods
    :code:`copy` and :code:`merge` for the supply of commutative comonoids.
    """
    @classmethod
    @abstractmethod
    def copy(cls, x: C0, n: int = 2) -> C1:
        """
        Make :code:`n` copies of a given object :code:`x`.

        Parameters:
            x : The object to copy.
            n : The number of copies.
        """

    @axiom
    def copy_counitality[X: Obj[C0]](
            cls, x: Var[C0, X]) -> Equation[Hom[C1, X, X]]:
        """ Counitality of copying. """
        copy, discard = cls.copy(x), cls.copy(x, 0)
        return cls.equation_factory(
            copy >> discard @ x, cls.id(x), copy >> x @ discard)

    @axiom
    def copy_coassociativity[X: Obj[C0]](
            cls, x: Var[C0, X]) -> Equation[Hom[C1, X, Tensor[X, X, X]]]:
        """ Coassociativity of copying. """
        copy = cls.copy(x)
        return cls.equation_factory(copy >> copy @ x, copy >> x @ copy)

    @axiom
    def copy_cocommutativity[X: Obj[C0]](
            cls, x: Var[C0, X]) -> Equation[Hom[C1, X, Tensor[X, X]]]:
        """ Cocommutativity of copying. """
        copy = cls.copy(x)
        return cls.equation_factory(copy >> cls.swap(x, x), copy)

    @axiom
    def discard_coherence[X: Obj[C0], Y: Obj[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y]
    ) -> Equation[Hom[C1, Tensor[X, Y], Unit[C0]]]:
        """ Monoidal coherence of discarding. """
        return cls.equation_factory(
            cls.copy(x @ y, 0), cls.copy(x, 0) @ cls.copy(y, 0))

    @axiom
    def copy_coherence[X: Obj[C0], Y: Obj[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y]
    ) -> Equation[Hom[C1, Tensor[X, Y], Tensor[X, Y, X, Y]]]:
        """ Monoidal coherence of copying. """
        return cls.equation_factory(
            cls.copy(x @ y),
            cls.copy(x) @ cls.copy(y) >> x @ cls.swap(x, y) @ y)

    @rule
    def comultiplication[X: Atom[C0], N: Count](
            cls, x: Var[C0, X], n: Var[int, N]
    ) -> Hom[C1, X, Repeat[X, N]]:
        """ The comultiplication of an atom, ``n`` copies ``X -> X ** n``. """
        return cls.copy(x, n)


class ClosedCategory[
        C0: ResiduatedMonoid, C1: ClosedCategory](
            BiclosedCategory[C0, C1], MarkovCategory[C0, C1]):
    """
    A closed category is a symmetric :class:`BiclosedCategory`. We also assume
    it comes with copy and discard so it is also a :class:`MarkovCategory`.
    """

    @axiom
    def currying_symmetry[Γ: Obj[C0], E: Obj[C0], Z: Obj[C0]](
            cls, f: Hom[C1, Tensor[Γ, E], Z], exponent: Var[C0, E]
    ) -> Equation[Hom[C1, Γ, Over[Z, E]]]:
        """ Currying on the left is currying on the right after a swap. """
        n = len(exponent)
        base = f.dom[:len(f.dom) - n]
        return cls.equation_factory(
            f.curry(n), (cls.swap(exponent, base) >> f).curry(n, left=False))


class FeedbackCategory[
        C0: ColouredMonoid, C1: FeedbackCategory](MarkovCategory[C0, C1]):
    """
    A feedback category is a :class:`MarkovCategory` with a :code:`delay`
    endofunctor and a :code:`feedback` operator.
    """
    @abstractmethod
    def delay(self, n_steps: int = 1) -> C1:
        """
        The delay endofunctor applied to a morphism.

        Parameters:
            n_steps : The number of time steps to delay.
        """

    @abstractmethod
    def feedback(self, dom: C0 | None = None, cod: C0 | None = None,
                 mem: C0 | None = None) -> C1:
        """
        The feedback operator on a morphism.

        Parameters:
            dom : The domain of the feedback.
            cod : The codomain of the feedback.
            mem : The memory type to trace over.
        """

    @axiom
    def delay_unit[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ Delaying by no time step is the identity. """
        return cls.equation_factory(f.delay(0), f)

    @axiom
    def delay_composition[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]
    ) -> Equation[Hom[C1, Delay[Delay[A]], Delay[Delay[B]]]]:
        """ Delaying twice is delaying by two time steps. """
        return cls.equation_factory(f.delay().delay(), f.delay(2))

    @axiom
    def feedback_vanishing[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ Vanishing of feedback over the unit. """
        return cls.equation_factory(f.feedback(mem=f.dom[:0]), f)

    @axiom
    def feedback_joining[Γ: Obj[C0], M: Obj[C0]](
            cls, f: Hom[C1, Tensor[Γ, Delay[M]], Tensor[Γ, M]],
            mem: Var[C0, M]) -> Equation[Hom[C1, Γ, Γ]]:
        """ The feedback of many wires is that of each, the last first. """
        joined = f
        for _ in range(len(mem)):
            joined = joined.feedback()
        return cls.equation_factory(f.feedback(mem=mem), joined)

    @axiom
    def feedback_tightening[
            Γ: Obj[C0], Δ: Obj[C0], Θ: Obj[C0], Λ: Obj[C0], M: Obj[C0]](
            cls, f: Hom[C1, Tensor[Γ, Delay[M]], Tensor[Δ, M]],
            g: Hom[C1, Θ, Γ], h: Hom[C1, Δ, Λ], mem: Var[C0, M]
    ) -> Equation[Hom[C1, Θ, Λ]]:
        """ Feedback is natural in its domain and codomain. """
        return cls.equation_factory(
            (g @ mem.delay() >> f >> h @ mem).feedback(mem=mem),
            g >> f.feedback(mem=mem) >> h)

    @axiom
    def feedback_sliding[Γ: Obj[C0], Δ: Obj[C0], M: Obj[C0], N: Obj[C0]](
            cls, f: Hom[C1, Tensor[Γ, Delay[M]], Tensor[Δ, N]],
            k: Hom[C1, N, M]) -> Equation[Hom[C1, Γ, Δ]]:
        """ A morphism slides around a feedback loop, delayed on its way
        back in. """
        base = f.dom[:len(f.dom) - len(k.cod)]
        cobase = f.cod[:len(f.cod) - len(k.dom)]
        return cls.equation_factory(
            (f >> cobase @ k).feedback(mem=k.cod),
            (base @ k.delay() >> f).feedback(mem=k.dom))

    @axiom
    def feedback_superposing[
            Γ: Obj[C0], Δ: Obj[C0], X: Obj[C0], M: Obj[C0]](
            cls, f: Hom[C1, Tensor[Γ, Delay[M]], Tensor[Δ, M]],
            x: Var[C0, X], mem: Var[C0, M]
    ) -> Equation[Hom[C1, Tensor[X, Γ], Tensor[X, Δ]]]:
        """ Feedback is natural with respect to the tensor. """
        return cls.equation_factory(
            (x @ f).feedback(mem=mem), x @ f.feedback(mem=mem))

    @rule
    def delaying[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]) -> Hom[C1, Delay[A], Delay[B]]:
        """ The delay of a morphism by one time step. """
        return f.delay()

    @rule
    def feedback_right[Γ: Obj[C0], Δ: Obj[C0], M: Obj[C0]](
            cls, f: Hom[C1, Tensor[Γ, Delay[M]], Tensor[Δ, M]],
            mem: Var[C0, M]) -> Hom[C1, Γ, Δ]:
        """ The feedback of a memory on the right. """
        return f.feedback(mem=mem)


class BalancedCategory[C0: ColouredMonoid, C1: BalancedCategory](
        BraidedCategory[C0, C1], TracedCategory[C0, C1]):
    """
    A balanced category is a :class:`BraidedCategory` and a
    :class:`TracedCategory` with a method :code:`twist` for the natural
    automorphism :code:`x -> x`.
    """
    @classmethod
    @abstractmethod
    def twist(cls, dom: C0) -> C1:
        """
        The twist on an object, to be instantiated.

        Parameters:
            dom : The object on which to take the twist.
        """

    @axiom
    def balanced_twist[X: Atom[C0], Y: Atom[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y]
    ) -> Equation[Hom[C1, Tensor[X, Y], Tensor[X, Y]]]:
        """ Compatibility of the twist and braid. """
        return cls.equation_factory(
            cls.twist(x @ y),
            cls.braid(x, y) >> cls.twist(y) @ cls.twist(x)
            >> cls.braid(y, x))

    @axiom
    def twist_naturality[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ Naturality of the twist. """
        return cls.equation_factory(
            f >> cls.twist(f.cod), cls.twist(f.dom) >> f)

    @axiom
    def twist_unit(cls) -> Equation[Hom[C1, Unit[C0], Unit[C0]]]:
        """ The twist of the unit is the identity. """
        unit = cls.ob()
        return cls.equation_factory(cls.twist(unit), cls.id(unit))

    @axiom
    def yanking[X: Atom[C0]](
            cls, x: Var[C0, X]) -> Equation[Hom[C1, X, X]]:
        """ The twist is the trace of a braid, on either side. """
        braid = cls.braid(x, x)
        return cls.equation_factory(
            braid.trace(left=True), cls.twist(x), braid.trace())

    @rule
    def twisting[X: Atom[C0]](cls, x: Var[C0, X]) -> Hom[C1, X, X]:
        """ The twist of an atom. """
        return cls.twist(x)

    @rule
    def inverse_twisting[X: Atom[C0]](
            cls, x: Var[C0, X]) -> Hom[C1, X, X]:
        """ The inverse of the twist of an atom. """
        return cls.twist(x).dagger()


class RibbonCategory[C0: Pregroup, C1: RibbonCategory](
        PivotalCategory[C0, C1], BalancedCategory[C0, C1]):
    """
    A ribbon category is a :class:`PivotalCategory` which is also a
    :class:`BalancedCategory`, i.e. where diagrams can draw knots and links.
    """


class CompactCategory[C0: Pregroup, C1: CompactCategory](
        RibbonCategory[C0, C1], SymmetricCategory[C0, C1]):
    """
    A compact category is a :class:`RibbonCategory` which is also a
    :class:`SymmetricCategory`, i.e. with cups, caps and swaps and where
    the twist is the identity.
    """
    @classmethod
    def twist(cls, dom: C0) -> C1:
        return cls.id(dom)

    @axiom
    def reidemeister_1_cap[X: Obj[C0]](
            cls, x: Var[C0, X]
    ) -> Equation[Hom[C1, Unit[C0], Tensor[R[X], X]]]:
        """ Reidemeister move 1 for caps. """
        return cls.equation_factory(
            cls.caps(x, x.r) >> cls.swap(x, x.r), cls.caps(x.r, x))

    @axiom
    def reidemeister_1_cup[X: Obj[C0]](
            cls, x: Var[C0, X]
    ) -> Equation[Hom[C1, Tensor[X, R[X]], Unit[C0]]]:
        """ Reidemeister move 1 for cups. """
        return cls.equation_factory(
            cls.swap(x, x.r) >> cls.cups(x.r, x), cls.cups(x, x.r))


class HypergraphCategory[C0: Pregroup, C1: HypergraphCategory](
        CompactCategory[C0, C1], MarkovCategory[C0, C1]):
    """
    A hypergraph category is a symmetric category with a supply of spiders,
    i.e. special commutative Frobenius algebras on each objects.

    This makes it both a :class:`CompactCategory` and a :class:`MarkovCategory`
    """
    @classmethod
    @abstractmethod
    def spiders(cls, n_legs_in: int, n_legs_out: int, typ: C0) -> C1:
        """
        The spiders on a given type with ``n_legs_in`` and ``n_legs_out``.

        Parameters:
            n_legs_in : The number of legs in for each spider.
            n_legs_out : The number of legs out for each spider.
            typ : The type of the spiders.
        """

    @axiom
    def spider_fusion[X: Atom[C0], M: Count, N: Count](
            cls, x: Var[C0, X], m: Var[int, M], n: Var[int, N]
    ) -> Equation[Hom[C1, Repeat[X, M], Repeat[X, N]]]:
        """ Two spiders connected by a leg fuse into one. """
        return cls.equation_factory(
            cls.spiders(m, 1, x) >> cls.spiders(1, n, x),
            cls.spiders(m, n, x))

    @axiom
    def spider_specialness[X: Atom[C0]](
            cls, x: Var[C0, X]) -> Equation[Hom[C1, X, X]]:
        """ Splitting a wire then merging it back is the identity. """
        return cls.equation_factory(
            cls.spiders(1, 2, x) >> cls.spiders(2, 1, x), cls.id(x))

    @axiom
    def spider_commutativity[X: Atom[C0]](
            cls, x: Var[C0, X]) -> Equation[Hom[C1, Tensor[X, X], X]]:
        """ Spiders do not see the order of their legs. """
        merge = cls.spiders(2, 1, x)
        return cls.equation_factory(cls.swap(x, x) >> merge, merge)

    @rule
    def spider[X: Atom[C0], M: Count, N: Count](
            cls, x: Var[C0, X], m: Var[int, M], n: Var[int, N]
    ) -> Hom[C1, Repeat[X, M], Repeat[X, N]]:
        """ The spider of an atom, ``X ** m -> X ** n``. """
        return cls.spiders(m, n, x)
