"""
The abstract base classes for categories.

These mirror the concrete hierarchy of :mod:`discopy` modules: each class adds
the characteristic generator of its categorical structure as an
:func:`abc.abstractmethod`, e.g. :class:`BraidedCategory` is a
:class:`MonoidalCategory` with an abstract :meth:`BraidedCategory.braid`.

.. raw:: html
    :file: ../docs/api/architecture.html

Software dependencies between modules go top-to-bottom, left-to-right and
forgetful functors between categories go the other way.

Each class also declares its :func:`discopy.axioms.axiom` equations, which
every free category inherits along with the structure they axiomatise:
:class:`Category` states the unitality and associativity of composition; a
law quantifies over the type parameters of its signature, objects of the
category, and over morphisms between them, see
:class:`discopy.axioms.Declaration`; a :class:`DaggerCategory`
adds the involution and contravariance of its dagger; a
:class:`ColouredMonoid` inherits the first two as the unitality and
associativity of its product, its composition; a :class:`Groupoid` is a
dagger category that adds the invertibility of its arrows, which its dagger
inverts.

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
    Groupoid
    Group
    Nat
    MonoidalCategory
    PRO
    FeedbackCategory
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
    ComarkovCategory
    ClosedCategory
    BalancedCategory
    RibbonCategory
    CompactCategory
    HypergraphCategory
    NamedGeneric

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        keyword_form
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from types import NoneType
from typing import ClassVar

from discopy import messages
from discopy.axioms import (
    Delay, Equation, Hom, Obj, Tensor, Testable, Var, axiom)
from discopy.utils import (  # noqa: F401  pylint: disable=unused-import
    AxiomError, NamedGeneric, classproperty, factory_name)


class Category[C0, C1: Category](ABC):
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
    ob: ClassVar[type[C0]]
    factory: ClassVar[type[C1]]
    dom: C0
    cod: C0

    #: Backward-compatible alias for :attr:`factory`, since types are
    #: themselves the objects of diagrams.
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
        return getattr(cls, "ob", cls), cls.ar

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
        return cls.equation_factory(
            cls.id(f.dom).then(f), f, f.then(cls.id(f.cod)))

    @axiom
    def associativity[A: Obj[C0], B: Obj[C0], C: Obj[C0], D: Obj[C0]](
            cls, f: Hom[C1, A, B], g: Hom[C1, B, C], h: Hom[C1, C, D]
    ) -> Equation[Hom[C1, A, D]]:
        """ Associativity of composition. """
        return cls.equation_factory(
            f.then(g).then(h), f.then(g.then(h)))

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
    def dagger(self) -> C1:
        """ The dagger of a morphism, to be instantiated. """

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


class ColouredMonoid[C0, C1: ColouredMonoid](Category[C0, C1]):
    """
    A coloured monoid is a category whose sequential composition ``then`` is
    given by a monoidal ``tensor``, with the objects ``C0`` (its colours) as
    the boundaries of its morphisms.

    An ordinary :obj:`Monoid` is the special case with a single, trivial
    colour, i.e. :class:`type(None)`. We do not enforce this so
    that e.g. :class:`monoidal.Ty` can take colours as objects.
    """
    @classmethod
    def id(cls, dom: C0 = None) -> C1:
        """The monoidal unit, i.e. the empty tensor ``cls()``."""
        return cls()

    @classmethod
    def unit(cls, colour: C0 = None) -> C0 | C1:
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
    def cast(cls, atoms) -> C1:
        """
        The element of a tuple of atoms, or of a single atom; an element of
        the monoid is unchanged.

        Parameters:
            atoms : An element, a tuple of atoms or a single atom.

        >>> assert Nat.cast(2) == Nat(2) == Nat.cast(Nat(2))
        """
        if isinstance(atoms, cls):
            return atoms
        return cls(*atoms) if isinstance(atoms, tuple) else cls(atoms)

    @classmethod
    def whisker(cls, other: C0 | C1) -> C1:
        """
        Do nothing if ``other`` is already a morphism else apply :meth:`id`.

        Parameters:
            other : The object or morphism to be tensored on the left or right.
        """
        return other if isinstance(other, cls) else cls.id(other)

    def __matmul__(self, other):
        return self.tensor(other)

    def __rmatmul__(self, other):
        return self.whisker(other).tensor(self)


class Monoid[C1: Monoid](ColouredMonoid[type(None), C1]):
    """ A monoid is a coloured monoid with a single, trivial colour. """


class Groupoid[C0, C1: Groupoid](DaggerCategory[C0, C1]):
    """
    A groupoid is a category in which every arrow is invertible, its inverse
    being its dagger: a dagger category whose dagger is the inverse.
    """
    @abstractmethod
    def dagger(self) -> C1:
        """ The inverse of an arrow. """

    @axiom
    def invertibility[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]
    ) -> Equation[tuple[Hom[C1, A, A], Hom[C1, B, B]]]:
        """ An arrow followed or preceded by its dagger is an identity. """
        return cls.equation_factory(
            (f.then(f.dagger()), f.dagger().then(f)),
            (cls.id(f.dom), cls.id(f.cod)))


class Group[C1: Group](Groupoid[type(None), C1], Monoid[C1]):
    """
    A group is a monoid whose elements are invertible, i.e. a groupoid with
    a single object :obj:`None`, which every element goes from and to.
    """
    dom = cod = None

    @classmethod
    def parameters(cls) -> tuple[type, type]:
        """ The objects of a group are the one :obj:`None`. """
        return NoneType, cls


@dataclass
class Nat(Testable["Nat"], Monoid["Nat"]):
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

    def to_tree(self) -> dict:
        """ The tree of a natural number, see :func:`discopy.utils.dumps`. """
        return {'factory': factory_name(type(self)), 'n': self.n}

    @classmethod
    def from_tree(cls, tree: dict) -> Nat:
        """ The natural number of a tree, see :func:`discopy.utils.loads`. """
        return cls(tree['n'])

    @classmethod
    def strategy(cls, *, max_size=3):
        """
        Generate natural numbers up to ``max_size``, so that a category
        whose objects they are, e.g. :class:`discopy.matrix.Matrix`,
        quantifies over them like any other :class:`discopy.axioms.Testable`.
        """
        from hypothesis import strategies as st

        return st.integers(min_value=0, max_value=max_size).map(cls)


class MonoidalCategory[C0: ColouredMonoid, C1: MonoidalCategory](
        Category[C0, C1]):
    """
    A monoidal category is a :class:`Category` with a method :code:`tensor` for
    both its objects and its morphisms.

    This base class also implements syntactic sugar :code:`@` for whiskering.

    A ``trace`` defined in a subclass, in the positional form
    ``trace(n, left)``, takes the keyword form ``trace(dom=, cod=, mem=)``
    of :meth:`FeedbackCategory.feedback` too, through :func:`keyword_form`,
    whether the subclass declares :class:`TracedCategory` or only has a
    trace, as :class:`discopy.matrix.Matrix` and
    :class:`discopy.hypergraph.Hypergraph` do.
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "trace" in vars(cls):
            cls.trace = keyword_form(vars(cls)["trace"])

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
        return other if isinstance(other, MonoidalCategory) else cls.id(other)

    def __matmul__(self, other):
        return self.tensor(self.whisker(other))

    def __rmatmul__(self, other):
        return self.whisker(other).tensor(self)

    def split_memory(
            self, n: int = None, left: bool = False, *,
            dom: C0 = None, cod: C0 = None, mem: C0 = None
            ) -> tuple[C0, C0, C0]:
        """
        The domain, codomain and memory of a feedback or a trace, read off
        either form of :meth:`FeedbackCategory.feedback`: the memory is
        ``mem``, or the objects of the codomain beside ``cod``, or the last
        ``n`` objects of the codomain, the first when ``left``, one by
        default; the domain and codomain default to what the memory leaves.
        Each of them is cast into the type of the codomain, e.g. an integer
        into the :class:`Nat` of a :class:`discopy.matrix.Matrix`.

        Example
        -------
        >>> from discopy.feedback import Ty, Box
        >>> x, y, m = map(Ty, "xym")
        >>> f = Box('f', x @ m.delay(), y @ m)
        >>> assert f.split_memory() == f.split_memory(mem=m) == (x, y, m)
        >>> assert f.split_memory(cod=y) == f.split_memory(1) == (x, y, m)
        """
        if n is not None and (dom, cod, mem) != (None, None, None):
            raise TypeError(messages.FEEDBACK_FORMS)
        dom, cod, mem = (x if x is None else type(self.cod).cast(x)
                         for x in (dom, cod, mem))
        if mem is None:
            size = len(self.cod) - len(cod) if cod is not None\
                else 1 if n is None else n
            if not 0 <= size <= len(self.cod):
                raise ValueError(
                    messages.NOT_SPLITTABLE.format(size, self.cod))
            mem = self.cod[:size] if left\
                else self.cod[len(self.cod) - size:]
        size = len(mem)
        if size > min(len(self.dom), len(self.cod)):
            raise ValueError(messages.NOT_SPLITTABLE.format(size, self))
        if dom is None:
            dom = self.dom[size:] if left else self.dom[:len(self.dom) - size]
        if cod is None:
            cod = self.cod[size:] if left else self.cod[:len(self.cod) - size]
        return dom, cod, mem

    def memory_size(self, mem: C0) -> int:
        """
        The number of objects in a memory, as the positional form of
        :meth:`FeedbackCategory.feedback` counts them, i.e. its length.

        Parameters:
            mem : The memory.
        """
        return len(mem)


class PRO[C1: PRO](MonoidalCategory[Nat, C1]):
    """
    A PRO is a :class:`MonoidalCategory` whose objects are the natural
    numbers :class:`Nat`, i.e. the free monoidal category on one generator.
    """


class FeedbackCategory[C0, C1](MonoidalCategory[C0, C1]):
    """
    A feedback category is a :class:`MonoidalCategory` with a monoidal
    endofunctor :meth:`delay` and an operator :meth:`feedback` from
    morphisms ``x @ m.delay() -> y @ m`` to morphisms ``x -> y``, see
    :cite:t:`KatisEtAl02` and :cite:t:`DiLavoreEtAl22`.

    The laws of feedback are stated once here: vanishing, joining, strength,
    tightening and sliding. A :class:`TracedCategory` is a feedback category
    whose delay is the identity.
    """
    def delay(self, n_steps: int = 1) -> C1:
        """
        The delay endofunctor applied to a morphism, the identity by default.

        Parameters:
            n_steps : The number of time steps to delay.
        """
        del n_steps
        return self

    @abstractmethod
    def feedback(
            self, n: int = None, left: bool = False, *,
            dom: C0 = None, cod: C0 = None, mem: C0 = None) -> C1:
        """
        The feedback operator on a morphism ``dom @ mem.delay() -> cod @ mem``
        giving a morphism ``dom -> cod``, to be instantiated.

        The keyword form ``dom``, ``cod``, ``mem`` is primary and the
        positional form ``n`` its fallback, the memory being the last ``n``
        objects of the codomain, the first when ``left``: the two forms are
        exclusive, see :meth:`split_memory`.

        Parameters:
            n : The number of objects to feed back.
            left : Whether to feed back on the left rather than the right.
            dom : The domain of the feedback.
            cod : The codomain of the feedback.
            mem : The memory to feed back.
        """

    @axiom
    def vanishing[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ Feeding back the unit does nothing. """
        return cls.equation_factory(f.feedback(0), f)

    @axiom
    def joining[X: Obj[C0], Y: Obj[C0], M: Obj[C0], N: Obj[C0]](
            cls, f: Hom[C1, Tensor[X, Delay[C1, Tensor[M, N]]],
                        Tensor[Y, M, N]],
            x: Var[C0, X], y: Var[C0, Y], m: Var[C0, M], n: Var[C0, N]
    ) -> Equation[Hom[C1, X, Y]]:
        """ Feeding back a tensor is feeding back its factors in turn. """
        return cls.equation_factory(
            f.feedback(dom=x, cod=y, mem=m @ n),
            f.feedback(dom=x @ cls.id(m).delay().dom, cod=y @ m, mem=n
                       ).feedback(dom=x, cod=y, mem=m))

    @axiom
    def strength[
            U: Obj[C0], V: Obj[C0], X: Obj[C0], Y: Obj[C0], M: Obj[C0]](
            cls, e: Hom[C1, U, V],
            f: Hom[C1, Tensor[X, Delay[C1, M]], Tensor[Y, M]],
            x: Var[C0, X], y: Var[C0, Y], m: Var[C0, M]
    ) -> Equation[Hom[C1, Tensor[U, X], Tensor[V, Y]]]:
        """ Feedback commutes with tensoring a morphism on the left. """
        return cls.equation_factory(
            e @ f.feedback(dom=x, cod=y, mem=m),
            (e @ f).feedback(dom=e.dom @ x, cod=e.cod @ y, mem=m))

    @axiom
    def tightening[
            W: Obj[C0], X: Obj[C0], Y: Obj[C0], Z: Obj[C0], M: Obj[C0]](
            cls, f: Hom[C1, Tensor[X, Delay[C1, M]], Tensor[Y, M]],
            g: Hom[C1, W, X], h: Hom[C1, Y, Z],
            x: Var[C0, X], y: Var[C0, Y], m: Var[C0, M]
    ) -> Equation[Hom[C1, W, Z]]:
        """ Feedback is natural in its domain and its codomain. """
        return cls.equation_factory(
            (g @ cls.id(m).delay() >> f >> h @ m).feedback(
                dom=g.dom, cod=h.cod, mem=m),
            g >> f.feedback(dom=x, cod=y, mem=m) >> h)

    @axiom
    def sliding[X: Obj[C0], Y: Obj[C0], M: Obj[C0]](
            cls, f: Hom[C1, Tensor[X, Delay[C1, M]], Tensor[Y, M]],
            k: Hom[C1, M, M], x: Var[C0, X], y: Var[C0, Y], m: Var[C0, M]
    ) -> Equation[Hom[C1, X, Y]]:
        """ A morphism on the memory slides along the feedback, delayed. """
        return cls.equation_factory(
            (f >> y @ k).feedback(dom=x, cod=y, mem=m),
            (x @ k.delay() >> f).feedback(dom=x, cod=y, mem=m))


def keyword_form(trace: Callable) -> Callable:
    """
    A method in the positional form ``(n, left)``, e.g. a trace, that takes
    the keyword form ``dom``, ``cod``, ``mem`` of
    :meth:`FeedbackCategory.feedback` as well, the two forms exclusive: the
    memory is read off the boundary with
    :meth:`MonoidalCategory.split_memory`, checked to be what the codomain
    feeds back and passed on as its :meth:`MonoidalCategory.memory_size`.
    The domain is checked on the result, since the boundary takes in the
    delay of the memory when the trace is the
    :meth:`TracedCategory.feedback` of a category that labels the two apart,
    e.g. a :class:`discopy.drawing.Drawing`. A method that takes ``mem``
    already is returned as it is, so that a class wraps the trace it defines
    once, whichever trace that one calls.

    Parameters:
        trace : The method in its positional form.

    Example
    -------
    >>> from discopy.matrix import Matrix
    >>> f = Matrix[bool]([[1, 1], [0, 0]], 2, 2)
    >>> assert f.trace(mem=1) == f.trace() != f.trace(mem=1, left=True)
    >>> assert keyword_form(Matrix.trace) is Matrix.trace
    """
    if "mem" in signature(trace).parameters:
        return trace

    @wraps(trace)
    def method(self, n: int = None, left: bool = False, *,
               dom=None, cod=None, mem=None):
        if (dom, cod, mem) == (None, None, None):
            return trace(self, left=left) if n is None\
                else trace(self, n, left)
        dom, cod, mem = self.split_memory(
            n, left, dom=dom, cod=cod, mem=mem)
        if self.cod != (mem @ cod if left else cod @ mem):
            raise AxiomError(messages.NOT_TRACEABLE.format(self, mem))
        result = trace(self, self.memory_size(mem), left)
        if result.dom != dom:
            raise AxiomError(messages.NOT_TRACEABLE.format(self, mem))
        return result
    method.__signature__ = signature(method, follow_wrapped=False)
    return method


class TracedCategory[C0, C1](FeedbackCategory[C0, C1]):
    """
    A traced category is a feedback category with the identity as delay plus
    the yanking axiom: its :meth:`feedback` is its :meth:`trace`, the partial
    trace of a morphism over some objects, and the trace of the swap of an
    object with itself is its identity, that of a braid its twist, see
    :meth:`yanking`, :meth:`BalancedCategory.yanking` and
    :cite:t:`JoyalEtAl96`. The free diagrams of :mod:`discopy.feedback` and the
    levels below it, whose wires carry a time step, are traced on their
    undelayed wires, the :class:`discopy.feedback.Trace` bubble being the
    feedback over an undelayed memory.
    """
    @abstractmethod
    def trace(self, n: int = 1, left: bool = False) -> C1:
        """
        The trace of a morphism, to be instantiated in its positional form,
        over the last ``n`` objects or the first when ``left``: it takes the
        keyword form ``dom``, ``cod``, ``mem`` of :meth:`feedback` as well,
        see :func:`keyword_form`.

        Tracing no object at all is the identity, i.e. the vanishing axiom
        ``f.trace(0) == f``, see `nLab
        <https://ncatlab.org/nlab/show/traced+monoidal+category>`_.

        Parameters:
            n : The number of objects to trace over.
            left : Whether to trace the wires on the left or right.
        """

    def feedback(
            self, n: int = None, left: bool = False, *,
            dom: C0 = None, cod: C0 = None, mem: C0 = None) -> C1:
        """
        The feedback of a traced category is its :meth:`trace` over the
        memory, whose delay is itself, in either form.
        """
        # pylint: disable-next=unexpected-keyword-arg  # see keyword_form
        return self.trace(n, left, dom=dom, cod=cod, mem=mem)

    @axiom
    def yanking[X: Obj[C0]](
            cls, x: Var[C0, X]) -> Equation[Hom[C1, X, X]]:
        """
        Tracing the swap of an object with itself gives its identity, the
        yanking axiom of a symmetric traced category: a braided one states it
        with its twist, see :meth:`BalancedCategory.yanking`, and a planar
        one, which has no swap, declares it inapplicable.
        """
        # pylint: disable-next=no-member  # stated where there is a swap
        return cls.equation_factory(cls.swap(x, x).trace(mem=x), cls.id(x))


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
    def curry(self, n: int = 1, left: bool = True) -> C1:
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

    def uncurry(self, n: int = 1, left: bool = True) -> C1:
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

    def curry(self, n: int = 1, left: bool = True) -> C1:
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


class PivotalCategory[C0, C1](RigidCategory[C0, C1], TracedCategory[C0, C1]):
    """
    A pivotal category is a :class:`RigidCategory` where the left and right
    adjoints coincide, hence it is also a :class:`TracedCategory`, a planar
    one with no swap to yank.
    """
    yanking = TracedCategory.yanking.inapplicable(
        messages.PLANAR_YANKING)


class BraidedCategory[C0, C1](MonoidalCategory[C0, C1]):
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


class PROB[C1: PROB](PRO[C1], BraidedCategory[Nat, C1]):
    """
    A PROB is a :class:`BraidedCategory` whose objects are the natural
    numbers :class:`Nat`, i.e. the free braided category on one generator.
    """


class SymmetricCategory[C0, C1](BraidedCategory[C0, C1]):
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
    def from_permutation(cls, perm: Sequence[int], dom: C0) -> C1:
        """
        The permutation of the atomic objects of ``dom`` by
        :meth:`permutation`, i.e. a composition of swaps unless the category
        writes it directly; a free category overrides it with a native box,
        see :meth:`discopy.symmetric.Diagram.from_permutation`.
        """
        return cls.permutation(perm, dom)

    @classmethod
    def braid(cls, left: C0, right: C0) -> C1:
        return cls.swap(left, right)


class PROP[C1: PROP](PROB[C1], SymmetricCategory[Nat, C1]):
    """
    A PROP is a :class:`SymmetricCategory` whose objects are the natural
    numbers :class:`Nat`, i.e. the free symmetric category on one generator.
    """


class MarkovCategory[C0, C1](SymmetricCategory[C0, C1]):
    """
    A Markov category is a :class:`SymmetricCategory` with a method
    :code:`copy` for the supply of commutative comonoids: ``copy(x, n)`` is
    the comultiplication into ``n`` copies and :meth:`discard` its counit,
    the case ``n = 0``. The laws of a commutative comonoid and the coherence
    of the supply with the tensor are stated here, see
    :class:`ComarkovCategory` for the dual.
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

    @classmethod
    def discard(cls, x: C0) -> C1:
        """
        The counit of the comonoid on :code:`x`, i.e. no copy of it at all.

        Parameters:
            x : The object to discard.
        """
        return cls.copy(x, 0)

    @axiom
    def copy_coassociativity[X: Obj[C0]](
            cls, x: Var[C0, X])\
            -> Equation[Hom[C1, X, Tensor[X, X, X]]]:
        """ Copying either of two copies gives the three copies. """
        copy, identity = cls.copy(x), cls.id(x)
        return cls.equation_factory(
            copy >> copy @ identity, cls.copy(x, 3), copy >> identity @ copy)

    @axiom
    def copy_cocommutativity[X: Obj[C0]](
            cls, x: Var[C0, X])\
            -> Equation[Hom[C1, X, Tensor[X, X]]]:
        """ The two copies of an object are interchangeable. """
        copy = cls.copy(x)
        return cls.equation_factory(copy >> cls.swap(x, x), copy)

    @axiom
    def copy_counitality[X: Obj[C0]](
            cls, x: Var[C0, X]) -> Equation[Hom[C1, X, X]]:
        """ Discarding either of two copies leaves the object itself. """
        copy, discard, identity = cls.copy(x), cls.discard(x), cls.id(x)
        return cls.equation_factory(
            copy >> discard @ identity, identity, copy >> identity @ discard)

    @axiom
    def copy_coherence[X: Obj[C0], Y: Obj[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y])\
            -> Equation[Hom[C1, Tensor[X, Y], Tensor[X, Y, X, Y]]]:
        """
        Copying a tensor is copying its factors, with the inner copies
        swapped so that each copy of the tensor comes together.
        """
        return cls.equation_factory(
            cls.copy(x @ y),
            cls.copy(x) @ cls.copy(y)
            >> cls.id(x) @ cls.swap(x, y) @ cls.id(y))


class ComarkovCategory[C0, C1](SymmetricCategory[C0, C1]):
    """
    A comarkov category is a :class:`SymmetricCategory` with a method
    :code:`merge` for the supply of commutative monoids: ``merge(x, n)`` is
    the multiplication of ``n`` copies and :meth:`noise` its unit, the case
    ``n = 0``. It is the dual of a :class:`MarkovCategory`: a category is
    comarkov when its opposite is Markov. The laws of a commutative monoid
    and the coherence of the supply with the tensor are stated here.
    """
    @classmethod
    @abstractmethod
    def merge(cls, x: C0, n: int = 2) -> C1:
        """
        Merge :code:`n` copies of a given object :code:`x` into one.

        Parameters:
            x : The object to merge.
            n : The number of copies.
        """

    @classmethod
    def noise(cls, x: C0) -> C1:
        """
        The unit of the monoid on :code:`x`, i.e. the merge of no copy at
        all: the state every object carries, dual to
        :meth:`MarkovCategory.discard`.

        Parameters:
            x : The object of the noise.
        """
        return cls.merge(x, 0)

    @axiom
    def merge_associativity[X: Obj[C0]](
            cls, x: Var[C0, X])\
            -> Equation[Hom[C1, Tensor[X, X, X], X]]:
        """ Merging either of two merges gives the merge of three. """
        merge, identity = cls.merge(x), cls.id(x)
        return cls.equation_factory(
            merge @ identity >> merge, cls.merge(x, 3),
            identity @ merge >> merge)

    @axiom
    def merge_commutativity[X: Obj[C0]](
            cls, x: Var[C0, X])\
            -> Equation[Hom[C1, Tensor[X, X], X]]:
        """ The two copies merged into an object are interchangeable. """
        merge = cls.merge(x)
        return cls.equation_factory(cls.swap(x, x) >> merge, merge)

    @axiom
    def merge_unitality[X: Obj[C0]](
            cls, x: Var[C0, X]) -> Equation[Hom[C1, X, X]]:
        """ Merging the object with noise on either side leaves it itself. """
        merge, noise, identity = cls.merge(x), cls.noise(x), cls.id(x)
        return cls.equation_factory(
            noise @ identity >> merge, identity, identity @ noise >> merge)

    @axiom
    def merge_coherence[X: Obj[C0], Y: Obj[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y])\
            -> Equation[Hom[C1, Tensor[X, Y, X, Y], Tensor[X, Y]]]:
        """
        Merging a tensor is swapping the inner factors so that the copies
        of each factor come together, then merging its factors.
        """
        return cls.equation_factory(
            cls.merge(x @ y),
            cls.id(x) @ cls.swap(y, x) @ cls.id(y)
            >> cls.merge(x) @ cls.merge(y))


class ClosedCategory[C0, C1](BiclosedCategory[C0, C1], MarkovCategory[C0, C1]):
    """
    A closed category is a symmetric :class:`BiclosedCategory`. We also assume
    it comes with copy and discard so it is also a :class:`MarkovCategory`.
    """


class BalancedCategory[C0, C1](
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
    def yanking[X: Obj[C0]](
            cls, x: Var[C0, X]) -> Equation[Hom[C1, X, X]]:
        """
        Tracing the braid of an object with itself gives its twist, the
        yanking axiom of a balanced traced category, see
        :cite:t:`JoyalEtAl96`: the twist is the identity when the braid is a
        swap.
        """
        return cls.equation_factory(
            cls.braid(x, x).trace(mem=x), cls.twist(x))


class RibbonCategory[C0, C1](
        PivotalCategory[C0, C1], BalancedCategory[C0, C1]):
    """
    A ribbon category is a :class:`PivotalCategory` which is also a
    :class:`BalancedCategory`, i.e. where diagrams can draw knots and links:
    it yanks as a balanced category does, with the braid a planar pivotal
    category lacks.
    """
    yanking = BalancedCategory.yanking


class CompactCategory[C0, C1](
        RibbonCategory[C0, C1], SymmetricCategory[C0, C1]):
    """
    A compact category is a :class:`RibbonCategory` which is also a
    :class:`SymmetricCategory`, i.e. with cups, caps and swaps and where
    the twist is the identity.
    """
    @classmethod
    def twist(cls, dom: C0) -> C1:
        return cls.id(dom)


class HypergraphCategory[C0, C1](
        CompactCategory[C0, C1], MarkovCategory[C0, C1],
        ComarkovCategory[C0, C1]):
    """
    A hypergraph category is a symmetric category with a supply of spiders,
    i.e. special commutative Frobenius algebras on each objects.

    This makes it a :class:`CompactCategory`, a :class:`MarkovCategory` and
    a :class:`ComarkovCategory`, the Frobenius law and speciality
    identifying the comonoids with the monoids.
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
