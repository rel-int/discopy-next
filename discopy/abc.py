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
structural methods carry their own :func:`discopy.pattern.rule`, from
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
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import count
from types import NoneType
from typing import (
    Any, ClassVar, Literal, Self, TYPE_CHECKING, TypeVar, get_args,
    get_origin, overload)

from discopy.axioms import (  # noqa: F401  pylint: disable=unused-import
    Axiom, axiom, declarations, Equation, Rule, rule, Serialisable, Testable)
from discopy.pattern import (  # noqa: F401  pylint: disable=unused-import
    AdjDir, Atom, Count, D, ExpDir, Hom, L, Obj, Over, R, Repeat, Tensor,
    TensorDir, Under, Unit, Var, OBJECTS, Match, Sort, heads, instantiate,
    match, variables)
from discopy.utils import (  # noqa: F401  pylint: disable=unused-import
    NamedGeneric, classproperty, factory_name)


class DeadEnd(Exception):
    """ A goal nothing closes within the depth: :meth:`Category.search`
    retries it a bounded number of times before rejecting the example. """


def atoms(value) -> list:
    """ The atoms of an object, each as an object of length one. """
    return [value[i:i + 1] for i in range(len(value))]


def materials(value) -> Iterator:
    """ The subformulae of an object: its atoms, and recursively the
    base and exponent of each exponential atom. """
    for i in range(len(value)):
        yield value[i:i + 1]
        for part in ("base", "exponent"):
            inner = getattr(value.inside[i], part, None)
            if inner is not None:
                yield from materials(inner)


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
        structural method carries, see :func:`discopy.pattern.rule`, bound
        to ``cls`` and owned by the class declaring it, the latest in the
        method resolution order winning like ordinary attribute lookup. A
        method implementing a rule is decorated
        :func:`discopy.pattern.rule` itself, restating its sequent, and
        the search calls it by name; one declared
        :meth:`discopy.pattern.Rule.inapplicable` or
        :meth:`discopy.pattern.Rule.admissible` is dropped.
        """
        return declarations(cls, Rule)

    @classproperty
    def generators(cls: type) -> dict[str, Rule]:
        """
        The logical constants inherited by ``cls``, by name: the rules
        with no hom premise, which the search builds in one step, see
        :meth:`discopy.pattern.Rule.recursive`. A class adjusts the set by
        assigning a dictionary of rules instead,
        :meth:`discopy.pattern.Rule.constant` giving the rule of one given
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
    def search(
            cls, free: Callable | None = None, *, dom=None, cod=None,
            types=None, max_depth: int = 3, epsilon: float = 0.05):
        """
        Generate a term of the category toward a goal whose two sides are
        patterns under one shared substitution: a side is a type, a
        pattern, a type parameter standing for its variable, or
        :obj:`None` for a fresh variable, so the goal ``A ⊢ A`` finds an
        endomorphism on anything.

        This is a template: a term is one of the :meth:`leaves` of the
        goal — a free box or a generator — or, below the depth bound, one
        of its :meth:`branches`, a recursive rule whose premises are
        searched in turn. The category says how the conclusion of the rule
        it applies meets the goal, :meth:`contexts`, and which rules it
        commits to without a choice, :meth:`focus`. A side guides the
        search once its variables are all bound, and constrains it
        afterwards: a failed unification is a :class:`DeadEnd`, retried a
        bounded number of times before the example is rejected, unless the
        side was fully bound already — then the term was built outside the
        declared conclusion, an :class:`discopy.utils.AxiomError`.

        Parameters:
            free : A strategy factory ``free(dom=, cod=, types=)`` for the
                free generator, e.g. the strategy of the boxes, or
                :obj:`None` for a category generated by a fixed vocabulary.
            dom : The domain of the goal.
            cod : The codomain of the goal.
            types : A strategy for the objects, overriding that of ``C0``.
            max_depth : The number of nested rules a term may apply.
            epsilon : The chance of escaping the focusing discipline: a
                goal with a :meth:`focus` rule commits to it, except with
                probability ``epsilon``.

        >>> from hypothesis import find
        >>> from discopy.monoidal import Ty, Diagram, Box
        >>> x, y = Ty('x'), Ty('y')
        >>> term = find(Diagram.search(Box.strategy, dom=x, cod=y),
        ...             lambda term: len(term.boxes) > 1)
        >>> assert (term.dom, term.cod) == (x, y) and len(term.boxes) > 1
        """
        from hypothesis import assume, strategies as st
        from discopy.utils import AxiomError

        scope, fresh = heads(cls), count()

        def as_pattern(side):
            return side if side is not None else TypeVar(  # ty: ignore[invalid-legacy-type-variable]
                f"?{next(fresh)}")

        def is_pattern(side):
            return isinstance(side, TypeVar) or get_origin(side) is not None

        def guidance(side, subst):
            if not is_pattern(side):
                return side
            if all(name in subst for name in variables(side)):
                return instantiate(side, subst, scope[OBJECTS])
            return None

        @st.composite
        def attempt(draw, goal, depth, subst, residuals):
            local, unchecked = dict(subst), list(residuals)
            dom, cod = (guidance(side, local) for side in goal)
            branches = cls.branches(dom, cod) if depth else []
            focus = cls.focus(branches, dom, cod)
            if focus and not (epsilon >= 1 or epsilon > 0 and draw(
                    st.floats(0, 1, exclude_max=True)) < epsilon):
                choice = focus[0]
            else:
                leaves = cls.leaves(dom, cod, free)
                candidates = branches if branches\
                    and (not leaves or draw(st.booleans())) else leaves
                if not candidates:
                    raise DeadEnd(f"{dom} -> {cod}")
                choice = draw(st.sampled_from(candidates))
            if choice is None:
                assert free is not None
                result = draw(free(dom=dom, cod=cod, types=types))
            else:
                rule, found = choice
                (rule_subst, rule_residuals), before, after = cls.contexts(
                    draw, rule, found, dom, cod)
                _, args = rule.generate(
                    draw, lambda _, dom, cod: terms(
                        (as_pattern(dom), as_pattern(cod)),
                        depth - 1, local, unchecked),
                    subst=rule_subst, residuals=rule_residuals, types=types)
                result = rule.apply(args)
                result = result if before is None else before.then(result)
                result = result if after is None else result.then(after)
            for side, guide, value in zip(
                    goal, (dom, cod), (result.dom, result.cod)):
                if is_pattern(side):
                    found = list(match(side, value, local, tuple(unchecked)))
                else:
                    found = [(local, tuple(unchecked))]\
                        if value == side else []
                if not found:
                    if guide is None:
                        raise DeadEnd(f"{dom} -> {cod}")
                    raise AxiomError(
                        f"The goal reads {guide} where "
                        f"{choice[0] if choice else free} built "
                        f"{result.dom} -> {result.cod}.")
                chosen = found[0] if len(found) == 1\
                    else draw(st.sampled_from(found))
                local, unchecked = dict(chosen[0]), list(chosen[1])
            subst.clear()
            subst.update(local)
            residuals[:] = unchecked
            return result

        @st.composite
        def terms(draw, goal, depth, subst, residuals, tries: int = 8):
            for _ in range(tries):
                try:
                    return draw(attempt(goal, depth, subst, residuals))
                except DeadEnd:
                    continue
            raise DeadEnd(f"{goal[0]} -> {goal[1]}")

        @st.composite
        def goals(draw, dom, cod, depth):
            subst: dict = {}
            residuals: list = []
            goal = (as_pattern(dom), as_pattern(cod))
            try:
                result = draw(terms(goal, depth, subst, residuals))
            except DeadEnd:
                assume(False)
            for pattern, value in residuals:
                for name in variables(pattern):
                    if name not in subst:
                        subst[name] = draw(Sort().strategy(scope, types))
                assume(instantiate(pattern, subst, scope[OBJECTS]) == value)
            return result

        return goals(dom, cod, max_depth)

    @classmethod
    def leaves(cls, dom, cod, free: Callable | None = None) -> list:
        """
        The ways to close a goal in one step: :obj:`None` for the free box
        when there is one, which fits any goal, and each generator whose
        conclusion unifies with the goal, with its matches.

        Parameters:
            dom : The domain of the goal, :obj:`None` when unknown.
            cod : The codomain of the goal, :obj:`None` when unknown.
            free : The strategy of the free box, if any.
        """
        return ([] if free is None else [None]) + cls.matching(
            cls.generators.values(), dom, cod)

    @classmethod
    def branches(cls, dom, cod) -> list:
        """
        The recursive rules whose conclusion unifies with a goal, each with
        its matches, whose premises the search proves in turn.

        Parameters:
            dom : The domain of the goal, :obj:`None` when unknown.
            cod : The codomain of the goal, :obj:`None` when unknown.
        """
        return cls.matching(
            (rule for rule in cls.rules.values() if rule.recursive), dom, cod)

    @staticmethod
    def matching(rules, dom, cod) -> list:
        """ The rules whose conclusion unifies with a goal, with their
        matches, those with none left out. """
        matches = [(rule, list(rule.match(dom, cod))) for rule in rules]
        return [(rule, found) for rule, found in matches if found]

    @classmethod
    def contexts(cls, draw, rule: Rule, found: list[Match], dom, cod
                 ) -> tuple[Match, Any, Any]:
        """
        How the conclusion of a rule meets the goal: a match, and the
        morphisms to compose before and after the term the rule builds,
        :obj:`None` for nothing. A mere category has no context to put a
        term in, so it draws one of the matches, which unify the
        conclusion with the goal as it is.

        Parameters:
            draw : The draw of the composite strategy.
            rule : The rule applied.
            found : The matches of its conclusion with the goal.
            dom : The domain of the goal, :obj:`None` when unknown.
            cod : The codomain of the goal, :obj:`None` when unknown.
        """
        from hypothesis import strategies as st

        # pylint: disable=unused-argument  # the goal is already matched
        return draw(st.sampled_from(found)), None, None

    @classmethod
    def focus(cls, branches: list, dom, cod) -> list:
        """
        The branches a goal commits to without a choice, none by default:
        a category whose rules are invertible on some goals says which,
        see :meth:`BiclosedCategory.focus`.

        Parameters:
            branches : The recursive rules matching the goal.
            dom : The domain of the goal, :obj:`None` when unknown.
            cod : The codomain of the goal, :obj:`None` when unknown.
        """
        # pylint: disable=unused-argument  # no rule is invertible
        return []

    @classmethod
    @rule
    @abstractmethod
    def id[A: Obj[C0]](cls, dom: Var[C0, A]) -> Hom[C1, A, A]:
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
    @rule
    @abstractmethod
    def dagger[A: Obj[C0], B: Obj[C0]](
            self: Hom[C1, A, B]) -> Hom[C1, B, A]:
        """ The dagger of a morphism, to be instantiated: as a rule, from
        ``a ⊢ b`` conclude ``b ⊢ a``. """

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
            cls, dom: Var[C0 | None, A] = None) -> Hom[C1, A, A]:
        """The monoidal unit, i.e. the empty tensor ``cls()``."""
        return cls()  # ty: ignore[invalid-return-type]

    @classmethod
    def unit[A: Obj[C0]](
            cls, colour: Var[C0 | None, A] = None) -> Var[C0 | C1, Unit[C0]]:
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
            cls, other: Var[C0, A] | Hom[C1, A, B]) -> Hom[C1, A, B]:
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
            cls, other: Var[C0, A] | Hom[C1, A, B]) -> Hom[C1, A, B]:
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
            left: Var[C0, X], right: Var[C0, Y]
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

    @classmethod
    def contexts(cls, draw, rule: Rule, found: list[Match], dom, cod
                 ) -> tuple[Match, Any, Any]:
        """
        A monoidal category puts the term of a recursive rule in context
        wherever its conclusion is a tensor: that side of the goal is
        :meth:`rewire`d first, and a match of the conclusion with the
        rewired goal is a split of it. The planar rewiring is the
        identity, so that the context is a split of the goal as it is.

        Parameters:
            draw : The draw of the composite strategy.
            rule : The rule applied.
            found : The matches of its conclusion with the goal.
            dom : The domain of the goal, :obj:`None` when unknown.
            cod : The codomain of the goal, :obj:`None` when unknown.
        """
        from hypothesis import strategies as st

        if not rule.recursive:
            return super().contexts(draw, rule, found, dom, cod)
        _, dom_pattern, cod_pattern = get_args(rule.conclusion)
        new_dom, before = (dom, None) if dom is None\
            or get_origin(dom_pattern) not in (Tensor, TensorDir)\
            else cls.rewire(draw, dom, True, cod)
        new_cod, after = (cod, None) if cod is None\
            or get_origin(cod_pattern) not in (Tensor, TensorDir)\
            else cls.rewire(draw, cod, False, dom)
        if before is None and after is None:
            return super().contexts(draw, rule, found, dom, cod)
        rewired = list(rule.match(new_dom, new_cod))
        if not rewired:
            raise DeadEnd(f"{new_dom} -> {new_cod}")
        return draw(st.sampled_from(rewired)), before, after

    @classmethod
    def rewire(cls, draw, value: C0, dom: bool, other: C0 | None
               ) -> tuple[C0, Any]:
        """
        The plumbing of a side of a goal, sampled: an object and a morphism
        from the side to it when ``dom``, from it to the side otherwise,
        :obj:`None` for the identity. A planar category has no plumbing
        but the identity, the levels with more structure sample their own:
        a permutation, copies, cups and caps or spiders — each only when it
        is among the :attr:`generators`, so that a category over a fixed
        vocabulary wires with nothing it does not have.

        Parameters:
            draw : The draw of the composite strategy.
            value : The side of the goal to rewire.
            dom : Whether it is the domain.
            other : The other side of the goal, :obj:`None` when unknown.
        """
        # pylint: disable=unused-argument  # a planar category has no wiring
        return value, None

    @classmethod
    def plumb(cls, plumbing, dom: bool, rest: tuple[C0, Any]
              ) -> tuple[C0, Any]:
        """
        Compose a piece of plumbing with the rewiring ``rest`` of its far
        end: after it on the domain side, before it on the codomain side.

        Parameters:
            plumbing : The piece of plumbing, from the side to its far end
                when ``dom``, from its far end to the side otherwise.
            dom : Whether it is the domain side.
            rest : The rewiring of the far end, as :meth:`rewire` returns.
        """
        value, more = rest
        if more is None:
            return value, plumbing
        return value, plumbing.then(more) if dom else more.then(plumbing)

    @classmethod
    def tensor_all(cls, parts: list, unit: C0) -> Self:
        """ The tensor of a list of morphisms, the identity on ``unit``
        when it is empty. """
        result = cls.id(unit)
        for part in parts:
            result = result @ part
        return result  # ty: ignore[invalid-return-type]

    @axiom
    def cut_derivation[
            A: Obj[C0], B: Obj[C0], C: Obj[C0], X: Obj[C0], Y: Obj[C0]](
            cls, f: Hom[C1, B, A], g: Hom[C1, Tensor[X, A, Y], C],
            x: Var[C0, X], y: Var[C0, Y]
    ) -> Equation[Hom[C1, Tensor[X, B, Y], C]]:
        """ The cut is derived from the tensor and composition. """
        return cls.Equation(f.cut(g, x, y), x @ f @ y >> g)

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
            cls, x: Var[C0, X], y: Var[C0, Y]
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
    A traced category is a :class:`MonoidalCategory` with a method
    :code:`trace` for the partial trace of a morphism over some objects on
    either side.
    """
    @rule
    @abstractmethod
    def trace[
            A: Obj[C0], B: Obj[C0], S: bool, N: Count, M: Obj[C0, N]](
            self: Hom[C1, TensorDir[M, A, S], TensorDir[M, B, S]],
            n: Var[int, N] = 1, left: Var[bool, S] = False
    ) -> Hom[C1, A, B]:
        """
        The trace of ``n`` wires on either side, to be instantiated: ``M``
        is of size ``n``, on the left of ``A`` and ``B`` when ``left`` and
        on their right otherwise. Tracing no object at all is the
        identity, i.e. the vanishing axiom ``f.trace(0) == f``, see `nLab
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
    def trace_iteration[
            A: Obj[C0], B: Obj[C0], N: Count, M: Obj[C0, N]](
            cls, f: Hom[C1, Tensor[M, A], Tensor[M, B]], n: Var[int, N]
    ) -> Equation[Hom[C1, A, B]]:
        """ The trace of ``n`` wires is ``n`` traces of one wire. """
        traced = f
        for _ in range(n):
            traced = traced.trace(1, left=True)
        return cls.Equation(f.trace(n, left=True), traced)

    @axiom
    def trace_superposing_left[
            A: Obj[C0], B: Obj[C0], X: Obj[C0], M: Atom[C0]](
            cls, f: Hom[C1, Tensor[M, A], Tensor[M, B]], x: Var[C0, X]
    ) -> Equation[Hom[C1, Tensor[A, X], Tensor[B, X]]]:
        """ Left-oriented superposing. """
        return cls.Equation(
            (f @ x).trace(left=True), f.trace(left=True) @ x)

    @axiom
    def trace_superposing_right[
            A: Obj[C0], B: Obj[C0], X: Obj[C0], M: Atom[C0]](
            cls, f: Hom[C1, Tensor[A, M], Tensor[B, M]], x: Var[C0, X]
    ) -> Equation[Hom[C1, Tensor[X, A], Tensor[X, B]]]:
        """ Right-oriented superposing. """
        return cls.Equation(
            (x @ f).trace(), x @ f.trace())

    @axiom
    def trace_naturality_left[M: Atom[C0], X: Obj[C0], A: Obj[C0], B: Obj[C0]](
            cls, x: Var[C0, Tensor[M, X]],
            f: Hom[C1, Tensor[M, X, B], Tensor[M, X, A]],
            g: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ Left-oriented trace naturality. """
        return cls.Equation(
            (x @ g).then(f).then(x @ g).trace(len(x), left=True),
            g.then(f.trace(len(x), left=True)).then(g))

    @axiom
    def trace_naturality_right[
            M: Atom[C0], X: Obj[C0], A: Obj[C0], B: Obj[C0]](
            cls, x: Var[C0, Tensor[M, X]],
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
            self: Var[C1, X], other: Var[C1, Y]) -> Var[C1, Over[X, Y]]:
        """ The right-to-left exponential object ``self`` to the ``other``. """

    @abstractmethod
    def under[X: Obj[C1], Y: Obj[C1]](
            self: Var[C1, X], other: Var[C1, Y]) -> Var[C1, Under[Y, X]]:
        """ The left-to-right exponential object ``self`` to the ``other``. """

    def __lshift__[X: Obj[C1], Y: Obj[C1]](
            self: Var[C1, X], other: Var[C1, Y]) -> Var[C1, Over[X, Y]]:
        return self.over(other)

    def __rshift__[X: Obj[C1], Y: Obj[C1]](
            self: Var[C1, X], other: Var[C1, Y]) -> Var[C1, Under[X, Y]]:
        return other.under(self)

    @axiom
    def exponential_unit[A: Obj[C0], B: Obj[C0]](
            cls, x: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ An exponential of the unit is its base. """
        return cls.Equation(
            x.over(cls.unit(x.cod)), x, x.under(cls.unit(x.dom)))


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
    def ev[Y: Obj[C0], E: Obj[C0], S: bool](
            cls, base: Var[C0, Y], exponent: Var[C0, E],
            left: Var[bool, S] = True
    ) -> Hom[C1, TensorDir[ExpDir[Y, E, S], E, S], Y]:
        """
        The evaluation of an exponential type on either side, to be
        instantiated: as a rule, ``(y << e) @ e ⊢ y`` on the left and
        ``e @ (e >> y) ⊢ y`` on the right.

        Parameters:
            base : The base of the exponential type.
            exponent : The exponent of the exponential type.
            left : Whether to take the left or right evaluation.
        """

    @rule
    @abstractmethod
    def curry[X: Obj[C0], S: bool, N: Count, Y: Obj[C0, N],
              Z: Obj[C0]](
            self: Hom[C1, TensorDir[X, Y, S], Z], n: Var[int, N] = 1,
            left: Var[bool, S] = True) -> Hom[C1, X, ExpDir[Z, Y, S]]:
        """
        The currying of ``n`` objects on either side, to be instantiated:
        ``Y`` is of size ``n``, curried out of the end of the domain into
        ``Z << Y`` when ``left``, out of its start into ``Y >> Z``
        otherwise.

        Parameters:
            n : The number of objects to curry.
            left : Whether to curry on the left or right.
        """

    def base_and_exponent[X: Obj[C0], S: bool, N: Count,
                          Y: Obj[C0, N], Z: Obj[C0]](
            self: Hom[C1, X, ExpDir[Z, Y, S]], n: Var[int, N],
            left: Var[bool, S]) -> tuple[Var[C0, Z], Var[C0, Y]]:
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

    def uncurry[X: Obj[C0], S: bool, N: Count, Y: Obj[C0, N],
                Z: Obj[C0]](
            self: Hom[C1, X, ExpDir[Z, Y, S]], n: Var[int, N] = 1,
            left: Var[bool, S] = True) -> Hom[C1, TensorDir[X, Y, S], Z]:
        """
        Uncurry a morphism by composing it with :meth:`ev`, assuming its
        codomain is an exponential object, i.e. undo :meth:`curry`, whose
        sequent its own states upside down: ``Y`` is the exponent, of size
        ``n``. If the exponent has less than ``n`` objects, we uncurry the
        remaining ones in turn, which the sequent leaves out. It is a
        method rather than a rule since it is admissible: the evaluation
        and a cut reach every uncurried morphism.

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

    @classmethod
    def uncurry_composition[
            A: Obj[C0], X: Atom[C0], E: Atom[C0], S: bool](
            cls, f: Hom[C1, TensorDir[A, E, S], X], base: Var[C0, X],
            exponent: Var[C0, E], left: Var[bool, S]
    ) -> Hom[C1, TensorDir[A, E, S], X]:
        """
        Curry ``f`` then evaluate it back, i.e. whisker the currying with
        ``exponent`` and compose with the evaluation, the roundtrip that
        :meth:`currying_left` and :meth:`currying_right` state equal to
        ``f``, through the rules :meth:`curry` and :meth:`ev`.

        Parameters:
            f : The morphism to curry and evaluate back.
            base : The base of the exponential, i.e. the codomain of ``f``.
            exponent : The objects curried out of the domain of ``f``.
            left : Whether to curry on the left or right.
        """
        curried, ev = f.curry(1, left), cls.ev(base, exponent, left)
        whiskered = curried @ exponent if left else exponent @ curried
        return whiskered.then(ev)

    @axiom
    def currying_left[A: Obj[C0], X: Atom[C0], E: Atom[C0]](
            cls, f: Hom[C1, Tensor[A, E], X],
            base: Var[C0, X],
            exponent: Var[C0, E]) -> Equation[Hom[C1, Tensor[A, E], X]]:
        """ Left currying followed by evaluation. """
        return cls.Equation(
            cls.uncurry_composition(f, base, exponent, left=True), f)

    @axiom
    def currying_right[A: Obj[C0], X: Atom[C0], E: Atom[C0]](
            cls, f: Hom[C1, Tensor[E, A], X],
            base: Var[C0, X],
            exponent: Var[C0, E]) -> Equation[Hom[C1, Tensor[E, A], X]]:
        """ Right currying followed by evaluation. """
        return cls.Equation(
            cls.uncurry_composition(f, base, exponent, left=False), f)

    @classmethod
    def focus(cls, branches: list, dom, cod) -> list:
        """
        The rules a goal applies deterministically: the conclusion unifies
        with the goal in exactly one way, the match binds every premise
        without residuals, and the premises keep to the subformulae of the
        goal — so committing to one samples nothing and manufactures
        nothing. These are the invertible rules of a focused proof search,
        read off the sequents at each goal: the curry of a biclosed
        category opens the goal's own exponential, while at a rigid level,
        where the exponential collapses into adjoints, the same rule would
        invert an adjoint into material the goal does not have, and stays
        a choice.

        >>> from discopy.biclosed import Diagram, Ty
        >>> x, y = Ty('x'), Ty('y')
        >>> [rule.name for rule, _ in Diagram.focus(
        ...     Diagram.branches(x, y << x), x, y << x)]
        ['curry']
        """
        unit = heads(cls)[OBJECTS]
        goal = {
            atom for side in (dom, cod) if side is not None
            for atom in materials(side)}

        def subformulae(rule, subst):
            for premise in rule.premises.values():
                if isinstance(premise, Sort)\
                        or not set(variables(premise)) <= subst.keys():
                    return False
                value = instantiate(premise, subst, unit)
                if isinstance(value, tuple) and value == (dom, cod):
                    return False  # No progress: the premise is the goal.
                for side in value if isinstance(value, tuple) else (value, ):
                    if hasattr(side, "inside")\
                            and not set(materials(side)) <= goal:
                        return False
            return True

        return [
            (rule, found) for rule, found in branches
            if len(found) == 1 and not found[0][1]
            and subformulae(rule, found[0][0])]

    @axiom
    def currying_eta[Y: Obj[C0], E: Obj[C0], S: bool](
            cls, base: Var[C0, Y], exponent: Var[C0, E],
            left: Var[bool, S]
    ) -> Equation[Hom[C1, ExpDir[Y, E, S], ExpDir[Y, E, S]]]:
        """ The currying of the evaluation is the identity. """
        exp = base << exponent if left else exponent >> base
        return cls.Equation(
            cls.ev(base, exponent, left).curry(len(exponent), left),
            cls.id(exp))

    @axiom
    def currying_naturality[
            A: Obj[C0], U: Obj[C0], E: Obj[C0], Z: Obj[C0], S: bool](
            cls, f: Hom[C1, TensorDir[A, E, S], Z], g: Hom[C1, U, A],
            exponent: Var[C0, E], left: Var[bool, S]
    ) -> Equation[Hom[C1, U, ExpDir[Z, E, S]]]:
        """ Currying is natural in the base of the domain. """
        whiskered = g @ exponent if left else exponent @ g
        n = len(exponent)
        return cls.Equation(
            whiskered.then(f).curry(n, left), g.then(f.curry(n, left)))


class Pregroup[C0, C1: Pregroup](ResiduatedMonoid[C0, C1]):
    """
    A pregroup is a residuated monoid where the left and right exponentials are
    given by tensoring with the chosen left and right duals for each object.
    """
    @property
    @abstractmethod
    def l[X: Obj[C1]](self: Var[C1, X]) -> Var[C1, L[X]]:
        """ The left adjoint, to be instantiated. """

    @property
    @abstractmethod
    def r[X: Obj[C1]](self: Var[C1, X]) -> Var[C1, R[X]]:
        """ The right adjoint, to be instantiated. """

    def over[X: Obj[C1], Y: Obj[C1]](
            self: Var[C1, X], other: Var[C1, Y]) -> Var[C1, Over[X, Y]]:
        return self @ other.l

    def under[X: Obj[C1], Y: Obj[C1]](
            self: Var[C1, X], other: Var[C1, Y]) -> Var[C1, Under[Y, X]]:
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
            cls, left: Var[C0, X], right: Var[C0, R[X]]
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
            cls, left: Var[C0, X], right: Var[C0, L[X]]
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
    def ev[Y: Obj[C0], E: Obj[C0], S: bool](
            cls, base: Var[C0, Y], exponent: Var[C0, E],
            left: Var[bool, S] = True
    ) -> Hom[C1, TensorDir[Y, TensorDir[AdjDir[E, S], E, S], S], Y]:
        """
        The evaluation of a rigid morphism is obtained using cups, as a
        rule ``y @ e.l @ e ⊢ y`` on the left and ``e @ e.r @ y ⊢ y`` on
        the right.

        Parameters:
            base : The base of the exponential type.
            exponent : The exponent of the exponential type.
            left : Whether to take the left or right evaluation.
        """
        if left:
            return base @ cls.cups(exponent.l, exponent)
        return cls.cups(exponent, exponent.r) @ base

    def curry[X: Obj[C0], S: bool, N: Count, Y: Obj[C0, N],
              Z: Obj[C0]](
            self: Hom[C1, TensorDir[X, Y, S], Z], n: Var[int, N] = 1,
            left: Var[bool, S] = True
    ) -> Hom[C1, X, TensorDir[Z, AdjDir[Y, S], S]]:
        """
        The curry of a rigid morphism is obtained using caps, ``Z @ Y.l``
        on the left and ``Y.r @ Z`` on the right. It is a method rather
        than a rule since it is admissible: caps and cut reach every
        transpose, and the self-dual types of a quantum circuit would
        otherwise let it focus on every goal.

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

    def base_and_exponent[X: Obj[C0], S: bool, N: Count,
                          Y: Obj[C0, N], Z: Obj[C0]](
            self: Hom[C1, X, TensorDir[Z, AdjDir[Y, S], S]], n: Var[int, N],
            left: Var[bool, S]) -> tuple[Var[C0, Z], Var[C0, Y]]:
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

    @classmethod
    def focus(cls, branches: list, dom, cod) -> list:
        """
        A rigid category does not focus: its exponentials collapse into
        adjoints, so its curry is a caps composition rather than an
        invertible rule, and the one rule left that a goal could apply
        without a choice is the dagger, which only sends the goal back
        and forth.
        """
        return Category.focus.__func__(cls, branches, dom, cod)

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
            cls, x: Var[C0, X]) -> Equation[Hom[C1, X, X]]:
        """ The two snake equations. """
        snake_r = (cls.id(x) @ cls.caps(x.r, x)).then(
            cls.cups(x, x.r) @ cls.id(x))
        snake_l = (cls.caps(x, x.l) @ cls.id(x)).then(
            cls.id(x) @ cls.cups(x.l, x))
        return cls.Equation(snake_r, cls.id(x), snake_l)

    @axiom
    def caps_coherence[M: Atom[C0], N: Atom[C0], X: Obj[C0], Y: Obj[C0]](
            cls, x: Var[C0, Tensor[M, X]],
            y: Var[C0, Tensor[N, Y]]) -> Equation[Hom[
                C1, Unit[C0], Tensor[M, X, N, Y, L[Y], L[N], L[X], L[M]]]]:
        """ Monoidal coherence of caps. """
        return cls.Equation(
            cls.caps(x @ y, (x @ y).l),
            cls.caps(x, x.l).then(x @ cls.caps(y, y.l) @ x.l))

    @axiom
    def cups_coherence[M: Atom[C0], N: Atom[C0], X: Obj[C0], Y: Obj[C0]](
            cls, x: Var[C0, Tensor[M, X]],
            y: Var[C0, Tensor[N, Y]]) -> Equation[Hom[
                C1, Tensor[M, X, N, Y, R[Y], R[N], R[X], R[M]], Unit[C0]]]:
        """ Monoidal coherence of cups. """
        return cls.Equation(
            cls.cups(x @ y, (x @ y).r),
            (x @ cls.cups(y, y.r) @ x.r).then(cls.cups(x, x.r)))

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
    def self_dual[X: Obj[C0]](cls, x: Var[C0, X]) -> Equation[Var[C0, R[X]]]:
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
            cls, left: Var[C0, X], right: Var[C0, Y]
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
            cls, left: Var[C0, X], right: Var[C0, Y]
    ) -> Hom[C1, Tensor[Y, X], Tensor[X, Y]]:
        """
        The inverse of the braid of two objects, crossing the other way.

        Parameters:
            left : The object on the left of the braid.
            right : The object on the right of the braid.
        """
        return cls.braid(left, right).dagger()

    @axiom
    def braid_then_inverse[X: Atom[C0], Y: Atom[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y]
    ) -> Equation[Hom[C1, Tensor[X, Y], Tensor[X, Y]]]:
        """ The braid followed by its inverse is the identity. """
        return cls.Equation(
            cls.braid(x, y).then(cls.braid_inverse(x, y)), cls.id(x @ y))

    @axiom
    def inverse_then_braid[X: Atom[C0], Y: Atom[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y]
    ) -> Equation[Hom[C1, Tensor[Y, X], Tensor[Y, X]]]:
        """ The inverse of the braid followed by it is the identity. """
        return cls.Equation(
            cls.braid_inverse(x, y).then(cls.braid(x, y)), cls.id(y @ x))

    @axiom
    def hexagon_left[X: Atom[C0], Y: Atom[C0], Z: Atom[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y], z: Var[C0, Z]
    ) -> Equation[Hom[C1, Tensor[X, Y, Z], Tensor[Y, Z, X]]]:
        """ The left hexagon equation. """
        return cls.Equation(
            cls.braid(x, y @ z),
            (cls.braid(x, y) @ z).then(y @ cls.braid(x, z)))

    @axiom
    def hexagon_right[X: Atom[C0], Y: Atom[C0], Z: Atom[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y], z: Var[C0, Z]
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
            cls, left: Var[C0, X], right: Var[C0, Y]
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
            cls, left: Var[C0, X], right: Var[C0, Y]
    ) -> Hom[C1, Tensor[X, Y], Tensor[Y, X]]:
        """ The braid of a symmetric category is its swap. """
        return cls.swap(left, right)

    @classmethod
    def rewire(cls, draw, value: C0, dom: bool, other: C0 | None
               ) -> tuple[C0, Any]:
        """
        A symmetric category permutes the wires of a side of a goal, with
        even odds of leaving them in place, so that the context of a term
        is any subset of the wires rather than a contiguous split.
        """
        from hypothesis import strategies as st

        parts = atoms(value)
        if "swap" not in cls.generators or len(parts) < 2\
                or not draw(st.booleans()):
            return super().rewire(draw, value, dom, other)
        xs = list(draw(st.permutations(range(len(parts)))))
        if xs == sorted(xs):
            return value, None
        if dom:
            plumbing = cls.permutation(xs, parts)
            return plumbing.cod, plumbing
        inverse = [xs.index(i) for i in range(len(xs))]
        plumbing = cls.permutation(inverse, [parts[i] for i in xs])
        return plumbing.dom, plumbing

    @classmethod
    @rule
    def braid_inverse[X: Atom[C0], Y: Atom[C0]](
            cls, left: Var[C0, X], right: Var[C0, Y]
    ) -> Hom[C1, Tensor[Y, X], Tensor[X, Y]]:
        """ The inverse of a swap is the swap the other way. """
        return cls.swap(right, left)

    @axiom
    def swap_inverse[X: Obj[C0], Y: Obj[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y]
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
            cls, x: Var[C0, X], n: Var[int, N]
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
            cls, x: Var[C0, X], n: Var[int, N]
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
            cls, x: Var[C0, X], n: Var[int, N]
    ) -> Equation[Hom[C1, Repeat[X, N], X]]:
        """ Merging is the dagger of copying. """
        return cls.Equation(cls.merge(x, n), cls.copy(x, n).dagger())

    @classmethod
    def rewire(cls, draw, value: C0, dom: bool, other: C0 | None
               ) -> tuple[C0, Any]:
        """
        A Markov category copies or discards each wire of the domain of a
        goal, with even odds of leaving them as they are, then permutes
        them: a term may use an input any number of times. The codomain
        is only permuted, since nothing merges in a Markov category.
        """
        from hypothesis import strategies as st

        if "copy" not in cls.generators or not dom or not value\
                or not draw(st.booleans()):
            return super().rewire(draw, value, dom, other)
        copies = [draw(st.sampled_from((1, 0, 2))) for _ in range(len(value))]
        plumbing = cls.tensor_all([
            cls.id(x) if n == 1 else cls.copy(x, n)
            for x, n in zip(atoms(value), copies)], value[:0])
        return cls.plumb(plumbing, dom, super().rewire(
            draw, plumbing.cod, dom, other))

    @axiom
    def copy_counitality[X: Obj[C0]](
            cls, x: Var[C0, X]) -> Equation[Hom[C1, X, X]]:
        """ Counitality of copying. """
        copy, discard = cls.copy(x, n=2), cls.copy(x, n=0)
        return cls.Equation(
            copy.then(discard @ x), cls.id(x),
            copy.then(x @ discard))

    @axiom
    def copy_coassociativity[X: Obj[C0]](
            cls, x: Var[C0, X]) -> Equation[Hom[C1, X, Tensor[X, X, X]]]:
        """ Coassociativity of copying. """
        copy = cls.copy(x, n=2)
        return cls.Equation(
            copy.then(copy @ x), copy.then(x @ copy))

    @axiom
    def copy_cocommutativity[X: Obj[C0]](
            cls, x: Var[C0, X]) -> Equation[Hom[C1, X, Tensor[X, X]]]:
        """ Cocommutativity of copying. """
        copy = cls.copy(x, n=2)
        return cls.Equation(copy.then(cls.swap(x, x)), copy)

    @axiom
    def discard_coherence[X: Obj[C0]](
            cls, x: Var[C0, X]
    ) -> Equation[Hom[C1, Tensor[X, X], Unit[C0]]]:
        """ Monoidal coherence of discarding. """
        return cls.Equation(
            cls.copy(x @ x, n=0),
            cls.copy(x, n=0) @ cls.copy(x, n=0))

    @axiom
    def copy_monoidal_coherence[X: Obj[C0]](
            cls, x: Var[C0, X]
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
    @axiom
    def currying_symmetry[A: Obj[C0], E: Obj[C0], Z: Obj[C0]](
            cls, f: Hom[C1, Tensor[A, E], Z], exponent: Var[C0, E]
    ) -> Equation[Hom[C1, A, Over[Z, E]]]:
        """ Currying on the left is currying on the right after a swap. """
        base = f.dom[:len(f.dom) - len(exponent)]
        n = len(exponent)
        return cls.Equation(
            f.curry(n, left=True),
            cls.swap(exponent, base).then(f).curry(n, left=False))


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
    def d[X: Obj[C1]](self: Var[C1, X]) -> Var[C1, D[X]]:
        """ Syntactic sugar for :meth:`delay` by one time step. """
        return self.delay()

    @axiom
    def delay_unit[A: Obj[C0], B: Obj[C0]](
            cls, x: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ Delaying by no time step is the identity. """
        return cls.Equation(x.delay(0), x)

    @axiom
    def delay_addition[A: Obj[C0], B: Obj[C0], N: Count](
            cls, x: Hom[C1, A, B], n: Var[int, N]
    ) -> Equation[Hom[C1, A, B]]:
        """ Delaying by one then by ``n`` steps is delaying by ``n + 1``. """
        return cls.Equation(x.delay().delay(n), x.delay(n + 1))

    @axiom
    def delay_tensor[A: Obj[C0], B: Obj[C0], C: Obj[C0]](
            cls, x: Hom[C1, A, B], y: Hom[C1, B, C]
    ) -> Equation[Hom[C1, A, C]]:
        """ The delay of a tensor is the tensor of the delays. """
        return cls.Equation((x @ y).delay(), x.delay() @ y.delay())


class FeedbackCategory[C0: DelayedMonoid, C1: FeedbackCategory](
        MarkovCategory[C0, C1]):
    """
    A feedback category is a :class:`MarkovCategory` whose objects are a
    :class:`DelayedMonoid`, with a :code:`delay` endofunctor and a
    :code:`feedback` operator.
    """
    @rule
    @abstractmethod
    def delay[A: Obj[C0], B: Obj[C0], N: Count](
            self: Hom[C1, A, B], n_steps: Var[int, N] = 1
    ) -> Hom[C1, D[A, N], D[B, N]]:
        """
        The delay endofunctor applied to a morphism, to be instantiated:
        as a rule, from ``a ⊢ b`` conclude ``a.delay(n) ⊢ b.delay(n)``.

        Parameters:
            n_steps : The number of time steps to delay.
        """

    @rule
    @abstractmethod
    def feedback[A: Obj[C0], B: Obj[C0], S: bool, M: Obj[C0]](
            self: Hom[C1, TensorDir[D[M], A, S], TensorDir[M, B, S]],
            dom: Var[C0 | None, A] = None, cod: Var[C0 | None, B] = None,
            mem: Var[C0 | None, M] = None, left: Var[bool, S] = False
    ) -> Hom[C1, A, B]:
        """
        The feedback operator on either side, to be instantiated: the
        memory ``mem`` is any object, delayed in the domain and on the
        left of ``A`` and ``B`` when ``left``, on their right otherwise.
        :meth:`feedback_joining` states that it is fed back one wire at a
        time.

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
    def feedback_joining[X: Obj[C0], M: Obj[C0]](
            cls, f: Hom[C1, Tensor[X, D[M]], Tensor[X, M]], mem: Var[C0, M]
    ) -> Equation[Hom[C1, X, X]]:
        """ Joining nested feedback loops: the feedback of a memory is
        the feedback of its wires, the last one first. """
        joined = f
        for _ in range(len(mem)):
            joined = joined.feedback()
        return cls.Equation(f.feedback(mem=mem), joined)

    @axiom
    def delay_unit[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ Delaying by no time step is the identity. """
        return cls.Equation(f.delay(0), f)

    @axiom
    def delay_composition[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]
    ) -> Equation[Hom[
            C1, D[D[A]], D[D[B]]]]:  # ty: ignore[invalid-type-arguments]
        """ Delaying twice is delaying by two time steps. """
        return cls.Equation(f.delay().delay(), f.delay(2))

    @axiom
    def feedback_tightening[
            A: Obj[C0], B: Obj[C0], U: Obj[C0], V: Obj[C0], M: Obj[C0]](
            cls, f: Hom[C1, Tensor[A, D[M]], Tensor[B, M]],
            g: Hom[C1, U, A], h: Hom[C1, B, V], mem: Var[C0, M]
    ) -> Equation[Hom[C1, U, V]]:
        """ Feedback is natural in its domain and codomain. """
        return cls.Equation(
            (g @ mem.d).then(f).then(h @ mem).feedback(mem=mem),
            g.then(f.feedback(mem=mem)).then(h))

    @axiom
    def feedback_sliding[
            A: Obj[C0], B: Obj[C0], M: Obj[C0], N: Obj[C0]](
            cls, f: Hom[C1, Tensor[A, D[M]], Tensor[B, N]],
            k: Hom[C1, N, M]) -> Equation[Hom[C1, A, B]]:
        """ A morphism slides around a feedback loop, delayed on the way
        back in. """
        base = f.dom[:len(f.dom) - len(k.cod)]
        cobase = f.cod[:len(f.cod) - len(k.dom)]
        return cls.Equation(
            f.then(cobase @ k).feedback(mem=k.cod),
            (base @ k.delay()).then(f).feedback(mem=k.dom))

    @axiom
    def feedback_superposing[A: Obj[C0], B: Obj[C0], X: Obj[C0], M: Obj[C0]](
            cls, f: Hom[C1, Tensor[A, D[M]], Tensor[B, M]],
            x: Var[C0, X], mem: Var[C0, M]
    ) -> Equation[Hom[C1, Tensor[X, A], Tensor[X, B]]]:
        """ Feedback is natural with respect to the tensor. """
        return cls.Equation(
            (x @ f).feedback(mem=mem), x @ f.feedback(mem=mem))

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
    def twist[X: Atom[C0]](cls, dom: Var[C0, X]) -> Hom[C1, X, X]:
        """
        The twist on an object, to be instantiated. As a rule, ``x ⊢ x``
        is a twist.

        Parameters:
            dom : The object on which to take the twist.
        """

    @axiom
    def balanced_twist[X: Atom[C0], Y: Atom[C0]](
            cls, x: Var[C0, X], y: Var[C0, Y]
    ) -> Equation[Hom[C1, Tensor[X, Y], Tensor[X, Y]]]:
        """ Compatibility of the twist and braid. """
        return cls.Equation(
            cls.twist(x @ y),
            cls.braid(x, y).then(
                cls.twist(y) @ cls.twist(x)).then(
                    cls.braid(y, x)))

    @axiom
    def twist_naturality[A: Obj[C0], B: Obj[C0]](
            cls, f: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
        """ Naturality of the twist. """
        return cls.Equation(
            f.then(cls.twist(f.cod)), cls.twist(f.dom).then(f))

    @axiom
    def twist_unit(cls) -> Equation[Hom[C1, Unit[C0], Unit[C0]]]:
        """ The twist of the unit is the identity. """
        return cls.Equation(cls.twist(cls.ob()), cls.id(cls.ob()))

    @axiom
    def yanking[X: Atom[C0]](
            cls, x: Var[C0, X]) -> Equation[Hom[C1, X, X]]:
        """ Yanking, i.e. the twist as both orientations of a traced braid:
        in a symmetric category the trace of a swap is the identity. """
        braid = cls.braid(x, x)
        return cls.Equation(
            braid.trace(left=True), cls.twist(x), braid.trace())


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
    def twist[X: Atom[C0]](cls, dom: Var[C0, X]) -> Hom[C1, X, X]:
        """ The twist of a compact category is the identity. """
        return cls.id(dom)

    twist = classmethod(  # ty: ignore[invalid-assignment]
        rule(twist).admissible("The twist is the identity."))

    @axiom
    def reidemeister_1_cap[X: Obj[C0]](
            cls, x: Var[C0, X]
    ) -> Equation[Hom[C1, Unit[C0], Tensor[R[X], X]]]:
        """ Reidemeister move 1 for caps. """
        return cls.Equation(
            cls.caps(x, x.r).then(cls.swap(x, x.r)),
            cls.caps(x.r, x))

    @axiom
    def reidemeister_1_cup[X: Obj[C0]](
            cls, x: Var[C0, X]
    ) -> Equation[Hom[C1, Tensor[X, R[X]], Unit[C0]]]:
        """ Reidemeister move 1 for cups. """
        return cls.Equation(
            cls.swap(x, x.r).then(cls.cups(x.r, x)),
            cls.cups(x, x.r))

    @classmethod
    def rewire(cls, draw, value: C0, dom: bool, other: C0 | None
               ) -> tuple[C0, Any]:
        """
        A compact category bends wires: on the domain of a goal, a cup
        may close a wire and its adjoint, and a cap may open a wire of the
        other side with its adjoint, then the wires are permuted; the
        codomain is rewired the other way round, a cap opening a pair of
        it and a cup closing a wire of the other side.
        """
        from hypothesis import strategies as st

        closing, opening = ("cups", "caps") if dom else ("caps", "cups")
        parts = atoms(value)
        pairs = [
            (i, j) for i, x in enumerate(parts) for j, y in enumerate(parts)
            if i != j and y == (x.r if dom else x.l)]
        if closing in cls.generators and pairs and draw(st.booleans()):
            i, j = draw(st.sampled_from(pairs))
            xs = [k for k in range(len(parts)) if k not in (i, j)] + [i, j]
            rest = value[:0].tensor(*(parts[k] for k in xs[:-2]))
            if dom:
                plumbing = cls.permutation(xs, parts).then(
                    rest @ cls.cups(parts[i], parts[j]))
                return cls.plumb(plumbing, dom, cls.rewire(
                    draw, rest, dom, other))
            inverse = [xs.index(k) for k in range(len(xs))]
            plumbing = (rest @ cls.caps(parts[i], parts[j])).then(
                cls.permutation(inverse, [parts[k] for k in xs]))
            return cls.plumb(plumbing, dom, cls.rewire(
                draw, rest, dom, other))
        if opening in cls.generators and other and draw(st.booleans()):
            x = draw(st.sampled_from(atoms(other)))
            plumbing = value @ cls.caps(x, x.l) if dom\
                else value @ cls.cups(x, x.r)
            bent = plumbing.cod if dom else plumbing.dom
            return cls.plumb(plumbing, dom, super().rewire(
                draw, bent, dom, None))
        return super().rewire(draw, value, dom, other)


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
            cls, n_legs_in: Var[int, M],
            n_legs_out: Var[int, N],
            typ: Var[C0, X]) -> Hom[C1, Repeat[X, M], Repeat[X, N]]:
        """
        The spiders on a given type with ``n_legs_in`` and ``n_legs_out``:
        as a rule, ``x @ .. @ x ⊢ x @ .. @ x`` is a spider, on up to three
        legs a side drawn.

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
        return cls.Equation(
            cls.spiders(m, 1, x).then(cls.spiders(1, n, x)),
            cls.spiders(m, n, x))

    @axiom
    def spider_specialness[X: Atom[C0]](
            cls, x: Var[C0, X]) -> Equation[Hom[C1, X, X]]:
        """ Splitting a wire then merging it back is the identity. """
        return cls.Equation(
            cls.spiders(1, 2, x).then(cls.spiders(2, 1, x)), cls.id(x))

    @axiom
    def spider_commutativity[X: Atom[C0]](
            cls, x: Var[C0, X]) -> Equation[Hom[C1, Tensor[X, X], X]]:
        """ Spiders do not see the order of their legs. """
        merge = cls.spiders(2, 1, x)
        return cls.Equation(cls.swap(x, x).then(merge), merge)

    @classmethod
    def rewire(cls, draw, value: C0, dom: bool, other: C0 | None
               ) -> tuple[C0, Any]:
        """
        A hypergraph category connects wires by spiders: on the domain of a
        goal, a spider may merge two equal wires or start a wire of the
        other side from nothing, and on the codomain split a wire in two or
        end a wire of the other side, before the cups, caps, copies and
        permutations of the levels it extends.
        """
        from hypothesis import strategies as st

        if "spiders" not in cls.generators:
            return super().rewire(draw, value, dom, other)
        parts = atoms(value)
        pairs = [(i, j) for i, x in enumerate(parts)
                 for j, y in enumerate(parts) if i < j and x == y]
        if pairs and draw(st.booleans()):
            i, j = draw(st.sampled_from(pairs))
            xs = [k for k in range(len(parts)) if k not in (i, j)] + [i, j]
            rest = value[:0].tensor(*(parts[k] for k in xs[:-2]))
            if dom:
                plumbing = cls.permutation(xs, parts).then(
                    rest @ cls.spiders(2, 1, parts[i]))
            else:
                inverse = [xs.index(k) for k in range(len(xs))]
                plumbing = (rest @ cls.spiders(1, 2, parts[i])).then(
                    cls.permutation(inverse, [parts[k] for k in xs]))
            merged = plumbing.cod if dom else plumbing.dom
            return cls.plumb(plumbing, dom, super().rewire(
                draw, merged, dom, other))
        if other and draw(st.booleans()):
            x = draw(st.sampled_from(atoms(other)))
            plumbing = value @ cls.spiders(0, 1, x) if dom\
                else value @ cls.spiders(1, 0, x)
            started = plumbing.cod if dom else plumbing.dom
            return cls.plumb(plumbing, dom, super().rewire(
                draw, started, dom, other))
        return super().rewire(draw, value, dom, other)
