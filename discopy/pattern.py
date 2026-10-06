"""
Sequent patterns and their matching: the language in which a category
states its rules, generators and axioms.

A sequent is the signature of a method on an abstract base class of
:mod:`discopy.abc`: its :pep:`695` type parameter list is the context,
each parameter one variable with its sort as the bound — ``A: Ob[C0]``
an object, ``X: Atom[C0]`` an atomic one, ``N: Count`` a number of
repetitions — its parameters the premises and its return annotation
the conclusion, each a subscript of the ``Ob`` and ``Hom`` aliases of
:mod:`discopy.abc`: ``f: Hom[C1, A, B]`` a morphism between two sides
and ``x: Ob[C0, p]`` a pattern beside its coarse type, the compound
sides built by the formers ``Tensor[A, B]``, ``Unit[C0]``, ``L[A]``,
``R[A]``, ``D[A]``, ``Over[A, B]``, ``Under[A, B]`` and
``Repeat[X, N]``.

.. code-block:: python

    def tensor[A: Ob[C0], B: Ob[C0], C: Ob[C0], D: Ob[C0]](
            self: Hom[C1, A, B], other: Hom[C1, C, D]
    ) -> Hom[C1, Tensor[A, C], Tensor[B, D]]:
        ...

The aliases are :pep:`695` ``type`` statements expanding to the
``Annotated[T, ...]`` a typechecker reads, so the coarse types stay
fully checked — a ``C0`` in a ``C1`` slot is an error — while
:func:`expand` rebuilds the pattern from the subscript when the
sequent is parsed, its args evaluated lazily in a scope where the
sibling parameters and the heads of the declaring class are visible.
The patterns remain plain values that an ``Annotated`` may carry
inline: ``Ob(A)`` lifts one of the declaration's own type parameters,
its sort the bound, and the operators build the compounds —
``Hom(p, q)``, ``p @ q``, ``p.l``, ``p.r``, ``p.d``, ``p << q``,
``p >> q``, ``p ** n`` and :data:`UNIT` — a side of :class:`Hom`
going through :func:`lift`. A premise over the terms of the category
itself states ``Self``, and one over the objects or arrows of a
functor's source or target its :class:`Sort`, ``Sort("In0")`` to
``Sort("Out1")``. Nothing is quoted: each pattern is built when the
annotation or bound is read, lazily by :pep:`649`, and :func:`parse`
collects the sequent without evaluating anything itself. Each pattern
class declares its :meth:`Pattern.level`, the least structure the
objects it stands in must have, and refuses objects bounded below what
its shape needs.

A conclusion is matched against a goal, a pair of an optional domain and
codomain, by unification over the free monoid of objects: a
:class:`Tensor` splits the goal at every position, an :class:`Ob` binds
once, an adjoint ``p.r`` inverts to ``p``. What cannot be inverted, an
exponential or a delay, is a residual equation checked once every
variable is instantiated.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Pattern
    Ob
    Unit
    Tensor
    Adjoint
    Delay
    Exp
    Repeat
    Hom
    Sort
    Atom
    Count
    L
    R
    D
    Over
    Under
    Sequent
    Declaration

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        lift
        alias_of
        expand
        parse
        cell
        declarations
"""

import __future__
import inspect
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import KW_ONLY, dataclass, field, replace
from functools import reduce
from typing import (
    Annotated, ClassVar, Self, TypeAliasType, TypeVar, get_args, get_origin)

from discopy import abc
from discopy.utils import factory_name


type Substitution = dict[str, object]
type Residuals = tuple[tuple[Pattern, object], ...]
type Match = tuple[Substitution, Residuals]


@dataclass(frozen=True)
class Sort:
    """
    The sort of a premise or a variable: the instances of the type its
    ``head`` resolves to in the scope of a bound declaration, atomic
    or not, and the class of :mod:`discopy.abc` bounding them when
    known.

    >>> print(Sort("C0", atomic=True))
    Atom[C0]
    """

    head: str = "C0"
    atomic: bool = False
    bound: type | None = field(default=None, compare=False, repr=False)

    def resolve(self, scope: dict) -> type:
        """ The type the head stands for in the scope. """
        return scope[self.head]

    def strategy(self, scope: dict, types=None):
        """
        Generate an object of the sort, from ``types`` in place of the
        strategy of the objects when given; a :class:`Count` draws a
        small number.
        """
        from hypothesis import strategies as st

        if self.head == "Count":
            return st.integers(min_value=0, max_value=3)
        resolved = self.resolve(scope)
        base = types if types is not None and resolved is scope["C0"]\
            else resolved.strategy()
        return base.filter(lambda value: len(value) == 1)\
            if self.atomic else base

    def __str__(self):
        return f"Atom[{self.head}]" if self.atomic else self.head


def head_of(head: TypeVar | str) -> str:
    """ The name of a head: a type parameter's, e.g. ``C0``, or its own. """
    if isinstance(head, TypeVar):
        return head.__name__
    if isinstance(head, str):
        return head
    raise TypeError(f"Expected a head, got {head!r}.")


class Atom[T]:
    """ The bound ``def cups[X: Atom]`` declares an atomic object; the
    former ``Atom[C0]`` names its head.

    >>> assert Atom["C0"] == Sort("C0", atomic=True)
    """

    def __class_getitem__(cls, head) -> Sort:
        return Sort(head_of(head), atomic=True)


class Count:
    """ The bound ``def spiders[N: Count]`` declares a number of
    repetitions, see :class:`Repeat`. """


def sort_of(bound, level: type | None = None) -> Sort:
    """
    The sort a bound declares: an object when :obj:`None`, an
    :class:`Atom` or a :class:`Count` when the bound says so, a
    :class:`Sort` as it is, ``Ob[C0]`` an object of the named head,
    or a class of :mod:`discopy.abc` bounding the objects; the
    ``level`` of the declaring class is carried by every object sort.
    """
    if bound is None:
        return Sort("C0", bound=level)
    if isinstance(bound, type) and issubclass(bound, Atom):
        return Sort("C0", atomic=True, bound=level)
    if isinstance(bound, type) and issubclass(bound, Count):
        return Sort("Count")
    if isinstance(bound, Sort):
        return replace(bound, bound=bound.bound or level)
    if isinstance(bound, type) and issubclass(bound, abc.Category):
        return Sort(bound=bound)
    if alias_of(bound) is getattr(abc, "Ob", None):
        try:
            (head, ) = get_args(bound)
        except ValueError:
            raise TypeError(
                f"A bound names one head, Ob[C0], got {bound!r}.") from None
        return Sort(head_of(head), bound=level)
    raise TypeError(f"Expected a sort, got {bound!r}.")


@dataclass(frozen=True)
class Pattern(ABC):
    """
    A pattern for the objects of a category, with variables to
    instantiate: matching a value yields every substitution unifying
    the pattern with it, each with the residual equations it could not
    invert. Compounds are built with the operators of the objects they
    stand for: ``@``, ``.l``, ``.r``, ``.d``, ``<<``, ``>>``, ``**``.
    """
    def __post_init__(self):
        """ Check a known bound against the :meth:`level`; a pattern
        of bare type parameters knows none until :func:`parse` reads it
        with the class stating it. """
        bound = self.bound
        if bound is None:
            return
        required = self.level()
        if not issubclass(bound, required):
            raise TypeError(
                f"{self} needs a {required.__name__}, its objects "
                f"are bounded by {bound.__name__}.")

    @classmethod
    def level(cls) -> type[abc.Category]:
        """ The least structure the pattern needs. """
        return abc.Category

    @property
    def parts(self) -> tuple[Pattern, ...]:
        """ The immediate sub-patterns, the fields holding one. """
        values = (
            getattr(self, name)
            for name in getattr(self, "__dataclass_fields__", ()))
        return tuple(
            part for value in values
            for part in (value if isinstance(value, tuple) else (value, ))
            if isinstance(part, Pattern))

    def walk(self) -> Iterator[Pattern]:
        """ The pattern and every sub-pattern below it. """
        yield self
        for part in self.parts:
            yield from part.walk()

    @property
    @abstractmethod
    def variables(self) -> tuple[str, ...]:
        """ The names of the variables of the pattern, in order. """

    @property
    @abstractmethod
    def bound(self) -> type | None:
        """
        The class of :mod:`discopy.abc` bounding the objects the pattern
        stands for, :obj:`None` when none is known.
        """

    @abstractmethod
    def instantiate(self, subst: Substitution, unit: Callable) -> object:
        """
        The object the pattern stands for under a substitution.

        Parameters:
            subst : The values of the variables, by name.
            unit : The object type, called to build the unit.
        """

    def match(self, value, subst: Substitution | None = None,
              residuals: Residuals = ()) -> Iterator[Match]:
        """
        Unify the pattern with a value, or with anything at all when the
        value is :obj:`None`.

        >>> from discopy.monoidal import Ty
        >>> x, y = Ty('x'), Ty('y')
        >>> A, B = Ob("A"), Ob("B")
        >>> for subst, _ in (A @ B).match(x @ y):
        ...     print(subst['A'], '|', subst['B'])
        Ty() | x @ y
        x | y
        x @ y | Ty()
        """
        subst = {} if subst is None else subst
        if value is None:
            yield subst, residuals
        else:
            yield from self.unify(value, subst, residuals)

    def unify(self, value, subst: Substitution,
              residuals: Residuals) -> Iterator[Match]:
        """
        Unify the pattern with a concrete value; by default the equation is
        kept as a residual, checked once the variables are instantiated.
        """
        yield subst, residuals + ((self, value), )

    def __matmul__(self, other: Pattern) -> Tensor:
        return Tensor(self, other)

    @property
    def l(self) -> Adjoint:
        """ The left adjoint of the pattern. """
        return Adjoint(self, "l")

    @property
    def r(self) -> Adjoint:
        """ The right adjoint of the pattern. """
        return Adjoint(self, "r")

    @property
    def d(self) -> Delay:
        """ The delay of the pattern by one time step. """
        return Delay(self)

    def __lshift__(self, other: Pattern) -> Exp:
        return Exp("<<", self, other)

    def __rshift__(self, other: Pattern) -> Exp:
        return Exp(">>", self, other)

    def __pow__(self, count: Ob) -> Repeat:
        return Repeat(self, count)


@dataclass(frozen=True, init=False)
class Ob(Pattern):
    """
    A variable over the objects of a category: the lift ``Ob(A)`` of
    one of the declaration's own type parameters, its sort the bound —
    or a name and a sort directly.

    >>> def cups[X: Atom](): ...
    >>> X, = cups.__type_params__
    >>> print(Ob(X) @ Ob(X).r)
    X @ X.r
    """

    name: str
    sort: Sort

    def __init__(self, var: TypeVar | str, sort: Sort | None = None):
        if not isinstance(var, (TypeVar, str)):
            raise TypeError(
                f"Expected a type parameter or a name, got {var!r}.")
        name = var if isinstance(var, str) else var.__name__
        if sort is None:
            sort = sort_of(getattr(var, "__bound__", None))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "sort", sort)
        super().__post_init__()

    @property
    def variables(self):
        return (self.name, )

    @property
    def bound(self):
        return self.sort.bound

    def instantiate(self, subst, unit):
        return subst[self.name]

    def unify(self, value, subst, residuals):
        if self.name in subst:
            if subst[self.name] == value:
                yield subst, residuals
        elif not self.sort.atomic or len(value) == 1:
            yield dict(subst, **{self.name: value}), residuals

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Unit[T](Pattern):
    """ The unit of a monoid of objects, :data:`UNIT` in an annotation
    and the former ``Unit[C0]`` in a bound. """

    sort: Sort = Sort("C0")

    def __class_getitem__(cls, head) -> Unit:
        return cls(Sort(head_of(head)))

    @classmethod
    def level(cls) -> type[abc.Category]:
        return abc.ColouredMonoid

    variables = ()

    @property
    def bound(self):
        return self.sort.bound

    def instantiate(self, subst, unit):
        return unit()

    def unify(self, value, subst, residuals):
        if not len(value):
            yield subst, residuals

    def __str__(self):
        return f"Unit[{self.sort}]"


UNIT = Unit()
""" The unit of the objects, e.g. the codomain of the cups. """


@dataclass(frozen=True, init=False)
class Tensor[*Ts](Pattern):
    """ The tensor of two or more patterns, flattened: ``p @ q`` and
    the former ``Tensor[p, q]`` in a bound. """

    factors: tuple[Pattern, ...]

    def __class_getitem__(cls, factors) -> Tensor:
        factors = factors if isinstance(factors, tuple) else (factors, )
        return cls(*map(lift, factors))

    def __init__(self, *factors: Pattern):
        if len(factors) < 2:
            raise TypeError(
                f"A tensor takes two or more factors: {factors!r}.")
        object.__setattr__(self, "factors", tuple(
            part for factor in factors
            for part in (
                factor.factors if isinstance(factor, Tensor)
                else (factor, ))))
        super().__post_init__()

    @classmethod
    def level(cls) -> type[abc.Category]:
        return abc.ColouredMonoid

    @property
    def variables(self):
        return tuple(name for f in self.factors for name in f.variables)

    @property
    def bound(self):
        return common(*self.factors)

    def instantiate(self, subst, unit):
        return reduce(operator.matmul, (
            factor.instantiate(subst, unit) for factor in self.factors))

    def unify(self, value, subst, residuals):
        yield from self.split(self.factors, value, subst, residuals)

    @classmethod
    def split(cls, factors, value, subst, residuals) -> Iterator[Match]:
        """ Unify the factors with the prefixes of a value, in turn. """
        if not factors:
            if not len(value):
                yield subst, residuals
            return
        head, *tail = factors
        for n in range(len(value) + 1):
            for subst_, residuals_ in head.unify(value[:n], subst, residuals):
                yield from cls.split(tail, value[n:], subst_, residuals_)

    def __str__(self):
        return " @ ".join(map(str, self.factors))


@dataclass(frozen=True)
class Adjoint(Pattern):
    """ The left or right adjoint ``p.l`` or ``p.r`` of a pattern,
    inverted by the adjoint on the other side when matching. """

    base: Pattern
    side: str

    INVERSE: ClassVar[dict] = {"l": "r", "r": "l"}

    @classmethod
    def level(cls) -> type[abc.Category]:
        return abc.Pregroup

    @property
    def variables(self):
        return self.base.variables

    @property
    def bound(self):
        return self.base.bound

    def instantiate(self, subst, unit):
        return getattr(self.base.instantiate(subst, unit), self.side)

    def unify(self, value, subst, residuals):
        inverse = getattr(value, self.INVERSE[self.side])
        yield from self.base.unify(inverse, subst, residuals)

    def __str__(self):
        base = f"({self.base})"\
            if isinstance(self.base, (Tensor, Repeat)) else str(self.base)
        return f"{base}.{self.side}"


@dataclass(frozen=True)
class Delay(Pattern):
    """ The delay ``p.d`` of a pattern by one time step. """

    base: Pattern

    @classmethod
    def level(cls) -> type[abc.Category]:
        return abc.DelayedMonoid

    @property
    def variables(self):
        return self.base.variables

    @property
    def bound(self):
        return self.base.bound

    def instantiate(self, subst, unit):
        return self.base.instantiate(subst, unit).d

    def __str__(self):
        base = f"({self.base})"\
            if isinstance(self.base, (Tensor, Repeat)) else str(self.base)
        return f"{base}.d"


@dataclass(frozen=True)
class Exp(Pattern):
    """ An exponential ``p << q`` or ``p >> q`` of two patterns,
    decomposed against a single exponential object that its base and
    exponent rebuild, kept as a residual otherwise — so a level that
    collapses its exponentials, e.g. into the adjoints of a pregroup,
    keeps the residual. """

    symbol: str
    left: Pattern
    right: Pattern

    OPERATORS: ClassVar[dict] = {"<<": operator.lshift, ">>": operator.rshift}

    @classmethod
    def level(cls) -> type[abc.Category]:
        return abc.ResiduatedMonoid

    @property
    def variables(self):
        return self.left.variables + self.right.variables

    @property
    def bound(self):
        return common(self.left, self.right)

    def instantiate(self, subst, unit):
        return self.OPERATORS[self.symbol](
            self.left.instantiate(subst, unit),
            self.right.instantiate(subst, unit))

    def unify(self, value, subst, residuals):
        atom = value.inside[0] if len(value) == 1 else None
        base = getattr(atom, "base", None)
        exponent = getattr(atom, "exponent", None)
        operands = (base, exponent) if self.symbol == "<<"\
            else (exponent, base)
        if base is None or exponent is None\
                or value != self.OPERATORS[self.symbol](*operands):
            yield from super().unify(value, subst, residuals)
            return
        for subst_, residuals_ in self.left.unify(
                operands[0], subst, residuals):
            yield from self.right.unify(operands[1], subst_, residuals_)

    def __str__(self):
        return f"({self.left} {self.symbol} {self.right})"


@dataclass(frozen=True)
class Repeat[P, N](Pattern):
    """
    An atomic pattern repeated a variable number of times, ``p ** n``
    and the former ``Repeat[p, n]`` in a bound, for the legs of a
    spider: matching binds the count to the number of atoms and the
    base to the one atom they all equal.
    """

    base: Pattern
    count: Ob

    def __class_getitem__(cls, args) -> Repeat:
        base, count = args
        return cls(lift(base), count if isinstance(count, Ob) else Ob(count))

    def __post_init__(self):
        if not isinstance(self.count, Ob)\
                or self.count.sort.head != "Count":
            raise TypeError(f"Expected a Count variable, got {self.count!r}.")
        if not getattr(getattr(self.base, "sort", None), "atomic", False):
            raise TypeError(
                f"Repeat takes an atomic base, e.g. `X: Atom`, got "
                f"{self.base!r}: matching cannot read a repeated compound "
                "back.")
        super().__post_init__()

    @classmethod
    def level(cls) -> type[abc.Category]:
        return abc.ColouredMonoid

    @property
    def variables(self):
        return self.base.variables + self.count.variables

    @property
    def bound(self):
        return self.base.bound

    def instantiate(self, subst, unit):
        return self.base.instantiate(subst, unit)\
            ** self.count.instantiate(subst, unit)

    def unify(self, value, subst, residuals):
        atoms = [value[i:i + 1] for i in range(len(value))]
        if any(atom != atoms[0] for atom in atoms[1:]):
            return
        for subst_, residuals_ in self.count.unify(
                len(atoms), subst, residuals):
            if atoms:
                yield from self.base.unify(atoms[0], subst_, residuals_)
            else:
                yield subst_, residuals_

    def __str__(self):
        return f"{self.base} ** {self.count}"


def lift(side: Pattern | TypeVar | str | list | tuple) -> Pattern:
    """
    The pattern a side of a :class:`Hom` stands for: a pattern as it
    is, a list or tuple tensored — each element a side itself, none
    the :data:`UNIT` — and anything else lifted by :class:`Ob`.

    >>> def cups[X: Atom](): ...
    >>> X, = cups.__type_params__
    >>> assert lift([X, Ob(X).r]) == Ob(X) @ Ob(X).r
    >>> assert lift(()) == UNIT and lift([X]) == Ob(X)
    """
    if isinstance(side, Pattern):
        return side
    if isinstance(side, (list, tuple)):
        parts = tuple(lift(part) for part in side)
        if not parts:
            return UNIT
        return parts[0] if len(parts) == 1 else Tensor(*parts)
    return Ob(side)


@dataclass(frozen=True, init=False)
class Hom(Pattern):
    """
    The type ``head[dom, cod]`` of the morphisms between two patterns,
    matched against a goal: a pair of an optional domain and codomain.
    A side goes through :func:`lift`: a rule writes ``Hom(A, B)`` on
    its own type parameters and ``Hom([M, A], [M, B])`` for tensors.

    >>> from discopy.monoidal import Ty
    >>> A, B = Ob("A"), Ob("B")
    >>> hom = Hom(A @ B, A)
    >>> print(hom)
    C1[A @ B, A]
    >>> x, y = Ty('x'), Ty('y')
    >>> [(str(s['A']), str(s['B'])) for s, _ in hom.match((x @ y, x))]
    [('x', 'y')]
    >>> list(hom.match((x @ y, y)))
    []
    >>> def then[A, B](): ...
    >>> assert Hom(*then.__type_params__) == Hom(Ob("A"), Ob("B"))
    >>> assert Hom([A, B], A) == hom
    """

    dom: Pattern
    cod: Pattern
    head: str = "C1"

    def __init__(self, dom: Pattern | TypeVar | str | list | tuple,
                 cod: Pattern | TypeVar | str | list | tuple,
                 head: str = "C1"):
        object.__setattr__(self, "dom", lift(dom))
        object.__setattr__(self, "cod", lift(cod))
        object.__setattr__(self, "head", head)
        super().__post_init__()

    @property
    def variables(self):
        return self.dom.variables + self.cod.variables

    @property
    def bound(self):
        return common(self.dom, self.cod)

    def resolve(self, scope: dict) -> type:
        """ The type the head stands for, see :meth:`Sort.resolve`. """
        return Sort(self.head).resolve(scope)

    def instantiate(self, subst, unit) -> tuple:
        """ The domain and codomain under a substitution. """
        return (self.dom.instantiate(subst, unit),
                self.cod.instantiate(subst, unit))

    def unify(self, value, subst, residuals):
        dom, cod = value
        for subst_, residuals_ in self.dom.match(dom, subst, residuals):
            yield from self.cod.match(cod, subst_, residuals_)

    def __str__(self):
        return f"{self.head}[{self.dom}, {self.cod}]"


class L[T]:
    """ The former ``L[p]`` of the left adjoint ``p.l``.

    >>> assert L[Ob("X")] == Ob("X").l
    """

    def __class_getitem__(cls, base) -> Adjoint:
        return lift(base).l


class R[T]:
    """ The former ``R[p]`` of the right adjoint ``p.r``. """

    def __class_getitem__(cls, base) -> Adjoint:
        return lift(base).r


class D[T]:
    """ The former ``D[p]`` of the delay ``p.d``. """

    def __class_getitem__(cls, base) -> Delay:
        return lift(base).d


class Over[A, B]:
    """ The former ``Over[p, q]`` of the exponential ``p << q``. """

    def __class_getitem__(cls, args) -> Exp:
        left, right = args
        return lift(left) << lift(right)


class Under[A, B]:
    """ The former ``Under[p, q]`` of the exponential ``p >> q``. """

    def __class_getitem__(cls, args) -> Exp:
        left, right = args
        return lift(left) >> lift(right)


def common(*patterns: Pattern) -> type | None:
    """ The bound of the objects of patterns standing together, if all do. """
    bounds = [pattern.bound for pattern in patterns]
    return None if None in bounds else bounds[0]

@dataclass(frozen=True)
class Sequent:
    """
    Variables and their sorts, named premises — a :class:`Hom` or a
    :class:`Sort` to generate, a :class:`Pattern` to instantiate — and
    an optional conclusion.

    >>> from discopy.abc import MonoidalCategory
    >>> print(MonoidalCategory.tensor.sequent)
    ... # doctest: +NORMALIZE_WHITESPACE
    A: C0, B: C0, C: C0, D: C0
    | self: C1[A, B], other: C1[C, D] ⊢ C1[A @ C, B @ D]
    """

    variables: dict[str, Sort] = field(default_factory=dict)
    premises: dict[str, Pattern | Sort] = field(default_factory=dict)
    conclusion: Hom | None = None

    def __str__(self):
        context = ", ".join(f"{n}: {s}" for n, s in self.variables.items())
        premises = ", ".join(f"{n}: {p}" for n, p in self.premises.items())
        left = " | ".join(part for part in (context, premises) if part)
        right = "" if self.conclusion is None else f" ⊢ {self.conclusion}"
        return left + right

def alias_of(annotation) -> TypeAliasType | None:
    """ The alias of :mod:`discopy.abc` an annotation subscripts,
    ``Ob[T, X]`` or ``Hom[C1, dom, cod]``, :obj:`None` otherwise. """
    origin = get_origin(annotation)
    if isinstance(origin, TypeAliasType) and origin in (
            getattr(abc, "Ob", None), getattr(abc, "Hom", None)):
        return origin
    return None


def expand(annotation) -> Pattern | None:
    """
    The pattern a subscripted alias of :mod:`discopy.abc` states:
    ``Hom[C1, dom, cod]`` the hom between its lifted sides, ``Ob[T,
    p]`` the pattern beside the coarse type, :obj:`None` for anything
    else — so a premise or a conclusion is one subscript, the
    ``Annotated`` inside the alias.
    """
    alias = alias_of(annotation)
    if alias is None:
        return None
    args = get_args(annotation)
    if alias is abc.Hom:
        try:
            head, dom, cod = args
        except ValueError:
            raise TypeError(
                f"Hom[head, dom, cod] takes three arguments, got "
                f"{annotation!r}.") from None
        return Hom(dom, cod, head=head_of(head))
    if len(args) != 2:
        raise TypeError(
            f"Ob[T, p] states one pattern beside the coarse type, "
            f"got {annotation!r}.")
    return lift(args[1])


def states_pattern(annotation) -> bool:
    """ Whether an annotation states a pattern or a sort. """
    return annotation is Self or get_origin(annotation) is Annotated\
        or alias_of(annotation) is not None


def premises_of(function: Callable, missing: bool = False) -> list[str]:
    """
    The names of the premises a function states: its parameters whose
    annotation states a pattern, an unannotated first ``cls`` or
    ``self`` skipped and an annotated ``*args`` standing for one
    argument at a time. With ``missing``, the parameters that state no
    pattern and have no default, which a call by the sequent could not
    fill.
    """
    parameters = list(inspect.signature(function).parameters.values())
    if parameters and parameters[0].annotation is inspect.Parameter.empty:
        parameters = parameters[1:]
    if missing:
        return [
            parameter.name for parameter in parameters
            if not states_pattern(parameter.annotation)
            and parameter.default is inspect.Parameter.empty
            and parameter.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD)]
    return [
        parameter.name for parameter in parameters
        if states_pattern(parameter.annotation)
        and parameter.kind is not inspect.Parameter.VAR_KEYWORD]


def parse(function: Callable, owner: type | None = None,
          conclusion: bool = True) -> Sequent:
    """
    The sequent a function states with its signature, collected from
    the pattern objects its annotations built when they were read:
    each parameter stating a pattern a premise, the return annotation
    the conclusion when asked for, the variables the :class:`Ob` s the
    patterns lift, carrying the bound of the objects of the ``owner``
    class stating it. A pattern needing more structure than the
    owner's objects have is refused. An annotation subscripting an
    alias of :mod:`discopy.abc` is the pattern :func:`expand` gives.

    >>> def then[A, B, C](
    ...         self: Annotated["object", Hom(Ob(A), Ob(B))],
    ...         other: Annotated["object", Hom(Ob(B), Ob(C))]
    ... ) -> Annotated["object", Hom(Ob(A), Ob(C))]:
    ...     ...
    >>> print(parse(then))
    A: C0, B: C0, C: C0 | self: C1[A, B], other: C1[B, C] ⊢ C1[A, C]
    >>> print(parse(then, conclusion=False))
    A: C0, B: C0, C: C0 | self: C1[A, B], other: C1[B, C]
    >>> from discopy import abc
    >>> def bracketed[A: abc.Ob["C0"], B: abc.Ob["C0"], C: abc.Ob["C0"]](
    ...         self: abc.Hom["C1", A, B], other: abc.Hom["C1", B, C]
    ... ) -> abc.Hom["C1", A, C]:
    ...     ...
    >>> assert str(parse(bracketed)) == str(parse(then))
    """
    function = inspect.unwrap(function)
    if function.__code__.co_flags & __future__.annotations.compiler_flag:
        raise TypeError(
            f"{function.__module__} defers its annotations with `from "
            f"__future__ import annotations`, so {function.__name__} "
            "states strings where patterns are read.")
    level = None
    if owner is not None:
        if getattr(owner, "__type_params__", ()):
            try:
                bound = owner.__type_params__[0].__bound__
            except NameError:  # A bound imported for typechecking only.
                bound = None
            level = bound if isinstance(bound, type) and issubclass(
                bound, abc.ColouredMonoid) else None
            # A monoid of objects is a level where the bound of a value
            # parameter, e.g. the category a CMap hosts, is not.
        if level is None:
            level = getattr(owner, "ob", None)
            level = level if isinstance(level, type) else None
    signature = inspect.signature(function)
    for name in premises_of(function, missing=True):
        raise TypeError(
            f"{function.__name__} states no pattern for {name}.")

    def stated(annotation) -> Pattern | Sort:
        if annotation is Self:
            return Sort("Self")
        value = expand(annotation)
        if value is None:
            try:
                (value, ) = annotation.__metadata__
            except (AttributeError, ValueError):
                raise TypeError(
                    "An annotation carries exactly one pattern, got "
                    f"{annotation!r}.") from None
        if not isinstance(value, (Pattern, Sort)):
            raise TypeError(f"Expected a pattern or a sort, got {value!r}.")
        if isinstance(value, Pattern) and level is not None:
            for node in value.walk():
                if isinstance(node, (Ob, Hom)):
                    continue
                required = type(node).level()
                if not issubclass(level, required):
                    raise TypeError(
                        f"{node} needs a {required.__name__}, the objects "
                        f"of {function.__name__} are bounded by "
                        f"{level.__name__}.")
        return value

    premises = {
        name: stated(signature.parameters[name].annotation)
        for name in premises_of(function)}
    returns = stated(signature.return_annotation) if conclusion\
        and signature.return_annotation is not inspect.Signature.empty\
        else None
    if conclusion and not isinstance(returns, Hom):
        raise TypeError(f"{function.__name__} concludes no hom type.")
    found = {
        node.name: replace(node.sort, bound=node.sort.bound or level)
        for value in (*premises.values(), returns)
        if isinstance(value, Pattern)
        for node in value.walk() if isinstance(node, Ob)}
    order = [
        parameter.__name__
        for parameter in getattr(function, "__type_params__", ())
        if parameter.__name__ in found]
    variables = {name: found[name] for name in order}
    variables.update(found)
    if isinstance(returns, Hom):
        named = {
            name for premise in premises.values()
            for name in getattr(premise, "variables", ())}
        missing = set(returns.variables) - named
        if missing:
            raise TypeError(
                f"{function.__name__} concludes on "
                f"{', '.join(sorted(missing))} that no premise states.")
    return Sequent(
        variables, premises, returns if isinstance(returns, Hom) else None)


@dataclass(repr=False)
class Declaration[**P, T]:
    """
    A sequent stated by a ``function`` on an abstract base class and
    inherited by every category below it: the base of the rules and
    generators of :mod:`discopy.search` and of the axioms of
    :mod:`discopy.axioms`. The ``category`` is the class the
    declaration is bound to, :obj:`None` until :meth:`bind` or the
    attribute access on a class binds it, ``name`` the attribute it is
    stored under and ``owner`` the class declaring the sequent, whose
    objects bound the sorts of its variables.

    >>> from discopy.abc import Category
    >>> Category.then
    abc.Category.then
    >>> print(Category.then.sequent)
    A: C0, B: C0, C: C0 | self: C1[A, B], other: C1[B, C] ⊢ C1[A, C]
    """

    function: Callable
    _: KW_ONLY
    category: type[T] | None = None
    name: str | None = None
    owner: type | None = None

    concludes: ClassVar[bool] = True
    """ Whether the return annotation is read as the conclusion. """

    def __post_init__(self):
        if isinstance(self.function, classmethod):
            raise TypeError(
                f"Decorate {self.function.__func__.__name__} inside "
                "classmethod, not outside.")
        self.name = self.name or self.function.__name__
        self.__doc__ = self.function.__doc__

    @property
    def sequent(self) -> Sequent:
        """ The parsed signature, cached — and read lazily, so the
        sorts carry the bounds of a class that does not exist when its
        body is decorated. """
        if "sequent" not in self.__dict__:
            self.__dict__["sequent"] = parse(
                self.function, self.owner or self.category,
                conclusion=self.concludes)
        return self.__dict__["sequent"]

    def __set_name__(self, owner: type, name: str):
        if self.category is None:
            self.name = name

    @property
    def __isabstractmethod__(self):
        return getattr(self.function, "__isabstractmethod__", False)

    @property
    def __signature__(self):
        """ The signature of the function, so that a declaration
        introspects as the method it decorates. """
        return inspect.signature(self.function)

    def __repr__(self):
        if self.category is None:
            return f"{type(self).__name__}({self.name})"
        return f"{factory_name(self.category)}.{self.name}"

    def __str__(self):
        return f"{self.name}: {self.sequent}"

    def __hash__(self):
        return hash((self.function, self.category, self.name))

    def bind(self, category: type,
             owner: type | None = None) -> Self:
        """ Bind the declaration to a concrete category. """
        return replace(
            self, category=category, owner=self.owner or owner)

    @property
    def scope(self) -> dict:
        """
        What the heads stand for: the category for ``Self``, its
        objects and arrows for ``C0`` and ``C1`` (a monoid stands for
        both), and those of a functor class's ``dom`` and ``cod`` as
        ``In0``, ``In1``, ``Out0`` and ``Out1``.
        """
        if self.category is None:
            raise TypeError(f"{self.name} is not bound to a class.")
        scope = {
            "Self": self.category,
            "C0": getattr(self.category, "ob", self.category),
            "C1": getattr(self.category, "ar", self.category)}
        dom, cod = (getattr(self.category, name, None)
                    for name in ("dom", "cod"))
        if isinstance(dom, type) and isinstance(cod, type):
            scope.update({
                "In0": getattr(dom, "ob", dom), "In1": dom,
                "Out0": getattr(cod, "ob", cod), "Out1": cod})
        return scope

    @property
    def unit(self) -> Callable:
        """ The object type of the category, called to build the unit. """
        return self.scope["C0"]

    def canonical(self) -> dict:
        """
        The canonical arguments of the sequent, by name: each variable an
        object named after it — a count the number two — each premise a
        :func:`cell` named after its parameter, so that a declaration
        reads as a schema.

        >>> from discopy.abc import MonoidalCategory
        >>> from discopy.monoidal import Diagram
        >>> tensor = MonoidalCategory.tensor.bind(Diagram)
        >>> for name, box in tensor.canonical().items():
        ...     print(f"{name}: {box.dom} -> {box.cod}")
        self: A -> B
        other: C -> D
        """
        subst = {
            name: 2 if sort.head == "Count"
            else cell(sort.resolve(self.scope), name)
            for name, sort in self.sequent.variables.items()}
        args = {}
        for name, premise in self.sequent.premises.items():
            if isinstance(premise, Hom):
                dom, cod = premise.instantiate(subst, self.unit)
                args[name] = cell(premise.resolve(self.scope), name, dom, cod)
            elif isinstance(premise, Sort):
                args[name] = cell(premise.resolve(self.scope), name)
            else:
                args[name] = premise.instantiate(subst, self.unit)
        return args

    def generate(self, draw: Callable, hom: Callable, subst=None,
                 residuals: Residuals = (), types=None) -> tuple:
        """
        Draw the arguments of the sequent inside a composite strategy —
        ``draw`` its draw function, ``hom(category, dom, cod)`` a strategy
        for the morphisms of that type, ``types`` one for the objects
        overriding that of ``C0`` — one premise at a time: a pattern is
        instantiated, a sort drawn, a hom drawn through ``hom``, a
        premise of the sort of the arrows being the hom with both sides
        free. A variable is drawn from its sort the first time a premise
        needs it, except one standing alone on a side of a hom, which is
        read off the term the search finds so that the goal guides the
        search. The ``subst`` and ``residuals`` of a match seed the draw,
        and the residuals are checked once every variable is bound,
        rejecting the example otherwise.
        """
        from hypothesis import assume

        subst = dict(subst or {})
        sorts = self.sequent.variables

        def side(pattern):
            if isinstance(pattern, Ob) and pattern.name not in subst:
                return None
            bound(*pattern.variables)
            return pattern.instantiate(subst, self.unit)

        def read_off(pattern, value):
            if isinstance(pattern, Ob) and pattern.name not in subst:
                assume(not pattern.sort.atomic or len(value) == 1)
                subst[pattern.name] = value

        def bound(*names):
            for name in names:
                if name not in subst:
                    subst[name] = draw(
                        sorts[name].strategy(self.scope, types), label=name)

        args = {}
        for name, premise in self.sequent.premises.items():
            if isinstance(premise, Hom):
                dom, cod = side(premise.dom), side(premise.cod)
                category = premise.resolve(self.scope)
                term = draw(hom(category, dom, cod), label=name)
                read_off(premise.dom, term.dom)
                read_off(premise.cod, term.cod)
                args[name] = term
            elif isinstance(premise, Sort)\
                    and premise.resolve(self.scope) is self.scope["C1"]\
                    and issubclass(self.scope["C1"], abc.Category):
                arrows = hom(self.scope["C1"], None, None)
                args[name] = draw(arrows, label=name)
            elif isinstance(premise, Sort):
                strategy = premise.strategy(self.scope, types)
                args[name] = draw(strategy, label=name)
            else:
                bound(*premise.variables)
                args[name] = premise.instantiate(subst, self.unit)
        for pattern, value in residuals:
            bound(*pattern.variables)
            assume(pattern.instantiate(subst, self.unit) == value)
        return subst, args


def cell(factory: type, name: str, dom=None, cod=None):
    """
    A cell of a class named after a parameter or a variable: a box of
    a class with a ``Box``, between ``dom`` and ``cod`` or objects
    named ``x`` and ``y``, else an instance of the class of that name.

    >>> from discopy.monoidal import Box, Diagram, Ty
    >>> assert cell(Diagram, 'f') == Box('f', Ty('x'), Ty('y'))
    >>> assert cell(Ty, 'A') == Ty('A')
    """
    box = getattr(factory, "Box", None)
    if box is not None and isinstance(box, type):
        dom = factory.ob("x") if dom is None else dom
        cod = factory.ob("y") if cod is None else cod
        return box(name, dom, cod)
    return factory(name)


def declarations[D: Declaration](cls: type, kind: type[D]) -> dict[str, D]:
    """
    The declarations of exactly a kind inherited by a class, bound to
    it and keyed by name, the latest in the method resolution order
    winning like ordinary attribute lookup, found under any inner
    decorator; a declaration marked inapplicable, or anything that is
    not a declaration, assigned over an inherited one drops it.

    >>> from discopy.monoidal import Diagram
    >>> from discopy.search import Rule
    >>> list(declarations(Diagram, Rule))
    ['id', 'tensor', 'cut']
    """
    result: dict[str, D] = {}
    for base in reversed(cls.__mro__):
        for name, value in base.__dict__.items():
            while isinstance(value, (classmethod, staticmethod)):
                value = value.__func__
            if type(value) is kind\
                    and getattr(value, "__inapplicable__", None) is None:
                result[name] = value.bind(cls, owner=base)
            else:
                result.pop(name, None)
    return result
