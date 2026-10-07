"""
The patterns in which a category states its rules, generators and
axioms, which :meth:`discopy.abc.Category.search` searches its terms by.

A sequent is the signature of a method on an abstract base class of
:mod:`discopy.abc`: its :pep:`695` type parameter list is the context,
each parameter one variable with its sort as the bound — ``A: Obj[C0]``
an object, ``X: Atom[C0]`` a single wire, ``M: Obj[C0, N]`` an object
of size ``N``, ``N: Count`` a number and ``S: bool`` a side — its
parameters the premises and its return annotation the conclusion.
A premise is a sort to sample, e.g. ``f: Hom[C1, A, B]`` a morphism
between two patterns, or ``Var[T, p]`` the value an already bound
pattern ``p`` stands for, e.g. ``x: Var[C0, X]`` or ``n: Var[int, N]``.

.. code-block:: python

    def tensor[A: Obj[C0], B: Obj[C0], C: Obj[C0], D: Obj[C0]](
            self: Hom[C1, A, B], other: Hom[C1, C, D]
    ) -> Hom[C1, Tensor[A, C], Tensor[B, D]]:
        ...

A :class:`Declaration` reads its sequent off :func:`inspect.signature`,
i.e. the annotations as :pep:`649` evaluates them, the same objects a
typechecker sees: ``Hom[C1, A, B]`` is a generic alias with ``Hom`` as
``__origin__`` and the type parameters ``C1, A, B`` as ``__args__``, a
pattern ``Tensor[A, C]`` a generic alias of its :class:`Pattern`, which
interprets the arguments of its own aliases, and a variable the
:class:`typing.TypeVar` of the declaration. The type parameters of a
pattern are bounded by the structure the objects it stands for need,
e.g. ``D[T: DelayedMonoid]``, so that a typechecker refuses a delay in
a sequent whose objects are not delayed.

A conclusion is matched against a goal, a pair of an optional domain and
codomain, by unification over the free monoid of objects: a ``Tensor``
splits the goal at every position, a variable binds once, an adjoint
``R[p]`` inverts to ``p``. What cannot be inverted, an exponential that
collapsed into adjoints, is a residual equation checked once every
variable is instantiated. A :class:`Rule` is a declaration with a
conclusion, and :meth:`discopy.abc.Category.search` builds a term of a
goal type by choosing at each step a free box, a rule with no hom
premise whose conclusion matches the goal — a generator, built in one
step — or, below the depth bound, a :meth:`Rule.recursive` one whose hom
premises are searched. The unification here is planar: a category with
more structure puts a term in context by its own plumbing, see
:meth:`discopy.abc.MonoidalCategory.rewire`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Pattern
    Tensor
    TensorDir
    ExpDir
    Over
    Under
    AdjDir
    L
    R
    D
    Repeat
    Image
    Count
    Sort
    Declaration
    Rule
    Constant

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        heads
        variables
        instantiate
        match
        unify
        cell
        declarations
        rule
"""

import inspect
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import KW_ONLY, dataclass, replace
from functools import reduce
from types import MethodType
from typing import (
    Annotated, Any, ClassVar, Literal, Self, TYPE_CHECKING, TypeVar,
    get_args, get_origin)

from discopy.utils import factory_name

if TYPE_CHECKING:
    pass


type Obj[Coarse, Size = None] = Annotated[Coarse, Size]
""" The sort ``Obj[T]`` of the objects of type ``T``, as the bound of a
variable ``A: Obj[C0]`` or as a premise to sample, of a given size when
one is given: ``Literal[n]`` or a variable ``N: Count``. A typechecker
reads it as ``T``. """

type Atom[Coarse] = Obj[Coarse, Literal[1]]
""" The sort of the objects of size one, i.e. a single wire. """

type Unit[Coarse] = Obj[Coarse, Literal[0]]
""" The sort of the objects of size zero, i.e. the unit. """

type Var[Coarse, Fine] = Annotated[Coarse, Fine]
""" The premise or conclusion ``Var[T, p]`` standing for the value of a
pattern ``p`` over variables bound elsewhere, e.g. ``x: Var[C0, X]``
for a variable ``X: Obj[C0]`` or ``n: Var[int, N]`` for ``N: Count``. A
typechecker reads it as ``T``. """

type Hom[Coarse, Dom, Cod] = Annotated[Coarse, Dom, Cod]
""" The sort ``Hom[C1, dom, cod]`` of the morphisms between two
patterns, as a premise to sample or a conclusion to match. A
typechecker reads it as ``C1``. """

type Substitution = dict[str, object]
type Residuals = tuple[tuple[object, object], ...]
type Match = tuple[Substitution, Residuals]

#: The heads of the sequents of a category.
OBJECTS, ARROWS, SELF = "C0", "C1", "Self"


class Count:
    """ The bound ``N: Count`` of a number, e.g. of repetitions. """


class Pattern(ABC):
    """
    A pattern for the objects of a category, with variables to
    instantiate. A pattern class is never instantiated: a pattern is the
    alias Python builds by subscripting it, e.g. ``Tensor[A, B]``, and
    the class interprets the arguments of its own aliases.
    """

    @classmethod
    def variables(cls, args: tuple) -> tuple[str, ...]:
        """ The names of the variables of the arguments, in order. """
        return tuple(label for arg in args for label in variables(arg))

    @classmethod
    @abstractmethod
    def instantiate(cls, args: tuple, subst: Substitution, unit):
        """ The object the pattern stands for under a substitution,
        ``unit`` the object type called to build the unit. """

    @classmethod
    @abstractmethod
    def unify(cls, args: tuple, value, subst: Substitution,
              residuals: Residuals) -> Iterator[Match]:
        """ Unify the pattern with a value, yielding every substitution
        with the residual equations it could not invert. """


class Tensor[*Ts](Pattern):
    """ The tensor ``Tensor[A, B, ...]`` of two or more patterns,
    matched by splitting the value at every position. """

    @classmethod
    def instantiate(cls, args, subst, unit):
        return reduce(operator.matmul, (
            instantiate(arg, subst, unit) for arg in args))

    @classmethod
    def unify(cls, args, value, subst, residuals):
        if not args:
            if not len(value):
                yield subst, residuals
            return
        head, *tail = args
        for n in range(len(value) + 1):
            for subst_, residuals_ in unify(
                    head, value[:n], subst, residuals):
                yield from cls.unify(tail, value[n:], subst_, residuals_)


class TensorDir[P, Q, S: bool](Pattern):
    """ The tensor ``P @ Q`` when the side ``S`` is true, ``Q @ P``
    otherwise: the ``left`` of a method puts ``P`` first. """

    @classmethod
    def instantiate(cls, args, subst, unit):
        first, second, left = (instantiate(arg, subst, unit) for arg in args)
        return first @ second if left else second @ first

    @classmethod
    def unify(cls, args, value, subst, residuals):
        first, second, side = args
        for subst_, left in sides(side, subst):
            yield from Tensor.unify(
                (first, second) if left else (second, first),
                value, subst_, residuals)


class ExpDir[Z: abc.ResiduatedMonoid, Y: abc.ResiduatedMonoid, S: bool](
        Pattern):
    """ The exponential ``Z << Y`` when the side ``S`` is true, ``Y >> Z``
    otherwise, decomposed against a single exponential object that its
    base and exponent rebuild, matching nothing else — except at a level
    that collapses its exponentials into the adjoints of a pregroup,
    which keeps the residual. """

    @classmethod
    def instantiate(cls, args, subst, unit):
        base, exponent, left = (instantiate(arg, subst, unit) for arg in args)
        return base << exponent if left else exponent >> base

    @classmethod
    def unify(cls, args, value, subst, residuals):
        pattern_base, pattern_exponent, side = args
        atom = value.inside[0] if len(value) == 1 else None
        base: Any = getattr(atom, "base", None)
        exponent: Any = getattr(atom, "exponent", None)
        for subst_, left in sides(side, subst):
            if base is None or exponent is None or value != (
                    base << exponent if left else exponent >> base):
                if hasattr(value, "r"):  # A pregroup: the residual stays.
                    oriented = ExpDir[
                        pattern_base, pattern_exponent, Literal[True]]\
                        if left else ExpDir[
                            pattern_base, pattern_exponent, Literal[False]]
                    yield subst_, residuals + ((oriented, value), )
                continue
            for subst__, residuals_ in unify(
                    pattern_base, base, subst_, residuals):
                yield from unify(
                    pattern_exponent, exponent, subst__, residuals_)


class Over[Z: abc.ResiduatedMonoid, Y: abc.ResiduatedMonoid](
        ExpDir[Z, Y, Literal[True]]):
    """ The exponential ``Over[Z, Y]``, i.e. ``Z << Y``. """

    @classmethod
    def instantiate(cls, args, subst, unit):
        return ExpDir.instantiate((*args, Literal[True]), subst, unit)

    @classmethod
    def unify(cls, args, value, subst, residuals):
        yield from ExpDir.unify((*args, Literal[True]), value, subst,
                                residuals)


class Under[Y: abc.ResiduatedMonoid, Z: abc.ResiduatedMonoid](
        ExpDir[Z, Y, Literal[False]]):
    """ The exponential ``Under[Y, Z]``, i.e. ``Y >> Z``. """

    @classmethod
    def instantiate(cls, args, subst, unit):
        exponent, base = args
        return ExpDir.instantiate(
            (base, exponent, Literal[False]), subst, unit)

    @classmethod
    def unify(cls, args, value, subst, residuals):
        exponent, base = args
        yield from ExpDir.unify(
            (base, exponent, Literal[False]), value, subst, residuals)


class AdjDir[T: abc.Pregroup, S: bool](Pattern):
    """ The left adjoint ``T.l`` when the side ``S`` is true, the right
    adjoint ``T.r`` otherwise, inverted by the adjoint on the other side
    when matching. """

    @classmethod
    def instantiate(cls, args, subst, unit):
        base, left = (instantiate(arg, subst, unit) for arg in args)
        return base.l if left else base.r

    @classmethod
    def unify(cls, args, value, subst, residuals):
        base, side = args
        for subst_, left in sides(side, subst):
            yield from unify(
                base, value.r if left else value.l, subst_, residuals)


class L[T: abc.Pregroup](AdjDir[T, Literal[True]]):
    """ The left adjoint ``L[T]``, i.e. ``T.l``. """

    @classmethod
    def instantiate(cls, args, subst, unit):
        return AdjDir.instantiate((*args, Literal[True]), subst, unit)

    @classmethod
    def unify(cls, args, value, subst, residuals):
        yield from AdjDir.unify((*args, Literal[True]), value, subst,
                                residuals)


class R[T: abc.Pregroup](AdjDir[T, Literal[False]]):
    """ The right adjoint ``R[T]``, i.e. ``T.r``. """

    @classmethod
    def instantiate(cls, args, subst, unit):
        return AdjDir.instantiate((*args, Literal[False]), subst, unit)

    @classmethod
    def unify(cls, args, value, subst, residuals):
        yield from AdjDir.unify((*args, Literal[False]), value, subst,
                                residuals)


class D[T: abc.DelayedMonoid, N = Literal[1]](Pattern):
    """ The delay ``D[T, N]`` of a pattern by ``N`` time steps, one by
    default, inverted by the delay ``-N`` steps back when matching: an
    unbound ``N`` binds to every number of steps up to
    :data:`MAX_COUNT` that the value is the delay of. """

    @classmethod
    def instantiate(cls, args, subst, unit):
        base, steps = args
        return instantiate(base, subst, unit).delay(
            instantiate(steps, subst, unit))

    @classmethod
    def unify(cls, args, value, subst, residuals):
        base, steps = args
        for n_steps, bound in counts(steps, subst):
            try:
                undelayed = value.delay(-n_steps)
            except NotImplementedError:  # Not the delay of anything.
                return
            if undelayed.delay(n_steps) == value:
                yield from unify(base, undelayed, bound, residuals)


class Repeat[X: abc.ColouredMonoid, N: Count](Pattern):
    """ A single wire ``X`` repeated ``N`` times, for the legs of a
    spider: matching binds the count to the number of wires and ``X`` to
    the one wire they all equal. """

    @classmethod
    def instantiate(cls, args, subst, unit):
        base, times = (instantiate(arg, subst, unit) for arg in args)
        return base ** times

    @classmethod
    def unify(cls, args, value, subst, residuals):
        base, times = args
        atoms = [value[i:i + 1] for i in range(len(value))]
        if any(atom != atoms[0] for atom in atoms[1:]):
            return
        sized = fits(times.__name__, value, subst)
        if sized is None:
            return
        if atoms:
            yield from unify(base, atoms[0], sized, residuals)
        else:
            yield sized, residuals


MAX_COUNT = 3
""" The largest number a ``Count`` stands for, drawn or matched. """


def counts(count, subst: Substitution
           ) -> Iterator[tuple[Any, Substitution]]:
    """ The numbers a count stands for, each with the substitution
    binding it: a ``Literal`` its own, a bound variable its binding, an
    unbound one every number up to :data:`MAX_COUNT`. """
    if get_origin(count) is Literal:
        yield get_args(count)[0], subst
    elif count.__name__ in subst:
        yield subst[count.__name__], subst
    else:
        for n in range(MAX_COUNT + 1):
            yield n, {**subst, count.__name__: n}


class Image[F, X](Pattern):
    """ The image ``Image[F, X]`` of a pattern under a functor, ``F`` the
    variable a premise ``functor: Var[Self, F]`` binds: instantiating
    applies the functor, which matching cannot invert, so the image is
    compared once the functor and the pattern are bound and is a
    residual until then. """

    @classmethod
    def instantiate(cls, args, subst, unit):
        functor, pattern = args
        return instantiate(functor, subst, unit)(
            instantiate(pattern, subst, unit))

    @classmethod
    def unify(cls, args, value, subst, residuals):
        functor, pattern = args
        if not all(label in subst for label in cls.variables(args)):
            yield subst, residuals + ((Image[functor, pattern], value), )
        elif cls.instantiate(args, subst, type(value)) == value:
            yield subst, residuals


def heads(category: type) -> dict[str, type]:
    """
    The types the heads of a sequent stand for in a category, by the
    name of the type parameter or class naming them, the objects and
    arrows of its domain and codomain too when it is a functor.

    >>> from discopy.monoidal import Ty, Diagram
    >>> assert heads(Diagram)[OBJECTS] is Ty
    """
    scope = {
        SELF: category,
        OBJECTS: getattr(category, "ob", category),
        ARROWS: getattr(category, "ar", category)}
    dom, cod = (getattr(category, name, None) for name in ("dom", "cod"))
    if isinstance(dom, type) and isinstance(cod, type):
        scope.update({
            "In0": getattr(dom, "ob", dom), "In1": dom,
            "Out0": getattr(cod, "ob", cod), "Out1": cod})
    return scope


@dataclass(frozen=True)
class Sort:
    """
    The sort of a variable, as its bound states it: ``Obj[C0]`` or no
    bound an object, ``Atom[C0]`` or ``Atom`` one of size one,
    ``Obj[C0, N]`` one of size ``N``, ``Count`` a number; or the
    sort of a premise, ``Obj[C0]`` or ``Atom[C0]`` with no pattern, or
    ``Self``. The head names what the sort ranges over: a type parameter
    of the declaring class, e.g. ``C0``, or else the objects of the
    category. The size is a number, the name of a ``Count`` variable or
    :obj:`None` for any. A side ``S: bool`` is a sort of its own, the
    value of the ``left`` of a method.

    >>> from discopy.monoidal import Ty
    >>> def cups[X: Atom[Ty]](): ...
    >>> print(Sort.of(cups.__type_params__[0]))
    Atom[Ty]
    >>> def trace[N: Count, M: Obj[Ty, N]](): ...
    >>> print(Sort.of(trace.__type_params__[1]))
    Obj[Ty, N]
    """

    head: str = OBJECTS
    size: int | str | None = None
    count: bool = False
    side: bool = False

    @property
    def atomic(self) -> bool:
        """ Whether the sort is of size one. """
        return self.size == 1

    @classmethod
    def of(cls, variable: TypeVar) -> Sort:
        """ The sort of a variable, read off its bound. """
        bound = variable.__bound__
        if bound is Count:
            return cls("Count", count=True)
        if bound is bool:
            return cls("bool", side=True)
        args = get_args(bound)
        return cls(name(args[0]) if args else OBJECTS, size=size(bound))

    @classmethod
    def premise(cls, annotation) -> Sort:
        """ The sort of a premise ``Obj[C0]`` or ``Atom[C0]``, or ``Self``. """
        if annotation is Self:
            return cls(SELF)
        return cls(name(get_args(annotation)[0]), size=size(annotation))

    def resolve(self, scope: dict) -> type:
        """ The type the head stands for in the scope. """
        if self.count or self.side:
            return int if self.count else bool
        return scope.get(self.head, scope[OBJECTS])

    def strategy(self, scope: dict, types=None, length: int | None = None):
        """ Generate an instance of the sort, from ``types`` in place of
        the strategy of the objects when given, of the given ``length``
        when the size is a variable. """
        from hypothesis import strategies as st

        if self.count:
            return st.integers(min_value=0, max_value=MAX_COUNT)
        if self.side:
            return st.booleans()
        resolved = self.resolve(scope)
        base = types if types is not None and resolved is scope[OBJECTS]\
            else resolved.strategy()
        length = self.size if isinstance(self.size, int) else length
        if length is None:
            return base
        atoms = base.filter(lambda value: len(value) == 1)
        return atoms if length == 1 else st.lists(
            atoms, min_size=length, max_size=length).map(
                lambda values: reduce(operator.matmul, values, resolved()))

    def canonical(self, scope: dict, label: str, length: int = 1):
        """ The canonical instance of the sort, named after a variable,
        ``length`` wires named ``label0, label1...`` when the size is a
        variable. """
        if self.count or self.side:
            return 2 if self.count else True
        if not isinstance(self.size, str):
            return cell(self.resolve(scope), label)
        return reduce(operator.matmul, (
            cell(self.resolve(scope), f"{label}{i}") for i in range(length)),
            self.resolve(scope)())

    def __str__(self):
        if self.size is None or self.count or self.side:
            return self.head
        return f"Atom[{self.head}]" if self.atomic\
            else f"Obj[{self.head}, {self.size}]"


def size(annotation) -> int | str | None:
    """ The size a sort states: a number, the name of its ``Count``
    variable or :obj:`None` for any. """
    if annotation is Atom or get_origin(annotation) is Atom:
        return 1
    if get_origin(annotation) is Unit:
        return 0
    args = get_args(annotation) if get_origin(annotation) is Obj else ()
    stated = args[1] if len(args) > 1 else None
    if isinstance(stated, TypeVar):
        return stated.__name__
    return get_args(stated)[0] if stated is not None else None


def fits(length: int | str | None, value,
         subst: Substitution) -> Substitution | None:
    """ The substitution with an object of a size, binding the size
    when it is an unbound variable, :obj:`None` when it does not fit. """
    if length is None:
        return subst
    if isinstance(length, int):
        return subst if len(value) == length else None
    if length in subst:
        return subst if subst[length] == len(value) else None
    return {**subst, length: len(value)}


def name(head) -> str:
    """ The name of a head: ``C0``, ``C1``, ``Ty``... or ``Self``. """
    return "Self" if head is Self else getattr(head, "__name__", str(head))


def premise(annotation) -> object | Sort | None:
    """
    What a parameter annotation states, :obj:`None` when it is no
    premise: ``Hom[...]`` and ``Var[T, p]`` are their own pattern, while
    ``Obj[C0]``, ``Atom[C0]``, ``Unit[C0]`` and ``Self`` are sorts.
    """
    if annotation is Self or get_origin(annotation) in (Obj, Atom, Unit):
        return Sort.premise(annotation)
    if get_origin(annotation) in (Hom, Var):
        return annotation
    return None


def variables(pattern) -> tuple[str, ...]:
    """ The names of the variables of a pattern, in order. """
    if isinstance(pattern, TypeVar):
        return (pattern.__name__, )
    origin, args = get_origin(pattern), get_args(pattern)
    if origin in (Hom, Var, Obj):
        return tuple(label for arg in args[1:] for label in variables(arg))
    if isinstance(origin, type) and issubclass(origin, Pattern):
        return origin.variables(args)
    return ()


def instantiate(pattern, subst: Substitution, unit: Any):
    """ The object a pattern stands for under a substitution, ``unit``
    the object type called to build the unit, a pair for a hom. """
    if isinstance(pattern, TypeVar):
        return subst[pattern.__name__]
    origin, args = get_origin(pattern), get_args(pattern)
    if origin is Literal:
        return args[0]
    if origin is Hom:
        return tuple(instantiate(arg, subst, unit) for arg in args[1:])
    if origin is Var:
        return instantiate(args[1], subst, unit)
    if size(pattern) == 0:
        return unit()
    if isinstance(origin, type) and issubclass(origin, Pattern):
        return origin.instantiate(args, subst, unit)
    raise TypeError(f"Expected a pattern, got {pattern!r}.")


def match(pattern, value, subst: Substitution | None = None,
          residuals: Residuals = ()) -> Iterator[Match]:
    """
    Unify a pattern with a value, or with anything at all when the
    value is :obj:`None`, a hom with a pair of optional sides.

    >>> from discopy.monoidal import Ty
    >>> A, B = TypeVar("A"), TypeVar("B")
    >>> x, y = Ty('x'), Ty('y')
    >>> for subst, _ in match(Tensor[A, B], x @ y):
    ...     print(subst['A'], '|', subst['B'])
    Ty() | x @ y
    x | y
    x @ y | Ty()
    """
    subst = {} if subst is None else subst
    if value is None:
        yield subst, residuals
    elif get_origin(pattern) is Hom:
        _, dom, cod = get_args(pattern)
        for subst_, residuals_ in match(dom, value[0], subst, residuals):
            yield from match(cod, value[1], subst_, residuals_)
    else:
        yield from unify(pattern, value, subst, residuals)


def unify(pattern, value, subst: Substitution,
          residuals: Residuals) -> Iterator[Match]:
    """ Unify a pattern with a concrete value, yielding every
    substitution with the residual equations it could not invert: a
    variable binds once, of the size its sort says, a sort matches any
    value of its size, and a :class:`Pattern` unifies its arguments. """
    if isinstance(pattern, TypeVar):
        label = pattern.__name__
        if label in subst:
            if subst[label] == value:
                yield subst, residuals
        elif (sized := fits(Sort.of(pattern).size, value, subst)) is not None:
            yield {**sized, label: value}, residuals
        return
    origin, args = get_origin(pattern), get_args(pattern)
    if origin is Var:
        yield from unify(args[1], value, subst, residuals)
    elif origin in (Obj, Atom, Unit):
        if (sized := fits(size(pattern), value, subst)) is not None:
            yield sized, residuals
    elif isinstance(origin, type) and issubclass(origin, Pattern):
        yield from origin.unify(args, value, subst, residuals)
    else:
        raise TypeError(f"Expected a pattern, got {pattern!r}.")


def sides(side, subst: Substitution) -> Iterator[tuple[Substitution, bool]]:
    """ The values of a side: a ``Literal`` its own, a bound variable
    its binding, an unbound one both, each binding it. """
    if not isinstance(side, TypeVar):
        yield subst, get_args(side)[0]
    elif side.__name__ in subst:
        yield subst, bool(subst[side.__name__])
    else:
        for left in (True, False):
            yield {**subst, side.__name__: left}, left


@dataclass(repr=False)
class Declaration[**P, T]:
    """
    A declaration is a sequent stated by a ``function`` on an abstract
    base class and inherited by every category below it: the base of the
    rules of :meth:`discopy.abc.Category.search` and of the axioms of
    :mod:`discopy.axioms`. The ``category`` is the class the
    declaration is bound to, ``name`` the attribute it is stored under
    and ``owner`` the class declaring the sequent.

    >>> from discopy.abc import Category
    >>> Category.then
    abc.Category.then
    >>> Category.then.conclusion
    Hom[C1, A, C]
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
    def variables(self) -> dict[str, Sort]:
        """ The type parameters of the function and their sorts. """
        return {
            variable.__name__: Sort.of(variable) for variable in getattr(
                inspect.unwrap(self.function), "__type_params__", ())}

    @property
    def premises(self) -> dict[str, object]:
        """ The parameters stating a premise, by name, each the pattern
        or the :class:`Sort` its annotation states. """
        parameters = inspect.signature(self.function).parameters.values()
        stated = {
            parameter.name: premise(parameter.annotation)
            for parameter in parameters
            if parameter.kind is not parameter.VAR_KEYWORD}
        return {
            label: value for label, value in stated.items()
            if value is not None}

    @property
    def conclusion(self):
        """ The return annotation, when the declaration concludes. """
        if not self.concludes:
            return None
        return inspect.signature(self.function).return_annotation

    @property
    def __isabstractmethod__(self):
        return getattr(self.function, "__isabstractmethod__", False)

    @property
    def __signature__(self):
        return inspect.signature(self.function)

    def __repr__(self):
        if self.category is None:
            return f"{type(self).__name__}({self.name})"
        return f"{factory_name(self.category)}.{self.name}"

    def __str__(self):
        return f"{self.name}{inspect.signature(self.function)}"

    def __hash__(self):
        return hash((self.function, self.category, self.name))

    def bind(self, category: type, owner: type | None = None) -> Self:
        """ Bind the declaration to a concrete category. """
        return replace(self, category=category, owner=self.owner or owner)

    @property
    def scope(self) -> dict[str, type]:
        """ What the heads stand for, see :func:`heads`. """
        if self.category is None:
            raise TypeError(f"{self.name} is not bound to a class.")
        return heads(self.category)

    @property
    def unit(self) -> Callable:
        """ The object type of the category, called to build the unit. """
        return self.scope[OBJECTS]

    def resolve(self, hom) -> type:
        """ The category a hom premise is a morphism of, by its head. """
        return self.scope.get(name(get_args(hom)[0]), self.scope[ARROWS])

    def canonical(self) -> dict[str, Any]:
        """
        The canonical arguments of the sequent, by name: each variable is
        the canonical instance of its sort and each premise a :func:`cell`
        named after its parameter.

        >>> from discopy.abc import MonoidalCategory
        >>> from discopy.monoidal import Diagram
        >>> tensor = MonoidalCategory.tensor.bind(Diagram)
        >>> for label, box in tensor.canonical().items():
        ...     print(f"{label}: {box.dom} -> {box.cod}")
        self: A -> B
        other: C -> D
        """
        sorts = self.variables
        counts = {
            label: sort.canonical(self.scope, label)
            for label, sort in sorts.items() if sort.count}
        subst = {
            label: counts.get(label) or sort.canonical(
                self.scope, label, counts.get(str(sort.size), 1))
            for label, sort in sorts.items()}
        args = {}
        for label, value in self.premises.items():
            if isinstance(value, Sort):
                args[label] = cell(value.resolve(self.scope), label)
            elif get_origin(value) is Hom:
                dom, cod = instantiate(value, subst, self.unit)
                args[label] = cell(self.resolve(value), label, dom, cod)
            else:
                args[label] = instantiate(value, subst, self.unit)
        return args

    def generate(self, draw: Callable, hom: Callable, subst=None,
                 residuals: Residuals = (), types=None) -> tuple:
        """
        Sample the arguments of the sequent inside a composite strategy,
        one premise at a time: a pattern is instantiated, a sort sampled,
        a hom sampled through ``hom(category, dom, cod)``. A variable
        standing alone on a side of a hom is read off the term found, so
        that the goal guides the search; the residuals of a match are
        checked once every variable is bound.
        """
        from hypothesis import assume

        subst = dict(subst or {})
        sorts = self.variables

        def side(pattern):
            if isinstance(pattern, TypeVar) and pattern.__name__ not in subst:
                return None
            bound(*variables(pattern))
            return instantiate(pattern, subst, self.unit)

        def read_off(pattern, value):
            if isinstance(pattern, TypeVar) and pattern.__name__ not in subst:
                check(sorts[pattern.__name__].size, value)
                subst[pattern.__name__] = value

        def check(length, value):
            sized = fits(length, value, subst)
            assume(sized is not None)
            subst.update(sized or {})

        def bound(*labels):
            for label in labels:
                if label in subst:
                    continue
                length = sorts[label].size
                if isinstance(length, str):
                    bound(length)
                subst[label] = draw(sorts[label].strategy(
                    self.scope, types, subst.get(str(length))), label=label)

        args = {}
        for label, value in self.premises.items():
            if isinstance(value, Sort) and value.resolve(self.scope)\
                    is self.scope[ARROWS] and hasattr(
                        self.scope[ARROWS], "rules"):
                args[label] = draw(
                    hom(self.scope[ARROWS], None, None), label=label)
            elif isinstance(value, Sort):
                if isinstance(value.size, str):
                    bound(value.size)
                args[label] = draw(value.strategy(
                    self.scope, types, subst.get(str(value.size))),
                    label=label)
            elif get_origin(value) is Hom:
                _, dom_pattern, cod_pattern = get_args(value)
                dom, cod = side(dom_pattern), side(cod_pattern)
                term = draw(hom(self.resolve(value), dom, cod), label=label)
                read_off(dom_pattern, term.dom)
                read_off(cod_pattern, term.cod)
                args[label] = term
            else:
                bound(*variables(value))
                args[label] = instantiate(value, subst, self.unit)
        for pattern, value in residuals:
            bound(*variables(pattern))
            assume(instantiate(pattern, subst, self.unit) == value)
        return subst, args



def cell(factory: type, label: str, dom=None, cod=None):
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
        return box(label, dom, cod)
    return factory(label)



def declarations[K: Declaration](cls: type, kind: type[K]) -> dict[str, K]:
    """
    The declarations of exactly a kind inherited by a class, bound to
    it and keyed by name, the latest in the method resolution order
    winning like ordinary attribute lookup; a declaration marked
    inapplicable or admissible, a declaration under a label other than
    its name, i.e. an alias, or anything that is not a declaration,
    assigned over an inherited one drops it.

    >>> from discopy.monoidal import Diagram
    >>> from discopy.pattern import Rule
    >>> list(declarations(Diagram, Rule))
    ['id', 'tensor', 'cut', 'dagger']
    """
    result: dict[str, K] = {}
    for base in reversed(cls.__mro__):
        for label, value in base.__dict__.items():
            while isinstance(value, (classmethod, staticmethod)):
                value = value.__func__
            if type(value) is kind and value.name == label\
                    and getattr(value, "__inapplicable__", None) is None\
                    and getattr(value, "__admissible__", None) is None:
                result[label] = value.bind(cls, owner=base)
            else:
                result.pop(label, None)
    return result



@dataclass(repr=False)
class Rule[**P, T](Declaration[P, T]):
    """
    An inference rule of a category, a
    :class:`Declaration` with a conclusion: every rule
    states its sequent as its own signature and
    :meth:`discopy.abc.Category.search` calls
    the attribute of the same name on the category. Accessed on a
    class, a rule binds to it, once per class; on an instance, it
    behaves as the method it decorates.

    >>> from discopy.abc import Category
    >>> print(Category.then)
    then(self: Hom[C1, A, B], other: Hom[C1, B, C]) -> Hom[C1, A, C]
    """

    __hash__ = Declaration.__hash__

    def __get__(self, instance, owner: type):
        if instance is not None:
            return MethodType(self.function, instance)
        bound = self.__dict__.get("bound")
        if bound is None:
            bound = self.__dict__["bound"] = {}
        if owner not in bound:
            declaring = next((
                base for base in owner.__mro__
                if self.is_declared(base.__dict__.get(self.name or ""))),
                None)
            bound[owner] = self.bind(owner, owner=declaring)
        return bound[owner]

    def is_declared(self, value) -> bool:
        """ Whether a class attribute is this very declaration. """
        value = getattr(value, "__func__", value)
        return value is self

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.function(*args, **kwargs)

    def match(self, dom=None, cod=None) -> Iterator[Match]:
        """ Unify the conclusion with a goal. """
        return match(self.conclusion, (dom, cod))

    @property
    def recursive(self) -> bool:
        """
        Whether a premise is a hom, which the search proves recursively
        below its depth bound; a rule with none is a generator, built in
        one step, see :meth:`discopy.abc.Category.generators`.

        >>> from discopy.abc import Category, RigidCategory
        >>> assert Category.then.recursive
        >>> assert not RigidCategory.cups.recursive
        """
        return any(
            get_origin(premise) is Hom for premise in self.premises.values())

    def apply(self, arguments: dict) -> T:
        """ The implementation of the rule on the category, applied
        positionally in premise order except the keyword-only parameters,
        applied by name. """
        function = getattr(self.category, self.name or "")
        try:
            keywords = {
                name for name, parameter
                in inspect.signature(function).parameters.items()
                if parameter.kind == parameter.KEYWORD_ONLY}
        except ValueError:
            keywords = set()
        return function(
            *(x for name, x in arguments.items() if name not in keywords),
            **{name: x for name, x in arguments.items() if name in keywords})

    @staticmethod
    def constant(box) -> Constant:
        """ The rule of one given box, see :class:`Constant`. """
        return Constant(box)

    def inapplicable(self, reason: str) -> Self:
        """
        The same rule dropped from the rules and generators of the
        class it is assigned on, because the structure it builds lies
        outside the category's terms, with the reason as its record:
        the method still runs, the search just never applies it, e.g.
        ``trace = frobenius.Diagram.trace.inapplicable("No loop in a
        sentence.")``. A rule the category does have, whose terms
        other rules reach, is :meth:`admissible` instead.
        """
        result = replace(self)
        result.__inapplicable__ = reason
        return result

    def admissible(self, reason: str) -> Self:
        """
        The same rule dropped from the rules and generators of the
        class it is assigned on, because it is `admissible
        <https://en.wikipedia.org/wiki/Admissible_rule>`_: the search
        reaches everything it builds through the other rules, which
        the reason names as its record, e.g. ``then =
        cat.Arrow.then.admissible("A cut with empty contexts.")``.
        The method still runs and remains applicable.
        """
        result = replace(self)
        result.__admissible__ = reason
        return result



@dataclass(repr=False)
class Constant(Rule):
    """
    The rule of one given box: it applies exactly to the sequent of the
    box and builds it, so that a category generated by a fixed vocabulary
    — the words of a grammar, the gates of a circuit — assigns a
    dictionary of these to its ``generators``.

    >>> from discopy.grammar import pregroup
    >>> n = pregroup.Ty('n')
    >>> rule = Rule.constant(pregroup.Word('Alice', n))
    >>> assert rule.apply({}) == pregroup.Word('Alice', n)
    """

    __hash__ = Declaration.__hash__

    def __post_init__(self):
        self.name = self.name or str(self.function)
        self.__doc__ = f"The constant {self.name}."

    variables = premises = {}
    conclusion = None

    def match(self, dom=None, cod=None) -> Iterator[Match]:
        box = self.function
        if dom in (None, box.dom) and cod in (None, box.cod):
            yield {}, ()

    def generate(self, draw, hom, subst=None, residuals=(), types=None):
        return dict(subst or {}), {}

    def apply(self, arguments: dict):
        return self.function



def rule[**P, T](function: Callable[P, T]) -> Rule[P, T]:
    """ Decorate a method as an inference rule, its signature the sequent. """
    return Rule(function)



from discopy import abc  # noqa: E402  pylint: disable=wrong-import-position
