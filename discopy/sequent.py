"""
The sequent of a method, as Python evaluates its signature, and its
matching against a goal.

A :class:`Declaration` reads its sequent off the function it decorates
with :func:`inspect.signature`, i.e. the annotations as :pep:`649`
evaluates them by default, the same objects a typechecker sees: the
variables are ``function.__type_params__``, each with its sort as
``__bound__``, the premises are the parameters annotated with
:data:`discopy.pattern.Obj`, :data:`discopy.pattern.Hom` or
:obj:`typing.Self` and the conclusion is the return annotation. Nothing
is rebuilt: a pattern is the generic alias Python returns, a variable
the :class:`typing.TypeVar` of the declaration, and :func:`unify` and
:func:`instantiate` interpret them by their ``__origin__`` and
``__args__``.

A conclusion is matched against a goal, a pair of an optional domain and
codomain, by unification over the free monoid of objects: a ``Tensor``
splits the goal at every position, a variable binds once, an adjoint
``R[p]`` inverts to ``p``. What cannot be inverted, an exponential that
collapsed into adjoints, is a residual equation checked once every
variable is instantiated.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Sort
    Declaration

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        heads
        variables
        instantiate
        unify
        match
        cell
        declarations
"""

import inspect
import operator
from collections.abc import Callable, Iterator
from dataclasses import KW_ONLY, dataclass, replace
from functools import reduce
from typing import Any, ClassVar, Self, TypeVar, get_args, get_origin

from discopy.pattern import (
    Atom, Count, D, Hom, L, Obj, Over, R, Repeat, Tensor, Under, Unit)
from discopy.utils import factory_name


type Substitution = dict[str, object]
type Residuals = tuple[tuple[object, object], ...]
type Match = tuple[Substitution, Residuals]

#: The heads of the sequents of a category.
OBJECTS, ARROWS, SELF = "C0", "C1", "Self"


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
    bound an object, ``Atom[C0]`` or ``Atom`` an atomic one, ``Count``
    a number of repetitions; or the sort of a premise, ``Obj[C0]`` with
    no pattern or ``Self``.

    >>> def cups[In0, X: Atom[In0]](): ...
    >>> print(Sort.of(cups.__type_params__[1]))
    Atom[In0]
    """

    head: str = OBJECTS
    atomic: bool = False
    count: bool = False

    @classmethod
    def of(cls, variable: TypeVar) -> Sort:
        """ The sort of a variable, read off its bound. """
        bound = variable.__bound__
        if bound is Count:
            return cls("Count", count=True)
        args = get_args(bound)
        head = name(args[0]) if args else OBJECTS
        return cls(head, atomic=bound is Atom or get_origin(bound) is Atom)

    def resolve(self, scope: dict) -> type:
        """ The type the head stands for in the scope. """
        if self.count:
            return int
        try:
            return scope[self.head]
        except KeyError:
            raise KeyError(
                f"{self.head} is no head of this scope, which names "
                f"{', '.join(scope)}.") from None

    def strategy(self, scope: dict, types=None):
        """ Generate an instance of the sort, from ``types`` in place of
        the strategy of the objects when given. """
        from hypothesis import strategies as st

        if self.count:
            return st.integers(min_value=0, max_value=3)
        resolved = self.resolve(scope)
        base = types if types is not None and resolved is scope[OBJECTS]\
            else resolved.strategy()
        return base.filter(lambda value: len(value) == 1)\
            if self.atomic else base

    def canonical(self, scope: dict, label: str):
        """ The canonical instance of the sort, named after a variable. """
        return 2 if self.count else cell(self.resolve(scope), label)

    def __str__(self):
        return f"Atom[{self.head}]" if self.atomic else self.head


def name(head) -> str:
    """ The name of a head: ``C0``, ``C1``, ``In0``... or ``Self``. """
    return "Self" if head is Self else getattr(head, "__name__", ARROWS)


def premise(annotation) -> object | Sort | None:
    """
    What a parameter annotation states, :obj:`None` when it is no
    premise: ``Hom[...]`` and ``Obj[T, p]`` are their own pattern, while
    ``Obj[C0]`` and ``Self`` are sorts.
    """
    if annotation is Self:
        return Sort(SELF)
    if get_origin(annotation) is Hom:
        return annotation
    if get_origin(annotation) is Obj:
        args = get_args(annotation)
        return annotation if len(args) == 2 and args[1] is not None\
            else Sort(name(args[0]))
    return None


def variables(pattern) -> tuple[str, ...]:
    """ The names of the variables of a pattern, in order. """
    if isinstance(pattern, TypeVar):
        return (pattern.__name__, )
    origin, args = get_origin(pattern), get_args(pattern)
    if origin is Unit or origin is None:
        return ()
    if origin in (Hom, Obj):
        args = args[1:]
    return tuple(label for arg in args for label in variables(arg))


def instantiate(pattern, subst: Substitution, unit: Any):
    """ The object a pattern stands for under a substitution, ``unit``
    the object type called to build the unit, a pair for a hom. """
    if isinstance(pattern, TypeVar):
        return subst[pattern.__name__]
    origin, args = get_origin(pattern), get_args(pattern)
    if origin is Hom:
        return tuple(instantiate(arg, subst, unit) for arg in args[1:])
    if origin is Obj:
        return instantiate(args[1], subst, unit)
    if origin is Unit:
        return unit()
    values: list[Any] = [instantiate(arg, subst, unit) for arg in args]
    if origin is Tensor:
        return reduce(operator.matmul, values)
    if origin in (L, R):
        return getattr(values[0], "l" if origin is L else "r")
    if origin is D:
        return values[0].d
    if origin is Over:
        return values[0] << values[1]
    if origin is Under:
        return values[0] >> values[1]
    if origin is Repeat:
        return values[0] ** values[1]
    raise TypeError(f"Expected a pattern, got {pattern!r}.")


def match(pattern, value, subst: Substitution | None = None,
          residuals: Residuals = ()) -> Iterator[Match]:
    """
    Unify a pattern with a value, or with anything at all when the
    value is :obj:`None`, a hom with a pair of optional sides.

    >>> from discopy.monoidal import Ty
    >>> from discopy.pattern import Tensor
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
    substitution with the residual equations it could not invert. """
    if isinstance(pattern, TypeVar):
        label, sort = pattern.__name__, Sort.of(pattern)
        if label in subst:
            if subst[label] == value:
                yield subst, residuals
        elif not sort.atomic or len(value) == 1:
            yield dict(subst, **{label: value}), residuals
        return
    origin, args = get_origin(pattern), get_args(pattern)
    if origin is Obj:
        yield from unify(args[1], value, subst, residuals)
    elif origin is Unit:
        if not len(value):
            yield subst, residuals
    elif origin is Tensor:
        yield from split(args, value, subst, residuals)
    elif origin in (L, R):
        inverse = getattr(value, "r" if origin is L else "l")
        yield from unify(args[0], inverse, subst, residuals)
    elif origin is D:
        yield from unify_delay(args[0], value, subst, residuals)
    elif origin in (Over, Under):
        yield from unify_exp(pattern, value, subst, residuals)
    elif origin is Repeat:
        yield from unify_repeat(args[0], args[1], value, subst, residuals)
    else:
        raise TypeError(f"Expected a pattern, got {pattern!r}.")


def split(factors, value, subst, residuals) -> Iterator[Match]:
    """ Unify factors with the prefixes of a value, in turn. """
    if not factors:
        if not len(value):
            yield subst, residuals
        return
    head, *tail = factors
    for n in range(len(value) + 1):
        for subst_, residuals_ in unify(head, value[:n], subst, residuals):
            yield from split(tail, value[n:], subst_, residuals_)


def unify_delay(base, value, subst, residuals) -> Iterator[Match]:
    """ Unify a delay, inverted by the delay ``-1`` steps back. """
    try:
        undelayed = value.delay(-1)
    except NotImplementedError:  # Not the delay of anything.
        return
    if undelayed.d == value:
        yield from unify(base, undelayed, subst, residuals)


def unify_exp(pattern, value, subst, residuals) -> Iterator[Match]:
    """ Unify an exponential against a single exponential object its
    base and exponent rebuild, keeping the residual at a level whose
    exponentials collapse into adjoints. """
    over = get_origin(pattern) is Over
    atom = value.inside[0] if len(value) == 1 else None
    base: Any = getattr(atom, "base", None)
    exponent: Any = getattr(atom, "exponent", None)
    operands = (base, exponent) if over else (exponent, base)
    if base is None or exponent is None or value != (
            base << exponent if over else exponent >> base):
        if hasattr(value, "r"):  # A pregroup: the residual is checked.
            yield subst, residuals + ((pattern, value), )
        return
    left, right = get_args(pattern)
    for subst_, residuals_ in unify(left, operands[0], subst, residuals):
        yield from unify(right, operands[1], subst_, residuals_)


def unify_repeat(base, count, value, subst, residuals) -> Iterator[Match]:
    """ Unify an atom repeated ``count`` times: the count binds to the
    number of atoms and the base to the one atom they all equal. """
    atoms = [value[i:i + 1] for i in range(len(value))]
    if any(atom != atoms[0] for atom in atoms[1:]):
        return
    label = count.__name__
    if label in subst and subst[label] != len(atoms):
        return
    subst = {**subst, label: len(atoms)}
    if atoms:
        yield from unify(base, atoms[0], subst, residuals)
    else:
        yield subst, residuals


@dataclass(repr=False)
class Declaration[**P, T]:
    """
    A declaration is a sequent stated by a ``function`` on an abstract
    base class and inherited by every category below it: the base of the
    rules of :mod:`discopy.search` and of the axioms of
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

    def __set_name__(self, owner: type, name: str):
        if self.category is None:
            self.name = name

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
        subst = {
            label: sort.canonical(self.scope, label)
            for label, sort in self.variables.items()}
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
                assume(not sorts[pattern.__name__].atomic or len(value) == 1)
                subst[pattern.__name__] = value

        def bound(*labels):
            for label in labels:
                if label not in subst:
                    subst[label] = draw(
                        sorts[label].strategy(self.scope, types), label=label)

        args = {}
        for label, value in self.premises.items():
            if isinstance(value, Sort) and value.resolve(self.scope)\
                    is self.scope[ARROWS] and hasattr(
                        self.scope[ARROWS], "rules"):
                args[label] = draw(
                    hom(self.scope[ARROWS], None, None), label=label)
            elif isinstance(value, Sort):
                args[label] = draw(
                    value.strategy(self.scope, types), label=label)
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


def declarations[D: Declaration](cls: type, kind: type[D]) -> dict[str, D]:
    """
    The declarations of exactly a kind inherited by a class, bound to
    it and keyed by name, the latest in the method resolution order
    winning like ordinary attribute lookup; a declaration marked
    inapplicable or admissible, or anything that is not a declaration,
    assigned over an inherited one drops it.

    >>> from discopy.monoidal import Diagram
    >>> from discopy.search import Rule
    >>> list(declarations(Diagram, Rule))
    ['id', 'tensor', 'cut']
    """
    result: dict[str, D] = {}
    for base in reversed(cls.__mro__):
        for label, value in base.__dict__.items():
            while isinstance(value, (classmethod, staticmethod)):
                value = value.__func__
            if type(value) is kind\
                    and getattr(value, "__inapplicable__", None) is None\
                    and getattr(value, "__admissible__", None) is None:
                result[label] = value.bind(cls, owner=base)
            else:
                result.pop(label, None)
    return result
