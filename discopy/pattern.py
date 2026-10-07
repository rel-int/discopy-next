"""
The patterns in which a category states its rules, generators and
axioms, which :meth:`discopy.abc.Category.search` searches its terms by.

A sequent is the signature of a method on an abstract base class of
:mod:`discopy.abc`: its :pep:`695` type parameter list is the context,
each parameter one variable with its sort as the bound — ``A: Obj[C0]``
an object, ``X: Atom[C0]`` a single wire, ``M: Obj[C0, N]`` an object
of size ``N`` and ``N: Count`` a number — its
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

    Count
    Substitution
    Pattern
    Tensor
    Exponential
    Over
    Under
    L
    R
    D
    Repeat
    Image
    Sort
    Objects
    Terms
    Counts
    Declaration
    Rule
    Constant

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        rule
"""

import inspect
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import KW_ONLY, dataclass, replace
from functools import reduce
from itertools import count
from types import MethodType
from typing import (
    Annotated, Any, ClassVar, Generic, Literal, Self, TypeVar, get_args,
    get_origin)

from discopy.utils import factory_name

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

class Count:
    """ The bound ``N: Count`` of a number, e.g. of repetitions. """


class Substitution(dict[str, Any]):
    """
    The values of the variables of patterns, by name, with the
    ``residuals`` unification could not invert: pairs of a pattern and the
    value it must instantiate to, checked once its variables are bound.

    >>> from discopy.monoidal import Ty
    >>> A, B = TypeVar("A"), TypeVar("B")
    >>> x, y = Ty('x'), Ty('y')
    >>> for subst in Substitution().unify(Tensor[A, B], x @ y):
    ...     print(subst['A'], '|', subst['B'])
    Ty() | x @ y
    x | y
    x @ y | Ty()
    """
    fresh_names: ClassVar = count()

    def __init__(self, *args, residuals: tuple = (), **kwargs):
        super().__init__(*args, **kwargs)
        self.residuals = tuple(residuals)

    @classmethod
    def fresh(cls) -> TypeVar:
        """ A variable named apart from every other. """
        return TypeVar(  # ty: ignore[invalid-legacy-type-variable]
            f"?{next(cls.fresh_names)}")

    def guide(self, side, ob: Any):
        """ The value a side of a goal stands for once its variables are
        bound, :obj:`None` until then. """
        if all(name in self for name in Pattern.variables(side)):
            return self.instantiate(side, ob)
        return None

    def __repr__(self):
        return f"Substitution({dict(self)!r}, residuals={self.residuals!r})"

    def bind(self, **values) -> Substitution:
        """ The substitution with more values. """
        return Substitution(self, residuals=self.residuals, **values)

    def residual(self, pattern, value) -> Substitution:
        """ The substitution with one more residual equation. """
        return Substitution(
            self, residuals=self.residuals + ((pattern, value), ))

    def fit(self, size: int | str | None, value) -> Substitution | None:
        """ The substitution with a value of a size, binding the size when
        it is an unbound variable, :obj:`None` when the value does not
        fit. """
        if size is None:
            return self
        if isinstance(size, int):
            return self if len(value) == size else None
        if size in self:
            return self if self[size] == len(value) else None
        return self.bind(**{size: len(value)})

    def instantiate(self, pattern, ob: Any):
        """ The value a pattern stands for, ``ob`` the type of objects
        called to build the unit, a pair of sides for a hom. """
        if isinstance(pattern, TypeVar):
            return self[pattern.__name__]
        origin, args = get_origin(pattern), get_args(pattern)
        if origin is Literal:
            return args[0]
        if origin is Hom:
            return tuple(self.instantiate(arg, ob) for arg in args[1:])
        if origin is Var:
            return self.instantiate(args[1], ob)
        if origin in (Obj, Atom, Unit) and Sort.read(pattern).size == 0:
            return ob()
        if isinstance(origin, type) and issubclass(origin, Pattern):
            return origin.instantiate(args, self, ob)
        if origin is None:  # A value is a pattern with no variable.
            return pattern
        raise TypeError(f"Expected a pattern, got {pattern!r}.")

    def unify(self, pattern, value) -> Iterator[Substitution]:
        """
        Every substitution extending this one under which a pattern stands
        for a value: anything at all when the value is :obj:`None`, a pair
        of optional sides for a hom. A variable binds once, of the size its
        sort says, a sort matches any value of its size and a
        :class:`Pattern` unifies its arguments.
        """
        if value is None:
            yield self
            return
        if isinstance(pattern, TypeVar):
            label = pattern.__name__
            if label in self:
                if self[label] == value:
                    yield self
            elif (fit := self.fit(getattr(
                    Sort.read(pattern.__bound__), "size", None), value)
                  ) is not None:
                yield fit.bind(**{label: value})
            return
        origin, args = get_origin(pattern), get_args(pattern)
        if origin is Hom:
            for subst in self.unify(args[1], value[0]):
                yield from subst.unify(args[2], value[1])
        elif origin is Var:
            yield from self.unify(args[1], value)
        elif origin in (Obj, Atom, Unit):
            if (fit := self.fit(Sort.read(pattern).size, value)) is not None:
                yield fit
        elif isinstance(origin, type) and issubclass(origin, Pattern):
            yield from origin.unify(args, value, self)
        elif origin is None:  # A value is a pattern with no variable.
            if pattern == value:
                yield self
        else:
            raise TypeError(f"Expected a pattern, got {pattern!r}.")


class Pattern(ABC):
    """
    A pattern for the objects of a category, with variables to
    instantiate. A pattern class is never instantiated: a pattern is the
    alias Python builds by subscripting it, e.g. ``Tensor[A, B]``, and
    the class interprets the arguments of its own aliases.
    """

    @staticmethod
    def variables(pattern) -> tuple[str, ...]:
        """ The names of the variables of a pattern, in order. """
        if isinstance(pattern, TypeVar):
            return (pattern.__name__, )
        origin, args = get_origin(pattern), get_args(pattern)
        if origin in (Hom, Var, Obj):
            args = args[1:]
        elif not (isinstance(origin, type) and issubclass(origin, Pattern)):
            return ()
        return tuple(
            label for arg in args for label in Pattern.variables(arg))

    @classmethod
    @abstractmethod
    def instantiate(cls, args: tuple, subst: Substitution, ob):
        """ The object the pattern of these arguments stands for. """

    @classmethod
    @abstractmethod
    def unify(cls, args: tuple, value, subst: Substitution
              ) -> Iterator[Substitution]:
        """ Every substitution under which the pattern of these arguments
        stands for a value. """


class Tensor[*Ts](Pattern):
    """ The tensor ``Tensor[A, B, ...]`` of two or more patterns,
    matched by splitting the value at every position. """

    @classmethod
    def instantiate(cls, args, subst, ob):
        return reduce(operator.matmul, (
            subst.instantiate(arg, ob) for arg in args))

    @classmethod
    def unify(cls, args, value, subst):
        if not args:
            if not len(value):
                yield subst
            return
        head, *tail = args
        for n in range(len(value) + 1):
            for unified in subst.unify(head, value[:n]):
                yield from cls.unify(tail, value[n:], unified)


class Exponential(Pattern):
    """
    The exponentials :class:`Over` and :class:`Under`, matched with a
    single exponential object that its base and exponent rebuild and
    nothing else — except at a level that collapses its exponentials into
    the adjoints of a pregroup, which keeps the residual.
    """
    left: ClassVar[bool]

    @classmethod
    def sides(cls, args: tuple) -> tuple:
        """ The base and the exponent, in this order. """
        return args if cls.left else args[::-1]

    @classmethod
    def instantiate(cls, args, subst, ob):
        base, exponent = (
            subst.instantiate(arg, ob) for arg in cls.sides(args))
        return base << exponent if cls.left else exponent >> base

    @classmethod
    def unify(cls, args, value, subst):
        base, exponent = cls.sides(args)
        atom = value.inside[0] if len(value) == 1 else None
        found: Any = (getattr(atom, "base", None),
                      getattr(atom, "exponent", None))
        if None in found or value != (
                found[0] << found[1] if cls.left else found[1] >> found[0]):
            if hasattr(value, "r"):  # A pregroup: the residual stays.
                exponential = Over if cls.left else Under
                yield subst.residual(exponential[args], value)
            return
        for unified in subst.unify(base, found[0]):
            yield from unified.unify(exponent, found[1])


class Over[Z: abc.ResiduatedMonoid, Y: abc.ResiduatedMonoid](Exponential):
    """ The exponential ``Over[Z, Y]``, i.e. ``Z << Y``. """
    left = True


class Under[Y: abc.ResiduatedMonoid, Z: abc.ResiduatedMonoid](Exponential):
    """ The exponential ``Under[Y, Z]``, i.e. ``Y >> Z``. """
    left = False


class L[T: abc.Pregroup](Pattern):
    """ The left adjoint ``L[T]``, i.e. ``T.l``, inverted by the right
    adjoint when matching. """

    @classmethod
    def instantiate(cls, args, subst, ob):
        (base, ) = args
        return subst.instantiate(base, ob).l

    @classmethod
    def unify(cls, args, value, subst):
        (base, ) = args
        yield from subst.unify(base, value.r)


class R[T: abc.Pregroup](Pattern):
    """ The right adjoint ``R[T]``, i.e. ``T.r``, inverted by the left
    adjoint when matching. """

    @classmethod
    def instantiate(cls, args, subst, ob):
        (base, ) = args
        return subst.instantiate(base, ob).r

    @classmethod
    def unify(cls, args, value, subst):
        (base, ) = args
        yield from subst.unify(base, value.l)


class D[T: abc.DelayedMonoid](Pattern):
    """ The delay ``D[T]`` of a pattern by one time step, inverted by the
    delay ``-1`` steps back when matching. """

    @classmethod
    def instantiate(cls, args, subst, ob):
        (base, ) = args
        return subst.instantiate(base, ob).d

    @classmethod
    def unify(cls, args, value, subst):
        (base, ) = args
        try:
            undelayed = value.delay(-1)
        except NotImplementedError:  # Not the delay of anything.
            return
        if undelayed.d == value:
            yield from subst.unify(base, undelayed)


class Repeat[X: abc.ColouredMonoid, N: Count](Pattern):
    """ A single wire ``X`` repeated ``N`` times, for the legs of a
    spider: matching binds the count to the number of wires and ``X`` to
    the one wire they all equal. """

    @classmethod
    def instantiate(cls, args, subst, ob):
        base, times = (subst.instantiate(arg, ob) for arg in args)
        return base ** times

    @classmethod
    def unify(cls, args, value, subst):
        base, times = args
        atoms = [value[i:i + 1] for i in range(len(value))]
        if any(atom != atoms[0] for atom in atoms[1:]):
            return
        fit = subst.fit(times.__name__, value)
        if fit is None:
            return
        if atoms:
            yield from fit.unify(base, atoms[0])
        else:
            yield fit


class Image[F, X](Pattern):
    """ The image ``Image[F, X]`` of a pattern under a functor, ``F`` the
    variable a premise ``functor: Var[Self, F]`` binds: instantiating
    applies the functor, which matching cannot invert, so the image is
    compared once the functor and the pattern are bound and is a
    residual until then. """

    @classmethod
    def instantiate(cls, args, subst, ob):
        functor, pattern = args
        return subst.instantiate(functor, ob)(subst.instantiate(pattern, ob))

    @classmethod
    def unify(cls, args, value, subst):
        functor, pattern = args
        image = Image[functor, pattern]
        if not all(label in subst for label in Pattern.variables(image)):
            yield subst.residual(image, value)
        elif cls.instantiate(args, subst, type(value)) == value:
            yield subst


@dataclass(frozen=True)
class Sort(ABC):
    """
    The sort of a variable, as its bound states it, or of a premise, as
    its annotation does: one subclass for each kind of head, the objects
    of a class or type parameter, the terms of the category ``Self`` and
    a number ``Count``.

    >>> from discopy.monoidal import Ty
    >>> def cups[X: Atom[Ty]](): ...
    >>> print(Sort.read(cups.__type_params__[0].__bound__))
    Atom[Ty]
    >>> def trace[N: Count, M: Obj[Ty, N]](): ...
    >>> print(Sort.read(trace.__type_params__[1].__bound__))
    Obj[Ty, N]
    """

    @staticmethod
    def read(annotation) -> Sort:
        """ The sort an annotation states: ``Count``, ``Self``, or the
        objects of ``Obj[T]``, ``Atom[T]``, ``Unit[T]`` or of anything. """
        if annotation is Count:
            return Counts()
        if annotation is Self:
            return Terms()
        origin, args = get_origin(annotation), get_args(annotation)
        if annotation is Atom or origin is Atom:
            return Objects(args[0] if args else None, 1)
        if origin is Unit:
            return Objects(args[0], 0)
        stated = args[1] if origin is Obj and len(args) > 1 else None
        size = stated.__name__ if isinstance(stated, TypeVar)\
            else None if stated is None else get_args(stated)[0]
        return Objects(args[0] if args else None, size)

    @staticmethod
    def position(owner: type, variable: TypeVar) -> int:
        """
        The position of a type parameter among those of the class at the
        root of its generic bases, e.g. ``C1`` of
        :class:`discopy.abc.Monoid` is the second parameter of
        :class:`discopy.abc.Category`, since a monoid is a
        ``ColouredMonoid[NoneType, C1]``. A class whose bases do not
        mention its parameter, e.g. a functor redeclaring its own, is
        such a root. ``owner`` is the class declaring the sequent, the
        parameter one of its own or of a class above it.

        >>> from discopy.abc import Monoid
        >>> assert Sort.position(Monoid, Monoid.__type_params__[0]) == 1
        """
        found = next(
            (base for base in owner.__mro__
             if variable in getattr(base, "__type_params__", ())), None)
        if found is None:
            raise TypeError(f"{variable} is no parameter above {owner}.")
        owner = found
        while True:
            alias = next((
                base for base in vars(owner).get("__orig_bases__", ())
                if get_origin(base) is not Generic
                and variable in get_args(base)), None)
            if alias is None:
                return owner.__type_params__.index(variable)
            owner = get_origin(alias)
            variable = owner.__type_params__[
                get_args(alias).index(variable)]

    @staticmethod
    def stands_for(head, category: type, default: type,
                   owner: type | None = None) -> type:
        """
        The type a head of a sequent stands for in a category: the
        category itself for ``Self``, what
        :meth:`discopy.abc.Category.parameters` says of a type parameter
        of the class ``owner`` declaring the sequent, by its
        :meth:`position`, and for a class the category's own subclass of
        it when it has one, e.g. the objects of a symmetric diagram for
        ``monoidal.Ty``. :obj:`None` and ``Any`` stand for the
        ``default``.

        >>> from discopy import monoidal, symmetric
        >>> assert Sort.stands_for(monoidal.Ty, symmetric.Diagram, None)\\
        ...     is symmetric.Diagram.ob
        """
        if head is Self:
            return category
        if head is None or head is Any:
            return default
        if isinstance(head, TypeVar):
            return category.parameters()[
                Sort.position(owner or category, head)]
        for own in (category, getattr(category, "ob", None)):
            if isinstance(own, type) and isinstance(head, type)\
                    and issubclass(own, head):
                return own
        return head

    @staticmethod
    def named(factory: type, label: str, dom=None, cod=None):
        """
        An instance of a class named after a variable or a parameter: a
        box of a class with a ``Box``, between ``dom`` and ``cod`` or
        objects named ``x`` and ``y``, else the instance of that name.

        >>> from discopy.monoidal import Box, Diagram, Ty
        >>> assert Sort.named(Diagram, 'f') == Box('f', Ty('x'), Ty('y'))
        >>> assert Sort.named(Ty, 'A') == Ty('A')
        """
        box = getattr(factory, "Box", None)
        if box is not None and isinstance(box, type):
            dom = factory.ob("x") if dom is None else dom
            cod = factory.ob("y") if cod is None else cod
            return box(label, dom, cod)
        return factory(label)

    @abstractmethod
    def resolve(self, category: type, owner: type | None = None) -> type:
        """ The type the sort ranges over in a category, for a sequent
        declared by ``owner``, see :meth:`stands_for`. """

    @abstractmethod
    def strategy(self, category: type, owner: type | None = None,
                 length: int | None = None):
        """ Generate an instance of the sort in a category, of the given
        ``length`` when its size is a variable. """

    @abstractmethod
    def canonical(self, category: type, label: str,
                  owner: type | None = None, length: int = 1):
        """ The canonical instance of the sort, named after a variable. """


@dataclass(frozen=True)
class Objects(Sort):
    """
    The objects of a class or of a type parameter, of a size: a number,
    the name of a ``Count`` variable or :obj:`None` for any. A head
    :obj:`None` stands for the objects of the category.
    """
    head: Any = None
    size: int | str | None = None

    def resolve(self, category: type, owner: type | None = None) -> type:
        return self.stands_for(self.head, category, category.ob, owner)

    def strategy(self, category: type, owner: type | None = None,
                 length: int | None = None):
        from hypothesis import strategies as st

        resolved = self.resolve(category, owner)
        length = self.size if isinstance(self.size, int) else length
        if length is None:
            return resolved.strategy()
        try:
            return resolved.strategy(min_length=length, max_length=length)
        except TypeError:  # A strategy with no length to ask for.
            atoms = resolved.strategy().filter(lambda value: len(value) == 1)
        return atoms if length == 1 else st.lists(
            atoms, min_size=length, max_size=length).map(
                lambda values: reduce(operator.matmul, values, resolved()))

    def canonical(self, category: type, label: str,
                  owner: type | None = None, length: int = 1):
        resolved = self.resolve(category, owner)
        if not isinstance(self.size, str):
            return self.named(resolved, label)
        return reduce(operator.matmul, (
            self.named(resolved, f"{label}{i}") for i in range(length)),
            resolved())

    def __str__(self):
        head = getattr(self.head, "__name__", None)
        if self.size is None:
            return f"Obj[{head}]" if head else "Obj"
        if self.size == 1:
            return f"Atom[{head}]" if head else "Atom"
        return f"Obj[{head or '_'}, {self.size}]"


@dataclass(frozen=True)
class Terms(Sort):
    """ The terms of the category, the sort of a premise ``Self``. """
    head: ClassVar = Self

    def resolve(self, category: type, owner: type | None = None) -> type:
        return category

    def strategy(self, category: type, owner: type | None = None,
                 length: int | None = None):
        return category.strategy()

    def canonical(self, category: type, label: str,
                  owner: type | None = None, length: int = 1):
        return self.named(category, label)

    def __str__(self):
        return "Self"


@dataclass(frozen=True)
class Counts(Sort):
    """ The numbers a ``Count`` stands for, up to :attr:`maximum` when
    drawn. """
    head: ClassVar = Count
    maximum: ClassVar[int] = 3

    def resolve(self, category: type, owner: type | None = None) -> type:
        return int

    def strategy(self, category: type, owner: type | None = None,
                 length: int | None = None):
        from hypothesis import strategies as st
        return st.integers(min_value=0, max_value=self.maximum)

    def canonical(self, category: type, label: str,
                  owner: type | None = None, length: int = 1):
        return 2

    def __str__(self):
        return "Count"


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
    >>> Category.cut
    abc.Category.cut
    >>> Category.cut.conclusion
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
            variable.__name__: Sort.read(variable.__bound__)
            for variable in getattr(
                inspect.unwrap(self.function), "__type_params__", ())}

    @property
    def premises(self) -> dict[str, object]:
        """ The parameters stating a premise, by name: ``Hom[...]`` and
        ``Var[T, p]`` are their own pattern, while ``Obj[C0]``,
        ``Atom[C0]``, ``Unit[C0]`` and ``Self`` state a :class:`Sort`. """
        result = {}
        for parameter in inspect.signature(
                self.function).parameters.values():
            annotation = parameter.annotation
            if parameter.kind is parameter.VAR_KEYWORD:
                continue
            if annotation is Self or get_origin(annotation) in (
                    Obj, Atom, Unit):
                result[parameter.name] = Sort.read(annotation)
            elif get_origin(annotation) in (Hom, Var):
                result[parameter.name] = annotation
        return result

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

    @classmethod
    def inherited(cls, category: type) -> dict[str, Self]:
        """
        The declarations of exactly this kind inherited by a class, bound
        to it and keyed by name, the latest in the method resolution order
        winning like ordinary attribute lookup; a declaration marked
        inapplicable or admissible, a declaration under a label other than
        its name, i.e. an alias, or anything that is not a declaration,
        assigned over an inherited one drops it.

        >>> from discopy.monoidal import Diagram
        >>> list(Rule.inherited(Diagram))
        ['ax', 'cut', 'mix', 'dagger']
        """
        result: dict[str, Self] = {}
        for base in reversed(category.__mro__):
            for label, value in base.__dict__.items():
                while isinstance(value, (classmethod, staticmethod)):
                    value = value.__func__
                if type(value) is cls and value.name == label\
                        and getattr(value, "__inapplicable__", None) is None\
                        and getattr(value, "__admissible__", None) is None:
                    result[label] = value.bind(category, owner=base)
                else:
                    result.pop(label, None)
        return result

    def bind(self, category: type, owner: type | None = None) -> Self:
        """ Bind the declaration to a concrete category. """
        return replace(self, category=category, owner=self.owner or owner)

    @property
    def bound(self) -> type:
        """ The category the declaration is bound to. """
        if self.category is None:
            raise TypeError(f"{self.name} is not bound to a class.")
        return self.category

    def resolve(self, hom) -> type:
        """ The category a hom premise is a morphism of, by its head. """
        return Sort.stands_for(
            get_args(hom)[0], self.bound, self.bound, self.owner)

    def canonical(self) -> dict[str, Any]:
        """
        The canonical arguments of the sequent, by name: each variable is
        the canonical instance of its sort and each premise an instance
        :meth:`Sort.named` after its parameter.

        >>> from discopy.abc import MonoidalCategory
        >>> from discopy.monoidal import Diagram
        >>> mix = MonoidalCategory.mix.bind(Diagram)
        >>> for label, box in mix.canonical().items():
        ...     print(f"{label}: {box.dom} -> {box.cod}")
        self: A -> B
        other: C -> D
        """
        category, sorts = self.bound, self.variables
        counts = {
            label: sort.canonical(category, label, self.owner)
            for label, sort in sorts.items() if isinstance(sort, Counts)}
        subst = Substitution(counts)
        for label, sort in sorts.items():
            if label not in counts:
                length = counts.get(str(getattr(sort, "size", None)), 1)
                subst[label] = sort.canonical(
                    category, label, self.owner, length)
        args = {}
        for label, value in self.premises.items():
            if isinstance(value, Sort):
                args[label] = Sort.named(
                    value.resolve(category, self.owner), label)
            elif get_origin(value) is Hom:
                dom, cod = subst.instantiate(value, category.ob)
                args[label] = Sort.named(self.resolve(value), label, dom, cod)
            else:
                args[label] = subst.instantiate(value, category.ob)
        return args

    def generate(self, draw: Callable, hom: Callable,
                 subst: Substitution | None = None) -> tuple:
        """
        Sample the arguments of the sequent inside a composite strategy,
        one premise at a time: a pattern is instantiated, a sort sampled
        from the strategy of the class it ranges over, a hom or a term of
        the category sampled through ``hom(category, dom, cod)``. A
        variable of any size standing alone on a side of a hom is read
        off the term found, so that the goal guides the search; the
        residuals of a match are checked once every variable is bound.
        """
        from hypothesis import assume

        category, sorts = self.bound, self.variables
        residuals = subst.residuals if subst else ()
        subst = Substitution(subst or {})

        def side(pattern):
            if isinstance(pattern, TypeVar) and pattern.__name__ not in subst\
                    and getattr(sorts[pattern.__name__], "size", None) is None:
                return None
            bound(*Pattern.variables(pattern))
            return subst.instantiate(pattern, category.ob)

        def read_off(pattern, value):
            if isinstance(pattern, TypeVar) and pattern.__name__ not in subst:
                check(getattr(sorts[pattern.__name__], "size", None), value)
                subst[pattern.__name__] = value

        def check(length, value):
            fit = subst.fit(length, value)
            assume(fit is not None)
            subst.update(fit or {})

        def sample(sort, label):
            length = getattr(sort, "size", None)
            if isinstance(length, str):
                bound(length)
            return draw(sort.strategy(
                category, self.owner, subst.get(str(length))), label=label)

        def bound(*labels):
            for label in labels:
                if label not in subst:
                    subst[label] = sample(sorts[label], label)

        args = {}
        for label, value in self.premises.items():
            if isinstance(value, Terms) and hasattr(category, "rules"):
                args[label] = draw(hom(category, None, None), label=label)
            elif isinstance(value, Sort):
                args[label] = sample(value, label)
            elif get_origin(value) is Hom:
                _, dom_pattern, cod_pattern = get_args(value)
                dom, cod = side(dom_pattern), side(cod_pattern)
                term = draw(hom(self.resolve(value), dom, cod), label=label)
                read_off(dom_pattern, term.dom)
                read_off(cod_pattern, term.cod)
                args[label] = term
            else:
                bound(*Pattern.variables(value))
                args[label] = subst.instantiate(value, category.ob)
        for pattern, value in residuals:
            bound(*Pattern.variables(pattern))
            assume(subst.instantiate(pattern, category.ob) == value)
        return subst, args



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
    >>> print(Category.cut)
    cut(self: Hom[C1, A, B], other: Hom[C1, B, C]) -> Hom[C1, A, C]
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
                if getattr(base.__dict__.get(self.name or ""), "__func__",
                           base.__dict__.get(self.name or "")) is self),
                None)
            bound[owner] = self.bind(owner, owner=declaring)
        return bound[owner]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.function(*args, **kwargs)

    def match(self, dom=None, cod=None) -> Iterator[Substitution]:
        """ Unify the conclusion with a goal. """
        return Substitution().unify(self.conclusion, (dom, cod))

    @property
    def recursive(self) -> bool:
        """
        Whether a premise is a hom, which the search proves recursively
        below its depth bound; a rule with none is a generator, built in
        one step, see :meth:`discopy.abc.Category.generators`.

        >>> from discopy.abc import Category, RigidCategory
        >>> assert Category.cut.recursive
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
        """ The rule of one given box, see :class:`Constant`, under a
        name no module of terms takes. """
        return Constant(box)

    def inapplicable(self, reason: str) -> Self:
        """
        The same rule dropped from the rules and generators of the
        class it is assigned on, because the structure it builds lies
        outside the category's terms, with the reason as its record:
        the method still runs, the search just never applies it, e.g.
        ``trace_left = frobenius.Diagram.trace_left.inapplicable("No loop
        in a sentence.")``. A rule the category does have, whose terms
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
        the reason names as its record, e.g. the curries of a
        rigid category, which caps and cuts reach.
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

    def match(self, dom=None, cod=None) -> Iterator[Substitution]:
        box = self.function
        if dom in (None, box.dom) and cod in (None, box.cod):
            yield Substitution()

    def generate(self, draw, hom, subst=None):
        return Substitution(subst or {}), {}

    def apply(self, arguments: dict):
        return self.function


def rule[**P, T](function: Callable[P, T]) -> Rule[P, T]:
    """ Decorate a method as an inference rule, its signature the sequent. """
    return Rule(function)


from discopy import abc  # noqa: E402  pylint: disable=wrong-import-position
