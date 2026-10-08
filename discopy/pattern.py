"""
The patterns in which a category states its rules and axioms, which
:meth:`discopy.axioms.Testable.sample` samples its terms by.

A pattern ``Pattern[T]`` stands for values of type ``T`` with variables
to instantiate: ``Hom(dom, cod)`` for the morphisms between two
patterns, ``Obj()`` for the objects, ``Var(name)`` for
the value a variable is bound to, and ``Tensor(p, q)``, ``Over(z, y)``,
``Under(y, z)``, ``L(p)``, ``R(p)``, ``D(p)``, ``Repeat(x, n)`` and
``Image(f, p)`` for the objects these operations build. A ground
pattern, with no variable left, is what
:meth:`discopy.axioms.Testable.strategy` samples, e.g.
``Diagram.strategy(Hom(x, y))`` the diagrams from ``x`` to ``y``.

A sequent is the signature of a method on an abstract base class of
:mod:`discopy.abc`, written with the aliases :data:`discopy.abc.Obj`,
:data:`discopy.abc.Hom` and :data:`discopy.abc.Var` that a typechecker
reads as plain types: its :pep:`695` type parameter list is the context,
each parameter one variable with its sort as the bound — ``A: Obj[C0]``
an object, ``X: Atom[C0]`` a single wire and ``N: Count`` a number —
its parameters the premises
and its return annotation the conclusion, e.g.

.. code-block:: python

    def tensor[A: Obj[C0], B: Obj[C0], C: Obj[C0], D: Obj[C0]](
            self: Hom[C1, A, B], other: Hom[C1, C, D]
    ) -> Hom[C1, Tensor[A, C], Tensor[B, D]]:
        ...

A :class:`Declaration` reads its sequent off :func:`inspect.signature`,
i.e. the annotations as :pep:`649` evaluates them, into patterns by
:meth:`Pattern.read`: a premise ``Hom[C1, A, B]`` reads as
``Hom(Var('A'), Var('B'), head=C1)``, the head saying which category it
is a morphism of, and a premise ``Self`` as ``Hom(head=Self)``, any term
of the category. The type parameters of a pattern class are bounded by
the structure the objects it stands for need, e.g. ``D[T:
DelayedMonoid]``, so that a typechecker refuses a delay in a sequent
whose objects are not delayed.

A conclusion is matched against a goal, a ground ``Hom`` whose sides
may be :obj:`None` for any, by unification over the free monoid of
objects: a ``Tensor`` splits the goal at every position, a variable binds
once, an adjoint ``R(p)`` inverts to ``p``. What cannot be inverted, an
exponential that collapsed into adjoints, is a residual equation checked
once every variable is instantiated.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Substitution
    Pattern
    Var
    Obj
    Count
    Hom
    Tensor
    Exponential
    Over
    Under
    L
    R
    D
    Repeat
    Image
    Declaration
    Rule

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
from dataclasses import KW_ONLY, dataclass, fields, replace
from functools import reduce
from types import MethodType
from typing import (
    Any, ClassVar, Generic, Literal, Self, TypeVar, get_args, get_origin)

from discopy.utils import factory_name


class Substitution(dict[str, Any]):
    """
    The values of the variables of patterns, by name, with the
    ``residuals`` unification could not invert: pairs of a pattern and the
    value it must instantiate to, checked once its variables are bound.

    >>> from discopy.monoidal import Ty
    >>> x, y = Ty('x'), Ty('y')
    >>> for subst in Substitution().unify(Tensor(Var('A'), Var('B')), x @ y):
    ...     print(subst['A'], '|', subst['B'])
    Ty() | x @ y
    x | y
    x @ y | Ty()
    """

    def __init__(self, *args, residuals: tuple = (), **kwargs):
        super().__init__(*args, **kwargs)
        self.residuals = tuple(residuals)

    def __repr__(self):
        return f"Substitution({dict(self)!r}, residuals={self.residuals!r})"

    def bind(self, **values) -> Substitution:
        """ The substitution with more values. """
        return Substitution(self, residuals=self.residuals, **values)

    def residual(self, pattern: Pattern, value) -> Substitution:
        """ The substitution with one more residual equation. """
        return Substitution(
            self, residuals=self.residuals + ((pattern, value), ))

    def instantiate(self, pattern, ob: Any):
        """ The value a pattern stands for, ``ob`` the type of objects
        called to build the unit: anything but a pattern is a value, its
        own pattern, and a sort is instantiated to a ground pattern. """
        if isinstance(pattern, Pattern):
            return pattern.instantiate(self, ob)
        return pattern

    def unify(self, pattern, value) -> Iterator[Substitution]:
        """
        Every substitution extending this one under which a pattern stands
        for a value: anything at all when either is :obj:`None`, the value
        itself for anything but a pattern.
        """
        if value is None or pattern is None:
            yield self
        elif isinstance(pattern, Pattern):
            yield from pattern.unify(value, self)
        elif pattern == value:
            yield self


@dataclass(frozen=True)
class Pattern[T](ABC):
    """
    A pattern standing for values of type ``T``, with variables to
    instantiate: a frozen dataclass whose fields are patterns or values.
    """

    @property
    def variables(self) -> tuple[str, ...]:
        """ The names of the variables of the pattern, in order. """
        values = [getattr(self, each.name) for each in fields(self)]
        parts = [
            part for value in values
            for part in (value if isinstance(value, tuple) else (value, ))]
        return tuple(
            label for part in parts if isinstance(part, Pattern)
            for label in part.variables)

    @abstractmethod
    def instantiate(self, subst: Substitution, ob) -> T:
        """ The value the pattern stands for under a substitution. """

    @abstractmethod
    def unify(self, value: T, subst: Substitution) -> Iterator[Substitution]:
        """ Every substitution extending ``subst`` under which the pattern
        stands for a value. """

    @staticmethod
    def read(annotation) -> Any:
        """
        The pattern a sequent states in an annotation: a type parameter is
        a :class:`Var` of the sort its bound reads as, ``Self`` any term,
        ``Count`` any number, the aliases of :mod:`discopy.abc` and the
        subscripts of a pattern class the instances they stand for, and
        anything else a value.

        >>> from discopy.abc import Atom, Hom, Obj
        >>> from discopy.monoidal import Ty, Diagram
        >>> def spiders[N: Count, X: Atom[Ty]](): ...
        >>> print(*map(Pattern.read, spiders.__type_params__))
        N X
        >>> def cups[X: Atom[Ty]](f: Hom[Diagram, Tensor[X, R[X]], X]): ...
        >>> print(Pattern.read(cups.__annotations__['f']))
        Hom[Diagram, Tensor[X, R[X]], X]
        """
        if isinstance(annotation, TypeVar):
            sort = Pattern.read(annotation.__bound__)
            return Var(annotation.__name__, sort if isinstance(sort, Pattern)
                       else Obj(sort))
        if annotation is Count:
            return Count()
        if annotation is Self:
            return Hom(head=Self)
        if annotation is abc.Atom:
            return Obj(size=1)
        origin, args = get_origin(annotation), get_args(annotation)
        if origin is Literal:
            return args[0]
        if origin is abc.Hom:
            return Hom(*map(Pattern.read, args[1:]), head=args[0])
        if origin is abc.Var:
            return Pattern.read(args[1])
        if origin in (abc.Atom, abc.Unit):
            return Obj(args[0], 1 if origin is abc.Atom else 0)
        if origin is abc.Obj:
            return Obj(args[0], *map(Pattern.read, args[1:]))
        if isinstance(origin, type) and issubclass(origin, Pattern):
            return origin(*map(Pattern.read, args))
        return annotation

    @staticmethod
    def heading(head) -> str:
        """ The name of the head of a pattern, as a sequent spells it. """
        if head is Self:
            return "Self"
        return getattr(head, "__name__", None) or "_"


@dataclass(frozen=True)
class Var[T](Pattern[T]):
    """
    The variable ``name``, bound once to a value of its ``sort``.

    >>> from discopy.monoidal import Ty
    >>> subst, = Substitution().unify(Var('X', Obj(size=1)), Ty('x'))
    >>> assert subst == {'X': Ty('x')}
    >>> assert not list(Substitution().unify(Var('X', Obj(size=1)), Ty()))
    """
    name: str
    sort: Pattern[T] | None = None

    @property
    def variables(self):
        return (self.name, )

    def instantiate(self, subst, ob):
        return subst[self.name]

    def unify(self, value, subst):
        if self.name in subst:
            if subst[self.name] == value:
                yield subst
            return
        for fit in subst.unify(self.sort, value):
            yield fit.bind(**{self.name: value})

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Obj[T](Pattern[T]):
    """
    The objects of a ``head``, a class or a type parameter, of a
    ``size``: :obj:`None` for any, one for a single wire and zero for the
    unit, the one object instantiating it. The head :obj:`None` stands for
    the objects of the category.

    >>> from discopy.monoidal import Ty
    >>> assert Obj(size=0).instantiate(Substitution(), Ty) == Ty()
    >>> assert Obj(size=1).instantiate(Substitution(), Ty) == Obj(size=1)
    """
    head: Any = None
    size: Literal[0, 1] | None = None

    def instantiate(self, subst, ob):
        return ob() if self.size == 0 else self

    def unify(self, value, subst):
        if self.size is None or len(value) == self.size:
            yield subst

    def __str__(self):
        head = self.head and self.heading(self.head)
        if self.size is None:
            return f"Obj[{head}]" if head else "Obj"
        if self.size == 1:
            return f"Atom[{head}]" if head else "Atom"
        return f"Unit[{head}]" if head else "Unit"


@dataclass(frozen=True)
class Count(Pattern[int]):
    """ The numbers, e.g. of repetitions, as the bound ``N: Count`` of a
    variable states them, up to :attr:`maximum` when sampled. """
    maximum: ClassVar[int] = 3

    def instantiate(self, subst, ob):
        return self

    def unify(self, value, subst):
        yield subst

    def __str__(self):
        return "Count"


@dataclass(frozen=True)
class Hom[T](Pattern[T]):
    """
    The morphisms from ``dom`` to ``cod``, patterns or values,
    :obj:`None` for any, of a ``head``: ``Self`` or :obj:`None` for the
    category, else a class or a type parameter. Matched against a goal,
    another ``Hom``, side by side.

    >>> from discopy.monoidal import Ty
    >>> x = Ty('x')
    >>> subst, = Substitution().unify(Hom(Var('A'), Var('A')), Hom(x, None))
    >>> assert subst == {'A': x}
    """
    dom: Any = None
    cod: Any = None
    _: KW_ONLY
    head: Any = None

    def instantiate(self, subst, ob):
        return replace(self, dom=subst.instantiate(self.dom, ob),
                       cod=subst.instantiate(self.cod, ob))

    def unify(self, value, subst):
        for unified in subst.unify(self.dom, value.dom):
            yield from unified.unify(self.cod, value.cod)

    def __str__(self):
        if self.head is Self and self.dom is self.cod is None:
            return "Self"
        return f"Hom[{self.heading(self.head)}, {self.dom}, {self.cod}]"


@dataclass(frozen=True, init=False)
class Tensor[*Ts](Pattern):
    """ The tensor ``Tensor(p, q, ...)`` of two or more patterns,
    matched by splitting the value at every position. """
    parts: tuple

    def __init__(self, *parts):
        object.__setattr__(self, "parts", parts)

    def instantiate(self, subst, ob):
        return reduce(operator.matmul, (
            subst.instantiate(part, ob) for part in self.parts))

    def unify(self, value, subst):
        if not self.parts:
            if not len(value):
                yield subst
            return
        head, *tail = self.parts
        for n in range(len(value) + 1):
            for unified in subst.unify(head, value[:n]):
                yield from Tensor(*tail).unify(value[n:], unified)

    def __str__(self):
        return f"Tensor[{', '.join(map(str, self.parts))}]"


@dataclass(frozen=True)
class Exponential(Pattern):
    """
    The exponentials :class:`Over` and :class:`Under` of a ``base`` and
    an ``exponent``, matched with a single exponential object that its
    base and exponent rebuild and nothing else — except at a level that
    collapses its exponentials into the adjoints of a pregroup, which
    keeps the residual.
    """
    left: ClassVar[bool]
    base: Any
    exponent: Any

    def build(self, base, exponent):
        """ The exponential object of a base and an exponent. """
        return base << exponent if self.left else exponent >> base

    def instantiate(self, subst, ob):
        return self.build(subst.instantiate(self.base, ob),
                          subst.instantiate(self.exponent, ob))

    def unify(self, value, subst):
        atom = value.inside[0] if len(value) == 1 else None
        found: Any = (getattr(atom, "base", None),
                      getattr(atom, "exponent", None))
        if None in found or value != self.build(*found):
            if hasattr(value, "r"):  # A pregroup: the residual stays.
                yield subst.residual(self, value)
            return
        for unified in subst.unify(self.base, found[0]):
            yield from unified.unify(self.exponent, found[1])

    def __str__(self):
        sides = (self.base, self.exponent) if self.left\
            else (self.exponent, self.base)
        return f"{type(self).__name__}[{sides[0]}, {sides[1]}]"


@dataclass(frozen=True, init=False)
class Over[Z: abc.ResiduatedMonoid, Y: abc.ResiduatedMonoid](Exponential):
    """ The exponential ``Over(z, y)``, i.e. ``z << y``. """
    left = True

    def __init__(self, base, exponent):
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "exponent", exponent)


@dataclass(frozen=True, init=False)
class Under[Y: abc.ResiduatedMonoid, Z: abc.ResiduatedMonoid](Exponential):
    """ The exponential ``Under(y, z)``, i.e. ``y >> z``. """
    left = False

    def __init__(self, exponent, base):
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "exponent", exponent)


@dataclass(frozen=True)
class L[T: abc.Pregroup](Pattern):
    """ The left adjoint ``L(p)``, i.e. ``p.l``, inverted by the right
    adjoint when matching. """
    base: Any

    def instantiate(self, subst, ob):
        return subst.instantiate(self.base, ob).l

    def unify(self, value, subst):
        yield from subst.unify(self.base, value.r)

    def __str__(self):
        return f"L[{self.base}]"


@dataclass(frozen=True)
class R[T: abc.Pregroup](Pattern):
    """ The right adjoint ``R(p)``, i.e. ``p.r``, inverted by the left
    adjoint when matching. """
    base: Any

    def instantiate(self, subst, ob):
        return subst.instantiate(self.base, ob).r

    def unify(self, value, subst):
        yield from subst.unify(self.base, value.l)

    def __str__(self):
        return f"R[{self.base}]"


@dataclass(frozen=True)
class D[T: abc.DelayedMonoid](Pattern):
    """ The delay ``D(p)`` of a pattern by one time step, inverted by the
    delay ``-1`` steps back when matching. """
    base: Any

    def instantiate(self, subst, ob):
        return subst.instantiate(self.base, ob).d

    def unify(self, value, subst):
        try:
            undelayed = value.delay(-1)
        except NotImplementedError:  # Not the delay of anything.
            return
        if undelayed.d == value:
            yield from subst.unify(self.base, undelayed)

    def __str__(self):
        return f"D[{self.base}]"


@dataclass(frozen=True)
class Repeat[X: abc.ColouredMonoid, N: Count](Pattern):
    """ A single wire repeated a number of ``times``, a :class:`Var` of
    sort :class:`Count`, for the legs of a spider: matching binds the
    count to the number of wires and the base to the one wire they all
    equal. """
    base: Any
    times: Var

    def instantiate(self, subst, ob):
        return subst.instantiate(self.base, ob) ** subst.instantiate(
            self.times, ob)

    def unify(self, value, subst):
        atoms = [value[i:i + 1] for i in range(len(value))]
        if any(atom != atoms[0] for atom in atoms[1:]):
            return
        label = self.times.name
        if subst.get(label, len(value)) != len(value):
            return
        fit = subst.bind(**{label: len(value)})
        if atoms:
            yield from fit.unify(self.base, atoms[0])
        else:
            yield fit

    def __str__(self):
        return f"Repeat[{self.base}, {self.times}]"


@dataclass(frozen=True)
class Image[F, X](Pattern):
    """ The image ``Image(f, p)`` of a pattern under a functor, ``f`` the
    variable a premise ``functor: Var[Self, F]`` binds: instantiating
    applies the functor, which matching cannot invert, so the image is
    compared once the functor and the pattern are bound and is a
    residual until then. """
    functor: Any
    base: Any

    def instantiate(self, subst, ob):
        return subst.instantiate(self.functor, ob)(
            subst.instantiate(self.base, ob))

    def unify(self, value, subst):
        if not all(label in subst for label in self.variables):
            yield subst.residual(self, value)
        elif self.instantiate(subst, type(value)) == value:
            yield subst

    def __str__(self):
        return f"Image[{self.functor}, {self.base}]"


@dataclass(repr=False)
class Declaration[**P, T]:
    """
    A declaration is a sequent stated by a ``function`` on an abstract
    base class and inherited by every category below it: the base of the
    rules of :meth:`discopy.axioms.Testable.sample` and of the axioms of
    :mod:`discopy.axioms`. The ``category`` is the class the
    declaration is bound to, ``name`` the attribute it is stored under
    and ``owner`` the class declaring the sequent.

    >>> from discopy.abc import Category
    >>> Category.cut
    abc.Category.cut
    >>> print(Category.cut.conclusion)
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
    def variables(self) -> dict[str, Pattern]:
        """ The type parameters of the function and their sorts, each the
        pattern its bound reads as. """
        return {
            variable.__name__: Pattern.read(variable).sort
            for variable in getattr(
                inspect.unwrap(self.function), "__type_params__", ())}

    @property
    def premises(self) -> dict[str, Any]:
        """ The parameters stating a premise, by name, each the pattern
        its annotation reads as: a sort ``Hom``, ``Obj`` or ``Count`` to
        sample, anything else the value of its variables. """
        return {
            parameter.name: Pattern.read(parameter.annotation)
            for parameter in inspect.signature(
                self.function).parameters.values()
            if parameter.kind is not parameter.VAR_KEYWORD
            and (parameter.annotation is Self or get_origin(
                parameter.annotation) in (
                    abc.Hom, abc.Var, abc.Obj, abc.Atom, abc.Unit))}

    @property
    def conclusion(self) -> Pattern | None:
        """ The return annotation, when the declaration concludes. """
        if not self.concludes:
            return None
        return Pattern.read(
            inspect.signature(self.function).return_annotation)

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
        >>> assert Declaration.position(
        ...     Monoid, Monoid.__type_params__[0]) == 1
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

    def resolve(self, pattern: Obj | Hom) -> type:
        """
        The type a sort ranges over in the category the declaration is
        bound to, by its head: the category itself for ``Self``, what
        :meth:`discopy.abc.Category.parameters` says of a type parameter
        of the class declaring the sequent, by its :meth:`position`, and
        for a class the category's own subclass of it when it has one,
        e.g. the objects of a symmetric diagram for ``monoidal.Ty``. The
        head :obj:`None` stands for the objects of the category or the
        category itself, as the pattern is an :class:`Obj` or a
        :class:`Hom`.

        >>> from discopy import monoidal, symmetric
        >>> from discopy.abc import Category
        >>> cut = Category.cut.bind(symmetric.Diagram)
        >>> assert cut.resolve(Obj(monoidal.Ty)) is symmetric.Diagram.ob
        """
        category, head = self.bound, pattern.head
        if head is Self:
            return category
        if head is None or head is Any:
            return category.ob if isinstance(pattern, Obj) else category
        if isinstance(head, TypeVar):
            return category.parameters()[
                self.position(self.owner or category, head)]
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
        >>> assert Declaration.named(Diagram, 'f')\\
        ...     == Box('f', Ty('x'), Ty('y'))
        >>> assert Declaration.named(Ty, 'A') == Ty('A')
        """
        box = getattr(factory, "Box", None)
        if box is not None and isinstance(box, type):
            dom = factory.ob("x") if dom is None else dom
            cod = factory.ob("y") if cod is None else cod
            return box(label, dom, cod)
        return factory(label)

    def context(self) -> Substitution:
        """
        The canonical values of the variables of the sequent: each number
        is two and each object a wire named after its variable.

        >>> from discopy.abc import MonoidalCategory
        >>> from discopy.monoidal import Diagram
        >>> print(*MonoidalCategory.mix.bind(Diagram).context().values())
        A B C D
        """
        subst = Substitution()
        for label, sort in self.variables.items():
            ground = subst.instantiate(sort, self.bound.ob)
            if isinstance(ground, Count):
                subst[label] = 2
            elif isinstance(ground, Obj):
                subst[label] = self.named(self.resolve(ground), label)
            else:
                subst[label] = ground
        return subst

    def canonical(self) -> dict[str, Any]:
        """
        The canonical arguments of the sequent, by name, in its
        :meth:`context`: each premise an instance :meth:`named` after its
        parameter.

        >>> from discopy.abc import MonoidalCategory
        >>> from discopy.monoidal import Diagram
        >>> mix = MonoidalCategory.mix.bind(Diagram)
        >>> for label, box in mix.canonical().items():
        ...     print(f"{label}: {box.dom} -> {box.cod}")
        self: A -> B
        other: C -> D
        """
        subst, args = self.context(), {}
        for label, premise in self.premises.items():
            ground = subst.instantiate(premise, self.bound.ob)
            if isinstance(ground, (Obj, Hom)):
                ground = self.named(self.resolve(ground), label, *(
                    (ground.dom, ground.cod) if isinstance(ground, Hom)
                    else ()))
            args[label] = ground
        return args


@dataclass(repr=False)
class Rule[**P, T](Declaration[P, T]):
    """
    An inference rule of a category, a
    :class:`Declaration` with a conclusion: every rule
    states its sequent as its own signature and
    :meth:`discopy.axioms.Testable.sample` calls
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

    def match(self, goal: Hom) -> Iterator[Substitution]:
        """ Unify the conclusion with a goal. """
        return Substitution().unify(self.conclusion, goal)

    @property
    def recursive(self) -> bool:
        """
        Whether a premise is a hom, which sampling proves recursively
        below its depth bound; a rule with none is a generator, built in
        one step.

        >>> from discopy.abc import Category, RigidCategory
        >>> assert Category.cut.recursive
        >>> assert not RigidCategory.cups.recursive
        """
        return any(
            isinstance(premise, Hom) for premise in self.premises.values())

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

    def inapplicable(self, reason: str) -> Self:
        """
        The same rule dropped from the rules of the class it is
        assigned on, because the structure it builds lies outside the
        category's terms, with the reason as its record: the method
        still runs, sampling just never applies it, e.g. ``dagger =
        monoidal.Diagram.dagger.inapplicable("Rigid types have no
        dagger.")``. A rule the category does have, whose terms other
        rules reach, is :meth:`admissible` instead.
        """
        result = replace(self)
        result.__inapplicable__ = reason
        return result

    def admissible(self, reason: str) -> Self:
        """
        The same rule dropped from the rules of the class it is
        assigned on, because it is `admissible
        <https://en.wikipedia.org/wiki/Admissible_rule>`_: sampling
        reaches everything it builds through the other rules, which
        the reason names as its record, e.g. the curries of a
        rigid category, which caps and cuts reach.
        The method still runs and remains applicable.
        """
        result = replace(self)
        result.__admissible__ = reason
        return result


def rule[**P, T](function: Callable[P, T]) -> Rule[P, T]:
    """ Decorate a method as an inference rule, its signature the sequent. """
    return Rule(function)


from discopy import abc  # noqa: E402  pylint: disable=wrong-import-position
