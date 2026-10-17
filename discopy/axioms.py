"""
Property-based testing of the axioms with `Hypothesis
<https://hypothesis.readthedocs.io>`_.

An :class:`Axiom` is an equation stated once on an abstract base class of
:mod:`discopy.abc` and inherited by every category below it, its signature a
sequent: its type parameters are variables, e.g. objects ``A: Obj[C0]``, its
parameters premises, e.g. a morphism ``f: Hom[C1, A, B]`` from ``A`` to
``B``, and its return annotation the type of its output, e.g. an equation
``Equation[Hom[C1, A, B]]``, which every value can be checked against, see
:func:`inhabits`. A :class:`Rule` is stated the same way, an operation of
the structure with a morphism as conclusion:
the terms of a free category are those its rules derive from boxes, see
:meth:`discopy.cat.Arrow.search`, and from symmetric categories on also
those of the combinatorial representation their equations compare by, see
:meth:`discopy.symmetric.Diagram.strategy`. Every strategy of a free category
honours any boundary it is asked for.

A law a category breaks is declared :meth:`Axiom.failing` where it breaks,
one that does not apply :meth:`Axiom.inapplicable`, and one that holds up to
a quotient or on a subspace :meth:`Axiom.modulo` and :meth:`Axiom.weaken`.
A :class:`Testable` category generates its own terms and
:meth:`Testable.matrix` lists every law of every such category, which the
suite in ``proptest/`` checks; CONTRIBUTING.md says how to run it. A
:class:`Serialisable` type states the laws of writing a term down and
reading it back, and an :class:`Equivalence` those of an equivalence
relation besides equality: reflexivity, symmetry and transitivity.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Equation
    Declaration
    Axiom
    Rule
    AxiomFailure
    Count
    Tensor
    Unit
    L
    R
    Delay
    Over
    Under
    Repeat
    Testable
    Serialisable
    Equivalence
    Related

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        axiom
        rule
        inhabits
"""

import inspect
import pickle
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, replace
from functools import cached_property, reduce, wraps
from operator import matmul
from types import GenericAlias
from typing import (
    TYPE_CHECKING, Annotated, Any, ClassVar, Concatenate, ParamSpec, Self,
    TypeVar,
    TypeVarTuple,
    get_args, get_origin)

from discopy.utils import (
    AxiomError,
    NamedGeneric,
    classproperty,
    dumps,
    factory_name,
    from_tree,
    loads,
)

if TYPE_CHECKING:
    from hypothesis import strategies as st


GENERATORS = tuple("abcde")
"""
The names the generators of a free category are sampled from.

They are finitely many and shared, so a generated functor can name every one
of them: composing two functors keeps only the keys of the left-hand map, so
a functor that named just a few would compose to one defined nowhere else.
"""

MAX_COUNT = 3
""" The largest number a declaration quantifying over ``int`` is given. """

FUEL = 3
""" How deep the rules nest in a term of a free category, by default. """


class Equation[T]:
    """
    An equation is a list of ``terms`` to be compared up to a function
    ``up_to``, the identity by default.  Casting it to ``bool`` checks
    whether its terms are all equal up to that function.

    Parameters:
        terms : The terms of the equation.
        symbol : The symbol between each pair of terms, ``"="`` by default.
        symbols : The symbols between each pair of consecutive terms, one
            fewer than the terms, overriding ``symbol``.
        up_to : The function up to which ``bool(equation)`` compares its
            terms, overriding the subclass' :attr:`up_to` if given.

    Example
    -------
    The number of boxes inside an arrow is left unchanged by associativity,
    so we can compare arrows up to the function that counts them modulo 2:

    >>> from discopy.cat import Ob, Box, Equation
    >>> x = Ob('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> parity = lambda term: len(term.inside) % 2
    >>> assert not Equation(f, f >> g >> g)
    >>> assert Equation(f, f >> g >> g, up_to=parity)
    """
    up_to = None

    def __init__(self, *terms, symbol="=", symbols=None, up_to=None):
        self.terms = terms
        gaps = max(len(terms) - 1, 0)
        self.symbols = gaps * (symbol, ) if symbols is None\
            else tuple(symbols)
        if len(self.symbols) != gaps:
            raise ValueError(
                f"Expected {gaps} symbols between {len(terms)} terms, "
                f"got {len(self.symbols)}.")
        if up_to is not None:
            self.up_to = up_to

    def modulo(self, up_to: Callable) -> "Equation":
        """
        The same equation compared up to the given function, rebinding
        :attr:`up_to`, whose name the attribute already takes.

        >>> from discopy.cat import Ob, Box, Equation
        >>> x = Ob('x')
        >>> f, g = Box('f', x, x), Box('g', x, x)
        >>> assert Equation(f >> g, g >> f).modulo(lambda _: True)
        """
        return type(self)(*self.terms, symbols=self.symbols, up_to=up_to)

    def __repr__(self):
        """
        >>> from discopy.cat import Ob, Box, Equation
        >>> Equation(Box('f', Ob('x'), Ob('x')))
        cat.Equation(cat.Box('f', cat.Ob('x'), cat.Ob('x')))
        """
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self.terms))})"

    def __str__(self):
        return f"Equation({', '.join(map(str, self.terms))})"

    def __bool__(self):
        terms = self.terms if self.up_to is None\
            else list(map(self.up_to, self.terms))
        return all(term == terms[0] for term in terms)


class AxiomFailure(AxiomError):
    """
    A law declared broken, raised when the bound axiom is called: the
    reason is the message and :attr:`equation` is the law evaluated on the
    arguments, whose sides say how it failed, or the error raised when its
    terms could not be built.
    """

    def __init__(self, reason: str, equation):
        super().__init__(reason, equation)
        self.equation = equation


type Obj[Coarse] = Annotated[Coarse, "object"]
"""
The sort of the objects of type ``Coarse``, as the bound of a variable
``A: Obj[C0]`` of a declaration. A typechecker reads it as ``Coarse``.
"""

type Atom[Coarse] = Annotated[Coarse, "atom"]
""" The sort of the atomic objects, i.e. a single wire, ``X: Atom[C0]``. """

type Var[Coarse, Value] = Annotated[Coarse, Value]
"""
The premise standing for the value of an expression over the variables of a
declaration, e.g. ``x: Var[C0, X]`` for ``X: Obj[C0]``, ``mem: Var[C0,
Tensor[M, X]]`` or ``n: Var[int, N]`` for ``N: Count``. A typechecker reads
it as ``Coarse``.
"""

type Hom[Coarse, Dom, Cod] = Annotated[Coarse, Dom, Cod]
"""
The morphisms of type ``Coarse`` from ``Dom`` to ``Cod``, as a premise to
sample, e.g. ``f: Hom[C1, A, B]``, or the conclusion of a rule. A typechecker
reads it as ``Coarse``.
"""


class Count(int):
    """
    The sort of the numbers up to :data:`MAX_COUNT`, as the bound of a
    variable ``N: Count`` of a declaration, e.g. the legs of a spider, two
    in the canonical instance of a law, the least that is neither the unit
    nor a single wire.
    """
    canonical = 2


def is_open(annotation) -> bool:
    """
    Whether an annotation has a variable left in it, i.e. a type parameter
    of a declaration or of the class stating it.

    >>> X = TypeVar("X")
    >>> assert is_open(Hom[None, X, X]) and not is_open(Hom[None, 1, 1])
    """
    if isinstance(annotation, TypeVar | TypeVarTuple | ParamSpec):
        return True
    return any(map(is_open, get_args(annotation)))


def substitute(annotation, scope: dict):
    """
    The annotation with each variable of the scope replaced by its value,
    from the leaves up, so that an object expression is computed once its
    variables are, see :class:`Former`.

    >>> from discopy.monoidal import Ty
    >>> A, B = TypeVar("A"), TypeVar("B")
    >>> x, y = Ty('x'), Ty('y')
    >>> assert substitute(Tensor[A, B], {A: x, B: y}) == x @ y
    """
    try:
        if annotation in scope:
            return scope[annotation]
    except TypeError:
        return annotation
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is None or not args:
        return annotation
    args = tuple(substitute(arg, scope) for arg in args)
    return origin[args if len(args) > 1 else args[0]]


class Former:
    """
    A former of object expressions in a signature, e.g. ``Tensor[A, B]``: a
    type for a typechecker, an expression while a variable is left in it,
    see :func:`is_open`, and the object :meth:`form` builds once there is
    none.
    """
    def __class_getitem__(cls, args):
        args = args if isinstance(args, tuple) else (args, )
        if any(map(is_open, args)):
            return GenericAlias(cls, args)
        return cls.form(*args)

    @staticmethod
    def form(*args):
        """ The object the former builds from objects, to be instantiated. """
        raise NotImplementedError


class Tensor[*Objects](Former):
    """
    The tensor of objects in a signature: ``Tensor[A, B]`` is a type for a
    typechecker and the object ``A @ B`` once the variables are sampled.

    >>> from discopy.monoidal import Ty
    >>> assert Tensor[Ty('x'), Ty('y')] == Ty('x', 'y')
    """
    @staticmethod
    def form(*objects):
        return reduce(matmul, objects)


class Unit[Coarse](Former):
    """
    The unit of a monoid of objects in a signature: ``Unit[C0]`` is the
    object ``C0()``.

    >>> from discopy.monoidal import Ty
    >>> assert Unit[Ty] == Ty()
    """
    @staticmethod
    def form(*args):
        ob, = args
        return ob()


class L[Object](Former):
    """ The left adjoint of an object in a signature, ``x.l``. """
    @staticmethod
    def form(*args):
        obj, = args
        return obj.l


class R[Object](Former):
    """ The right adjoint of an object in a signature, ``x.r``. """
    @staticmethod
    def form(*args):
        obj, = args
        return obj.r


class Delay[Object](Former):
    """
    The delay of an object by one time step in a signature, ``M.delay()``.

    >>> from discopy.feedback import Ty
    >>> assert Delay[Ty('x')] == Ty('x').delay()
    """
    @staticmethod
    def form(*args):
        obj, = args
        return obj.delay()


class Over[Base, Exponent](Former):
    """ The exponential on the left in a signature, ``base << exponent``. """
    @staticmethod
    def form(*args):
        base, exponent = args
        return base << exponent


class Under[Exponent, Base](Former):
    """ The exponential on the right in a signature, ``exponent >> base``. """
    @staticmethod
    def form(*args):
        exponent, base = args
        return exponent >> base


class Repeat[Object, N](Former):
    """
    An object repeated a number of times in a signature, ``x ** n``.

    >>> from discopy.monoidal import Ty
    >>> assert Repeat[Ty('x'), 2] == Ty('x', 'x')
    """
    @staticmethod
    def form(*args):
        obj, n = args
        return obj ** n


def position(variable: TypeVar, category: type) -> int | None:
    """
    The position of a type parameter of a class above a category among
    those of the class that says what they stand for, i.e. that defines
    ``parameters``, following each generic base that passes the parameter
    on, e.g. the ``C1`` of a ``Monoid`` is the second parameter of a
    ``Category``, since a monoid is a ``ColouredMonoid[NoneType, C1]``.
    :obj:`None` for a parameter that is not passed on to such a class.

    >>> from discopy.abc import Category, Monoid
    >>> assert position(Monoid.__type_params__[0], Monoid) == 1
    >>> assert position(Category.__type_params__[0], Category) == 0
    """
    owner = next((
        klass for klass in category.__mro__
        if variable in getattr(klass, "__type_params__", ())), None)
    while owner is not None and "parameters" not in vars(owner):
        base = next((
            base for base in vars(owner).get("__orig_bases__", ())
            if variable in get_args(base)
            and getattr(get_origin(base), "__type_params__", ())), None)
        if base is None:
            return None
        owner, variable = get_origin(base), get_origin(base).__type_params__[
            get_args(base).index(variable)]
    return None if owner is None else owner.__type_params__.index(variable)


def inhabits(value, kind) -> bool:
    """
    Whether a value has an evaluated type of a sequent: a morphism of
    ``Hom[C, dom, cod]`` has that boundary, the value of ``Var[T, x]`` is
    ``x``, an equation of ``Equation[T]`` has terms of type ``T``, a tuple of
    ``tuple[T, ...]`` has its elements of these types and ``type[T]`` is a
    subclass of ``T``.

    >>> from discopy.monoidal import Ty, Box, Diagram
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> assert inhabits(f, Hom[Diagram, x, y])
    >>> assert not inhabits(f, Hom[Diagram, y, x])
    >>> assert inhabits(Equation(f, f), Equation[Hom[Diagram, x, y]])
    """
    origin, args = get_origin(kind), get_args(kind)
    if origin is Hom:
        return isinstance(value, args[0])\
            and (value.dom, value.cod) == args[1:]
    if origin is Var:
        return value == args[1]
    if origin is Equation:
        return all(inhabits(term, args[0]) for term in value.terms)
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if len(args) == 2 and args[1] is Ellipsis:
            return all(inhabits(item, args[0]) for item in value)
        return len(value) == len(args) and all(map(inhabits, value, args))
    if origin is type:
        return isinstance(value, type) and issubclass(value, args[0])
    return isinstance(value, kind)


@dataclass
class Declaration[**P, T](classmethod):
    """
    A declaration of a category, stated once on an abstract base class and
    inherited by every category below it: an :class:`Axiom` or a
    :class:`Rule`.

    The category is the class the declaration is bound to, e.g.
    :class:`discopy.cat.Arrow` for those of :class:`discopy.abc.Category`.
    Its function is a classmethod of it, implicitly: its first parameter is
    the category and the others are its arguments, sampled in order. Its
    signature is a sequent: each type parameter is a variable of the sort
    its bound states, ``A: Obj[C0]`` an object, ``X: Atom[C0]`` an atomic
    one and ``N: Count`` a number, each parameter a premise, ``f: Hom[C1, A,
    B]`` a morphism from ``A`` to ``B``, ``x: Var[C0, X]`` the value of an
    object expression and :data:`typing.Self` a term, and the return
    annotation the type of its output, see :meth:`output`. Object
    expressions are built by :class:`Tensor`, :class:`Unit`, :class:`L`,
    :class:`R`, :class:`Delay`, :class:`Over`, :class:`Under` and
    :class:`Repeat`, which stay expressions until the variables are
    substituted, see :func:`substitute`. The annotations are read as
    objects, the type parameters of the classes above the category standing
    for what :meth:`discopy.abc.Category.parameters` says, see
    :func:`position`: a function compiled with ``from __future__ import
    annotations`` has strings for annotations, and is refused.

    Parameters:
        function : The function the declaration states, from the category and
            its arguments.
        category : The class the declaration is bound to, :obj:`None` until
            :meth:`bind` or the attribute access on a class binds it.
        name : The attribute the declaration is stored under, the name of the
            function by default.
        params : The parameters passed to the strategy of every morphism the
            declaration quantifies over, see :meth:`Axiom.weaken`.
    """

    function: Callable[Concatenate[Any, P], T]
    _: KW_ONLY
    category: type = None  # ty: ignore[invalid-assignment]
    name: str = None  # ty: ignore[invalid-assignment]
    params: dict[str, Any] = None  # ty: ignore[invalid-assignment]

    def __post_init__(self):
        function = inspect.unwrap(self.function)
        if any(isinstance(annotation, str)
               for annotation in function.__annotations__.values()):
            raise TypeError(
                f"{function.__module__} states {function.__name__} with "
                "string annotations, e.g. under `from __future__ import "
                "annotations`.")
        classmethod.__init__(self, self.function)
        self.name = self.name or self.function.__name__
        self.params = dict(self.params or {})
        self.__doc__ = self.function.__doc__

    def __repr__(self):
        """
        A bound declaration is the attribute of its category, e.g.
        ``cat.Arrow.unitality``; an unbound one wraps a function and has no
        transparent representation.
        """
        if self.category is None:
            return f"{type(self).__name__}({self.name})"
        return f"{factory_name(self.category)}.{self.name}"

    def __hash__(self):
        return hash((self.function, self.category, self.name))

    def bind(self, category: type[T]) -> Self:
        """ Bind the declaration to a concrete category. """
        return replace(self, category=category)

    def __get__(self, instance, owner: type[T]) -> Self:
        return self.bind(owner)

    @property
    def parameters(self) -> tuple[inspect.Parameter, ...]:
        """
        The parameters whose arguments are generated: all but the first,
        which is the category, and those with a default.
        """
        return tuple(
            parameter for parameter in tuple(
                inspect.signature(self.function).parameters.values())[1:]
            if parameter.default is inspect.Parameter.empty)

    def scope(self, variable: Callable, given: dict | None = None) -> dict:
        """
        The values of the variables of a signature: the category for
        :data:`typing.Self`, what each type parameter of a class above it
        stands for, see :func:`position`, and then ``variable(name, sort)``
        for each type parameter of the declaration not ``given`` by name,
        the sort being its bound.
        """
        if self.category is None:
            raise TypeError(f"{self.name} is not bound to a class.")
        given, category = given or {}, self.category
        scope: dict[Any, Any] = {Self: category}
        for klass in category.__mro__:
            for parameter in getattr(klass, "__type_params__", ()):
                index = position(parameter, category)
                if index is not None:
                    scope[parameter] = category.parameters()[index]
        for parameter in inspect.unwrap(self.function).__type_params__:
            name = parameter.__name__
            scope[parameter] = given[name] if name in given\
                else variable(name, substitute(parameter.__bound__, scope))
        return scope

    def instantiate(self, scope: dict, value: Callable) -> dict:
        """
        The arguments of the bound declaration: ``value(name, kind)`` for each
        of its :attr:`parameters`, their annotation substituted in the scope.
        """
        return {
            parameter.name: value(
                parameter.name, substitute(parameter.annotation, scope))
            for parameter in self.parameters}

    def output(self, scope: dict):
        """
        The type of the output of the declaration, its return annotation
        substituted in the scope its arguments were sampled in, which the
        output :func:`inhabits`.
        """
        return substitute(
            inspect.signature(self.function).return_annotation, scope)

    def sequents(
            self, given: dict | None = None, **params
    ) -> "st.SearchStrategy[tuple]":
        """
        Generate the scope and the arguments of the bound declaration: a
        variable of sort ``Obj`` from the strategy of ``category.ob``, of
        sort ``Atom`` from the ``generators`` of ``category.ob`` and of sort
        :class:`Count` a number up to :data:`MAX_COUNT`, then a premise
        ``Hom[C1, A, B]`` from the strategy of the category with its boundary
        and ``params``, a premise ``Var[T, x]`` the value ``x`` and
        :data:`typing.Self` a term, from the strategy of the category with
        ``params`` too.
        """
        from hypothesis import strategies as st

        @st.composite
        def sequents(draw):
            def variable(name, sort):
                if sort is Count:
                    return draw(st.integers(0, MAX_COUNT), label=name)
                ob = self.category.ob
                return draw(ob.generators() if get_origin(sort) is Atom
                            else ob.strategy(), label=name)

            def value(name, kind):
                if get_origin(kind) is Var:
                    return get_args(kind)[1]
                if get_origin(kind) is Hom:
                    _, dom, cod = get_args(kind)
                    strategy = self.category.strategy(
                        dom=dom, cod=cod, **params)
                elif isinstance(kind, type) and issubclass(kind, Testable)\
                        and getattr(kind, "factory", None) is Self:
                    strategy = kind.__origin__[self.category].strategy()
                else:
                    strategy = self.category.strategy(**params)
                return draw(strategy, label=name)

            scope = self.scope(variable, given)
            return scope, self.instantiate(scope, value)

        return sequents()

    def arguments(
            self, given: dict | None = None, **params
    ) -> "st.SearchStrategy[dict]":
        """ Generate the arguments of the bound declaration, see
        :meth:`sequents`. """
        return self.sequents(given, **params).map(lambda sequent: sequent[1])

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if self.category is None:
            raise TypeError(f"{self.name} is not bound to a class.")
        return self.function(self.category, *args, **kwargs)


@dataclass(repr=False)
class Axiom[**P, T](Declaration[P, T]):
    """
    An axiom of a category: a :class:`Declaration` whose function states a
    law, from the category and its arguments to an :class:`Equation`, or to
    :obj:`NotImplemented` when the structure does not apply.

    Calling a bound axiom returns its own verdict: :obj:`NotImplemented`
    when the structure does not apply to the category, and the equation
    itself otherwise; a law declared broken raises an
    :class:`AxiomFailure` carrying that equation instead of returning it.

    A law is broken when *some* argument is a counterexample, not every one,
    so :attr:`broken` is declared by :meth:`failing` before any argument is
    generated — the property matrix marks such an axiom as an expected
    failure and lets the search find the counterexample.

    Parameters:
        broken : Whether the law is declared broken by :meth:`failing`.
    """

    _: KW_ONLY
    broken: bool = False

    __hash__ = Declaration.__hash__

    def modulo(self, up_to) -> "Axiom[P, T]":
        """
        The same law with its equation compared up to a function, so that a
        category weakens an inherited axiom in one statement, e.g.
        ``CMap.dagger_contravariance = CompactCategory.dagger_contravariance
        .modulo(CMap.to_hypergraph)``.
        """
        @wraps(self.function)
        def equation(*args, **kwargs):
            return self.function(*args, **kwargs).modulo(up_to)
        return replace(self, function=equation)

    def failing(self, reason: str) -> "Axiom[P, T]":
        """
        The same law declared broken: calling it raises an
        :class:`AxiomFailure` with the reason as message and the equation
        evaluated on the arguments, or the error raised when the
        implementation refuses to build its terms, e.g. ``braid_naturality =
        BraidedCategory.braid_naturality.failing("A free braid is a box.")``.
        """
        @wraps(self.function)
        def equation(*args, **kwargs):
            try:
                evaluated = self.function(*args, **kwargs)
            except Exception as error:
                raise AxiomFailure(reason, error) from error
            raise AxiomFailure(reason, evaluated)
        equation.__doc__ = reason
        return replace(self, function=equation, broken=True)

    def inapplicable(self, reason: str) -> "Axiom[P, T]":
        """
        The same law declared not to apply to the category: it takes no
        argument and returns :obj:`NotImplemented`, with the reason as its
        documentation, e.g. ``trace_vanishing =
        TracedCategory.trace_vanishing.inapplicable("No trace.")``.
        """
        def law(_category):
            return NotImplemented
        law.__doc__ = reason
        return replace(self, function=law, params={}, broken=False)

    def weaken(self, **params) -> "Axiom[P, T]":
        """
        The same law quantified over the morphisms the strategy of the
        category generates with the given parameters, e.g. the rigid
        diagrams of the underlying monoidal category
        ``RigidCategory.dagger_involution.weaken(within=MonoidalCategory)``,
        see :meth:`discopy.cat.Arrow.search`. Assigned to its own attribute
        beside a ``.failing`` declaration, it shows the matrix one expected
        failure and one green cell instead of one blanket expected failure.
        """
        return replace(self, params=dict(self.params, **params))

    def strategy(self) -> "st.SearchStrategy[dict]":
        """
        Generate the arguments of the bound axiom, with the parameters
        :meth:`weaken` declared, see :meth:`Declaration.arguments`.
        """
        return self.arguments(**self.params)

    def canonical(self) -> Equation:
        """
        The law as a schema: its equation with an object named after each
        type parameter, a ``box_factory`` named after each morphism, cast
        into the category when it is a generator rather than a term, e.g.
        the wire of a type, and two for each number, the equation a law
        declared broken raises included. A category whose objects take no
        name, e.g. the shapes of a network, has none and raises
        ``TypeError``.

        >>> from discopy.cat import Arrow
        >>> print(Arrow.associativity.canonical())
        Equation(f >> g >> h, f >> g >> h)
        """
        try:
            self.category.ob("A")
        except TypeError as error:
            raise TypeError(f"{self} has no canonical instance: the objects "
                            f"of {self.category} take no name.") from error

        def value(name, kind):
            if get_origin(kind) is Var:
                return get_args(kind)[1]
            dom, cod = get_args(kind)[1:] if get_origin(kind) is Hom\
                else (self.category.ob(name), ) * 2
            box = self.category.box_factory(name, dom=dom, cod=cod)
            return box if isinstance(box, self.category)\
                else self.category.cast(box)

        scope = self.scope(lambda name, sort: Count.canonical if sort is Count
                           else self.category.ob(name))
        arguments = self.instantiate(scope, value)
        try:
            return self(**arguments)  # ty: ignore[invalid-return-type]
        except AxiomFailure as failure:
            return failure.equation

    def draw(self, **params):
        """
        Draw the :meth:`canonical` equation of the law, the parameters those
        of :meth:`discopy.monoidal.Equation.draw`.

        >>> from discopy import rigid
        >>> rigid.Diagram.snake_equations.draw(
        ...     doctest="docs/_static/rigid/snake-equations-law.svg")

        .. image:: /_static/rigid/snake-equations-law.svg
            :align: center
        """
        equation = self.canonical()
        if equation is NotImplemented:
            raise TypeError(f"{self} does not apply, so has nothing to draw.")
        draw = getattr(equation, "draw", None)
        if draw is None:
            raise TypeError(f"{self} is not an equation of diagrams.")
        return draw(**params)

    def falsify(self, **params) -> dict:
        """
        Search for a shrunk counterexample to the bound axiom: arguments for
        which the verdict fails — the equation is false, or the
        implementation refuses to build its terms — raising
        :class:`hypothesis.errors.NoSuchExample` when no counterexample is
        found. Keyword arguments are passed to :func:`hypothesis.find`.

        >>> from hypothesis import settings
        >>> from discopy.cat import Arrow
        >>> Arrow.associativity.falsify(
        ...     settings=settings(max_examples=10))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
         ...
        hypothesis.errors.NoSuchExample: No examples found of condition ...
        """
        from hypothesis import find

        def refutes(arguments):
            try:
                verdict = self(**arguments)
            except AxiomFailure:
                return True
            return verdict is not NotImplemented and not verdict

        return find(self.strategy(), refutes, **params)


@dataclass(repr=False)
class Rule[**P, T](Declaration[P, T]):
    """
    An inference rule of a category: a :class:`Declaration` whose function
    concludes a morphism from its premises, the objects and morphisms it
    quantifies over, forwarding to the method that implements the operation,
    e.g. the cut ``def cut[A: Obj[C0], B: Obj[C0], C: Obj[C0]](cls, f:
    Hom[C1, A, B], g: Hom[C1, B, C]) -> Hom[C1, A, C]`` composes. Its return
    annotation ``Hom[C1, dom, cod]`` is the
    boundary of its conclusion, read by :meth:`strategy`, and never
    evaluated.

    >>> from discopy.cat import Arrow, Ob, Box
    >>> sorted(Arrow.rules)
    ['ax', 'cut']
    >>> f, g = Box('f', Ob('x'), Ob('y')), Box('g', Ob('y'), Ob('z'))
    >>> assert Arrow.cut(f, g) == f >> g
    """

    __hash__ = Declaration.__hash__

    @cached_property
    def conclusion(self) -> tuple[str | None, ...]:
        """
        The type parameter each side of the conclusion is, or :obj:`None`
        for a side that is not a variable.

        >>> from discopy.abc import MonoidalCategory
        >>> MonoidalCategory.cut.conclusion
        ('A', 'C')
        >>> MonoidalCategory.mix.conclusion
        (None, None)
        """
        function = inspect.unwrap(self.function)
        sides = get_args(inspect.signature(function).return_annotation)[1:]
        return tuple(
            side.__name__ if side in function.__type_params__ else None
            for side in sides)

    @cached_property
    def is_recursive(self) -> bool:
        """
        Whether the rule has a morphism among its premises, i.e. a parameter
        annotated with ``Hom``.
        """
        return any(
            get_origin(parameter.annotation) is Hom
            for parameter in self.parameters)

    def given(self, dom=None, cod=None) -> dict:
        """
        The variables a goal from ``dom`` to ``cod`` binds: each side of the
        conclusion that is a variable takes the side of the goal, unless the
        variable is an :class:`Atom` and the side is not.

        >>> from discopy.abc import Category
        >>> from discopy.cat import Ob
        >>> Category.cut.given(Ob('x'), Ob('y'))
        {'A': cat.Ob('x'), 'C': cat.Ob('y')}
        """
        variables = {
            variable.__name__: variable
            for variable in inspect.unwrap(self.function).__type_params__}
        return {
            name: side for name, side in zip(self.conclusion, (dom, cod))
            if name is not None and side is not None
            and (get_origin(variables[name].__bound__) is not Atom
                 or len(side) == 1)}

    def strategy(
            self, dom=None, cod=None, **params) -> "st.SearchStrategy[dict]":
        """
        Generate the premises of the rule for a conclusion from ``dom`` to
        ``cod``, see :meth:`Declaration.arguments`, the variables it binds
        :meth:`given` and the others sampled. The conclusion need not meet the
        goal on a side that is not a variable.
        """
        return self.arguments(self.given(dom, cod), **params)


def axiom[**P, T](
        equation: Callable[Concatenate[Any, P], Equation[T]]
) -> Axiom[P, Equation[T]]:
    """
    Decorate an equation as a categorical axiom: a classmethod of its
    category, implicitly, whose remaining parameters are generated.
    """
    return Axiom(equation)


def rule[**P, T](
        function: Callable[Concatenate[Any, P], T]) -> Rule[P, T]:
    """
    Decorate an operation as an inference rule: a classmethod of its
    category, implicitly, from its premises to its conclusion.
    """
    return Rule(function)


class Testable[T](ABC):
    """
    A testable class states axioms, which its subclasses inherit along with
    the structure they axiomatise, and says how to generate the terms those
    axioms quantify over with :meth:`strategy`.

    A :class:`discopy.abc.Category` states the laws of a categorical
    structure, a :class:`Serialisable` those of writing a term down and
    reading it back, which is why the two meet here rather than in either.
    """

    @classmethod
    def strategy(cls, **params) -> "st.SearchStrategy[T]":
        """
        Build a `search strategy
        <https://hypothesis.readthedocs.io/en/latest/data.html>`_ for the
        instances of ``cls``: implementing it is how a category enrols in
        the property matrix, see :meth:`enrolled`.

        An override that delegates to another strategy accepts
        ``**params``, pops the parameters it consumes and forwards the
        rest, so that a caller's bounds pass through unchanged. A terminal
        strategy instead declares exactly the parameters it implements: a
        constraint it cannot honour fails loudly as an unexpected keyword
        rather than being silently dropped.

        The default raises, deliberately not as an
        :func:`abc.abstractmethod`, which would make every category that has
        not implemented one uninstantiable rather than merely unchecked.

        >>> from discopy.abc import Nat
        >>> Nat.strategy()
        Traceback (most recent call last):
         ...
        NotImplementedError: No strategy implemented for abc.Nat.
        """
        raise NotImplementedError(
            f"No strategy implemented for {factory_name(cls)}.")

    @classmethod
    def declared[D: Declaration](cls, kind: type[D]) -> dict[str, D]:
        """
        The declarations of a kind inherited by ``cls``, by name, subclasses
        overriding bases, bound to ``cls``.

        Names are collected before they are filtered, so that assigning
        anything that is not a declaration over an inherited one drops it
        altogether, rather than restating it. Each declaration is named after
        the attribute it is stored under, so that a law weakened beside its
        failing declaration, e.g. ``dagger_involution_monoidal``, is a
        cell of its own in the matrix.
        """
        visible = {
            name: value
            for base in reversed(cls.__mro__)
            for name, value in base.__dict__.items()}
        return {name: replace(value.bind(cls), name=name)
                for name, value in visible.items()
                if isinstance(value, kind)}

    axioms = classproperty(lambda cls: cls.declared(Axiom))
    """ The axioms inherited by ``cls``, see :meth:`declared`. """

    rules = classproperty(lambda cls: cls.declared(Rule))
    """ The rules inherited by ``cls``, see :meth:`declared`. """

    @classmethod
    def subclasses(cls) -> "tuple[type[Testable], ...]":
        """
        Every transitive subclass of ``cls``, ``cls`` itself included, once,
        in the order a breadth-first walk of the subclass graph meets it.

        >>> from discopy.cat import Arrow, Box
        >>> assert Arrow.subclasses()[0] is Arrow
        >>> assert Box in Arrow.subclasses()
        """
        found, queue = {cls: None}, [cls]
        while queue:
            for subclass in queue.pop(0).__subclasses__():
                if subclass not in found:
                    found[subclass] = None
                    queue.append(subclass)
        return tuple(found)

    @classmethod
    def enrolled(cls) -> "tuple[type[Testable], ...]":
        """
        The categories the property matrix checks, by name: those that state
        laws, generate their terms and are their own ``ar``. A box inherits
        the strategy of the category it lives in, whose terms it is part of,
        rather than being a category of its own, while a type is a category
        of its own, the free coloured monoid on its wires. A box that is not
        the generator of its category but defines a strategy of its own,
        e.g. a lambda term, is checked on the terms that strategy generates.

        >>> from discopy.cat import Arrow, Box
        >>> from discopy.monoidal import Ty
        >>> from discopy.biclosed import TermBase
        >>> assert Arrow in Testable.enrolled() and Ty in Testable.enrolled()
        >>> assert Box not in Testable.enrolled()
        >>> assert TermBase in Testable.enrolled()
        """
        def generates(testable):
            try:
                testable.strategy()
            except NotImplementedError:
                return False
            return True

        def own_terms(testable):
            ar = getattr(testable, "ar", None)
            return ar is testable or isinstance(ar, type)\
                and "strategy" in vars(testable)\
                and getattr(testable, "box_factory", None) is not testable

        return tuple(sorted((
            testable for testable in cls.subclasses()
            if own_terms(testable)
            and testable.axioms and generates(testable)), key=factory_name))

    @classmethod
    def matrix(cls, once: bool = False) -> list[Axiom]:
        """
        Every law of every :meth:`enrolled` category, bound to it, or only
        ``once`` per declaration of a law: bound to the enrolled category
        nearest the class declaring it, rather than to every category
        inheriting it. A category restating a law, broken, weakened or
        modulo a quotient, declares it anew and is the nearest to that.

        >>> from discopy.cat import Arrow
        >>> assert Arrow.unitality in Testable.matrix(once=True)
        """
        cells = [
            law for testable in cls.enrolled()
            for law in testable.axioms.values()]
        if not once:
            return cells
        nearest = {}
        for law in cells:
            distance, declaring = next(
                (i, base) for i, base in enumerate(law.category.__mro__)
                if law.name in vars(base))
            key = (declaring, law.name)
            if key not in nearest or distance < nearest[key][0]:
                nearest[key] = (distance, law)
        return sorted(
            (law for _, law in nearest.values()),
            key=lambda law: (factory_name(law.category), law.name))


no_strategy = Testable.__dict__["strategy"]
"""
The default :meth:`Testable.strategy`, under a name that can be assigned: a
category that would inherit a strategy for terms it cannot read back, e.g.
a pregroup diagram, declares ``strategy = no_strategy`` to stay out of the
matrix until it implements its own.
"""


class Serialisable[T](Testable[T]):
    """
    A type whose terms are written down and read back: by their
    representation, their pickle, their deep copy and their tree, each
    with the law that it is a roundtrip.
    """

    @classmethod
    def environment(cls) -> dict:
        """
        The namespace the representation of a term reads back in: the
        public names of the package, as ``from discopy import *`` binds
        them, so that a representation qualified by module such as
        ``cat.Box('f', cat.Ob('x'), cat.Ob('y'))`` evaluates, and then
        those of the module the type is defined in, so that one
        printing bare names such as ``Tensor[int]([0], dom=Dim(1),
        cod=Dim(1))`` evaluates too. The module comes second because a
        term prints the names its own module binds: ``Dim`` in
        ``discopy.tensor`` is the one a tensor is built from.

        The import is local because the package imports this module.
        """
        import discopy

        public = lambda namespace: {
            name: value for name, value in namespace.items()
            if not name.startswith("_")}
        module = sys.modules[cls.__module__]
        return dict(public(vars(discopy)), **public(vars(module)))

    @classproperty
    def axioms(cls) -> dict[str, Axiom]:
        """
        The axioms inherited by ``cls``, by name, subclasses overriding bases.

        Names are collected before they are filtered, so that assigning
        anything that is not an axiom over an inherited one drops it
        altogether, rather than restating it.
        """
        visible = {
            name: value
            for base in reversed(cls.__mro__)
            for name, value in base.__dict__.items()}
        return {name: value.bind(cls) for name, value in visible.items()
                if isinstance(value, Axiom)}

    @axiom
    def repr_transparency(cls, term: Self) -> Equation[Self]:
        """
        The representation of a term evaluates back to it, in the
        :meth:`environment` of its type.
        """
        return Equation(eval(repr(term), cls.environment()), term)

    @axiom
    def pickling(cls, term: Self) -> Equation[tuple[type[Self], Self]]:
        """
        A term loads back from its pickle, of the same class: the equation
        is between the pairs of a class and a term, since a subscript of a
        :class:`discopy.utils.NamedGeneric` is part of what a pickle keeps.
        """
        loaded = pickle.loads(pickle.dumps(term))
        return Equation((type(loaded), loaded), (type(term), term))

    @axiom
    def copying(cls, term: Self) -> Equation[tuple[type[Self], Self]]:
        """
        A term is equal to its deep copy, of the same class. Copying goes
        through the same protocol as :meth:`pickling` without the bytes, so
        a class whose reduction drops what its state needs breaks one law
        with the other.
        """
        copied = deepcopy(term)
        return Equation((type(copied), copied), (type(term), term))

    @axiom
    def serialisation(cls, term: Self) -> Equation[Self]:
        """
        A term decodes back from its tree and from the JSON of its tree.
        A type without a tree declares the law inapplicable.
        """
        return Equation(from_tree(term.to_tree()), loads(dumps(term)), term)


class Related[factory](Testable, NamedGeneric, tuple):
    """
    Three instances of the ``factory``, an :class:`Equivalence`: two related
    by construction and a third that may be or not, see
    :meth:`Equivalence.related`.
    """
    def __new__(cls, terms):
        terms = tuple(terms)
        if len(terms) != 3:
            raise ValueError("Expected three terms.")
        return super().__new__(cls, terms)

    @classmethod
    def strategy(cls, **params) -> "st.SearchStrategy[Related]":
        """ Generate the triples the factory relates. """
        from hypothesis import strategies as st

        return st.builds(cls, cls.factory.related(**params))


class Equivalence(Testable):
    """
    A type with an equivalence relation on its instances besides equality,
    e.g. lambda terms up to the names of their bound variables, and the
    laws of an equivalence relation stated once: reflexivity, symmetry and
    transitivity.

    The relation is the :class:`Equation` subclass ``equivalence_factory``,
    which holds when its terms are related. A search over arbitrary
    instances would rarely meet related ones, so the two laws that need
    them quantify over :class:`Related` instances, which the type generates
    with :meth:`related`: two related by construction and a third that may
    be or not, so that a law runs in both directions.

    Example
    -------
    >>> from discopy.biclosed import Ty, TermBase
    >>> X = Ty("X")
    >>> f, g = (X << X)("f"), (X << X)("g")
    >>> terms = X(lambda x: f(x)), X(lambda y: f(y)), g
    >>> assert TermBase.equivalent(*terms[:2])
    >>> assert not TermBase.equivalent(terms[0], g)
    >>> assert TermBase.transitivity(Related[TermBase](terms))
    """
    equivalence_factory: ClassVar[type[Equation]]

    @classmethod
    @abstractmethod
    def related(cls, **params) -> "st.SearchStrategy[tuple]":
        """
        Generate three instances: two related by construction and a third
        comparable to them, related or not.
        """

    @classmethod
    def equivalent(cls, *terms) -> bool:
        """ Whether the terms are related, by the ``equivalence_factory``. """
        return bool(cls.equivalence_factory(*terms))

    @axiom
    def reflexivity(cls, term: Self) -> Equation:
        """ A term is related to itself. """
        return cls.equivalence_factory(term, term)

    @axiom
    def symmetry(cls, terms: Related[Self]) -> Equation:
        """ A term is related to another exactly when the other is to it. """
        first, _, third = terms
        return Equation(
            cls.equivalent(first, third), cls.equivalent(third, first))

    @axiom
    def transitivity(cls, terms: Related[Self]) -> Equation:
        """
        Terms related to each other are related to the same terms: a third
        term is related to the first exactly when it is to the second,
        which is transitivity in both directions given symmetry.
        """
        first, second, third = terms
        return Equation(
            cls.equivalent(first, third), cls.equivalent(second, third))
