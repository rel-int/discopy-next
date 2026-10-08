# -*- coding: utf-8 -*-

"""
The free (dagger) category
with formal sums, unary operators and symbolic variables.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ob
    FreeCategory
    Arrow
    Box
    Id
    Sum
    Bubble
    Functor
    Equivalence
    Transformation
    Equation
    Generator

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        factory
        dumps
        loads

Axioms
------

We can create boxes with objects as domain and codomain:

>>> x, y, z = Ob('x'), Ob('y'), Ob('z')
>>> f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, x)

We can create arbitrary arrows with identity and composition:

>>> arrow = Arrow.id(x).then(f, g, h)
>>> assert arrow == f >> g >> h == h << g << f

We can create dagger functors from the free category to itself:

>>> ob = {x: z, y: y, z: x}
>>> ar = {f: g[::-1], g: f[::-1], h: h[::-1]}
>>> F = Functor(ob, ar)
>>> assert F(arrow) == (h >> f >> g)[::-1]

We can check the axioms of dagger (i.e. a contravariant involutive
identity-on-objects endofunctor):

>>> x, y, z = Ob('x'), Ob('y'), Ob('z')
>>> f, g = Box('f', x, y), Box('g', y, z)
>>> assert f[::-1][::-1] == f
>>> assert Id(x)[::-1] == Id(x)
>>> assert (f >> g)[::-1] == g[::-1] >> f[::-1]

We can check the axioms of dagger functors.

>>> assert F(Id(x)) == Id(F(x))
>>> assert F(f >> g) == F(f) >> F(g)
>>> assert F(f[::-1]) == F(f)[::-1]
>>> assert F(f.dom) == F(f).dom and F(f.cod) == F(f).cod

Functors are bubble-preserving.

>>> assert F(f.bubble()) == F(f).bubble()
"""

from dataclasses import dataclass
from functools import total_ordering, cached_property
from typing import (
    Any, Callable, ClassVar, Mapping, Iterable, Self,
    TYPE_CHECKING, overload)

from discopy import messages, utils
from discopy.pattern import Image, Pattern, rule
from discopy.axioms import (
    Serialisable,
    axiom, Equation as AbstractEquation, MAX_FUEL, no_strategy)
from discopy.abc import Category, DaggerCategory, Hom, Obj, Var
from discopy.utils import (  # noqa: F401
    factory,
    Generator,
    factory_name,
    from_tree,
    rsubs,
    unbiased,
    MappingOrCallable,
    assert_isinstance,
    assert_iscomposable,
    assert_isparallel,
    get_origin,
)

if TYPE_CHECKING:
    import sympy
    from hypothesis import strategies as st

dumps, loads = utils.dumps, utils.loads


@total_ordering
class Ob(Serialisable):
    """
    An object with a string as :code:`name`.

    Parameters:
        name : The name of the object.

    Example
    -------
    >>> x, x_, y = Ob('x'), Ob('x'), Ob('y')
    >>> assert x == x_ and x != y
    >>> assert x.to_tree() == {'factory': 'cat.Ob', 'name': 'x'}
    """
    serialised_attrs = ('name', )

    def __setstate__(self, state):
        if "name" not in state and "_name" in state:
            state["name"] = state["_name"]
            del state["_name"]
        super().__setstate__(state)

    def __init__(self, name: str = ""):
        assert_isinstance(name, str)
        self.name = name

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.name == other.name

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return self.name < other.name

    alphabet: ClassVar[tuple[str, ...]] = tuple("abcde")
    """
    The names the strategy of a free category samples its generators
    from: finitely many and shared, so that a generated functor names
    every one of them, since composing two functors keeps only the keys
    of the left-hand map.
    """

    @classmethod
    def strategy(cls, pattern: Pattern[Self] | None = None,
                 fuel: int = MAX_FUEL) -> "st.SearchStrategy[Self]":
        """ Generate named objects, whatever the pattern and fuel. """
        from hypothesis import strategies as st

        del pattern, fuel
        return st.sampled_from(cls.alphabet).map(cls)


class FreeCategory(Category):
    """
    A category whose arrows are paths of generating arrows.

    Note
    ----
    Subclasses are assumed to have an arrow factory ``ar`` whose
    constructor accepts ``inside``, ``dom``, ``cod`` and ``_scan`` as
    keyword arguments. New arrows are always built internally through that
    constructor by keyword (passing ``_scan=False`` to skip the
    composability check when it is guaranteed by construction), so that
    subclasses are free to expose a different, more user-friendly positional
    signature without breaking the machinery below. Subclasses check the
    type of what they put ``inside`` themselves, e.g. :class:`Arrow` its
    ``Box`` and :class:`discopy.monoidal.Diagram` its layers.
    """
    is_generator = utils.DeprecatedAttribute("is_atom")

    @classmethod
    def strategy(cls, pattern: Pattern[Self] | None = None,
                 fuel: int = MAX_FUEL) -> "st.SearchStrategy[Self]":
        """ Sample terms by applying rules at random to the
        :meth:`generators`, see :meth:`discopy.axioms.Testable.sample`. """
        return cls.sample(pattern, fuel)

    @classmethod
    def generators(cls, goal: Pattern) -> "st.SearchStrategy[Self]":
        """ Generate the terms of length one from ``goal.dom`` to
        ``goal.cod``, any for a side left :obj:`None`. """
        raise NotImplementedError(
            f"No generators implemented for {cls.__name__}")

    @classmethod
    def generator[U](cls, root: type[U]) -> type[U]:
        """
        Declare ``root`` as a generator of the category, bound under its
        own name, e.g. ``@Diagram.generator`` above ``class Swap``.
        """
        binding = Generator.subclass(root)
        binding.__set_name__(cls, root.__name__)
        setattr(cls, root.__name__, binding)
        return root

    def __init__(self, inside, dom, cod, _scan=True):
        ob = type(self).ob
        dom = dom if isinstance(dom, ob) else ob(dom)
        cod = cod if isinstance(cod, ob) else ob(cod)
        self.dom, self.cod, self.inside = dom, cod, tuple(inside)
        if _scan:
            previous = dom
            for arrow in inside:
                if previous != arrow.dom:
                    raise utils.AxiomError(messages.NOT_COMPOSABLE.format(
                        previous, arrow, previous, arrow.dom))
                previous = arrow.cod
            if previous != cod:
                raise utils.AxiomError(messages.NOT_COMPOSABLE.format(
                    previous, cod, previous, cod))

    @classmethod
    def id(cls, dom: Any | None = None) -> Any:
        """The identity path on ``dom``, with no generators inside."""
        dom = cls.ob() if dom is None else dom
        return cls.ar(inside=(), dom=dom, cod=dom, _scan=False)

    def then(self, *others: Self) -> Self:
        inside, dom, cod = self.inside, self.dom, self.cod
        for other in others:
            assert_isinstance(other, self.ar)
            assert_isinstance(self, other.ar)
            if cod != other.dom:
                raise utils.AxiomError(messages.NOT_COMPOSABLE.format(
                    self, other, cod, other.dom))
            inside, cod = inside + other.inside, other.cod
        return self.ar(inside=inside, dom=dom, cod=cod, _scan=False)

    def __iter__(self):
        return iter(self.inside)

    def __len__(self):
        return len(self.inside)

    def __getitem__(self, key):
        if isinstance(key, int):
            if key >= len(self) or key < -len(self):
                raise IndexError
            key = key + len(self) if key < 0 else key
            return self[key:key + 1]
        if not isinstance(key, slice):
            raise TypeError
        start, _, step = key.indices(len(self))
        inside = self.inside[key]
        if step < 0:  # A negative step reverses the path, hence the dagger.
            inside = tuple(gen.dagger() for gen in inside)
        if inside:
            dom, cod = inside[0].dom, inside[-1].cod
        elif 0 <= start < len(self):
            dom = cod = self.inside[start].dom
        else:
            dom = cod = self.cod if step > 0 else self.dom
        return self.ar(inside=inside, dom=dom, cod=cod, _scan=abs(step) > 1)

    def dagger(self):
        """ Contravariant involution, called with :code:`[::-1]`. """
        return self[::-1]


@factory
class Arrow(FreeCategory, DaggerCategory, Serialisable):
    """
    An arrow is a tuple of composable boxes :code:`inside` with a pair of
    objects :code:`dom` and :code:`cod` as domain and codomain.

    Parameters:
        inside: The tuple of boxes inside an arrow.
        dom: The domain of an arrow, i.e. its input.
        cod: The codomain of an arrow, i.e. output
        _scan: Whether to check composition.

    .. admonition:: Summary

        .. autosummary::

            id
            then
            dagger
            bubble

    Tip
    ---
    For code clarity, it is recommended not to initialise arrows directly but
    to use :meth:`Arrow.id` and :meth:`Arrow.then` instead. For example:

    >>> x, y, z = map(Ob, "xyz")
    >>> f, g = Box('f', x, y), Box('g', y, z)
    >>> arrow = Arrow.id(x).then(f, g)  # Do this...
    >>> arrow_ = Arrow((f, g), x, z)    # ...rather than that!
    >>> assert arrow == arrow_

    Note
    ----
    Arrows can be indexed and sliced using square brackets. Indexing behaves
    like that of strings, i.e. when we index an arrow we get an arrow back.

    >>> assert (f >> g)[0] == f and (f >> g)[1] == g
    >>> assert f[:0] == Arrow.id(f.dom)
    >>> assert f[1:] == Arrow.id(f.cod)

    Note
    ----
    If ``dom`` or ``cod`` are not instances of ``ob``, they are
    automatically cast. This means one can use e.g. ``int`` instead of ``Ob``,
    see :class:`monoidal.Nat`.
    """

    @classmethod
    def generators(cls, goal: Pattern) -> "st.SearchStrategy[Self]":
        """ Generate fresh boxes from ``goal.dom`` to ``goal.cod``, any
        object for a side left :obj:`None`. """
        from hypothesis import strategies as st

        objects = cls.ob.strategy()
        return st.builds(
            lambda name, dom, cod: cls.Box(str(name), dom, cod), st.uuids(),
            objects if goal.dom is None else st.just(goal.dom),
            objects if goal.cod is None else st.just(goal.cod))

    ob = Ob
    Box: ClassVar[Generator]
    Sum: ClassVar[Generator]
    Bubble: ClassVar[Generator]
    serialised_attrs = ('inside', 'dom', 'cod')

    def __init__(self, inside, dom, cod, _scan=True):
        if _scan:
            generator = self.Box
            for box in inside:
                assert_isinstance(box, generator)
        super().__init__(inside, dom, cod, _scan=_scan)

    def __setstate__(self, state):
        if '_dom' in state:  # Backward compatibility
            self.dom, self.cod, self.inside = (
                state['_dom'], state['_cod'], tuple(state['_boxes']))
            del state['_dom'], state['_cod'], state['_boxes']
        super().__setstate__(state)

    def __repr__(self):
        if not self.inside:  # i.e. self is identity.
            return f"{factory_name(type(self))}.id({repr(self.dom)})"
        if self.atom is self:
            return super().__repr__()
        return f"{factory_name(self.ar)}(inside={repr(self.inside)}, " \
               f"dom={repr(self.dom)}, cod={repr(self.cod)})"

    def __str__(self):
        return ' >> '.join(map(str, self.inside)) or f"Id({self.dom})"

    def __add__(self, other):
        return self.Sum((self, )) + other

    def __radd__(self, other):
        return self if other == 0 else NotImplemented

    @property
    def is_atom(self):
        """ Whether an `Arrow` is an atom, i.e. it has length 1. """
        return len(self.inside) == 1

    @property
    def atom(self):
        """ The only box in an `Arrow` of length 1, i.e. its generator. """
        return self.inside[0] if self.is_atom else None

    def setoid(self):
        """
        Returns data that faithfully describes an `Arrow` making sure that
        `self.atom.setoid == self.setoid` when `self.is_atom`.
        This is used to define `Arrow.__eq__` and `Arrow.__hash__`.

        Abstract
        --------
        We are defining a [setoid](https://en.wikipedia.org/wiki/Setoid) with
        the type `Arrow` quotiented by `f.setoid() == g.setoid()` so that the
        equivalence class satisfies the axioms of category theory e.g.

        >>> f = Box('f', Ob("X"), Ob("Y"))
        >>> f_ = f >> Id(f.cod)
        >>> assert f.setoid() == f_.setoid()
        >>> assert f is not f_ and f == f_

        Warning
        -------
        Messing around with this method can lead to so-called **setoid hell**.
        In Python there is no way to give a formal proof that a function, e.g.
        functor application, is in fact a morphism of setoids, i.e. that it
        sends equal inputs to equal outputs.
        """
        atom = self.atom
        if atom is None:
            return (self.inside, self.dom, self.cod)
        return atom.setoid()

    def __eq__(self, other):
        return isinstance(other, self.ar) and self.setoid() == other.setoid()

    def __hash__(self):
        return hash(self.setoid())

    @rule
    def dagger[A, B](self: Hom[Arrow, A, B]) -> Hom[Arrow, B, A]:
        """ Contravariant involution, called with :code:`[::-1]`. """
        return self[::-1]

    def then(self, *others: Self) -> Self:
        """
        Sequential composition, called with :code:`>>` and :code:`<<`.

        Parameters:
            others : The other arrows to compose.

        Raises:
            AxiomError : Whenever `self` and `others` do not compose.

        Example
        -------
        >>> assert Arrow.id() == Id() == Id(Ob())
        >>> assert Arrow.id('x') == Id('x') == Id(Ob('x'))
        """
        if any(isinstance(other, Sum) for other in others):
            return self.Sum((self, )).then(*others)
        return super().then(*others)

    @classmethod
    def zero(cls, dom, cod):
        """
        Return the empty sum with a given domain and codomain.

        Parameters:
            dom : The domain of the empty sum.
            cod : The codomain of the empty sum.
        """
        return cls.Sum((), dom, cod)

    def bubble(self, *args, **kwargs) -> Bubble:
        """ Unary operator on homsets. """
        return self.Bubble(self, *args, **kwargs)

    @property
    def free_symbols(self) -> "set[sympy.Symbol]":
        """
        The free :code:`sympy` symbols in an arrow.

        Example
        -------

        >>> from sympy.abc import phi, psi  # doctest: +EXTRA
        >>> x, y = Ob('x'), Ob('y')
        >>> f = Box('f', x, y, data={"Alice": [phi + 1]})
        >>> g = Box('g', y, x, data={"Bob": [psi / 2]})
        >>> diagram = (f >> g).bubble() + Id(x)
        >>> assert diagram.free_symbols == {phi, psi}
        """
        return {x for box in self.inside for x in box.free_symbols}

    def subs(self, *args) -> Arrow:
        """
        Substitute a variable by an expression.

        Parameters:
            var (sympy.Symbol) : The subtituted variable.
            expr (sympy.Expr) : The substituting expression.

        Tip
        ---
        You can give a list of :code:`(var, expr)` for multiple substitution.

        Example
        -------
        >>> from sympy.abc import phi, psi  # doctest: +EXTRA
        >>> x, y = Ob('x'), Ob('y')
        >>> f = Box('f', x, y, data={"Alice": [phi + 1]})
        >>> g = Box('g', y, x, data={"Bob": [psi / 2]})
        >>> assert (f >> g).subs(phi, phi + 1) == f.subs(phi, phi + 1) >> g
        >>> assert (f >> g).subs(phi, 1) == f.subs(phi, 1) >> g
        >>> assert (f >> g).subs(psi, 1) == f >> g.subs(psi, 1)
        """
        inside = tuple(box.subs(*args) for box in self.inside)
        return self.ar(inside, self.dom, self.cod, _scan=False)

    def lambdify(self, *symbols: "sympy.Symbol", **kwargs) -> Callable:
        """
        Turn a symbolic diagram into a function from parameters to diagram.

        Parameters:
            symbols : The inputs of the function.
            kwargs : Passed to :code:`sympy.lambdify`.

        Example
        -------
        >>> from sympy.abc import phi, psi  # doctest: +EXTRA
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Box('f', x, y, data=phi), Box('g', y, z, data=psi)
        >>> assert f.lambdify(psi)(42) == f
        >>> assert (f >> g).lambdify(phi, psi)(42, 43)\\
        ...     == Box('f', x, y, data=42) >> Box('g', y, z, data=43)
        """
        return lambda *xs: self.ar(
            dom=self.dom, cod=self.cod, inside=tuple(
                box.lambdify(*symbols, **kwargs)(*xs) for box in self.inside))


@total_ordering
@Arrow.generator
class Box(Arrow):
    """
    A box is an arrow with a :code:`name` and the tuple of just itself inside.

    Parameters:
        name : The name of the box.
        dom : The domain of the box, i.e. the input.
        cod : The codomain of the box, i.e. the output.
        data (any) : Extra data in the box, default is :code:`None`.
        is_dagger : Whether the box is dagger.

    Example
    -------

    >>> x, y = Ob('x'), Ob('y')
    >>> f = Box('f', x, y, data=[42])
    >>> assert f.inside == (f, )
    """
    data, is_dagger = None, False
    serialised_attrs = ('name', 'dom', 'cod', 'is_dagger', 'data')

    strategy = no_strategy  # A box is sampled by its category.

    def __setstate__(self, state):
        if '_name' in state:  # Backward compatibility
            self.name, self.data, self.is_dagger = (
                state['_name'], state['_data'], state['_dagger'])
            del state['_name'], state['_data'], state['_dagger']
        super().__setstate__(state)

    def __init__(
            self, name: str, dom: Ob, cod: Ob, data=None, is_dagger=False):
        assert_isinstance(name, str)
        self.name, self.data, self.is_dagger = name, data, is_dagger
        Arrow.__init__(self, (self, ), dom, cod, _scan=False)

    @cached_property
    def free_symbols(self) -> "set[sympy.Symbol]":
        def recursive_free_symbols(data):
            if isinstance(data, Mapping):
                data = data.values()
            if isinstance(data, Iterable):
                # Handles numpy 0-d arrays, which are actually not iterable.
                if not hasattr(data, "shape") or data.shape != ():
                    return set().union(*map(recursive_free_symbols, data))
            return getattr(data, "free_symbols", set())
        return recursive_free_symbols(self.data)

    def subs(self, *args) -> Box:
        if not any(var in self.free_symbols for var in (
                {var for var, _ in args[0]} if len(args) == 1 else {args[0]})):
            return self
        return type(self)(
            self.name, self.dom, self.cod, is_dagger=self.is_dagger,
            data=rsubs(self.data, *args))

    def lambdify(self, *symbols: "sympy.Symbol", **kwargs) -> Callable:
        if not any(x in self.free_symbols for x in symbols):
            return lambda *xs: self
        from sympy import lambdify
        return lambda *xs: type(self)(
            self.name, self.dom, self.cod, is_dagger=self.is_dagger,
            data=lambdify(symbols, self.data, **kwargs)(*xs))

    def dagger(self) -> Self:
        return type(self)(
            self.name, self.cod, self.dom,
            data=self.data, is_dagger=not self.is_dagger)

    def __getitem__(self, key):
        if key == slice(None, None, -1):
            return self.dagger()
        return super().__getitem__(key)

    def __repr__(self):
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        return super().__repr__()

    def __str__(self):
        return str(self.name) + ("[::-1]" if self.is_dagger else '')

    def setoid(self):
        """
        The equality and hash of a box is given by hashing its type as well as
        its internal attributes `name, dom, cod, is_dagger` and `data`. In
        particular if the `data` is not hashable then neither is the `Box`.
        """
        attributes = self.name, self.dom, self.cod, self.is_dagger, self.data
        return (type(self), ) + attributes

    def __lt__(self, other):
        return self.name < other.name


@Arrow.generator
class Sum(Box):
    """
    A sum is a tuple of arrows :code:`terms` with the same domain and codomain.

    Parameters:
        terms : The terms of the formal sum.
        dom : The domain of the formal sum.
        cod : The codomain of the formal sum.

    Example
    -------
    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> f, g = Box('f', x, y), Box('g', y, z)
    >>> unit = Sum((), x, y)
    >>> assert f + unit == f == unit + f
    >>> assert f >> (g + g) == (f >> g) + (f >> g) == (f + f) >> g

    Important
    ---------
    Domain and codomain are optional only if the terms are non-empty.

    Note
    ----
    The sum is non-commutative, i.e. :code:`Sum([f, g]) != Sum([g, f])`.
    """
    serialised_attrs = ('terms', 'dom', 'cod')
    strategy = no_strategy

    def __init__(self, terms: Iterable[Arrow],
                 dom: Ob | None = None, cod: Ob | None = None):
        terms = tuple(terms)
        if not terms and (dom is None or cod is None):
            raise ValueError(messages.MISSING_TYPES_FOR_EMPTY_SUM)
        dom = terms[0].dom if dom is None else dom
        cod = terms[0].cod if cod is None else cod
        for arrow in terms:
            assert_isparallel(Sum((), dom, cod), arrow)
        str_args = f", dom={repr(dom)}, cod={repr(cod)}" if not terms else ""
        name = f"{factory_name(type(self))}(terms={repr(terms)}{str_args})"
        self.terms = terms
        super().__init__(name, dom, cod)

    @property
    def is_atom(self):
        return len(self.terms) == 1 and self.terms[0].is_atom

    @property
    def atom(self):
        return self.terms[0].atom if self.is_atom else None

    def setoid(self):
        """ Ensure that a singleton sum is in fact equal to its only term. """
        return self.terms[0].setoid() if len(self.terms) == 1 else (
            type(self), self.terms, self.dom, self.cod)

    def __repr__(self):
        return self.name

    def __str__(self):
        return " + ".join(f"({arrow})" for arrow in self.terms)\
            if self.terms else\
            f"{factory_name(type(self))}((), {self.dom}, {self.cod})"

    def __add__(self, other):
        assert_isparallel(self, other)
        other = other if isinstance(other, Sum)\
            else self.Sum((other, ))
        return self.Sum(self.terms + other.terms, self.dom, self.cod)

    def __iter__(self):
        for arrow in self.terms:
            yield arrow

    def __len__(self):
        return len(self.terms)

    @unbiased
    def then(self, other: Arrow) -> Sum:
        other = other if isinstance(other, Sum)\
            else self.Sum((other, ))
        terms = tuple(f.then(g) for f in self.terms for g in other.terms)
        return self.Sum(terms, self.dom, other.cod)

    def dagger(self):
        terms = tuple(f.dagger() for f in self.terms)
        return self.Sum(terms, self.cod, self.dom)

    @property
    def free_symbols(self):
        return {x for box in self.terms for x in box.free_symbols}

    def subs(self, *args):
        terms = tuple(f.subs(*args) for f in self.terms)
        return self.Sum(terms, self.dom, self.cod)

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: self.Sum(
            tuple(box.lambdify(*symbols, **kwargs)(*xs) for box in self.terms),
            dom=self.dom, cod=self.cod)


@Arrow.generator
class Bubble(Box):
    """
    A bubble is a box with arrow :code:`args` inside and an optional pair of
    objects :code:`dom` and :code:`cod`.

    Parameters:
        args : The arrows inside the bubble.
        dom : The domain of the bubble, default is that of :code:`args`.
        cod : The codomain of the bubble, default is that of :code:`args`.
        name (str) : An optional name for the bubble.
        method (str) : The method to call when a functor is applied to it.
        kwargs : Passed to the `__init__` of :class:`Box`.

    Raises:
        ValueError : When dom is None but all the args have the same dom.
    """
    serialised_attrs = ('args', 'dom', 'cod')
    strategy = no_strategy

    def __init__(self, *args: Arrow, dom: Ob | None = None,
                 cod: Ob | None = None, name="", method="bubble", **kwargs):
        dom, = set(arg.dom for arg in args) if dom is None else (dom, )
        cod, = set(arg.cod for arg in args) if cod is None else (cod, )
        self.args, self.method = args, method
        Box.__init__(self, name, dom, cod, **kwargs)

    @property
    def arg(self):
        """ The arrow inside the bubble if there is exactly one. """
        if len(self.args) == 1:
            return self.args[0]
        raise ValueError(f"{self} has multiple args.")

    @property
    def is_id_on_objects(self):
        """ Whether the bubble is identity on objects. """
        return len(self.args) == 1 and (
            self.dom, self.cod) == (self.arg.dom, self.arg.cod)

    def setoid(self):
        """
        Ensure that bubbles are equal if they have the same type, their `args`
        are equal as well as their attributes `dom, cod, name, method`.
        """
        args_data = tuple(f.setoid() for f in self.args)
        return (type(self), ) + args_data + tuple(getattr(self, x) for x in (
            "dom", "cod", "name", "method"))

    def __str__(self):
        str_args = ",".join(map(str, self.args))
        str_dom_cod = '' if self.is_id_on_objects else (
            f'dom={self.dom}, cod={self.cod}')
        return f"({str_args}).bubble({str_dom_cod})"

    def __repr__(self):
        repr_args = ", ".join(map(repr, self.args))
        repr_dom_cod = "" if self.is_id_on_objects else (
            f", dom={repr(self.dom)}, cod={repr(self.cod)}")
        return factory_name(type(self)) + (f"({repr_args}{repr_dom_cod})")

    @property
    def free_symbols(self):
        return super().free_symbols.union(*[f.free_symbols for f in self.args])

    def dagger(self) -> Bubble:
        return type(self)(
            *(arg.dagger() for arg in self.args),
            dom=self.cod, cod=self.dom, name=self.name, method=self.method,
            data=self.data, is_dagger=not self.is_dagger)

    @classmethod
    def from_tree(cls, tree):
        """
        Decode a serialised bubble, whose ``args`` unpack as positional
        arguments, see :func:`discopy.utils.loads`.

        Parameters:
            tree : DisCoPy serialisation.
        """
        dom, cod = map(from_tree, (tree['dom'], tree['cod']))
        return cls(*map(from_tree, tree['args']), dom=dom, cod=cod)


@factory
class Functor[In0, In1, Out0, Out1](Category, Serialisable):
    """
    A functor is a pair of maps :code:`ob_map` and :code:`ar_map` and an
    optional codomain category :code:`cod`.

    Parameters:
        ob_map : Mapping from :class:`Ob` to :code:`cod.ob`.
        ar_map : Mapping from :class:`Box` to :code:`cod`.
        cod : The codomain, :code:`Arrow` by default.

    Example
    -------
    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> f, g = Box('f', x, y), Box('g', y, z)
    >>> ob_map, ar_map = {x: y, y: z, z: y}, {f: g, g: g[::-1]}
    >>> F = Functor(ob_map, ar_map)
    >>> assert F(x) == y and F(f) == g

    Tip
    ---
    Both :code:`ob_map` and :code:`ar_map` can be a function rather than a
    dictionary.
    In conjunction with :attr:`Box.data`, this can be used to create a
    :class:`Functor` from a free category with infinitely many generators.

    >>> ob_map = lambda x: x
    >>> ar_map = lambda f: Box(f.name, f.dom, f.cod, data=f.data + 1)
    >>> F = Functor(ob_map, ar_map)
    >>> h = Box('h', x, x, data=42)
    >>> assert F(h).data == 43 and F(F(h)).data == 44

    If :attr:`Box.data` is a mutable object, then so can be the image of a
    :class:`Functor` on it.

    >>> ar_map = lambda f: f if all(f.data) else f[::-1]
    >>> F = Functor(ob_map, ar_map)
    >>> m = Box('m', x, x, data=[True])
    >>> assert F(m) == m
    >>> m.data.append(False)
    >>> assert F(m) == m[::-1]
    """
    ob = type[Category]
    dom = cod = Arrow

    @classmethod
    def parameters(cls) -> tuple[type, ...]:
        """
        What the type parameters ``In0, In1, Out0, Out1`` of a functor
        stand for: the objects and arrows of its domain then those of its
        codomain.
        """
        return (getattr(cls.dom, "ob", cls.dom), cls.dom,
                getattr(cls.cod, "ob", cls.cod), cls.cod)

    @classmethod
    def id(cls, dom: type | None = None) -> Functor:
        """
        The identity functor on a given category ``dom``.

        Parameters:
            dom : The domain of the functor.
        """
        return cls(lambda x: x, lambda f: f, dom=dom, cod=dom)

    @unbiased
    def then(self, other: Functor) -> Functor:
        """
        The composition of functor with another.

        Parameters:
            other : The other functor with which to compose.

        Note
        ----
        Functor composition is unital only on the left. Indeed, we cannot check
        equality of functors defined with functions instead of dictionaries.

        Example
        -------
        >>> x, y = Ob('x'), Ob('y')
        >>> F, G = Functor({x: y}, {}), Functor({y: x}, {})
        >>> print(F >> G)
        cat.Functor(ob_map={cat.Ob('x'): cat.Ob('x')}, ar_map={})
        >>> assert F >> Functor.id() == F != Functor.id() >> F
        >>> print(Functor.id() >> F)  # doctest: +ELLIPSIS
        cat.Functor(ob_map=<function ...>, ar_map=...)
        """
        assert_isinstance(other, Functor)
        assert_iscomposable(self, other)
        ob_map, ar_map = self.ob_map.then(other), self.ar_map.then(other)
        return type(self)(ob_map, ar_map, dom=self.dom, cod=other.cod)

    def __init__(
            self,
            ob_map: Mapping[Ob, Ob] | Callable[[Ob], Ob] | None = None,
            ar_map: Mapping[Box, Arrow] | Callable[[Box], Arrow] | None = None,
            dom: type | None = None, cod: type | None = None):
        self.dom, self.cod = (  # ty: ignore[invalid-assignment]
            dom or type(self).dom, cod or type(self).cod)
        self.ob_map: MappingOrCallable[Ob, Ob] = MappingOrCallable(
            ob_map or {})
        self.ar_map: MappingOrCallable[Box, Arrow] = MappingOrCallable(
            ar_map or {})

    def __eq__(self, other):
        return type(self) is type(other)\
            and (self.ob_map, self.ar_map, self.cod)\
            == (other.ob_map, other.ar_map, other.cod)

    def __repr__(self):
        cod_repr = "" if self.cod == type(self).cod\
            else f", cod={factory_name(self.cod)}"
        return factory_name(type(self))\
            + f"(ob_map={self.ob_map}, ar_map={self.ar_map}{cod_repr})"

    @overload
    def __call__(self, other: In0) -> Out0:
        ...

    @overload
    def __call__(self, other: In1) -> Out1:
        ...

    def __call__(self, other):
        """ The action of the functor, an object of ``dom`` to an object
        of ``cod`` and an arrow of ``dom`` to an arrow of ``cod``, as the
        two overloads spell. """
        if isinstance(other, Ob):
            result = self.ob_map[other]
            origin = get_origin(self.cod.ob)
            return result if isinstance(result, origin)\
                else self.cod.ob(result)
        if isinstance(other, Sum):
            return sum(map(self, other.terms),  # ty: ignore[invalid-argument-type]
                       self.cod.zero(self(other.dom), self(other.cod)))
        if isinstance(other, Bubble) and hasattr(self.cod, other.method):
            dom, cod = map(self, (other.dom, other.cod))
            return getattr(self.cod, other.method)(
                *map(self, other.args),  # ty: ignore[invalid-argument-type]
                dom=dom, cod=cod)
        if isinstance(other, Box) and other.is_dagger:
            return self(other.dagger()).dagger()
        if isinstance(other, Box):
            result = self.ar_map[other]
            # This allows some nice syntactic sugar for the ar mapping.
            return result if isinstance(result, self.cod)\
                else self.cod(result, self(other.dom), self(other.cod))
        assert_isinstance(other, Arrow)
        result = self.cod.id(self(other.dom))
        for box in other.inside:
            result = result >> self(box)
        return result

    @classmethod
    def strategy(cls, pattern: Pattern[Self] | None = None,
                 fuel: int = MAX_FUEL) -> "st.SearchStrategy[Self]":
        """
        Generate an endofunctor relabelling every generator, whatever the
        pattern and fuel, for the functors whose domain has objects freely
        generated by names: a relabelling is an endofunctor, so the
        functors into another category — tensors, intertwiners, channels
        — stay unchecked, as does a category whose objects are dimensions
        or information units, which has no relabelling.
        """
        from hypothesis import strategies as st

        del pattern, fuel
        if cls.dom is not cls.cod:
            raise NotImplementedError(
                f"A relabelling is an endofunctor, {cls.__name__} goes "
                f"from {cls.dom.__name__} to {cls.cod.__name__}.")
        try:
            atoms = [cls.dom.ob(name) for name in Ob.alphabet]
        except (TypeError, ValueError) as error:
            raise NotImplementedError from error

        def relabel(images):
            """ The endofunctor sending each atom to its image. """
            labelling = Relabelling(tuple(zip(atoms, images)))
            return cls(labelling, labelling)

        return st.tuples(
            *(st.sampled_from(atoms) for _ in atoms)).map(relabel)

    serialisation = Serialisable.serialisation.inapplicable(
        "A functor has no tree.")

    @axiom
    def functor_identity[F: Obj[Self], A: Obj[In0]](
            cls, functor: Var[Self, F], x: Var[In0, A]
    ) -> AbstractEquation[Hom[Out1, Image[F, A], Image[F, A]]]:
        """ A functor preserves identities. """
        return functor.cod.Equation(
            functor(functor.dom.id(x)), functor.cod.id(functor(x)))

    @axiom
    def functor_composition[
            F: Obj[Self], A: Obj[In0], B: Obj[In0], C: Obj[In0]](
            cls, functor: Var[Self, F], f: Hom[In1, A, B], g: Hom[In1, B, C]
    ) -> AbstractEquation[Hom[Out1, Image[F, A], Image[F, C]]]:
        """ A functor preserves composition. """
        return functor.cod.Equation(
            functor(f.then(g)), functor(f).then(functor(g)))

    unitality = Category.unitality.failing(
        "The identity functor is a pair of functions: composing it on the "
        "left of a functor given by mappings acts the same but compares "
        "unequal (#648).")

    @axiom
    def associativity(cls, f: Self, g: Self, h: Self):
        """
        Associativity of the composition of functors.

        The objects of ``Cat`` are categories, which the property matrix does
        not generate, so this is stated of the endofunctors it does.
        """
        return AbstractEquation(f.then(g).then(h), f.then(g.then(h)))


class Equivalence(Functor, DaggerCategory):
    """
    A functor with an inverse, an instance rather than a subclass: the
    functor applies ``encode`` to the arrows of ``dom``, letting
    objects pass through, and :meth:`dagger` is the equivalence the
    other way around, swapping ``encode`` and ``decode``, so the
    involution holds by construction. Functors given by mappings
    compare unequal to the identity functor (#648), so the laws of an
    equivalence quantify the composites pointwise, over the arrows of
    its domain: see e.g. :meth:`discopy.monoidal.Diagram\
.hypergraph_equivalence` and the laws it satisfies.

    Parameters:
        encode : The action on the arrows of ``dom``.
        decode : The action on the arrows of ``cod``, its inverse.
        dom : The domain category.
        cod : The codomain category.

    Example
    -------
    >>> from discopy.monoidal import Ty, Box, Diagram
    >>> encode = Diagram.hypergraph_equivalence()
    >>> f = Box('f', Ty('x'), Ty('y'))
    >>> assert encode.dagger()(encode(f)) == f
    """
    #: The laws of an equivalence are stated on the category that
    #: builds the instance, not on this class.
    strategy = no_strategy

    def __init__(self, encode: Callable, decode: Callable,
                 dom: type, cod: type):
        self.encode, self.decode = encode, decode
        self.dom, self.cod = dom, cod  # ty: ignore[invalid-assignment]

    def __call__(self, other):
        return self.encode(other) if isinstance(other, self.dom)\
            else other

    def dagger(self) -> Self:
        return type(self)(self.decode, self.encode, self.cod, self.dom)

    def __eq__(self, other):
        return isinstance(other, Equivalence)\
            and (self.encode, self.decode, self.dom, self.cod)\
            == (other.encode, other.decode, other.dom, other.cod)

    def __hash__(self):
        return hash((self.encode, self.decode, self.dom, self.cod))

    def __repr__(self):
        return factory_name(type(self)) + (
            f"({self.encode.__qualname__}, {self.decode.__qualname__})")


@dataclass(frozen=True, eq=False)
class Relabelling(Mapping):
    """
    A map on the generators of a free category, sending the atoms it names to
    a chosen object, every other atom to itself, and a box to one of the
    same name on the relabelled boundary.

    It is a :class:`Mapping` rather than a closure so that functors built
    from it can be composed and compared, which is what makes the axioms of
    ``Cat`` itself checkable: :meth:`discopy.utils.MappingOrCallable.then`
    composes by iterating the keys of the left-hand map, and equality
    compares the wrapped maps. Iterating yields only the atoms it renames,
    while looking up is total, so a functor built from it as both its
    object and its arrow map applies to any diagram and still composes to
    something comparable.
    """
    images: tuple[tuple[object, object], ...] = ()

    def __repr__(self):
        return factory_name(type(self)) + f"(images={self.images!r})"

    def __getitem__(self, key):
        """
        The image of an atom, looked up by name, or of a box, relabelled on
        its boundary by the functor of the box's own category: a rotation or
        a delay of an atom is that functor's business, e.g.
        :class:`discopy.rigid.Functor`'s, on a box's boundary as on an
        object.

        """
        if not isinstance(key, Ob):
            functor = getattr(type(key), "Functor", Functor)
            relabel = functor(self, self)
            return type(key)(key.name, relabel(key.dom), relabel(key.cod))
        wire, = getattr(key, "inside", (key, ))
        for atom, image in self.images:
            other, = getattr(atom, "inside", (atom, ))
            if other.name == wire.name:
                return image
        return key

    def __iter__(self):
        return iter([atom for atom, _ in self.images])

    def __len__(self):
        return len(self.images)

    def __bool__(self):
        """ A relabelling is total, even when it renames nothing. """
        return True


@factory
class Transformation(Category):
    """
    A (not necessarily natural) transformation between two parallel functors.

    Parameters:
        components :
            A mapping from objects ``x`` in the domain category to arrows
            ``components[x] : dom(x) -> cod(x)`` in the codomain category.
        dom : The domain functor.
        cod : The codomain functor.

    Example
    -------
    >>> x, y = Ob('x'), Ob('y')
    >>> f = Box('f', x, y)
    >>> F, G = Functor.id(), Functor({x: y, y: x}, {})
    >>> alpha = Transformation({x: f, y: f[::-1]}, F, G)
    >>> assert alpha(x) == f and alpha(y) == f[::-1]
    >>> beta = Transformation.id(G)
    >>> assert (alpha >> beta)(x) == alpha(x) >> beta(x)
    """
    ob = Functor

    def __init__(
            self, components: Mapping[Ob, Arrow] | Callable[[Ob], Arrow],
            dom: Functor, cod: Functor):
        assert_isinstance(dom, Functor)
        assert_isinstance(cod, Functor)
        if dom.dom != cod.dom or dom.cod != cod.cod:
            raise utils.AxiomError(
                "Transformation.dom and Transformation.cod must "
                "have the same domain and codomain.")
        self.dom, self.cod = dom, cod
        self.components: MappingOrCallable[Ob, Arrow] = MappingOrCallable(
            components)

    def __call__(self, x: Ob) -> Arrow:
        """
        The component of the transformation at a given object ``x``,
        i.e. the arrow from ``dom(x)`` to ``cod(x)`` -- from ``F(x)`` to
        ``G(x)`` for the functors ``F = self.dom`` and ``G = self.cod``.

        Parameters:
            x : The object at which to take the component.

        Example
        -------
        >>> x, y = Ob('x'), Ob('y')
        >>> f = Box('f', x, y)
        >>> F, G = Functor.id(), Functor({x: y, y: x}, {})
        >>> alpha = Transformation({x: f, y: f[::-1]}, F, G)
        >>> alpha(x)
        cat.Box('f', cat.Ob('x'), cat.Ob('y'))
        >>> assert alpha(x).dom == F(x) and alpha(x).cod == G(x)
        """
        component = self.components[x]
        if component.dom != self.dom(x) or component.cod != self.cod(x):
            raise utils.AxiomError(
                f"The component at {x} must be an arrow "
                f"from {self.dom(x)} to {self.cod(x)}.")
        return component

    @classmethod
    def id(cls, dom: Functor) -> Transformation:
        """
        The identity transformation on a given functor ``dom``, i.e. the
        transformation whose component at each object ``x`` is the
        identity arrow on ``dom(x)``.

        Parameters:
            dom : The functor on which to take the identity transformation.

        Example
        -------
        >>> x, y = Ob('x'), Ob('y')
        >>> F = Functor({x: y, y: x}, {})
        >>> alpha = Transformation.id(F)
        >>> alpha(x)
        cat.Arrow.id(cat.Ob('y'))
        >>> assert alpha(x) == F.cod.id(F(x))
        """
        return cls(lambda x: dom.cod.id(dom(x)), dom, dom)

    @unbiased
    def then(self, other: Transformation) -> Transformation:
        """
        The vertical composition of a transformation with another.

        Parameters:
            other : The other transformation with which to compose.
        """
        assert_isinstance(other, Transformation)
        if self.cod != other.dom:
            raise utils.AxiomError(messages.NOT_COMPOSABLE.format(
                self, other, self.cod, other.dom))
        components = lambda x: self(x) >> other(x)
        return type(self)(components, self.dom, other.cod)

    def __eq__(self, other):
        return isinstance(other, Transformation) and (
            self.components, self.dom, self.cod) == (
                other.components, other.dom, other.cod)

    def __repr__(self):
        return factory_name(type(self)) + (
            f"(components={self.components}, "
            f"dom={self.dom!r}, cod={self.cod!r})")


class Equation(AbstractEquation[Arrow]):
    """
    An :class:`.axioms.Equation` between arrows, see its docstring for the
    parameters and :meth:`.axioms.Equation.modulo` for quotients.

    Example
    -------
    >>> x = Ob('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> parity = lambda term: len(term.inside) % 2
    >>> assert not Equation(f, f >> g >> g)
    >>> assert Equation(f, f >> g >> g, up_to=parity)
    """


Ob.Equation = Arrow.Equation = Equation
Id = Arrow.id
