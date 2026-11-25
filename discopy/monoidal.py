# -*- coding: utf-8 -*-

"""
The free (pre)monoidal category, i.e. planar diagrams.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Colour
    Wire
    List
    Ty
    Nat
    Dim
    Layer
    Diagram
    Box
    Bubble
    Functor
    Match
    Equation
    TermBase
    Variable
    Constant
    Application
    Kappa
    InterchangeEquation
    Sampler

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        uncolour_state
        let

Axioms
------
We can check the axioms for :class:`Ty` being a monoid.

>>> x, y, z, unit = Ty('x'), Ty('y'), Ty('z'), Ty()
>>> assert x @ unit == x == unit @ x
>>> assert (x @ y) @ z == x @ y @ z == x @ (y @ z)

We can check the axioms for dagger monoidal categories, up to interchanger.

>>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
>>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
>>> d = Id(x) @ f1 >> f0 @ Id(w)
>>> assert d == (f0 @ f1).interchange(0, 1)
>>> assert f0 @ f1 == d.interchange(0, 1)
>>> assert (f0 @ f1)[::-1][::-1] == f0 @ f1
>>> assert (f0 @ f1)[::-1].interchange(0, 1) == f0[::-1] @ f1[::-1]

We can check the Eckmann-Hilton argument, up to interchanger.

>>> s0, s1 = Box('s0', Ty(), Ty()), Box('s1', Ty(), Ty())
>>> assert s0 @ s1 == s0 >> s1 == (s1 @ s0).interchange(0, 1)
>>> assert s1 @ s0 == s1 >> s0 == (s0 @ s1).interchange(0, 1)

.. image:: /_static/monoidal/EckmannHilton.gif
    :align: center

Terms
-----

A diagram also reads as a term of the planar linear kappa calculus, see
:class:`TermBase`: the terms of :mod:`discopy.cat` with a list of variables
as context, each used once and in order.

A term has exactly one output, and the translation between diagrams and
terms is faithful for the diagrams that have exactly one too:
:meth:`Diagram.to_term` reads one variable for each wire
and a ``let`` for each box but the last, which gives the output and needs
none, and :meth:`TermBase.eval` gives the diagram back, box for box. A
diagram with any other number of outputs is refused rather than read as the
term of a different diagram.

>>> x, y, z = map(Ty, "xyz")
>>> f, g, h = Box('f', x, y), Box('g', x, z), Box('h', y @ z, x)
>>> diagram = f @ x >> y @ g >> h
>>> print(diagram.to_term())
(x @ x)(lambda x0, x1: let(f(x0), lambda x2: \
let(g(x1), lambda x3: h(x2, x3))))
>>> assert diagram.to_term().eval() == diagram
>>> (f @ g).to_term()
Traceback (most recent call last):
    ...
ValueError: Expected exactly one output, got y @ z: a term has one.

This loses no diagram where the category has products: a
:class:`discopy.closed.Pack` box turns any closed diagram into one with
exactly one output, of :class:`discopy.closed.Product` type. A closed diagram
reads as a lambda term by default, see :meth:`discopy.closed.Diagram.to_term`,
and as a kappa term with :meth:`TermBase.from_diagram`.

>>> from discopy import closed
>>> X, Y, Z = map(closed.Ty, "XYZ")
>>> F, G = closed.Box('f', X, Y), closed.Box('g', X, Z)
>>> packed = F @ G >> closed.Pack(Y * Z)
>>> assert len(packed.cod) == 1
>>> print(TermBase.from_diagram(packed).normal_form())
(X @ X)(lambda x0, x1: Pack((Y * Z))(f(x0), g(x1)))
>>> assert TermBase.from_diagram(packed).eval() == packed

Beta reduction :math:`(\\kappa x. e)(c) = e[c / x]` inlines the ``let`` of a
box with one output, and :meth:`TermBase.normal_form` inlines them all. It is
sound up to interchange: the two orders of ``f @ g`` have one normal form.

>>> print(diagram.to_term().normal_form())
(x @ x)(lambda x0, x1: h(f(x0), g(x1)))
>>> other = x @ g >> f @ z >> h
>>> assert other != diagram and InterchangeEquation(other, diagram)
>>> assert other.to_term().normal_form() == diagram.to_term().normal_form()

When every box has one output the normal form is a tree, and two diagrams
with one output are equal up to interchange exactly when their normal forms
are. A box with several outputs keeps its ``let``, which binds them all at
once, and two such statements do not commute: there the normal form is
sound, not complete.

>>> copy = Box('copy', x, x @ x)
>>> print((copy >> f @ g >> h).to_term().normal_form())
x(lambda x0: let(copy(x0), lambda x1, x2: h(f(x1), g(x2))))
>>> k = Box('k', x @ x @ x @ x, x)
>>> first = copy @ x >> x @ x @ copy >> k
>>> second = x @ copy >> copy @ x @ x >> k
>>> assert first.to_hypergraph() == second.to_hypergraph()
>>> assert first.to_term().normal_form() != second.to_term().normal_form()

Eta expansion :math:`\\kappa x. e(x) = e` holds on the nose.

>>> term = Constant(h)
>>> assert (y @ z)(lambda a, b: term(a, b)).eval() == term.eval() == h

A term is drawn as its diagram:

>>> diagram.to_term().draw(doctest='docs/_static/monoidal/let.svg')

.. image:: /_static/monoidal/let.svg
    :align: center

The modules above have terms of their own, down to :mod:`discopy.biclosed`
where they are lambda terms: a :class:`discopy.symmetric.Kappa` binds its
variables in any order, a :class:`discopy.markov.Kappa` need not use them
and a :class:`discopy.markov.Application` may use one twice.
"""

from __future__ import annotations

import itertools
from copy import copy
from itertools import count
from dataclasses import dataclass, field
from functools import cached_property
from operator import index
from types import NoneType
from typing import (
    Iterable, Iterator, Callable, ClassVar, Mapping, Self, Sequence,
    TYPE_CHECKING, cast)
from warnings import warn

from discopy import abc, cat, drawing, hypergraph, cmap, messages, testable
from discopy import sum as formal_sum
from discopy.abc import (
    ColouredMonoid, DaggerMonoidalCategory, Monoid, MonoidalCategory)
# pylint: disable-next=unused-import  # the redex a term prints as
from discopy.cat import let  # noqa: F401
from discopy.testable import (
    GENERATORS, Equation as AbstractEquation, Hom, Obj, TestableGeneric,
    Var, axiom)
from discopy.drawing import Drawing
from discopy.config import (
    BOX_DRAWING_ATTRIBUTES, WIRE_DRAWING_ATTRIBUTES,
    COLOUR_DRAWING_ATTRIBUTES, TRANSPARENT)
from discopy.utils import (
    factory,
    factory_key,
    factory_name,
    from_tree,
    assert_isinstance,
    assert_iscomposable,
    AxiomError,
    classproperty,
    deprecated_alias,
    MappingOrCallable,
    RichDisplay,
    text_width,
    unbiased,
)

if TYPE_CHECKING:
    import sympy
    from hypothesis import strategies as st


@dataclass(frozen=True)
class Colour(cat.Ob):
    """
    A 0-cell, drawn using its matplotlib-compatible ``name``. A region with
    no colour is ``None``, drawn as the page.

    An optional ``label`` gives the region a human-readable name for the
    drawing legend (e.g. a category) while still filling with ``name``. It
    is ignored for equality and hashing, so two regions with the same fill
    colour still merge.
    """

    name: str
    label: "str | None" = field(default=None, compare=False)

    @classmethod
    def strategy(cls):
        """ Generate the colours of a few regions, by their names. """
        from hypothesis import strategies as st

        return st.sampled_from(("red", "green", "blue")).map(cls)

    def __post_init__(self):
        assert_isinstance(self.name, str)
        if self.label is not None:
            assert_isinstance(self.label, str)

    @property
    def legend_label(self) -> str:
        """ The name shown for this colour in a drawing legend. """
        return self.name if self.label is None else self.label

    def __repr__(self):
        label = "" if self.label is None else f", label={self.label!r}"
        return f"{factory_name(type(self))}({self.name!r}{label})"

    def to_tree(self):
        tree = super().to_tree()
        if self.label is not None:
            tree['label'] = self.label
        return tree

    @classmethod
    def from_tree(cls, tree):
        return cls(tree['name'], label=tree.get('label'))


def uncolour_state(state: dict) -> dict:
    """
    The pickled ``state`` of a wire or a type, with no colour as ``None``:
    a pickle from before ``None`` stood for no colour holds a
    :class:`Colour` named :data:`discopy.config.TRANSPARENT` instead.
    """
    for key in ('dom', 'cod'):
        colour = state.get(key)
        if isinstance(colour, Colour) and colour.name == TRANSPARENT:
            state[key] = None
        state.setdefault(key, None)
    return state


class Wire(cat.Ob):
    """
    A generating 1-cell with a colour on either side.

    Its :meth:`dagger` is a generator of its own, the colours swapped and
    printed with ``[::-1]``, as a daggered box is: equality and hashing count
    ``is_dagger``, so that a functor, which sends the dagger of a wire to the
    dagger of its image, sends equal wires to equal images.

    >>> red, green = Colour('red'), Colour('green')
    >>> x = Wire('x', red, green)
    >>> assert x.dagger() != Wire('x', green, red) and x.dagger().dagger() == x
    >>> assert str(x.dagger()) == "x[::-1]" and str(x) == "x"
    """

    def __init__(self, name: str, dom: Colour | None = None,
                 cod: Colour | None = None, is_dagger: bool = False):
        assert_isinstance(dom, (Colour, type(None)))
        assert_isinstance(cod, (Colour, type(None)))
        self.is_dagger = is_dagger
        self.dom, self.cod = dom, cod
        super().__init__(name)

    def __setstate__(self, state):
        uncolour_state(state)
        state.setdefault('is_dagger', False)
        super().__setstate__(state)

    def dagger(self):
        return type(self)(
            self.name, self.cod, self.dom, is_dagger=not self.is_dagger)

    def __getitem__(self, key):
        """
        The dagger of a wire, written ``x[::-1]`` as for a box, so that the
        string of a daggered wire reads back.

        >>> x = Wire('x')
        >>> assert eval(str(x.dagger())) == x.dagger() == x[::-1]
        """
        if key != slice(None, None, -1):
            raise TypeError(messages.ONLY_REVERSED.format(
                factory_name(type(self))))
        return self.dagger()

    def __eq__(self, other):
        return type(self) is type(other) and (
            self.name, self.dom, self.cod, self.is_dagger) == (
                other.name, other.dom, other.cod, other.is_dagger)

    def __hash__(self):
        return hash(
            (factory_key(type(self)),
             self.name, self.dom, self.cod, self.is_dagger))

    def __lt__(self, other):
        """
        Wires are ordered by name, then by ``repr``, which tells apart two
        unequal wires of one name, e.g. a wire and its dagger: of two unequal
        wires, exactly one is less than the other.

        >>> x = Wire('x')
        >>> assert Wire('w') < x < x.dagger() and not x.dagger() < x
        >>> assert x.dagger() > x and not x > x.dagger() and not x < x
        """
        return self != other\
            and (self.name, repr(self)) < (other.name, repr(other))

    def __repr__(self):
        colours = "" if self.dom is None and self.cod is None\
            else f", dom={self.dom!r}, cod={self.cod!r}"
        dagger = ", is_dagger=True" if self.is_dagger else ""
        return f"{factory_name(type(self))}({self.name!r}{colours}{dagger})"

    __str__ = cat.Box.__str__

    def to_tree(self):
        tree = super().to_tree()
        tree['factory'] = factory_name(type(self))
        if self.dom is not None:
            tree['dom'] = self.dom.to_tree()
        if self.cod is not None:
            tree['cod'] = self.cod.to_tree()
        if self.is_dagger:
            tree['is_dagger'] = True
        return tree

    @classmethod
    def from_tree(cls, tree):
        dom = from_tree(tree['dom']) if 'dom' in tree else None
        cod = from_tree(tree['cod']) if 'cod' in tree else None
        return cls(tree['name'], dom, cod,
                   is_dagger=tree.get('is_dagger', False))


class List[generator_factory](Monoid["List"], TestableGeneric):
    """
    The free monoid on a ``generator_factory``, i.e. lists of its instances
    with concatenation as :meth:`tensor` and the empty list as unit.

    ``List[X]`` is the free monoid on ``X`` the way ``Hypergraph[C]`` is the
    hypergraph category over a category ``C``, e.g. ``python.Function.ob``
    is ``List[type]``, the free monoid on Python's ``type``. It has a single,
    trivial colour: :class:`Ty` is the free *coloured* monoid, with
    :class:`Colour` boundaries on its :class:`Wire` generators, and
    :class:`Nat` the free monoid on a single generator.

    >>> assert List[int](2, 3) @ List[int](4) == List[int](2, 3, 4)
    >>> assert List[int](2, 3) ** 2 == List[int](2, 3, 2, 3)
    >>> assert List[int](2, 3)[1:] == List[int](3) and not List[int]()

    Note
    ----
    A list is a sequence of its length-one sublists, e.g.
    ``List[int](2, 3)[0] == List[int](2)``; the atoms themselves are its
    :attr:`inside`, e.g. ``List[int](2, 3).inside[0] == 2``.

    Note
    ----
    A ``List[X]`` is transparent, i.e. ``eval(repr(x)) == x``, exactly when
    the representation of an atom of ``X`` is an expression that evaluates
    back to it. :meth:`generator_repr` is what secures that for every
    generator the library uses: ``repr`` is such an expression for an
    ``int``, and for a class it is not — ``repr(int)`` is ``<class 'int'>``
    where the name ``int`` is what reads back — so the free monoid on
    Python's ``type``, i.e. :obj:`discopy.python.Ty`, prints its atoms with
    :func:`.factory_name` instead. A generator with no such representation
    at all, e.g. ``object()``, has no transparent list until it overrides the
    hook.

    >>> from discopy import monoidal
    >>> x = List[type](int, str)
    >>> assert repr(x) == "monoidal.List[type](int, str)"
    >>> assert eval(repr(x)) == x
    """
    @classmethod
    def checked_params(cls) -> tuple:
        """
        Python types, the only generators :meth:`generator_strategy` draws.
        """
        return (type, )

    def __init__(self, *inside):
        self.inside = inside

    def tensor(self, *others: "List") -> "List":
        for other in others:
            assert_isinstance(other, type(self))
        return type(self)(
            *self.inside, *(x for other in others for x in other.inside))

    def __matmul__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented  # This allows whiskering on the left.
        return self.tensor(other)

    def __len__(self) -> int:
        return len(self.inside)

    def unwind(self) -> "List":
        """
        The identity: a list has no winding, see :meth:`Ty.unwind`, so that
        a :class:`hypergraph.Hypergraph` over any free monoid labels its
        spiders the same way.

        >>> assert List[int](2).unwind() == List[int](2)
        """
        return self

    def __getitem__(self, key: int | slice) -> "List":
        if isinstance(key, slice):
            return type(self)(*self.inside[key])
        return type(self)(self.inside[key])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i:i + 1]

    def __pow__(self, n_times: int) -> "List":
        n_times = index(n_times)
        return self.tensor(*(n_times - 1) * [self]) if n_times > 0\
            else type(self)()

    def __eq__(self, other):
        return type(self) is type(other) and self.inside == other.inside

    def __hash__(self):
        """
        A list's generators can themselves be classes, `python.Ty` being
        `List[type]`, and a class hashes as its address: those take their
        :func:`discopy.utils.factory_key` here, as a setoid's leading
        component does.
        """
        return hash((factory_key(type(self)), tuple(
            factory_key(x) if isinstance(x, type) else x
            for x in self.inside)))

    @classmethod
    def generator_repr(cls, generator) -> str:
        """
        The representation of one atom, which :meth:`__repr__` joins.

        It is the ``repr`` of a generator whose own representation is an
        expression, e.g. an ``int``, and :func:`.factory_name` for a class,
        whose is not: ``repr(int)`` is ``<class 'int'>`` where ``int`` is
        the name that evaluates back to it. A generator with neither — one
        printing its address, say — overrides this to say what reads back
        to it, since nothing can be inferred from such a ``repr``.

        Example
        -------
        >>> assert List[int].generator_repr(2) == "2"
        >>> assert List[type].generator_repr(int) == "int"
        >>> assert List[type].generator_repr(cat.Ob) == "cat.Ob"
        """
        return factory_name(generator) if isinstance(generator, type)\
            else repr(generator)

    def __repr__(self):
        return factory_name(type(self))\
            + f"({', '.join(map(self.generator_repr, self.inside))})"

    @classmethod
    def generator_strategy(cls):
        """
        Generate the lists of one Python type, one of those
        :func:`testable.values` samples, the free monoid on any other
        generator having no strategy yet.

        >>> from discopy.python import Ty
        >>> generator = Ty.generator_strategy().example()
        >>> assert generator in map(Ty, testable.VALUES)
        """
        from hypothesis import strategies as st
        if cls.generator_factory is not type:
            raise NotImplementedError
        return st.sampled_from(tuple(testable.VALUES)).map(cls)

    @classmethod
    def strategy(cls, dom=None, cod=None):
        """
        Generate lists of at most three generators, each from the only
        object :obj:`None` to itself.
        """
        from hypothesis import strategies as st
        if (dom, cod) != (None, None):
            raise ValueError("A list has a single object, None.")
        return st.lists(cls.generator_strategy(), max_size=3).map(
            lambda xs: cls().tensor(*xs))


@factory
class Ty(cat.Ob, cat.FreeCategory, ColouredMonoid):
    """
    A type is a composable path of objects with :meth:`Ty.tensor`
    as concatenation.

    Parameters:
        inside : The objects inside the type (or their names).

    Tip
    ---
    Types can be instantiated with a name rather than object.

    >>> assert Ty('x') == Ty(cat.Ob('x'))

    Tip
    ---
    A type can be exponentiated by a natural number.

    >>> assert Ty('x') ** 3 == Ty('x', 'x', 'x')

    Tip
    ---
    Types can also be instantiated by keyword, passing the path of
    generators as ``inside=``; this is what the free-category machinery
    uses internally, while the variadic form above is the user-friendly
    ``Ty('x', 'y')`` API.

    >>> assert Ty(inside=(Wire('x'), Wire('y'))) == Ty('x', 'y')

    Note
    ----
    Types can be indexed and sliced using square brackets. Indexing behaves
    like that of strings, i.e. when we index a type we get a type back.
    The objects inside the type are still accessible using ``.inside``.

    >>> t = Ty(*"xyz")
    >>> assert t[0] == t[:1] == Ty('x')
    >>> assert t[0] != t.inside[0] == Wire('x')
    >>> assert t[1:] == t[-2:] == Ty('y', 'z')

    Note
    ----
    The ``name`` of a type is that of its class: its objects are printed by
    ``str``.

    >>> assert t.name == "Ty" and str(t) == "x @ y @ z"
    """
    ob = Colour
    generator_factory = Wire
    box_factory = classproperty(lambda cls: cls.generator_factory)
    """
    The generators of a type as a free category, i.e. its wires, a
    :class:`cat.Ob` that :meth:`cast` embeds as a type of length one.
    """

    @classmethod
    def typeparams(cls) -> tuple[type, type]:
        """
        The objects of types as a category are their colours, which a law
        quantifies over as ``None``, the colour of every type generated:
        no coloured type is generated yet.

        >>> assert Ty.typeparams() == (NoneType, Ty)
        """
        return NoneType, cls

    @classmethod
    def strategy(cls, **params):
        """
        Generate types by the :meth:`search` of the free monoid, i.e. from
        the identity and the composition of its rules with a wire as the
        generator, with less fuel than a diagram since its types are the
        boundaries of one.

        Parameters are those of :meth:`search`.

        >>> from hypothesis import find
        >>> assert len(find(Ty.strategy(), lambda x: len(x) > 1)) == 2
        """
        return cls.search(**{"fuel": 1, **params})

    @classmethod
    def generator_strategy(cls, *, dom=None, cod=None):
        """
        Generate the types of one wire, with no colour, as those of
        :meth:`typeparams` are: a coloured type is not generated yet, so
        the boundary is not read.
        """
        # pylint: disable=unused-argument  # no type generated is coloured
        return cls.generator_factory.strategy().map(cls)

    def cast_wire(self, x: str | cat.Ob) -> cat.Ob:
        """
        Turn a constructor argument into a ``self.generator_factory``.

        Old dumps and pickles used a plain ``cat.Ob``, with no colour, as
        the generators: upgrade it to ``Wire(x.name)`` for subclasses whose
        generators are plain ``Wire``.
        """
        if isinstance(x, self.generator_factory):
            return x
        if isinstance(x, str):
            return self.generator_factory(x)
        if self.generator_factory is Wire and type(x) is cat.Ob:
            return self.generator_factory(x.name)
        raise TypeError(messages.TYPE_ERROR.format(
            factory_name(self.generator_factory), factory_name(type(x))))

    def __init__(self, *inside: str | cat.Ob,
                 dom: Colour | None = None, cod: Colour | None = None,
                 _scan: bool = True, **kwargs):
        inside = tuple(kwargs.pop('inside', inside))
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}.")
        for obj in inside:
            assert_isinstance(obj, (str, self.generator_factory) + (
                (cat.Ob, ) if self.generator_factory is Wire else ()))
        inside = tuple(map(self.cast_wire, inside))
        if dom is None and inside:
            dom = inside[0].dom
        if cod is None and inside:
            cod = inside[-1].cod
        cat.FreeCategory.__init__(self, inside, dom, cod, _scan)
        cat.Ob.__init__(self, type(self).__name__)

    @classmethod
    def cast_ob(cls, x: Colour | str | None) -> Colour | None:
        """ No colour is ``None``, any other boundary a :class:`Colour`. """
        return None if x is None else super().cast_ob(x)

    @classmethod
    def id(cls, dom: Colour | None = None) -> Ty:
        """ The empty type on a colour, with no colour by default. """
        return cls.ar(inside=(), dom=dom, cod=dom, _scan=False)

    def tensor(self, *others: "Ty") -> Self:
        if any(not isinstance(other, self.factory) for other in others):
            return NotImplemented  # This allows whiskering on the left.
        return cat.FreeCategory.then(self, *others)

    then = tensor

    def __pow__(self, n_times: int) -> Self:
        n_times = index(n_times)
        if n_times <= 0:
            assert self.dom == self.cod
            return self.factory.id(self.dom)
        return self.tensor(*(n_times - 1) * [self])

    __iter__ = List.__iter__

    def __eq__(self, other):
        return type(self) is type(other) and self.inside == other.inside\
            and (self.dom, self.cod) == (other.dom, other.cod)

    def __hash__(self):
        return hash((factory_key(type(self)), self.inside, self.dom, self.cod))

    def __repr__(self):
        if not self.inside and self.dom is not None:
            return f"{factory_name(type(self))}.id({self.dom!r})"
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self.inside))})"

    @property
    def is_generator(self) -> bool:
        """ Whether a type is a single generating object. """
        return len(self.inside) == 1

    @property
    def generator(self) -> Wire | None:
        """ The single object inside a generator type. """
        return self.inside[0] if self.is_generator else None

    def count(self, obj: cat.Ob) -> int:
        """
        Counts the occurrence of a given object (or a type of length 1).

        Parameters:
            obj : The object to count.

        Example
        -------

        >>> x = Ty('x')
        >>> xs = x ** 5
        >>> assert xs.count(x) == xs.inside.count(x.inside[0])
        """
        obj, = obj.inside if isinstance(obj, Ty) else (obj, )
        return self.inside.count(obj)

    def unwind(self) -> "Ty":
        """
        Rotate an atomic type to winding number zero.

        This is the identity for monoidal types, which have no winding. It is
        overridden by :class:`rigid.Ty` to give a canonical representative for
        the compact quotient, i.e. the base type on which spiders are labelled.
        """
        return self

    def __str__(self):
        name = type(self).__name__
        if not self.inside:
            if self.dom is None:
                return f"{name}()"
            return f"{name}.id({self.dom})"
        parts = []
        for ob in self.inside:
            s = str(ob)
            parts.append(f'{name}("")' if s == '' else s)
        return ' @ '.join(parts)

    def __call__(self, function: Callable, offset: int | None = None):
        """
        The kappa abstraction of a Python function from a variable for each
        wire of the type to a term, see :class:`Kappa`, as a type of a
        closed category applied to a function is a lambda abstraction.

        Parameters:
            function : The body, as a function of the variables.
            offset : Passed to :class:`Kappa`.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> f = Box('f', x @ y, x)
        >>> term = (x @ y)(lambda a, b: Constant(f)(a, b))
        >>> assert term.eval() == f
        """
        return Kappa.from_callable(self, offset)(function)

    def atoms(self) -> list[Ty]:
        """
        The atoms of a type: those of its objects, each one with no adjoint
        nor delay as a type of its own, i.e. the generators its string names.

        >>> x, y = Ty('x'), Ty('y')
        >>> assert (x @ y @ x).atoms() == [x, y, x]
        """
        return [
            atom if isinstance(atom, Ty) else self.factory(atom.name)
            for ob in self.inside if isinstance(ob, cat.Ob)
            for atom in ob.atoms()]

    def __lt__(self, other):
        """
        Types are ordered by length first, then lexicographically on the
        objects inside, e.g. ``Ty('a') < Ty('b') < Ty('a', 'b')``, then by
        ``repr``, which tells apart the empty types of two colours: of two
        unequal types, exactly one is less than the other. The remaining
        comparisons are filled in by :func:`functools.total_ordering` on the
        :class:`cat.Ob` base class.

        >>> x, y, z = map(Ty, "xyz")
        >>> assert sorted([z, x @ y, x, y]) == [x, y, z, x @ y]
        >>> assert x < x[::-1] and x[::-1] > x and not x > x[::-1]
        >>> assert Ty.id(Colour('green')) < Ty.id(Colour('red'))
        """
        assert_isinstance(other, Ty)
        return self != other\
            and (len(self.inside), self.inside, repr(self))\
            < (len(other.inside), other.inside, repr(other))

    def __setstate__(self, state):
        if 'inside' not in state and "_objects" in state:
            state["inside"] = state['_objects']
            del state['_objects']
        cat.Ob.__setstate__(self, uncolour_state(state))

    def to_tree(self):
        tree = {
            'factory': factory_name(type(self)),
            'inside': [x.to_tree() for x in self.inside]}
        if not self.inside and self.dom is not None:
            tree['dom'] = self.dom.to_tree()
            tree['cod'] = self.cod.to_tree()
        return tree

    @classmethod
    def from_tree(cls, tree):
        if "inside" not in tree:
            warn("Outdated dumps", DeprecationWarning)
            return cls(*map(from_tree, tree['objects']))
        inside = tuple(map(from_tree, tree['inside']))
        # Old dumps used cat.Ob as the generators of monoidal.Ty.
        inside = tuple(
            cls.generator_factory(x.name) if type(x) is cat.Ob else x
            for x in inside)
        if inside:
            return cls(*inside)
        if 'dom' in tree:
            return cls(dom=from_tree(tree['dom']), cod=from_tree(tree['cod']))
        return cls()

    def to_drawing(self) -> "Ty":
        """
        The type of a drawing: a wire for each object, labelled by its
        ``drawing_name`` when it has one, e.g. ``4`` for ``Dim(4)``, else by
        its string.
        """
        if not self.inside:
            return Ty.id(self.dom)
        result = Ty(*(
            Wire(getattr(x, 'drawing_name', None) or str(x),
                 getattr(x, 'dom', None), getattr(x, 'cod', None))
            for x in self.inside))
        for new, old in zip(result.inside, self.inside):
            if getattr(old, "frame_boundary", False):
                new.frame_boundary = True
            for attr, default in (
                    WIRE_DRAWING_ATTRIBUTES | COLOUR_DRAWING_ATTRIBUTES
            ).items():
                setattr(new, attr, getattr(old, attr, default(new)))
        return result

    def wire_offsets(self) -> list:
        """
        The x-position of each wire of the type relative to the first, i.e. the
        sum of the cell widths ``max(1, right_margin)`` of the objects before
        it: each wire takes up at least a unit, more if its label is longer.
        A wire followed by a region carrying a ``width`` (e.g. the inside of
        a :class:`discopy.balanced.Ribbon`) is only that width away from the
        next wire, i.e. the region carries the width.

        >>> assert Ty('x', 'y').to_drawing().wire_offsets() == [0, 1]
        """
        offsets, total = [], 0
        for ob in self.inside:
            offsets.append(total)
            ribbon = getattr(ob, "ribbon", None)
            if ribbon is not None:
                total += ribbon.width
                continue
            cell_width = max(1, ob.right_margin)
            total += max(cell_width, 1 + getattr(ob, "min_right_margin", 0))
        return offsets


@factory
class Nat(abc.Nat, Ty):  # ty: ignore[invalid-generic-class]
    """
    ``Nat`` is a natural number ``n`` seen as a type with addition as
    tensor, i.e. the free monoid on one generator.

    Parameters
    ----------
    inside : int | tuple
        The natural number, or a tuple of generators whose length is taken.

    Example
    -------
    >>> assert Nat(1) @ Nat(2) == Nat(3)

    ``Nat`` is also a sequence over its unary encoding, so slicing, iterating
    and taking its length reads it the same way as any other :class:`Ty`.

    >>> assert len(Nat(3)) == 3 and list(Nat(3)) == 3 * [Nat(1)]
    >>> assert Nat(3)[:1] == Nat(1) == Nat(3)[0]

    Note
    ----
    If ``ob`` is ``Nat`` then :class:`Diagram` will automatically turn
    any ``n: int`` into ``Nat(n)``. Thus ``Nat`` never needs to be called,
    and the resulting diagrams live in a :class:`discopy.abc.PRO`.

    >>> @factory
    ... class Circuit(abc.PRO, Diagram):
    ...     ob = Nat
    >>> class Gate(Box, Circuit): ...
    >>> CX = Gate('CX', 2, 2)

    >>> assert CX @ 2 >> 2 @ CX == CX @ CX
    """
    ob = Colour
    generator_factory = int
    box_factory = None
    """ No generator of one colour is named, so there is no canonical
    instance of a law, see :meth:`discopy.testable.Axiom.canonical`. """

    @classmethod
    def generator_strategy(cls, *, dom=None, cod=None):
        """ Generate the one generator of the free monoid, ``Nat(1)``. """
        # pylint: disable=unused-argument  # a Nat has no colour
        from hypothesis import strategies as st

        return st.just(cls(1))

    def __init__(self, inside: int | tuple = 0, dom: Colour | None = None,
                 cod: Colour | None = None, _scan: bool = True):
        # pylint: disable=unused-argument  # a Nat has no colour
        abc.Nat.__init__(
            self, inside if isinstance(inside, int) else len(inside))
        self.dom = self.cod = None
        cat.Ob.__init__(self, type(self).__name__)

    def __setstate__(self, state: dict):
        if "n" not in state:
            state = {"n": len(state["_objects"])}
        uncolour_state(state)
        state.setdefault("name", type(self).__name__)
        cat.Ob.__setstate__(self, state)

    @property
    def inside(self):
        return self.n * (1, )

    def tensor(self, *others: "Nat") -> "Nat":
        for other in others:
            if not isinstance(other, Ty):
                return NotImplemented  # This allows whiskering on the left.
            assert_isinstance(self, other.factory)
            assert_isinstance(other, self.factory)
        return self.factory(self.n + sum(other.n for other in others))

    then = tensor

    def __repr__(self):
        return factory_name(type(self)) + f"({self.n})"

    def __eq__(self, other):
        return isinstance(other, self.factory) and self.n == other.n

    def __hash__(self):
        return hash(repr(self))

    def __pow__(self, n_times):
        return self.factory(index(n_times) * self.n)

    def delay(self, n_steps: int = 1) -> Self:
        """
        The identity: a type whose atoms are not wires has no time step, so
        its :attr:`head` and :attr:`tail` are itself too, see
        :class:`discopy.feedback.Ty`.

        Parameters:
            n_steps : The number of time steps to delay.

        Example
        -------
        >>> assert Nat(2).delay() == Nat(2).head == Nat(2).tail == Nat(2)
        """
        del n_steps
        return self

    head = tail = property(lambda self: self)

    def to_tree(self):
        return {'factory': factory_name(type(self)), 'n': self.n}

    @classmethod
    def from_tree(cls, tree):
        return cls(tree['n'])


@factory
class Dim(Ty):
    """
    A dimension is a tuple of positive integers
    with product ``@`` and unit ``Dim(1)``.

    Example
    -------
    >>> Dim(1) @ Dim(2) @ Dim(3)
    Dim(2, 3)
    >>> from discopy.utils import dumps, loads
    >>> assert loads(dumps(Dim(2, 3))) == Dim(2, 3)
    """
    generator_factory = int
    box_factory = None
    """ No generator of one colour is named, so there is no canonical
    instance of a law, see :meth:`discopy.testable.Axiom.canonical`. """

    @classmethod
    def generator_strategy(cls, *, dom=None, cod=None):
        """ Generate the dimensions of one wire, two or three. """
        # pylint: disable=unused-argument  # a Dim has one colour
        from hypothesis import strategies as st

        return st.integers(2, 3).map(cls)

    def __init__(self, *inside: int, dom=None, cod=None, _scan=True, **kwargs):
        inside = tuple(kwargs.pop('inside', inside))
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}.")
        for dim in inside:
            assert_isinstance(dim, int)
            if dim < 1:
                raise ValueError(f"Expected a positive dimension, got {dim}.")
        inside = tuple(dim for dim in inside if dim > 1)
        cat.FreeCategory.__init__(self, inside, dom, cod, _scan=False)
        cat.Ob.__init__(self, type(self).__name__)

    def __getitem__(self, key: int | slice) -> Self:
        if isinstance(key, slice):
            return self.factory(*self.inside[key])
        return self.factory(self.inside[key])

    def __repr__(self):
        return f"Dim({', '.join(map(repr, self.inside)) or '1'})"

    __str__ = __repr__

    @property
    def drawing_name(self) -> str:
        """
        The label of a wire of this dimension in a drawing: its numbers,
        without the ``Dim`` that :meth:`__repr__` prints.

        >>> Dim(4).drawing_name, Dim(4, 4).drawing_name, Dim(1).drawing_name
        ('4', '4, 4', '1')
        """
        return ", ".join(map(str, self.inside)) or "1"

    delay, head, tail = Nat.delay, Nat.head, Nat.tail

    def to_tree(self) -> dict:
        return {'factory': factory_name(type(self)),
                'inside': list(self.inside)}

    @classmethod
    def from_tree(cls, tree: dict) -> "Dim":
        return cls(*tree['inside'])


class Layer(cat.Box, ColouredMonoid):
    """
    A layer is a tensor product of boxes and plumbing, i.e. non-empty types.

    Parameters:
        inside : Boxes and plumbing, with at least one box.
        normalise : Whether to type check the components and normalise them,
            i.e. remove empty plumbing and tensor consecutive types. The
            methods below build layers that satisfy this by construction and
            pass ``False``, which skips every pass over ``inside`` so that
            tensoring ``n`` layers takes linear rather than quadratic time.
    """
    ob = Ty

    def __setstate__(self, state):
        if 'boxes_or_types' not in state:
            state['boxes_or_types'] = tuple(
                state[key] for key in ['_left', '_box', '_right'])
            del state['_left'], state['_box'], state['_right']
        state['boxes_or_types'] = type(self).normalise(
            state['boxes_or_types'])
        super().__setstate__(state)
        for key in [key for key in vars(self) if isinstance(
                getattr(type(self), key, None), cached_property)]:
            delattr(self, key)

    def __init__(self, *inside: "Ty | Box", normalise: bool = True):
        if normalise:
            type(self).check(inside)
            inside = type(self).normalise(inside)
        self.boxes_or_types = tuple(inside)
        if normalise and not self.boxes:
            raise ValueError(messages.LAYERS_MUST_HAVE_A_BOX)

    data, is_dagger = None, False

    @classmethod
    def id(cls, dom: Ty | None = None) -> "Layer":
        """
        There is no identity layer: a layer has at least one box, and a
        layer of empty plumbing would denote the identity diagram, which is
        the empty sequence of layers rather than a one-element one.

        Whiskering merges a type into the boundary instead, see
        :meth:`whisker` and :meth:`tensor`.
        """
        raise ValueError(messages.LAYERS_MUST_HAVE_A_BOX)

    def then(self, *others: cat.Arrow) -> cat.Arrow:
        """
        Layers do not compose with layers: a diagram is the sequence of its
        layers, so it is diagrams that compose, e.g. ``f >> g`` and not
        ``Layer(f) >> Layer(g)``. Identities compose as in :mod:`cat`.

        Parameters:
            others : The arrows to compose.

        Raises:
            AxiomError : When the composite would hold two layers, saying to
                compose diagrams instead.

        Example
        -------
        >>> x = Ty('x')
        >>> f = Box('f', x, x)
        >>> Layer(f) >> Layer(f)
        Traceback (most recent call last):
        ...
        discopy.utils.AxiomError: Compose diagrams, not layers: f >> f.
        >>> assert (Layer(f) >> cat.Arrow.id(x)).inside == (Layer(f), )
        """
        result = cat.Arrow.then(self, *others)
        if sum(isinstance(box, Layer) for box in result.inside) > 1:
            raise AxiomError(messages.LAYERS_DO_NOT_COMPOSE.format(
                " >> ".join(map(str, result.inside))))
        return result

    def __rrshift__(self, other: cat.Arrow) -> cat.Arrow:
        """
        The composite ``other >> self``, called before ``other``'s own for an
        arrow of :mod:`cat`, so that it cannot hold another layer either.

        Example
        -------
        >>> x = Ty('x')
        >>> f = Box('f', x, x)
        >>> cat.Arrow.id(x) >> Layer(f) >> Layer(f)
        Traceback (most recent call last):
        ...
        discopy.utils.AxiomError: Compose diagrams, not layers: f >> f.
        """
        return Layer.then(other, self) if isinstance(other, cat.Arrow)\
            else NotImplemented

    __lshift__ = __rrshift__

    @classmethod
    def whisker(cls, other: "Ty | Box | Layer") -> "Ty | Layer":
        """
        A type stays a type, which :meth:`tensor` merges into the boundary;
        anything else is embedded as a layer.

        Parameters:
            other : The type, box or layer to be tensored on either side.
        """
        return other if isinstance(other, (cls, cls.ob)) else cls(
            other, normalise=False)  # ty: ignore[invalid-argument-type]

    @classmethod
    def unit(cls, colour: Colour | None = None) -> Ty:
        """
        The unit of the layer product, i.e. the empty type on a colour.

        A layer has at least one box, so the unit is not one: it lands in the
        types rather than in the layers, which :meth:`tensor` accepts.

        Example
        -------
        >>> x = Ty('x')
        >>> f = Box('f', x, x)
        >>> unit, layer = Layer.unit(), Layer(f)
        >>> assert unit @ layer == layer == layer @ unit
        """
        return cls.ob.unit(colour)

    @cached_property
    def inside(self):
        return (self, )

    @cached_property
    def name(self):
        return " @ ".join(map(str, self.boxes_or_types))

    @cached_property
    def dom(self):
        pieces = [x if isinstance(x, Ty) else x.dom
                  for x in self.boxes_or_types]
        return pieces[0][:0].tensor(*pieces)

    @cached_property
    def cod(self):
        pieces = [x if isinstance(x, Ty) else x.cod
                  for x in self.boxes_or_types]
        return pieces[0][:0].tensor(*pieces)

    plumbing: type | tuple[type, ...] = Ty

    @staticmethod
    def check(inside):
        """
        Check that the components are types and boxes whose domains and
        codomains tensor, i.e. one linear pass comparing the colours of
        each adjacent pair on both rows.
        """
        for value in inside:
            assert_isinstance(value, (Ty, Box))
        rows = [(x, x) if isinstance(x, Ty) else (x.dom, x.cod)
                for x in inside]
        for (dom, cod), (dom_, cod_) in zip(rows, rows[1:]):
            if dom.cod != dom_.dom:
                raise AxiomError(messages.NOT_COMPOSABLE.format(
                    dom, dom_, dom.cod, dom_.dom))
            if cod.cod != cod_.dom:
                raise AxiomError(messages.NOT_COMPOSABLE.format(
                    cod, cod_, cod.cod, cod_.dom))

    @classmethod
    def is_plumbing_box(cls, value: Ty | Box) -> bool:
        """
        Whether a component of a layer is plumbing other than a type, e.g. a
        :class:`discopy.symmetric.Permutation` in a symmetric layer, none in
        a monoidal one.

        Parameters:
            value : The component of a layer.
        """
        return isinstance(value, cls.plumbing) and not isinstance(value, Ty)

    @classmethod
    def is_plumbing_component(cls, value: Ty | Box) -> bool:
        """
        Whether a component of a layer is plumbing, i.e. a type or a box of
        plumbing, see :meth:`is_plumbing_box`.

        Parameters:
            value : The component of a layer.
        """
        return isinstance(value, Ty) or cls.is_plumbing_box(value)

    @classmethod
    def normalise(cls, inside):
        """
        The canonical word of boxes and non-empty plumbing: either we
        append or we tensor with the last element. Plumbing that tensors
        into an identity, e.g. a spider with one leg on each side, leaves
        its type behind, and any other diagram leaves the two apart.
        """
        result = []
        for value in (x for x in inside if not isinstance(x, Ty) or x):
            if not result or not cls.is_plumbing_component(value)\
                    or not cls.is_plumbing_component(result[-1]):
                result.append(value)
                continue
            merged = result[-1] @ value
            if isinstance(merged, (Ty, Box)):
                result[-1] = merged
            elif not merged.inside:
                result[-1] = merged.dom
            else:
                result.append(value)
        return tuple(result)

    def __iter__(self):
        for box_or_typ in self.boxes_or_types:
            yield box_or_typ

    @property
    def boxes(self):
        return [
            box_or_typ for box_or_typ in self.boxes_or_types
            if isinstance(box_or_typ, Box)]

    def atoms(self) -> list:
        """ The atoms of a layer: those of its boxes and types. """
        return [atom for box_or_typ in self.boxes_or_types
                for atom in box_or_typ.atoms()]

    @property
    def size(self):
        return sum(box.size for box in self.boxes)

    def __getitem__(self, key):
        return self.boxes_or_types[key]

    def __eq__(self, other):
        return isinstance(other, type(self)) and tuple(self) == tuple(other)

    def __hash__(self):
        return hash(tuple(self))

    def __repr__(self):
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self))})"

    def tensor(self, *others: "Ty | Box | Layer") -> "Layer":
        """
        Tensor more layers on the right, normalising each common boundary.

        A type is merged into the boundary rather than embedded as a layer,
        so that whiskering never builds one out of empty plumbing.

        Parameters:
            others : The types, boxes or layers to tensor on the right.

        Example
        -------
        >>> x = Ty('x')
        >>> f, g, h = (Box(name, x, x) for name in "fgh")
        >>> assert Layer(f).tensor(Layer(g), Layer(h)) == Layer(f, g, h)
        >>> assert Layer(f).tensor(x, Layer(g)) == Layer(f, x, g)
        """
        factory = type(self)
        inside = list(self.boxes_or_types)
        for other in others:
            other = factory.whisker(other)
            pieces = (other, ) if isinstance(other, self.ob)\
                else other.boxes_or_types
            factory.check((inside[-1], pieces[0]))
            inside[-1:] = [
                *factory.normalise((inside[-1], pieces[0])), *pieces[1:]]
        return factory(*inside, normalise=False)

    def __rmatmul__(self, other):
        other = type(self).whisker(other)
        if isinstance(other, self.ob):
            type(self).check((other, self[0]))
            return type(self)(
                *type(self).normalise((other, self[0])), *self[1:],
                normalise=False)
        return other.tensor(self)  # ty: ignore[invalid-argument-type]

    @property
    def free_symbols(self) -> "set[sympy.Symbol]":
        return {x for box in self.boxes for x in box.free_symbols}

    def subs(self, *args) -> "Layer":
        return type(self)(*(
            x if self.is_plumbing_component(x) else x.subs(*args)
            for x in self),
            normalise=False)

    @property
    def is_generator(self):
        return len(self.boxes_or_types) == 1\
            and isinstance(self.boxes_or_types[0], Box)

    @property
    def generator(self):
        return self.boxes_or_types[0] if self.is_generator else None

    def dagger(self) -> "Layer":
        return type(self)(*(
            x if isinstance(x, Ty) else x.dagger() for x in self),
            normalise=False)

    @property
    def boxes_and_offsets(self) -> "list[tuple[Box, int]]":
        """
        The offsets of each box inside the layer.

        Example
        -------
        >>> a, b, c, d, e = map(Ty, "abcde")
        >>> f, g = Box('f', a, b), Box('g', c, d)
        >>> assert Layer(e, f, e, g, e).boxes_and_offsets == [(f, 1), (g, 3)]
        """
        result, offset = [], 0
        for box_or_typ in self.boxes_or_types:
            if isinstance(box_or_typ, Ty):
                offset += len(box_or_typ)
            else:
                result.append((box_or_typ, offset))
                offset += len(box_or_typ.cod)
        return result

    def merge(self, other: "Layer") -> "Layer":
        """
        Merge two layers into one or raise :class:`AxiomError`,
        used by :meth:`Diagram.foliation`: two layers merge when their
        composite is one layer up to interchangers alone, whatever rewrites
        the level of the diagram adds, e.g. a snake is never yanked.

        Parameters:
            other : The other layer with which to merge.

        Example
        -------
        >>> a, b, c, d, e = map(Ty, "abcde")
        >>> f, g = Box('f', a, b), Box('g', c, d)
        >>> layer0 = Layer(e,  f,  e @ c @ e)
        >>> layer1 = Layer(e @ b @ e,  g,  e)
        >>> assert layer0.merge(layer1) == Layer(e, f, e, g, e)
        """
        assert_iscomposable(self, other)
        staircases = self.boxes[0].ar(
            (self, other), self.dom, other.cod).to_staircases()
        try:
            diagram = Diagram.normal_form(
                staircases, normalize=Diagram.normalize)
        except NotImplementedError as exception:  # Eckmann-Hilton argument.
            diagram = exception.last_step
        boxes_or_types, offset = [self.dom[:0]], 0
        for layer, box, box_offset in zip(
                diagram.inside, diagram.boxes, diagram.offsets):
            if box_offset < offset:
                raise AxiomError(
                    messages.NOT_MERGEABLE.format(self, other))
            boxes_or_types[-1] @= layer.dom[offset:box_offset]
            boxes_or_types += [box, box.cod[len(box.cod):]]
            offset = box_offset + len(box.cod)
        boxes_or_types[-1] @= diagram.cod[offset:]
        return type(self)(*boxes_or_types)

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: type(self)(*(
            x if self.is_plumbing_component(x)
            else x.lambdify(*symbols, **kwargs)(*xs) for x in self),
            normalise=False)

    def to_tree(self) -> dict:
        return dict(factory=factory_name(type(self)),
                    inside=[x.to_tree() for x in self])

    @classmethod
    def from_tree(cls, tree: dict) -> "Layer":
        return cls(*(map(from_tree, tree['inside'])))


@factory
class Diagram(cat.Arrow, DaggerMonoidalCategory, RichDisplay):
    """
    A diagram is a tuple of composable layers :code:`inside` with a pair of
    types :code:`dom` and :code:`cod` as domain and codomain.

    Parameters:
        inside : The layers of the diagram.
        dom : The domain of the diagram, i.e. its input.
        cod : The codomain of the diagram, i.e. its output.

    Attributes:
        draw_left_to_right : Whether the diagrams of the class are drawn left
            to right rather than top to bottom, a property of the category
            rather than of any diagram: see :class:`Bubble` for diagrams of
            one direction drawn inside the boxes of another.

    .. admonition:: Summary

        .. autosummary::

            tensor
            boxes
            offsets
            draw
            interchange
            normalize
            normal_form
    """
    bifunctoriality = MonoidalCategory.bifunctoriality.failing(
        messages.NO_PLANAR_NORMAL_FORM)
    dagger_monoidality = DaggerMonoidalCategory.dagger_monoidality.failing(
        messages.NO_PLANAR_NORMAL_FORM)

    box_factory: ClassVar[type["Box"]]
    sum_factory: ClassVar[type[formal_sum.Sum]]
    bubble_factory: ClassVar[type["Bubble"]]
    functor_factory: ClassVar[type["Functor"]]
    draw_left_to_right: ClassVar[bool] = False
    ob = Ty
    layer_factory = Layer

    def __setstate__(self, state):
        if 'inside' not in state:  # Backward compatibility
            state |= {
                'dom': state['_dom'], 'cod': state['_cod'],
                'inside': tuple(state['_layers'])}
        super().__setstate__(state)

    def __init__(
            self, inside: tuple[Layer, ...], dom: Ty, cod: Ty, _scan=True):
        inside = tuple(inside)
        if _scan:
            for layer in inside:
                assert_isinstance(layer, Layer)
                if not layer.boxes:
                    raise ValueError(messages.LAYERS_MUST_HAVE_A_BOX)
        super().__init__(inside, dom, cod, _scan=_scan)

    @classproperty
    # pylint: disable-next=no-self-argument  # a property of the class
    def structural_boxes(cls) -> tuple[type[Box], ...]:
        """
        The structural boxes of the free category, i.e. the box classes
        among its factories that draw themselves with a ``strategy`` of
        their own rather than that of its :attr:`box_factory`, e.g.
        :class:`discopy.braided.Braid` for
        :attr:`discopy.braided.Diagram.braid_factory`. A box class of
        another category, e.g. the braid a symmetric diagram inherits from
        a balanced one, and a factory that is a method rather than a class
        are left out. :func:`discopy.testable.assert_structural_boxes`
        checks that each one is written down and read back.

        Example
        -------
        >>> assert Diagram.structural_boxes == ()
        >>> from discopy import braided, symmetric
        >>> assert braided.Diagram.structural_boxes == (braided.Braid, )
        >>> assert braided.Braid not in symmetric.Diagram.structural_boxes
        """
        factories = {
            name: value
            # pylint: disable-next=no-member  # a class has a __mro__
            for base in reversed(cls.__mro__)
            for name, value in vars(base).items()
            if name.endswith("_factory")}
        free = cls.box_factory.strategy.__func__
        return tuple(dict.fromkeys(
            factory for factory in factories.values()
            if isinstance(factory, type) and issubclass(factory, cls)
            and factory.strategy.__func__ is not free))

    @property
    def size(self):
        return sum(box.size for box in self.inside)

    @property
    def is_generator(self):
        """ Whether a `Diagram` is a generator, i.e. a single box. """
        return len(self) == 1 and self.inside[0].is_generator

    @property
    def generator(self):
        """ The single box in a generator `Diagram`. """
        return self.inside[0].generator if self.is_generator else None

    @classmethod
    def from_callable(
            cls, dom: Ty, cod: Ty) -> "Callable[[Callable], Diagram]":
        """
        Define a diagram using the standard syntax for Python functions.

        Note that we can specify the offset as argument.

        Example
        -------
        >>> x = Ty('x')
        >>> cup, cap = Box('cup', x @ x, Ty()), Box('cap', Ty(), x @ x)
        >>> @Diagram.from_callable(x, x)
        ... def snake(left):
        ...     middle, right = cap(offset=1)
        ...     cup(left, middle)
        ...     return right
        >>> snake.draw(doctest='docs/_static/monoidal/diagramize.svg')

        .. image:: /_static/monoidal/diagramize.svg
            :align: center
        """
        def decorator(func):
            graph = hypergraph.Hypergraph[
                cls.ar].from_callable(dom, cod)(func)
            return graph.to_diagram()

        return decorator

    def tensor(self, *others: "Diagram") -> "Diagram":
        """
        Parallel composition, called using :code:`@`.

        Parameters:
            others : The other diagrams to tensor.

        Important
        ---------
        The definition of tensor is biased to the left, i.e.::

            self @ other == self @ other.dom >> self.cod @ other

        Each layer of the ``i``-th diagram is whiskered by the codomains of
        the diagrams before it and the domains of those after it. A layer
        whose plumbing whiskers into the identity, e.g. a frobenius spider
        with one leg on each side, has no box left and is dropped, since the
        identity is the empty sequence of layers.

        Example
        -------
        >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
        >>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
        >>> assert f0 @ f1 == f0.tensor(f1) == f0 @ Id(z) >> Id(y) @ f1
        >>> assert f0 @ f1 @ f0 == f0.tensor(f1, f0)

        >>> (f0 @ f1).draw(
        ...     doctest='docs/_static/monoidal/tensor-example.svg')

        .. image:: /_static/monoidal/tensor-example.svg
            :align: center
        """
        if not others:
            return self
        for other in others:
            if isinstance(other, formal_sum.Sum):
                return self.sum_factory((self, )).tensor(*others)
            assert_isinstance(other, self.ar)
            assert_isinstance(self, other.ar)
        suffixes = [others[-1].dom]
        for other in reversed(others[:-1]):
            suffixes.append(other.dom @ suffixes[-1])
        dom, prefix, inside = self.dom @ suffixes[-1], None, []
        for diagram in (self, ) + others:
            suffix = suffixes.pop() if suffixes else None
            if prefix and suffix:
                whiskered = (
                    prefix @ layer @ suffix for layer in diagram.inside)
            elif prefix:
                whiskered = (prefix @ layer for layer in diagram.inside)
            elif suffix:
                whiskered = (layer @ suffix for layer in diagram.inside)
            else:
                whiskered = diagram.inside
            inside += [layer for layer in whiskered if layer.boxes]
            prefix = diagram.cod if prefix is None else prefix @ diagram.cod
        return self.ar(tuple(inside), dom, prefix, _scan=False)

    @property
    def boxes(self) -> "list[Box]":
        """ The boxes in each layer of the diagram. """
        return [box for layer in self.inside for box in layer.boxes]

    @property
    def offsets(self) -> list[int]:
        """ The offset of a box is the length of the type on its left. """
        return [
            offset
            for layer in self.inside
            for _, offset in layer.boxes_and_offsets]

    @property
    def width(self):
        """
        The width of a diagram, i.e. the maximum number of parallel wires.

        Example
        -------
        >>> x = Ty('x')
        >>> f = Box('f', x, x ** 4)
        >>> diagram = f @ x ** 2 >> x ** 2 @ f.dagger()
        >>> assert diagram.width == 6 and Diagram.id(x ** 2).width == 2
        """
        return max(
            len(self.dom),
            max((len(layer.cod) for layer in self), default=0))

    def encode(self) -> "tuple[Ty, list[list[tuple[Box, int]]]]":
        """
        Compact encoding of a diagram as its domain and its layers, each the
        list of its boxes with their offsets, see
        :attr:`Layer.boxes_and_offsets`: the offset of a box is the length of
        the type on its left, the boxes on its left counted by their
        codomain, so that flattening the layers gives the offsets of
        :meth:`to_staircases`.

        Example
        -------
        >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
        >>> f0, f1, g = Box('f0', x, y), Box('f1', z, w), Box('g', y @ w, y)
        >>> diagram = f0 @ f1 >> g
        >>> dom, layers = diagram.encode()
        >>> assert dom == x @ z
        >>> assert layers == [[(f0, 0)], [(f1, 1)], [(g, 0)]]
        >>> assert diagram == Diagram.decode(*diagram.encode())
        >>> diagram.draw(doctest='docs/_static/monoidal/arrow-example.svg')

        .. image:: /_static/monoidal/arrow-example.svg
            :align: center

        A layer with several boxes, as :meth:`foliation` makes, is one list:

        >>> foliated = diagram.foliation()
        >>> assert foliated.encode()[1] == [[(f0, 0), (f1, 1)], [(g, 0)]]
        >>> assert foliated == Diagram.decode(*foliated.encode())
        """
        return self.dom, [layer.boxes_and_offsets for layer in self.inside]

    @classmethod
    def decode(
            cls,
            dom: Ty,
            boxes_and_offsets: Iterable[
                "tuple[Box, int] | list[tuple[Box, int]]"] | None = None,
            boxes: "list[Box] | None" = None,
            offsets: list[int] | None = None,
            cod: Ty | None = None) -> "Diagram":
        """
        Turn a domain and layers of boxes and offsets into a diagram, the
        inverse of :meth:`encode`.

        Parameters:
            dom : The domain of the diagram.
            cod : The codomain of the diagram.
            boxes_and_offsets : The layers of the diagram, each a list of
                ``(box, offset)`` as :meth:`encode` writes them, or a single
                ``(box, offset)`` for a layer with one box.
            boxes : The list of boxes, one per layer.
            offsets : The list of offsets, one per layer.

        Example
        -------
        >>> x, y, z, w = map(Ty, "xyzw")
        >>> f, g = Box('f', x, y), Box('g', z, w)
        >>> assert f @ z >> y @ g == Diagram.decode(
        ...     dom=x @ z, cod=y @ w, boxes=[f, g], offsets=[0, 1])
        >>> assert f @ z >> y @ g == Diagram.decode(x @ z, [(f, 0), (g, 1)])
        >>> assert (f @ g).foliation() == Diagram.decode(
        ...     x @ z, [[(f, 0), (g, 1)]])

        Note
        ----
        If ``boxes_and_offsets is None``
        then we set it to ``zip(boxes, offsets)``.
        """
        if boxes_and_offsets is None:
            if boxes is None or offsets is None:
                raise ValueError(messages.MISSING_BOXES_AND_OFFSETS)
            boxes_and_offsets = zip(boxes, offsets)
        diagram = cls.id(dom)
        for layer in boxes_and_offsets:
            is_pair = layer and not isinstance(layer[0], tuple | list)
            pairs = cast(
                "list[tuple[Box, int]]", [layer] if is_pair else layer)
            diagram = diagram >> cls.decode_layer(diagram.cod, pairs)
        if cod is not None:
            assert_iscomposable(diagram, cls.id(cod))
        return diagram

    @classmethod
    def decode_layer(
            cls, dom: Ty, boxes_and_offsets: "list[tuple[Box, int]]"
    ) -> "Diagram":
        """
        The diagram of one layer, given its domain and its boxes with their
        offsets as :attr:`Layer.boxes_and_offsets` writes them.

        Parameters:
            dom : The domain of the layer.
            boxes_and_offsets : The boxes of the layer, left to right.

        Example
        -------
        >>> x, y = map(Ty, "xy")
        >>> f, g = Box('f', x, y), Box('g', x, y)
        >>> layer = Diagram.decode_layer(x @ x @ x, [(f, 0), (g, 2)])
        >>> assert layer.inside == (Layer(f, x, g), )
        >>> assert Diagram.decode_layer(x @ x, [(g, 1)]) == x @ g
        """
        if len(boxes_and_offsets) == 1:
            (box, offset), = boxes_and_offsets
            return dom[:offset] @ box @ dom[offset + len(box.dom):]
        inside, position, end = [], 0, 0
        for box, offset in boxes_and_offsets:
            if offset < end:
                raise ValueError(messages.OVERLAPPING_BOXES.format(
                    box, offset, end))
            inside += [dom[position:position + offset - end], box]
            position += offset - end + len(box.dom)
            end = offset + len(box.cod)
        layer = cls.layer_factory(*inside, dom[position:])
        return cls((layer, ), layer.dom, layer.cod)

    def to_drawing(self, functor_factory=None) -> Drawing:
        """ Called before :meth:`Diagram.draw`. """
        ob = ar = lambda x: x.to_drawing()
        dom = self.ar
        cod = Drawing
        return (functor_factory or Functor)(ob, ar, dom, cod)(self)

    def to_typst(self, **params):
        """
        Return a Typst Document AST for this diagram.

        Parameters
        ----------
        params : Passed to :class:`Typst`.

        Returns
        -------
        doc : Document
            A Typst Document AST with imports and a ``cetz.canvas`` block.

        Example
        -------
        >>> from discopy.monoidal import Ty, Box
        >>> f = Box('f', Ty('x'), Ty('y'))
        >>> doc = f.to_typst()
        >>> source = doc.render()
        >>> 'cetz' in source and 'canvas' in source
        True
        """
        from discopy.drawing import to_typst
        return to_typst(self, **params)

    def to_map(self) -> "CMap":
        """
        Translate a diagram into a combinatorial map.

        A map does not say in which face a closed component lies, so it
        identifies diagrams up to moving a scalar across a wire, as
        :meth:`to_hypergraph` does. From braided categories on, a scalar
        slides past a wire anyway, by naturality of the braiding; below, the
        map presents the free spacial category, see :mod:`discopy.spacial`
        and :meth:`discopy.cmap.CMap.match`, rather than the free category.

        >>> x, c = Ty('x'), Box('c', Ty(), Ty())
        >>> assert c @ x != x @ c and (c @ x).to_map() == (x @ c).to_map()
        """
        return cmap.CMap[self.ar].from_diagram(self)

    def to_term(self):
        """
        Read a diagram with exactly one output as a term of the planar
        linear kappa calculus, see :class:`TermBase`: a variable for each
        wire, ``x0, ..., x{n - 1}`` for the inputs, and a ``let`` for each
        box in the order of the layers, but the last when it gives the
        output, so that :meth:`TermBase.eval` gives the
        diagram back, box for box.

        A term has one output: a diagram with any other number of them is
        refused. Every box is a constant, structural ones included, e.g. a
        swap, and no wire is ever copied, discarded or crossed. Closed
        Markov diagrams read as lambda terms instead, see
        :meth:`discopy.closed.Diagram.to_term`.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> f, g = Box('f', x, y @ y), Box('g', y @ y, x)
        >>> print((f >> g).to_term())
        x(lambda x0: let(f(x0), lambda x1, x2: g(x1, x2)))
        >>> e = Box('e', y, Ty())
        >>> print((f >> y @ e >> Box('h', y, x)).to_term())
        x(lambda x0: let(f(x0), lambda x1, x2: \
let(e(x2), lambda: h(x1), offset=1)))
        >>> assert (f >> g).to_term().eval() == f >> g
        """
        return self.term_factory.from_diagram(self)

    def to_staircases(self):
        """
        Splits layers with more than one box into staircases, i.e. flattens
        the layers of :meth:`encode` and decodes each box as its own layer.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> f0, f1 = Box('f0', x, y), Box('f1', y, x)
        >>> diagram = f0 @ y >> y @ f1
        >>> print(diagram.foliation())
        f0 @ f1
        >>> print(diagram.foliation().to_staircases())
        f0 @ y >> y @ f1
        """
        dom, layers = self.encode()
        return self.ar.decode(dom, [
            pair for layer in layers for pair in layer])

    @axiom
    def hypergraph_identity[X: Obj[Ty]](
            cls, x: Var[Ty, X]) -> testable.Equation[hypergraph.Hypergraph]:
        """ The hypergraph of an identity is the identity hypergraph. """
        return cat.Equation(
            cls.id(x).to_hypergraph(), hypergraph.Hypergraph[cls].id(x))

    @axiom
    def hypergraph_composition[A: Obj[Ty], B: Obj[Ty], C: Obj[Ty]](
            cls, f: Hom[Self, A, B], g: Hom[Self, B, C]
    ) -> testable.Equation[hypergraph.Hypergraph]:
        """ The hypergraph of a composite composes the hypergraphs. """
        return cat.Equation(
            (f >> g).to_hypergraph(), f.to_hypergraph() >> g.to_hypergraph())

    @axiom
    def hypergraph_tensor[A: Obj[Ty], B: Obj[Ty], C: Obj[Ty], D: Obj[Ty]](
            cls, f: Hom[Self, A, B], g: Hom[Self, C, D]
    ) -> testable.Equation[hypergraph.Hypergraph]:
        """ The hypergraph of a tensor tensors the hypergraphs. """
        return cat.Equation(
            (f @ g).to_hypergraph(), f.to_hypergraph() @ g.to_hypergraph())

    @axiom
    def hypergraph_section[A: Obj[Ty], B: Obj[Ty]](
            cls, f: Hom[Self, A, B]
    ) -> testable.Equation[hypergraph.Hypergraph]:
        """ Decoding a hypergraph gives a diagram that encodes back to it. """
        graph = f.to_hypergraph()
        return cat.Equation(graph.to_diagram().to_hypergraph(), graph)

    @axiom
    def map_agreement[A: Obj[Ty], B: Obj[Ty]](
            cls, f: Hom[Self, A, B]
    ) -> testable.Equation[hypergraph.Hypergraph]:
        """ Encoding through a map or directly gives the same hypergraph. """
        return cat.Equation(f.to_map().to_hypergraph(), f.to_hypergraph())

    @axiom
    def drawing_dagger_involution[A: Obj[Ty], B: Obj[Ty]](
            cls, f: Hom[Self, A, B]) -> testable.Equation[Drawing]:
        """ Reflecting the drawing of a diagram twice gives it back. """
        drawing = f.to_drawing()
        return cat.Equation(drawing.dagger().dagger(), drawing)

    @axiom
    def foliation_idempotence[A: Obj[Ty], B: Obj[Ty]](
            cls, f: Hom[Self, A, B]) -> testable.Equation[Hom[Self, A, B]]:
        """ A foliation is its own foliation, on the nose. """
        foliated = f.foliation()
        return cat.Equation(foliated.foliation(), foliated)

    @classmethod
    def equation_factory(cls, *terms: "Diagram") -> "Equation":
        """
        The equation of a law, compared on the nose. A hypergraph is faithful
        only from :class:`discopy.symmetric.Diagram` on, whose
        ``equation_factory`` compares up to it: below, it is a quotient that
        forgets the regions of a planar diagram and the side of its scalars,
        identifies a braid with its inverse, a twist with the identity and
        the adjoints of a type, so a planar law is compared syntactically
        until each doctrine has a normal form of its own.

        >>> x = Ty('x')
        >>> f, g = Box('f', Ty(), x), Box('g', x, Ty())
        >>> assert not Diagram.equation_factory(f @ g, g >> f)
        >>> assert Diagram.equation_factory(f @ g, f @ g)
        """
        return Equation(*terms)

    def to_hypergraph(self) -> "Hypergraph":
        """
        Translate a planar diagram into a hypergraph.

        The offset of each *state* (a box with an empty domain) is recorded on
        the hypergraph, so :meth:`discopy.hypergraph.Hypergraph.to_diagram` can
        place it back without a swap. A state has no input wires for the
        boundary scan to follow, so without its offset that scan would default
        every state to the left and then need swaps to reorder them -- which a
        category without symmetry (e.g. a formal grammar) does not have.
        The offsets are recorded for the states before the first box that
        the functor into the hypergraph turns into wiring, e.g. a swap, a cup
        or a copy: up to there the boxes of the hypergraph are those of the
        diagram, in order, and so are the open wires, while after it an
        offset counts wires the hypergraph no longer has, so the states that
        follow are left to the decoder.

        Like :meth:`to_map`, it identifies diagrams up to moving a closed
        component, i.e. a scalar, across a wire, which below braided
        categories presents the free spacial category, see
        :mod:`discopy.spacial`. Decoding gives back a diagram with the same
        hypergraph, i.e. ``d.to_hypergraph().to_diagram().to_hypergraph()``
        is ``d.to_hypergraph()``, and that diagram is ``d`` up to the axioms
        and, below braided, up to where its scalars lie.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> f0, f1 = Box('f0', x, y), Box('f1', y, x)
        >>> diagram = f0 @ f1.dagger() >> f0.dagger() @ f1
        >>> assert diagram.to_hypergraph().to_diagram() == diagram.foliation()

        >>> c = Box('c', Ty(), Ty())
        >>> assert c @ x != x @ c
        >>> assert (c @ x).to_hypergraph() == (x @ c).to_hypergraph()
        """
        graph = getattr(self, "hypergraph_factory", hypergraph.Hypergraph)[
            type(self).ar].from_diagram(self)
        if all(box.dom for box in self.boxes):
            return graph
        offsets = []
        for box, offset, image in zip(self.boxes, self.offsets, graph.boxes):
            if image != box:
                break
            offsets.append(None if box.dom else offset)
        offsets += (len(graph.boxes) - len(offsets)) * [None]
        return type(graph)(
            graph.dom, graph.cod, graph.boxes, graph.wires,
            graph.spider_types, tuple(offsets), graph.phases)

    def foliation(self):
        """
        Merges layers together to reduce the length of a diagram.

        Example
        -------
        >>> from discopy.monoidal import *
        >>> x, y = Ty('x'), Ty('y')
        >>> f0, f1 = Box('f0', x, y), Box('f1', y, x)
        >>> diagram = f0 @ f1.dagger() >> f0.dagger() @ f1
        >>> print(diagram)
        f0 @ x >> y @ f1[::-1] >> f0[::-1] @ y >> x @ f1
        >>> diagram.foliation().draw(
        ...     doctest='docs/_static/monoidal/foliation-example.svg')

        .. image:: /_static/monoidal/foliation-example.svg
            :align: center

        Note
        ----
        If one defines a foliation as a sequence of unmergeable layers,
        there may exist many distinct foliations for the same diagram. When the
        diagram lives in a symmetric category with self-dual objects and its
        hypergraph is monogamous, causal and boundary-connected, the foliation
        is read off the hypergraph in one pass over the boundary (see
        :meth:`discopy.hypergraph.Hypergraph.to_diagram`). The self-duality is
        needed because that pass reconstructs the diagram with swaps, which
        would not be faithful in a rigid category such as pregroup, where the
        left and right adjoints of an object differ. Otherwise this scans top
        to bottom and merges layers eagerly.
        """
        if not isinstance(self, abc.SymmetricCategory):
            return self.merge_layers()
        graph = self.to_hypergraph()
        if graph.is_monogamous and graph.is_causal\
                and graph.is_boundary_connected and all(
                    not isinstance(obj, abc.Pregroup) or obj.l == obj.r
                    for obj in graph.spider_types):
            return graph.to_diagram().merge_layers()
        return self.merge_layers()

    def merge_layers(self):
        diagram, i = self, 0
        while i < len(diagram.inside) - 1:
            try:
                merged = diagram.inside[i].merge(diagram.inside[i + 1])
            except AxiomError:
                i += 1
                continue
            inside = diagram.inside[:i] + (merged, ) + diagram.inside[i + 2:]
            diagram = diagram.ar(inside, diagram.dom, diagram.cod)
            i = max(i - 1, 0)
        return diagram

    def dependencies(self) -> "tuple[tuple[int, ...], ...]":
        """
        The boxes each box of a staircase diagram must stay below: no
        interchange moves it above them. These are

        * the boxes whose outputs it takes as inputs,
        * for a box with no inputs, the boxes whose outputs enclose the gap it
          sits in, with the boxes above them: a box with no outputs joins two
          gaps, enclosed by the boxes that enclose both,
        * the boxes with no outputs that joined a gap between its inputs.

        Example
        -------
        >>> x = Ty('x')
        >>> f, g, h = Box('f', x, x), Box('g', x, x), Box('h', x @ x, x)
        >>> cup, s = Box('cup', Ty(), x @ x), Box('s', Ty(), x)
        >>> e = Box('e', x, Ty())
        >>> (f @ g >> h).to_staircases().dependencies()
        ((), (), (0, 1))
        >>> (cup >> f @ x >> x @ s @ x).dependencies()
        ((), (0,), (0,))
        >>> (x @ e @ x >> h).dependencies()
        ((), (0,))
        """
        wires: "list[int | None]" = len(self.dom) * [None]
        gaps: "list[tuple[frozenset[int], frozenset[int]]]" = (
            len(self.dom) + 1) * [(frozenset(), frozenset())]
        above: "list[frozenset[int]]" = []
        result: "list[tuple[int, ...]]" = []
        for box, offset in zip(self.boxes, self.offsets):
            n, m, j = len(box.dom), len(box.cod), len(result)
            inputs = {i for i in wires[offset:offset + n] if i is not None}\
                .union(*(gap[1] for gap in gaps[offset + 1:offset + n]))\
                if n else set(gaps[offset][0])
            above.append(frozenset(inputs).union(*(above[i] for i in inputs)))
            left, right = gaps[offset], gaps[offset + n]
            if not n:
                left, right = (left[0], frozenset()), (right[0], frozenset())
            wires[offset:offset + n] = m * [j]
            gaps[offset:offset + n + 1] = [left] + (m - 1) * [
                (above[j] | {j}, frozenset())] + [right] if m else [
                    (left[0] & right[0], left[1] | right[1] | {j})]
            result.append(tuple(sorted(inputs)))
        return tuple(result)

    @staticmethod
    def layerings(
            dependencies: "tuple[tuple[int, ...], ...]"
    ) -> "Iterator[tuple[tuple[int, ...], ...]]":
        """
        The ways to split boxes into layers that respect their dependencies,
        each layer depending on the one above it, i.e. no two adjacent layers
        could be merged, the largest layers first.

        Parameters:
            dependencies : As given by :meth:`dependencies`.

        Example
        -------
        >>> list(Diagram.layerings(((), (), (0, ))))
        [((0, 1), (2,)), ((0,), (1, 2))]
        """
        def layers(placed, previous):
            free = [i for i, before in enumerate(dependencies)
                    if i not in placed and set(before) <= placed]
            if previous and not any(
                    set(dependencies[i]) & set(previous) for i in free):
                return iter(())
            return (layer for size in range(len(free), 0, -1)
                    for layer in itertools.combinations(free, size)
                    if not previous or any(
                        set(dependencies[i]) & set(previous) for i in layer))
        if not dependencies:
            yield ()
            return
        stack: "list[tuple[tuple, frozenset[int], Iterator]]" = [
            ((), frozenset(), layers(frozenset(), ()))]
        while stack:
            layering, placed, candidates = stack[-1]
            layer = next(candidates, None)
            if layer is None:
                stack.pop()
                continue
            layering, placed = layering + (layer, ), placed | set(layer)
            if len(placed) == len(dependencies):
                yield layering
                continue
            stack.append((layering, placed, layers(placed, layer)))

    def foliate(
            self, layering: "tuple[tuple[int, ...], ...]", left=False
    ) -> "Diagram":
        """
        The foliation of a staircase diagram with the given layers: each box
        is interchanged up to its place, then each layer is merged.

        Parameters:
            layering : The indices of the boxes in each layer, as given by
                :meth:`layerings`.
            left : Passed to :meth:`interchange`, which side a box with no
                inputs takes of a box with no outputs it moves above.

        Raises:
            AxiomError : When a layer does not merge, e.g. a box with no
                inputs moved between the outputs of a box in its layer.

        Example
        -------
        >>> x = Ty('x')
        >>> f, g, h = Box('f', x, x), Box('g', x, x), Box('h', x @ x, x)
        >>> print((f @ g >> h).to_staircases().foliate(((1, ), (0, ), (2, ))))
        x @ g >> f @ x >> h
        """
        diagram, order = self, list(range(len(self)))
        for position, i in enumerate(i for layer in layering for i in layer):
            diagram = diagram.interchange(
                order.index(i), position, left=left)
            order.insert(position, order.pop(order.index(i)))
        inside, start = [], 0
        for layer in layering:
            merged = diagram.inside[start]
            for other in diagram.inside[start + 1:start + len(layer)]:
                merged = merged.merge(other)
            inside.append(merged)
            start += len(layer)
        return self.ar(tuple(inside), self.dom, self.cod)

    def foliations(self, limit: int | None = None) -> "list[Diagram]":
        """
        Foliations of a diagram, i.e. diagrams equal to it up to interchange
        whose adjacent layers cannot be merged, the largest layers first, from
        the top: :meth:`foliation` merges layers top-down instead, so it may
        come later, e.g. ``(f >> g) @ h``. Each layering of
        :meth:`dependencies` is foliated, then merged by the level's own
        :meth:`Layer.merge`, e.g. two permutations fuse in
        :mod:`discopy.symmetric`.

        Parameters:
            limit : The maximum number of foliations to return.

        Example
        -------
        >>> x = Ty('x')
        >>> f, g, h = Box('f', x, x), Box('g', x, x), Box('h', x @ x, x)
        >>> for foliation in (f @ g @ f >> h @ x).foliations():
        ...     print(foliation)
        f @ g @ f >> h @ x
        f @ g @ x >> h @ f
        >>> assert len((f @ f @ f @ f).foliations()) == 1
        >>> assert len((f @ f @ f >> f @ f @ f).foliations(limit=2)) == 2

        A box with no inputs can go either side of a box with no outputs:

        >>> s, e = Box('s', Ty(), x), Box('e', x @ x, Ty())
        >>> for foliation in (e >> s >> f).foliations(): print(foliation)
        s @ e >> f
        s @ x @ x >> f @ e
        x @ x @ s >> e @ f

        Note
        ----
        Each layering is built with right interchangers, then with left ones:
        a foliation that needs both, e.g. a box with no inputs moved left of
        one box and right of another, is missed, and so is the order of a
        box with no inputs and one with no outputs inside a layer that
        :meth:`Layer.merge` does not choose, e.g. ``e @ s >> f`` beside
        ``s @ e >> f``: see `#576
        <https://github.com/rel-int/discopy-next/issues/576>`_.
        """
        staircase, result = self.to_staircases(), []
        candidates = (
            (layering, left) for layering in self.layerings(
                staircase.dependencies()) for left in (False, True))
        for layering, left in candidates:
            if limit is not None and len(result) >= limit:
                break
            try:
                foliation = staircase.foliate(layering, left).merge_layers()
            except AxiomError:
                continue
            if foliation not in result:
                result.append(foliation)
        return result

    def depth(self):
        """
        Computes (an upper bound to) the depth of a diagram by foliating it.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> f, g = Box('f', x, y), Box('g', y, x)
        >>> assert Id(x @ y).depth() == 0
        >>> assert f.depth() == 1
        >>> assert (f @ g).depth() == 1
        >>> assert (f >> g).depth() == 2

        Note
        ----
        The depth of a diagram is the minimum length over all its foliations,
        this method just returns the length of :meth:`Diagram.foliation`.
        """
        return len(self.foliation())

    def interchange(self, i: int, j: int, left=False) -> "Diagram":
        """
        Interchange a box from layer ``i`` to layer ``j``, a planar rewrite:
        see :meth:`normalize` for its scope.

        Parameters:
            i : Index of the box to interchange.
            j : Index of the new position for the box.
            left : Whether to apply left interchangers.

        Raises:
            IndexError : If ``i`` or ``j`` is not the index of a layer.
            NotImplementedError : If some layer has more than one box.

        Note
        ----
        By default, we apply right interchangers::

            top >> left @ box1.dom @ mid @ box0     @ right\\
                >> left @ box1     @ mid @ box0.cod @ right >> bottom

        gets rewritten to::

            top >> left @ box1     @ mid @ box0.dom @ right\\
                >> left @ box1.cod @ mid @ box0     @ right >> bottom
        """
        if not 0 <= i < len(self) or not 0 <= j < len(self):
            raise IndexError
        if any(len(layer.boxes) != 1 for layer in self.inside):
            raise NotImplementedError
        if i == j:
            return self
        if j < i - 1:
            result = self
            for k in range(i - j):
                result = result.interchange(i - k, i - k - 1, left=left)
            return result
        if j > i + 1:
            result = self
            for k in range(j - i):
                result = result.interchange(i + k, i + k + 1, left=left)
            return result
        if j < i:
            i, j = j, i
        box0, box1 = self.boxes[i], self.boxes[j]
        off0, off1 = self.offsets[i], self.offsets[j]
        box0_left_of_box1 = off1 >= off0 + len(box0.cod)
        box0_right_of_box1 = off0 >= off1 + len(box1.dom)
        # By default, we check if box0 is to the right first, then to the left.
        if box0_left_of_box1 and (left or not box0_right_of_box1):
            off1 += len(box0.dom) - len(box0.cod)
        elif box0_right_of_box1:
            off0 += len(box1.cod) - len(box1.dom)
        else:
            raise AxiomError(messages.INTERCHANGER_ERROR.format(box0, box1))
        interchanged = self.ar.decode(
            self.inside[i].dom, [(box1, off1), (box0, off0)])
        return self[:i] >> interchanged >> self[i + 2:]

    def substitute(self, i: int, other: "Diagram") -> "Diagram":
        """
        Implements operadic composition of nested diagrams,
        replacing box :code:`i` with diagram :code:`other`.
        See Patterson et al :cite:t:`Patterson21`.

        Parameters:
            i : Index of the box to substitute.
            other : The diagram to substitute with.

        A diagram with more than one box in a layer is split into staircases
        first, see :meth:`to_staircases`, so that box ``i`` is layer ``i``.

        Example
        -------
        >>> x = Ty('x')
        >>> f, g = Box('f', x, x), Box('g', x, x)
        >>> assert (f @ f).foliation().substitute(1, g >> g)\\
        ...     == f @ x >> x @ g >> x @ g
        """
        if any(len(layer.boxes) > 1 for layer in self.inside):
            return self.to_staircases().substitute(i, other)
        box, offset = self.boxes[i], self.offsets[i]
        dom = self.inside[i].dom
        outside = Match(
            self[:i], self[i + 1:], dom[:offset], dom[offset + len(box.dom):])
        return outside.substitute(other)

    def normalize(self, left=False) -> "Iterator[Diagram]":
        """
        Implements normalisation of boundary-connected diagrams,
        see Delpeuch and Vicary :cite:t:`DelpeuchVicary22`.

        Parameters:
            left : Passed to :meth:`Diagram.interchange`.

        Example
        -------
        >>> from discopy.monoidal import *
        >>> s0, s1 = Box('s0', Ty(), Ty()), Box('s1', Ty(), Ty())
        >>> gen = (s0 @ s1).normalize()
        >>> for _ in range(3): print(next(gen))
        s1 >> s0
        s0 >> s1
        s1 >> s0

        A diagram with more than one box in a layer is split into staircases
        first, see :meth:`to_staircases`, and that is the first step.

        >>> x = Ty('x')
        >>> f, g = Box('f', x, x), Box('g', x, x)
        >>> for step in (f @ g).foliation().normalize(left=True): print(step)
        f @ x >> x @ g
        x @ g >> f @ x

        Note
        ----
        This is a planar rewrite, as are :meth:`interchange` and
        :meth:`discopy.rigid.Diagram.snake_removal`, which yanks snakes
        before it interchanges boxes: their scope is the planar levels, from
        :mod:`discopy.monoidal` to :mod:`discopy.rigid`. Above them, a braid,
        a swap or a trace is just one more box, so they do not compute a
        normal form for the axioms of the level, e.g. in
        :mod:`discopy.symmetric` the diagrams
        ``Swap(x, y) >> y @ f >> Swap(y, x)`` and ``f @ y`` have the same
        hypergraph and neither is rewritten into the other. From
        :mod:`discopy.symmetric` up, the ``Equation`` of the level compares
        diagrams up to their hypergraph instead.
        """
        diagram = self
        if any(len(layer.boxes) > 1 for layer in self.inside):
            diagram = self.to_staircases()
            yield diagram
        while True:
            no_more_moves = True
            boxes, offsets = diagram.boxes, diagram.offsets
            for i in range(len(diagram) - 1):
                box0, box1 = boxes[i], boxes[i + 1]
                off0, off1 = offsets[i], offsets[i + 1]
                if left and off1 >= off0 + len(box0.cod)\
                        or not left and off0 >= off1 + len(box1.dom):
                    diagram = diagram.interchange(i, i + 1, left=left)
                    boxes, offsets = diagram.boxes, diagram.offsets
                    yield diagram
                    no_more_moves = False
            if no_more_moves:
                break

    def normal_form(
            self, normalize: Callable | None = None, **params) -> "Diagram":
        """
        Returns the normal form of a diagram, up to the planar rewrites of
        :meth:`normalize`, whose note says where they apply.

        Parameters:
            normalize : The rewriting to normalise with, the diagram's own
                :meth:`normalize` by default, which from :mod:`rigid` up
                yanks snakes before it interchanges boxes:
                :meth:`Layer.merge` passes :meth:`Diagram.normalize`
                to interchange alone.
            params : Passed to ``normalize``.

        Raises
        ------
        NotImplementedError
            Whenever ``normalize`` yields the same rewrite steps twice, e.g.
            the diagram is not boundary-connected.
        """
        normalize = type(self).normalize if normalize is None else normalize
        cache = set()
        for diagram in itertools.chain([self], normalize(self, **params)):
            if str(diagram) in cache:
                exception = NotImplementedError(
                    messages.NOT_CONNECTED.format(self))
                exception.last_step = diagram
                raise exception
            cache.add(str(diagram))
        return diagram

    @classmethod
    def from_tree(cls, tree):
        if "inside" not in tree:
            warn("Outdated dumps", DeprecationWarning)
            boxes, offsets = map(from_tree, tree['boxes']), tree['offsets']
            return cls.decode(from_tree(tree['dom']), zip(boxes, offsets))
        return super().from_tree(tree)


class Box(cat.Box, Diagram):
    """
    A box is a diagram with a :code:`name` and the layer of just itself inside.

    Parameters:
        name : The name of the box.
        dom : The domain of the box, i.e. the input.
        cod : The codomain of the box, i.e. the output.
        data (any) : Extra data in the box, default is :code:`None`.
        is_dagger (bool, optional) : Whether the box is dagger.

    Other parameters
    ----------------

    draw_as_spider : bool, optional
        Whether to draw the box as a spider.
    draw_as_wires : bool, optional
        Whether to draw the box as wires, e.g. :class:`discopy.symmetric.Swap`.
    draw_as_braid : bool, optional
        Whether to draw the box as a a braid, e.g. :class:`braided.Braid`.
    drawing_name : str, optional
        The name to use when drawing the box.
    tikzstyle_name : str, optional
        The name a spider's TikZ style starts with, followed by its colour and
        shape, e.g. :code:`"Z_green_circle"` for a green circle named "Z".
    color : str, optional
        The color to use when drawing the box, one of
        :code:`"transparent", "red", "green", "blue", "yellow", "black"` or any
        other matplotlib colour, e.g. :data:`discopy.config.TRANSPARENT`
        for a spider drawn unfilled.
        Default is :code:`"black" if draw_as_spider else "transparent"`.
    shape : str, optional
        The shape to use when drawing a spider,
        one of :code:`"circle", "rectangle"`.

    Examples
    --------
    >>> f = Box('f', Ty('x', 'y'), Ty('z'))
    >>> assert Id(Ty('x', 'y')) >> f == f == f >> Id(Ty('z'))
    >>> assert Id(Ty()) @ f == f == f @ Id(Ty())
    >>> assert f == f[::-1][::-1]

    Coloured wires separate matplotlib regions.

    >>> red, green, blue = map(Colour, ("red", "green", "blue"))
    >>> x = Ty(Wire("x", red, green))
    >>> y = Ty(Wire("y", green, blue))
    >>> z = Ty(Wire("z", red, blue))
    >>> coloured = Box("coloured", x @ y, z)
    >>> coloured.draw(doctest='docs/_static/monoidal/coloured-box.svg')

    .. image:: /_static/monoidal/coloured-box.svg
        :align: center
    """

    def __init__(self, name: str, dom: Ty, cod: Ty, **params):
        dom = dom if isinstance(dom, self.ob) else self.ob(dom)
        cod = cod if isinstance(cod, self.ob) else self.ob(cod)
        if (dom.dom, dom.cod) != (cod.dom, cod.cod):
            raise AxiomError(messages.NOT_GLOBULAR.format(
                dom.dom, dom.cod, cod.dom, cod.cod))
        for attr in BOX_DRAWING_ATTRIBUTES:
            if attr in params:
                setattr(self, attr, params.pop(attr))
        cat.Box.__init__(self, name, dom, cod, **params)
        inside = () if self.is_identity\
            else (self.layer_factory(self, normalise=False), )
        Diagram.__init__(self, inside, dom, cod)

    def rebuild(self, **changes):
        """
        Rebuild a box, keeping the drawing attributes it was given that its
        constructor does not set, e.g. so that a dagger keeps its shape.

        >>> x = Ty('x')
        >>> merge = Box('t', x @ x, x, shape="triangle", color="red")
        >>> assert merge.dagger().shape == "triangle"
        >>> assert merge.dagger().dagger().color == "red"
        """
        box = super().rebuild(**changes)
        for attr in BOX_DRAWING_ATTRIBUTES:
            if attr in vars(self) and attr not in vars(box):
                setattr(box, attr, getattr(self, attr))
        return box

    is_identity = False

    @classmethod
    def atomic_strategy(cls, types=None):
        """
        Generate the atomic types a structural box is built on, those of
        one wire, drawn from ``types`` if given: a structural box of a
        composite type is a diagram of those of its atoms.

        Parameters:
            types : The strategy to draw the types from, e.g.
                :meth:`Ty.strategy`, the one wire types otherwise.

        Example
        -------
        >>> from hypothesis import find
        >>> assert find(Box.atomic_strategy(), lambda _: True) == Ty('a')
        """
        # pylint: disable-next=no-member  # the ob of a Box is a Ty
        return cls.ob.generator_strategy() if types is None\
            else types.filter(lambda typ: len(typ) == 1)

    @property
    def size(self):
        return 1

    def to_drawing(self):
        result = Drawing.from_box(self)
        if type(self).draw_left_to_right and "height" not in vars(self)\
                and not result.box.no_label:
            # The height of a box drawn left to right is its width on the page.
            box = copy(self)
            box.height = max(1, text_width(
                getattr(self, "drawing_name", self.name)) + 1)
            if "min_width" not in vars(self):
                box.min_width = 0
            return Drawing.from_box(box)
        return result


class Bubble(cat.Bubble, Box):
    """
    A bubble is a box with diagrams :code:`args` inside and an optional pair of
    types :code:`dom` and :code:`cod`.

    Parameters:
        args (Diagram) : The diagrams inside the bubble.
        drawing_name (str) : The label to use when drawing, empty by default.
        draw_as_square (bool) : Whether to draw the bubble as a square.
        draw_as_frame (bool) : Whether to draw the bubble as a frame.
        draw_vertically (bool) : Whether to draw the frame slots vertically.
        kwargs : Passed to :class:`cat.Bubble`.

    Raises:
        ValueError : When dom (resp. cod) is None and the args do not all
            have the same dom (resp. cod).

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, g, h = Box('f', x, y ** 3), Box('g', y, y @ y), Box('h', x, y)
    >>> d = (f.bubble(dom=x ** 3, cod=y, draw_as_square=True) >> g).bubble()
    >>> d.draw(doctest='docs/_static/monoidal/bubble-example.svg')

    .. image:: /_static/monoidal/bubble-example.svg
        :align: center

    >>> b = Bubble(f, g, h >> h[::-1], dom=x, cod=y @ y)
    >>> b.draw(doctest='docs/_static/monoidal/bubble-multiple-args.svg')

    .. image:: /_static/monoidal/bubble-multiple-args.svg
        :align: center

    >>> b = Bubble(f, g, h, dom=x, cod=y @ y, draw_vertically=True)
    >>> b.draw(doctest='docs/_static/monoidal/frame-vertical-args.svg')

    .. image:: /_static/monoidal/frame-vertical-args.svg
        :align: center

    Coloured frames distinguish their outside, frame and slot regions.

    >>> red, blue = map(Colour, ("red", "blue"))
    >>> x = Ty(Wire("x", red, blue))
    >>> f = Box("f", x, x)
    >>> frame = f.bubble(
    ...     dom=Ty(Wire("boundary", blue, red)),
    ...     cod=Ty(Wire("boundary", blue, red)),
    ...     draw_as_frame=True)
    >>> frame.draw(doctest='docs/_static/monoidal/coloured-frame.svg')

    .. image:: /_static/monoidal/coloured-frame.svg
        :align: center

    The diagrams of a category with ``draw_left_to_right`` are drawn left to
    right, and a bubble whose args are drawn in another direction than its
    own category is drawn as one box with its args across it, on a
    background of its ``frame_colour``. This is how to draw "Very big car" in
    higher-order DisCoCat :cite:p:`ToumiDeFelice25`: each word is a lambda
    term, drawn top to bottom, whose body is a diagram, drawn left to right:

    >>> @factory
    ... class Body(Diagram):
    ...     draw_left_to_right = True
    >>> class Primitive(Box, Body):
    ...     pass
    >>> n, n_n = Ty('n'), Ty('n ← n')
    >>> x, B = Primitive('x', Ty(), n), Primitive('B', n, n)
    >>> C = Primitive('C', Ty(), n)
    >>> style = dict(color="green", frame_colour="red")
    >>> very = Box('λfx · ffx', Ty(), Ty('(n ← n) ← (n ← n)'),
    ...            color="green", height=2)
    >>> big = Bubble(x >> B, dom=Ty(), cod=n_n, drawing_name="λx ·", **style)
    >>> car = Bubble(C, dom=Ty(), cod=n, **style)
    >>> very_big = Bubble(x >> B >> B, dom=very.cod @ n_n, cod=n_n,
    ...                   drawing_name="λx ·", **style)
    >>> very_big_car = Bubble(C >> B >> B, dom=n_n @ n, cod=n, **style)
    >>> diagram = (very @ big @ car).foliation()\\
    ...     >> very_big @ n >> very_big_car
    >>> diagram.draw(doctest='docs/_static/monoidal/very-big-car.svg')

    .. image:: /_static/monoidal/very-big-car.svg
        :align: center

    """
    serialised_attrs = cat.Bubble.serialised_attrs + (
        "drawing_name", "draw_as_frame", "draw_as_square", "draw_vertically")

    @property
    def frame_colour(self):
        """ The colour of the frame, the default of
        :data:`discopy.config.BOX_DRAWING_ATTRIBUTES` unless given. """
        return vars(self).get(
            "frame_colour", BOX_DRAWING_ATTRIBUTES['frame_colour'](self))

    @frame_colour.setter
    def frame_colour(self, colour):
        vars(self)["frame_colour"] = colour

    def __init__(
            self, *args: Diagram,
            drawing_name: str | None = None,
            draw_as_frame: bool | None = None,
            draw_as_square: bool | None = None,
            draw_vertically=False, **kwargs):
        params = {attr: kwargs.pop(attr)
                  for attr in BOX_DRAWING_ATTRIBUTES if attr in kwargs}
        cat.Bubble.__init__(self, *args, **kwargs)
        Box.__init__(self, self.name, self.dom, self.cod,
                     data=self.data, is_dagger=self.is_dagger, **params)
        self.drawing_name = "" if drawing_name is None else drawing_name
        self.draw_vertically = draw_vertically
        can_draw_as_square = len(args) == 1
        can_draw_as_bubble = (can_draw_as_square
                              and len(self.dom) == len(self.arg.dom)
                              and len(self.cod) == len(self.arg.cod))
        if len(args) == 1:
            can_draw_as_bubble = (len(self.dom), len(self.cod)) == (
                len(self.arg.dom), len(self.arg.cod))
            self.draw_as_square = draw_as_square or not can_draw_as_bubble
            self.draw_as_frame = draw_as_frame or (
                not can_draw_as_bubble and not self.draw_as_square)
        else:
            self.draw_as_frame = True
            self.draw_as_square = False

    @property
    def size(self):
        """ The number of boxes in a bubble, counting its arguments. """
        return 1 + sum(arg.size for arg in self.args)

    def to_drawing(self):
        args = [arg.to_drawing() for arg in self.args]
        directions = {type(arg).draw_left_to_right for arg in self.args}
        if len(directions) > 1:
            raise ValueError(messages.MIXED_DIRECTIONS.format(self))
        if directions != {self.draw_left_to_right}:
            framed = [arg.framed() for arg in args]
            height = max(arg.height for arg in framed)
            nested = tuple(
                arg.stretch(height - arg.height) for arg in framed)
            width = sum(arg.width for arg in nested)
            label = text_width(self.drawing_name) + .25\
                if self.drawing_name else 0
            return Box(
                self.drawing_name, self.dom, self.cod, nested=nested,
                color=getattr(self, "color", "white"),
                frame_colour=self.frame_colour,
                min_width=max(label + height + .5,
                              getattr(self, "min_width", 0)),
                height=max(width + 1,
                           getattr(self, "height", 0))).to_drawing()
        method = "frame" if self.draw_as_frame else "bubble"
        kwargs = dict(
            dom=self.dom.to_drawing(),
            cod=self.cod.to_drawing(),
            name=self.drawing_name)
        if self.draw_as_frame:
            kwargs['draw_vertically'] = self.draw_vertically
            kwargs['frame_colour'] = self.frame_colour
        else:
            kwargs['draw_as_square'] = self.draw_as_square
        return getattr(Drawing, method)(*args, **kwargs)


class Functor(cat.Functor):
    """
    A monoidal functor is a functor that preserves the tensor product.

    Parameters:
        ob_map (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
        colour_map (Mapping[Colour, Colour]) :
            Map from region :class:`Colour` to :code:`cod` colour.

    Important
    ---------
    The keys of the objects mapping must be atomic types, i.e. of length 1.

    Note
    ----
    Colour maps are expected to send colours to colours, so the image of an
    empty coloured identity ``Ty.id(c)`` keeps its (mapped) colour whenever
    ``cod.ob`` has an ``id`` method, e.g. ``F(Ty.id(c)) == Ty.id(F(c))``.

    Note
    ----
    A :class:`Layer` maps to the tensor of its images. When ``cod`` is a
    :class:`discopy.abc.MonoidalCategory`, the first two are tensored with
    ``@`` as in a fold, e.g. so that a type next to a
    :class:`discopy.symmetric.Permutation` gives a wider permutation, and the
    others are whiskered and tensored in one ``n``-ary call. Any other
    ``cod`` folds ``@``, e.g. :class:`discopy.kleisli.multiplicative.Channel`.

    Example
    -------
    >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    >>> f0, f1 = Box('f0', x, y, data=0.1), Box('f1', z, w, data=1.1)
    >>> F = Functor({x: z, y: w, z: x, w: y}, {f0: f1, f1: f0})
    >>> assert F(f0) == f1 and F(f1) == f0
    >>> assert F(F(f0)) == f0
    >>> assert F(f0 @ f1) == f1 @ f0
    >>> assert F(f0 >> f0[::-1]) == f1 >> f1[::-1]
    >>> source, target = f0 >> f0[::-1], F(f0 >> f0[::-1])

    >>> Equation(source, target, symbol='$\\\\mapsto$').draw(
    ...     doctest='docs/_static/monoidal/functor-example.svg')

    .. image:: /_static/monoidal/functor-example.svg
        :align: center
    """

    dom = cod = Diagram

    def __init__(
            self, ob_map=None, ar_map=None,
            dom=None, cod=None, colour_map=None):
        super().__init__(ob_map, ar_map, dom=dom, cod=cod)
        self.colour_map = MappingOrCallable(colour_map or {})

    @unbiased
    def then(self, other):
        assert_isinstance(other, Functor)
        result = super().then(other)
        if result is self or result is other:
            return result
        result.colour_map = MappingOrCallable(
            self.colour_map.then(other) if self.colour_map
            else other.colour_map)
        return result

    def __eq__(self, other):
        return super().__eq__(other) and self.colour_map == other.colour_map

    def __repr__(self):
        result = super().__repr__()
        if not self.colour_map:
            return result
        suffix = ')' if result.endswith(')') else ''
        return result[:-len(suffix) if suffix else None] + (
            f", colour_map={self.colour_map!r}{suffix}")

    def to_tree(self) -> dict:
        """
        Serialise a functor whose maps are dictionaries, its colour map
        included, see :meth:`discopy.cat.Functor.to_tree`.

        >>> from discopy.utils import dumps, loads
        >>> x, y, red, blue = Ty('x'), Ty('y'), Colour('red'), Colour('blue')
        >>> F = Functor({x: y @ y}, {}, colour_map={red: blue})
        >>> assert loads(dumps(F)) == F
        """
        tree, colour_map = super().to_tree(), self.colour_map.mapping
        if not isinstance(colour_map, Mapping):
            raise TypeError(messages.NO_TREE_CALLABLE.format(
                "A functor", "colour_map"))
        return tree | ({'colour_map': [
            [key.to_tree(), value.to_tree()]
            for key, value in colour_map.items()]} if colour_map else {})

    @classmethod
    def from_tree(cls, tree: dict) -> Functor:
        functor = super().from_tree(tree)
        functor.colour_map = MappingOrCallable({
            from_tree(key): from_tree(value)
            for key, value in tree.get('colour_map', [])})
        return functor

    def __call__(self, other):
        if other is None or isinstance(other, Colour):
            if not self.colour_map or (
                    other is None and other not in self.colour_map):
                return other
            return self.colour_map[other]
        if isinstance(other, Dim) and isinstance(
                other, self.dom.ob.generator_factory):
            return self.cod.ob.cast(self.ob_map[self.dom.ob(other)])
        if isinstance(other, Dim):
            return self.cod.ob().tensor(*(  # ty: ignore[invalid-argument-type]
                self.ob_map[x] for x in other))
        if isinstance(other, Nat):
            image = super().__call__(other.factory(1))
            return image[:0].tensor(*other.n * [image])
        if isinstance(other, Ty):
            if not other.inside:
                if not issubclass(self.cod.ob, abc.Category):
                    return self.cod.ob()
                return self.cod.ob.id(self(other.dom))
            head, *tail = map(self, other.inside)
            return head.tensor(*tail)
        if isinstance(other, self.dom.ob.generator_factory):
            if isinstance(other, Wire) and other.is_dagger:
                # Map a daggered coloured generator functorially: its image is
                # the dagger of the image of the underlying generator.
                return self(other.dagger()).dagger()
            result = super().__call__(self.dom.ob(other))
            if isinstance(other, Wire) and isinstance(result, Ty):
                expected = self(other.dom), self(other.cod)
                if (result.dom, result.cod) != expected:
                    raise AxiomError(messages.NOT_GLOBULAR.format(
                        result.dom, result.cod, *expected))
            return result
        if isinstance(other, Layer) and issubclass(
                self.cod.ar, abc.MonoidalCategory):
            images = list(map(self, other))
            head = images[0] if len(images) == 1 else images[0] @ images[1]
            return head.tensor(*map(self.cod.ar.whisker, images[2:]))
        if isinstance(other, Layer):
            head, *tail = other
            result = self(head)
            for box_or_typ in tail:
                result = result @ self(box_or_typ)
            return result
        if isinstance(other, Bubble) and self.cod is Drawing:
            return other.to_drawing()
        if isinstance(other, Bubble) and other.method == "bubble"\
                and issubclass(self.cod, Diagram):
            factory = self.cod.bubble_factory
            kept = {
                attr: getattr(other, attr) for attr in other.serialised_attrs
                if attr in factory.serialised_attrs
                and attr not in ("*args", "dom", "cod", "method")}
            return factory(*map(self, other.args), dom=self(other.dom),
                           cod=self(other.cod), **kept)
        return super().__call__(other)


@dataclass
class Match:
    """ A match is a diagram with a hole, given by:

    Parameters:
        above : The diagram above the hole.
        below : The diagram below the hole.
        left : The wires left of the hole.
        right : The wires right of the hole.
    """

    above: Diagram
    below: Diagram
    left: Ty
    right: Ty

    def substitute(self, target: Diagram) -> Diagram:
        """
        Substitute a diagram inside the hole.

        Parameters:
            target : The diagram to substitute inside the hole.
        """
        return self.above >> self.left @ target @ self.right >> self.below


CMap = cmap.CMap[Diagram]


class Equation(cat.Equation, RichDisplay):
    """
    An :class:`.cat.Equation` of diagrams, i.e. with a :meth:`draw` method.

    Parameters:
        terms : The terms of the equation.
        symbol : The symbol between each pair of terms, ``"="`` by default.
        symbols : The symbols between each pair of consecutive terms, one
            fewer than the terms, overriding ``symbol``.
        space : The space between the terms when drawing the equation.
        up_to : The function up to which ``bool(equation)`` compares its terms,
            overriding the subclass' :attr:`up_to` if given.

    Example
    -------
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> print(Equation(f, g))
    Equation(f, g)
    """
    def __init__(self, *terms: Diagram, symbol="=", symbols=None, space=1,
                 up_to=None):
        super().__init__(*terms, symbol=symbol, symbols=symbols, up_to=up_to)
        self.space = space

    def __eq__(self, other):
        """
        Two equations of diagrams are equal when they are equal as
        :class:`.cat.Equation` and have the same space between their terms.

        >>> x = Ty('x')
        >>> f = Box('f', x, x)
        >>> assert Equation(f, f) == Equation(f, f) != Equation(f, f, space=2)
        """
        return super().__eq__(other) and self.space == other.space

    __hash__ = cat.Equation.__hash__

    def __repr__(self):
        """
        The representation of a :class:`.cat.Equation`, then the space
        unless it is ``1``, after the terms if there are any.

        >>> Equation(Ty(), Ty(), space=2)
        monoidal.Equation(monoidal.Ty(), monoidal.Ty(), space=2)
        >>> Equation(space=2)
        monoidal.Equation(space=2)
        """
        if self.space == 1:
            return super().__repr__()
        comma = ", " if self.terms else ""
        return f"{super().__repr__()[:-1]}{comma}space={self.space!r})"

    def to_tree(self) -> dict:
        """
        Serialise an equation of diagrams, with the space between its terms,
        see :meth:`discopy.cat.Equation.to_tree`.

        >>> from discopy.utils import dumps, loads
        >>> x = Ty('x')
        >>> equation = Equation(Box('f', x, x), Id(x), symbol="->", space=2)
        >>> loaded = loads(dumps(equation))
        >>> assert (loaded.terms, loaded.symbols, loaded.space)\\
        ...     == ((Box('f', x, x), Id(x)), ("->", ), 2)
        >>> dumps(Equation(Id(x), Id(x), space="2"))
        Traceback (most recent call last):
        ...
        TypeError: Expected int | float, got str instead.
        """
        assert_isinstance(self.space, (int, float))
        return super().to_tree() | ({} if self.space == 1 else {
            'space': self.space})

    @classmethod
    def from_tree(cls, tree: dict) -> Equation:
        equation = super().from_tree(tree)
        equation.space = tree.get('space', 1)
        assert_isinstance(equation.space, (int, float))
        return equation

    def to_drawing(self):
        result = self.terms[0].to_drawing()
        for symbol, term in zip(self.symbols, self.terms[1:]):
            result = result.add(term.to_drawing(), symbol, self.space)
        return result

    def draw(self, path=None, **params):
        """
        Drawing an equation.

        Parameters:
            path : Where to save the drawing.
            params : Passed to :meth:`Diagram.draw`.
        """
        return self.to_drawing().draw(path=path, **params)


Diagram.draw = drawing.draw
Diagram.to_gif = drawing.to_gif

Diagram.box_factory = Box
Diagram.sum_factory = formal_sum.Sum[Diagram]
Diagram.bubble_factory = Bubble
Diagram.functor_factory = Functor
Hypergraph = hypergraph.Hypergraph[Diagram]
Drawing.ob = Ty
Id = Diagram.id


class TermBase(cat.TermBase):
    """
    A term of a monoidal category, i.e. of the planar linear kappa calculus:
    a :class:`discopy.cat.TermBase` with a list of variables as context,
    each one wire, and a type as stack. A term in a context
    :math:`\\Gamma` with a stack :math:`A` of inputs still to be applied
    and one output :math:`b`

    .. math::

        \\Gamma_1, \\Gamma_2 \\vdash e : A \\to b
        \\qquad \\text{denotes} \\qquad
        \\Gamma_1 \\otimes A \\otimes \\Gamma_2 \\to b

    - an :class:`Application` :math:`e(c)` feeds a term :math:`c` with no
      stack to the first inputs of the stack of :math:`e`, putting the
      context of :math:`c` where they were
    - a :class:`Kappa` :math:`\\kappa x. e` moves a block of variables from
      the context to the stack, where they stand

    Terms are linear, each variable occurs once, and planar, the context is
    in the order of the occurrences and nothing ever crosses, so that they
    evaluate with composition and whiskering alone.

    Attributes:
        freevars (list[Variable]): The context, in the order of the domain.
        stack (Ty): The inputs still to be applied.
        offset (int | None): The number of free variables left of the stack,
            ``None`` when the term has no place for one: a term with no
            stack that is not a :class:`Kappa`.
        cod (Ty): The codomain, one wire unless the term is the expression
            of a :func:`let` binding several variables, or none.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, a, b = Constant(Box('f', x @ y, x)), Variable('a', x), \\
    ...     Variable('b', y)
    >>> term = f(a)
    >>> assert term.freevars == [a] and term.offset == 1
    >>> assert term.stack == y and term.dom == x @ y and term.cod == x
    >>> assert term(b).dom == x @ y and term(b).offset is None
    >>> g, h = Box('g', x, y), Box('h', x @ y @ x, y)
    >>> term = (x @ x @ x)(
    ...     lambda a, b, c: Constant(h)(a, Constant(g)(b), c))
    >>> assert term.eval() == x @ g @ x >> h
    """
    stack: Ty
    cod: Ty
    sampler_factory: ClassVar[type[Sampler]]

    strategy = classmethod(cat.TermBase.strategy.__func__)

    @staticmethod
    def wires(typ: Ty) -> list[Ty]:
        """ The types of one wire a type is made of. """
        return [typ[i:i + 1] for i in range(len(typ))]

    def types(self, variables: Sequence[cat.Variable]) -> Ty:
        """ The tensor of the types of some variables. """
        return self.cod[:0].tensor(*(x.cod for x in variables))

    @property
    def dom(self) -> Ty:
        """ The domain: the stack between the two halves of the context. """
        offset = len(self.freevars) if self.offset is None else self.offset
        return self.types(self.freevars[:offset]) @ self.stack\
            @ self.types(self.freevars[offset:])

    @classmethod
    def from_diagram(cls, diagram: Diagram) -> cat.TermBase:
        """
        The term of a diagram with one output, see
        :meth:`Diagram.to_term`.

        Parameters:
            diagram : The diagram to read.
        """
        if len(diagram.cod) != 1:
            raise ValueError(messages.NOT_ONE_OUTPUT.format(diagram.cod))
        fresh = count()

        def name(typ):
            return [cls.variable_factory(f"x{next(fresh)}", wire)
                    for wire in cls.wires(typ)]

        inputs = name(diagram.dom)
        wires, steps = list(inputs), []
        for layer in diagram.inside:
            for box, offset in layer.boxes_and_offsets:
                arguments = wires[offset:offset + len(box.dom)]
                outputs = cls.rewire(box, arguments)
                if outputs is None:
                    outputs = name(box.cod)
                    steps.append((cls.constant_factory(box)(
                        *arguments), outputs, offset))
                wires[offset:offset + len(box.dom)] = outputs
        result, = wires
        for expression, outputs, offset in reversed(steps):
            result = expression if outputs == [result]\
                else cls.bind(outputs, result, offset)(expression)
        return cls.bind(inputs, result) if inputs else result

    @classmethod
    def rewire(cls, box: Box, arguments: list) -> list | None:
        """
        The variables on the outputs of a box that a term reads as wiring
        rather than as a constant, given those on its inputs: ``None`` in a
        monoidal category, where every box is a constant, a permutation of
        them in a symmetric one.
        """
        # pylint: disable=unused-argument  # every box is a constant here
        return None

    @classmethod
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def generate(cls, cod: Ty, choices: Sequence[int], types: Sequence[Ty],
                 letters: Sequence[str | None]) -> cat.TermBase:
        """
        Build a term of codomain ``cod`` from a sequence of ``choices``, as
        :meth:`discopy.biclosed.TermBase.generate` does: its inputs are
        bound by a kappa or left free, named ``v0, v1, ...``, and the
        variable bound at each level is named by the letter at that level,
        see :class:`Sampler`.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> print(TermBase.generate(y, [1, 0, 0, 2, 2, 1, 2], [x], "ab"))
        x(lambda a0: let(c0(a0), lambda b1, a2: c1(b1, a2)))
        >>> print(TermBase.generate(y, [1, 2, 1, 2, 2, 0, 2], [x], "ab"))
        let(c0, lambda a0, b1: c1(a0, b1, v0))
        """
        sampler = cls.sampler_factory(cls, iter(choices), types, letters)
        dom = [sampler.choose(types) for _ in range(sampler.choose(range(3)))]
        if sampler.choose((False, True)):
            return sampler.state(cod, tuple(
                cls.variable_factory(f"v{i}", typ)
                for i, typ in enumerate(dom)))
        variables = tuple(
            sampler.bound(i, typ) for i, typ in enumerate(dom))
        body = sampler.state(cod, variables, len(dom))
        return cls.bind(variables, body) if variables else body

    @classmethod
    def shapes(cls, *, types=None, cod=None) -> st.SearchStrategy[tuple]:
        """
        Generate the arguments of :meth:`generate` but its letters, with
        types of one wire each.
        """
        from hypothesis import strategies as st

        types = st.sampled_from(GENERATORS).map(cls.functor.cod.ob)\
            if types is None else types
        return st.tuples(
            types if cod is None else st.just(cod), cls.choices(),
            st.lists(types, min_size=1, max_size=3))

    @axiom
    def normal_form_soundness(cls, term: Self) -> AbstractEquation:
        """
        Beta reduction is sound up to interchange, see
        :class:`InterchangeEquation`: a term, its normal form and that of
        the reading of its foliation evaluate to diagrams of one orbit. The
        reading is there for beta reduction to change the order of its
        boxes, those of a layer left to right: a generated term already
        evaluates its arguments in the order its normal form does.
        """
        diagram = term.eval()
        reading = diagram.foliation().to_term()
        return InterchangeEquation(
            diagram, term.normal_form().eval(), reading.normal_form().eval())

    normal_form_completeness = cat.TermBase.normal_form_completeness\
        .failing(messages.STATEMENTS_DO_NOT_COMMUTE)


class Variable(TermBase, cat.Variable):
    """
    A variable with a name and a type of one wire, an identity diagram.

    Parameters:
        name : The name of the variable.
        cod : The type of the variable.

    Example
    -------
    >>> x = Ty('x')
    >>> assert Variable('a', x).eval() == Diagram.id(x)
    >>> Variable('a', x @ x)
    Traceback (most recent call last):
        ...
    ValueError: Expected exactly one output, got x @ x: a term has one.
    """

    def __init__(self, name: str, cod: Ty):
        if len(cod) != 1:
            raise ValueError(messages.NOT_ONE_OUTPUT.format(cod))
        super().__init__(name, cod)
        self.stack = cod[:0]


class Constant(TermBase, cat.Constant):
    """
    A box as a term, with its domain as stack. It prints as the name of the
    box, so that a term reads as a program calling its boxes.

    Parameters:
        box : The box.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> assert Constant(f).stack == x and Constant(f).eval() == f
    >>> print(Constant(f)(Variable('a', x)))
    f(a)
    """


class Application(TermBase, cat.Application):
    """
    The application of a term to an argument with no stack, which takes the
    first inputs of the stack: the context of the argument goes where they
    were, so that nothing crosses.

    Parameters:
        func : The term that is applied.
        arg : The argument, a term with no stack.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> g, a, b = Constant(Box('g', x @ y, x)), Variable('a', x), \\
    ...     Variable('b', y)
    >>> assert g(a, b) == Application(Application(g, a), b)
    >>> assert g(a, b).freevars == [a, b]
    >>> g(b)
    Traceback (most recent call last):
        ...
    ValueError: g of stack x @ y does not apply to b of stack Ty() and \
type y.
    """

    def __init__(self, func: TermBase, arg: TermBase):
        # pylint: disable=super-init-not-called  # a context is a list here
        assert_isinstance(func, TermBase)
        assert_isinstance(arg, TermBase)
        if arg.stack or func.offset is None\
                or func.stack[:len(arg.cod)] != arg.cod:
            raise ValueError(messages.NOT_APPLICABLE.format(
                func, func.stack, arg, arg.stack, arg.cod))
        self.func, self.arg = func, arg
        before = self.share(func.freevars[:func.offset], arg.freevars)
        self.freevars = self.share(before, func.freevars[func.offset:])
        self.stack, self.cod = func.stack[len(arg.cod):], func.cod
        self.offset = len(before) if self.stack else None

    @staticmethod
    def share(*contexts: list) -> list:
        """
        The context of terms side by side: their free variables in turn,
        which have to be disjoint where nothing is copied.
        """
        result = [x for context in contexts for x in context]
        if len(set(result)) < len(result):
            raise ValueError("Expected disjoint free variables.")
        return result

    def eval(self, functor=None):
        functor = self.identity(functor)
        func, identity = self.func, functor.cod.id
        left = identity(functor(self.types(func.freevars[:func.offset])))
        right = identity(functor(
            self.stack @ self.types(func.freevars[func.offset:])))
        return left @ self.arg.eval(functor) @ right >> func.eval(functor)


class Kappa(TermBase, cat.Kappa):
    """
    The abstraction of a block of ``variables`` of the ``body``, contiguous
    in its context and next to its stack if it has one: they join the stack
    where they stand, at its front if they are left of it and at its back
    otherwise, so that the kappa of a term is the same diagram.

    Parameters:
        variables : The variables to bind, in the order of the context.
        body : The term in which they are bound.
        offset : The number of free variables left of the stack, needed
            when nothing says where it is, no variable and no stack in the
            body: the place for a box with no output.

    Note
    ----
    Applying a type to a Python function names the variables after the
    parameters, one for each wire, see :meth:`Ty.__call__`.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, a, b = Constant(Box('f', x @ y, x)), Variable('a', x), \\
    ...     Variable('b', y)
    >>> term = Kappa((b, ), f(a, b))
    >>> assert term.freevars == [a] and term.stack == y
    >>> assert term == y(lambda b: f(a, b)) and term.eval() == f.box
    >>> assert Kappa((a, ), term).stack == x @ y == Kappa((a, ), f(a)).stack
    >>> Kappa((b, a), f(a, b))
    Traceback (most recent call last):
        ...
    ValueError: Expected a block of variables next to the stack of f(a, b), \
got b, a.

    A box with no output is put between two wires by an ``offset``:

    >>> e, c = Constant(Box('e', x, Ty())), Variable('c', x)
    >>> term = let(e(a), lambda: f(c, b), offset=1)
    >>> assert term.freevars == [c, a, b]
    >>> assert term.eval() == x @ e.box @ y >> f.box
    >>> let(e(a), lambda: f(c, b))
    Traceback (most recent call last):
        ...
    ValueError: Expected an offset for the abstraction of no variable in \
f(c, b), which has no stack to say where it stands.
    >>> y(lambda b: f(c, b), offset=0)
    Traceback (most recent call last):
        ...
    ValueError: Expected offset 1 for the abstraction of b in f(c, b), got 0.
    """
    offset: int
    unbound: ClassVar[str] = messages.NOT_A_BLOCK

    @staticmethod
    def place(variables, body):
        """
        The number of free variables left of the stack: where the first of
        the variables stands in the context of the body, or else where its
        stack does, ``None`` when it has none.
        """
        return body.freevars.index(variables[0]) if variables\
            else body.offset

    def __init__(self, variables: tuple[Variable, ...], body: cat.TermBase,
                 offset: int | None = None):
        # pylint: disable=super-init-not-called  # a context is a list here
        assert_isinstance(body, cat.TermBase)
        variables = tuple(variables)
        start = self.start(variables, body, offset)
        stop = start + len(variables)
        if tuple(body.freevars[start:stop]) != variables\
                or body.offset not in (None, start, stop):
            raise ValueError(self.unbound.format(
                body, ", ".join(x.name for x in variables)))
        self.variables, self.body, self.offset = variables, body, start
        self.freevars = body.freevars[:start] + body.freevars[stop:]
        self.cod, bound = body.cod, body.types(variables)
        self.stack = bound @ body.stack if self.front else body.stack @ bound

    @staticmethod
    def binds(variables: tuple, body: cat.TermBase) -> bool:
        """
        Whether a body has the variables to bind: distinct ones, free in
        it, each used exactly once where nothing is copied or discarded.
        """
        return len(set(variables)) == len(variables)\
            and set(variables) <= set(body.freevars)

    @classmethod
    def start(cls, variables: tuple, body: cat.TermBase,
              offset: int | None) -> int:
        """
        The number of free variables left of the stack: where the
        variables or the stack of the body :meth:`place` it, or else the
        ``offset``, which is refused when it says otherwise.
        """
        names = ", ".join(x.name for x in variables) or "no variable"
        if not cls.binds(variables, body):
            raise ValueError(cls.unbound.format(body, names))
        expected = cls.place(variables, body)
        start = expected if offset is None else offset
        if start is None:
            raise ValueError(messages.NO_OFFSET.format(body))
        rest = len(set(body.freevars).difference(variables))
        wanted = f"from 0 to {rest}" if expected is None else expected
        valid = expected is not None or 0 <= start <= rest
        if not valid or expected not in (None, start):
            raise ValueError(
                messages.WRONG_OFFSET.format(wanted, names, body, start))
        return start


type Term = Variable | Constant | Application | Kappa


class InterchangeEquation(Equation):
    """
    An :class:`Equation` between diagrams which holds when
    they are equal up to interchange, i.e. as morphisms of the free monoidal
    category rather than of the premonoidal one: when the others are in the
    :meth:`orbit` of the first under interchangers.

    The hypergraphs of such diagrams are equal, and the converse fails: a
    hypergraph forgets on which side of a wire a box with no wire sits. No
    interchanger moves it across a wire that runs from the domain to the
    codomain, only round the end of one: above a state or below an effect.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, s = Box('f', x, y), Box('s', Ty(), Ty())
    >>> assert InterchangeEquation(s @ x >> f, f >> s @ y, s @ f)
    >>> assert not InterchangeEquation(s @ x, x @ s)
    >>> assert (s @ x).to_hypergraph() == (x @ s).to_hypergraph()
    >>> cap, cup = Box('cap', Ty(), x), Box('cup', x @ x, Ty())
    >>> assert InterchangeEquation(
    ...     x @ s >> x @ cap >> cup, s @ x >> x @ cap >> cup)

    Note
    ----
    The search is exhaustive and an orbit grows as the factorial of the
    number of boxes that commute: this is for small diagrams, such as a law
    draws. A boundary-connected diagram has a normal form instead, see
    :cite:t:`DelpeuchVicary22` and :meth:`Diagram.normal_form`.
    """
    def __bool__(self):
        first, *others = (term.to_staircases() for term in self.terms)
        orbit = self.orbit(first, *others)
        return all(other in orbit for other in others)

    @staticmethod
    def swaps(diagram: Diagram, i: int) -> list[Diagram]:
        """
        The interchanges of layer ``i`` with the one above it: none when
        the two boxes do not commute, two when a box with no input goes
        either side of a box with no output, one otherwise.

        Example
        -------
        >>> x = Ty('x')
        >>> e, s = Box('e', x, Ty()), Box('s', Ty(), x)
        >>> for diagram in InterchangeEquation.swaps(e >> s, 1):
        ...     print(diagram)
        s @ x >> x @ e
        x @ s >> e @ x
        >>> assert not InterchangeEquation.swaps(s >> e, 1)
        """
        result = []
        for left in (False, True):
            try:
                swapped = diagram.interchange(i, i - 1, left=left)
            except AxiomError:
                continue
            result += [swapped] if swapped not in result else []
        return result

    @classmethod
    def orbit(cls, diagram: Diagram, *until: Diagram) -> list[Diagram]:
        """
        The diagrams that a diagram with one box in each layer rewrites
        into by interchangers, itself first, breadth first: all of them,
        or as many as it takes to find every diagram of ``until``.

        Example
        -------
        >>> x = Ty('x')
        >>> f, g = Box('f', x, x), Box('g', x, x)
        >>> for diagram in InterchangeEquation.orbit(f @ x >> x @ g):
        ...     print(diagram)
        f @ x >> x @ g
        x @ g >> f @ x
        >>> assert len(InterchangeEquation.orbit(f @ g @ f @ g)) == 24
        >>> assert len(InterchangeEquation.orbit(f >> g)) == 1
        """
        result, seen, missing = [diagram], {diagram}, set(until) - {diagram}
        position = 0
        while position < len(result) and (missing or not until):
            current, position = result[position], position + 1
            found = dict.fromkeys(
                swapped for i in range(1, len(current))
                for swapped in cls.swaps(current, i) if swapped not in seen)
            result += found
            seen.update(found)
            missing.difference_update(found)
        return result


@dataclass
class Sampler:
    """
    Samples a term with no stack from a sequence of choices, one per node,
    taking leaves once it runs out, see :meth:`TermBase.generate`.

    The term is planar and linear: each subterm is given the variables it
    uses, in order, and a node :meth:`split` them between its subterms. It
    is a leaf, a fresh constant applied to subterms, or a statement binding
    up to two outputs between the variables left for its body.

    Parameters:
        terms : The terms to sample, i.e. their factories.
        choices : The choices left, consumed one per node.
        types : The types of one wire the variables are drawn from.
        letters : The letters naming the bound variables, one per level.
        counter : The numbers of the constants built.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> sampler = Sampler(TermBase, iter([2, 0, 1, 1, 0, 0, 0]), [x], "a")
    >>> print(sampler.state(y, (Variable('v', x), Variable('w', x))))
    let(c0(v), lambda: c1(w), offset=0)
    """
    terms: type[TermBase]
    choices: Iterator[int]
    types: Sequence[Ty]
    letters: Sequence[str | None]
    counter: Iterator[int] = field(default_factory=count)

    def choose(self, options: Sequence):
        """ Pick an option by the next choice, the first when they ran out. """
        return options[next(self.choices, 0) % len(options)]

    def constant(
            self, dom: Sequence[Ty], cod: Sequence[Ty]) -> cat.Constant:
        """ A fresh constant between the tensors of two lists of types. """
        category = self.terms.functor.cod
        return self.terms.constant_factory(category.box_factory(
            f"c{next(self.counter)}", category.ob().tensor(*dom),
            category.ob().tensor(*cod)))

    def bound(self, level: int, cod: Ty,
              shadowed: cat.Variable | None = None) -> cat.Variable:
        """
        The variable bound at a given level, named by its letter or after
        the variable it shadows, see :meth:`discopy.cat.TermBase.bound_name`.
        """
        return self.terms.variable_factory(
            self.terms.bound_name(self.letters, level, shadowed), cod)

    def shuffle(self, bound: tuple) -> tuple:
        """
        The bound variables in the order a term uses them: their own in a
        monoidal category, picked by a choice where there is exchange.
        """
        return bound

    def spine(
            self, cod: Sequence[Ty], bound: tuple[cat.Variable, ...]
            ) -> cat.TermBase:
        """ A fresh constant applied to the bound variables in order. """
        return self.constant([x.cod for x in bound], cod)(*bound)

    def split(self, bound: tuple, n: int) -> list[tuple]:
        """ The bound variables cut into ``n`` contiguous blocks. """
        cuts = sorted(
            self.choose(range(len(bound) + 1)) for _ in range(n - 1))
        return [bound[i:j] for i, j in zip([0] + cuts, cuts + [len(bound)])]

    def state(self, cod: Ty, bound: tuple[cat.Variable, ...] = (),
              level: int = 0) -> cat.TermBase:
        """
        A term of a given type with no stack and the bound variables as
        context: a leaf when the choices ran out, else what the next
        choice picks.
        """
        kind, bound = next(self.choices, 0) % 3, self.shuffle(bound)
        if kind == 0:
            return bound[0] if [x.cod for x in bound] == [cod]\
                else self.spine([cod], bound)
        if kind == 1:
            dom = [self.choose(self.types)
                   for _ in range(1 + self.choose(range(2)))]
            return self.constant(dom, [cod])(*(
                self.state(typ, block, level)
                for typ, block in zip(dom, self.split(bound, len(dom)))))
        left, middle, right = self.split(bound, 3)
        types = [
            self.choose(self.types) for _ in range(self.choose((1, 0, 2)))]
        used_up = [x for x in dict.fromkeys(middle) if x not in left + right]
        variables = tuple(
            self.bound(level + i, typ, next(iter(used_up[i:]), None))
            for i, typ in enumerate(types))
        inside = self.shuffle(left + variables + right)
        expression = self.state(types[0], middle, level) if len(types) == 1\
            else self.spine(types, middle)
        body = self.state(cod, inside, level + len(types))
        return self.terms.bind(variables, body, len(left))(expression)


TermBase.functor, TermBase.sampler_factory = Functor.id(Diagram), Sampler
Diagram.term_factory = TermBase
TermBase.variable_factory, TermBase.constant_factory = Variable, Constant
TermBase.application_factory, TermBase.kappa_factory = Application, Kappa

__getattr__ = deprecated_alias(
    __name__, {"PRO": "Nat", "Sum": Diagram.sum_factory})
