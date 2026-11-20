# -*- coding: utf-8 -*-

"""
The free linear category, i.e. a monoidal category with a bang: a comonad
``!`` whose objects are comonoids, central and lax monoidal, see
:class:`discopy.abc.LinearCategory`. It needs no symmetry and no closure,
so it sits right above :mod:`discopy.monoidal`, the way :mod:`discopy.para`
and :mod:`discopy.monad` sit beside the hierarchy.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    BangOb
    Diagram
    Box
    Sum
    Bubble
    Bang
    Structural
    Dereliction
    Digging
    Contraction
    Weakening
    Exchange
    Laxator
    Functor

Example
-------
The translation of the non-linear term ``lambda x: (f(x), f(x))`` banged its
argument: it copies ``!x`` by contraction and reads each copy once by
dereliction, after the bang of ``f``.

>>> x, y = Ty('x'), Ty('y')
>>> f = Box('f', x, y)
>>> twice = f.bang() >> Contraction(y) >> Dereliction(y) @ Dereliction(y)
>>> assert twice.dom == x.bang() and twice.cod == y @ y
>>> twice.draw(doctest='docs/_static/bang/twice.svg')

.. image:: /_static/bang/twice.svg
    :align: center

A :class:`Functor` into a category without a bang, e.g.
:class:`discopy.tensor.Diagram`, sends banged wires and boxes through its
``ob_map`` and ``ar_map``, e.g. ``!x`` to a Fock space and ``f.bang()`` to
the second quantisation of ``f``, see the notebook
:doc:`/notebooks/optyx/linear_logic`.
"""

from __future__ import annotations

from functools import cached_property
from typing import ClassVar, Self

from discopy import monoidal, messages
from discopy.abc import LinearCategory, MonoidalCategory
from discopy.cat import factory
from discopy.drawing import Drawing
from discopy.utils import (
    AxiomError, assert_isinstance, factory_name, from_tree, get_origin)


@factory
class Ty(monoidal.Ty):
    """
    A type with a bang, a monoidal type whose wires may be
    :class:`BangOb`.
    """
    bang_factory: ClassVar[type[BangOb]]

    def bang(self) -> Self:
        """
        The bang of a type, a single wire :class:`BangOb` holding it: the
        bang is lax monoidal, ``!(x @ y)`` is one wire and not ``!x @ !y``.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> assert x.bang() == Ty(BangOb(x)) != Ty(BangOb(x @ y))
        >>> assert (x @ y).bang() != x.bang() @ y.bang()
        >>> assert str(x.bang()) == "x.bang()"
        >>> assert eval(str((x @ y).bang().bang())) == (x @ y).bang().bang()
        """
        return type(self)(self.bang_factory(self))


class BangOb(monoidal.Wire):
    """
    The bang ``!x`` of a type ``x``, a wire of its own holding it, the way an
    :class:`Exp` holds its base and exponent.

    Parameters:
        arg : The type to bang.

    Example
    -------
    >>> from discopy import bang
    >>> x, y = Ty('x'), Ty('y')
    >>> assert Ty(BangOb(x @ y)) == (x @ y).bang()
    >>> assert eval(repr(BangOb(x))) == BangOb(x)
    >>> assert BangOb(x @ y).drawing_name == "!(x @ y)"
    """
    ob = Ty

    def __init__(self, arg: Ty):
        assert_isinstance(arg, self.ob)
        self.arg = arg
        super().__init__("")
        del self.name

    @cached_property
    def name(self):  # pylint: disable=method-hidden  # deleted by __init__
        "The name of a bang is its print, computed when first read."
        return str(self)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.arg == other.arg

    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        arg = str(self.arg) if len(self.arg) == 1 else f"({self.arg})"
        return arg + ".bang()"

    @property
    def drawing_name(self) -> str:
        """ The label of a banged wire, e.g. ``!x`` or ``!(x @ y)``. """
        arg = self.arg.to_drawing()
        return "!" + (str(arg) if len(arg) == 1 else f"({arg})")

    def atoms(self) -> list:
        """ The atoms of a bang: those of the type it holds. """
        return self.arg.atoms()

    def __repr__(self):
        return factory_name(type(self)) + f"({self.arg!r})"

    def to_tree(self):
        return {'factory': factory_name(type(self)),
                'arg': self.arg.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        return cls(from_tree(tree['arg']))

    def dagger(self) -> BangOb:
        """ A banged wire is its own dagger, as it has no colour. """
        return self


Ty.bang_factory = BangOb


@factory
class Diagram(monoidal.Diagram, LinearCategory):
    """
    A diagram in the free linear category: a monoidal diagram with
    :class:`Bang` bubbles and the :class:`Dereliction`, :class:`Digging`,
    :class:`Contraction`, :class:`Weakening`, :class:`Exchange` and
    :class:`Laxator` boxes, see :class:`discopy.abc.LinearCategory`.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    box_factory: ClassVar[type["Box"]]
    sum_factory: ClassVar[type["Sum"]]
    functor_factory: ClassVar[type["Functor"]]
    bang_factory: ClassVar[type["Bang"]]
    dereliction_factory: ClassVar[type["Dereliction"]]
    digging_factory: ClassVar[type["Digging"]]
    contraction_factory: ClassVar[type["Contraction"]]
    weakening_factory: ClassVar[type["Weakening"]]
    exchange_factory: ClassVar[type["Exchange"]]
    laxator_factory: ClassVar[type["Laxator"]]

    ob = Ty

    def bang(self) -> Diagram:
        """
        Wrapper around :class:`Bang` called by :class:`Functor`: the bang of
        a free diagram is a bubble, so it is a functor only up to the laws
        the free category declares failing.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> f = Box('f', x, y)
        >>> assert f.bang() == Bang(f) and f.bang().dom == x.bang()
        >>> assert Diagram.id(x).bang() != Diagram.id(x.bang())
        """
        return self.bang_factory(self)

    @classmethod
    def dereliction(cls, x: Ty) -> Dereliction:
        """ Wrapper around :class:`Dereliction`. """
        return cls.dereliction_factory(x)

    @classmethod
    def digging(cls, x: Ty) -> Digging:
        """ Wrapper around :class:`Digging`. """
        return cls.digging_factory(x)

    @classmethod
    def contraction(cls, x: Ty) -> Contraction:
        """ Wrapper around :class:`Contraction`. """
        return cls.contraction_factory(x)

    @classmethod
    def weakening(cls, x: Ty) -> Weakening:
        """ Wrapper around :class:`Weakening`. """
        return cls.weakening_factory(x)

    @classmethod
    def exchange(cls, x: Ty, y: Ty, is_dagger: bool = False) -> Exchange:
        """ Wrapper around :class:`Exchange`. """
        return cls.exchange_factory(x, y, is_dagger)

    @classmethod
    def laxator(cls, x: Ty | None = None, y: Ty | None = None) -> Laxator:
        """ Wrapper around :class:`Laxator`. """
        return cls.laxator_factory(x, y)


class Box(monoidal.Box, Diagram):
    """
    A box in the free linear category, i.e. a monoidal box of its diagrams.
    """


class Sum(monoidal.Sum, Box):
    """
    A sum in the free linear category, i.e. a monoidal sum and its box.
    """


class Bubble(monoidal.Bubble, Box):
    """
    A bubble in the free linear category, i.e. a monoidal bubble and its box.
    """


class Bang(monoidal.Bubble, Box):
    """
    The bang ``!f : !x -> !y`` of a diagram ``f : x -> y``, a bubble the way
    a :class:`Curry` is.

    Parameters:
        arg : The diagram to bang.

    Example
    -------
    >>> from discopy import bang
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> assert Bang(f).dom == x.bang() and Bang(f).cod == y.bang()
    >>> assert eval(repr(Bang(f))) == Bang(f) and eval(str(Bang(f))) == Bang(f)
    >>> print(Bang(f >> f[::-1]))
    (f >> f[::-1]).bang()

    The promotion of !L*, from ``!x -> y`` to ``!x -> !y``, is the digging
    followed by the bang:

    >>> g = Box('g', x.bang(), y)
    >>> promotion = Digging(x) >> g.bang()
    >>> assert promotion.dom == x.bang() and promotion.cod == y.bang()
    """
    serialised_attrs = ("*args", )

    def __init__(self, arg: Diagram):
        dom, cod = arg.dom.bang(), arg.cod.bang()
        monoidal.Bubble.__init__(
            self, arg, dom=dom, cod=cod, drawing_name="!")
        Box.__init__(self, f"({arg}).bang()", dom, cod)

    def __repr__(self):
        return f"{factory_name(type(self))}({self.arg!r})"

    def __str__(self):
        return self.name

    def to_tree(self):
        return {'factory': factory_name(type(self)),
                'arg': self.arg.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        return cls(from_tree(tree['arg']))

    def dagger(self) -> Bang:
        """ The dagger of a bang is the bang of the dagger. """
        return type(self)(self.arg.dagger())


class Structural(Box):
    """
    A structural box of a linear category, built from the types of its
    :attr:`serialised_attrs` rather than a name and a boundary: its name is
    the call on them, which is its string.
    """
    def __getnewargs__(self):
        """ The arguments of the constructor. """
        return tuple(getattr(self, attr) for attr in self.serialised_attrs)

    def __repr__(self):
        arguments = (
            repr(value) for value in self.__getnewargs__()
            if value is not None)
        return f"{factory_name(type(self))}({', '.join(arguments)})"

    def to_tree(self):
        return {'factory': factory_name(type(self))} | {
            attr: value.to_tree() if isinstance(value, Ty) else value
            for attr, value in zip(
                self.serialised_attrs, self.__getnewargs__())}

    @classmethod
    def from_tree(cls, tree):
        return cls(**{
            attr: from_tree(value) if isinstance(value, dict) else value
            for attr, value in tree.items() if attr != "factory"})

    def dagger(self):
        """
        A structural box has no dagger but for the exchange.

        >>> Dereliction(Ty('x')).dagger()
        Traceback (most recent call last):
        ...
        discopy.utils.AxiomError: Dereliction(x) has no dagger: ...
        """
        raise AxiomError(messages.BANG_HAS_NO_DAGGER.format(self))


class Dereliction(Structural):
    """
    The dereliction ``!x -> x``, the counit of the bang and the rule of !L*
    deriving ``Γ, !x, Δ ⊢ y`` from ``Γ, x, Δ ⊢ y``.

    Parameters:
        x : The type to derelict.

    Example
    -------
    >>> x = Ty('x')
    >>> assert Dereliction(x).dom == x.bang() and Dereliction(x).cod == x
    """
    serialised_attrs = ("x", )
    drawing_name = "ε"

    def __init__(self, x: Ty):
        self.x = x
        super().__init__(f"Dereliction({x})", x.bang(), x)

    @classmethod
    # pylint: disable-next=arguments-differ  # drawn with no boundary
    def strategy(cls, *, types=None):
        """
        Generate the box at an atomic type.

        Parameters:
            types : The strategy to draw the types from, see
                :meth:`atomic_strategy`.
        """
        return cls.atomic_strategy(types).map(cls)


class Digging(Structural):
    """
    The digging ``!x -> !!x``, the comultiplication of the bang.

    Parameters:
        x : The type to dig.

    Example
    -------
    >>> x = Ty('x')
    >>> assert Digging(x).cod == x.bang().bang()
    """
    serialised_attrs = ("x", )
    drawing_name = "δ"
    strategy = classmethod(Dereliction.strategy.__func__)

    def __init__(self, x: Ty):
        self.x = x
        super().__init__(f"Digging({x})", x.bang(), x.bang().bang())


class Contraction(Structural):
    """
    The contraction ``!x -> !x @ !x``, the rule of !L* deriving
    ``Γ, !x, Δ ⊢ y`` from ``Γ, !x, !x, Δ ⊢ y``.

    Parameters:
        x : The type to contract.

    Example
    -------
    >>> x = Ty('x')
    >>> assert Contraction(x).cod == x.bang() @ x.bang()
    """
    serialised_attrs = ("x", )
    drawing_name = "Δ"
    strategy = classmethod(Dereliction.strategy.__func__)

    def __init__(self, x: Ty):
        self.x = x
        super().__init__(f"Contraction({x})", x.bang(), x.bang() @ x.bang())


class Weakening(Structural):
    """
    The weakening ``!x -> Ty()``, the counit of the contraction. !L* is
    relevant: it has no weakening, which an example of it never uses.

    Parameters:
        x : The type to weaken.

    Example
    -------
    >>> x = Ty('x')
    >>> assert Weakening(x).dom == x.bang() and not Weakening(x).cod
    """
    serialised_attrs = ("x", )
    drawing_name = "e"
    strategy = classmethod(Dereliction.strategy.__func__)

    def __init__(self, x: Ty):
        self.x = x
        super().__init__(f"Weakening({x})", x.bang(), x[:0])


class Exchange(Structural):
    """
    The exchange ``!x @ y -> y @ !x`` of a banged type with any type, and its
    inverse ``y @ !x -> !x @ y`` when ``is_dagger``, the permutation rule of
    !L*: it makes ``!x`` central rather than the category symmetric.

    Parameters:
        x : The type that is banged.
        y : The type it is exchanged with.
        is_dagger : Whether to take the inverse.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> assert Exchange(x, y).dom == x.bang() @ y
    >>> assert Exchange(x, y).dagger() == Exchange(x, y, True)
    >>> assert Exchange(x, y, True).dom == y @ x.bang()
    >>> assert eval(str(Exchange(x, y)[::-1])) == Exchange(x, y, True)
    """
    serialised_attrs = ("x", "y", "is_dagger")
    drawing_name = "σ"

    def __init__(self, x: Ty, y: Ty, is_dagger: bool = False):
        self.x, self.y = x, y
        dom, cod = x.bang() @ y, y @ x.bang()
        super().__init__(f"Exchange({x}, {y})", *(
            (cod, dom) if is_dagger else (dom, cod)), is_dagger=is_dagger)

    def dagger(self) -> Exchange:
        return type(self)(self.x, self.y, not self.is_dagger)

    @classmethod
    # pylint: disable-next=arguments-differ  # drawn with no boundary
    def strategy(cls, *, types=None):
        """
        Generate the exchange of two atomic types, or its inverse.

        Parameters:
            types : The strategy to draw the types from, see
                :meth:`atomic_strategy`.
        """
        from hypothesis import strategies as st

        atoms = cls.atomic_strategy(types)
        return st.builds(cls, atoms, atoms, st.booleans())


class Laxator(Structural):
    """
    The laxator ``!x @ !y -> !(x @ y)`` of the bang, and its unit
    ``Ty() -> !Ty()`` when given no type.

    Parameters:
        x : The type on the left.
        y : The type on the right.

    Example
    -------
    >>> from discopy import bang
    >>> x, y = Ty('x'), Ty('y')
    >>> assert Laxator(x, y).dom == x.bang() @ y.bang()
    >>> assert Laxator(x, y).cod == (x @ y).bang()
    >>> assert not Laxator().dom and Laxator().cod == Ty().bang()
    >>> assert eval(repr(Laxator())) == Laxator()
    """
    serialised_attrs = ("x", "y")
    drawing_name = "m"

    def __init__(self, x: Ty | None = None, y: Ty | None = None):
        if (x is None) != (y is None):
            raise ValueError(messages.LAXATOR_ARITY.format((x, y)))
        self.x, self.y = x, y
        if x is None or y is None:
            name, dom = "Laxator()", self.ob()
            cod = Ty.bang(dom)
        else:
            name, dom = f"Laxator({x}, {y})", x.bang() @ y.bang()
            cod = (x @ y).bang()
        super().__init__(name, dom, cod)

    @classmethod
    # pylint: disable-next=arguments-differ  # drawn with no boundary
    def strategy(cls, *, types=None):
        """
        Generate the laxator of two atomic types, or its unit.

        Parameters:
            types : The strategy to draw the types from, see
                :meth:`atomic_strategy`.
        """
        from hypothesis import strategies as st

        atoms = cls.atomic_strategy(types)
        return st.one_of(st.just(cls()), st.builds(cls, atoms, atoms))


Id = Diagram.id
Diagram.box_factory = Box
Diagram.sum_factory = Sum
Diagram.bubble_factory = Bubble
Diagram.bang_factory = Bang
Diagram.dereliction_factory = Dereliction
Diagram.digging_factory = Digging
Diagram.contraction_factory = Contraction
Diagram.weakening_factory = Weakening
Diagram.exchange_factory = Exchange
Diagram.laxator_factory = Laxator
for law in sorted(
        set(LinearCategory.axioms) - set(MonoidalCategory.axioms)):
    setattr(Diagram, law, getattr(LinearCategory, law).failing(
        messages.FREE_BANG))


class Functor(monoidal.Functor):
    """
    A functor from the free linear category: into a linear category it
    sends a :class:`BangOb`, a :class:`Bang` and the structural boxes
    through the bang and the structure of its codomain, and into any other
    category, e.g. :class:`discopy.tensor.Diagram`, through its ``ob_map``
    and ``ar_map``.

    Example
    -------
    >>> from discopy import tensor
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> F = Functor({x: y, y: x}, {f: f[::-1]})
    >>> assert F(Contraction(x) >> f.bang() @ Weakening(x)) == (
    ...     Contraction(y) >> f[::-1].bang() @ Weakening(y))
    >>> e = tensor.Box('e', tensor.Dim(3), tensor.Dim(2))
    >>> G = Functor(ob_map={x: tensor.Dim(2), x.bang(): tensor.Dim(3)},
    ...             ar_map={Dereliction(x): e}, cod=tensor.Diagram)
    >>> assert G(Dereliction(x)) == e
    """
    dom = cod = Diagram

    @property
    def has_bang(self) -> bool:
        """
        Whether the codomain is a linear category, which a :class:`BangOb`,
        a :class:`Bang` and the structural boxes are sent through.
        """
        return issubclass(get_origin(self.cod), LinearCategory)

    def map_bang(self, other):
        """
        The image of a :class:`BangOb`, a :class:`Bang` or a structural box
        in a codomain with a bang, see :attr:`has_bang`.

        Parameters:
            other : The wire or box to map.
        """
        if isinstance(other, (BangOb, Bang)):
            return self(other.arg).bang()
        if isinstance(other, Laxator):
            return self.cod.laxator() if other.x is None\
                else self.cod.laxator(self(other.x), self(other.y))
        if isinstance(other, Exchange):
            return self.cod.exchange(
                self(other.x), self(other.y), other.is_dagger)
        method = getattr(self.cod, type(other).__name__.lower())
        return method(self(other.x))

    def __call__(self, other):
        if isinstance(other, (BangOb, Bang, Structural))\
                and self.cod is not Drawing:
            if self.has_bang:
                return self.map_bang(other)
            if isinstance(other, Bang):
                return self.ar_map[other]
        return super().__call__(other)


Diagram.functor_factory = Functor
