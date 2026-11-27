# -*- coding: utf-8 -*-

"""
The free comonad category, i.e. a category with a comonad ``!``, see
:class:`~discopy.abc.ComonadCategory`: the :class:`Bang` of an arrow, a box
with the arrow as its arg, the :class:`Dereliction` ``!x -> x`` and the
:class:`Digging` ``!x -> !!x``. It needs no tensor: the contraction, the
weakening, the laxator and the exchange are those of
:mod:`linear.monoidal <discopy.linear.monoidal>`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ob
    BangOb
    Arrow
    Box
    Bang
    Structural
    Dereliction
    Digging
    Functor

Example
-------
The promotion of an arrow ``!x -> y`` to ``!x -> !y`` is the digging then
the bang, and derelicting it is the arrow up to the comonad laws, which the
free category does not quotient by.

>>> x, y = Ob('x'), Ob('y')
>>> g = Box('g', x.bang(), y)
>>> promotion = Digging(x) >> g.bang()
>>> assert (promotion.dom, promotion.cod) == (x.bang(), y.bang())
>>> assert promotion >> Dereliction(y) != g
>>> print(promotion >> Dereliction(y))
Digging(x) >> (g).bang() >> Dereliction(y)
"""

from __future__ import annotations

from functools import cached_property
from typing import ClassVar

from discopy import cat, messages
from discopy.abc import Category, ComonadCategory, DaggerCategory
from discopy.cat import factory
from discopy.utils import (
    AxiomError, assert_isinstance, factory_name, from_tree, get_origin)


class Ob(cat.Ob):
    """
    An object with a bang, a :class:`~discopy.cat.Ob` that may be banged.
    """
    bang_factory: ClassVar[type[BangOb]]

    def bang(self) -> Ob:
        """
        The bang ``!x`` of an object, a :class:`BangOb` holding it.

        Example
        -------
        >>> x = Ob('x')
        >>> assert x.bang() == BangOb(x) != x
        >>> assert eval(str(x.bang().bang())) == x.bang().bang()
        """
        return self.bang_factory(self)


class BangOb(Ob):
    """
    The bang ``!x`` of an object ``x``, an object of its own holding it.

    Parameters:
        arg : The object to bang.

    Example
    -------
    >>> from discopy import linear
    >>> x = Ob('x')
    >>> assert eval(repr(BangOb(x))) == BangOb(x)
    >>> assert str(BangOb(x)) == "x.bang()"
    """
    ob = Ob

    def __init__(self, arg: Ob):
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
        return f"{self.arg}.bang()"

    def __repr__(self):
        return factory_name(type(self)) + f"({self.arg!r})"

    def to_tree(self):
        return {'factory': factory_name(type(self)),
                'arg': self.arg.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        return cls(from_tree(tree['arg']))


Ob.bang_factory = BangOb


@factory
class Arrow(cat.Arrow, ComonadCategory):
    """
    An arrow in the free comonad category: a :class:`~discopy.cat.Arrow` with
    :class:`Bang` boxes and the :class:`Dereliction` and :class:`Digging`
    boxes, see :class:`~discopy.abc.ComonadCategory`.

    Parameters:
        inside (tuple[Box, ...]) : The boxes inside the arrow.
        dom (Ob) : The domain of the arrow, i.e. its input.
        cod (Ob) : The codomain of the arrow, i.e. its output.
    """
    box_factory: ClassVar[type["Box"]]
    functor_factory: ClassVar[type["Functor"]]
    bang_factory: ClassVar[type["Bang"]]
    dereliction_factory: ClassVar[type["Dereliction"]]
    digging_factory: ClassVar[type["Digging"]]

    ob = Ob
    dagger_involution = DaggerCategory.dagger_involution.failing(
        messages.STRUCTURAL_HAS_NO_DAGGER)
    dagger_contravariance = DaggerCategory.dagger_contravariance.failing(
        messages.STRUCTURAL_HAS_NO_DAGGER)

    def bang(self) -> Arrow:
        """
        Wrapper around :class:`Bang` called by :class:`Functor`: the bang of
        a free arrow is a box with the arrow as its arg, so it is a functor
        only up to the laws the free category declares failing.

        Example
        -------
        >>> x, y = Ob('x'), Ob('y')
        >>> f = Box('f', x, y)
        >>> assert f.bang() == Bang(f) and f.bang().dom == x.bang()
        >>> assert Arrow.id(x).bang() != Arrow.id(x.bang())
        """
        return self.bang_factory(self)

    @classmethod
    def dereliction(cls, x: Ob) -> Dereliction:
        """ Wrapper around :class:`Dereliction`. """
        return cls.dereliction_factory(x)

    @classmethod
    def digging(cls, x: Ob) -> Digging:
        """ Wrapper around :class:`Digging`. """
        return cls.digging_factory(x)


class Box(cat.Box, Arrow):
    """ A box in the free comonad category. """


class Bang(Box):
    """
    The bang ``!f : !x -> !y`` of an arrow ``f : x -> y``, a box with ``f``
    as its arg.

    Parameters:
        arg : The arrow to bang.

    Example
    -------
    >>> from discopy import linear
    >>> x, y = Ob('x'), Ob('y')
    >>> f = Box('f', x, y)
    >>> assert Bang(f).dom == x.bang() and Bang(f).cod == y.bang()
    >>> assert eval(repr(Bang(f))) == Bang(f) and eval(str(Bang(f))) == Bang(f)
    >>> print(Bang(f >> f[::-1]))
    (f >> f[::-1]).bang()
    """
    serialised_attrs = ("*args", )

    def __init__(self, arg: Arrow):
        dom, cod = arg.dom.bang(), arg.cod.bang()
        Box.__init__(self, f"({arg}).bang()", dom, cod, args=(arg, ))

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
    A structural box of a comonad category, built from the objects of its
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
            attr: value.to_tree() if isinstance(value, cat.Ob) else value
            for attr, value in zip(
                self.serialised_attrs, self.__getnewargs__())}

    @classmethod
    def from_tree(cls, tree):
        return cls(**{
            attr: from_tree(value) if isinstance(value, dict) else value
            for attr, value in tree.items() if attr != "factory"})

    def dagger(self):
        """
        A structural box has no dagger, but for the exchange of
        :mod:`linear.monoidal <discopy.linear.monoidal>`.

        >>> Dereliction(Ob('x')).dagger()
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
        x : The object to derelict.

    Example
    -------
    >>> x = Ob('x')
    >>> assert Dereliction(x).dom == x.bang() and Dereliction(x).cod == x
    """
    serialised_attrs = ("x", )
    drawing_name = "ε"

    def __init__(self, x: Ob):
        self.x = x
        super().__init__(f"Dereliction({x})", x.bang(), x)


class Digging(Structural):
    """
    The digging ``!x -> !!x``, the comultiplication of the bang.

    Parameters:
        x : The object to dig.

    Example
    -------
    >>> x = Ob('x')
    >>> assert Digging(x).cod == x.bang().bang()
    """
    serialised_attrs = ("x", )
    drawing_name = "δ"

    def __init__(self, x: Ob):
        self.x = x
        super().__init__(f"Digging({x})", x.bang(), x.bang().bang())


Id = Arrow.id
Arrow.box_factory = Box
Arrow.bang_factory = Bang
Arrow.dereliction_factory = Dereliction
Arrow.digging_factory = Digging
for law in sorted(set(ComonadCategory.axioms) - set(Category.axioms)):
    setattr(Arrow, law, getattr(ComonadCategory, law).failing(
        messages.FREE_BANG))


class Functor(cat.Functor):
    """
    A functor from the free comonad category: into a comonad category it
    sends a :class:`BangOb`, a :class:`Bang` and the structural boxes through
    the bang and the structure of its codomain, and into any other category
    through its ``ob_map`` and ``ar_map``.

    Example
    -------
    >>> x, y = Ob('x'), Ob('y')
    >>> f = Box('f', x, y)
    >>> F = Functor({x: y, y: x}, {f: f[::-1]})
    >>> assert F(Digging(x) >> f.bang().bang() >> Dereliction(y).bang())\\
    ...     == Digging(y) >> f[::-1].bang().bang() >> Dereliction(x).bang()
    """
    dom = cod = Arrow

    @property
    def has_bang(self) -> bool:
        """
        Whether the codomain is a comonad category, which a :class:`BangOb`,
        a :class:`Bang` and the structural boxes are sent through.
        """
        return issubclass(get_origin(self.cod), ComonadCategory)

    def map_bang(self, other):
        """
        The image of a :class:`BangOb`, a :class:`Bang` or a structural box
        in a codomain with a bang, see :attr:`has_bang`.

        Parameters:
            other : The object or box to map.
        """
        if isinstance(other, BangOb):
            return self.cod.banged(self(other.arg))
        if isinstance(other, Bang):
            return self(other.arg).bang()
        method = getattr(self.cod, type(other).__name__.lower())
        return method(*map(self, other.__getnewargs__()))

    def __call__(self, other):
        if isinstance(other, (BangOb, Bang, Structural)):
            if self.has_bang:
                return self.map_bang(other)
            if isinstance(other, Bang):
                return self.ar_map[other]
        return super().__call__(other)


Arrow.functor_factory = Functor
