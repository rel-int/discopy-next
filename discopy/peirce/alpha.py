"""
Peirce's alpha graphs: propositions as diagrams on the unit type, with two
colours of region, the page :data:`white` and the inside of an odd number of
cuts :data:`black`, see :cite:t:`BonchiEtAl24`.

A white region is :class:`discopy.tensor.Tensor` of ``bool`` as it is: a box
is a relation, composition is ∃ over the middle wire and tensor is ∧. A black
region is the same structure conjugated by elementwise negation ¬: a black
box ``R`` is ¬R, black composition is the relative sum and black tensor is
∨. A :class:`Cut` is where the colour changes: the diagram inside it is the
:meth:`Diagram.negative` of the one outside, i.e. the same with its colours
swapped, so that the complement ``~d`` is the cut of the negative of ``d``,
with the boundary of ``d``.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Cut
    Functor
    Model

Example
-------
>>> p, q = Box('p'), Box('q')
>>> M = Model(ar={'p': True, 'q': False})
>>> assert M(p & q) == M(p) @ M(q) and not M(p & q).array.any()
>>> assert M(p | q).array.all() and not M(~p).array.any()

The double cut is the identity in every model:

>>> from itertools import product
>>> for P, Q in product((False, True), repeat=2):
...     M = Model(ar={'p': P, 'q': Q})
...     assert M(~~(p & ~q)) == M(p & ~q)
"""

from __future__ import annotations

import operator
from copy import copy

import numpy

from discopy import monoidal
from discopy.matrix import Matrix
from discopy.monoidal import Colour
from discopy.tensor import Dim, Tensor
from discopy.testable import no_strategy
from discopy.utils import AxiomError, factory, factory_name, from_tree

white, black = Colour("white"), Colour("black")
""" The colours of the two regions, the page and the inside of a cut. """

NEGATIVE = {white: black, black: white}
""" The other colour of each region, which a cut changes to. """


def negative_wire(wire: monoidal.Wire) -> monoidal.Wire:
    """
    The wire with the colour on each side swapped, all else kept.

    >>> wire = monoidal.Wire('x', dom=white, cod=white)
    >>> assert negative_wire(wire).dom == black == negative_wire(wire).cod
    """
    result = copy(wire)
    dom, cod = wire.dom, wire.cod
    result.dom, result.cod = NEGATIVE[dom], NEGATIVE[cod]  # ty: ignore
    return result


@factory
class Diagram(monoidal.Diagram):
    """
    An alpha diagram is a monoidal diagram on regions coloured :data:`white`
    or :data:`black`, the dunders its connectives.

    >>> p, q = Box('p'), Box('q')
    >>> assert p & q == p @ q
    >>> assert p | q == ~(~p & ~q)
    >>> assert (~p).colour == p.colour == white and (~p).arg.colour == black
    """
    strategy = no_strategy

    @classmethod
    def id(cls, dom=None) -> Diagram:
        """
        The identity, on the empty type of the white region by default.

        >>> assert Diagram.id().dom == Diagram.ob.id(white)
        """
        return super().id(cls.ob.id(white) if dom is None else dom)

    @property
    def colour(self) -> Colour:
        """ The colour of the region around the diagram. """
        return self.dom.dom

    @classmethod
    def negative_functor(cls) -> Functor:
        """
        The functor that swaps the colours of every region, from ``cls`` to
        itself, the category of diagrams ``cls`` is a box of.
        """
        diagram = cls.factory

        def ob_map(typ):
            return typ.factory(*map(negative_wire, typ.inside))

        def ar_map(box):
            negative = diagram.negative_functor()
            return box.rebuild(dom=negative(box.dom), cod=negative(box.cod))
        return diagram.functor_factory(
            ob_map, ar_map, dom=diagram, cod=diagram, colour_map=NEGATIVE)

    def negative(self) -> Diagram:
        """
        The photographic negative: the same diagram with the colour of every
        region swapped, which denotes its complement.

        >>> p = Box('p')
        >>> assert p.negative().colour == black
        >>> assert p.negative().negative() == p
        """
        return self.negative_functor()(self)

    def bubble(self, *args, **params) -> Cut:
        """ The :class:`Cut` of a diagram, the bubble of this category. """
        return Cut(self, *args, **params)

    def __invert__(self):
        """
        The complement ``~d``, the cut of the :meth:`negative` of ``d`` with
        the boundary of ``d``.

        >>> p = Box('p')
        >>> assert ~p == Cut(p.negative()) and (~p).colour == white
        """
        return self.negative().bubble(dom=self.dom, cod=self.cod)

    def conjunction(self, other: Diagram) -> Diagram:
        """ The conjunction of two white propositions, their tensor. """
        return self @ other

    def __and__(self, other: Diagram) -> Diagram:
        """
        The conjunction, :meth:`conjunction` in the white region and the
        negative of the disjunction of the negatives in the black one, so
        that a model reads it as ∧ in either.

        >>> p, q = Box('p').negative(), Box('q').negative()
        >>> M = Model(ar={'p': True, 'q': False})
        >>> assert M(p & q) == M(p) @ M(q) and M(q).array.all()
        """
        if self.colour == black:
            return (self.negative() | other.negative()).negative()
        return self.conjunction(other)

    def __or__(self, other: Diagram) -> Diagram:
        """ The disjunction, by de Morgan, ``~(~d & ~e)``. """
        return ~(~self & ~other)

    def relative_sum(self, other: Diagram) -> Diagram:
        """
        The relative sum, ∀ over the middle wire: ``~(~d >> ~e)`` in the
        white region, and the composition of the black one, i.e. the
        negative of the composition of the negatives.
        """
        if self.colour == black:
            return (self.negative() >> other.negative()).negative()
        return ~(~self >> ~other)

    def par(self, other: Diagram) -> Diagram:
        """
        The product ∨ of two relations: ``~(~d @ ~e)`` in the white region,
        and the tensor of the black one, i.e. the negative of the tensor of
        the negatives.
        """
        if self.colour == black:
            return (self.negative() @ other.negative()).negative()
        return ~(~self @ ~other)


class Box(monoidal.Box, Diagram):
    """
    An alpha box is a proposition, a box on the unit type of its region,
    :data:`white` by default.

    >>> assert Box('p').dom == Box('p').cod == Diagram.ob.id(white)
    """
    def __init__(self, name: str, dom=None, cod=None, **params):
        dom = self.ob.id(white) if dom is None else dom
        cod = dom if cod is None else cod
        super().__init__(name, dom, cod, **params)


class Cut(Box):
    """
    A cut is a bubble around a diagram of the other colour, its boundary the
    :meth:`Diagram.negative` of that of the diagram by default.

    >>> p = Box('p').negative()
    >>> assert Cut(p).colour == white and Cut(p).arg.colour == black
    >>> Cut(Box('p'), dom=Box('p').dom)
    Traceback (most recent call last):
    ...
    discopy.utils.AxiomError: A cut changes the colour of its region, \
got white inside white.
    """
    serialised_attrs = ("*args", ) + tuple(
        attr for attr in Box.serialised_attrs
        if attr not in ("*", "args", "method"))

    def __init__(self, arg: Diagram, dom=None, cod=None, name="", **params):
        negative = arg.negative_functor()
        dom = negative(arg.dom) if dom is None else dom
        cod = negative(arg.cod) if cod is None else cod
        if arg.colour in (dom.dom, cod.dom):
            raise AxiomError(
                "A cut changes the colour of its region, "
                f"got {arg.colour.name} inside {dom.dom.name}.")
        super().__init__(name, dom, cod, args=(arg, ), **params)

    def __repr__(self):
        kwargs = "" if self.is_id_on_objects else (
            f", dom={self.dom!r}, cod={self.cod!r}")
        kwargs += f", name={self.name!r}" if self.name else ""
        kwargs += "" if self.data is None else f", data={self.data!r}"
        kwargs += ", is_dagger=True" if self.is_dagger else ""
        return f"{factory_name(type(self))}({self.arg!r}{kwargs})"

    @classmethod
    def from_tree(cls, tree: dict) -> Cut:
        """
        Decode a serialised cut from its arg and boundary.

        >>> from discopy.utils import dumps, loads
        >>> assert loads(dumps(~Box('p'))) == ~Box('p')
        """
        kwargs = {
            k: tree[k] for k in ("name", "data", "is_dagger") if k in tree}
        arg, dom, cod = map(
            from_tree, (tree['args'][0], tree['dom'], tree['cod']))
        return cls(arg, dom=dom, cod=cod, **kwargs)


class Sum(monoidal.Sum, Box):
    """
    A formal sum of alpha diagrams with the same boundary, which a model
    reads as the union of their relations in the white region. Like every
    structure of the black region, a black sum is the white one conjugated
    by ¬, i.e. the intersection, so that the negative of a sum is still
    its complement.

    >>> p, q = Box('p'), Box('q')
    >>> M = Model(ar={'p': True, 'q': False})
    >>> assert isinstance(p + q, Sum) and M(p + q) == M(p) + M(q)
    >>> assert M((p + q).negative()) == M(~(p + q))
    >>> assert not M((p + q).negative()).array.any()
    """


class Functor(monoidal.Functor):
    """
    An alpha functor is a monoidal functor that sends a :class:`Cut` to the
    cut of its image, or to its elementwise negation in a category of
    matrices, e.g. :class:`discopy.tensor.Tensor` of ``bool``.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Cut) and issubclass(self.cod, Matrix):
            result = self(other.arg).map(operator.not_, dtype=bool)
            for _ in range(abs(getattr(other, "z", 0))):
                result = result.r if other.z > 0 else result.l
            return result
        return super().__call__(other)


class Model:
    """
    A model sends each wire to a finite set, by its name, and each box to a
    relation, its truth values as a Boolean array: a box and its
    :meth:`Diagram.negative` share their name, and so their relation.

    The image of a diagram is that of the functor :meth:`functor` gives,
    which forgets the colours and negates each cut, negated once more when
    the diagram is in the black region: a black box ``R`` is ¬R.

    Parameters:
        ob : The size of the set of each wire, by name.
        ar : The truth values of each box, by name.

    >>> p = Box('p')
    >>> M = Model(ar={'p': True})
    >>> assert M(p).array.all() and not M(p.negative()).array.any()
    >>> from discopy import peirce
    >>> assert eval(repr(M)) == M
    """
    def __init__(self, ob: dict | None = None, ar: dict | None = None):
        self.ob, self.ar = dict(ob or {}), dict(ar or {})

    def __eq__(self, other):
        """
        Two models are equal when they send each name to the same size and
        the same truth values, in whatever order they were given.

        >>> assert Model(ar={'p': 1, 'q': 0}) == Model(ar={'q': 0, 'p': 1})
        >>> assert Model(ar={'p': [1]}) == Model(ar={'p': 1})
        """
        def values(mapping):
            return sorted(
                (key, numpy.asarray(value).reshape(-1).tolist())
                for key, value in mapping.items())
        return type(self) is type(other) and all(
            values(getattr(self, attr)) == values(getattr(other, attr))
            for attr in ("ob", "ar"))

    __hash__ = None

    def __repr__(self):
        return f"{factory_name(type(self))}(ob={self.ob!r}, ar={self.ar!r})"

    diagram: type[Diagram]
    """ The category of diagrams the model interprets, by default. """

    def functor(self, dom: type[Diagram] | None = None) -> monoidal.Functor:
        """
        The functor from a category of diagrams ``dom`` into
        :class:`discopy.tensor.Tensor` of ``bool`` this model gives.
        """
        dom = self.diagram if dom is None else dom

        def ob_map(typ):
            return Dim(*(self.ob[wire.name] for wire in typ.inside))

        def ar_map(box):
            return Tensor[bool](self.ar[box.name], *map(
                functor, (box.dom, box.cod)))

        functor = dom.functor_factory(
            ob_map, ar_map, dom=dom, cod=Tensor[bool],
            colour_map=lambda _: None)
        return functor

    def __call__(self, diagram: Diagram) -> Tensor:
        result = self.functor(
            diagram.factory)(diagram)  # ty: ignore[invalid-argument-type]
        return result if diagram.colour == white\
            else result.map(operator.not_, dtype=bool)


Diagram.box_factory = Box
Diagram.sum_factory = Sum
Diagram.functor_factory = Functor
Model.diagram = Diagram
Id = Diagram.id
