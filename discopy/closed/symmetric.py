# -*- coding: utf-8 -*-

"""
The free symmetric closed category, i.e. a symmetric monoidal category with
exponentials, where left and right exponentials coincide:
``x << y == x ** y == y >> x``. It has no copy and no discard, see
:mod:`~discopy.closed.markov` for those.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Wire
    Ty
    Exp
    Diagram
    Box
    Eval
    Coeval
    Curry
    Swap
    Permutation
    Sum
    Functor
    Equation

Axioms
------
Currying on the left is currying on the right after a swap, which the free
category does not quotient by.

>>> x, y, z = map(Ty, "xyz")
>>> f = Box('f', x @ y, z)
>>> assert f.curry().cod == (Diagram.swap(y, x) >> f).curry(left=False).cod
>>> assert f.curry() != (Diagram.swap(y, x) >> f).curry(left=False)
"""

from __future__ import annotations

from functools import cached_property
from typing import ClassVar

from discopy import closed, cmap, hypergraph, messages, monoidal, symmetric
from discopy.abc import BalancedCategory, ClosedCategory
from discopy.cat import factory
from discopy.utils import factory_name


class Wire(closed.monoidal.Wire):
    """ A symmetric closed object is a biclosed object. """


@factory
class Ty(closed.monoidal.Ty):
    """
    A symmetric closed type is a biclosed type where left and right
    exponentials coincide.

    >>> x, y = Ty('x'), Ty('y')
    >>> assert x << y == x ** y == y >> x

    Its terms, those of the linear lambda calculus, are not implemented: a
    type applied to a name or a function raises, see
    :mod:`~discopy.closed.markov` for the terms of a closed Markov category.

    >>> x('a')
    Traceback (most recent call last):
    ...
    NotImplementedError: A symmetric closed type has no terms yet, see \
discopy.closed.markov.
    """
    generator_factory = Wire

    def __call__(self, arg):
        raise NotImplementedError(messages.NO_SYMMETRIC_CLOSED_TERMS)


class Exp(Wire, closed.monoidal.Exp):
    """
    An exponential object in a symmetric closed category, called with
    ``**``, ``<<`` or ``>>`` alike.
    """
    ob = Ty

    @cached_property
    def name(self):  # pylint: disable=method-hidden  # deleted by __init__
        "The name of an exponential is its print, read when needed."
        return Exp.__str__(self)

    def show(self, base, exponent) -> str:
        return f"({exponent} >> {base})"

    def __repr__(self):
        return factory_name(type(self))\
            + f"({self.base!r}, {self.exponent!r})"


@factory
class Diagram(
        closed.monoidal.Diagram, symmetric.Diagram, ClosedCategory,
        BalancedCategory):
    """
    A symmetric closed diagram is a biclosed diagram and a symmetric diagram.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    box_factory: ClassVar[type["Box"]]
    functor_factory: ClassVar[type["Functor"]]
    swap_factory: ClassVar[type["Swap"]]
    permutation_factory: ClassVar[type["Permutation"]]
    curry_factory: ClassVar[type["Curry"]]
    eval_factory: ClassVar[type["Eval"]]
    coeval_factory: ClassVar[type["Coeval"]]
    sum_factory: ClassVar[type["Sum"]]
    ob = Ty

    @classmethod
    def ev(cls, base: Ty, exponent: Ty, left: bool = True):
        if not exponent:
            return cls.id(base)
        return cls.eval_factory(exponent >> base, left=left)

    def to_drawing(self):
        return monoidal.Diagram.to_drawing(self, functor_factory=Functor)

    currying_symmetry = ClosedCategory.currying_symmetry.failing(
        messages.FREE_CURRY)


class Box(closed.monoidal.Box, symmetric.Box, Diagram):
    """
    A symmetric closed box is a biclosed and symmetric box.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
    """


class Eval(closed.monoidal.Eval, Box):
    """ The evaluation of an exponential type, on either side. """
    serialised_attrs = closed.monoidal.Eval.serialised_attrs + ("*", )
    drawing_name = "__call__"

    def __getnewargs__(self):
        """ The arguments of the constructor, which a pickle reads. """
        return self.x, self.left

    def __str__(self):
        """
        The name of the evaluation, with its side when it is on the right:
        an exponential of a closed category is on either side, so its type
        alone does not say which.

        >>> x, y = Ty('x'), Ty('y')
        >>> print(Eval(x << y, left=False))
        Eval(y >> x, left=False)
        """
        return self.name if self.left else self.name[:-1] + ", left=False)"


class Coeval(closed.monoidal.Coeval, Box):
    "The coevaluation of an exponential type, i.e. the dagger of an Eval."
    serialised_attrs = closed.monoidal.Coeval.serialised_attrs + ("*", )
    __getnewargs__ = Eval.__getnewargs__
    __str__ = Eval.__str__


class Curry(closed.monoidal.Curry, Box):
    """ The currying of a symmetric closed diagram. """


class Permutation(symmetric.Permutation, Box):
    """ A permutation in a symmetric closed diagram. """


class Swap(Permutation, symmetric.Swap, Box):
    """ The symmetric swap in a symmetric closed diagram. """


class Sum(closed.monoidal.Sum, symmetric.Sum, Box):
    """
    A symmetric closed sum is a biclosed sum and a symmetric sum.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(closed.monoidal.Functor, symmetric.Functor):
    """
    A symmetric closed functor is a biclosed functor and a symmetric functor.

    Parameters:
        ob_map (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram


CMap = cmap.CMap[Diagram]
Diagram.box_factory = Box
Diagram.functor_factory = Functor
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
Diagram.coeval_factory = Coeval
Diagram.sum_factory = Sum
Hypergraph = hypergraph.Hypergraph[Diagram]
Ty.under_factory = Ty.over_factory = staticmethod(Exp)
Id = Diagram.id


class Equation(symmetric.Equation):
    """ The :class:`symmetric.Equation` of symmetric closed diagrams. """


Bubble = Box.from_args
