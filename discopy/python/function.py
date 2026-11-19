# -*- coding: utf-8 -*-

"""
The category of Python functions with sequential composition.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Function
    Composition
    EndoFunctor
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Self

from discopy import messages
from discopy.abc import Category
from discopy.testable import Extensional, values
from discopy.cat import Functor
from discopy.monoidal import List
from discopy.utils import (
    assert_iscomposable, assert_isinstance,
    tuplify, untuplify, classproperty, factory)


Ty = List[type]
""" Lists of Python types, i.e. the free monoid on ``type``. """

NO_TREE = messages.NO_TREE_CALLABLE.format("A function", "inside")


@factory
@dataclass(frozen=True, eq=False)
class Composition:
    """
    The callable inside a composition of functions: it calls its ``factors``
    in turn, each on the outputs of the one before, see
    :meth:`Function.then`.

    Parameters:
        factors : The functions composed, at least one.

    Example
    -------
    >>> succ = Function(lambda x: x + 1, (int, ), (int, ))
    >>> assert Composition((succ, succ, succ))(0) == 3
    >>> assert (succ >> succ >> succ).inside.factors == (succ, succ, succ)
    """
    factors: tuple[Function, ...]

    def __call__(self, *args):
        *before, last = self.factors
        for factor in before:
            args = tuplify(factor(*args), len(factor.cod))
        return last(*args)


@dataclass(unsafe_hash=True)
class Function(Category[Ty, "Function"]):
    """
    Python function with sequential composition.

    Parameters:
        inside : The callable Python object inside the function.
        dom : The domain of the function, i.e. its input type.
        cod : The codomain of the function, i.e. its output type.

    Note
    ----
    The objects are :class:`monoidal.List` ``[type]``, the free monoid on
    Python's ``type``: a type or a tuple of types is ``cast`` into one.

    .. admonition:: Summary

        .. autosummary::

            id
            then
    """
    inside: Callable
    dom: Ty
    cod: Ty

    ob = Ty
    type_checking = True

    @classmethod
    def equation_factory(cls, *terms: "Function") -> Extensional:
        """
        The equation of a law, compared by the graphs of its terms on the
        :meth:`arguments` of their domain, see :class:`testable.Extensional`.
        """
        return Extensional(*terms)

    @classmethod
    def arguments(cls, dom: Ty) -> tuple[tuple, ...]:
        """
        The arguments a function on ``dom`` is called with by an
        extensional equation: one value of one of its types.

        >>> assert Function.arguments(Ty(bool)) == ((False, ), (True, ))
        """
        return tuple((value, ) for typ in dom.inside for value in values(typ))

    def __init__(self, inside: Callable, dom: Ty | type | tuple[type, ...],
                 cod: Ty | type | tuple[type, ...]):
        self.inside, self.dom, self.cod = (
            inside, self.ob.cast(dom), self.ob.cast(cod))

    @classmethod
    def id(cls, dom: Ty | type | tuple[type, ...]) -> Self:
        """
        The identity function on a given list of types :code:`dom`.

        Parameters:
            dom : The list of types on which to take the identity.
        """
        return cls(lambda *xs: untuplify(xs), dom, dom)

    def then(self, *others: Function) -> Self:
        """
        The sequential composition of ``n`` functions, called with
        :code:`>>`.

        Parameters:
            others : The other functions to compose in sequence.

        Example
        -------
        >>> from discopy.python.multiplicative import Function
        >>> succ = Function(lambda x: x + 1, (int, ), (int, ))
        >>> assert succ.then(succ, succ)(0) == 3

        Note
        ----
        The ``inside`` of a composition is a :class:`Composition` of the
        functions it was made of, and composing it again extends them, so
        that calling a long composition, built one function at a time as a
        functor does, takes a loop rather than as many nested calls.

        >>> from functools import reduce
        >>> assert reduce(Function.then, 2000 * [succ])(0) == 2000
        """
        if not others:
            return self
        factors = (self, ) + others
        for factor, other in zip(factors, others):
            assert_isinstance(other, type(self))
            assert_iscomposable(factor, other)
        inside = Composition(tuple(
            part for factor in factors for part in (
                factor.inside.factors
                if isinstance(factor.inside, Composition) else (factor, ))))
        return type(self)(inside, self.dom, others[-1].cod)

    @classproperty
    @contextmanager
    def no_type_checking(cls):
        had = "type_checking" in vars(cls)
        tmp, cls.type_checking = cls.type_checking, False
        try:
            yield
        finally:
            if had:
                cls.type_checking = tmp
            else:
                del cls.type_checking

    def __call__(self, arg):
        if self.type_checking:
            assert_isinstance(arg, self.dom.inside)
        result = self.inside(arg)
        if self.type_checking:
            assert_isinstance(result, self.cod.inside)
        return result

    def to_tree(self) -> dict:
        """
        A function has no tree, see :func:`discopy.utils.dumps`.

        >>> from discopy.utils import dumps
        >>> dumps(Function(abs, int, int))
        Traceback (most recent call last):
        ...
        TypeError: A function has no tree: its inside is a Python callable, \
which JSON cannot hold.
        """
        raise TypeError(NO_TREE)


@factory
class EndoFunctor(Functor):
    """
    An endofunctor on the category of Python functions, i.e. a functorial
    mapping from types to types and from functions to functions.

    Parameters:
        ob_map : Mapping from lists of types to lists of types, e.g.
            sending ``X`` to some container type over ``X``.
        ar_map : Mapping from functions to functions, e.g. ``fmap``.

    Note
    ----
    Since :attr:`Function.ob` is :class:`~discopy.monoidal.List` ``[type]``,
    calling an :class:`EndoFunctor` on a type or a tuple of types casts it
    into a list for convenience, but the result is always a list of types,
    matching :attr:`Function.dom` and :attr:`Function.cod`.

    Example
    -------
    >>> Maybe = EndoFunctor(
    ...     lambda X: Ty(X.inside[0] | None),
    ...     lambda f: Function(
    ...         lambda x: None if x is None else f(x),
    ...         f.dom.inside[0] | None, f.cod.inside[0] | None))
    >>> Maybe(int)
    monoidal.List[type](int | None)
    >>> f = Function(lambda x: x + 1, int, int)
    >>> Maybe(f)(1), Maybe(f)(None)
    (2, None)
    >>> assert EndoFunctor.id()(int) == Ty(int)
    """
    ob = Ty
    dom = cod = Function

    def __call__(self, other):
        if isinstance(other, Function):
            return self.ar_map[other]
        return self.ob_map[Ty.cast(other)]  # ty: ignore[invalid-argument-type]
