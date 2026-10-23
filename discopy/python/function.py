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
    EndoFunctor
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from contextlib import contextmanager

from discopy.abc import Category
from discopy.axioms import Extensional, values
from discopy.cat import Functor
from discopy.monoidal import List
from discopy.utils import (
    assert_iscomposable, assert_isinstance,
    tuplify, untuplify, classproperty, factory)


Ty = List[type]
""" Lists of Python types, i.e. the free monoid on ``type``. """


@factory
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
        :meth:`arguments` of their domain, see :class:`axioms.Extensional`.
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

    def __init__(self, inside: Callable, dom: Ty | type, cod: Ty | type):
        self.inside, self.dom, self.cod = (
            inside, self.ob.cast(dom), self.ob.cast(cod))

    @classmethod
    def id(cls, dom: type) -> Function:
        """
        The identity function on a given list of types :code:`dom`.

        Parameters:
            dom (type) : The list of types on which to take the identity.
        """
        return cls(lambda *xs: untuplify(xs), dom, dom)

    def then(self, *others: Function) -> Function:
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
        """
        if not others:
            return self
        factors = (self, ) + others
        for factor, other in zip(factors, others):
            assert_isinstance(other, type(self))
            assert_iscomposable(factor, other)
        *before, last = factors

        def inside(*args):
            for factor in before:
                args = tuplify(factor(*args), len(factor.cod))
            return last(*args)
        return type(self)(inside, self.dom, others[-1].cod)

    @classproperty
    @contextmanager
    def no_type_checking(cls):
        tmp, cls.type_checking = cls.type_checking, False
        try:
            yield
        finally:
            cls.type_checking = tmp

    def __call__(self, arg):
        if self.type_checking:
            assert_isinstance(arg, self.dom.inside)
        result = self.inside(arg)
        if self.type_checking:
            assert_isinstance(result, self.cod.inside)
        return result


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
