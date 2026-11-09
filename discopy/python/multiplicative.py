# -*- coding: utf-8 -*-

"""
The category of Python functions with tuple as monoidal product.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Function

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        exp
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import accumulate
from typing import Self

from discopy import messages
from discopy.abc import ClosedCategory, TracedCategory
from discopy.utils import (
    AxiomError, assert_isinstance, assert_istraceable, tuplify, untuplify,
    factory)
from discopy.python import finset, function
from discopy.python.function import Ty


def exp(base: Ty, exponent: Ty) -> Ty:
    """
    The exponential of a list of Python types by another.

    Parameters:
        base : The base types.
        exponent : The exponent types.
    """
    base, exponent = map(Ty.cast, (base, exponent))
    return Ty(Callable[
        list(exponent.inside),  # ty: ignore[invalid-type-form]
        tuple[base.inside]])


@factory
class Function(function.Function, ClosedCategory, TracedCategory):
    """
    Python function with tuple as tensor.

    Parameters:
        inside : The callable Python object inside the function.
        dom : The domain of the function, i.e. its list of input types.
        cod : The codomain of the function, i.e. its list of output types.

    .. admonition:: Summary

        .. autosummary::

            tensor
            swap
            copy
            discard
            ev
            curry
            uncurry
            fix
            trace
    """

    def __call__(self, *xs):
        if self.type_checking:
            if len(xs) != len(self.dom):
                raise ValueError
            for (x, t) in zip(xs, self.dom.inside):
                callable(x) or assert_isinstance(x, t)
        ys = self.inside(*xs)
        if self.type_checking:
            if len(self.cod) != 1 and (
                    not isinstance(ys, tuple) or len(self.cod) != len(ys)):
                raise RuntimeError
            for (y, t) in zip(tuplify(ys, len(self.cod)), self.cod.inside):
                callable(y) or assert_isinstance(y, t)
        return ys

    def tensor(self, *others: Function) -> Self:
        """
        The parallel composition of ``n`` functions, called with :code:`@`.

        Parameters:
            others : The other functions to compose in parallel.

        Example
        -------
        >>> add = Function(lambda x, y: x + y, (int, int), (int, ))
        >>> assert add.tensor(add, add)(1, 2, 3, 4, 5, 6) == (3, 7, 11)
        """
        if not others:
            return self
        factors = (self, ) + others
        offsets = [0, *accumulate(len(factor.dom) for factor in factors)]

        def inside(*xs):
            return untuplify(sum((
                tuplify(
                    factor(*xs[offsets[i]:offsets[i + 1]]), len(factor.cod))
                for i, factor in enumerate(factors)), ()))
        dom = self.dom.tensor(*(other.dom for other in others))
        cod = self.cod.tensor(*(other.cod for other in others))
        return type(self)(inside, dom, cod)

    @classmethod
    def swap(cls, x: Ty, y: Ty) -> Self:
        """
        The function for swapping two lists of types :code:`x` and :code:`y`.

        Parameters:
            x : The list of types on the left.
            y : The list of types on the right.
        """
        x, y = map(cls.ob.cast, (x, y))

        def inside(*xs):
            return untuplify(xs[len(x):] + xs[:len(x)])
        return cls(inside, dom=x @ y, cod=y @ x)

    @classmethod
    def permutation(cls, xs, doms) -> Self:
        """ Permute blocks of arguments. """
        doms = list(map(cls.ob.cast, doms))
        xs = finset.Permutation(xs, len(doms))
        offsets = [0, *accumulate(map(len, doms))]

        def inside(*args):
            blocks = [args[offsets[i]:offsets[i + 1]]
                      for i in range(len(doms))]
            return untuplify(sum((blocks[i] for i in xs), ()))

        dom = cls.ob().tensor(*doms)
        cod = cls.ob().tensor(*(doms[i] for i in xs))
        return cls(inside, dom, cod)

    braid = swap

    @classmethod
    def copy(cls, x: Ty, n=2) -> Self:
        """
        The function for making :code:`n` copies of a list of types :code:`x`.

        Parameters:
            x : The list of types to copy.
            n : The number of copies.
        """
        x = cls.ob.cast(x)
        return cls(lambda *xs: untuplify(n * xs), dom=x, cod=x ** n)

    @classmethod
    def discard(cls, dom: Ty) -> Self:
        """
        The function discarding a list of types, i.e. making zero copies.

        Parameters:
            dom : The list of types to discard.
        """
        return cls.copy(dom, 0)

    @classmethod
    def ev(cls, base: Ty, exponent: Ty, left=True) -> Self:
        """
        The evaluation function,
        i.e. take a function and apply it to an argument.

        Parameters:
            base : The output type.
            exponent : The input type.
            left : Whether to take the function on the left or right.
        """
        base, exponent = map(cls.ob.cast, (base, exponent))
        if left:
            dom, cod = cls.exp(base, exponent) @ exponent, base
            return cls(lambda f, *xs: f(*xs), dom, cod)
        dom, cod = exponent @ cls.exp(base, exponent), base
        return cls(lambda *xs: xs[-1](*xs[:-1]), dom, cod)

    def curry(self, n=1, left=True) -> Self:
        """
        Currying, i.e. turn a binary function into a function-valued function.

        Parameters:
            n : The number of types to curry.
            left : Whether to curry on the left or right.

        Raises:
            ValueError : When ``n`` is negative or more than the domain has.
        """
        if n < 0 or n > len(self.dom):
            raise ValueError(messages.NOT_SPLITTABLE.format(n, self.dom))
        if left:
            dom = self.dom[:len(self.dom) - n]
            cod = self.exp(self.cod, self.dom[len(self.dom) - n:])
        else:
            dom, cod = self.dom[n:], self.exp(self.cod, self.dom[:n])
        return type(self)(dom=dom, cod=cod, inside=lambda *xs: lambda *ys:
                          self(*(xs + ys) if left else (ys + xs)))

    def uncurry(self, left=True) -> Self:
        """
        Uncurrying,
        i.e. turn a function-valued function into a binary function.

        Parameters:
            left : Whether to uncurry on the left or right.
        """
        traced = self.cod.inside[0].__args__
        base, exponent = map(Ty.cast, (traced[-1].__args__, traced[:-1]))
        return self @ exponent >> self.ev(base, exponent) if left\
            else exponent @ self >> self.ev(base, exponent, left=False)

    def fix(self, n=1) -> Self:
        """
        The parameterised fixed point of a function ``f : A @ X -> X`` over
        the last ``n`` types ``X`` of its domain, i.e. ``fix(f) : A -> X``
        sends ``a`` to the fixed point of ``x -> f(a, x)`` reached by
        iteration from the default instance ``t()`` of each traced type
        ``t``, e.g. ``0`` for ``int``, which stops when two iterates are
        equal.

        Parameters:
            n : The number of types to take the fixed point over.

        Example
        -------
        The golden ratio is the fixed point of ``x -> 1 + x / (1 + x)``:

        >>> phi = Function(lambda x: 1 + x / (1 + x), float, float).fix()
        >>> assert phi() == (1 + 5 ** .5) / 2

        Note
        ----
        A feedback with no fixed point reachable from the seed does not
        terminate, as the function it would compute has no value.
        """
        if n < 0 or n > len(self.dom):
            raise ValueError
        if n == 0:
            return self
        traced = self.dom[len(self.dom) - n:]
        if self.cod != traced:
            raise AxiomError(messages.NOT_TRACEABLE.format(traced, self.cod))

        def inside(*xs):
            ys = tuple(t() for t in traced.inside)
            while (zs := tuplify(self(*xs, *ys), n)) != ys:
                ys = zs
            return untuplify(ys)
        return type(self)(inside, self.dom[:len(self.dom) - n], self.cod)

    def trace(self, n=1, left=False):
        """
        The multiplicative trace of a function ``f : A @ X -> B @ X`` over
        the last ``n`` types ``X``, i.e. the function sending ``a`` to the
        ``B`` part of ``f(a, x)`` at the fixed point ``x`` of ``x -> f(a, x)``
        given by :meth:`fix`: the trace of a cartesian category with a
        fixed-point operator, see `nLab
        <https://ncatlab.org/nlab/show/traced+monoidal+category>`_.

        Parameters:
            n : The number of types to trace over.
            left : Whether to trace the first ``n`` types instead, see
                :meth:`discopy.abc.SymmetricCategory.left_trace`.

        Example
        -------
        >>> f = Function(lambda x, y: (x + y, 1), (int, int), (int, int))
        >>> assert f.trace()(5) == 6
        >>> g = Function(lambda y, x: (1, x + y), (int, int), (int, int))
        >>> assert g.trace(left=True)(5) == 6
        """
        if n == 0:
            return self
        if left:
            return self.left_trace(n)
        assert_istraceable(self, n)
        dom, cod, traced = self.dom[:-n], self.cod[:-n], self.dom[-n:]
        fixed = (self >> self.discard(cod) @ traced).fix(n)
        return self.copy(dom) >> dom @ fixed\
            >> self >> cod @ self.discard(traced)

    exp = over = under = staticmethod(lambda x, y: exp(x, y))
