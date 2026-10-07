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

from collections.abc import Callable
from itertools import accumulate
from typing import Self

from discopy.abc import ClosedCategory
from discopy.utils import (
    assert_isinstance, tuplify, untuplify, factory, unbiased)
from discopy.python import finset, function
from discopy.python.function import Ty
from discopy.pattern import Count, Hom, Obj, Repeat, rule, Tensor, Var


def exp(base: Ty, exponent: Ty) -> Ty:
    """
    The exponential of a list of Python types by another.

    Parameters:
        base : The base types.
        exponent : The exponent types.
    """
    base, exponent = map(Ty.cast, (base, exponent))
    return Ty(
        Callable[list(exponent.inside),  # ty: ignore[invalid-type-form]
                 tuple[base.inside]])  # ty: ignore[invalid-type-form]


@factory
class Function(function.Function, ClosedCategory):
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
            for (y, t) in zip(tuplify(ys), self.cod.inside):
                callable(y) or assert_isinstance(y, t)
        return ys

    @unbiased
    def tensor(self, other: Function) -> Function:
        """
        The parallel composition of two functions, called with :code:`@`.

        Parameters:
            other : The other function to compose in sequence.
        """
        def inside(*xs):
            left, right = xs[:len(self.dom)], xs[len(self.dom):]
            return untuplify(tuplify(self(*left)) + tuplify(other(*right)))
        return Function(inside, self.dom @ other.dom, self.cod @ other.cod)

    @classmethod
    @rule
    def swap[X: Obj[Ty], Y: Obj[Ty]](
            cls, x: Var[Ty | type, X], y: Var[Ty | type, Y]
    ) -> Hom[Function, Tensor[X, Y], Tensor[Y, X]]:
        """
        The function for swapping two lists of types :code:`x` and :code:`y`.

        Parameters:
            x : The list of types on the left.
            y : The list of types on the right.
        """
        x, y = map(Ty.cast, (x, y))

        def inside(*xs):
            return untuplify(tuplify(xs)[len(x):] + tuplify(xs)[:len(x)])
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

    @classmethod
    @rule
    def copy[X: Obj[Ty], N: Count](
            cls, x: Var[Ty | type, X], n: Var[int, N] = 2
    ) -> Hom[Function, X, Repeat[X, N]]:
        """
        The function for making :code:`n` copies of a list of types :code:`x`.

        Parameters:
            x : The list of types to copy.
            n : The number of copies.
        """
        x = Ty.cast(x)
        return cls(lambda *xs: n * xs, dom=x, cod=x ** n)

    merge = classmethod(ClosedCategory.merge.__func__.inapplicable(
        "A Python function cannot merge copies."))

    @staticmethod
    def discard(dom: Ty) -> Function:
        """
        The function discarding a list of types, i.e. making zero copies.

        Parameters:
            dom : The list of types to discard.
        """
        return Function.copy(dom, 0)

    @staticmethod
    def ev(base: Ty, exponent: Ty, left: bool = True) -> Function:
        """
        The evaluation function, a method rather than a rule since the
        exponential of Python types is :meth:`exp` and not ``<<``, which
        no pattern states,
        i.e. take a function and apply it to an argument.

        Parameters:
            base : The output type.
            exponent : The input type.
            left : Whether to take the function on the left or right.
        """
        base, exponent = map(Ty.cast, (base, exponent))
        if left:
            dom, cod = Function.exp(base, exponent) @ exponent, base
            return Function(lambda f, *xs: f(*xs), dom, cod)
        dom, cod = exponent @ Function.exp(base, exponent), base
        return Function(lambda *xs: xs[-1](*xs[:-1]), dom, cod)

    def curry(self, context=None, base=None, exponent=None, left=True
              ) -> Function:
        """
        Currying, i.e. turn a binary function into a function-valued function,
        a method rather than a rule as :meth:`ev` is.

        Parameters:
            context : The domain of the curry.
            base : The base of the exponential, the codomain.
            exponent : The objects to curry, one wire by default.
            left : Whether to curry on the left or right.
        """
        context, _, exponent = self.curry_boundary(
            context, base, exponent, left)
        if not exponent:
            return self
        return Function(
            dom=context, cod=Function.exp(self.cod, exponent),
            inside=lambda *xs: lambda *ys: self(
                *(xs + ys) if left else (ys + xs)))

    def uncurry(self, base=None, exponent=None, left=True) -> Function:
        """
        Uncurrying, i.e. turn a function-valued function into a binary
        function, its base and exponent read off the callable type of its
        codomain where not given.

        Parameters:
            base : The base of the exponential.
            exponent : The exponent of the exponential.
            left : Whether to uncurry on the left or right.
        """
        traced = self.cod.inside[0].__args__
        base = Ty.cast(traced[-1].__args__) if base is None else base
        exponent = Ty.cast(traced[:-1]) if exponent is None else exponent
        if not exponent:
            return self
        return self @ exponent >> Function.ev(base, exponent) if left\
            else exponent @ self >> Function.ev(base, exponent, left=False)

    def fix(self, n=1) -> Function:
        """
        The parameterised fixed point of a function.

        Parameters:
            n : The number of types to take the fixed point over.
        """
        def inside(*xs, y=None):
            result = self.inside(*xs + (() if y is None else (y, )))
            return y if result == y else inside(*xs, y=result)
        return self if n == 0\
            else Function(inside, self.dom[:-1], self.cod).fix(n - 1)

    def trace(self, dom=None, cod=None, mem=None, left=False):
        """
        The multiplicative trace of a function.

        Parameters:
            dom : The domain of the trace.
            cod : The codomain of the trace.
            mem : The objects to trace over, one wire by default.
            left : Whether to trace the wires on the left or right.
        """
        dom, cod, traced = self.trace_boundary(dom, cod, mem, left)
        if not traced:
            return self
        if left:
            raise NotImplementedError
        fixed = (self >> self.discard(cod) @ traced).fix()
        return self.copy(dom) >> dom @ fixed\
            >> self >> cod @ self.discard(traced)

    exp = over = under = staticmethod(lambda x, y: exp(x, y))
