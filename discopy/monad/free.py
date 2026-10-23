# -*- coding: utf-8 -*-

"""
The free algebras of a monad and their homomorphisms.

Given a :class:`~discopy.kleisli.monad.Monad` ``M``, the free algebra on a
type ``X`` is ``M(X)`` with the multiplication
:math:`\\mu_X : M(M(X)) \\to M(X)` as structure, see
:meth:`discopy.monad.algebra.Algebra.free`. A homomorphism from the free
algebra on ``X`` to the free algebra on ``Y`` is a function
``h : M(X) -> M(Y)`` commuting with the multiplication

.. math::
    h \\circ \\mu_X = \\mu_Y \\circ M(h)

Homomorphisms compose as functions, with the identity on ``M(X)`` as
identity: the monad is needed to build them, not to compose them. Every
channel ``f : X -> M(Y)`` of the Kleisli category extends to its *bind*

.. math::
    f^* = M(f) \\mathbin{;} \\mu_Y \\;:\\; M(X) \\to M(Y)

and every homomorphism ``h`` restricts along the unit to the channel
:math:`\\eta_X \\mathbin{;} h`. The two are inverse to each other, so the
Kleisli category is isomorphic to the category of free algebras, the full
subcategory of the Eilenberg-Moore category on them, see
:mod:`discopy.monad.algebra`: Kleisli composition is bind followed by plain
composition.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Homomorphism
"""
from __future__ import annotations

from collections.abc import Callable

from discopy.abc import Category, NamedGeneric
from discopy.kleisli.channel import Channel
from discopy.kleisli.monad import Monad
from discopy.python.function import Function
from discopy.utils import (
    assert_iscomposable, assert_ismember, factory, factory_name)


@factory
class Homomorphism(Category, NamedGeneric['monad']):
    """
    A homomorphism between the free algebras of a monad ``M`` on two types
    ``X`` and ``Y``, i.e. a function from ``M(X)`` to ``M(Y)`` commuting
    with the multiplication.

    Parameters:
        inside : The underlying function, from ``M(X)`` to ``M(Y)``.
        dom : The type ``X`` of generators of the domain.
        cod : The type ``Y`` of generators of the codomain.

    Note
    ----
    That ``inside`` commutes with the multiplication is not checked: a
    function cannot be compared with another, but the law can be checked
    at a point through the inclusion into the Eilenberg-Moore category, see
    :meth:`discopy.monad.algebra.Homomorphism.preservation`.

    Example
    -------
    A channel of the powerset monad extends to sets of inputs:

    >>> from discopy.kleisli.monad import Powerset
    >>> neighbours = Channel[Powerset](
    ...     lambda x: frozenset({x - 1, x + 1}), int, int)
    >>> step = Homomorphism[Powerset].from_channel(neighbours)
    >>> sorted(step(frozenset({0, 10})))
    [-1, 1, 9, 11]

    Composition is the composition of functions, and it agrees with the
    composition of channels:

    >>> assert (step >> step)(frozenset({0}))\\
    ...     == (neighbours >> neighbours)(0) == frozenset({-2, 0, 2})
    >>> assert step.to_channel()(0) == neighbours(0)
    """
    ob = type
    monad: Monad

    def __init__(self, inside: Callable, dom: type, cod: type):
        monad = type(self).monad
        self.inside = inside if isinstance(inside, Function)\
            else Function(inside, monad(dom), monad(cod))
        self.dom, self.cod = dom, cod

    @classmethod
    def id(cls, dom: type) -> Homomorphism:
        """
        The identity homomorphism on the free algebra on ``dom``, i.e. the
        identity function on ``M(dom)``.

        Parameters:
            dom : The type of generators.
        """
        return cls(Function.id(cls.monad(dom)), dom, dom)

    def then(self, other: Homomorphism) -> Homomorphism:
        """
        The composition of two homomorphisms, i.e. of their functions,
        called with :code:`>>`.

        Parameters:
            other : The other homomorphism to compose in sequence.
        """
        assert_ismember(other, type(self))
        assert_iscomposable(self, other)
        return type(self)(self.inside >> other.inside, self.dom, other.cod)

    @classmethod
    def from_channel(cls, channel: Channel) -> Homomorphism:
        """
        The homomorphism extending a channel ``f : X -> M(Y)``, i.e. its
        bind :math:`M(f) \\mathbin{;} \\mu_Y`.

        Parameters:
            channel : The channel to extend.
        """
        monad = cls.monad
        assert_ismember(channel, Channel[monad])
        return cls(
            monad(channel.inside) >> monad.mult(channel.cod),
            channel.dom, channel.cod)

    def to_channel(self) -> Channel:
        """
        The channel ``X -> M(Y)`` a homomorphism restricts to along the
        unit, inverse to :meth:`from_channel`.
        """
        monad = type(self).monad
        return Channel[monad](
            monad.unit(self.dom) >> self.inside, self.dom, self.cod)

    def __call__(self, m):
        return self.inside(m)

    def __repr__(self):
        return factory_name(type(self))\
            + f"({self.inside!r}, dom={self.dom!r}, cod={self.cod!r})"
