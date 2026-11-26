# -*- coding: utf-8 -*-

"""
Parametric maps over an actegory, i.e. parameters that act on the data.

A `right action <https://ncatlab.org/nlab/show/actegory>`_ of a monoidal
category ``M`` on a category ``C`` sends an object ``X`` of ``C`` and an object
``P`` of ``M`` to an object ``X ◁ P`` of ``C``, and an arrow ``f`` of ``C`` and
an arrow ``r`` of ``M`` to an arrow ``f ◁ r`` of ``C``, functorially in both
arguments, strictly unital and multiplicative:

.. math::
    X ◁ I = X \\qquad\\qquad (X ◁ P) ◁ Q = X ◁ (P \\otimes Q)

The parametric maps of the action, as in :mod:`discopy.para`, are the arrows
``X ◁ P -> Y`` of ``C`` with ``P`` in ``M`` and parameters after the data. They
compose by tensoring the parameters, i.e.
``(P, f) >> (Q, g) = (P @ Q, f ◁ Q >> g)``,
and a reparametrisation is an arrow of ``M``, not of ``C``, acting on the
parameters of a map ``(P, f)`` by precomposition with ``X ◁ r``.

Weights tie by copying, which only deterministic maps can do: a random weight
sampled once and copied is not one sampled twice. So functions act on channels.
The tensor of parametric maps is left out, since it needs a monoidal structure
on ``C`` compatible with the action. For the self-action ``X ◁ P = X @ P`` of a
symmetric category, :class:`Para` is :mod:`discopy.para` with empty
coparameters. The action is a field of :class:`Para`, a value and not a class,
since the same ``M`` and ``C`` carry many actions.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Action
    Para

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        dirac

Example
-------
The action of a monoidal category on itself, ``X ◁ P = X @ P``, gives back the
parametric maps of :class:`discopy.para.Markov`, with the same composition:

>>> from discopy import para
>>> from discopy.markov import Ty, Box, Diagram
>>> x, y, p, q = map(Ty, "xypq")
>>> act = Action.right(Diagram)
>>> f = Para(x, y, Box('f', x @ p, y), p, act)
>>> g = Para(y, x, Box('g', y @ q, x), q, act)
>>> assert (f >> g).param == p @ q
>>> Markov = para.Markov[Diagram]
>>> assert Markov.equation_factory(
...     Markov(x, x, (f >> g).inside, p @ q),
...     Markov(x, y, f.inside, p) >> Markov(y, x, g.inside, q))

Two noisy layers read the one weight, the copy being the diagonal of ``M``:

>>> from discopy.kleisli.monad import Subdistribution
>>> from discopy.kleisli.multiplicative import Channel
>>> from discopy.python.multiplicative import Function, Ty
>>> Stochastic = Channel[Subdistribution]
>>> act = Action.deterministic(Stochastic)
>>> gate = Stochastic(
...     lambda x, w: frozenset({(x and not w, .9), (not (x and not w), .1)}),
...     (bool, bool), bool)
>>> assert act.unitality(gate) and act.identities(Ty(bool), Ty(bool))
>>> layer = Para(Ty(bool), Ty(bool), gate, Ty(bool), act)
>>> tied = (layer >> layer).reparam(Function.copy(Ty(bool)))
>>> assert tied.param == Ty(bool)
>>> sorted((x, round(p, 2)) for x, p in tied.inside(True, False))
[(False, 0.18), (True, 0.82)]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from operator import matmul
from typing import Any, Self

from discopy import messages
from discopy.kleisli.multiplicative import pack, pack_value
from discopy.python.multiplicative import Function
from discopy.testable import Equation, quotient
from discopy.utils import (
    AxiomError, assert_iscomposable, assert_isinstance, factory_name,
    tuplify)


def dirac(f, r: Function):
    """
    The action ``f ◁ r = f @ dirac(r)`` of a deterministic function ``r`` on a
    channel ``f``: the channel that runs ``f`` and applies ``r``, wrapped with
    the unit of the monad, to the parameters.

    This is lawful because ``r`` is a deterministic, i.e. pure, function: it
    is a map of the cartesian category of functions, which embeds in the
    Kleisli category by the unit of the monad, see Theorem G.10 of
    `arXiv:2402.15332 <https://arxiv.org/abs/2402.15332>`_.

    Parameters:
        f : A channel of :mod:`discopy.kleisli.multiplicative`.
        r : A function of :mod:`discopy.python.multiplicative`.

    Example
    -------
    >>> from discopy.kleisli.monad import Maybe
    >>> from discopy.kleisli.multiplicative import Channel
    >>> half = Channel[Maybe](lambda x: x // 2, int, int)
    >>> succ = Function(lambda x: x + 1, int, int)
    >>> assert dirac(half, succ)(4, 1) == (2, 2)
    """
    channel = type(f)
    unit = channel.monad.unit(pack(r.cod))
    point = channel(
        lambda *xs: unit(pack_value(tuplify(r(*xs), len(r.cod)))),
        r.dom, r.cod)
    return f @ point


@dataclass(frozen=True)
class Action:
    """
    A strict right action ``X ◁ P`` of the arrows of a class ``parameters``
    on the arrows of a class ``data``, given by its action on objects and on
    arrows, together with the laws it satisfies.

    Parameters:
        parameters : The class of arrows of ``M``, with ``ob``, ``id``, ``@``.
        data : The class of arrows of ``C``.
        ob : The action on objects, ``(X, P) -> X ◁ P``.
        ar : The action on arrows, ``(f, r) -> f ◁ r``.

    Example
    -------
    >>> from discopy.monoidal import Ty, Box, Diagram
    >>> x, p, q = map(Ty, "xpq")
    >>> f, r = Box('f', x, x), Box('r', q, p)
    >>> act = Action.right(Diagram)
    >>> assert act.ar(f, r) == f @ r and act.ob(x, p) == x @ p
    >>> assert act.unitality(f) and act.multiplicativity(f, r, r)
    """
    parameters: type
    data: type
    ob: Callable
    ar: Callable

    @classmethod
    def right(cls, category: type) -> Self:
        """
        The action of a monoidal category on itself, ``X ◁ P = X @ P`` and
        ``f ◁ r = f @ r``.

        Parameters:
            category : The monoidal category.
        """
        return cls(category, category, matmul, matmul)

    @classmethod
    def deterministic(cls, channel: type) -> Self:
        """
        The action of Python functions on the channels of a monad, i.e. of
        deterministic parameters on stochastic data, with
        ``X ◁ P = X @ P`` and ``f ◁ r = f @ dirac(r)``, see :func:`dirac`.

        Parameters:
            channel : A subscripted channel of
                :mod:`discopy.kleisli.multiplicative`, e.g. ``Channel[Maybe]``.
        """
        return cls(Function, channel, matmul, dirac)

    def __repr__(self):
        """
        The named constructor of the action, so that
        ``eval(repr(action)) == action`` for ``right`` and ``deterministic``.
        An action built by hand has the names of its callables, which a
        lambda does not give back.
        """
        callables = (self.ob, self.ar)
        name = factory_name(self.data)
        if callables == (matmul, matmul) and self.parameters is self.data:
            return f"Action.right({name})"
        if callables == (matmul, dirac) and self.parameters is Function:
            return f"Action.deterministic({name})"
        return f"Action({factory_name(self.parameters)}, {name}, " + ", ".join(
            f.__name__ for f in callables) + ")"

    @property
    def empty(self):
        """ The unit ``I`` of the monoidal category of parameters. """
        return self.parameters.ob()

    def unitality(self, f) -> Equation:
        """
        The unit law ``f ◁ id_I = f``.

        Parameters:
            f : An arrow of the data.
        """
        return self.data.equation_factory(
            self.ar(f, self.parameters.id(self.empty)), f)

    def multiplicativity(self, f, r, s) -> Equation:
        """
        The multiplication law ``(f ◁ r) ◁ s = f ◁ (r @ s)``.

        Parameters:
            f : An arrow of the data.
            r : An arrow of the parameters.
            s : Another arrow of the parameters.
        """
        return self.data.equation_factory(
            self.ar(self.ar(f, r), s), self.ar(f, r @ s))

    def functoriality(self, f, g, r, s) -> Equation:
        """
        Functoriality ``(f >> g) ◁ (r >> s) = (f ◁ r) >> (g ◁ s)``.

        Parameters:
            f : An arrow of the data.
            g : An arrow of the data, composable with ``f``.
            r : An arrow of the parameters.
            s : An arrow of the parameters, composable with ``r``.
        """
        return self.data.equation_factory(
            self.ar(f >> g, r >> s), self.ar(f, r) >> self.ar(g, s))

    def identities(self, x, p) -> Equation:
        """
        Functoriality on identities, ``id_X ◁ id_P = id_(X ◁ P)``.

        Parameters:
            x : An object of the data.
            p : An object of the parameters.
        """
        return self.data.equation_factory(
            self.ar(self.data.id(x), self.parameters.id(p)),
            self.data.id(self.ob(x, p)))


@dataclass(frozen=True)
class Para:
    """
    A parametric map over an :class:`Action`, i.e. an arrow
    ``inside : dom ◁ param -> cod`` of the data, with ``param`` an object of
    the parameters.

    Parameters:
        dom : The domain, an object of the data.
        cod : The codomain, an object of the data.
        inside : The arrow ``dom ◁ param -> cod``.
        param : The parameter space, an object of the parameters.
        action : The action of the parameters on the data.

    Example
    -------
    >>> from discopy.markov import Ty, Box, Diagram
    >>> x, p, q = map(Ty, "xpq")
    >>> act = Action.right(Diagram)
    >>> f = Para(x, x, Box('f', x @ p, x), p, act)
    >>> assert Para.id(act, x) >> f == f == f >> Para.id(act, x)
    >>> assert f.reparam(Box('r', q, p)).param == q
    """
    dom: object
    cod: object
    inside: Any
    param: Any
    action: Action

    def __post_init__(self):
        data, ob = self.action.data, self.action.ob
        assert_isinstance(self.param, self.action.parameters.ob)
        assert_isinstance(self.inside, data)
        assert_iscomposable(data.id(ob(self.dom, self.param)), self.inside)
        assert_iscomposable(self.inside, data.id(self.cod))

    @staticmethod
    def equation_factory(*terms: Para) -> Equation:
        """
        The equation of parametric maps, compared by their boundary and the
        parameters, and their inside up to the equation of the data.
        """
        return Equation(*terms, up_to=lambda para: (
            para.dom, para.cod, para.param,
            quotient(para.action.data, para.inside)))

    @classmethod
    def id(cls, action: Action, dom) -> Para:
        """
        The identity, with the empty parameter space.

        Parameters:
            action : The action.
            dom : The object of the data, also the codomain.
        """
        empty = action.empty
        return cls(dom, dom, action.data.id(action.ob(dom, empty)),
                   empty, action)

    def then(self, other: Para) -> Para:
        """
        The composite of ``(P, f)`` and ``(Q, g)`` is
        ``(P @ Q, (f ◁ Q) >> g)``.

        Parameters:
            other : The parametric map to compose with.
        """
        assert_isinstance(other, Para)
        if self.action != other.action:
            raise AxiomError(messages.DIFFERENT_ACTIONS.format(self, other))
        if self.cod != other.dom:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                self, other, self.cod, other.dom))
        action = self.action
        inside = action.ar(self.inside, action.parameters.id(other.param))\
            >> other.inside
        return Para(
            self.dom, other.cod, inside, self.param @ other.param, action)

    __rshift__ = then

    def reparam(self, other) -> Para:
        """
        Precompose the parameters with an arrow ``other : Q -> P`` of the
        *parameters*, i.e. the inside with ``dom ◁ other``.

        Parameters:
            other : The reparametrisation, an arrow of the parameters.

        Raises:
            TypeError : When ``other`` is not an arrow of the parameters.
        """
        action = self.action
        assert_isinstance(other, action.parameters)
        assert_iscomposable(other, action.parameters.id(self.param))
        inside = action.ar(action.data.id(self.dom), other) >> self.inside
        return Para(self.dom, self.cod, inside, other.dom, action)
