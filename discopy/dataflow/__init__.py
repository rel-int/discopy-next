# -*- coding: utf-8 -*-

"""
Dataflow diagrams, i.e. feedback diagrams whose wires carry streams with a
head and a tail, after :cite:t:`DiLavoreEtAl22`.

A wire ``x`` has a :attr:`StreamWire.head`, its first value, and a
:attr:`StreamWire.tail`, the values after it, which is ``x`` itself when ``x``
is constant. A diagram has a :attr:`StreamDiagram.head` and a
:attr:`StreamDiagram.tail`, and :class:`FollowedBy` is the isomorphism from
``x.head @ x.tail.delay()`` to ``x``, the ``fby`` of Lucid.

This package holds the structure as mixins, which a level of the hierarchy
takes as they are, each class of the level subclassing its mixin and the
class of the feedback level below: :mod:`~discopy.dataflow.symmetric` over
:mod:`discopy.feedback.symmetric`, since monoidal streams are defined over
symmetric categories, and :mod:`~discopy.dataflow.markov` over
:mod:`discopy.feedback.markov`. See :mod:`discopy.stream` for their
interpretation as streams.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    symmetric
    markov

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    StreamWire
    HeadOb
    TailOb
    StreamTy
    StreamDiagram
    Head
    Tail
    FollowedBy
    Functor

Axioms
------
The head and tail satisfy the following equations:

>>> from discopy.dataflow.symmetric import Ty
>>> x = Ty('x')
>>> assert x.head.head == x.head
>>> assert x.head.tail == Ty()
>>> assert x.delay().head == Ty()
>>> assert x.delay().tail == x == x.tail

In the category of streams, :class:`FollowedBy` is just the identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from discopy import feedback, messages
from discopy.abc import DaggerCategory
from discopy.utils import AxiomError, factory_name, from_tree


class StreamWire:
    """
    The mixin of a wire that carries a stream: a :attr:`head`, a
    :attr:`tail` and whether it :attr:`is_constant`, a keyword it takes on
    top of those of the wire it is mixed into.

    Parameters:
        is_constant : Whether the wire is a constant stream.
    """
    is_constant = True
    head_factory: ClassVar[type]
    tail_factory: ClassVar[type]

    def __init__(self, *args, is_constant: bool = True, **params):
        super().__init__(*args, **params)
        self.is_constant = is_constant

    def __setstate__(self, state):
        state.setdefault('is_constant', True)
        super().__setstate__(state)

    @property
    def head(self):
        """ The :attr:`head_factory` of the wire, or `None` if delayed. """
        return None if self.time_step else self.head_factory(self)

    @property
    def tail(self):
        """ The :attr:`tail_factory` of the wire, or itself if constant. """
        return self.delay(-1) if self.time_step > 0 else (
            self if self.is_constant else self.tail_factory(self))

    def dagger(self):
        result = super().dagger()
        result.is_constant = self.is_constant
        return result

    def __eq__(self, other):
        return super().__eq__(other)\
            and self.is_constant == getattr(other, "is_constant", True)

    def __hash__(self):
        return hash((super().__hash__(), self.is_constant))

    def repr_time_step(self) -> str:
        """ The keyword arguments of the time step and of the constancy. """
        return super().repr_time_step() + (
            "" if self.is_constant else ", is_constant=False")

    def to_tree(self):
        tree = super().to_tree()
        if not self.is_constant:
            tree['is_constant'] = False
        return tree

    @classmethod
    def from_tree(cls, tree):
        result = super().from_tree(tree)
        result.is_constant = tree.get('is_constant', True)
        return result


class HeadOb:
    """
    The mixin of the head of a wire, interpreted as the first element of a
    stream followed by the constant stream on the empty type.

    Note the wire `arg` cannot be itself a `HeadOb` or be delayed.

    Parameters:
        arg : The wire to take the head of.
        time_step : The number of times the head has been delayed.
    """
    part = "head"

    def __init__(self, arg, time_step: int = 0):
        if isinstance(arg, HeadOb) or arg.time_step:
            raise ValueError
        self.arg = arg
        super().__init__(
            f"{arg}.{self.part}", time_step=time_step, is_constant=False)

    def __repr__(self):
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return factory_name(type(self)) + f"({repr(self.arg)}{time_step})"

    def delay(self, n_steps=1):
        return type(self)(self.arg, self.time_step + n_steps)

    def reset(self):
        return type(self)(self.arg)

    @property
    def head(self):
        return None if self.time_step else self

    @property
    def tail(self):
        return self.delay(-1) if self.time_step else None

    def to_tree(self) -> dict:
        """
        Serialise the head of a wire by its argument and its time step.

        >>> from discopy.dataflow.symmetric import Ty
        >>> from discopy.utils import dumps, loads
        >>> x = Ty('x')
        >>> assert loads(dumps(x.head.delay())) == x.head.delay()
        """
        tree = {'factory': factory_name(type(self)), 'arg': self.arg.to_tree()}
        if self.time_step:
            tree['time_step'] = self.time_step
        return tree

    @classmethod
    def from_tree(cls, tree: dict):
        return cls(from_tree(tree['arg']), tree.get('time_step', 0))


class TailOb:
    """
    The mixin of the tail of a wire that is not constant, interpreted as the
    stream starting from the second time step.

    Parameters:
        arg : The wire to take the tail of.
        time_step : The number of times the tail has been delayed.
    """
    part = "tail"

    def __init__(self, arg, time_step: int = 0):
        if isinstance(arg, HeadOb) or arg.is_constant or arg.time_step > 0:
            raise ValueError
        self.arg = arg
        super().__init__(
            f"{arg}.{self.part}", time_step=time_step, is_constant=False)

    delay, reset, __repr__ = HeadOb.delay, HeadOb.reset, HeadOb.__repr__
    to_tree, from_tree = HeadOb.to_tree, classmethod(HeadOb.from_tree.__func__)


class StreamTy:
    """ The mixin of a type whose wires carry streams. """
    @property
    def head(self):
        """ The head of a type, see :class:`HeadOb`. """
        return self.factory(*(x.head for x in self.inside if x.head))

    @property
    def tail(self):
        """ The tail of a type, see :class:`TailOb`. """
        return self.factory(*(x.tail for x in self.inside if x.tail))


class StreamDiagram:
    """
    The mixin of a diagram with a :attr:`head`, a :attr:`tail` and
    :meth:`followed_by`, each built by its factory.
    """
    head_factory: ClassVar[type]
    tail_factory: ClassVar[type]
    followed_by_factory: ClassVar[type]

    @property
    def head(self):
        """ Syntactic sugar for :class:`Head`. """
        return self.head_factory(self)

    @property
    def tail(self):
        """ Syntactic sugar for :class:`Tail`. """
        return self.tail_factory(self)

    @classmethod
    def followed_by(cls, dom):
        """ Syntactic sugar for :class:`FollowedBy`. """
        return cls.followed_by_factory(dom)


class Head:
    """
    The mixin of the head of a diagram, interpreted as the first element
    followed by the identity stream on the empty type.

    Parameters:
        arg : The diagram to take the head of.
        time_step : The number of times the head has been delayed.
    """
    serialised_attrs = ("*args", "time_step")
    part = "head"

    def __init__(self, arg, time_step=0):
        dom, cod = (
            getattr(x, self.part).delay(time_step) for x in [arg.dom, arg.cod])
        super().__init__(f"({arg}).{self.part}", dom, cod,
                         time_step=time_step, args=(arg, ))

    def dagger(self):
        """
        The dagger of a head is the head of the dagger, and the same for a
        tail: on streams both are functors, which commute with the dagger.

        >>> from discopy.dataflow.symmetric import Box, Ty, Wire
        >>> x, y = (Ty(Wire(name, is_constant=False)) for name in "xy")
        >>> f = Box('f', x, y)
        >>> assert f.head.dagger() == f.dagger().head
        >>> assert f.tail.delay().dagger() == f.dagger().tail.delay()
        """
        return type(self)(self.arg.dagger(), self.time_step)

    def __str__(self):
        return self.name + feedback.str_delayed(self.time_step)

    delay, reset, __repr__ = HeadOb.delay, HeadOb.reset, HeadOb.__repr__
    to_tree, from_tree = HeadOb.to_tree, classmethod(HeadOb.from_tree.__func__)


class Tail(Head):
    """
    The mixin of the tail of a diagram, interpreted as the stream starting
    from the second time step with the identity on the empty type at the
    first.

    Parameters:
        arg : The diagram to take the tail of.
        time_step : The number of times the tail has been delayed.
    """
    part = "tail"


class FollowedBy:
    """
    The mixin of the isomorphism between `x.head @ x.tail.delay()` and `x`,
    the ``fby`` of Lucid: in the category of streams, it is the identity.

    Parameters:
        arg : The type it is the followed-by of.
        is_dagger : Whether it goes from `x` to `x.head @ x.tail.delay()`.
        time_step : The number of times it has been delayed.
    """
    serialised_attrs = ("arg", "is_dagger", "time_step")
    arg = None  # shadows Box.arg, the type rather than a diagram

    def __init__(self, arg, is_dagger=False, time_step=0):
        self.arg = arg
        dagger_name = ", is_dagger=True" if is_dagger else ""
        name = f"FollowedBy({arg}{dagger_name})"
        dom, cod = arg.head @ arg.tail.delay(), arg
        dom, cod = (cod, dom) if is_dagger else (dom, cod)
        dom, cod = [x.delay(time_step) for x in (dom, cod)]
        super().__init__(
            name, dom, cod, time_step=time_step, is_dagger=is_dagger)

    def __repr__(self):
        is_dagger = ", is_dagger=True" if self.is_dagger else ""
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return factory_name(type(self))\
            + f"({repr(self.arg)}{is_dagger}{time_step})"

    def delay(self, n_steps=1):
        return type(self)(self.arg, self.is_dagger, self.time_step + n_steps)

    def reset(self):
        return type(self)(self.arg, self.is_dagger)

    def dagger(self):
        return type(self)(self.arg, not self.is_dagger, self.time_step)

    def to_tree(self) -> dict:
        """
        Serialise by the type, the direction and the time step.

        >>> from discopy.dataflow.symmetric import FollowedBy, Ty, Wire
        >>> from discopy.utils import dumps, loads
        >>> x = Ty(Wire('x', is_constant=False))
        >>> fby = FollowedBy(x, is_dagger=True).delay()
        >>> assert loads(dumps(fby)) == fby
        """
        tree = {'factory': factory_name(type(self)), 'arg': self.arg.to_tree()}
        if self.is_dagger:
            tree['is_dagger'] = True
        if self.time_step:
            tree['time_step'] = self.time_step
        return tree

    @classmethod
    def from_tree(cls, tree: dict):
        return cls(from_tree(tree['arg']), tree.get('is_dagger', False),
                   tree.get('time_step', 0))


class Functor:
    """
    The mixin of a functor that sends heads, tails and followed-bys to those
    of its codomain, e.g. :class:`discopy.stream.Stream`. Into a codomain
    that has none, they are boxes and wires like any other: a callable map,
    e.g. that of :meth:`discopy.symmetric.Diagram.to_hypergraph`, sends them
    as it sends a generator, and a mapping that does not have them raises
    rather than make them up.
    """
    def __call__(self, other):
        if not isinstance(other, (HeadOb, TailOb, Head, FollowedBy))\
                or other.time_step:
            return super().__call__(other)
        if hasattr(self.cod, "followed_by"):
            if not isinstance(other, FollowedBy):
                return getattr(self(other.arg), other.part)
            arg = other.dom if other.is_dagger else other.cod
            result = self.cod.followed_by(self(arg))
            return result.dagger() if other.is_dagger\
                and issubclass(self.cod, DaggerCategory)\
                and self.cod.has_dagger() else result
        mapping, key = (self.ob_map.mapping, self.dom.ob(other))\
            if isinstance(other, (HeadOb, TailOb))\
            else (self.ar_map.mapping, other)
        if isinstance(mapping, Mapping) and key not in mapping:
            raise AxiomError(messages.NO_DATAFLOW.format(
                factory_name(self.cod)))
        return super().__call__(other)


# The levels subclass these mixins, so they are imported last.
# pylint: disable-next=wrong-import-position
from discopy.dataflow import symmetric, markov  # noqa: E402,F401
