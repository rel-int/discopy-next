# -*- coding: utf-8 -*-

"""
The free symmetric dataflow category, i.e. symmetric feedback diagrams whose
wires carry streams with a head and a tail, see :mod:`~discopy.dataflow`.

Its classes are those of the base level: the next level, e.g.
:mod:`dataflow.markov <discopy.dataflow.markov>`, mixes each of them into the
class of the feedback level it extends.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Wire
    HeadOb
    TailOb
    Ty
    Diagram
    Box
    Permutation
    Swap
    Feedback
    Feedforward
    Trace
    Head
    Tail
    FollowedBy
    Sum
    Functor

Axioms
------
The head and tail satisfy the following equations:

>>> x = Ty('x')
>>> assert x.head.head == x.head
>>> assert x.head.tail == Ty()
>>> assert x.delay().head == Ty()
>>> assert x.delay().tail == x == x.tail

In the category of streams, :class:`FollowedBy` is just the identity.

Example
-------
>>> from discopy import stream
>>> x = Ty(Wire('x', is_constant=False))
>>> FollowedBy(x).draw(doctest="docs/_static/dataflow/followed-by.svg")

.. image:: /_static/dataflow/followed-by.svg
    :align: center

>>> F = Functor({x: stream.Ty.sequence('x')}, cod=stream.Stream)
>>> X, Xh, Xtd = map(F, (x, x.head, x.tail.delay()))
>>> for xh, xtd in [(Xh.now, Xtd.now),
...                 (Xh.later.now, Xtd.later.now),
...                 (Xh.later.later.now, Xtd.later.later.now)]:
...     print(f"({xh}, {xtd})")
(x0, Ty())
(Ty(), x1)
(Ty(), x2)
>>> eq_up_to_F = lambda f, g: F(f).unroll(2).now == F(g).unroll(2).now
>>> assert eq_up_to_F(FollowedBy(x), Id(x))
"""

from __future__ import annotations

from typing import ClassVar

from discopy import feedback, hypergraph, messages, stream
from discopy.abc import (
    DaggerCategory, DaggerDataflowCategory, DataflowCategory,
    SymmetricCategory)
from discopy.cat import factory, kappa_calculus
from discopy.feedback import symmetric
from discopy.symmetric import (  # noqa: F401  pylint: disable=unused-import
    Equation)
from discopy.testable import Equation as AbstractEquation, Obj, Var, axiom
from discopy.utils import (
    AxiomError, assert_isinstance, factory_name, from_tree)


class Wire(feedback.Wire):
    """
    A dataflow wire is a feedback wire that carries a stream: a :attr:`head`,
    a :attr:`tail` and whether it :attr:`is_constant`.

    Parameters:
        name : The name of the wire.
        time_step : The number of times the wire has been delayed.
        is_constant : Whether the wire is a constant stream.

    Example
    -------
    >>> from discopy import dataflow
    >>> x = Wire('x', is_constant=False)
    >>> assert eval(repr(x.delay(2))) == x.delay(2) != Wire('x').delay(2)
    """
    is_constant = True
    head_factory: ClassVar[type]
    tail_factory: ClassVar[type]

    def __init__(self, *args, is_constant: bool = True, **params):
        assert_isinstance(is_constant, bool)
        super().__init__(*args, **params)
        self.is_constant = is_constant

    def __setstate__(self, state):
        state.setdefault('is_constant', True)
        super().__setstate__(state)

    @classmethod
    def strategy(cls):
        """ Generate named wires, constant or not. """
        from hypothesis import strategies as st

        return st.tuples(super().strategy(), st.booleans()).map(
            lambda args: cls(args[0].name, is_constant=args[1]))

    def atoms(self):
        """
        The atom of a wire, its undelayed wire, constant or not, which its
        string does not show.

        >>> x = Wire('x', is_constant=False)
        >>> assert Ty(x.delay()).atoms() == [Ty(x)] != Ty('x').atoms()
        """
        return [Wire(self.name, is_constant=self.is_constant)]

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


class HeadOb(Wire):
    """
    The head of a dataflow wire, interpreted as the first element of a
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

    def atoms(self):
        """ The atoms of the head of a wire, those of the wire. """
        return self.arg.atoms()

    def delay(self, n_steps=1):
        return type(self)(self.arg, self.time_step + n_steps)

    def reset(self):
        return type(self)(self.arg)

    def dagger(self):
        """
        The dagger of the head of a wire, the head of its dagger.

        >>> x = Ty(Wire('x', is_constant=False))
        >>> assert x.head.delay().dagger() == x[::-1].head.delay()
        >>> assert x.tail.dagger() == x[::-1].tail
        """
        return type(self)(self.arg.dagger(), self.time_step)

    @property
    def head(self):
        return None if self.time_step else self

    @property
    def tail(self):
        return self.delay(-1) if self.time_step else None

    def to_tree(self) -> dict:
        """
        Serialise the head of a wire by its argument and its time step.

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


class TailOb(Wire):
    """
    The tail of a dataflow wire that is not constant, interpreted as the
    stream starting from the second time step.

    Parameters:
        arg : The wire to take the tail of.
        time_step : The number of times the tail has been delayed.

    Example
    -------
    >>> x = Wire('x', is_constant=False)
    >>> assert x.tail == TailOb(x)
    """
    part = "tail"

    def __init__(self, arg, time_step: int = 0):
        if isinstance(arg, HeadOb) or arg.is_constant or arg.time_step > 0:
            raise ValueError
        self.arg = arg
        super().__init__(
            f"{arg}.{self.part}", time_step=time_step, is_constant=False)

    delay, reset, dagger = HeadOb.delay, HeadOb.reset, HeadOb.dagger
    __repr__, to_tree, atoms = HeadOb.__repr__, HeadOb.to_tree, HeadOb.atoms
    from_tree = classmethod(HeadOb.from_tree.__func__)


Wire.head_factory, Wire.tail_factory = HeadOb, TailOb


@factory
class Ty(feedback.Ty):
    """ A dataflow type is a feedback type with a head and a tail. """
    generator_factory = Wire

    def atoms(self):
        """ The atoms of a type, one wire each, see :meth:`Wire.atoms`. """
        return [self.factory(atom)
                for wire in self.inside for atom in wire.atoms()]

    @property
    def head(self):
        """ The head of a type, see :class:`HeadOb`. """
        return self.factory(*(x.head for x in self.inside if x.head))

    @property
    def tail(self):
        """ The tail of a type, see :class:`TailOb`. """
        return self.factory(*(x.tail for x in self.inside if x.tail))


def up_to_streams(term):
    """
    A diagram up to its stream, i.e. the first steps of its
    :meth:`Diagram.to_stream` up to the order of its memory, see
    :func:`~discopy.stream.up_to_memory`, a type as itself and a tuple of
    them each: how the laws of :class:`~discopy.abc.DataflowCategory` that
    hold of streams but not of their syntax are compared.

    >>> x = Ty(Wire('x', is_constant=False))
    >>> assert up_to_streams(FollowedBy(x)) == up_to_streams(Id(x))
    >>> assert up_to_streams((x, FollowedBy(x))) == (x, up_to_streams(Id(x)))
    """
    if isinstance(term, tuple):
        return tuple(map(up_to_streams, term))
    return term if isinstance(term, Ty)\
        else stream.up_to_memory(term.to_stream())


@factory
class Diagram(symmetric.Diagram, DaggerDataflowCategory):
    """
    A symmetric dataflow diagram is a symmetric feedback diagram with a
    :attr:`head`, a :attr:`tail` and :meth:`followed_by`, each built by its
    factory.

    The laws of :class:`~discopy.abc.DataflowCategory` hold of their streams,
    see :func:`up_to_streams`, on the diagrams of the symmetric rules: in a
    symmetric category, the head of a feedback has no stream, as the memory
    its first step writes cannot be discarded. A constant type is its own
    tail, see :meth:`constant_tail`, and the head and the tail commute with
    the dagger on the nose.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty
    head_factory: ClassVar[type]
    tail_factory: ClassVar[type]
    followed_by_factory: ClassVar[type]
    stream_factory = stream.Stream

    (head_monoidality, tail_monoidality, head_idempotence, tail_of_head,
     head_of_delay, tail_of_delay, followed_by_naturality) = (
            law.modulo(up_to_streams).weaken(within=SymmetricCategory)
            for law in (
                DataflowCategory.head_monoidality,
                DataflowCategory.tail_monoidality,
                DataflowCategory.head_idempotence,
                DataflowCategory.tail_of_head,
                DataflowCategory.head_of_delay,
                DataflowCategory.tail_of_delay,
                DataflowCategory.followed_by_naturality))
    followed_by_delay = DataflowCategory.followed_by_delay.modulo(
        up_to_streams)
    followed_by_unitarity = DaggerDataflowCategory.followed_by_unitarity\
        .modulo(up_to_streams)

    @axiom
    def constant_tail[X: Obj[Ty]](
            cls, x: Var[Ty, X]) -> AbstractEquation[Ty]:
        """
        A constant type is its own tail, here the wires of a type that are
        constant and not delayed.
        """
        constant = x.factory(*(
            wire for wire in x.inside
            if wire.is_constant and not wire.time_step))
        return AbstractEquation(constant.tail, constant)

    def to_stream(self) -> stream.Stream:
        """
        The stream of a dataflow diagram, in the :attr:`stream_factory` of
        the free category on its boxes: a wire ``x`` goes to the sequence
        ``x0, x1, ...``, or to ``x`` at each step when it is constant, and a
        box ``f`` to the sequence ``f0, f1, ...``.

        >>> x = Ty(Wire('x', is_constant=False))
        >>> f = Box('f', x, x)
        >>> print(f.head.to_stream().now)
        f0
        >>> print(f.tail.to_stream().now, "|", f.tail.to_stream().later.now)
        f1 | f2
        """
        cod = self.stream_factory

        def ob(typ):
            wire, = typ.inside
            return cod.ob(cod.category.ob(wire.name)) if wire.is_constant\
                else cod.ob.sequence(wire.name)

        def ar(box):
            return cod.sequence(str(box), functor(box.dom), functor(box.cod))

        functor = self.functor_factory(ob, ar, dom=type(self), cod=cod)
        return functor(self)

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


class Box(symmetric.Box, Diagram):
    """
    A symmetric dataflow box is a symmetric feedback box in a dataflow
    diagram.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
        time_step (int) : The number of times the box has been delayed.
    """


class Permutation(symmetric.Permutation, Box):
    """ A permutation in a symmetric dataflow diagram. """


class Swap(Permutation, symmetric.Swap, Box):
    """ A swap in a symmetric dataflow diagram. """


class Sum(symmetric.Sum, Box):
    """ A sum of symmetric dataflow diagrams. """


class Head(Box):
    """
    The head of a dataflow diagram, interpreted as the first element
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
    The tail of a dataflow diagram, interpreted as the stream starting from
    the second time step with the identity on the empty type at the first.

    Parameters:
        arg : The diagram to take the tail of.
        time_step : The number of times the tail has been delayed.
    """
    part = "tail"


class FollowedBy(Box):
    """
    The isomorphism between `x.head @ x.tail.delay()` and `x`, the ``fby``
    of Lucid: in the category of streams, it is the identity.

    Parameters:
        arg : The type it is the followed-by of.
        is_dagger : Whether it goes from `x` to `x.head @ x.tail.delay()`.
        time_step : The number of times it has been delayed.
    """
    serialised_attrs = ("arg", "is_dagger", "time_step")
    arg = None  # a type, not the diagram Box.arg reads

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
        return type(self)(
            self.arg, self.is_dagger,
            self.time_step + n_steps)  # ty: ignore[unsupported-operator]

    def reset(self):
        return type(self)(self.arg, self.is_dagger)

    def dagger(self):
        return type(self)(self.arg, not self.is_dagger, self.time_step)

    def to_tree(self) -> dict:
        """
        Serialise by the type, the direction and the time step.

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


class Functor(symmetric.Functor):
    """
    A symmetric dataflow functor is a symmetric feedback functor that sends
    heads, tails and followed-bys to those of its codomain, e.g.
    :class:`~discopy.stream.Stream` or :class:`Hypergraph`, and raises on
    a codomain that has none, whatever its map: they are the structure of
    the category, not generators.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to
            :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain, :code:`Diagram` by default.

    Example
    -------
    >>> x, y, m = [Ty(Wire(n, is_constant=False)) for n in "xym"]
    >>> f = Box('f', x @ m.d, y @ m)
    >>> g = Box('g', y.d @ m.d.d, x.d @ m.d)
    >>> F = Functor({x: y.d, y: x.d, m: m.d}, {f: g})
    >>> assert F(x.head) == F(x).head and F(x.tail) == F(x).tail
    >>> assert F(FollowedBy(x)) == FollowedBy(F(x))
    >>> assert F(f.head) == F(f).head and F(f.tail) == F(f).tail

    >>> from discopy import python
    >>> G = Functor(lambda _: int, lambda _: None, cod=python.Function)
    >>> G(FollowedBy(x))
    Traceback (most recent call last):
    ...
    discopy.utils.AxiomError: python.multiplicative.Function has no head, ...
    """
    dom = cod = Diagram

    def __call__(self, other):
        if not isinstance(other, (HeadOb, TailOb, Head, FollowedBy)):
            return super().__call__(other)
        if not hasattr(self.cod, "followed_by"):
            raise AxiomError(messages.NO_DATAFLOW.format(
                factory_name(self.cod)))
        if other.time_step:
            result = self(other.reset())
            for _ in range(other.time_step):
                result = result.delay()
            return result
        if not isinstance(other, FollowedBy):
            return getattr(self(other.arg), other.part)
        result = self.cod.followed_by(self(other.arg))
        return result.dagger() if other.is_dagger\
            and issubclass(self.cod, DaggerCategory)\
            and self.cod.has_dagger() else result


class Hypergraph(hypergraph.Hypergraph):
    """
    A dataflow hypergraph is a hypergraph of dataflow diagrams with their
    structure, so that :meth:`~discopy.monoidal.Diagram.to_hypergraph` and
    :class:`Equation` map into a codomain that has it: the head and the tail
    of a hypergraph are those of the diagram it decodes to, as boxes, its
    :meth:`followed_by` is that box and its :meth:`delay` delays each box
    and each spider.

    Example
    -------
    >>> x = Ty(Wire('x', is_constant=False))
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> assert (f >> g).head.to_hypergraph()\\
    ...     == (f >> g).to_hypergraph().head == Head(f >> g).to_hypergraph()
    >>> assert FollowedBy(x).delay().to_hypergraph()\\
    ...     == Hypergraph.followed_by(x).delay()
    """
    category = Diagram

    @classmethod
    def checked_params(cls) -> tuple:
        """ The dataflow diagrams, of both levels. """
        # pylint: disable-next=import-outside-toplevel  # it imports this
        from discopy.dataflow import markov
        return Diagram, markov.Diagram

    @property
    def head(self):
        """ The hypergraph of the head of the diagram it decodes to. """
        return self.from_box(self.category.head_factory(self.to_diagram()))

    @property
    def tail(self):
        """ The hypergraph of the tail of the diagram it decodes to. """
        return self.from_box(self.category.tail_factory(self.to_diagram()))

    @classmethod
    def followed_by(cls, dom):
        """ The hypergraph of the followed-by of a type. """
        return cls.from_box(cls.category.followed_by_factory(dom))

    def delay(self, n_steps=1):
        """ The delay of each box and each spider of the hypergraph. """
        return type(self)(
            self.dom.delay(n_steps), self.cod.delay(n_steps),
            tuple(box.delay(n_steps) for box in self.boxes), self.wires,
            tuple(typ.delay(n_steps) for typ in self.spider_types),
            self.offsets, self.phases)


Feedback, Feedforward, Trace = feedback.structural_boxes(Diagram, Box)
Diagram.box_factory = Box
Diagram.functor_factory = Functor
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.sum_factory = Sum
Diagram.head_factory, Diagram.tail_factory = Head, Tail
Diagram.followed_by_factory = FollowedBy
Diagram.hypergraph_factory = Hypergraph
Id = Diagram.id

TermBase, Variable, Constant, Application, Kappa = kappa_calculus(Diagram)
Term = Variable | Constant | Application | Kappa
