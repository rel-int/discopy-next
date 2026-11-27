# -*- coding: utf-8 -*-

"""
The free closed Markov feedback category, i.e. closed Markov diagrams with
feedback, whose wires carry a time step, see :mod:`~discopy.feedback`.

An exponential and a product are wires of their own, delayed as a whole,
leaving their factors alone: the delay is not a closed functor.

>>> x, y = Ty('x'), Ty('y')
>>> assert (x << y).delay() != x.delay() << y.delay()
>>> assert (x * y).delay() != x.delay() * y.delay()

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Wire
    Ty
    Exp
    Product
    Unitype
    Diagram
    Box
    Timed
    Eval
    Coeval
    Curry
    Pack
    Unpack
    Function
    Swap
    Permutation
    Copy
    Discard
    Feedback
    Feedforward
    Trace
    Sum
    Functor
"""

from __future__ import annotations

from functools import cached_property
from typing import ClassVar

from discopy import closed, cmap, feedback, hypergraph, messages, monoidal
from discopy.abc import DaggerMonoidalCategory
from discopy.cat import factory
# pylint: disable-next=unused-import
from discopy.closed.markov import Equation  # noqa: F401
from discopy.drawing import Drawing
from discopy.feedback import markov
from discopy.utils import factory_name, from_tree


class Wire(feedback.Wire, closed.markov.Wire):
    """ A closed feedback object is a closed object with a time step. """


@factory
class Ty(closed.markov.Ty, feedback.Ty):
    """ A closed feedback type is a closed type and a feedback type. """
    generator_factory = Wire


class Exp(Wire, closed.markov.Exp):
    """ An exponential object with a time step, delayed as a whole. """
    ob = Ty

    @cached_property
    def name(self):  # pylint: disable=method-hidden  # deleted by __init__
        "The name of an exponential is its undelayed print, read when needed."
        return str(self.reset())

    def __str__(self):
        return closed.markov.Exp.__str__(self)\
            + feedback.str_delayed(self.time_step)

    @property
    def drawing_name(self) -> str:
        return closed.markov.Exp.drawing_name.fget(self)\
            + feedback.str_delayed(self.time_step)

    def __repr__(self):
        return factory_name(type(self))\
            + f"({self.base!r}, {self.exponent!r}{self.repr_time_step()})"


class Product(closed.markov.Product, Wire):
    """
    The product of a tuple of types with a time step, delayed as a whole.

    >>> X, Y = Ty("X"), Ty("Y")
    >>> assert eval(repr((X * Y).delay())) == (X * Y).delay()
    """
    ob = Ty

    @cached_property
    def name(self):  # pylint: disable=method-hidden  # deleted by __init__
        "The name of a product is its undelayed print, read when needed."
        return str(self.reset())

    def __eq__(self, other):
        return closed.markov.Product.__eq__(self, other)\
            and self.time_step == other.time_step

    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        return closed.markov.Product.__str__(self)\
            + feedback.str_delayed(self.time_step)

    def __repr__(self):
        args = ', '.join(map(repr, self.factors)) + self.repr_time_step()
        return factory_name(type(self)) + f"({args.removeprefix(', ')})"

    def to_tree(self):
        return feedback.Wire.to_tree(self) | {
            'factors': [typ.to_tree() for typ in self.factors]}

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, tree['factors']),
                   time_step=tree.get('time_step', 0))


class Unitype(closed.markov.Unitype, Exp):
    """
    The unitype with a time step: delayed, it is the delay of its own
    exponential, which is not its own.

    Parameters:
        time_step : The number of times the unitype has been delayed.

    Example
    -------
    >>> U = Ty(Unitype())
    >>> assert U == U >> U
    >>> assert U.delay() == (U >> U).delay() != U.delay() >> U.delay()
    """
    def __init__(self, time_step: int = 0):
        monoidal.Wire.__init__(self, "U")
        self.time_step = time_step
        self.base = self.exponent = Ty(Unitype() if time_step else self)

    def __eq__(self, other):
        return isinstance(other, Exp)\
            and self.time_step == other.time_step\
            and (isinstance(other, Unitype) or (
                other.base, other.exponent) == (self.base, self.exponent))

    def __hash__(self):
        return hash(Exp(
            self.base,  # ty: ignore[invalid-argument-type]
            self.exponent, time_step=self.time_step))

    def __str__(self):
        return "U" + feedback.str_delayed(self.time_step)

    @property
    def drawing_name(self) -> str:
        return str(self)

    def __repr__(self):
        return f"{factory_name(type(self))}("\
            f"{self.repr_time_step().removeprefix(', ')})"

    def to_tree(self):
        return {'factory': factory_name(type(self)),
                'time_step': self.time_step}

    @classmethod
    def from_tree(cls, tree):
        return cls(tree.get('time_step', 0))


@factory
class Diagram(closed.markov.Diagram, markov.Diagram):
    """
    A closed feedback diagram is a closed diagram and a Markov feedback
    diagram.
    """
    trace_factory: ClassVar[type["Trace"]]  # ty: ignore[invalid-type-form]
    ob = Ty

    def to_compact(self) -> Diagram:
        """
        Open the curry bubbles into coevaluation and feedback, which stays
        a :class:`Diagram` as a closed feedback category is traced: each curry
        becomes its argument followed by :class:`Coeval`, traced over the
        curried wires.

        Example
        -------
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box("f", x @ y, z)
        >>> assert f.curry().to_compact() == (
        ...     f >> Coeval(z << y, left=True)).trace()
        """
        def image(box):
            if box.time_step:
                return image(box.reset()).delay(box.time_step)
            if isinstance(box, Curry) and not box.n:
                return box.arg.to_compact()
            if isinstance(box, Curry):
                return (box.arg.to_compact() >> Coeval(
                    box.cod, left=box.left)).trace(
                        len(box.cod.exponent), left=not box.left)
            return box
        result = self.id(self.dom)
        for layer in self.inside:
            for box, offset in layer.boxes_and_offsets:
                cod = result.cod
                result >>= cod[:offset] @ image(box)\
                    @ cod[offset + len(box.dom):]
        return result

    def to_drawing(self):
        return monoidal.Diagram.to_drawing(self, functor_factory=Functor)


class Box(closed.markov.Box, markov.Box, Diagram):
    """ A closed feedback box is a closed and a Markov feedback box. """
    serialised_attrs = markov.Box.serialised_attrs


class Timed(Box):
    """
    A closed feedback box built from arguments rather than a name and a
    boundary, with a time step of its own: delayed, it keeps its arguments
    and delays its boundary, as the delay is not a closed functor. E.g. the
    delay of the evaluation of ``x << y`` evaluates ``(x << y).delay()`` at
    ``y.delay()`` into ``x.delay()``, where ``x.delay() << y.delay()`` is
    another exponential.

    Parameters:
        args : The arguments of the box at time step zero.
        time_step : The number of times the box has been delayed.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> ev = Eval(x << y, left=True).delay()
    >>> assert ev.dom == (x << y).delay() @ y.delay() and ev.cod == x.d
    >>> assert ev.x == x << y and ev == Eval(x << y, True, time_step=1)
    """
    def __init__(self, *args, time_step: int = 0, **params):
        super().__init__(*args, **params)
        if time_step:
            self._time_step = time_step
            monoidal.Box.__init__(self, self.name, self.dom.delay(time_step),
                                  self.cod.delay(time_step), args=self.args,
                                  method=self.method)

    time_step = property(lambda self: self._time_step)

    def delay(self, n_steps=1):
        return type(self)(
            *self.__getnewargs__(), time_step=self.time_step + n_steps)

    def dagger(self):
        return super().dagger().delay(self.time_step)

    def repr_time_step(self) -> str:
        """ The keyword of the time step, as :meth:`__repr__` reads it. """
        return f", time_step={self.time_step}" if self.time_step else ""

    def __repr__(self):
        return super().__repr__()[:-1] + self.repr_time_step() + ")"

    def __str__(self):
        return super().__str__() + feedback.str_delayed(self.time_step)

    def to_tree(self):
        tree = super().to_tree()
        if self.time_step:
            tree['time_step'] = self.time_step
        return tree

    @classmethod
    def from_tree(cls, tree):
        return super().from_tree(tree).delay(tree.get('time_step', 0))


class Eval(Timed, closed.markov.Eval):
    """
    The evaluation of an exponential type, with a time step.

    >>> x, y = Ty('x'), Ty('y')
    >>> print(Eval(x << y, left=False).delay())
    Eval(y >> x, left=False).d
    """
    serialised_attrs = closed.markov.Eval.serialised_attrs + ("time_step", )


class Coeval(Timed, closed.markov.Coeval):
    """ The coevaluation of an exponential type, with a time step. """
    serialised_attrs = closed.markov.Coeval.serialised_attrs + ("time_step", )


class Curry(Timed, closed.markov.Curry):
    """ The currying of a closed feedback diagram, with a time step. """
    serialised_attrs = closed.markov.Curry.serialised_attrs + ("time_step", )

    def to_drawing(self):
        """
        A delayed curry is drawn as the delay of its argument followed by the
        delay of the coevaluation, traced over the delayed wires it curries.
        """
        if not self.time_step:
            return super().to_drawing()
        if not self.n:
            return self.arg.delay(self.time_step).to_drawing()
        coeval = self.coeval_factory(self.reset().cod, left=self.left)
        return (self.arg >> coeval).delay(self.time_step).to_drawing()\
            .trace(self.n, left=not self.left)


class Pack(Timed, closed.markov.Pack):
    """
    The packing of a product type, with a time step.

    >>> X, Y = Ty("X"), Ty("Y")
    >>> assert Pack(X * Y).delay().dagger() == Unpack(X * Y, time_step=1)
    """
    def __init__(self, typ: Ty, time_step: int = 0):
        Timed.__init__(self, typ, time_step=time_step)

    def __getnewargs__(self):
        """ The product type at time step zero, which a delay reads. """
        return (getattr(self, self.side).delay(-self.time_step), )

    def dagger(self):
        return Unpack(*self.__getnewargs__(), time_step=self.time_step)

    def __repr__(self):
        product, = self.__getnewargs__()
        return factory_name(type(self))\
            + f"({product!r}{self.repr_time_step()})"

    __str__ = closed.markov.Box.__str__

    def to_tree(self):
        return closed.markov.Pack.to_tree(self) | (
            {'time_step': self.time_step} if self.time_step else {})

    @classmethod
    def from_tree(cls, tree):
        return cls(from_tree(tree[cls.side]), time_step=tree.get(
            'time_step', 0))


class Unpack(Timed, closed.markov.Unpack):
    """ The unpacking of a product type, with a time step. """
    __init__ = Pack.__init__

    def dagger(self):
        return Pack(*self.__getnewargs__(), time_step=self.time_step)

    __getnewargs__, __repr__, __str__ = (
        Pack.__getnewargs__, Pack.__repr__, Pack.__str__)
    to_tree, from_tree = Pack.to_tree, classmethod(Pack.from_tree.__func__)


class Function(closed.markov.Function, markov.Function, Box):
    """ A function in a closed feedback diagram. """


class Permutation(closed.markov.Permutation, markov.Permutation, Box):
    """ A permutation in a closed feedback diagram. """


Function.register(Permutation)


class Swap(Permutation, closed.markov.Swap, markov.Swap, Box):
    """ A swap in a closed feedback diagram. """


class Copy(closed.markov.Copy, markov.Copy, Function):
    """ A copy in a closed feedback diagram. """


class Discard(closed.markov.Discard, markov.Discard, Copy):
    """ A discard in a closed feedback diagram. """


class Sum(closed.markov.Sum, markov.Sum, Box):
    """ A closed feedback sum is a closed and a feedback sum. """


class Functor(closed.markov.Functor, markov.Functor):
    """
    A closed feedback functor is a closed and a feedback functor: a delayed
    wire or box goes through the delay of the codomain.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to
            :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if self.cod is not Drawing and isinstance(other, (Wire, Box))\
                and other.time_step:
            return feedback.Functor.__call__(self, other)
        return super().__call__(other)


Diagram.functor_factory = Functor
CMap = cmap.CMap[Diagram]
Hypergraph = hypergraph.Hypergraph[Diagram]
for law in (
        "dagger_involution", "dagger_contravariance", "dagger_monoidality"):
    setattr(Hypergraph, law, getattr(DaggerMonoidalCategory, law).failing(
        messages.CURRY_HAS_NO_DAGGER))
Diagram.copy_factory = Copy
Diagram.function_factory = Function
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Feedback, Feedforward, Trace = feedback.structural_boxes(Diagram, Box)
for structural in (Feedback, Feedforward):
    structural.is_linear = property(lambda self: self.arg.is_linear)
Diagram.box_factory = Box
Diagram.pack_factory = Pack
Diagram.unpack_factory = Unpack
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
Diagram.coeval_factory = Coeval
Diagram.discard_factory = Discard
Diagram.sum_factory = Sum
Ty.under_factory = Ty.over_factory = staticmethod(Exp)
Ty.product_factory = Product

Id = Diagram.id


Bubble = Box.from_args
