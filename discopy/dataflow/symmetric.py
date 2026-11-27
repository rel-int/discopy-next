# -*- coding: utf-8 -*-

"""
The free symmetric dataflow category, i.e. symmetric feedback diagrams whose
wires carry streams with a head and a tail, see :mod:`discopy.dataflow`.

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

from discopy import dataflow, feedback, hypergraph
from discopy.cat import factory
from discopy.feedback import symmetric
from discopy.symmetric import (  # noqa: F401  pylint: disable=unused-import
    Equation)


class Wire(dataflow.StreamWire, feedback.Wire):
    """
    A dataflow wire is a feedback wire that carries a stream.

    Parameters:
        name : The name of the wire.
        time_step : The number of times the wire has been delayed.
        is_constant : Whether the wire is a constant stream.

    Example
    -------
    >>> from discopy.dataflow import symmetric
    >>> x = Wire('x', is_constant=False)
    >>> assert eval(repr(x.delay(2))) == x.delay(2) != Wire('x').delay(2)
    """


class HeadOb(dataflow.HeadOb, Wire):
    """ The head of a dataflow wire, see :class:`discopy.dataflow.HeadOb`. """


class TailOb(dataflow.TailOb, Wire):
    """
    The tail of a dataflow wire, see :class:`discopy.dataflow.TailOb`.

    Example
    -------
    >>> x = Wire('x', is_constant=False)
    >>> assert x.tail == TailOb(x)
    """


Wire.head_factory, Wire.tail_factory = HeadOb, TailOb


@factory
class Ty(dataflow.StreamTy, feedback.Ty):
    """ A dataflow type is a feedback type with a head and a tail. """
    generator_factory = Wire


@factory
class Diagram(dataflow.StreamDiagram, symmetric.Diagram):
    """
    A symmetric dataflow diagram is a symmetric feedback diagram with a head,
    a tail and a followed-by.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty


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


class Head(dataflow.Head, Box):
    """ The head of a dataflow diagram, see :class:`discopy.dataflow.Head`. """


class Tail(dataflow.Tail, Box):
    """ The tail of a dataflow diagram, see :class:`discopy.dataflow.Tail`. """


class FollowedBy(dataflow.FollowedBy, Box):
    """
    The followed-by of a dataflow type, see
    :class:`discopy.dataflow.FollowedBy`.
    """


class Functor(dataflow.Functor, symmetric.Functor):
    """
    A symmetric dataflow functor is a symmetric feedback functor that sends
    heads, tails and followed-bys to those of its codomain, see
    :class:`discopy.dataflow.Functor`.

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
    >>> G = Functor({x: int}, {}, cod=python.Function)
    >>> G(FollowedBy(x))
    Traceback (most recent call last):
    ...
    discopy.utils.AxiomError: python.multiplicative.Function has no head, ...
    """
    dom = cod = Diagram


Feedback, Feedforward, Trace = feedback.structural_boxes(Diagram, Box)
Diagram.box_factory = Box
Diagram.functor_factory = Functor
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.sum_factory = Sum
Diagram.head_factory, Diagram.tail_factory = Head, Tail
Diagram.followed_by_factory = FollowedBy
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id
