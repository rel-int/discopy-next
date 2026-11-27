# -*- coding: utf-8 -*-

"""
The free Markov dataflow category, i.e. Markov feedback diagrams whose wires
carry streams with a head and a tail, see :mod:`~discopy.dataflow`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Function
    Permutation
    Swap
    Copy
    Discard
    Feedback
    Feedforward
    Trace
    Head
    Tail
    FollowedBy
    Sum
    Functor

Random walk
-----------
A random walk is a feedback loop: start at ``0`` and then add a random step.

>>> x = Ty('x')
>>> zero = Box('0', Ty(), x.head)
>>> rand = Box('rand', Ty(), x)
>>> plus = Box('+', x @ x, x)
>>> walk = (rand.delay() @ x.delay() >> zero @ plus.delay()
...         >> FollowedBy(x) >> Copy(x)).feedback()
>>> walk.draw(doctest="docs/_static/dataflow/feedback-random-walk.svg")

.. image:: /_static/dataflow/feedback-random-walk.svg
    :align: center

Fibonacci
---------
We can define the Fibonacci sequence as a dataflow diagram, interpreted in
the category of streams of python types and functions.

>>> from discopy import python, stream
>>> from discopy.markov import Equation

>>> X = Ty('X')
>>> fby, wait = FollowedBy(X), Swap(X, X.d).feedback()
>>> zero, one = Box('zero', Ty(), X.head), Box('one', Ty(), X.head)
>>> copy, plus = Copy(X), Box('plus', X @ X, X)

>>> @Diagram.feedback
... @Diagram.from_callable(X.d, X @ X)
... def fib(x):
...     y = fby(zero(), plus.d(fby.d(one.d(), wait.d(x)), x))
...     return (y, y)

>>> fib_ = (copy.d >> one.d @ wait.d @ X.d
...                >> fby.d @ X.d
...                >> plus.d
...                >> zero @ X.d
...                >> fby >> copy).feedback()
>>> assert Equation(fib.arg, fib_.arg)
>>> fib_.draw(wire_labels=False, figsize=(5, 5),
...           doctest="docs/_static/dataflow/fibonacci-feedback.svg")

.. image:: /_static/dataflow/fibonacci-feedback.svg
    :align: center

>>> cod = stream.Stream[python.Function]
>>> F = Functor(
...     ob_map={X: int},
...     ar_map={zero: cod.singleton(python.Function(lambda: 0, (), int)),
...         one: cod.singleton(python.Function(lambda: 1, (), int)),
...         plus: lambda x, y: x + y}, cod=cod)
>>> assert F(fib).unroll(9).now()[:10] == (0, 1, 1, 2, 3, 5, 8, 13, 21, 34)

The same in the category of streams of python types and probabilistic
functions gives the random walk:

>>> from random import choice, seed; seed(420)
>>> rand = Box('rand', Ty(), X)
>>> F.ar_map[rand] = lambda: choice([-1, +1])

>>> @Diagram.feedback
... @Diagram.from_callable(X.d, X @ X)
... def walk(x):
...     x = plus.d(rand.d(), x)
...     x = fby(zero(), x)
...     return (x, x)

>>> walk.draw(wire_labels=False, figsize=(5, 5),
...           doctest="docs/_static/dataflow/random-walk-feedback.svg")

.. image:: /_static/dataflow/random-walk-feedback.svg
    :align: center

>>> assert F(walk).unroll(9).now()[:10] == (0, -1, 0, 1, 2, 1, 0, -1, 0, 1)
>>> assert F(walk).unroll(9).now()[:10] == (0, -1, -2, -1, 0, 1, 0, 1, 2, 1)
>>> assert F(walk).unroll(9).now()[:10] == (0, -1, 0, 1, 0, 1, 0, -1, 0, -1)
"""

from __future__ import annotations

from discopy import feedback, hypergraph
from discopy.cat import factory, kappa_calculus
from discopy.dataflow import symmetric
# pylint: disable-next=unused-import
from discopy.dataflow.symmetric import (  # noqa: F401
    Wire, HeadOb, TailOb, Ty)
from discopy.feedback import markov
from discopy.markov import (  # noqa: F401  pylint: disable=unused-import
    Equation)


@factory
class Diagram(markov.Diagram, symmetric.Diagram):
    """
    A Markov dataflow diagram is a Markov feedback diagram and a symmetric
    dataflow diagram.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty


class Box(markov.Box, symmetric.Box, Diagram):
    """
    A Markov dataflow box is a Markov feedback box and a symmetric dataflow
    box.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
        is_deterministic : Whether copying commutes with the box.
        is_affine : Whether discarding commutes with the box.
        time_step (int) : The number of times the box has been delayed.
    """


class Function(markov.Function, Box):
    """ A function in a Markov dataflow diagram. """


class Permutation(markov.Permutation, symmetric.Permutation, Box):
    """ A permutation in a Markov dataflow diagram. """


Function.register(Permutation)


class Swap(Permutation, markov.Swap, symmetric.Swap, Box):
    """ A swap in a Markov dataflow diagram. """


class Copy(markov.Copy, Function):
    """ A copy in a Markov dataflow diagram. """


class Discard(markov.Discard, Copy):
    """ A discard in a Markov dataflow diagram. """


class Sum(markov.Sum, symmetric.Sum, Box):
    """ A sum of Markov dataflow diagrams. """


class Head(symmetric.Head, Box):
    """ The head of a Markov dataflow diagram. """


class Tail(symmetric.Tail, Box):
    """ The tail of a Markov dataflow diagram. """


class FollowedBy(symmetric.FollowedBy, Box):
    """ The followed-by of a type in Markov dataflow diagrams. """


class Functor(markov.Functor, symmetric.Functor):
    """
    A Markov dataflow functor is a Markov feedback functor and a symmetric
    dataflow functor.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to
            :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain, :code:`Diagram` by default.
    """
    dom = cod = Diagram


Feedback, Feedforward, Trace = feedback.structural_boxes(Diagram, Box)
Diagram.box_factory = Box
Diagram.functor_factory = Functor
Diagram.copy_factory = Copy
Diagram.function_factory = Function
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.discard_factory = Discard
Diagram.sum_factory = Sum
Diagram.head_factory, Diagram.tail_factory = Head, Tail
Diagram.followed_by_factory = FollowedBy
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id

TermBase, Variable, Constant, Application, Kappa = kappa_calculus(Diagram)
Term = Variable | Constant | Application | Kappa
