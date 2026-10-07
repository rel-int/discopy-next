# -*- coding: utf-8 -*-

"""
The free compact category, i.e. diagrams with swaps, cups and caps.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Cup
    Cap
    Permutation
    Swap
    Sum
    Bubble
    Functor

Axioms
------

>>> x, y = Ty('x'), Ty('y')

Snake equations
===============

>>> snake = Equation(Id(x.l).transpose(left=True), Id(x), Id(x.r).transpose())
>>> assert snake
>>> snake.draw(doctest="docs/_static/compact/snake.svg")

.. image:: /_static/compact/snake.svg
    :align: center

Yanking
=======
a.k.a. Reidemeister move 1

>>> cap_yanking = Equation(Cap(x, x.r) >> Swap(x, x.r), Cap(x.r, x))
>>> cup_yanking = Equation(Swap(x, x.r) >> Cup(x.r, x), Cup(x, x.r))
>>> assert cap_yanking and cup_yanking
>>> Equation(cap_yanking, cup_yanking, symbol='', space=1).draw(
...     doctest="docs/_static/compact/yanking_cup_and_cap.svg")

.. image:: /_static/compact/yanking_cup_and_cap.svg
    :align: center

Coherence
=========

>>> assert Equation(Diagram.caps(x @ y, y.r @ x.r),
...     Cap(x, x.r) @ Cap(y, y.r) >> x @ Diagram.swap(x.r, y @ y.r))
"""

from typing import Any, ClassVar

from discopy import symmetric, ribbon, rigid, cmap, hypergraph, monoidal
from discopy.abc import (
    BalancedCategory, BiclosedCategory, CompactCategory, PivotalCategory)
from discopy.axioms import Serialisable
from discopy.cat import factory, Generator
from discopy.pivotal import Wire, Ty  # noqa: F401  pylint: disable=unused-import
from discopy.utils import deprecated_alias


class Layer(symmetric.Layer, rigid.Layer):
    """ A compact layer with permutation plumbing and rigid rotation. """


@factory
class Diagram(symmetric.Diagram, ribbon.Diagram, CompactCategory):
    """
    A compact diagram is a symmetric diagram and a ribbon diagram.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (pivotal.Ty) : The domain of the diagram, i.e. its input.
        cod (pivotal.Ty) : The codomain of the diagram, i.e. its output.
    """
    serialisation = Serialisable.serialisation

    twist = classmethod(ribbon.Diagram.twist.__func__.admissible(
        "The twist is the identity."))

    ob = Ty
    Layer: ClassVar[Generator] = Generator.subclass(Layer)
    Permutation: ClassVar[Generator]
    Functor: ClassVar[Generator]

    pivotality = PivotalCategory.pivotality

    yanking = BalancedCategory.yanking

    foliation_idempotence = ribbon.Diagram.foliation_idempotence.failing(
        "The hypergraph decode follows the box order of its encoding, so "
        "the foliation of a composition written backwards through a snake "
        "keeps a cut that foliating it again straightens.")

    currying_left = BiclosedCategory.currying_left.weaken(
        boundary_connected=True)

    currying_right = BiclosedCategory.currying_right.weaken(
        boundary_connected=True)

    currying_eta_left = BiclosedCategory.currying_eta_left.weaken(
        boundary_connected=True)

    currying_eta_right = BiclosedCategory.currying_eta_right.weaken(
        boundary_connected=True)

    currying_naturality_left = BiclosedCategory\
        .currying_naturality_left.weaken(
        boundary_connected=True)

    currying_naturality_right = BiclosedCategory\
        .currying_naturality_right.weaken(
        boundary_connected=True)

    @classmethod
    def rewire(cls, draw, value: monoidal.Ty, dom: bool,
               other: monoidal.Ty | None) -> tuple[monoidal.Ty, Any]:
        """
        A compact category bends wires: on the domain of a goal, a cup
        may close a wire and its adjoint, and a cap may open a wire of the
        other side with its adjoint, then the wires are permuted; the
        codomain is rewired the other way round, a cap opening a pair of
        it and a cup closing a wire of the other side.
        """
        from hypothesis import strategies as st

        closing, opening = ("cups", "caps") if dom else ("caps", "cups")
        parts = value.atoms
        pairs = [
            (i, j) for i, x in enumerate(parts) for j, y in enumerate(parts)
            if i != j and y == (x.r if dom else x.l)]
        if closing in cls.generators and pairs and draw(st.booleans()):
            i, j = draw(st.sampled_from(pairs))
            xs = [k for k in range(len(parts)) if k not in (i, j)] + [i, j]
            rest = value[:0].tensor(*(parts[k] for k in xs[:-2]))
            if dom:
                plumbing = cls.permutation(xs, parts).then(
                    rest @ cls.cups(parts[i], parts[j]))
                return cls.plumb(plumbing, dom, cls.rewire(
                    draw, rest, dom, other))
            inverse = [xs.index(k) for k in range(len(xs))]
            plumbing = (rest @ cls.caps(parts[i], parts[j])).then(
                cls.permutation(inverse, [parts[k] for k in xs]))
            return cls.plumb(plumbing, dom, cls.rewire(
                draw, rest, dom, other))
        if opening in cls.generators and other and draw(st.booleans()):
            x = draw(st.sampled_from(other.atoms))
            plumbing = value @ cls.caps(x, x.l) if dom\
                else value @ cls.cups(x, x.r)
            bent = plumbing.cod if dom else plumbing.dom
            return cls.plumb(plumbing, dom, super().rewire(
                draw, bent, dom, None))
        return super().rewire(draw, value, dom, other)


Box, Cup, Cap = (
    Diagram.Box, Diagram.Cup, Diagram.Cap)


@Diagram.generator
class Permutation(symmetric.Permutation, Box):
    """
    A compact permutation is a symmetric permutation in a compact category.

    Parameters:
        dom (pivotal.Ty) : The domain, i.e. the wires to permute.
        perm : The permutation as a :class:`finset.Permutation` or a list.
    """
    def rotate(self, left=False):
        dom = self.cod.l if left else self.cod.r
        return type(self)(dom, self.perm.rotate())

    l = property(lambda self: self.rotate(left=True))
    r = property(lambda self: self.rotate(left=False))


Swap, Sum, Bubble, Eval, Coeval, Curry = (
    Diagram.Swap, Diagram.Sum, Diagram.Bubble,
    Diagram.Eval, Diagram.Coeval, Diagram.Curry)


@Diagram.generator
class Functor(symmetric.Functor, ribbon.Functor):
    """
    A compact functor is both a symmetric functor and a ribbon functor.

    Parameters:
        ob_map (Mapping[pivotal.Ty, pivotal.Ty]) :
            Map from atomic :class:`pivotal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, (symmetric.Swap, symmetric.Permutation)):
            return symmetric.Functor.__call__(self, other)
        return ribbon.Functor.__call__(self, other)


CMap = cmap.CMap[Diagram]

TermBase, Constant, Variable, Application, Abstraction = (
    Diagram.TermBase, Diagram.Constant, Diagram.Variable,
    Diagram.Application, Diagram.Abstraction)
Id = Diagram.id

Hypergraph = hypergraph.Hypergraph[Diagram]


class Equation(symmetric.Equation):
    """ The :class:`symmetric.Equation` of compact diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)


Diagram.Equation = Equation

__getattr__ = deprecated_alias(__name__, {"Ob": "Wire"})
