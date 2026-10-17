# -*- coding: utf-8 -*-

"""
The free traced category, i.e. diagrams where outputs can feedback into inputs.

Note that these diagrams are planar traced so that e.g. :mod:`pivotal` diagrams
are traced in this sense. See :mod:`symmetric` for the usual notion of trace.

Whenever the diagrams are also :mod:`symmetric`, their equality can be checked
by translation to monogamous :mod:`hypergraph`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Trace
    Functor

Axioms
------

A monoidal category is right-traced when it comes with an operator of shape:

>>> from discopy.monoidal import Equation
>>> x, y, z = map(Ty, "xyz")
>>> f = Box("f", x @ z, y @ z)
>>> Equation(f, f.trace(), symbol="$\\\\mapsto$").draw(
...     doctest='docs/_static/traced/right-trace.svg')

It is left-traced when it comes with an operator of the following shape:

>>> g = Box("g", z @ x, z @ y)
>>> Equation(g, g.trace(left=True), symbol="$\\\\mapsto$").draw(
...     doctest='docs/_static/traced/left-trace.svg')


These are subjects to the axioms listed below. Note however that at the moment
equality of planar traced diagrams is not implemented, only symmetric traced.

>>> from discopy.symmetric import Ty, Box, Swap, Id, Equation
>>> from discopy import symmetric
>>> x = Ty('x')
>>> f, g = Box('f', x @ x, x @ x), Box('g', x, x)

Vanishing
=========

>>> assert f.trace(n=0) == f == f.trace(n=0, left=True)
>>> assert f.trace(n=2) == f.trace().trace()
>>> assert f.trace(n=2, left=True) == f.trace(left=True).trace(left=True)

Superposing
===========

>>> assert Equation((x @ f).trace(), x @ f.trace())
>>> assert Equation((f @ x).trace(left=True), f.trace(left=True) @ x)

Yanking
=======

>>> yanking = Equation(
...     Swap(x, x).trace(left=True), Id(x), Swap(x, x).trace())
>>> yanking.draw(
...     doctest='docs/_static/traced/yanking.svg',
...     wire_labels=False, figsize=(4, 1))

.. image:: /_static/traced/yanking.svg
    :align: center

>>> assert yanking

Naturality
==========

>>> tightening_left = Equation(
...     (x @ g >> f >> x @ g).trace(left=True),
...     g >> f.trace(left=True) >> g)
>>> tightening_left.draw(
...     doctest='docs/_static/traced/tightening-left.svg', wire_labels=False)

.. image:: /_static/traced/tightening-left.svg
    :align: center

>>> tightening_right = Equation(
...     (g @ x >> f >> g @ x).trace(),
...     g >> f.trace() >> g)
>>> tightening_right.draw(
...     doctest='docs/_static/traced/tightening-right.svg',
...     wire_labels=False)

.. image:: /_static/traced/tightening-right.svg
    :align: center

>>> assert tightening_left and tightening_right

Dinaturality
============

>>> sliding_left = Equation(
...     (f >> g @ x).trace(left=True),
...     (g @ x >> f).trace(left=True))
>>> sliding_left.draw(
...     doctest='docs/_static/traced/sliding-left.svg', wire_labels=False)

.. image:: /_static/traced/sliding-left.svg
    :align: center

>>> sliding_right = Equation(
...     (f >> x @ g).trace(),
...     (x @ g >> f).trace())
>>> sliding_right.draw(
...     doctest='docs/_static/traced/sliding-right.svg', wire_labels=False)

.. image:: /_static/traced/sliding-right.svg
    :align: center

>>> assert sliding_left and sliding_right
"""

from typing import ClassVar

from discopy import monoidal, cmap, hypergraph, messages
from discopy.abc import TracedCategory
from discopy.cat import factory
from discopy.monoidal import Ty  # noqa: F401  pylint: disable=unused-import
from discopy.utils import (
    factory_name,
    from_tree,
    assert_isinstance,
    assert_istraceable,
    from_tree,
)


@factory
class Diagram(monoidal.Diagram, TracedCategory):
    """
    A traced diagram is a monoidal diagram with :class:`Trace` boxes.

    Parameters:
        inside(monoidal.Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.
    """
    trace_naturality_left = TracedCategory.trace_naturality_left.failing(
        messages.NO_PLANAR_NORMAL_FORM)
    trace_naturality_right = TracedCategory.trace_naturality_right.failing(
        messages.NO_PLANAR_NORMAL_FORM)
    trace_dinaturality_left = TracedCategory.trace_dinaturality_left.failing(
        messages.NO_PLANAR_NORMAL_FORM)
    trace_dinaturality_right = TracedCategory.trace_dinaturality_right.failing(
        messages.NO_PLANAR_NORMAL_FORM)
    trace_superposing_left = TracedCategory.trace_superposing_left.failing(
        messages.NO_PLANAR_NORMAL_FORM)
    trace_superposing_right = TracedCategory.trace_superposing_right.failing(
        messages.NO_PLANAR_NORMAL_FORM)

    sum_factory: ClassVar[type["Sum"]]
    box_factory: ClassVar[type["Box"]]
    functor_factory: ClassVar[type["Functor"]]
    trace_factory: ClassVar[type["Trace"]]

    def trace(self, n=1, left=False):
        """
        Feed ``n`` outputs back into inputs.

        Parameters:
            n : The number of output wires to feedback into inputs.
            left : Whether to trace the wires on the left or right.

        Example
        -------
        >>> from discopy.monoidal import Equation as Eq
        >>> x = Ty('x')
        >>> f = Box('f', x @ x, x @ x)
        >>> LHS, RHS = f.trace(left=True), f.trace(left=False)
        >>> Eq(Eq(LHS, f, symbol="$\\\\mapsfrom$"),
        ...     RHS, symbol="$\\\\mapsto$").draw(
        ...         doctest="docs/_static/traced/trace.svg")

        .. image:: /_static/traced/trace.svg
        """
        return self if n == 0\
            else self.trace_factory(self, left).trace(n - 1, left)

    @property
    def is_strict(self) -> bool:
        """
        Whether the diagram is strict, i.e. made of strict boxes only.

        A map ``h : U -> V`` is strict when the trace is uniform along it,
        i.e. when ``f >> B @ h == A @ h >> g`` implies that ``f.trace()``
        equals ``g.trace()`` for every ``f : A @ U -> B @ U`` and
        ``g : A @ V -> B @ V``, after Hasegawa's *Recursion from cyclic
        sharing* (1997) and Simpson and Plotkin's *Complete axioms for
        categorical fixed-point operators* (2000), where the trace is a
        least fixed point and strict maps preserve the least element.
        Strict maps form a monoidal subcategory, so a diagram is strict
        when each of its boxes is. A free box is not strict, nothing being
        known about it, while a permutation, a copy, a discard and a merge
        are, see :meth:`markov.Diagram.trace_uniformity`.
        """
        return all(box.is_strict for box in self.boxes)

    def to_drawing(self):
        return monoidal.Diagram.to_drawing(self, functor_factory=Functor)

    hypergraph_section = monoidal.Diagram.hypergraph_section.failing(
        messages.SECTION_NEEDS_SWAPS)


class Box(monoidal.Box, Diagram):
    """
    A traced box is a monoidal box in a traced diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """
    is_strict = False


class Trace(Box, monoidal.Bubble):
    """
    A trace is a diagram ``arg`` with an output wire fed back into an input.

    Parameters:
        arg : The diagram to trace.
        left : Whether to trace the wires on the left or right.

    See also
    --------
    :meth:`Diagram.trace`
    """
    def __init__(self, arg: Diagram, left=False):
        assert_isinstance(arg, self.ar)
        assert_istraceable(arg, n=1, left=left)
        self.left = left
        name = f"Trace({arg}, left=True)" if left else f"Trace({arg})"
        dom, cod = (arg.dom[1:], arg.cod[1:]) if left\
            else (arg.dom[:-1], arg.cod[:-1])
        monoidal.Bubble.__init__(self, arg, dom=dom, cod=cod)
        Box.__init__(self, name, dom, cod)

    def __str__(self):
        return self.name

    def __repr__(self):
        """
        >>> from discopy import cat, monoidal, traced
        >>> x = Ty('x')
        >>> trace = Box('f', x @ x, x @ x).trace()
        >>> assert eval(repr(trace)) == trace
        """
        return factory_name(type(self)) + f"({self.arg!r}, left={self.left!r})"

    def to_tree(self) -> dict:
        """
        Serialise a trace by its argument and the side it traces.

        >>> from discopy.utils import dumps, loads
        >>> x = Ty('x')
        >>> trace = Box('f', x @ x, x @ x).trace(left=True)
        >>> assert loads(dumps(trace)) == trace
        """
        return {'factory': factory_name(type(self)),
                'arg': self.arg.to_tree(), 'left': self.left}

    @classmethod
    def from_tree(cls, tree: dict) -> "Trace":
        return cls(from_tree(tree['arg']), left=tree.get('left', False))

    def dagger(self):
        return self.arg.dagger().trace(left=self.left)

    @property
    def is_strict(self) -> bool:
        """ The fixed point of a strict map is strict. """
        return self.arg.is_strict

    def to_drawing(self):
        return self.ar.to_drawing(self)


class Sum(monoidal.Sum, Box):
    """
    A traced sum is a sum of traced diagrams.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(monoidal.Functor):
    """
    A traced functor is a monoidal functor that preserves traces.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.

    Example
    -------
    Let's compute the golden ratio as the trace of a Python function, i.e.
    the nested radical iterated from ``float()``, see
    :meth:`python.Function.trace <discopy.python.Function.trace>`.

    >>> from math import sqrt
    >>> from discopy import python
    >>> x = Ty('$\\\\mathbb{R}$')
    >>> f = Box('$\\\\lambda x . (x, \\\\sqrt{1 + x})$', x, x @ x)
    >>> g = Box('$\\\\frac{1 + \\\\sqrt{5}}{2}$', Ty(), x)
    >>> F = Functor(
    ...     ob_map={x: float},
    ...     ar_map={
    ...         f: lambda x: (x, sqrt(1 + x)),
    ...         g: lambda: (1 + sqrt(5)) / 2},
    ...     cod=python.Function)
    >>> assert F(f.trace())() == F(g)()

    >>> from discopy.monoidal import Equation
    >>> Equation(f.trace(), g).draw(doctest="docs/_static/traced/golden.svg")

    .. image:: /_static/traced/golden.svg
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Trace):
            n = len(self(other.arg.dom)) - len(self(other.dom))
            return self.cod.trace(self(other.arg), n, left=other.left)
        return super().__call__(other)


CMap = cmap.CMap[Diagram]

Diagram.box_factory = Box
Diagram.functor_factory = Functor
Diagram.trace_factory = Trace
Diagram.sum_factory = Sum
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id
