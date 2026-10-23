# -*- coding: utf-8 -*-

"""
The free pivotal category,
i.e. diagrams with cups and caps that can rotate by a full turn.

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
    Layer
    Diagram
    Box
    Cup
    Cap
    Feedback
    Trace
    FollowedBy
    Head
    Tail
    Functor

Axioms
------
A pivotal category is a rigid category where left and right transpose coincide.

>>> x, y, z = map(Ty, "xyz")
>>> assert x.r == x.l and x.l.l == x == x.r.r
>>> f = Box('f', x, y)

>>> Equation(f.transpose(left=True), f.r, f.transpose(left=False)).draw(
...     doctest="docs/_static/pivotal/axiom.svg")

.. image:: /_static/pivotal/axiom.svg
    :align: center

For each diagram, we have its conjugate:

>>> d = Box('g', x @ y, z).curry()
>>> Equation(d, d.conjugate(), symbol="").draw(
...     space=2, doctest="docs/_static/pivotal/box-conjugate.svg")

.. image:: /_static/pivotal/box-conjugate.svg
    :align: center

We also have its dagger and its transpose:

>>> Equation(d.dagger(), d.rotate(), symbol="").draw(
...     space=2, doctest="docs/_static/pivotal/dagger-transpose.svg")

.. image:: /_static/pivotal/dagger-transpose.svg
    :align: center
"""

from __future__ import annotations

from copy import copy

from discopy import cat, cmap, rigid, feedback
from discopy.abc import PivotalCategory
from discopy.cat import factory
from discopy.utils import deprecated_alias, rsubs


class Wire(feedback.Wire, rigid.Wire):
    """
    A pivotal object is a rigid object where left and right adjoints coincide,
    and a feedback object with a time step.

    Parameters:
        name : The name of the object.
        z (bool) : Whether the object is an adjoint or not.
        time_step : The number of times the object has been delayed.
        is_constant : Whether the object is a constant stream.

    Example
    -------
    >>> from discopy import pivotal
    >>> x = Wire('x', time_step=2)
    >>> assert x.l == x.r == Wire('x', 1, time_step=2) and x.l.l == x
    >>> assert eval(repr(x.l)) == x.l and str(x.l) == 'x.r.d.d'
    """
    @property
    def l(self) -> Wire:
        """ The adjoint of the object, at the same time step. """
        result = copy(self)
        result.z, result.dom, result.cod = (self.z + 1) % 2, self.cod, self.dom
        return result

    r = l

    def __repr__(self):
        return rigid.Wire.__repr__(self)[:-1] + self.repr_time_step() + ")"

    def dagger(self) -> Wire:
        """
        The dagger of a pivotal object coincides with its left and right
        adjoints, i.e. it flips the parity of the winding number ``z`` and
        swaps its domain and codomain colours.
        """
        return self.l


HeadOb, TailOb = feedback.wire_generators(Wire)


@factory
class Ty(rigid.Ty, feedback.Ty):
    """
    A pivotal type is a rigid type with pivotal objects inside, and a
    feedback type that can be delayed.

    Parameters:
        inside (Wire) : The objects inside the type.
    """
    generator_factory = Wire


@factory
class Nat(rigid.Nat, Ty):
    """
    A pivotal ``Nat`` is a natural number ``n``
    seen as a pivotal type of length ``n``.

    Parameters
    ----------
    n : int
        The natural number.
    """
    l = r = property(lambda self: self)


class Layer(rigid.Layer, feedback.Layer):
    """ A pivotal layer is a rigid layer that can be delayed. """


@factory
class Diagram(rigid.Diagram, feedback.Diagram, PivotalCategory):
    """
    A pivotal diagram is a rigid diagram and a feedback diagram
    with pivotal types as domain and codomain.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty
    layer_factory = Layer

    def dagger(self):
        """
        The dagger of a pivotal diagram is its vertical reflection.

        Example
        -------
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box('f', x @ y, z).curry()

        >>> Equation(f, f.dagger(), symbol="$\\\\mapsto$").draw(
        ...     asymmetry=.1,
        ...     doctest="docs/_static/pivotal/dagger.svg")

        .. image:: /_static/pivotal/dagger.svg
            :align: center
        """
        return cat.Arrow.dagger(self)

    def conjugate(self):
        """
        The horizontal reflection of a diagram,
        defined as the dagger of the rotation.

        Equivalently, it is the rotation of the dagger.

        Example
        -------
        >>> x, y, z = map(Ty, "xyz")
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box('f', x @ y, z).curry()
        >>> assert f.conjugate() == f[::-1].rotate() == f.rotate()[::-1]

        >>> Equation(f, f.conjugate(), symbol="$\\\\mapsto$").draw(
        ...     doctest="docs/_static/pivotal/conjugate.svg")

        .. image:: /_static/pivotal/conjugate.svg
            :align: center
        """
        return self.rotate().dagger()

    @classmethod
    def trace_factory(cls, diagram: Diagram, left=False):
        """
        The trace of a pivotal diagram is its pre- and post-composition with
        cups and caps to form a feedback loop.

        Parameters:
            diagram : The diagram to trace.
            left : Whether to trace on the left or right.
        """
        traced_wire = diagram.dom[:1] if left else diagram.dom[-1:]
        dom, cod = (diagram.dom[1:], diagram.cod[1:]) if left\
            else (diagram.dom[:-1], diagram.cod[:-1])
        return cls.cap_factory(traced_wire.r, traced_wire) @ dom\
            >> traced_wire.r @ diagram\
            >> cls.cup_factory(traced_wire.r, traced_wire) @ cod if left\
            else dom @ cls.cap_factory(traced_wire, traced_wire.r)\
            >> diagram @ traced_wire.r\
            >> cod @ cls.cup_factory(traced_wire, traced_wire.r)


class Box(rigid.Box, feedback.Box, Diagram):
    """
    A pivotal box is a rigid box in a pivotal diagram.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
        time_step (int) : The number of times the box has been delayed.
    """
    def __init__(self, name, dom, cod, *args, time_step: int = 0, **params):
        rigid.Box.__init__(self, name, dom, cod, *args, **params)
        self._time_step, self._args, self._params = time_step, args, params

    def __repr__(self):
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return rigid.Box.__repr__(self)[:-1] + time_step + ")"

    def __str__(self):
        return rigid.Box.__str__(self) + feedback.str_delayed(self.time_step)

    def rotate(self, left=False):
        del left
        return type(self)(
            self.name, dom=self.cod.r, cod=self.dom.r, data=self.data,
            is_dagger=self.is_dagger, z=(self.z + 1) % 2,
            time_step=self.time_step)

    def dagger(self) -> Box:
        return type(self)(
            name=self.name, dom=self.cod, cod=self.dom, data=self.data,
            is_dagger=not self.is_dagger, z=self.z, time_step=self.time_step)

    def lambdify(self, *symbols, **kwargs):
        if not self.free_symbols & set(symbols):
            return lambda *xs: self
        return lambda *xs: type(self)(
            name=self.name, dom=self.dom, cod=self.cod,
            data=rsubs(self.data, dict(zip(symbols, xs)), **kwargs),
            is_dagger=self.is_dagger, z=self.z, time_step=self.time_step)

    @property
    def is_conjugate(self):
        """ Whether the box is a conjugate, i.e. the transpose of a dagger. """
        return self.is_dagger and bool(self.z)

    def to_drawing(self):
        result = super().to_drawing()
        result.is_conjugate = self.is_conjugate
        return result


class Cup(rigid.Cup, Box):
    """
    A pivotal cup is a rigid cup of pivotal types.

    Parameters:
        left (Ty) : The atomic type.
        right (Ty) : Its adjoint.
    """

    def dagger(self) -> Cap:
        """ The dagger of a pivotal cup. """
        return self.cap_factory(self.left, self.right)


class Cap(rigid.Cap, Box):
    """
    A pivotal cap is a rigid cap of pivotal types.

    Parameters:
        left (Ty) : The atomic type.
        right (Ty) : Its adjoint.
    """

    def dagger(self) -> Cup:
        """ The dagger of a pivotal cap. """
        return self.cup_factory(self.left, self.right)


class Sum(rigid.Sum, feedback.Sum, Box):
    """
    A pivotal sum is a sum of pivotal diagrams.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(rigid.Functor, feedback.Functor):
    """
    A pivotal functor is a rigid functor on a pivotal category.

    Parameters:
        ob_map (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram


Feedback, Trace, FollowedBy, Head, Tail = feedback.generators(Diagram, Box)
Diagram.box_factory = Box
Diagram.functor_factory = Functor
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
Diagram.sum_factory = Sum
CMap = cmap.CMap[Diagram]
Id = Diagram.id


class Equation(rigid.Equation):
    """ The :class:`rigid.Equation` of pivotal diagrams. """


__getattr__ = deprecated_alias(__name__, {"Ob": "Wire", "PRO": "Nat"})
