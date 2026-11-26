# -*- coding: utf-8 -*-

"""
The free braided category, i.e. diagrams with braids.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Wire
    Diagram
    Box
    Braid
    Slide
    Unbraid
    Sum
    Functor
    TermBase
    Variable
    Constant
    Application
    Kappa
    Equation

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        hexagon

Axioms
------
Braids have their dagger as inverse, up to :meth:`Diagram.simplify`.

>>> x, y, z = map(Ty, "xyz")
>>> LHS = Braid(x, y) >> Braid(x, y)[::-1]
>>> RHS = Braid(y, x)[::-1] >> Braid(y, x)
>>> assert LHS.simplify() == Id(x @ y) == RHS.simplify()

>>> Equation(LHS, Id(x @ y), RHS).draw(
...     doctest='docs/_static/braided/inverse.svg')

.. image:: /_static/braided/inverse.svg
    :align: center

The hexagon equations hold on the nose.

>>> left_hexagon = Braid(x, y) @ z >> y @ Braid(x, z)
>>> assert left_hexagon == Diagram.braid(x, y @ z)
>>> right_hexagon = x @ Braid(y, z) >> Braid(x, z) @ y
>>> assert right_hexagon == Diagram.braid(x @ y, z)

>>> Equation(left_hexagon, right_hexagon, symbol='').draw(
...     space=2, doctest='docs/_static/braided/hexagons.svg')

.. image:: /_static/braided/hexagons.svg
    :align: center
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import ClassVar

from discopy import monoidal, messages, rewrite
from discopy.abc import BraidedCategory, SpacialCategory
from discopy.cat import factory, kappa_calculus
from discopy.drawing import Drawing
from discopy.drawing.drawing import between
from discopy.monoidal import Ty
from discopy.utils import (
    assert_isatomic, AxiomError, BinaryBoxConstructor, deprecated_alias,
    factory_name, from_tree, Point)


class Wire(monoidal.Wire):
    """
    A braided object is a self-dagger :class:`monoidal.Wire`. From braided
    categories onwards colours stop making sense, i.e. we cannot add colours to
    braids or swaps in any meaningful way, so it defaults to no colour.
    """
    def dagger(self) -> Wire:
        return self


@factory
class Diagram(monoidal.Diagram, BraidedCategory):
    """
    A braided diagram is a monoidal diagram with :class:`Braid` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.
    """
    braid_naturality = BraidedCategory.braid_naturality.failing(
        messages.NO_BRAID_GROUP)
    braid_then_inverse = BraidedCategory.braid_then_inverse.failing(
        messages.NO_BRAID_GROUP)
    inverse_then_braid = BraidedCategory.inverse_then_braid.failing(
        messages.NO_BRAID_GROUP)

    box_factory: ClassVar[type["Box"]]
    braid_factory: ClassVar[type["Braid"]]
    functor_factory: ClassVar[type["Functor"]]
    sum_factory: ClassVar[type["Sum"]]
    scalar_sliding = SpacialCategory.scalar_sliding.failing(
        messages.FREE_SCALAR)

    @classmethod
    def braid(cls, left: monoidal.Ty, right: monoidal.Ty) -> Diagram:
        """
        The diagram braiding :code:`left` over :code:`right`.

        Parameters:
            left : The type on the top left and bottom right.
            right : The type on the top right and bottom left.

        Note
        ----
        This calls :func:`hexagon` and :attr:`braid_factory`.
        """
        return hexagon(cls, cls.braid_factory)(left, right)

    def simplify(self) -> Diagram:
        """
        Remove braids followed by their dagger, splitting the layers of more
        than one box into staircases first, see :meth:`to_staircases`.

        Example
        -------
        >>> x, y = map(Ty, "xy")
        >>> f, g = Box('f', x, x), Box('g', y, y)
        >>> diagram = (f @ g >> Braid(x, y) >> Braid(x, y)[::-1]).foliation()
        >>> assert diagram.simplify() == f @ y >> x @ g
        """
        if any(len(layer.boxes) > 1 for layer in self.inside):
            return self.to_staircases().simplify()
        boxes, offsets = self.boxes, self.offsets
        for i, (f, g) in enumerate(zip(boxes, boxes[1:])):
            if offsets[i] == offsets[i + 1]\
                    and isinstance(f, Braid) and f == g[::-1]:
                inside = self.inside[:i] + self.inside[i + 2:]
                return self.ar(
                    inside, self.dom, self.cod, _scan=False).simplify()
        return self

    def rewrite_to(self, other: Diagram, left=False) -> rewrite.Rewrite:
        """
        The rewrite of one step from the diagram to ``other``: an
        interchanger, see :meth:`discopy.monoidal.Diagram.rewrite_to`, or
        an :class:`Unbraid` when ``other`` is the diagram without a braid and
        its inverse, as :meth:`simplify` removes them.

        Parameters:
            other : The diagram to rewrite into.
            left : Passed to :meth:`discopy.monoidal.Diagram.rewrite_to`.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> diagram = Braid(x, y) >> Braid(x, y)[::-1]
        >>> print(diagram.rewrite_to(diagram.simplify()))
        Unbraid(x, y)
        """
        if len(self) == len(other) + 2:
            i = next((i for i, (layer, that) in enumerate(
                zip(self.inside, other.inside)) if layer != that),
                len(other))
            try:
                step = Unbraid.at(self, i)
            except AxiomError:
                step = None
            if step is not None and step.cod == other:
                return step
        return super().rewrite_to(other, left)

    def naturality(self, i: int, left=True, down=True, braid=None) -> Diagram:
        """
        Slide a box through a braid.

        Parameters:
            i : The index of the box to slide.
            left : Whether to slide left or right.
            down : Whether to slide down or up.
            braid : The braiding method to be used.

        A diagram with more than one box in a layer is split into staircases
        first, see :meth:`to_staircases`, so that box ``i`` is layer ``i``.

        Examples
        --------
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box('f', x, y)
        >>> top_left = f @ z >> Braid(y, z)
        >>> top_right = z @ f >> Braid(z, y)
        >>> bot_left = Braid(z, x) >> f @ z
        >>> bot_right = Braid(x, z) >> z @ f
        >>> assert top_right.naturality(0) == bot_left
        >>> assert top_left.naturality(0, left=False) == bot_right
        >>> assert bot_right.naturality(1, down=False) == top_left
        >>> assert bot_left.naturality(1, left=False, down=False) == top_right

        This is the codomain of the rewrite
        :meth:`Slide.at`, which a gif draws as the box
        sliding through the braids:

        >>> g, w = Box('g', x, y @ z, draw_as_spider=True), Ty('w')
        >>> slide = Slide.at(w @ g >> Diagram.braid(w, y @ z), 0)
        >>> _ = slide.to_gif(tween=12, timestep=80, loop=True,
        ...     doctest="docs/_static/braided/naturality.gif")

        .. image:: /_static/braided/naturality.gif
            :align: center
        """
        if any(len(layer.boxes) > 1 for layer in self.inside):
            return self.to_staircases().naturality(i, left, down, braid)
        return Slide.at(self, i, left=left, down=down, braid=braid).cod


class Box(monoidal.Box, Diagram):
    """
    A braided box is a monoidal box in a braided diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """


class Braid(BinaryBoxConstructor, Box):
    """
    The braiding of atomic types :code:`left` and :code:`right`.

    Parameters:
        left : The type on the top left and bottom right.
        right : The type on the top right and bottom left.
        is_dagger : Braiding over or under.

    Important
    ---------
    :class:`Braid` is only defined for atomic types (i.e. of length 1).
    For complex types, use :meth:`Diagram.braid` instead.
    """
    serialised_attrs = ("left", "right", "is_dagger")

    def __init__(self, left: monoidal.Ty, right: monoidal.Ty, is_dagger=False):
        assert_isatomic(left, monoidal.Ty)
        assert_isatomic(right, monoidal.Ty)
        name = type(self).__name__\
            + (f"({right}, {left})" if is_dagger else f"({left}, {right})")
        dom, cod = left @ right, right @ left
        Box.__init__(
            self, name, dom, cod, is_dagger=is_dagger, draw_as_braid=True)
        BinaryBoxConstructor.__init__(self, left, right)

    def __getnewargs__(self):
        """ The arguments of the constructor, which a pickle and a delay read.
        """
        return self.left, self.right, self.is_dagger

    def __repr__(self):
        str_is_dagger = ", is_dagger=True" if self.is_dagger else ""
        return factory_name(type(self)) + \
            f"({repr(self.left)}, {repr(self.right)}{str_is_dagger})"

    @classmethod
    # pylint: disable-next=arguments-differ  # drawn with no boundary
    def strategy(cls, *, types=None):
        """
        Generate the braid of two atomic types, over or under.

        Parameters:
            types : The strategy to draw the types from, see
                :meth:`atomic_strategy`.
        """
        from hypothesis import strategies as st

        atoms = cls.atomic_strategy(types)
        return st.builds(cls, atoms, atoms, st.booleans())

    def dagger(self):
        return type(self)(self.right, self.left, not self.is_dagger)

    def to_tree(self) -> dict:
        """
        Serialise a braid, keeping whether it is an inverse.

        >>> from discopy.utils import dumps, loads
        >>> x, y = Ty('x'), Ty('y')
        >>> assert loads(dumps(Braid(x, y).dagger())) == Braid(x, y).dagger()
        """
        tree = BinaryBoxConstructor.to_tree(self)
        if self.is_dagger:
            tree['is_dagger'] = True
        return tree

    @classmethod
    def from_tree(cls, tree: dict) -> Braid:
        left, right = map(from_tree, (tree['left'], tree['right']))
        return cls(left, right, is_dagger=tree.get('is_dagger', False))


def hexagon(cls: type, factory: Callable) -> Callable[[Ty, Ty], Diagram]:
    """
    Take a ``factory`` for braids of atomic types and extend it recursively.

    Parameters:
        cls : A diagram factory, e.g. :class:`Diagram`.
        factory : A factory for braids of atomic types, e.g. :class:`Braid`.
    """
    def method(left: Ty, right: Ty) -> Diagram:
        if len(left) == 0:
            return cls.id(right)
        if len(right) == 0:
            return cls.id(left)
        if len(left) == len(right) == 1:
            return factory(left[0], right[0])
        if len(left) == 1:
            return method(left, right[:1]) @ right[1:]\
                >> right[:1] @ method(left, right[1:])
        return left[:1] @ method(left[1:], right)\
            >> method(left[:1], right) @ left[1:]

    return method


class Sum(monoidal.Sum, Box):
    """
    A braided sum is a monoidal sum and a braided box.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(monoidal.Functor):
    """
    A braided functor is a monoidal functor that preserves braids.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Braid) and not other.is_dagger\
                and hasattr(self.cod, "braid"):
            return self.cod.braid(self(other.dom[0]), self(other.dom[1]))
        return super().__call__(other)


Diagram.box_factory = Box
Diagram.braid_factory, Diagram.functor_factory = Braid, Functor
Diagram.sum_factory = Sum
Rewrite = rewrite.Rewrite[Diagram]
TILT = .2  # The angle in radians a braid turns as it slides through a wire.


class Slide[category](rewrite.Step):
    """
    A box slides through the braid with a wire next to it: the instance of
    ``braid_naturality`` with an identity, see
    :meth:`Diagram.naturality`, a step of the rewrites of the ``category``,
    that of ``box`` when not given.

    Parameters:
        box : The box that slides.
        wire : The type of one wire it slides through.
        left : Whether the wire is on the left of the box, else right.
        down : Whether the box slides down, else up.
        braid : The braid, the :meth:`Diagram.braid` of the
            box's diagram by default.

    Example
    -------
    >>> x, y, z = map(Ty, "xyz")
    >>> f = Box('f', x, y)
    >>> step = Slide(f, z)
    >>> assert type(step).ar is Rewrite
    >>> print(step.dom, "|", step.cod)
    z @ f >> Braid(z, y) | Braid(z, x) >> f @ z
    >>> assert step.dagger() == Slide(f, z, left=False, down=False)
    """
    serialised_attrs = ("box", "wire", "*", "left", "down", "braid")

    @classmethod
    def subscript_bases(cls, category) -> tuple[type, ...]:
        """ The steps of the category, of which a slide is one. """
        return (rewrite.Step[category], )

    @staticmethod
    def category_of(*args, **kwargs) -> type:
        """ The category of the box that slides, the first argument. """
        return (args[0] if args else kwargs["box"]).factory

    # pylint: disable-next=too-many-arguments  # naturality's arguments
    def __init__(self, box: Box, wire: Ty, *, left=True,
                 down=True, braid=None):
        self.box, self.wire, self.left, self.down = box, wire, left, down
        self.braid = braid
        braid = braid or box.braid
        if left and down:
            dom = wire @ box >> braid(wire, box.cod)
            cod = braid(wire, box.dom) >> box @ wire
        elif left:
            dom = braid(box.dom, wire) >> wire @ box
            cod = box @ wire >> braid(box.cod, wire)
        elif down:
            dom = box @ wire >> braid(box.cod, wire)
            cod = braid(box.dom, wire) >> wire @ box
        else:
            dom = braid(wire, box.dom) >> box @ wire
            cod = wire @ box >> braid(wire, box.cod)
        rewrite.Step.__init__(
            self, f"Slide({box}, {wire}{self.flags})", dom, cod)

    @property
    def flags(self) -> str:
        """ The arguments ``left`` and ``down`` that are not the default. """
        return (", left=False" if not self.left else "")\
            + (", down=False" if not self.down else "")

    def __repr__(self):
        return f"braided.Slide({self.box!r}, {self.wire!r}{self.flags})"

    def dagger(self) -> Slide:
        return type(self)(self.box, self.wire, left=not self.left,
                          down=not self.down, braid=self.braid)

    def frame(  # pylint: disable=too-many-locals  # Reidemeister III
            self, t: float) -> Drawing:
        """
        The crossings shrink into the box and grow out of its other side,
        see :meth:`discopy.drawing.Drawing.slide`. When the box is itself a
        braid, this is the third Reidemeister move, the Yang-Baxter equation:
        the braid moves in a straight line through the wire, turning by up to
        ``TILT`` radians on the way, the wire keeps its curve and only moves
        up or down, and the crossings of the wire with the legs of the braid
        shrink into the braid's ports until the three strands meet at
        ``t = .5``.

        Parameters:
            t : The fraction of the way, from ``0`` to ``1``.

        Example
        -------
        >>> x, y, z = map(Ty, "xyz")
        >>> step = Slide(Braid(y, z), x)
        >>> print(step.dom)
        x @ Braid(y, z) >> Braid(x, z) @ y >> z @ Braid(x, y)
        >>> print(step.cod)
        Braid(x, y) @ z >> y @ Braid(x, z) >> Braid(y, z) @ x
        >>> assert step.frame(0) == step.dom.to_drawing()
        >>> assert step.frame(1) == step.cod.to_drawing()
        >>> middle = step.frame(.5)
        >>> for node in middle.box_nodes: print(middle.positions[node])
        Point(x=1.0, y=2.0)
        Point(x=2.0, y=1.0)
        Point(x=1.5, y=1.5)
        >>> _ = step.to_gif(tween=12, timestep=80, loop=True,
        ...     doctest="docs/_static/braided/reidemeister-3.gif")

        .. image:: /_static/braided/reidemeister-3.gif
            :align: center
        """
        source, target = self.dom.to_drawing(), self.cod.to_drawing()
        if not getattr(self.box, "draw_as_braid", False):
            return source.slide(target, t)
        k, drawing = (0, source) if t < .5 else (1, target)
        indices = (0, len(self.cod.inside) - 1) if self.down\
            else (len(self.dom.inside) - 1, 0)
        j, strands = indices[k], drawing.strands()
        legs = {strands[node] for node in drawing.nodes
                if node.kind == "box_dom" and node.j == j}
        centres = [d.positions[d.box_nodes[i]]
                   for d, i in zip((source, target), indices)]
        centre, angle = between(*centres, t), TILT * math.sin(math.pi * t)

        def turned(p):
            dx, dy = p.x - centres[k].x, p.y - centres[k].y
            return Point(
                centre.x + dx * math.cos(angle) - dy * math.sin(angle),
                centre.y + dx * math.sin(angle) + dy * math.cos(angle))

        def top(d):
            return max((p for node, p in d.positions.items()
                        if node.kind == "box_dom"
                        and d.strands()[node] not in legs), key=lambda p: p.y)
        shift = (top(target).y - top(source).y) * (t - k)

        def position(node, p):
            if node.kind in ("dom", "cod"):
                return p
            if node.data.get("j") == j:
                return turned(p)
            if node.kind == "box" or strands[node] not in legs:
                return p.shift(0, shift)
            port = min((n for n in drawing.nodes if n.kind != "box"
                        and n.data.get("j") == j
                        and strands[n] == strands[node]),
                       key=lambda n: abs(drawing.positions[n].y - p.y))
            start = drawing.positions[port]
            shrink = 1 - 2 * abs(t - k)
            return turned(start).shift(
                shrink ** 2 * (p.x - start.x), shrink * (p.y - start.y))
        return drawing.relabel_nodes(positions={
            node: position(node, p)
            for node, p in drawing.positions.items()}).resized(
                source, target, t)

    @classmethod
    # pylint: disable-next=too-many-arguments  # naturality's arguments
    def at(cls, diagram: Diagram, i: int, *, left=True, down=True,
           braid=None) -> rewrite.Rewrite:
        """
        The slide of box ``i`` of a diagram with one box in each layer,
        whiskered by the rest, see
        :meth:`Diagram.naturality`, which is its codomain.

        Parameters:
            diagram : The diagram to rewrite.
            i : The index of the box to slide.
            left : Whether to slide left or right.
            down : Whether to slide down or up.
            braid : The braiding method to be used.

        Raises:
            AxiomError : When there is no braid to slide through.
        """
        box, offset = diagram.boxes[i], diagram.offsets[i]
        dom = diagram.inside[i].dom
        left_wires, right_wires = dom[:offset], dom[offset + len(box.dom):]
        wire = left_wires[-1] if left else right_wires[0]
        step = cls(box, wire, left=left, down=down, braid=braid)
        start = i if down else i - len(step.dom) + 1
        outside = (left_wires[:-1], right_wires) if left\
            else (left_wires, right_wires[1:])
        if start < 0 or diagram[start:start + len(step.dom)]\
                != outside[0] @ step.dom @ outside[1]:
            raise AxiomError(messages.NO_BRAID_TO_SLIDE.format(box, i))
        return step.whisker(
            diagram[:start], diagram[start + len(step.dom):], *outside)


class Unbraid[category](rewrite.Step):
    """
    A braid followed by its inverse pulled apart into parallel wires: the
    instance of ``braid_then_inverse``, or ``inverse_then_braid``, of
    :class:`discopy.abc.BraidedCategory`, the second Reidemeister move, a
    step of the rewrites of the ``category``, :class:`Diagram` when not
    given.

    Parameters:
        x : The type that starts on the left.
        y : The type that starts on the right.
        inverse : Whether the inverse comes first, from ``y @ x``.
        is_dagger : Whether the step goes the other way, from the wires to
            the braids.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> step = Unbraid(x, y)
    >>> print(step.dom, "|", step.cod)
    Braid(x, y) >> Braid(x, y)[::-1] | Id(x @ y)
    >>> print(Unbraid(x, y, inverse=True).dom)
    Braid(x, y)[::-1] >> Braid(x, y)
    >>> _ = Unbraid.at(Braid(x, y) >> Braid(x, y)[::-1], 0).to_gif(
    ...     tween=12, timestep=60, loop=True,
    ...     doctest="docs/_static/braided/unbraid.gif")

    .. image:: /_static/braided/unbraid.gif
        :align: center
    """
    serialised_attrs = ("x", "y", "*", "inverse", "is_dagger")

    @classmethod
    def subscript_bases(cls, category) -> tuple[type, ...]:
        """ The steps of the category, of which an unbraid is one. """
        return (rewrite.Step[category], )

    @staticmethod
    def category_of(*args, **kwargs) -> type:
        """ Braided diagrams, unless the class is given a category. """
        return Diagram

    def __init__(self, x: Ty, y: Ty, *, inverse=False, is_dagger=False):
        self.x, self.y, self.inverse = x, y, inverse
        braid = self.category.braid(x, y)
        braids = braid[::-1] >> braid if inverse else braid >> braid[::-1]
        wires = self.category.id(braids.dom)
        rewrite.Step.__init__(
            self, f"Unbraid({x}, {y}"
            + (", inverse=True)" if inverse else ")"),
            *((wires, braids) if is_dagger else (braids, wires)),
            is_dagger=is_dagger)

    def __repr__(self):
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        return f"braided.Unbraid({self.x!r}, {self.y!r}"\
            + (", inverse=True)" if self.inverse else ")")

    def dagger(self) -> Unbraid:
        return type(self)(self.x, self.y, inverse=self.inverse,
                          is_dagger=not self.is_dagger)

    def frame(self, t: float) -> Drawing:
        """
        The strands pull apart and shrink: each node of a strand, see
        :meth:`discopy.drawing.Drawing.strands`, moves in a straight line to
        the bottom of its wire and each crossing to the middle of its
        strands, while the height goes to zero, that of wires composed with
        other drawings, as for :meth:`discopy.rigid.Unsnake.frame`.

        Parameters:
            t : The fraction of the way, from ``0`` to ``1``.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> step = Unbraid(x, y)
        >>> assert step.frame(0) == step.dom.to_drawing()
        >>> assert step.frame(1) == step.cod.to_drawing()
        >>> step.frame(.5).height
        1.0
        """
        if self.is_dagger:
            return self.dagger().frame(1 - t)
        braids, wires = self.dom.to_drawing(), self.cod.to_drawing()
        if t == 1:
            return wires
        strands = braids.strands()
        bottoms = {node.i: Point(p.x, 0) for node, p in wires.positions.items()
                   if node.kind == "dom"}
        target = {node: bottoms[k] for node, k in strands.items()}
        for node in braids.nodes:
            if node.kind == "box":
                ports = [p for port, p in target.items()
                         if port.kind == "box_dom" and port.j == node.j]
                target[node] = between(*ports, .5)
        result = braids.relabel_nodes(positions={
            node: between(p, target[node], t)
            for node, p in braids.positions.items()}).resized(
                braids, wires, t)
        result.height = (1 - t) * braids.height
        return result

    @classmethod
    def at(cls, diagram: Diagram, i: int) -> rewrite.Rewrite:
        """
        The unbraid of the braid at layer ``i`` and its inverse at layer
        ``i + 1`` of a diagram with one box in each layer, whiskered by the
        rest: the diagram without them is its codomain.

        Parameters:
            diagram : The diagram to rewrite.
            i : The layer of the first braid.

        Raises:
            AxiomError : When the two layers are not a braid and its inverse.

        Example
        -------
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> diagram = z @ Braid(x, y) >> z @ Braid(x, y)[::-1]
        >>> print(Unbraid.at(diagram, 0))
        z @ Unbraid(x, y)
        >>> print(Unbraid.at(diagram[::-1] >> diagram, 1))
        z @ Braid(x, y) >> z @ Unbraid(x, y, inverse=True) \
>> z @ Braid(x, y)[::-1]
        >>> Unbraid.at(Braid(x, y) >> Braid(y, x), 0)
        Traceback (most recent call last):
        ...
        discopy.utils.AxiomError: Layers 0 and 1 are not a braid and its \
inverse.
        """
        factory_ = cls if cls.category is not None else cls[diagram.factory]
        first, second = diagram.boxes[i:i + 2] if i + 1 < len(diagram)\
            else (None, None)
        if isinstance(first, Braid) and first[::-1] == second\
                and diagram.offsets[i] == diagram.offsets[i + 1]:
            x, y = (first.cod if first.is_dagger else first.dom)
            step = factory_(x, y, inverse=first.is_dagger)
            offset, wires = diagram.offsets[i], diagram.inside[i].dom
            outside = wires[:offset], wires[offset + 2:]
            return step.whisker(diagram[:i], diagram[i + 2:], *outside)
        raise AxiomError(messages.NOT_A_BRAID_PAIR.format(i, i + 1))


Id = Diagram.id


class Equation(monoidal.Equation):
    """ The :class:`monoidal.Equation` of braided diagrams. """


__getattr__ = deprecated_alias(__name__, {"Ob": "Wire"})

TermBase, Variable, Constant, Application, Kappa = kappa_calculus(
    Diagram)
Term = Variable | Constant | Application | Kappa


Bubble = Box.from_args
