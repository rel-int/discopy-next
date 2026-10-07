# -*- coding: utf-8 -*-

"""
The free symmetric category, i.e. diagrams with swaps.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Layer
    Diagram
    Box
    Permutation
    Swap
    Trace
    Sum
    Bubble
    Functor

Axioms
------

>>> x, y, z, w = map(Ty, "xyzw")
>>> f, g = Box("f", x, y), Box("g", z, w)

Triangle
========

>>> assert Diagram.swap(Ty(), x) == Id(x) == Diagram.swap(x, Ty())

Hexagon
=======

>>> assert Diagram.swap(x, y @ z) == Swap(x, y) @ z >> y @ Swap(x, z)
>>> assert Diagram.swap(x @ y, z) == x @ Swap(y, z) >> Swap(x, z) @ y
>>> Equation(Diagram.swap(x, y @ z), Diagram.swap(x @ y, z), symbol='').draw(
...     space=2, doctest='docs/_static/symmetric/hexagons.svg', figsize=(5, 2))

.. image:: /_static/symmetric/hexagons.svg
    :align: center

Involution
==========
a.k.a. Reidemeister move 2

>>> assert Swap(x, y)[::-1] == Swap(y, x)
>>> assert Equation(Swap(x, y) >> Swap(y, x), Id(x @ y))
>>> Equation(Swap(x, y) >> Swap(y, x), Id(x @ y)).draw(
...     doctest='docs/_static/symmetric/inverse.svg', figsize=(3, 2))

.. image:: /_static/symmetric/inverse.svg
    :align: center

Naturality
==========

>>> naturality = Equation(
...     f @ g >> Swap(f.cod, g.cod), Swap(f.dom, g.dom) >> g @ f)
>>> assert naturality
>>> naturality.draw(
...     doctest='docs/_static/symmetric/naturality.svg', figsize=(3, 2))

.. image:: /_static/symmetric/naturality.svg
    :align: center

Yang-Baxter
===========
a.k.a. Reidemeister move 3

This is a special case of naturality.

>>> yang_baxter_left = Swap(x, y) @ z >> y @ Swap(x, z) >> Swap(y, z) @ x
>>> yang_baxter_right = x @ Swap(y, z) >> Swap(x, z) @ y >> z @ Swap(x, y)
>>> assert Equation(yang_baxter_left, yang_baxter_right)

Both sides foliate to the same single permutation.

>>> yang_baxter_middle = yang_baxter_left.foliation()
>>> assert yang_baxter_middle == yang_baxter_right.foliation()
>>> assert yang_baxter_middle == Permutation(x @ y @ z, [2, 1, 0])
>>> Equation(yang_baxter_left, yang_baxter_middle, yang_baxter_right).draw(
...     doctest='docs/_static/symmetric/yang-baxter.svg', figsize=(5, 2))

.. image:: /_static/symmetric/yang-baxter.svg
    :align: center

"""

from typing import ClassVar, Self

from collections.abc import Sequence

from discopy import cat, monoidal, balanced, hypergraph, cmap, messages
from discopy.pattern import Var, Tensor  # noqa: F401
from discopy.abc import (
    BalancedCategory, BraidedCategory, MonoidalCategory, SymmetricCategory,
    TracedCategory)
from discopy.axioms import (
    Atom, axiom, Equation as AbstractEquation, Hom, rule)
from discopy.cat import factory, Generator
from discopy.monoidal import Wire, Ty, Nat  # noqa: F401  pylint: disable=unused-import
from discopy.python import finset
from discopy.utils import (
    AxiomError, assert_iscomposable, assert_isatomic, factory_name, from_tree)


class Layer(monoidal.Layer):
    """
    A tensor product of generators and non-empty plumbing, where plumbing is a
    type when it is the identity and a :class:`Permutation` otherwise.
    :class:`Swap` is the permutation ``[1, 0]`` on two atomic wires.

    Plumbing components are coalesced, so a permutation given between two
    types becomes one permutation. Generators can be consecutive. An
    identity permutation is stored as its type, hence a layer with a single
    permutation always permutes: the identity is the empty diagram, not a
    layer.

    A layer with no crossing is stored exactly as a
    :class:`discopy.monoidal.Layer`.

    Parameters:
        inside : Generators and plumbing, with at least one generator or one
                 non-identity permutation.

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, perm = Box('f', x, y), Permutation(x @ y, [1, 0])
    >>> assert Layer(x, f, y).boxes_or_types == (x, f, y)
    >>> assert Layer(x, f, perm).boxes_or_types == (x, f, perm)
    >>> assert Layer(x, perm, y) == Layer(Permutation(x @ x @ y @ y,
    ...     [0, 2, 1, 3]))

    Forgetting the distinction between plumbing and generators gives the
    ordinary alternating view of a :class:`discopy.monoidal.Layer`, which is
    what :attr:`boxes`, :attr:`boxes_and_offsets` and the rewrites indexed by
    them are computed from.

    >>> assert Layer(x, f, perm).boxes_and_types == (
    ...     x, f, Ty(), perm, Ty())
    >>> assert Layer(x, f, perm).boxes_and_offsets == [(f, 1), (perm, 2)]
    """
    @classmethod
    def normalise(cls, inside):
        """
        Normalise identity permutations to their underlying types, so a
        layer whose only component is an identity permutation raises the
        same :class:`ValueError` as a layer without a box.
        """
        return super().normalise(
            value.dom
            if isinstance(value, Permutation) and hasattr(value, 'perm')
            and value.is_identity else value
            for value in inside)

    @property
    def is_plumbing(self) -> bool:
        """
        Whether the layer plumbs its wires non-trivially, i.e. one of its
        plumbing components is a :class:`Permutation` rather than a type.

        >>> x, y = Ty('x'), Ty('y')
        >>> assert Layer(Permutation(x @ y, [1, 0])).is_plumbing
        >>> assert not Layer(x, Box('f', x, y), y).is_plumbing
        """
        return any(isinstance(value, Permutation) for value in self)

    def merge(self, other: Layer) -> Layer:
        """
        Merge two layers of pure plumbing by composing their permutations,
        otherwise fall back to :meth:`discopy.monoidal.Layer.merge`.

        Parameters:
            other : The other layer with which to merge.

        Example
        -------
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> layer0 = Layer(Permutation(x @ y @ z, [1, 0, 2]))
        >>> layer1 = Layer(Permutation(y @ x @ z, [0, 2, 1]))
        >>> assert layer0.merge(layer1) == Layer(
        ...     Permutation(x @ y @ z, [1, 2, 0]))
        """
        if len(self.boxes_or_types) == 1 == len(other.boxes_or_types)\
                and self.is_plumbing and other.is_plumbing:
            assert_iscomposable(self, other)
            first, second = self.boxes_or_types[0], other.boxes_or_types[0]
            perm = [first.perm[i] for i in second.perm]
            if perm == sorted(perm):
                raise AxiomError(messages.NOT_MERGEABLE.format(self, other))
            return type(self)(first.Permutation(self.dom, perm))
        return super().merge(other)  # ty: ignore[invalid-return-type]


@factory
class Diagram(balanced.Diagram, SymmetricCategory):
    """
    A symmetric diagram is a balanced diagram with :class:`Swap` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.

    Note
    ----
    Equality and hashing of symmetric diagrams is always syntactic: two
    diagrams are equal if and only if they are built from the same layers.
    To compare diagrams up to hypergraph isomorphism (swaps, spider fusion,
    trace plumbing) use ``from discopy.symmetric import Equation``, i.e. the
    :class:`Equation` whose :attr:`~Equation.up_to` is :attr:`to_hypergraph`.

    >>> x, y = Ty("x"), Ty("y")
    >>> a = Swap(x, y) >> Swap(y, x)
    >>> assert a != Id(x @ y)
    >>> assert Equation(a, Id(x @ y))

    Note
    ----
    Symmetric diagrams can be defined using the standard syntax for functions.

    >>> x = Ty('x')
    >>> f = Box('f', x @ x, x)
    >>> g = Box('g', x, x @ x)

    >>> @Diagram.from_callable(x @ x @ x, x @ x @ x)
    ... def diagram(x0, x1, x2):
    ...     x3 = f(x2, x0)
    ...     x4, x5 = g(x1)
    ...     return x5, x3, x4
    >>> diagram.draw(wire_labels=False,
    ...              doctest='docs/_static/symmetric/decorator.svg')

    .. image:: /_static/symmetric/decorator.svg
        :align: center

    Every variable must be used exactly once or this will raise an error.

    >>> from pytest import raises
    >>> from discopy.utils import AxiomError

    >>> with raises(AxiomError) as err:
    ...     Diagram.from_callable(x, x @ x)(lambda x: (x, x))
    >>> print(err.value)
    symmetric.Diagram has no cups or caps for the wiring of this map.

    >>> with raises(AxiomError) as err:
    ...     Diagram.from_callable(x, Ty())(lambda x: ())
    >>> print(err.value)
    symmetric.Diagram has no cups or caps for the wiring of this map.

    Note
    ----
    As for :class:`discopy.balanced.Diagram`, our symmetric diagrams are traced
    by default. However now we have that the axioms for trace hold on the nose.

    Note
    ----
    The swaps of atomic types are generated by :attr:`Swap`, which
    subclasses should set to their own subclass of :class:`Swap`. It is the
    braid of a symmetric category, i.e. :attr:`braided.Diagram.Braid`
    reads it:

    >>> class Permutation(Diagram): ...
    >>> class Transposition(Swap, Permutation): ...
    >>> Permutation.Swap = Transposition
    >>> assert Permutation.Braid is Transposition
    """
    Braid = Generator.alias("Swap")
    Layer: ClassVar[Generator] = Generator.subclass(Layer)
    Twist = Generator.classmethod(lambda cls, dom: cls.id(dom))
    Permutation: ClassVar[Generator]
    Swap: ClassVar[Generator]
    Functor: ClassVar[Generator]

    @property
    def is_plumbing(self) -> bool:
        """ Whether one of the layers plumbs its wires non-trivially. """
        return any(layer.is_plumbing for layer in self.inside)

    @classmethod
    @rule
    def swap[X: Atom, Y: Atom](
            cls, left: Var[monoidal.Ty, X],
            right: Var[monoidal.Ty, Y]
    ) -> Hom[Diagram, Tensor[X, Y], Tensor[Y, X]]:
        """
        The diagram that swaps the ``left`` and ``right`` wires.

        Parameters:
            left : The type at the top left and bottom right.
            right : The type at the top right and bottom left.

        Note
        ----
        This calls :func:`balanced.hexagon` and :attr:`Swap`.
        """
        return cls.braid(left, right)

    @classmethod
    def permutation(cls, xs: Sequence[int],
                    doms: Sequence | monoidal.Ty | None = None) -> Diagram:
        """
        The diagram that encodes a given permutation as a composition of
        swaps.

        Parameters:
            xs : A permutation, as a sequence of integers or a
                 :class:`finset.Permutation`.
            dom : A type of the same length as :code:`xs`,
                  default is :code:`Nat(len(xs))`.
        """

        if doms is None:
            doms = Nat(len(xs))
        size = len(doms)
        unit = type(doms)() if isinstance(doms, Nat) else cls.ob()
        tensor = lambda tys: unit.tensor(*tys)
        dom = tensor(doms)

        xs = finset.Permutation(xs, size)
        if xs.is_identity:
            return cls.id(dom)
        i = xs[0]
        left, head, right = (
            doms[slice]
            for slice in (
                slice(0, i), i, slice(i + 1, None)
            )
        )
        rest = (left @ right  # ty: ignore[unsupported-operator]
                if isinstance(doms, monoidal.Ty)
                else left + right)  # ty: ignore[unsupported-operator]
        return cls.swap(
            tensor(left), head  # ty: ignore[invalid-argument-type]
        ) @ tensor(right)\
            >> head @ cls.permutation(
                [x - 1 if x > i else x for x in xs[1:]], rest)

    @classmethod
    @rule
    def cycle[X: Atom, A](
            cls, x: Var[Ty, X], a: Var[Ty, A]
    ) -> Hom[Diagram, Tensor[X, A], Tensor[A, X]]:
        """
        The permutation moving a wire past a type, a native
        :class:`Permutation` of any length.

        Parameters:
            x : The wire to move.
            a : The type to move it past.

        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> assert Diagram.cycle(x, y @ z) == Permutation(x @ y @ z, [1, 2, 0])
        """
        assert_isatomic(x, cls.ob)
        return cls.from_permutation([*range(1, len(a) + 1), 0], x @ a)

    @classmethod
    def from_permutation(cls, perm: Sequence[int],
                         dom: monoidal.Ty | None = None) -> Diagram:
        """
        Encode a permutation natively when the category has a matching
        :class:`Permutation` factory. Descendant categories without one use
        their own swap decomposition instead. An identity permutation always
        becomes the identity diagram.

        Parameters:
            perm : A permutation, as a sequence of integers or a
                   :class:`finset.Permutation`.
            dom : A type of the same length as :code:`perm`,
                  default is :code:`Nat(len(perm))`.

        Examples
        --------
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> assert Diagram.from_permutation([1, 2, 0], x @ y @ z)\\
        ...     == Permutation(x @ y @ z, [1, 2, 0])
        >>> assert Diagram.from_permutation(
        ...     [0, 1, 2], x @ y @ z) == Id(x @ y @ z)
        """
        dom = Nat(len(perm)) if dom is None else dom
        perm = finset.Permutation(perm, len(dom))
        if perm.is_identity:
            return cls.id(dom)
        if cls.Permutation.ar is cls:
            return cls.Permutation(dom, perm)
        return cls.permutation(perm, dom)

    def permute(self, *xs: int) -> Diagram:
        """
        Post-compose with a permutation written as the historical swap
        decomposition. Use :meth:`from_permutation` to construct a native
        :class:`Permutation` box.

        Parameters:
            xs : A list of integers representing a permutation.

        Examples
        --------
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> assert Id(x @ y @ z).permute(2, 0, 1).cod == z @ x @ y
        """
        return self >> self.permutation(list(xs), self.cod)

    def simplify(self):
        """ Simplify by translating back and forth to hypergraph. """
        return self.to_hypergraph().to_diagram()

    def foliation(self):
        """
        Merge independent generators, keeping native plumbing compact.

        A hypergraph forgets that plumbing is native, so a diagram with a
        :class:`Permutation` is foliated by merging its layers instead.

        >>> x, y = Ty('x'), Ty('y')
        >>> perm = Permutation(x @ y, [1, 0])
        >>> assert perm.foliation() == perm
        """
        if self.is_plumbing:
            return self.merge_layers()
        return super().foliation()

    def depth(self):
        """
        The depth of a symmetric diagram.

        Examples
        --------
        >>> x = Ty('x')
        >>> f = Box('f', x, x)
        >>> assert Id(x).depth() == Id().depth() == 0
        >>> assert f.depth() == (f @ f).depth() == 1
        >>> assert (f @ f >> Swap(x, x)).depth() == 1
        >>> assert (f >> f).depth() == 2 and (f >> f >> f).depth() == 3
        """
        return self.to_hypergraph().depth()

    bifunctoriality = MonoidalCategory.bifunctoriality

    dagger_monoidality = MonoidalCategory.dagger_monoidality

    #: A free braid is a box, but the braid of a symmetric category is
    #: its swap, whose naturality holds in the hypergraph quotient.
    braid_naturality = BraidedCategory.braid_naturality

    braid_then_inverse = BraidedCategory.braid_then_inverse

    inverse_then_braid = BraidedCategory.inverse_then_braid

    #: A free twist is a box, but the twist of a symmetric category is
    #: the identity, natural and the trace of a swap.
    twist_naturality = BalancedCategory.twist_naturality

    yanking = BalancedCategory.yanking

    #: The category has the swaps that decoding asks for, so the
    #: section of the hypergraph encoding comes back.
    hypergraph_section = monoidal.Diagram.hypergraph_section

    #: A free trace is a box, but the trace of a symmetric category is a
    #: feedback wire, whose (di)naturality and superposing hold in the
    #: hypergraph quotient.
    trace_naturality_left = TracedCategory.trace_naturality_left

    trace_naturality_right = TracedCategory.trace_naturality_right

    trace_dinaturality_left = TracedCategory.trace_dinaturality_left

    trace_dinaturality_right = TracedCategory.trace_dinaturality_right

    trace_superposing_left = TracedCategory.trace_superposing_left

    trace_superposing_right = TracedCategory.trace_superposing_right

    @axiom
    def hypergraph_retract(cls, f: Self):
        """
        Decoding the hypergraph of a diagram gives it back: from
        symmetric on the equation is the hypergraph quotient, which is
        where decoding lands.
        """
        functor = cls.hypergraph_equivalence()
        return cls.Equation(functor.decode(functor(f)), f)

    @classmethod
    def map_equivalence(cls) -> cat.Equivalence:
        """
        The equivalence sending a diagram to its combinatorial map:
        :meth:`~discopy.monoidal.Diagram.to_map` encodes and
        :meth:`discopy.cmap.CMap.to_diagram` decodes. A map is compact
        whatever category hosts it, so decoding one asks for swaps: the
        equivalence is stated where the swaps are.
        """
        return cat.Equivalence(
            cls.ar.to_map, cmap.CMap.to_diagram,
            cls.ar, cmap.CMap[cls.ar])

    @axiom
    def map_section(cls, f: Self):
        """
        Decoding is a section of the map encoding, modulo the
        hypergraph, which does not order the boxes: a diagram orders
        its boxes totally where a map orders them only by their wiring,
        so decoding picks one topological order among the diagrams of
        the same map and re-encoding can permute independent boxes.
        """
        functor = cls.map_equivalence()
        image = functor(f)
        return AbstractEquation(
            functor(functor.decode(image)), image,
            up_to=cmap.CMap.to_hypergraph)

    @axiom
    def map_retract(cls, f: Self):
        """
        Decoding the map of a diagram gives it back, up to the
        equation of the level, which is essential: a map is spacial —
        it cannot distinguish nested scalars from scalars side by
        side — and so is the hypergraph the equation compares by,
        while the syntactic comparison up to
        :meth:`~discopy.monoidal.Diagram.foliation` is false even on
        the boundary-connected subspace.
        """
        functor = cls.map_equivalence()
        return cls.Equation(functor.decode(functor(f)), f)

    @axiom
    def map_composition(cls, f: Self):
        """
        The encoding preserves composition: the map of a diagram is
        the composition of the maps of any two halves of it.
        """
        functor = cls.map_equivalence()
        top, bottom = f[:len(f) // 2], f[len(f) // 2:]
        return AbstractEquation(functor(f), functor(top) >> functor(bottom))

    @axiom
    def map_identity[X](cls, x: Var[Ty, X]):
        """ The encoding preserves identities. """
        functor = cls.map_equivalence()
        return AbstractEquation(functor(cls.id(x)), functor.cod.id(x))


Box = Diagram.Box


@Diagram.generator
class Permutation(Box):
    """
    A permutation box, i.e. a :class:`Box` that reorders its input wires.

    A permutation holds a :class:`discopy.python.finset.Permutation` ``perm``
    as attribute, with the convention that output wire ``i`` comes from input
    wire ``perm[i]``, i.e. ``cod[i] == dom[perm[i]]``.

    A :class:`Layer` stores it as plumbing rather than as a generator, and the
    identity permutation is the identity diagram. The transposition ``[1, 0]``
    on two atomic wires constructs a :class:`Swap`. Other permutations draw as
    a single band of crossing wires rather than a staircase of swaps.

    Parameters:
        dom : The domain, i.e. the wires to permute.
        perm : The permutation as a :class:`finset.Permutation` or a list.

    Examples
    --------
    >>> x, y, z, w = map(Ty, "xyzw")
    >>> perm = Permutation(x @ y @ z, [1, 2, 0])
    >>> assert perm.cod == y @ z @ x
    >>> assert perm.dagger() == Permutation(y @ z @ x, [2, 0, 1])
    >>> assert Equation(perm >> perm.dagger(), Id(x @ y @ z))
    >>> assert perm @ Id(w) == Permutation(x @ y @ z @ w, [1, 2, 0, 3])
    >>> assert Permutation(x @ y, [1, 0]) == Swap(x, y)
    >>> assert Permutation(x @ y, [0, 1]) == Id(x @ y)

    Writing permutations by hand keeps swap-heavy diagrams compact: a whole
    permutation occupies a single layer rather than a quadratic staircase of
    swaps. Reversing four wires before a single layer of boxes is a
    permutation layer followed by a box layer.

    >>> f0, f1 = Box("f0", w, x), Box("f1", z, y)
    >>> g0, g1 = Box("g0", y, z), Box("g1", x, w)
    >>> reverse = Permutation(x @ y @ z @ w, [3, 2, 1, 0])
    >>> diagram = reverse >> f0 @ f1 @ g0 @ g1
    >>> diagram.depth()
    1
    >>> diagram.draw(
    ...     doctest='docs/_static/symmetric/foliation.svg', figsize=(4, 4))

    .. image:: /_static/symmetric/foliation.svg
        :align: center
    """

    def __new__(cls, dom: monoidal.Ty | None = None,
                perm: Sequence[int] | None = None):
        if dom is None or cls is cls.ar.Swap:
            return super().__new__(cls)
        factory = cls.ar.Swap\
            if isinstance(perm, Sequence)\
            and finset.Permutation.is_swap(perm) else cls
        return super(Permutation, factory).__new__(factory)

    def __init__(self, dom: monoidal.Ty, perm: Sequence[int]):
        self.perm = finset.Permutation(perm, len(dom))
        cod = dom[:0].tensor(*(dom[i] for i in self.perm))
        name = f"Permutation({list(self.perm)})"
        super().__init__(
            name, dom, cod, drawing_name=name,
            draw_as_wires=True, draw_as_permutation=True,
            permutation_indices=tuple(self.perm))

    @property
    def is_identity(self) -> bool:
        """
        Whether the underlying permutation is the identity.

        >>> assert Permutation(Ty('x', 'y'), [0, 1]).is_identity
        """
        return self.perm.is_identity

    @property
    def size(self) -> int:
        """ Structural permutations are not generator boxes in a layer. """
        return 0

    def setoid(self):
        if self.is_identity:
            return (), self.dom, self.cod
        return type(self), self.dom, tuple(self.perm)

    def to_drawing(self):
        """ Draw as a compact band, or as wires for the identity. """
        from discopy.drawing import Drawing
        return Drawing.id(self.dom) if self.is_identity\
            else Drawing.from_box(self)

    def to_swaps(self) -> Diagram:
        """
        The same permutation built as a composition of swaps.

        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> perm = Permutation(x @ y @ z, [1, 2, 0])
        >>> assert Equation(perm.to_swaps(), perm)
        """
        doms = self.dom if isinstance(self.dom, Nat)\
            else list(map(self.ob, self.dom.inside))
        return self.ar.permutation(self.perm, doms)

    def to_tree(self) -> dict:
        """
        Serialise a permutation, see :func:`discopy.utils.dumps`.

        >>> from discopy.utils import dumps, loads
        >>> x, y = Ty('x'), Ty('y')
        >>> assert loads(dumps(Permutation(x @ y, [1, 0])))\\
        ...     == Permutation(x @ y, [1, 0])
        """
        return dict(factory=factory_name(type(self)),
                    dom=self.dom.to_tree(), perm=list(self.perm))

    @classmethod
    def from_tree(cls, tree: dict) -> Permutation:
        return cls(from_tree(tree['dom']), tree['perm'])

    def dagger(self) -> Permutation:
        return type(self)(self.cod, self.perm.dagger())

    def tensor[A, B, C, D](
            self: Hom[Permutation, A, B],
            other: Hom[Diagram | monoidal.Ty | None, C, D] = None,
            *others) -> Hom[Diagram, Tensor[A, C], Tensor[B, D]]:
        if other is None:
            return self
        if isinstance(other, Permutation):
            result = self.Permutation(
                self.dom @ other.dom, self.perm.tensor(other.perm))
        elif isinstance(other, monoidal.Ty)\
                or isinstance(other, Diagram) and not other.inside:
            typ = other if isinstance(other, monoidal.Ty) else other.dom
            result = self.Permutation(self.dom @ typ, self.perm.tensor(
                finset.Permutation.id(len(typ))))
        else:
            result = super().tensor(other)
        return result.tensor(*others)

    def __rmatmul__(self, other):
        if not isinstance(other, monoidal.Ty):
            return super().__rmatmul__(other)
        perm = finset.Permutation.id(len(other)).tensor(self.perm)
        return self.Permutation(other @ self.dom, perm)

    def __repr__(self):
        return f"{factory_name(type(self))}({self.dom!r}, {list(self.perm)})"

    def __str__(self):
        return f"Permutation({self.dom}, {list(self.perm)})"


Layer.plumbing = (monoidal.Ty, Permutation)


@Diagram.generator
class Swap(Permutation, balanced.Braid, Box):
    """
    The permutation ``[1, 0]`` of two atomic types.

    Parameters:
        left : The type on the top left and bottom right.
        right : The type on the top right and bottom left.

    Important
    ---------
    :class:`Swap` is only defined for atomic types (i.e. of length 1).
    For complex types, use :meth:`Diagram.swap` instead.
    """
    def __setstate__(self, state):
        state.setdefault('perm', finset.Permutation([1, 0], 2))
        super().__setstate__(state)

    def __init__(self, left, right):
        if len(left) == 2:
            perm = finset.Permutation(right, len(left))
            if not perm.is_swap():
                raise ValueError
            left, right = left[:1], left[1:]
        self.perm = finset.Permutation([1, 0], 2)
        balanced.Braid.__init__(self, left, right)
        self.Box.__init__(
            self, self.name, self.dom, self.cod,
            draw_as_wires=True, draw_as_braid=False)

    def dagger(self):
        return type(self)(self.right, self.left)

    def to_swaps(self):
        return self

    def to_drawing(self):
        return Box.to_drawing(self)

    def __repr__(self):
        return balanced.Braid.__repr__(self)

    def __str__(self):
        return self.name


Trace, Sum, Bubble = (
    Diagram.Trace, Diagram.Sum, Diagram.Bubble)


@Diagram.generator
class Functor[In0, In1, Out0, Out1](balanced.Functor):
    """
    A symmetric functor is a monoidal functor that preserves swaps.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.
    """
    dom = cod = Diagram

    @axiom
    def symmetric[X: Atom[In0], Y: Atom[In0]](
            cls, functor: Self, x: Var[Ty, X], y: Var[Ty, Y]) -> Equation:
        """ A symmetric functor preserves the swap. """
        return functor.cod.Equation(
            functor(functor.dom.swap(x, y)),
            functor.cod.swap(functor(x), functor(y)))

    def __call__(self, other):
        if isinstance(other, Swap) and hasattr(self.cod.ar, "swap"):
            return self.cod.ar.swap(self(other.dom[0]), self(other.dom[1]))
        if isinstance(other, Permutation) and hasattr(
                self.cod.ar, "permutation"):
            if isinstance(other.dom, Nat):
                doms = self(other.dom)
            else:
                doms = list(map(self, other.dom))
            atoms = list(doms)
            if hasattr(self.cod.ar, "Permutation")\
                    and len(atoms) == len(other.perm)\
                    and all(len(atom) == 1 for atom in atoms):
                dom = self.cod.ar.ob().tensor(*atoms)
                return self.cod.ar.Permutation(dom, other.perm)
            return self.cod.ar.permutation(other.perm, doms)
        return super().__call__(other)


CMap = cmap.CMap[Diagram]

Hypergraph = hypergraph.Hypergraph[Diagram]


Id = Diagram.id


class Equation(monoidal.Equation):
    """
    The :class:`monoidal.Equation` of symmetric diagrams compared up to
    hypergraph isomorphism, i.e. up to swaps, spider fusion and trace plumbing.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> assert Equation(Swap(x, y) >> Swap(y, x), Id(x @ y))
    """
    up_to = staticmethod(Diagram.to_hypergraph)


Diagram.Equation = Equation
