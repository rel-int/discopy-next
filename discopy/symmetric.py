# -*- coding: utf-8 -*-

"""
The free symmetric category, i.e. diagrams with swaps.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Permutation
    Layer
    Diagram
    Box
    Swap
    Sum
    Functor
    DualRail
    Equation
    TermBase
    Variable
    Constant
    Application
    Kappa
    Sampler

Terms
-----

A symmetric diagram with exactly one output reads as a term of the linear
kappa calculus, see :class:`TermBase`: a swap is the order of the variables,
and the term evaluates back to the diagram up to the axioms below.

>>> x, y = Ty('x'), Ty('y')
>>> diagram = Swap(x, y) >> Box('f', y @ x, x)
>>> print(diagram.to_term())
(x @ y)(lambda x0, x1: f(x1, x0))
>>> assert Equation(diagram.to_term().eval(), diagram)

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

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

from discopy import cat, monoidal, balanced, hypergraph, cmap, messages
from discopy.abc import (
    BalancedCategory, BiclosedCategory, CompactCategory,
    DaggerMonoidalCategory, PivotalCategory, RigidCategory, SpacialCategory,
    SymmetricCategory)
from discopy.testable import (
    FUEL, Axiom, Equation as AbstractEquation, Rule, axiom)
# pylint: disable-next=unused-import  # the redex a term prints as
from discopy.cat import let  # noqa: F401
from discopy.cat import factory
from discopy.monoidal import (  # noqa: F401  pylint: disable=unused-import
    Wire, Ty, Nat)
from discopy.python import finset
from discopy.utils import (
    AxiomError, assert_iscomposable, assert_isinstance, classproperty,
    factory_key, factory_name, from_tree)


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
    :class:`~discopy.monoidal.Layer`.

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

    Forgetting the distinction between plumbing and generators, a
    permutation is one more box of the layer: :attr:`boxes`,
    :attr:`boxes_and_offsets` and the rewrites indexed by them count it as
    one.

    >>> assert Layer(x, f, perm).boxes_and_offsets == [(f, 1), (perm, 2)]
    """
    @classmethod
    def normalise(cls, inside):
        """
        Normalise identity plumbing to its underlying type, so a layer whose
        only component is an identity permutation raises the same
        :class:`ValueError` as a layer without a box. A box being unpickled
        has no data yet, hence is not an identity.
        """
        return super().normalise(
            value.dom if cls.is_plumbing_box(value)
            and getattr(value, 'is_identity', False) else value
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
        return any(map(self.is_plumbing_box, self))

    def merge(self, other: Layer) -> Layer:
        """
        Merge two layers of pure plumbing by fusing their boxes, see
        :meth:`Permutation.fuse`, otherwise fall back to
        :meth:`~discopy.monoidal.Layer.merge`.

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
            result = self.boxes_or_types[0].fuse(other.boxes_or_types[0])
            if not result.inside:
                raise AxiomError(messages.NOT_MERGEABLE.format(self, other))
            return type(self)(result)
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
    The swaps of atomic types are generated by :attr:`swap_factory`, which
    subclasses should set to their own subclass of :class:`Swap`. It is the
    braid of a symmetric category, i.e. :attr:`braided.Diagram.braid_factory`
    reads it:

    >>> class Permutation(Diagram): ...
    >>> class Transposition(Swap, Permutation): ...
    >>> Permutation.swap_factory = Transposition
    >>> assert Permutation.braid_factory is Transposition
    """
    # A hypergraph is faithful from symmetric categories on.
    bifunctoriality = SymmetricCategory.bifunctoriality
    dagger_monoidality = DaggerMonoidalCategory.dagger_monoidality
    dagger_monoidality_symmetric = DaggerMonoidalCategory.dagger_monoidality\
        .weaken(within=SymmetricCategory)
    braid_naturality = SymmetricCategory.braid_naturality
    braid_then_inverse = SymmetricCategory.braid_then_inverse
    inverse_then_braid = SymmetricCategory.inverse_then_braid
    inverse_braiding = SymmetricCategory.inverse_braiding
    twist_naturality = BalancedCategory.twist_naturality

    box_factory: ClassVar[type["Box"]]
    functor_factory: ClassVar[type["Functor"]]
    swap_factory: ClassVar[type["Swap"]]
    permutation_factory: ClassVar[type["Permutation"]]
    sum_factory: ClassVar[type["Sum"]]
    braid_factory = classproperty(lambda cls: cls.swap_factory)
    layer_factory = Layer
    twist_factory = classmethod(lambda cls, dom: cls.id(dom))

    @classmethod
    def equation_factory(cls, *terms: Diagram) -> Equation:
        """
        The equation of a law, compared up to the isomorphism of the
        hypergraphs of its terms, the coherence theorem of symmetric
        categories: sliding boxes past each other and composing swaps is
        absorbed, while a free trace, curry or feedback is a box of the
        hypergraph like any other.

        >>> x = Ty('x')
        >>> f, g = Box('f', Ty(), x), Box('g', x, Ty())
        >>> assert Diagram.equation_factory(f @ g, g >> f)
        """
        return Equation(*terms)

    @classmethod
    def strategy(cls, *, dom=None, cod=None, fuel=FUEL, within=None):
        """
        Draw a diagram by one of two ways: :meth:`search`, the derivation from
        the rules of the category, or :meth:`quotient_strategy`, the downgrade
        of its combinatorial representation, which draws larger diagrams in
        one go.
        A morphism premise of the search is drawn by this strategy in turn,
        so a derived diagram can contain a sampled one. A search ``within``
        the rules of a weaker structure is not mixed with the sample, which
        has all the structure of the category.

        Parameters are those of :meth:`search`.

        >>> from hypothesis import find
        >>> x = Ty('x')
        >>> diagram = find(Diagram.strategy(dom=x, cod=x),
        ...                lambda d: len(d.boxes) > 1)
        >>> assert (diagram.dom, diagram.cod) == (x, x)
        """
        from hypothesis import strategies as st

        derivations = cls.search(dom=dom, cod=cod, fuel=fuel, within=within)
        if within is not None:
            return derivations
        return st.one_of(derivations, cls.quotient_strategy(
            dom=dom, cod=cod, fuel=fuel))

    @classmethod
    def quotient_strategy(cls, *, dom=None, cod=None, fuel=FUEL):
        """
        Sample symmetric diagrams through their quotient: a combinatorial map
        drawn by :meth:`~discopy.cmap.CMap.strategy` with at most ``fuel + 1``
        boxes, monogamous unless the category has cups and caps, downgraded
        with :meth:`~discopy.cmap.CMap.to_diagram`, which makes its cycles
        explicit as traces and its crossings as swaps.
        """
        return cmap.CMap[cls].strategy(  # ty: ignore[invalid-type-form]
            dom=dom, cod=cod, max_leaves=fuel + 1).map(
                lambda graph: graph.to_diagram())

    @property
    def is_plumbing(self) -> bool:
        """ Whether one of the layers plumbs its wires non-trivially. """
        return any(layer.is_plumbing for layer in self.inside)

    @classmethod
    def swap(cls, left: monoidal.Ty, right: monoidal.Ty) -> Diagram:
        """
        The diagram that swaps the ``left`` and ``right`` wires.

        Parameters:
            left : The type at the top left and bottom right.
            right : The type at the top right and bottom left.

        Note
        ----
        This calls :func:`balanced.hexagon` and :attr:`swap_factory`.
        """
        return cls.braid(left, right)  # ty: ignore[invalid-return-type]

    @classmethod
    def permutation(
            cls, xs: Sequence[int],
            doms: Sequence[monoidal.Ty] | Nat | None = None) -> Diagram:
        """
        The diagram that encodes a given permutation as a composition of
        swaps.

        Parameters:
            xs : A permutation, as a sequence of integers or a
                 :class:`finset.Permutation`.
            doms : A sequence of types of the same length as :code:`xs`,
                   default is :code:`Nat(len(xs))`.
        """

        types = Nat(len(xs)) if doms is None else doms
        size = len(types)
        nat = types if isinstance(types, Nat) else next(iter(types), None)
        unit = type(nat)() if isinstance(nat, Nat)\
            else cls.ob() if nat is None else nat.id(nat.dom)
        tensor = lambda tys: unit.tensor(*tys)
        dom = tensor(types)

        xs = finset.Permutation(xs, size)
        if xs.is_identity:
            return cls.id(dom)
        i = xs[0]
        left, head, right = (
            types[slice]
            for slice in (
                slice(0, i), i, slice(i + 1, None)
            )
        )
        rest = (left @ right if isinstance(types, monoidal.Ty)
                else left + right)  # ty: ignore[unsupported-operator]
        return cls.swap(
            tensor(left), head  # ty: ignore[invalid-argument-type]
        ) @ tensor(right) >> head @ cls.permutation(
            [x - 1 if x > i else x for x in xs[1:]], rest)

    @classmethod
    def from_permutation(
            cls, perm: Sequence[int], dom: monoidal.Ty | None = None
    ) -> Diagram:
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
        if cls.permutation_factory.ar is cls:
            return cls.permutation_factory(dom, perm)
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
        :class:`Permutation` is foliated by merging its layers instead, see
        :meth:`merge_layers`.

        >>> x, y = Ty('x'), Ty('y')
        >>> perm = Permutation(x @ y, [1, 0])
        >>> assert perm.foliation() == perm
        """
        if self.is_plumbing:
            return self.merge_layers()
        return super().foliation()

    def merge_layers(self):
        """
        Merge layers as :meth:`~discopy.monoidal.Diagram.merge_layers` does,
        dropping two layers of plumbing alone that compose to the identity,
        which is the empty diagram rather than a layer, see
        :meth:`Permutation.fuse`.

        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> perm = Permutation(x @ y @ z, [1, 2, 0])
        >>> assert (perm >> perm[::-1] >> perm).foliation() == perm
        """
        diagram = super().merge_layers()
        for i, (first, second) in enumerate(
                zip(diagram.inside, diagram.inside[1:])):
            if all(layer.is_generator and layer.is_plumbing
                   for layer in (first, second))\
                    and not first.generator.fuse(second.generator).inside:
                inside = diagram.inside[:i] + diagram.inside[i + 2:]
                return self.ar(inside, self.dom, self.cod).merge_layers()
        return diagram

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

    hypergraph_section = monoidal.Diagram.hypergraph_section


class Box(balanced.Box, Diagram):
    """
    A symmetric box is a balanced box in a symmetric diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """


class Permutation(Box):
    """
    A permutation box, i.e. a :class:`Box` that reorders its input wires.

    A permutation holds a :class:`~discopy.python.finset.Permutation` ``perm``
    as attribute, with the convention that output wire ``i`` comes from input
    wire ``perm[i]``, i.e. ``cod[i] == dom[perm[i]]``.

    A :class:`Layer` stores it as plumbing rather than as a generator, and the
    identity permutation is the identity diagram. The transposition ``[1, 0]``
    on two atomic wires constructs a :class:`Swap`. Other permutations draw as
    a single band of crossing wires rather than a staircase of swaps.

    Parameters:
        dom : The domain, i.e. the wires to permute.
        perm : The permutation as a :class:`finset.Permutation`, a list or
            any iterable, which is read once into a list unless it is a
            sequence.

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
    >>> assert Permutation(x @ y, iter([1, 0])) == Swap(x, y)

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
    serialised_attrs = ("dom", "perm")

    def __new__(cls, dom: monoidal.Ty | None = None,
                perm: Iterable[int] | None = None):
        listed = finset.Permutation(perm, len(dom))\
            if isinstance(dom, monoidal.Ty) and perm is not None\
            and not isinstance(perm, Sequence | monoidal.Ty) else None
        perm = perm if listed is None else listed
        factory = cls.ar.swap_factory\
            if dom is not None and cls is not cls.ar.swap_factory\
            and isinstance(perm, Sequence)\
            and finset.Permutation.is_swap(perm) else cls
        result = super(Permutation, factory).__new__(factory)
        if listed is not None:
            result.perm = listed
        return result

    def __init__(self, dom: monoidal.Ty, perm: Iterable[int]):
        self.perm = finset.Permutation(
            perm if isinstance(perm, Sequence) else self.perm, len(dom))
        cod = dom[:0].tensor(*(dom[i] for i in self.perm))
        name = f"Permutation({list(self.perm)})"
        super().__init__(
            name, dom, cod, drawing_name=name,
            draw_as_wires=True, draw_as_permutation=True,
            permutation_indices=tuple(self.perm))

    @classmethod
    # pylint: disable-next=arguments-differ  # drawn with no boundary
    def strategy(cls, *, types=None):
        """
        Generate a permutation of two or three atomic types, other than the
        identity, which is no box.

        Parameters:
            types : The strategy to draw the types from, see
                :meth:`atomic_strategy`.
        """
        from hypothesis import strategies as st

        def permutations(dom):
            return st.permutations(range(len(dom))).map(
                lambda perm: cls(dom, perm))

        atoms = st.lists(cls.atomic_strategy(types), min_size=2, max_size=3)
        return atoms.map(lambda xs: xs[0].tensor(*xs[1:])).flatmap(
            permutations).filter(lambda perm: not perm.is_identity)

    def __getnewargs__(self):
        """
        The arguments of :meth:`__new__`, which a pickle reads.

        >>> x, y, z = map(Ty, "xyz")
        >>> perm = Permutation(x @ y @ z, [2, 0, 1])
        >>> assert Permutation(*perm.__getnewargs__()) == perm
        """
        return self.dom, list(self.perm)

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
        return factory_key(type(self)), self.dom, tuple(self.perm)

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
        Serialise a permutation, see :func:`~discopy.utils.dumps`.

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

    def fuse(self, other: Permutation) -> Diagram:
        """
        The composite with another permutation as one permutation, i.e. the
        identity diagram when they are inverse.

        Parameters:
            other : The permutation to compose with.

        >>> x, y = Ty('x'), Ty('y')
        >>> assert Swap(x, y).fuse(Swap(y, x)) == Id(x @ y)
        """
        assert_iscomposable(self, other)
        return self.ar.from_permutation(
            [self.perm[i] for i in other.perm], self.dom)

    def tensor(self, *others):
        """
        Tensoring plumbing with plumbing gives one wider permutation: a run
        of permutations and types coalesces into the permutation on the
        disjoint union of their underlying :class:`.finset.Permutation`.

        A generator interrupts the run, from which point the layers are
        those of an ordinary :meth:`.monoidal.Diagram.tensor`.

        >>> x = Ty('x')
        >>> perm = Permutation(x @ x, [1, 0])
        >>> assert perm.tensor(x, perm) == Permutation(
        ...     x ** 5, [1, 0, 2, 4, 3])
        """
        perms, dom, plumbing = [self.perm], self.dom, 0
        while plumbing < len(others) and (
                isinstance(others[plumbing], (Permutation, monoidal.Ty))
                or isinstance(others[plumbing], Diagram)
                and not others[plumbing].inside):
            other, plumbing = others[plumbing], plumbing + 1
            if isinstance(other, Permutation):
                perms.append(other.perm)
                dom = dom @ other.dom
            else:
                typ = other if isinstance(other, monoidal.Ty) else other.dom
                perms.append(finset.Permutation.id(len(typ)))
                dom = dom @ typ
        result = self if len(perms) == 1 else self.permutation_factory(
            dom, perms[0].tensor(*perms[1:]))
        rest = others[plumbing:]
        return result if not rest\
            else super(Permutation, result).tensor(*rest)

    def __rmatmul__(self, other):
        if not isinstance(other, monoidal.Ty):
            return super().__rmatmul__(other)
        perm = finset.Permutation.id(len(other)).tensor(self.perm)
        return self.permutation_factory(other @ self.dom, perm)

    def __repr__(self):
        return f"{factory_name(type(self))}({self.dom!r}, {list(self.perm)})"

    def __str__(self):
        return f"Permutation({self.dom}, {list(self.perm)})"


Layer.plumbing = (monoidal.Ty, Permutation)


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
    serialised_attrs = ("left", "right")

    def __setstate__(self, state):
        state.setdefault('perm', finset.Permutation([1, 0], 2))
        super().__setstate__(state)

    def __init__(self, left, right):
        if not isinstance(right, monoidal.Ty) and len(left) == 2:
            perm = finset.Permutation(
                right if isinstance(right, Sequence) else self.perm, 2)
            if not perm.is_swap():
                raise ValueError(messages.NOT_A_SWAP.format(list(perm)))
            left, right = left[:1], left[1:]
        self.perm = finset.Permutation([1, 0], 2)
        balanced.Braid.__init__(self, left, right)
        Box.__init__(self, self.name, self.dom, self.cod,
                     draw_as_wires=True, draw_as_braid=False)

    @classmethod
    # pylint: disable-next=arguments-differ  # drawn with no boundary
    def strategy(cls, *, types=None):
        """
        Generate the swap of two atomic types.

        Parameters:
            types : The strategy to draw the types from, see
                :meth:`atomic_strategy`.
        """
        from hypothesis import strategies as st

        atoms = cls.atomic_strategy(types)
        return st.builds(cls, atoms, atoms)

    def dagger(self):
        return type(self)(self.right, self.left)

    def __getnewargs__(self):
        return self.left, self.right

    def to_swaps(self):
        return self

    def to_drawing(self):
        return Box.to_drawing(self)

    def __repr__(self):
        return balanced.Braid.__repr__(self)

    def __str__(self):
        return self.name


class Sum(balanced.Sum, Box):
    """
    A symmetric sum is a balanced sum and a symmetric box.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(balanced.Functor):
    """
    A symmetric functor is a monoidal functor that preserves swaps.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.

    Note
    ----
    A :class:`Permutation` is sent to the permutation of the images of its
    wires, one block per wire: the codomain's ``from_permutation`` when it
    has a ``permutation_factory``, i.e. a native permutation box, so that the
    identity functor is the identity, and its ``permutation`` otherwise. Into
    a codomain with neither, it is sent to the images of the swaps of
    :meth:`Permutation.to_swaps`, so that a whiskered swap maps like a swap.

    A :class:`Swap` is a permutation of two wires and goes the same way, so
    that a functor commutes with whiskering on the nose even when it sends an
    atomic type to a longer one.

    >>> x, y, z = map(Ty, "xyz")
    >>> perm = Permutation(x @ y @ z, [1, 2, 0])
    >>> assert Functor.id(Diagram)(perm) == perm
    >>> assert Functor.id(Diagram)(Swap(x, y)) == Swap(x, y)
    >>> G = Functor(lambda ob: ob @ ob, lambda box: box)
    >>> assert G(Swap(x, y) @ z) == G(Swap(x, y)) @ G(z)
    >>> F = Functor({x: x, y: y, z: z}, {}, cod=balanced.Diagram)
    >>> print(F(Swap(x, y) @ z))
    Braid(x, y) @ z
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Permutation) and issubclass(self.cod.ar, Diagram):
            sizes = [len(self(x)) for x in other.dom]
            return self.cod.ar.from_permutation(finset.Function.permutation(
                other.perm, sizes).inside, self(other.dom))
        if isinstance(other, Permutation) and issubclass(
                self.cod.ar, SymmetricCategory):
            return self.cod.ar.permutation(
                other.perm, list(map(self, other.dom)))
        if isinstance(other, Permutation) and not isinstance(other, Swap):
            swaps, dom = other.to_swaps(), other.dom
            if len(swaps) != 1:
                return self(swaps)
            i = next(i for i, j in enumerate(other.perm) if i != j)
            swap = other.swap_factory(dom[i:i + 1], dom[i + 1:i + 2])
            return self(monoidal.Layer(dom[:i], swap, dom[i + 2:]))
        return super().__call__(other)


class DualRail(balanced.DualRail, Functor):
    """
    The functor sending a symmetric diagram to its dual rail encoding, i.e.
    :class:`~discopy.balanced.DualRail` sending a :class:`Permutation` to the
    dual rail braids of its swaps.

    >>> x, y, z = map(Ty, "xyz")
    >>> diagram = (Swap(x, y) @ z).to_braided()
    >>> assert [type(box) for box in diagram.boxes] == [balanced.DualRailBraid]
    """
    def __call__(self, other):
        if isinstance(other, Permutation) and not isinstance(other, Swap):
            return Functor.__call__(self, other)
        return super().__call__(other)


CMap = cmap.CMap[Diagram]

for level in (
        BiclosedCategory, RigidCategory, PivotalCategory, CompactCategory):
    for name, law in vars(level).items():
        if isinstance(law, (Axiom, Rule)):
            setattr(CMap, name, law.inapplicable(messages.MAP_HAS_NO_ADJOINTS))

Diagram.box_factory = Box
Diagram.functor_factory = Functor
Diagram.dual_rail_factory = DualRail
Hypergraph = hypergraph.Hypergraph[Diagram]
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.sum_factory = Sum
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


Diagram.scalar_sliding = SpacialCategory.scalar_sliding.modulo(
    Diagram.to_hypergraph)


class TermBase(balanced.TermBase):  # ty: ignore[unsupported-base]
    """
    A term of a symmetric category, i.e. of the linear kappa calculus: a
    :class:`~discopy.monoidal.TermBase` with exchange, where a :class:`Kappa`
    binds its variables in any order, from anywhere in the context.

    A variable still occurs once and the context of a term is still in the
    order of the occurrences: a kappa is where the wires cross, so that it
    evaluates to a permutation followed by its body, and a term to a
    diagram up to the axioms of symmetric categories, see :class:`Equation`.

    Example
    -------
    >>> x, y, z = map(Ty, "xyz")
    >>> f = Box('f', x @ y, z)
    >>> term = (Swap(y, x) >> f).to_term()
    >>> print(term)
    (y @ x)(lambda x0, x1: f(x1, x0))
    >>> assert term.eval() == Swap(y, x) >> f

    A diagram reads box for box up to its swaps, which a term does not
    remember but as the order of its variables:

    >>> g = Box('g', x, y)
    >>> diagram = Swap(x, x) >> x @ g >> Swap(x, y) >> Swap(y, x) >> f
    >>> print(diagram.to_term())
    (x @ x)(lambda x0, x1: let(g(x0), lambda x2: f(x1, x2)))
    >>> assert diagram.to_term().eval() == Swap(x, x) >> x @ g >> f
    >>> assert diagram.to_term().eval() != diagram
    >>> assert Equation(diagram.to_term().eval(), diagram)
    >>> print(diagram.to_term().normal_form())
    (x @ x)(lambda x0, x1: f(x1, g(x0)))
    """
    strategy = classmethod(balanced.TermBase.strategy.__func__)

    @classmethod
    def rewire(cls, box, arguments):
        """ A permutation is read as its inputs, in the order it gives. """
        return [arguments[i] for i in box.perm]\
            if isinstance(box, Permutation) else None

    @staticmethod
    def wire(functor, wiring: list[int], dom: Ty):
        """
        The image under a functor of the wiring from a type to its wires
        at the given indices: that of their permutation in its domain, the
        identity when nothing moves, so that a term evaluates wherever its
        diagram does, e.g. to braids in a braided category.

        Parameters:
            functor : The functor to evaluate the permutation.
            wiring : The index of the input each output reads.
            dom : The type of the inputs.
        """
        return functor(functor.dom.from_permutation(wiring, dom))

    @axiom
    def normal_form_soundness(cls, term: Self) -> AbstractEquation:
        """
        Beta reduction is sound: a term and its normal form evaluate to
        diagrams equal up to the axioms of symmetric categories.
        """
        return Equation(term.eval(), term.normal_form().eval())

    @axiom
    def reading(cls, term: Self) -> AbstractEquation:
        """
        Reading is a section of evaluation up to the axioms of symmetric
        categories: the diagram of a term reads as a term that evaluates
        back to it, box for box but for its swaps.
        """
        diagram = term.eval()
        return Equation(diagram.to_term().eval(), diagram)


class Variable(TermBase, balanced.Variable):  # ty: ignore[unsupported-base]
    """
    A variable with a name and a type of one wire, an identity diagram.

    Parameters:
        name : The name of the variable.
        cod : The type of the variable.
    """


class Constant(TermBase, balanced.Constant):  # ty: ignore[unsupported-base]
    """
    A box as a term, with its domain as stack.

    Parameters:
        box : The box.
    """


class Application(
        TermBase, balanced.Application):  # ty: ignore[unsupported-base]
    """
    The application of a term to an argument with no stack, as in a
    monoidal category: the context of the argument goes where the first
    inputs of the stack were.

    Parameters:
        func : The term that is applied.
        arg : The argument, a term with no stack.
    """


class Kappa(TermBase, balanced.Kappa):  # ty: ignore[unsupported-base]
    """
    The abstraction of some of the free ``variables`` of the ``body``, in
    any order and from anywhere in its context: they join its stack, which
    stands where the stack of the body does among the variables left, or
    else where the first of the variables did.

    Parameters:
        variables : The variables to bind, in the order of the stack.
        body : The term in which they are bound.
        offset : The number of free variables left of the stack, needed
            when nothing says where it is, no variable and no stack in the
            body: the place for a box with no output.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, g = Constant(Box('f', x @ y, x)), Constant(Box('g', x @ x @ y, x))
    >>> a, b, c = Variable('a', x), Variable('b', y), Variable('c', x)
    >>> term = Kappa((b, a), f(a, b))
    >>> assert term == (y @ x)(lambda b, a: f(a, b))
    >>> assert term.stack == y @ x and term.eval() == Swap(y, x) >> f.box
    >>> term = Kappa((c, a), g(a, c, b))
    >>> assert term.freevars == [b] and term.offset == 0
    >>> assert term.eval()\\
    ...     == Diagram.permutation([1, 0, 2], x @ x @ y) >> g.box
    >>> Kappa((a, a), f(a, b))
    Traceback (most recent call last):
        ...
    ValueError: Expected distinct free variables of f(a, b), got a, a.
    """
    unbound = messages.NOT_FREE

    @staticmethod
    def place(variables, body):
        """
        The number of free variables left of the stack: those that are not
        bound and stand left of the stack of the body, or else left of the
        first of the variables it uses, ``None`` when there is neither.
        """
        used = [x for x in variables if x in body.freevars]
        if not body.stack and not used:
            return None
        anchor = body.offset if body.stack else body.freevars.index(used[0])
        return len(set(body.freevars[:anchor]).difference(variables))

    def __init__(self, variables: tuple[Variable, ...], body: cat.TermBase,
                 offset: int | None = None):
        # pylint: disable=super-init-not-called  # no block of variables
        assert_isinstance(body, cat.TermBase)
        self.variables, self.body = tuple(variables), body
        self.offset = self.start(self.variables, body, offset)
        self.freevars = [x for x in body.freevars if x not in self.variables]
        self.cod, bound = body.cod, body.types(self.variables)
        self.stack = bound @ body.stack if self.front else body.stack @ bound

    def eval(self, functor=None):
        functor, body = self.identity(functor), self.body
        stack = list(range(len(body.stack)))
        bound = list(self.variables)
        left, _, right = body.context
        source = self.freevars[:self.offset] + (
            bound + stack if self.front else stack + bound
        ) + self.freevars[self.offset:]
        wiring = [source.index(wire) for wire in left + stack + right]
        return self.wire(functor, wiring, self.dom) >> body.eval(functor)


type Term = Variable | Constant | Application | Kappa


@dataclass
class Sampler(monoidal.Sampler):
    """
    Samples a term of a symmetric category as a
    :class:`~discopy.monoidal.Sampler` does, each subterm using its
    variables in an order of its own.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> sampler = Sampler(TermBase, iter([0, 1]), [x], "a")
    >>> print(sampler.state(y, (Variable('v', x), Variable('w', x))))
    c0(w, v)
    """
    def shuffle(self, bound):
        """ A rotation of the variables, picked by a choice. """
        start = self.choose(range(len(bound))) if len(bound) > 1 else 0
        return bound[start:] + bound[:start]


TermBase.functor, TermBase.sampler_factory = Functor.id(Diagram), Sampler
TermBase.variable_factory, TermBase.constant_factory = Variable, Constant
TermBase.application_factory, TermBase.kappa_factory = Application, Kappa
Diagram.term_factory, monoidal.Kappa.above = TermBase, Kappa


Bubble = Box.from_args
