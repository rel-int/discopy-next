# -*- coding: utf-8 -*-

"""
The free Markov category, i.e. a semicartesian category with a supply of
commutative comonoid, see :cite:t:`FritzLiang23`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Swap
    Permutation
    Copy
    Functor


Axioms
------

>>> x = Ty('x')

>>> copy, merge = Copy(x), Merge(x)
>>> unit, delete = Merge(x, n=0), Copy(x, n=0)

Commutative monoid
==================

>>> unitality = Equation(unit @ x >> merge, Id(x), x @ unit >> merge)
>>> associativity = Equation(merge @ x >> merge, x @ merge >> merge)
>>> commutativity = Equation(Swap(x, x) >> merge, merge)
>>> assert unitality and associativity and commutativity
>>> Equation(unitality, associativity, commutativity, symbol='').draw(
...     doctest="docs/_static/frobenius/monoid.svg")

.. image:: /_static/frobenius/monoid.svg
    :align: center

Cocommutative comonoid
======================

>>> counitality = Equation(copy >> delete @ x, Id(x), copy >> x @ delete)
>>> coassociativity = Equation(copy >> copy @ x, copy >> x @ copy)
>>> cocommutativity = Equation(copy >> Swap(x, x), copy)
>>> assert counitality and coassociativity and cocommutativity
>>> Equation(counitality, coassociativity, cocommutativity, symbol='').draw(
...     doctest="docs/_static/frobenius/comonoid.svg")

.. image:: /_static/frobenius/comonoid.svg
    :align: center

Coherence
=========

>>> assert Equation(Diagram.copy(x @ x, n=0), delete @ delete)
>>> assert Equation(Diagram.copy(x @ x),
...     copy @ copy >> x @ Swap(x, x) @ x)
>>> assert Equation(Diagram.merge(x @ x, n=0), unit @ unit)
>>> assert Equation(Diagram.merge(x @ x),
...     x @ Swap(x, x) @ x >> merge @ merge)

Note
----
Equality of Markov diagrams is computed by translation to hypergraph.
Both copy and merge boxes are translated to spiders, thus when they appear
in the same diagram they automatically satisfy the :mod:`frobenius` axioms.
"""


from typing import TYPE_CHECKING, ClassVar, Self

from discopy import axioms, symmetric, monoidal, cmap, hypergraph, messages
from discopy.abc import HypergraphCategory, MarkovCategory
from discopy.axioms import FUEL, Testable, axiom
from discopy.cat import factory
from discopy.monoidal import Ty  # noqa: F401  pylint: disable=unused-import
from discopy.utils import (
    NamedGeneric, assert_isatomic, factory_name, from_tree)

if TYPE_CHECKING:
    from hypothesis import strategies as st


Layer = symmetric.Layer


class Square[factory](Testable, NamedGeneric, tuple):
    """
    A commuting square ``f >> B @ h == A @ h >> g`` of the ``factory``, a
    traced Markov category that is not a hypergraph one, along a strict map
    ``h : M -> M ** n``, the ``n``-fold copy of an atomic memory ``M`` not
    in ``A`` or ``B``, as the triple ``(f, h, g)``, so that the square
    commutes by construction.

    The template ``f : A @ M -> B @ M`` is drawn acyclic and ``g`` is its
    :meth:`double`: its image under the functor sending ``M`` to ``M ** n``
    and each box to itself, reading one copy of each memory wire it reads
    and copying each memory wire it writes. The square of each box commutes
    by the counit of the copy, and squares paste. No box or boundary type
    contains the memory but as one of its wires, as the functor would
    rewrite it, e.g. inside an exponential.

    Example
    -------
    >>> from hypothesis import find
    >>> f, h, g = find(Square[Diagram].strategy(), lambda s: s[0].boxes)
    >>> A, B = f.dom[:-1], f.cod[:-1]
    >>> assert Diagram.equation_factory(f >> B @ h, A @ h >> g)
    >>> assert h.is_strict
    """
    def __new__(cls, square):
        square = tuple(square)
        if len(square) != 3:
            raise ValueError("Expected a morphism, a strict map and another.")
        return super().__new__(cls, square)

    @classmethod
    def double(cls, mem: monoidal.Ty, n: int, reads=None):
        """
        The functor sending the memory ``mem`` to ``mem ** n`` and each box
        to itself, reading the copies ``reads[box]`` of the memory wires
        of its domain, the first by default, and copying the memory wires
        of its codomain.

        Parameters:
            mem : The memory, an atomic type.
            n : The number of copies.
            reads : The copy each box reads of each wire of its domain.
        """
        category, reads = cls.factory, reads or {}

        def read(box):
            indices = reads.get(box, [0] * len(box.dom))
            return category.id().tensor(*(
                category.discard(mem ** i) @ mem
                @ category.discard(mem ** (n - 1 - i)) if x == mem
                else category.id(x) for x, i in zip(box.dom, indices)))

        def write(box):
            return category.id().tensor(*(
                category.copy(mem, n) if x == mem else category.id(x)
                for x in box.cod))

        return category.functor_factory(
            ob_map=lambda x: mem ** n if x == mem else x,
            ar_map=lambda box: read(box) >> box >> write(box), cod=category)

    @classmethod
    def canonical(cls) -> "Square":
        """
        The square of a box ``f : A @ M -> B @ M`` along ``copy(M)``.
        """
        category = cls.factory
        A, B, M = map(category.ob, "ABM")
        f = category.box_factory("f", A @ M, B @ M)
        return cls((f, category.copy(M), cls.double(M, 2)(f)))

    @classmethod
    def strategy(cls, *, written=False, **params
                 ) -> "st.SearchStrategy[Square]":
        """
        Generate a commuting square along the copy of a memory.

        Parameters:
            written : Whether the memory ``f`` outputs is written by a box,
                ``f`` ending with a writer from some wires to the memory, so
                that every loop of ``f.trace()`` has a producer.
            params : Passed to :meth:`discopy.hypergraph.Hypergraph.strategy`
                for the template.
        """
        from hypothesis import strategies as st

        category = cls.factory
        if issubclass(category, HypergraphCategory):
            raise NotImplementedError(messages.NO_SQUARE)

        @st.composite
        def squares(draw):
            mem, n = draw(category.ob.generators()), draw(st.integers(2, 3))
            reads = {}
            functor = cls.double(mem, n, reads)
            types = category.ob.strategy().filter(lambda ty: all(
                x == mem or functor(x) == x for x in ty))
            boundaries = types.filter(lambda ty: all(x != mem for x in ty))

            def diagrams(dom, cod):
                return hypergraph.Hypergraph[category].strategy(
                    types=types, dom=dom, cod=cod, acyclic=True,
                    max_scalars=0, **params
                ).map(lambda graph: graph.to_diagram())

            dom, cod = draw(boundaries), draw(boundaries)
            if written:
                wires = draw(boundaries)
                f = draw(diagrams(dom @ mem, cod @ wires))\
                    >> cod @ draw(diagrams(wires, mem))
            else:
                f = draw(diagrams(dom @ mem, cod @ mem))
            reads.update({box: draw(st.lists(
                st.integers(0, n - 1), min_size=len(box.dom),
                max_size=len(box.dom))) for box in f.boxes})
            return cls((f, category.copy(mem, n), functor(f)))
        return squares()


@factory
class Diagram(symmetric.Diagram, MarkovCategory):
    """
    A Markov diagram is a symmetric diagram with :class:`Copy` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.

    Note
    ----
    We can create arbitrary Markov diagrams with the standard notation for
    Python functions.

    >>> x = Ty('x')
    >>> f = Box('f', x, x)

    >>> copy_then_apply = Diagram.from_callable(x, x @ x)(
    ...     lambda x: (f(x), f(x)))

    >>> @Diagram.from_callable(x, x @ x)
    ... def apply_then_copy(x):
    ...     y = f(x)
    ...     return y, y

    >>> Equation(copy_then_apply, apply_then_copy, symbol="$\\\\neq$").draw(
    ...     doctest="docs/_static/markov/copy_and_apply.svg")

    .. image:: /_static/markov/copy_and_apply.svg
    """
    box_factory: ClassVar[type["Box"]]
    functor_factory: ClassVar[type["Functor"]]
    copy_factory: ClassVar[type["Copy"]]
    merge_factory: ClassVar[type["Merge"]]
    swap_factory: ClassVar[type["Swap"]]
    permutation_factory: ClassVar[type["Permutation"]]
    trace_factory: ClassVar[type["Trace"]]
    discard_factory: ClassVar[type["Discard"]]
    sum_factory: ClassVar[type["Sum"]]

    @classmethod
    def sample(cls, *, dom=None, cod=None, fuel=FUEL):
        """
        Sample Markov diagrams through their quotient: a hypergraph drawn
        by :meth:`discopy.hypergraph.Hypergraph.strategy` with at most
        ``fuel + 1`` boxes, besides the one balancing box that may close its
        boundary, and the spiders that the category supports, i.e.
        copies and discards here and every spider in a hypergraph category,
        downgraded with :meth:`discopy.hypergraph.Hypergraph.to_diagram`.
        """
        category = hypergraph.Hypergraph[cls]  # ty: ignore[invalid-type-form]
        return category.strategy(
            dom=dom, cod=cod, max_leaves=fuel + 1).map(
                lambda graph: graph.to_diagram())

    @classmethod
    def spider_factory(cls, n_legs_in, n_legs_out, typ, phase=None):
        if phase is not None or 1 not in (n_legs_in, n_legs_out):
            raise ValueError
        return cls.copy_factory(typ, n_legs_out) if n_legs_in == 1\
            else cls.merge_factory(typ, n_legs_in)

    @classmethod
    def copy(cls, x: monoidal.Ty, n=2) -> "Diagram":
        """
        Make :code:`n` copies of a given type :code:`x`.

        Parameters:
            x : The type to copy.
            n : The number of copies.
        """
        from discopy import frobenius
        return frobenius.Diagram.spiders.__func__(
            cls, 1, n, x)  # ty: ignore[invalid-argument-type]

    @classmethod
    def merge(cls, x: monoidal.Ty, n=2) -> "Diagram":
        """
        Merge :code:`n` copies of a given type :code:`x`.

        Parameters:
            x : The type to copy.
            n : The number of copies.
        """
        return cls.copy(x, n).dagger()

    @classmethod
    def discard(cls, x: monoidal.Ty) -> "Diagram":
        """
        The discard of an atomic type :code:`x`.

        Parameters:
            x : The type to discard.
        """
        return cls.copy(x, 0)

    map_agreement = monoidal.Diagram.map_agreement.failing(
        messages.MAP_HAS_NO_SPIDERS)

    @axiom
    def trace_uniformity(cls, square: Square[Self]) -> axioms.Equation:
        """
        The trace is uniform along a strict map: if ``f >> B @ h`` equals
        ``A @ h >> g`` then ``f.trace()`` equals ``g.trace()``, the premise
        holding by construction of the :class:`Square`, as composability
        does by construction of the premises of associativity.
        """
        f, h, g = square
        return cls.equation_factory(
            f.trace(len(h.dom)), g.trace(len(h.cod)))

    trace_uniformity_written = trace_uniformity.weaken(written=True)
    trace_uniformity = trace_uniformity.failing(messages.NOT_UNIFORM)


class Box(symmetric.Box, Diagram):
    """
    A Markov box is a symmetric box in a Markov diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """


class Permutation(symmetric.Permutation, Box):
    """
    A permutation in a Markov category.

    Parameters:
        dom (monoidal.Ty) : The domain, i.e. the wires to permute.
        perm : The permutation as a :class:`finset.Permutation` or a list.
    """


class Swap(Permutation, symmetric.Swap, Box):
    """
    Symmetric swap in a Markov diagram.

    Parameters:
        left (monoidal.Ty) : The type on the top left and bottom right.
        right (monoidal.Ty) : The type on the top right and bottom left.
    """


class Trace(symmetric.Trace, Box):
    """
    A trace in a Markov category.

    Parameters:
        arg : The diagram to trace.
        left : Whether to trace the wires on the left or right.

    See also
    --------
    :meth:`Diagram.trace`
    """


class Copy(Box):
    """
    The copy of an atomic type :code:`x` some :code:`n` number of times.

    Parameters:
        x : The type to copy.
        n : The number of copies.
    """
    is_strict = True

    def __init__(self, x: monoidal.Ty, n: int = 2):
        assert_isatomic(x, monoidal.Ty)
        name = f"Copy({x}" + ("" if n == 2 else f", {n}") + ")"
        Box.__init__(self, name, dom=x, cod=x ** n,
                     draw_as_spider=True, color="black", drawing_name="")

    def __new__(cls, x: monoidal.Ty, n: int = 2):
        return super().__new__(cls) if n else\
            cls.discard_factory.__new__(cls.discard_factory, x)

    def __getnewargs__(self):
        """ The arguments of :meth:`__new__`, which a pickle needs. """
        return self.dom, len(self.cod)

    def dagger(self) -> "Merge":
        """
        The merge of the same type, in the category of the copy.

        >>> x = Ty('x')
        >>> assert Copy(x).dagger() == Merge(x)
        >>> assert Diagram.discard(x).dagger().dagger() == Discard(x)
        """
        return self.merge_factory(self.dom, len(self.cod))

    def __repr__(self):
        return (
            factory_name(type(self)) + f"({repr(self.dom)}, {len(self.cod)})")

    def to_tree(self) -> dict:
        """
        Serialise a copy by its type and number of copies.

        >>> from discopy.utils import dumps, loads
        >>> x = Ty('x')
        >>> for box in (Copy(x), Copy(x, 3), Discard(x), Merge(x)):
        ...     assert loads(dumps(box)) == box
        """
        return {'factory': factory_name(type(self)),
                'x': self.dom.to_tree(), 'n': len(self.cod)}

    @classmethod
    def from_tree(cls, tree: dict) -> "Copy":
        return cls(from_tree(tree['x']), tree['n'])


class Merge(Box):
    """
    The merge of an atomic type :code:`x` some :code:`n` number of times.

    Parameters:
        x : The type of wires to merge.
        n : The number of wires to merge.
    """
    is_strict = True

    def __init__(self, x: monoidal.Ty, n: int = 2):
        assert_isatomic(x, monoidal.Ty)
        name = f"Merge({x}" + ("" if n == 2 else f", {n}") + ")"
        Box.__init__(self, name, dom=x ** n, cod=x,
                     draw_as_spider=True, color="black", drawing_name="")

    def dagger(self) -> "Copy":
        """ The copy of the same type, in the category of the merge. """
        return self.copy_factory(self.cod, len(self.dom))

    def __repr__(self):
        return (
            factory_name(type(self)) + f"({repr(self.cod)}, {len(self.dom)})")

    def to_tree(self) -> dict:
        """ Serialise a merge by its type and number of wires merged. """
        return {'factory': factory_name(type(self)),
                'x': self.cod.to_tree(), 'n': len(self.dom)}

    @classmethod
    def from_tree(cls, tree: dict) -> "Merge":
        return cls(from_tree(tree['x']), tree['n'])


class Discard(Copy):
    """
    The discard of an atomic type :code:`x`.

    Parameters:
        x : The type to discard.
    """
    def __init__(self, x: monoidal.Ty, n: int = 0):
        if n:
            raise ValueError(f"A discard has no copies, got n={n}.")
        super().__init__(x, 0)


class Sum(symmetric.Sum, Box):
    """
    A markov sum is a symmetric sum and a markov box.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(symmetric.Functor):
    """
    A Markov functor is a symmetric functor that preserves copies.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.

    Example
    -------

    We build a functor into python functions.

    >>> x = Ty('x')
    >>> add = Box('add', x @ x, x)
    >>> from discopy import python
    >>> F = Functor({x: int}, {add: lambda a, b: a + b},
    ...             cod=python.Function)
    >>> copy = Copy(x)
    >>> bialgebra_l = copy @ copy >> Id(x) @ Swap(x, x) @ Id(x) >> add @ add
    >>> bialgebra_r = add >> copy
    >>> assert F(bialgebra_l)(54, 46) == F(bialgebra_r)(54, 46)

    >>> Equation(bialgebra_l, bialgebra_r, symbol="=").draw(
    ...     doctest="docs/_static/markov/bialgebra.svg")

    .. image:: /_static/markov/bialgebra.svg
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Copy) and hasattr(self.cod, "copy"):
            return self.cod.copy(self(other.dom), len(other.cod))
        if isinstance(other, Merge) and hasattr(self.cod, "merge"):
            return self.cod.merge(self(other.cod), len(other.dom))
        return super().__call__(other)


CMap = cmap.CMap[Diagram]

Diagram.box_factory = Box
Diagram.functor_factory = Functor
Hypergraph = hypergraph.Hypergraph[Diagram]
Diagram.copy_factory, Diagram.merge_factory = Copy, Merge
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.trace_factory = Trace
Diagram.discard_factory = Discard
Diagram.sum_factory = Sum
Id = Diagram.id


class Equation(symmetric.Equation):
    """ The :class:`symmetric.Equation` of Markov diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)
