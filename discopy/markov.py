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
    Feedback
    Trace
    FollowedBy
    Head
    Tail
    Functor


Axioms
------

>>> x = Ty('x')

>>> copy, delete = Copy(x), Copy(x, n=0)

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

Note
----
Equality of Markov diagrams is computed by translation to hypergraph.
Copy boxes are translated to spiders, thus when they appear in the same
diagram as the merge boxes of :mod:`comarkov` they automatically satisfy
the :mod:`frobenius` axioms.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, ClassVar, Self

from discopy import (
    axioms, symmetric, monoidal, feedback, cmap, hypergraph, messages)
from discopy.abc import ComarkovCategory, HypergraphCategory, MarkovCategory
from discopy.axioms import Testable, axiom
from discopy.cat import factory
from discopy.feedback import (  # noqa: F401  pylint: disable=unused-import
    Wire, Ty)
from discopy.utils import (
    AxiomError, NamedGeneric, assert_isatomic, deprecated_alias,
    factory_name, from_tree)

if TYPE_CHECKING:
    from hypothesis import strategies as st


Layer = symmetric.Layer


class Square(Testable, NamedGeneric["factory"], tuple):
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
    >>> assert Equation(f >> B @ h, A @ h >> g)
    >>> assert h.is_strict
    """
    def __new__(cls, square):
        square = tuple(square)
        if len(square) != 3:
            raise ValueError("Expected a morphism, a strict map and another.")
        if any(box.is_dagger for box in square[0].boxes):
            raise ValueError("A template with a dagger box has no double.")
        return super().__new__(cls, square)

    @classmethod
    def double(cls, mem: monoidal.Ty, n: int, reads=None):
        """
        The functor sending the memory ``mem`` to ``mem ** n`` and each box
        to itself, reading the copies ``reads[box]`` of the memory wires
        of its domain, the first by default, and copying the memory wires
        of its codomain. A dagger box goes to the dagger of its double,
        which is not its double, so :class:`Square` refuses one.

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
    def strategy(cls, *, written=False, **params
                 ) -> st.SearchStrategy[Square]:
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
            atoms = category.ob.generator_factory.strategy().map(category.ob)
            mem, n = draw(atoms), draw(st.integers(2, 3))
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

    Note
    ----
    A Markov diagram is a feedback diagram whose wires carry a time step, so
    that it is the syntax of streams, see :mod:`discopy.stream`.

    >>> x = Ty('x')
    >>> zero = Box('0', Ty(), x.head)
    >>> rand = Box('rand', Ty(), x)
    >>> plus = Box('+', x @ x, x)
    >>> walk = (rand.delay() @ x.delay() >> zero @ plus.delay()
    ...         >> FollowedBy(x) >> Copy(x)).feedback()
    >>> walk.draw(doctest="docs/_static/feedback/feedback-random-walk.svg")

    .. image:: /_static/feedback/feedback-random-walk.svg
        :align: center
    """
    copy_factory: ClassVar[type["Copy"]]
    discard_factory: ClassVar[type["Discard"]]

    @classmethod
    def spider_factory(cls, n_legs_in, n_legs_out, typ, phase=None):
        if phase is not None or n_legs_in != 1:
            raise ValueError(messages.NO_SUCH_SPIDER.format(
                factory_name(cls), "copies"))
        return cls.copy_factory(typ, n_legs_out)

    @classmethod
    def copy(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        Make :code:`n` copies of a given type :code:`x`.

        Parameters:
            x : The type to copy.
            n : The number of copies.
        """
        from discopy import frobenius
        return frobenius.Diagram.spiders.__func__(cls, 1, n, x)

    @classmethod
    def discard(cls, x: monoidal.Ty) -> Diagram:
        """
        The discard of a type :code:`x`, i.e. no copy of it at all.

        Parameters:
            x : The type to discard.
        """
        return cls.copy(x, 0)

    @classmethod
    def merge(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        Deprecated: a Markov diagram has no merge, see
        :meth:`discopy.comarkov.Diagram.merge`, which a diagram that is
        also comarkov, e.g. :class:`discopy.frobenius.Diagram`, inherits.

        Parameters:
            x : The type to merge.
            n : The number of copies.
        """
        if issubclass(cls, ComarkovCategory):
            return super().merge(x, n)
        warnings.warn(
            f"{factory_name(cls)}.merge is deprecated, "
            "use discopy.comarkov.Diagram.merge instead.",
            DeprecationWarning, stacklevel=2)
        from discopy import comarkov
        return comarkov.Diagram.merge(x, n)

    @property
    def is_deterministic(self) -> bool:
        """
        Whether copying commutes with the diagram, i.e. its boxes are
        deterministic, see :class:`Box`.

        >>> x = Ty('x')
        >>> sample = Box('sample', x, x, is_deterministic=False)
        >>> assert Copy(x).is_deterministic
        >>> assert not (Copy(x) >> sample @ x).is_deterministic
        """
        return all(box.is_deterministic for box in self.boxes)

    @property
    def is_affine(self) -> bool:
        """
        Whether discarding commutes with the diagram, i.e. its boxes are
        affine, see :class:`Box`.

        >>> x = Ty('x')
        >>> condition = Box('condition', x, Ty(), is_affine=False)
        >>> assert Copy(x).is_affine
        >>> assert not (Copy(x) >> condition @ x).is_affine
        """
        return all(box.is_affine for box in self.boxes)

    @property
    def is_cartesian(self) -> bool:
        """
        Whether the diagram is deterministic and affine, i.e. a total
        function: copying and discarding both commute with it.
        """
        return self.is_deterministic and self.is_affine

    @axiom
    def trace_uniformity(cls, square: Square[Self]) -> axioms.Equation:
        """
        The trace is uniform along a strict map: if ``f >> B @ h`` equals
        ``A @ h >> g`` then ``f.trace()`` equals ``g.trace()``, the premise
        holding by construction of the :class:`Square`, as composability
        does by construction of the premises of associativity. Both sides
        are compared up to hypergraph, where the trace is wiring.
        """
        f, h, g = square
        return cls.equation_factory(
            f.trace(len(h.dom)), g.trace(len(h.cod)))

    trace_uniformity = trace_uniformity.modulo(
        symmetric.Diagram.to_hypergraph)
    trace_uniformity_written = trace_uniformity.weaken(written=True)
    trace_uniformity = trace_uniformity.failing(messages.NOT_UNIFORM)


class Box(symmetric.Box, Diagram):
    """
    A Markov box is a symmetric box in a Markov diagram, a total function
    unless it says otherwise.

    A box ``f`` is deterministic when ``f >> copy == copy >> f @ f`` and
    affine when ``f >> discard == discard``, e.g. a sample is affine but not
    deterministic, a condition deterministic but not affine. A deterministic
    box that is not affine is a hard constraint, i.e. a partial function: a
    weight ``w`` copied is ``w * w``, so it is zero or one, and a soft score
    is not deterministic. Both flags are attributes of each box rather than
    of its class, as an identity or a swap is a total function whatever its
    class, and they are part of its equality. They are declared of the box
    as ``deterministic`` and ``affine`` and hold of it but not of its dagger,
    as the converse of a total function need not be one, so that the dagger
    of a dagger is the box again.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
        is_deterministic : Whether copying commutes with the box.
        is_affine : Whether discarding commutes with the box.

    Example
    -------
    >>> p, x = Ty('p'), Ty('x')
    >>> sample = Box('sample', p, x, is_deterministic=False)
    >>> assert sample != Box('sample', p, x)
    >>> sample
    markov.Box('sample', feedback.Ty(feedback.Wire('p')), \
feedback.Ty(feedback.Wire('x')), is_deterministic=False)
    >>> assert not sample.dagger().dagger().is_deterministic

    >>> f = Box('f', p, x)
    >>> assert f.is_cartesian and not f.dagger().is_cartesian
    >>> assert f.dagger().dagger() == f and f.dagger().deterministic
    """
    deterministic = affine = True

    def __init__(self, *args, is_deterministic=True, is_affine=True,
                 **params):
        self.deterministic, self.affine = is_deterministic, is_affine
        super().__init__(*args, **params)

    @property
    def is_deterministic(self) -> bool:
        """ Whether the box is declared deterministic and is no dagger. """
        return self.deterministic and not self.is_dagger

    @property
    def is_affine(self) -> bool:
        """ Whether the box is declared affine and is no dagger. """
        return self.affine and not self.is_dagger

    def flags(self) -> dict[str, bool]:
        """
        The flags declared of the box that differ from those of a total
        function, by name, as :meth:`__repr__` and :meth:`to_tree` write them
        down.
        """
        return {name: False for name, value in [
            ("is_deterministic", self.deterministic),
            ("is_affine", self.affine)] if not value}

    def __repr__(self):
        flags = "".join(f", {name}=False" for name in self.flags())
        if self.is_dagger or not flags:
            return super().__repr__()
        return super().__repr__()[:-1] + flags + ")"

    def setoid(self):
        """
        Markov boxes are equal when they are equal as :class:`cat.Box` and
        their flags are equal too.
        """
        return super().setoid() + (self.deterministic, self.affine)

    def flagged(self, other: Box) -> Box:
        """
        Give a box built from this one the flags declared of this one, as
        :meth:`dagger`, :meth:`subs` and :meth:`lambdify` do on top of those
        of the class they extend, e.g. one that keeps a winding number, and so
        do the rotations of a :class:`discopy.frobenius.Box`.
        """
        other.deterministic, other.affine = self.deterministic, self.affine
        return other

    def dagger(self) -> Box:
        return self.flagged(super().dagger())

    def subs(self, *args) -> Box:
        """
        Substitute a variable in the data of a box, keeping its flags.

        >>> from sympy.abc import phi  # doctest: +EXTRA
        >>> sample = Box('sample', Ty('x'), Ty('x'), data=phi,
        ...              is_deterministic=False)
        >>> assert not sample.subs(phi, 1).is_deterministic
        >>> assert not sample.lambdify(phi)(1).is_deterministic
        """
        return self.flagged(super().subs(*args))

    def lambdify(self, *symbols, **kwargs) -> Callable:
        function = super().lambdify(*symbols, **kwargs)
        return lambda *xs: self.flagged(function(*xs))

    def to_tree(self) -> dict:
        """
        Serialise a box, with the flags it does not have.

        >>> from discopy.utils import dumps, loads
        >>> box = Box('condition', Ty('x'), Ty(), is_affine=False)
        >>> assert loads(dumps(box)) == box
        """
        return {**super().to_tree(), **self.flags()}

    @classmethod
    def from_tree(cls, tree: dict):
        result = super().from_tree(tree)
        for name in ("is_deterministic", "is_affine"):
            if name in tree:
                setattr(result, name.removeprefix("is_"), tree[name])
        return result


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
        name = f"Copy({x}" + ("" if n == 2 else f", {n}") + ")" if n\
            else f"Discard({x})"
        Box.__init__(self, name, dom=x, cod=x ** n,
                     draw_as_spider=True, color="black", drawing_name="")

    def __new__(cls, x: monoidal.Ty, n: int = 2):
        return super().__new__(cls) if n else\
            cls.discard_factory.__new__(cls.discard_factory, x)

    def __getnewargs__(self):
        """
        The arguments of :meth:`__new__`, which a pickle needs.

        >>> from pickle import dumps, loads
        >>> x = Ty('x')
        >>> assert loads(dumps(Copy(x))) == Copy(x)
        >>> assert loads(dumps(Discard(x))) == Discard(x)
        """
        return self.dom, len(self.cod)

    def dagger(self) -> Diagram:
        """
        The merge of the same type, when the category is also comarkov.

        >>> from discopy import closed
        >>> closed.Copy(closed.Ty('x')).dagger()
        Traceback (most recent call last):
            ...
        discopy.utils.AxiomError: Copy(x) has no dagger: closed.Diagram \
is not a dagger category.
        """
        if not issubclass(self.ar, ComarkovCategory):
            raise AxiomError(messages.NOT_A_DAGGER.format(
                self, factory_name(self.ar)))
        return self.ar.merge_factory(self.dom, len(self.cod))

    def __repr__(self):
        return (
            factory_name(type(self)) + f"({repr(self.dom)}, {len(self.cod)})")

    @classmethod
    def from_tree(cls, tree: dict) -> Copy:
        return cls(from_tree(tree['dom']), len(from_tree(tree['cod'])))


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

    Its flags are read off its terms: copying and discarding are linear, so
    that a sum is deterministic when it has at most one term and that term
    is, and affine when it has exactly one and that term is.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.

    Example
    -------
    >>> x = Ty('x')
    >>> f = Box('f', x, x)
    >>> assert Sum((f, )).is_cartesian and not (f + f).is_deterministic
    >>> zero = Sum((), x, x)
    >>> assert zero.is_deterministic and not zero.is_affine
    """
    def __init__(self, terms, dom=None, cod=None):
        super().__init__(terms, dom, cod)
        self.deterministic = len(self.terms) < 2 and all(
            term.is_deterministic for term in self.terms)
        self.affine = len(self.terms) == 1 and self.terms[0].is_affine


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
        return super().__call__(other)


CMap = cmap.CMap[Diagram]

Feedback, Trace, FollowedBy, Head, Tail = feedback.generators(Diagram, Box)
Diagram.box_factory = Box
Diagram.functor_factory = Functor
Hypergraph = hypergraph.Hypergraph[Diagram]
Diagram.copy_factory = Copy
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.discard_factory = Discard
Diagram.sum_factory = Sum
Id = Diagram.id


class Equation(symmetric.Equation):
    """ The :class:`symmetric.Equation` of Markov diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)


__getattr__ = deprecated_alias(__name__, {"Merge": "discopy.comarkov.Merge"})
