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

from __future__ import annotations

from typing import ClassVar

from discopy import symmetric, monoidal, cmap, hypergraph, messages
from discopy.abc import MarkovCategory
from discopy.axioms import FUEL
from discopy.cat import factory
from discopy.monoidal import Ty  # noqa: F401  pylint: disable=unused-import
from discopy.utils import assert_isatomic, factory_name, from_tree


Layer = symmetric.Layer


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
    def copy(cls, x: monoidal.Ty, n=2) -> Diagram:
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
    def merge(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        Merge :code:`n` copies of a given type :code:`x`.

        Parameters:
            x : The type to copy.
            n : The number of copies.
        """
        return cls.copy(x, n).dagger()

    @classmethod
    def discard(cls, x: monoidal.Ty) -> Diagram:
        """
        The discard of an atomic type :code:`x`.

        Parameters:
            x : The type to discard.
        """
        return cls.copy(x, 0)

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
        >>> assert Copy(x).is_affine and not Merge(x).is_affine
        """
        return all(box.is_affine for box in self.boxes)

    @property
    def is_cartesian(self) -> bool:
        """
        Whether the diagram is deterministic and affine, i.e. a total
        function: copying and discarding both commute with it.
        """
        return self.is_deterministic and self.is_affine

    map_agreement = monoidal.Diagram.map_agreement.failing(
        messages.MAP_HAS_NO_SPIDERS)


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
    class, and they are part of its equality.

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
    markov.Box('sample', monoidal.Ty(cat.Ob('p')), monoidal.Ty(cat.Ob('x')), \
is_deterministic=False)
    >>> assert not sample.dagger().dagger().is_deterministic
    """
    is_deterministic = is_affine = True

    def __init__(self, *args, is_deterministic=True, is_affine=True,
                 **params):
        self.is_deterministic, self.is_affine = is_deterministic, is_affine
        super().__init__(*args, **params)

    def flags(self) -> dict[str, bool]:
        """
        The flags that differ from those of a total function, by name, as
        :meth:`__repr__` and :meth:`to_tree` write them down.
        """
        return {name: False for name, value in [
            ("is_deterministic", self.is_deterministic),
            ("is_affine", self.is_affine)] if not value}

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
        return super().setoid() + (self.is_deterministic, self.is_affine)

    def flagged(self, other: Box) -> Box:
        """
        Give a box built from this one the flags of this one, as
        :meth:`dagger` does on top of the dagger of the class it extends,
        e.g. one that keeps a winding number, and so do the rotations of a
        :class:`discopy.frobenius.Box`.
        """
        for name, value in self.flags().items():
            setattr(other, name, value)
        return other

    def dagger(self) -> Box:
        return self.flagged(super().dagger())

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
                setattr(result, name, tree[name])
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

    def dagger(self) -> Merge:
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
    def from_tree(cls, tree: dict) -> Copy:
        return cls(from_tree(tree['x']), tree['n'])


class Merge(Box):
    """
    The merge of an atomic type :code:`x` some :code:`n` number of times.

    Parameters:
        x : The type of wires to merge.
        n : The number of wires to merge.
    """
    def __init__(self, x: monoidal.Ty, n: int = 2):
        assert_isatomic(x, monoidal.Ty)
        name = f"Merge({x}" + ("" if n == 2 else f", {n}") + ")"
        Box.__init__(self, name, dom=x ** n, cod=x, is_affine=False,
                     draw_as_spider=True, color="black", drawing_name="")

    def dagger(self) -> Copy:
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
    def from_tree(cls, tree: dict) -> Merge:
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
        self.is_deterministic = len(self.terms) < 2 and all(
            term.is_deterministic for term in self.terms)
        self.is_affine = len(self.terms) == 1 and self.terms[0].is_affine


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
