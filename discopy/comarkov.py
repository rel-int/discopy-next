# -*- coding: utf-8 -*-

"""
The free comarkov category, i.e. a symmetric category with a supply of
commutative monoids, the dual of :mod:`markov`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Layer
    Diagram
    Box
    Cofunction
    Swap
    Permutation
    Merge
    Noise
    Feedback
    Trace
    FollowedBy
    Head
    Tail
    Functor

Axioms
------

>>> x = Ty('x')

>>> merge, noise = Merge(x), Merge(x, n=0)

Commutative monoid
==================

>>> unitality = Equation(noise @ x >> merge, Id(x), x @ noise >> merge)
>>> associativity = Equation(merge @ x >> merge, x @ merge >> merge)
>>> commutativity = Equation(Swap(x, x) >> merge, merge)
>>> assert unitality and associativity and commutativity
>>> Equation(unitality, associativity, commutativity, symbol='').draw(
...     doctest="docs/_static/frobenius/monoid.svg")

.. image:: /_static/frobenius/monoid.svg
    :align: center

Coherence
=========

>>> assert Equation(Diagram.merge(x @ x, n=0), noise @ noise)
>>> assert Equation(Diagram.merge(x @ x),
...     x @ Swap(x, x) @ x >> merge @ merge)

Note
----
Equality of comarkov diagrams is computed by translation to hypergraph.
Merge boxes are translated to spiders, thus when they appear in the same
diagram as the copy boxes of :mod:`markov` they automatically satisfy the
:mod:`frobenius` axioms.
"""

from __future__ import annotations

from typing import ClassVar

from discopy import (
    symmetric, monoidal, feedback, cmap, hypergraph, messages, op)
from discopy.abc import ComarkovCategory, MarkovCategory
from discopy.cat import factory
from discopy.feedback import (  # noqa: F401  pylint: disable=unused-import
    Wire, Ty)
from discopy.python import finset
from discopy.utils import (
    AxiomError, assert_isatomic, assert_iscomposable, factory_name,
    from_tree)


class Layer(symmetric.Layer):
    """
    A tensor product of generators and non-empty plumbing, where plumbing is
    a type when it is the identity and a :class:`Cofunction` otherwise, i.e.
    merges, noise and permutations, the dual of :class:`discopy.markov.Layer`.

    Parameters:
        inside : Generators and plumbing, with at least one generator or one
                 cofunction other than the identity.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> assert Layer(Merge(x), y, f, Noise(y)).boxes_or_types == (
    ...     Cofunction(x @ y, [0, 0, 1]), f, Noise(y))
    """


@factory
class Diagram(symmetric.Diagram, ComarkovCategory):
    """
    A comarkov diagram is a symmetric diagram with :class:`Merge` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.
    """
    merge_factory: ClassVar[type["Merge"]]
    noise_factory: ClassVar[type["Noise"]]
    cofunction_factory: ClassVar[type["Cofunction"]]
    layer_factory = Layer

    @classmethod
    def spider_factory(cls, n_legs_in, n_legs_out, typ, phase=None):
        if phase is not None or n_legs_out != 1:
            raise ValueError(messages.NO_SUCH_SPIDER.format(
                factory_name(cls), "merges"))
        return cls.merge_factory(typ, n_legs_in)

    @classmethod
    def merge(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        Merge :code:`n` copies of a given type :code:`x` into one.

        Parameters:
            x : The type to merge.
            n : The number of copies.

        Note
        ----
        A category with native cofunctions merges a type as one
        :class:`Cofunction`.

        >>> x, y = Ty('x'), Ty('y')
        >>> assert Diagram.merge(x @ y) == Cofunction(x @ y, [0, 1, 0, 1])
        """
        if cls.cofunction_factory.ar is cls:
            return cls.from_cofunction(finset.Function.copy(len(x), n), x)
        from discopy import frobenius
        return frobenius.Diagram.spiders.__func__(cls, n, 1, x)

    @classmethod
    def from_cofunction(cls, cofunc, cod: monoidal.Ty) -> Diagram:
        """
        A native :class:`Cofunction` box when the category has a matching
        cofunction factory, its decomposition by
        :meth:`discopy.abc.ComarkovCategory.cofunction` otherwise. A
        bijection becomes the permutation that undoes it, see
        :meth:`from_permutation`, the identity the identity diagram.

        Parameters:
            cofunc : The output each input goes to, as a list or a
                     :class:`finset.Function`.
            cod : The type to merge into.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> assert Diagram.from_cofunction([1, 0, 0], x @ y)\\
        ...     == Cofunction(x @ y, [1, 0, 0])
        >>> assert Diagram.from_cofunction([1, 0], x @ y) == Swap(y, x)
        """
        cofunc = list(cofunc)
        if sorted(cofunc) == list(range(len(cod))):
            dom = cod[:0].tensor(*(cod[i] for i in cofunc))
            return cls.from_permutation(
                finset.Permutation(cofunc).dagger(), dom)
        if cls.cofunction_factory.ar is cls:
            return cls.cofunction_factory(cod, cofunc)
        return super().from_cofunction(cofunc, cod)


class Box(symmetric.Box, Diagram):
    """
    A comarkov box is a symmetric box in a comarkov diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """


class Cofunction(Box):
    """
    A function between finite sets as plumbing: input ``i`` goes to output
    ``cofunc[i]``, i.e. ``dom[i] == cod[cofunc[i]]``. An output that no input
    goes to is noise, one that several inputs go to is their merge, so that
    a cofunction holds an arrow of :class:`op.Comarkov[finset.Function]
    <discopy.op.Comarkov>`, the free comarkov category on one object.

    It is the formal reverse of :class:`discopy.markov.Function` with the
    same arguments, which is its dagger in a category that is both. A
    :class:`Layer` stores it as plumbing rather than as a generator. A
    bijection constructs a :class:`Permutation`, a cofunction onto one wire
    a :class:`Merge` or a :class:`Noise`, see
    :meth:`Diagram.from_cofunction`.

    Parameters:
        cod : The codomain, i.e. the wires to merge into.
        cofunc : The output each input goes to, as a list or a
                 :class:`finset.Function <discopy.python.finset.Function>`.

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Cofunction(x @ y, [1, 0, 0])
    >>> assert f.dom == y @ x @ x
    >>> assert Equation(f, y @ Merge(x) >> Swap(y, x))
    >>> assert Cofunction(x, [0, 0]) == Merge(x)
    >>> assert Cofunction(x, []) == Noise(x)
    >>> assert Cofunction(x @ y, [1, 0]) == Swap(y, x)

    Tensoring cofunctions gives one cofunction:

    >>> assert Merge(x) @ Noise(y) == Cofunction(x @ y, [0, 0])
    >>> f.draw(doctest='docs/_static/comarkov/cofunction.svg')

    .. image:: /_static/comarkov/cofunction.svg
        :align: center
    """
    def __new__(cls, cod: monoidal.Ty = None, cofunc=None):
        if cod is None or cls is not cls.ar.cofunction_factory:
            return super().__new__(cls)
        cofunc = list(cofunc)
        if sorted(cofunc) == list(range(len(cod))):
            return cls.ar.from_cofunction(cofunc, cod)
        if len(cod) == 1 and not any(cofunc):
            return cls.ar.merge_factory(cod, len(cofunc))
        return super().__new__(cls)

    def __init__(self, cod: monoidal.Ty, cofunc):
        self.cofunc = op.Comarkov(
            finset.Function(list(cofunc), len(cod), len(cofunc)))
        dom = cod[:0].tensor(*(cod[i] for i in cofunc))
        Box.__init__(self, f"Cofunction({list(cofunc)})", dom, cod)

    def __getnewargs__(self):
        """
        The arguments of :meth:`__new__`, which a pickle and a delay read.

        >>> x, y = map(Ty, "xy")
        >>> f = Cofunction(x @ y, [1, 0, 0])
        >>> assert f.delay() == Cofunction((x @ y).delay(), [1, 0, 0])
        """
        return self.cod, list(self.cofunc.inside)

    @property
    def is_identity(self) -> bool:
        """ Whether the underlying cofunction is the identity. """
        return list(self.cofunc.inside) == list(range(len(self.cod)))

    @property
    def size(self) -> int:
        """
        The number of merges and noises the cofunction fuses, i.e. of its
        outputs not reached exactly once, so that a permutation counts zero.

        >>> x, y = Ty('x'), Ty('y')
        >>> assert Merge(x).size == Noise(x).size == 1
        >>> assert Cofunction(x @ y, [0, 0]).size == 2
        """
        counts, _ = self.cofunc.inside.factorise()
        return sum(n != 1 for n in counts)

    def setoid(self):
        if self.is_identity:
            return (), self.dom, self.cod
        return type(self), self.cod, tuple(self.cofunc.inside)

    def to_drawing(self):
        """ Draw the permutation of the inputs, then the merges. """
        from discopy.drawing import Drawing
        counts, perm = self.cofunc.inside.factorise()
        merges = Drawing.id().tensor(*(
            self.merge_factory(self.cod[j:j + 1], n).to_drawing() if n != 1
            else Drawing.id(self.cod[j:j + 1].to_drawing())
            for j, n in enumerate(counts)))
        return Drawing.permutation(perm.dagger(), [
            self.dom[i:i + 1].to_drawing() for i in range(len(self.dom))])\
            >> merges

    def to_tree(self) -> dict:
        """
        Serialise a cofunction, see :func:`discopy.utils.dumps`.

        >>> from discopy.utils import dumps, loads
        >>> x, y = Ty('x'), Ty('y')
        >>> assert loads(dumps(Cofunction(x @ y, [1, 0, 0])))\\
        ...     == Cofunction(x @ y, [1, 0, 0])
        """
        return dict(factory=factory_name(type(self)),
                    cod=self.cod.to_tree(), cofunc=list(self.cofunc.inside))

    @classmethod
    def from_tree(cls, tree: dict) -> Cofunction:
        return cls(from_tree(tree['cod']), tree['cofunc'])

    def dagger(self) -> Diagram:
        """
        The function of the same list, when the category is also Markov.

        >>> Merge(Ty('x')).dagger()
        Traceback (most recent call last):
            ...
        discopy.utils.AxiomError: Merge(x) has no dagger: comarkov.Diagram \
is not a dagger category.
        """
        if not issubclass(self.ar, MarkovCategory):
            raise AxiomError(messages.NOT_A_DAGGER.format(
                self, factory_name(self.ar)))
        return self.ar.function_factory(self.cod, list(self.cofunc.inside))

    def fuse(self, other: Cofunction) -> Diagram:
        """
        The composite with another cofunction as one cofunction, i.e. the
        identity diagram when it is the identity.

        Parameters:
            other : The cofunction to compose with.

        >>> x = Ty('x')
        >>> assert (Noise(x) @ x).fuse(Merge(x)) == Id(x)
        """
        assert_iscomposable(self, other)
        return self.ar.from_cofunction(
            (self.cofunc >> other.cofunc).inside, other.cod)

    def tensor(self, *others):
        """
        Tensoring plumbing with plumbing gives one wider cofunction, the
        dual of :meth:`discopy.markov.Function.tensor`.

        >>> x, y = Ty('x'), Ty('y')
        >>> assert Merge(x).tensor(y, Noise(x))\\
        ...     == Cofunction(x @ y @ x, [0, 0, 1])
        """
        cofuncs, cod, plumbing = [self.cofunc], self.cod, 0
        while plumbing < len(others) and (
                isinstance(others[plumbing], monoidal.Ty)
                or isinstance(others[plumbing], Cofunction)
                and self.ar.layer_factory.is_plumbing_box(others[plumbing])
                or isinstance(others[plumbing], Diagram)
                and not others[plumbing].inside):
            other, plumbing = others[plumbing], plumbing + 1
            typ = other if isinstance(other, monoidal.Ty) else other.cod
            cofuncs.append(other.cofunc if isinstance(other, Cofunction)
                           else op.Comarkov.id(len(typ)))
            cod = cod @ typ
        result = self if len(cofuncs) == 1 else self.ar.from_cofunction(
            cofuncs[0].tensor(*cofuncs[1:]).inside, cod)
        rest = others[plumbing:]
        return result if not rest else result.ar.tensor(result, *rest)

    def __rmatmul__(self, other):
        if not isinstance(other, monoidal.Ty):
            return super().__rmatmul__(other)
        cofunc = op.Comarkov.id(len(other)) @ self.cofunc
        return self.ar.from_cofunction(cofunc.inside, other @ self.cod)

    def __repr__(self):
        return f"{factory_name(type(self))}"\
            f"({self.cod!r}, {list(self.cofunc.inside)})"

    def __str__(self):
        return f"Cofunction({self.cod}, {list(self.cofunc.inside)})"


Layer.plumbing = (monoidal.Ty, Cofunction)


class Permutation(symmetric.Permutation, Box):
    """
    A permutation in a comarkov category, i.e. a bijective
    :class:`Cofunction`, registered as a virtual subclass so that it keeps
    the constructor of :class:`discopy.symmetric.Permutation`: input ``i``
    goes to the output that reads it.

    Parameters:
        dom (monoidal.Ty) : The domain, i.e. the wires to permute.
        perm : The permutation as a :class:`finset.Permutation` or a list.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> assert isinstance(Swap(x, y), Cofunction)
    >>> assert Swap(x, y) @ Merge(x) == Cofunction(y @ x @ x, [1, 0, 2, 2])
    """
    cofunc = property(lambda self: op.Comarkov(finset.Function(
        list(self.perm.dagger()), len(self.cod), len(self.dom))))
    fuse, tensor = Cofunction.fuse, Cofunction.tensor


Cofunction.register(Permutation)


class Swap(Permutation, symmetric.Swap, Box):
    """
    Symmetric swap in a comarkov diagram.

    Parameters:
        left (monoidal.Ty) : The type on the top left and bottom right.
        right (monoidal.Ty) : The type on the top right and bottom left.
    """


class Merge(Cofunction):
    """
    The merge of an atomic type :code:`x` some :code:`n` number of times,
    i.e. the :class:`Cofunction` onto one wire that ``n`` inputs go to.

    Parameters:
        x : The type of wires to merge.
        n : The number of wires to merge, or the cofunction ``[0] * n``
            when a :class:`Cofunction` constructs it.
    """
    def __init__(self, x: monoidal.Ty, n: int = 2):
        assert_isatomic(x, monoidal.Ty)
        n = n if isinstance(n, int) else len(n)
        name = f"Merge({x}" + ("" if n == 2 else f", {n}") + ")"
        Box.__init__(self, name, dom=x ** n, cod=x,
                     draw_as_spider=True, color="black", drawing_name="")

    def __new__(cls, x: monoidal.Ty, n: int = 2):
        return super().__new__(cls) if n else\
            cls.noise_factory.__new__(cls.noise_factory, x)

    def __getnewargs__(self):
        """
        The arguments of :meth:`__new__`, which a pickle needs.

        >>> from pickle import dumps, loads
        >>> x = Ty('x')
        >>> assert loads(dumps(Merge(x))) == Merge(x)
        >>> assert loads(dumps(Noise(x))) == Noise(x)
        """
        return self.cod, len(self.dom)

    cofunc = property(lambda self: op.Comarkov(
        finset.Function.copy(1, len(self.dom))))

    def to_drawing(self):
        return super(Cofunction, self).to_drawing()

    def to_tree(self) -> dict:
        return super(Cofunction, self).to_tree()

    def __repr__(self):
        return (
            factory_name(type(self)) + f"({repr(self.cod)}, {len(self.dom)})")

    def __str__(self):
        return self.name

    @classmethod
    def from_tree(cls, tree: dict) -> Merge:
        return cls(from_tree(tree['cod']), len(from_tree(tree['dom'])))


class Noise(Merge):
    """
    The noise on an atomic type :code:`x`, i.e. the merge of no copy: the
    state every object of a comarkov category carries, dual to the effect
    :class:`discopy.markov.Discard` every object of a Markov category
    carries. "The time reversal of a causal theory is eternal noise"
    (toumix, 2026-10-17).

    Parameters:
        x : The type of the noise.
    """
    def __init__(self, x: monoidal.Ty, n: int = 0):
        if n:
            raise ValueError(f"A noise has no copies, got n={n}.")
        super().__init__(x, 0)


class Sum(symmetric.Sum, Box):
    """
    A comarkov sum is a symmetric sum and a comarkov box.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(symmetric.Functor):
    """
    A comarkov functor is a symmetric functor that preserves merges.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.

    Example
    -------

    We build a functor into Python functions with the disjoint union as
    tensor, where the merge of a type is the codiagonal: whichever side its
    input comes from, it comes out the same.

    >>> from discopy.python import additive
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> F = Functor({x: int}, {f: lambda n: n + 1, g: lambda n: 2 * n},
    ...             cod=additive.Function)
    >>> assert F(f @ g >> Merge(x))(41, 0) == 42
    >>> assert F(f @ g >> Merge(x))(21, 1) == 42
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Merge) and hasattr(self.cod, "merge"):
            return self.cod.merge(self(other.cod), len(other.dom))
        if isinstance(other, Cofunction) and hasattr(self.cod, "merge")\
                and not isinstance(other, symmetric.Permutation):
            cofunction = getattr(self.cod, "cofunction", None)\
                or ComarkovCategory.cofunction.__func__.__get__(self.cod)
            return cofunction(
                list(other.cofunc.inside), list(map(self, other.cod)))
        return super().__call__(other)


CMap = cmap.CMap[Diagram]

Feedback, Trace, FollowedBy, Head, Tail = feedback.generators(Diagram, Box)
Diagram.box_factory = Box
Diagram.functor_factory = Functor
Hypergraph = hypergraph.Hypergraph[Diagram]
Diagram.merge_factory = Merge
Diagram.cofunction_factory = Cofunction
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.noise_factory = Noise
Diagram.sum_factory = Sum
Id = Diagram.id


class Equation(symmetric.Equation):
    """ The :class:`symmetric.Equation` of comarkov diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)
