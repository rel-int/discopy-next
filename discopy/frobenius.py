# -*- coding: utf-8 -*-

"""
The free symmetric category with a supply of spiders, also known as special
commutative Frobenius algebras.

Diagrams in the free hypergraph category are faithfully encoded as
:class:`Hypergraph`, see :cite:t:`BonchiEtAl22`.

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
    Dim
    Layer
    Diagram
    Box
    Cospan
    Cup
    Cap
    Swap
    Spider
    Bubble
    Feedback
    Trace
    FollowedBy
    Head
    Tail
    Functor
    CMap

Axioms
------

>>> x, y, z = map(Ty, "xyz")

>>> split, merge = Spider(1, 2, x), Spider(2, 1, x)
>>> unit, counit = Spider(0, 1, x), Spider(1, 0, x)

Frobenius
=========

>>> frobenius = Equation(
...     split @ x >> x @ merge, merge >> split, x @ split >> merge @ x)
>>> assert frobenius
>>> frobenius.draw(doctest="docs/_static/frobenius/frobenius.svg")

.. image:: /_static/frobenius/frobenius.svg
    :align: center

Speciality
==========

>>> special = Equation(split >> merge, Spider(1, 1, x), Id(x))
>>> assert special
>>> special.draw(doctest="docs/_static/frobenius/special.svg")

.. image:: /_static/frobenius/special.svg
    :align: center
"""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from operator import index
from typing import ClassVar, Self

from discopy import (
    monoidal, rigid, markov, comarkov, compact, pivotal, feedback, cmap,
    hypergraph, symmetric)
from discopy.abc import HypergraphCategory
from discopy.cat import factory
from discopy.python import finset
from discopy.utils import (
    assert_isatomic, assert_iscomposable, deprecated_alias, factory_name,
    from_tree)


class Wire(pivotal.Wire):
    """
    A frobenius object is a self-dual pivotal object: its adjoints are the
    object itself the other way round, from its codomain to its domain.

    Parameters:
        name : The name of the object.

    >>> from discopy.monoidal import Colour
    >>> x = Wire('x', dom=Colour('a'), cod=Colour('b'))
    >>> assert x.l == x.r and (x.l.dom, x.l.cod) == (x.cod, x.dom)
    >>> assert x.l.l == x and Wire('y').l == Wire('y')
    """
    @property
    def l(self) -> Self:
        """ The object the other way round, from its codomain to its domain.
        """
        result = copy(self)
        result.dom, result.cod = self.cod, self.dom
        return result

    r = l
    windings = (0, )


HeadOb, TailOb = feedback.wire_generators(Wire)


@factory
class Ty(pivotal.Ty):
    """
    A frobenius type is a pivotal type with frobenius objects inside.

    Parameters:
        inside (frobenius.Wire) : The objects inside the type.
    """
    generator_factory = Wire


@factory
class Nat(rigid.Nat, Ty):
    """
    A ``Nat`` is a natural number ``n`` seen as a frobenius type with
    unnamed objects.

    Parameters
    ----------
    n : int
        The natural number.
    """

    l = r = property(lambda self: self)


@factory
class Dim(monoidal.Dim, Ty):
    """
    A dimension is a tuple of integers greater than one seen as a type. Its
    adjoints are its reversal and it has no winding, so :meth:`unwind` is
    the identity.

    Example
    -------
    >>> assert Dim(2, 3).l == Dim(2, 3).r == Dim(3, 2)
    >>> assert Dim(2).unwind() == Dim(2)
    """

    l = r = property(lambda self: self.ar(*self.inside[::-1]))

    def unwind(self) -> Dim:
        """
        The identity: a dimension has no winding.

        >>> assert Dim(2).unwind() == Dim(2)
        """
        return self


class Layer(compact.Layer):
    """
    A tensor product of generators and non-empty plumbing, where plumbing is
    a type when it is the identity and a :class:`Cospan` otherwise, i.e.
    permutations, phase-free spiders and the cups and caps of a self-dual
    wire. Plumbing components are coalesced, so that the layer alternates
    between cospans and generators.

    A category below frobenius whose spiders are not its own cospans, e.g.
    :mod:`discopy.tensor`, keeps them as generators, and only its
    permutations as plumbing.

    Parameters:
        inside : Generators and plumbing, with at least one generator or one
                 cospan other than the identity.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> assert Layer(Spider(1, 2, x), y, f).boxes_or_types == (
    ...     Cospan(x @ y, [0, 1], [0, 0, 1]), f)
    """
    @classmethod
    def is_plumbing_box(cls, value: Ty | Box) -> bool:
        """
        Whether a component is a permutation, or a cospan of a category that
        has its own, see :attr:`Diagram.cospan_factory`.
        """
        return isinstance(value, symmetric.Permutation)\
            or getattr(value, "legs", None) is not None\
            and value.ar.cospan_factory.ar is value.ar


@factory
class Diagram(
        compact.Diagram, markov.Diagram, comarkov.Diagram,
        HypergraphCategory):
    """
    A frobenius diagram is a compact, Markov and comarkov diagram.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    sum_factory: ClassVar[type["Sum"]]
    box_factory: ClassVar[type["Box"]]
    functor_factory: ClassVar[type["Functor"]]
    cup_factory: ClassVar[type["Cup"]]
    cap_factory: ClassVar[type["Cap"]]
    swap_factory: ClassVar[type["Swap"]]
    permutation_factory: ClassVar[type["Permutation"]]
    bubble_factory: ClassVar[type["Bubble"]]

    ob = Ty
    layer_factory = Layer
    str_transparency = compact.Diagram.str_transparency

    @classmethod
    def from_cospan(cls, apex: Ty, left, right) -> Diagram:
        """
        A native :class:`Cospan` box when the category has a matching
        cospan factory, its decomposition by
        :meth:`discopy.abc.HypergraphCategory.cospan` otherwise. A bijection
        becomes :meth:`from_permutation`, the identity the identity diagram.

        Parameters:
            apex : The type of each point of the apex.
            left : The point of each input.
            right : The point of each output.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> assert Diagram.from_cospan(x @ y, [1, 0], [0, 1]) == Swap(y, x)
        """
        left, right = list(left), list(right)
        dom = apex[:0].tensor(*(apex[k:k + 1] for k in left))
        if len(left) == len(right) == len(apex)\
                and sorted(left) == sorted(right) == list(range(len(apex))):
            perm = [left.index(k) for k in right]
            result = cls.from_permutation(perm, dom)
            return result  # ty: ignore[invalid-return-type]
        if cls.cospan_factory.ar is cls:
            return cls.cospan_factory(apex, left, right)
        return super().from_cospan(apex, left, right)

    @classmethod
    def from_function(cls, func, dom: Ty) -> Diagram:
        """
        The cospan of a function, i.e. with the identity as left leg.

        >>> x, y = Ty('x'), Ty('y')
        >>> assert Diagram.from_function([1, 0, 0], x @ y)\\
        ...     == Cospan(x @ y, [0, 1], [1, 0, 0])
        """
        if cls.cospan_factory.ar is not cls:
            return super().from_function(
                func, dom)  # ty: ignore[invalid-return-type]
        return cls.from_cospan(dom, range(len(dom)), func)

    @classmethod
    def from_cofunction(cls, cofunc, cod: Ty) -> Diagram:
        """
        The cospan of a cofunction, i.e. with the identity as right leg.

        >>> x, y = Ty('x'), Ty('y')
        >>> assert Diagram.from_cofunction([1, 0, 0], x @ y)\\
        ...     == Cospan(x @ y, [1, 0, 0], [0, 1])
        """
        if cls.cospan_factory.ar is not cls:
            return super().from_cofunction(
                cofunc, cod)  # ty: ignore[invalid-return-type]
        return cls.from_cospan(cod, cofunc, range(len(cod)))

    @classmethod
    def caps(cls, left, right):
        return cls.cups(left, right).dagger()

    @classmethod
    def spiders(cls, n_legs_in: int, n_legs_out: int, typ: Ty, phases=None
                ) -> Diagram:
        """
        The spiders on a given type with ``n_legs_in`` and ``n_legs_out`` and
        some optional vector of ``phases``.

        Parameters:
            n_legs_in : The number of legs in for each spider.
            n_legs_out : The number of legs out for each spider.
            typ : The type of the spiders.
            phases : The phase for each spider.
        """
        n_legs_in, n_legs_out = index(n_legs_in), index(n_legs_out)
        if phases is None and len(typ) != 1\
                and getattr(cls, "cospan_factory", None)\
                and cls.cospan_factory.ar is cls:
            spiders = finset.Cospan.spiders(n_legs_in, n_legs_out, len(typ))
            return cls.from_cospan(typ, spiders.left, spiders.right)
        return interleaving(cls, cls.spider_factory)(
            n_legs_in, n_legs_out, typ, phases)

    def unfuse(self) -> Diagram:
        """
        Unfuse arbitrary spiders into spiders with one or three legs.

        See Also
        --------
        This calls :func:`coherence`.

        Example
        -------
        >>> spider = Spider(3, 5, Ty(''), "$\\\\phi$") @ Ty()
        >>> Equation(spider, spider.unfuse(), symbol="$\\\\mapsto$").draw(
        ...     doctest='docs/_static/hypergraph/unfuse.svg')

        .. image:: /_static/hypergraph/unfuse.svg
            :align: center
        """
        F = compact.Functor(
            ob_map=lambda x: x, ar_map=lambda f:
                f.unfuse() if isinstance(f, Spider) else f,
            dom=Diagram, cod=Diagram)
        return F(self)


class Box(compact.Box, markov.Box, comarkov.Box, Diagram):
    """
    A frobenius box is a compact, Markov and comarkov box in a frobenius
    diagram, its rotations keeping its flags as its dagger does, which hold
    of neither, as the transpose of a total function need not be one.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.

    Example
    -------
    >>> x = Ty('x')
    >>> f = Box('f', x, x, is_affine=False)
    >>> assert f.r[::-1][::-1] == f.r and f.r.l == f
    >>> assert Box('g', x, x).is_cartesian and not Box('g', x, x).r.is_affine
    """
    serialised_attrs = pivotal.Box.serialised_attrs + (
        "is_deterministic", "is_affine")

    @property
    def is_deterministic(self) -> bool:
        return super().is_deterministic and not self.z

    @property
    def is_affine(self) -> bool:
        return super().is_affine and not self.z

    @is_deterministic.setter
    def is_deterministic(self, value: bool):
        self.deterministic = value

    @is_affine.setter
    def is_affine(self, value: bool):
        self.affine = value

    def rotate(self, left=False):
        return self.flagged(super().rotate(left))


class Cospan(Box):
    """
    A cospan of finite sets as plumbing: input ``i`` and output ``j`` are
    legs of the spiders ``left[i]`` and ``right[j]`` of the apex, i.e.
    ``dom[i] == apex[left[i]]`` and ``cod[j] == apex[right[j]]``, so that a
    cospan holds a :class:`finset.Cospan <discopy.python.finset.Cospan>`, an
    arrow of the free hypergraph category on one object, see
    :cite:t:`Lack04`. A point that no leg reaches is a spider with no legs.

    It is the :class:`discopy.comarkov.Cofunction` ``left`` into its apex
    then the :class:`discopy.markov.Function` ``right`` out of it, i.e. its
    merges then its copies. A :class:`Layer` stores it as plumbing. A
    bijection constructs a :class:`Permutation`, a cospan with one point a
    :class:`Spider`, see :meth:`Diagram.from_cospan`, and the points are
    numbered as :class:`finset.Cospan <discopy.python.finset.Cospan>` does.

    Parameters:
        apex : The type of each point of the apex.
        left : The point of each input.
        right : The point of each output.

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Cospan(x @ y, [0, 1, 0], [1, 1])
    >>> assert f.dom == x @ y @ x and f.cod == y @ y
    >>> assert Equation(f, x @ Swap(y, x) >> Spider(2, 0, x) @ Spider(1, 2, y))
    >>> assert Cospan(x, [0], [0, 0]) == Spider(1, 2, x)
    >>> assert Cospan(x @ y, [0, 1], [1, 0]) == Swap(x, y)

    Tensoring cospans gives one cospan, and so does a cup or a spider:

    >>> assert Spider(2, 0, x) @ Spider(1, 2, y) @ x\\
    ...     == Cospan(x @ y @ x, [0, 0, 1, 2], [1, 1, 2])
    >>> f.draw(doctest='docs/_static/frobenius/cospan.svg')

    .. image:: /_static/frobenius/cospan.svg
        :align: center
    """
    def __new__(cls, apex: Ty | None = None, left=None, right=None):
        if apex is None or cls is not cls.ar.cospan_factory:
            return super().__new__(cls)
        left, right = list(
            left), list(right)  # ty: ignore[invalid-argument-type]
        if len(apex) == 1 and not any(left + right)\
                and not len(left) == len(right) == 1:
            factory = {(2, 0): cls.ar.cup_factory, (0, 2): cls.ar.cap_factory
                       }.get((len(left), len(right)))
            return factory(apex, apex) if factory\
                else cls.ar.spider_factory(len(left), len(right), apex)
        if len(left) == len(right) == len(apex)\
                and sorted(left) == sorted(right) == list(range(len(apex))):
            return cls.ar.from_cospan(apex, left, right)
        return super().__new__(cls)

    def __init__(self, apex: Ty, left, right):
        order = finset.Cospan.order(left, right, len(apex))
        self.apex = apex[:0].tensor(*(apex[k:k + 1] for k in order))
        self.legs = finset.Cospan(left, right, len(apex))
        dom, cod = (self.apex[:0].tensor(*(
            self.apex[k:k + 1] for k in leg))
            for leg in (self.legs.left, self.legs.right))
        ins = [self.legs.left.count(k) for k in range(self.legs.apex)]
        Box.__init__(
            self, f"Cospan({self.legs.left}, {self.legs.right})", dom,
            cod, is_deterministic=all(ins), is_affine=set(ins) <= {1})

    def __getnewargs__(self):
        """
        The arguments of :meth:`__new__`, which a pickle and a delay read.

        >>> x, y = map(Ty, "xy")
        >>> f = Cospan(x @ y, [0, 1, 0], [1, 1])
        >>> assert f.delay() == Cospan((x @ y).delay(), [0, 1, 0], [1, 1])
        """
        return self.apex, self.legs.left, self.legs.right

    @property
    def is_identity(self) -> bool:
        """ Whether the underlying cospan is the identity. """
        return self.legs == finset.Cospan.id(len(self.dom))

    @property
    def size(self) -> int:
        """
        The number of spiders the cospan fuses, i.e. of the points of its
        apex without exactly one leg in and one out, so that a permutation
        counts zero and a cup or a cap one, as their boxes do.

        >>> x, y = Ty('x'), Ty('y')
        >>> assert Diagram.spiders(2, 1, x @ y).size == 2
        >>> assert Cospan(x @ y, [0, 1], [1, 0]).size == 0
        """
        left, right = self.legs.left, self.legs.right
        return sum((left.count(k), right.count(k)) != (1, 1)
                   for k in range(self.legs.apex))

    def setoid(self):
        if self.is_identity:
            return (), self.dom, self.cod
        return type(self), self.apex, tuple(self.legs.left), tuple(
            self.legs.right)

    def to_spiders(self) -> Diagram:
        """
        The same cospan as the permutation that gathers the inputs of each
        point, its spider, then the permutation that scatters its outputs, a
        point with one leg on each side being a wire and one with two legs on
        the same side a cup or a cap. It is how a cospan is drawn and how a
        functor sends it, as each of these boxes.

        >>> x, y = Ty('x'), Ty('y')
        >>> print(Cospan(x @ y, [0, 1, 0], [1, 1]).to_spiders())
        Permutation(x @ y @ x, [0, 2, 1]) >> Cup(x, x) @ Spider(1, 2, y)
        """
        left, right, apex = self.legs.left, self.legs.right, self.legs.apex
        m, n = len(self.dom), len(self.cod)
        ins, gather = finset.Function(
            left, apex, m).factorise()  # ty: ignore[invalid-argument-type]
        outs, scatter = finset.Function(
            right, apex, n).factorise()  # ty: ignore[invalid-argument-type]
        points = [self.apex[k:k + 1] for k in range(len(self.apex))]
        spiders = []
        for x, m, n in zip(points, ins, outs):
            spider = x if (m, n) == (1, 1) else Cup(x, x) if (m, n) == (2, 0)\
                else Cap(x, x) if (m, n) == (0, 2) else Spider(m, n, x)
            if spiders and isinstance(spider, Ty)\
                    and isinstance(spiders[-1], Ty):
                spiders[-1] = spiders[-1] @ spider
            else:
                spiders.append(spider)
        middle = self.layer_factory(*spiders, normalise=False)
        return self.ar.from_permutation(gather.dagger(), self.dom)\
            >> self.ar((middle, ), middle.dom, middle.cod)\
            >> self.ar.from_permutation(scatter, middle.cod)

    def to_drawing(self):
        """ Draw the cospan as its spiders, see :meth:`to_spiders`. """
        return self.to_spiders().to_drawing()

    def to_tree(self) -> dict:
        """
        Serialise a cospan, see :func:`discopy.utils.dumps`.

        >>> from discopy.utils import dumps, loads
        >>> x, y = Ty('x'), Ty('y')
        >>> f = Cospan(x @ y, [0, 1, 0], [1, 1])
        >>> assert loads(dumps(f)) == f
        """
        return dict(factory=factory_name(type(self)),
                    apex=self.apex.to_tree(), left=self.legs.left,
                    right=self.legs.right)

    @classmethod
    def from_tree(cls, tree: dict) -> Cospan:
        return cls(from_tree(tree['apex']), tree['left'], tree['right'])

    def dagger(self) -> Diagram:
        """ The cospan the other way round, i.e. its legs swapped. """
        return self.ar.from_cospan(
            self.apex, self.legs.right, self.legs.left)

    def rotate(self, left=False):
        """ The cospan with each leg reversed and swapped with the other. """
        del left
        return self.ar.from_cospan(
            self.apex, self.legs.right[::-1], self.legs.left[::-1])

    def fuse(self, other: Cospan) -> Diagram:
        """
        The composite with another cospan as one cospan, i.e. their pushout,
        the identity diagram when it is the identity.

        Parameters:
            other : The cospan to compose with.

        >>> x = Ty('x')
        >>> assert Spider(1, 2, x).fuse(Spider(2, 1, x)) == Id(x)
        >>> assert Cap(x, x).fuse(Cup(x, x)) == Spider(0, 0, x)
        """
        assert_iscomposable(self, other)
        cospan, points = self.legs.pushout(other.legs)
        apex = [None] * cospan.apex
        for k, x in zip(points, self.apex @ other.apex):
            apex[k] = x
        return self.ar.from_cospan(
            self.apex[:0].tensor(*apex), cospan.left, cospan.right)

    def tensor(self, *others):
        """
        Tensoring plumbing with plumbing gives one wider cospan, a run of
        cospans and types coalescing into the disjoint union of their
        apexes, from which point the layers are those of an ordinary
        :meth:`discopy.monoidal.Diagram.tensor`.

        >>> x, y = Ty('x'), Ty('y')
        >>> assert Spider(1, 0, x).tensor(y, Cup(x, x))\\
        ...     == Cospan(x @ y @ x, [0, 1, 2, 2], [1])
        """
        if not self.ar.layer_factory.is_plumbing_box(self):
            return self.ar.tensor(self, *others)
        left, right, apex = self.legs.left, self.legs.right, self.apex
        plumbing = 0
        while plumbing < len(others) and (
                isinstance(others[plumbing], Ty)
                or self.ar.layer_factory.is_plumbing_box(others[plumbing])
                or isinstance(others[plumbing], Diagram)
                and not others[plumbing].inside):
            other, plumbing = others[plumbing], plumbing + 1
            typ = other if isinstance(other, Ty) else other.dom
            ins, outs = 2 * (range(len(typ)), ) if isinstance(other, Ty)\
                or not other.inside else (other.legs.left, other.legs.right)
            left = left + [len(apex) + i for i in ins]
            right = right + [len(apex) + j for j in outs]
            apex = apex @ (typ if isinstance(other, Ty) or not other.inside
                           else other.apex)
        result = self.ar.from_cospan(apex, left, right) if plumbing\
            else self
        rest = others[plumbing:]
        return result if not rest else result.ar.tensor(result, *rest)

    def __rmatmul__(self, other):
        if not isinstance(other, Ty)\
                or not self.ar.layer_factory.is_plumbing_box(self):
            return super(Box, self).__rmatmul__(other)
        cospan = finset.Cospan.id(len(other)) @ self.legs
        return self.ar.from_cospan(
            other @ self.apex, cospan.left, cospan.right)

    def __repr__(self):
        return f"{factory_name(type(self))}({self.apex!r}, "\
            f"{self.legs.left}, {self.legs.right})"

    def __str__(self):
        return f"Cospan({self.apex}, {self.legs.left}, {self.legs.right})"


class Cup(compact.Cup, Box):
    """
    A frobenius cup is a compact cup in a frobenius diagram, i.e. the spider
    with two legs in, deterministic but not affine.

    Parameters:
        left (Ty) : The atomic type.
        right (Ty) : Its adjoint.
    """
    apex = property(lambda self: self.left)
    legs = property(lambda self: finset.Cospan.cups(1, 1)
                    if self.left == self.right else None)
    tensor, __rmatmul__, fuse = Cospan.tensor, Cospan.__rmatmul__, Cospan.fuse

    def __init__(self, left, right):
        super().__init__(left, right)
        self.affine = False

    def __setstate__(self, state):
        """
        Load a cup with its flags, which a pickle from before them lacks.

        >>> x = Ty('x')
        >>> cup, state = Cup(x, x.r), dict(Cup(x, x.r).__dict__)
        >>> del state["affine"]
        >>> old = Cup.__new__(Cup)
        >>> old.__setstate__(state)
        >>> assert old == cup and not old.is_affine
        """
        super().__setstate__(state)
        if "affine" not in state:
            self.affine = False


class Cap(compact.Cap, Box):
    """
    A frobenius cap is a compact cap in a frobenius diagram, i.e. the spider
    with two legs out, neither deterministic nor affine.

    Parameters:
        left (Ty) : The atomic type.
        right (Ty) : Its adjoint.
    """
    apex = Cup.apex
    legs = property(lambda self: finset.Cospan.caps(1, 1)
                    if self.left == self.right else None)
    tensor, __rmatmul__, fuse = Cospan.tensor, Cospan.__rmatmul__, Cospan.fuse

    def __init__(self, left, right):
        super().__init__(left, right)
        self.deterministic = self.affine = False

    def __setstate__(self, state):
        super().__setstate__(state)
        if "affine" not in state:
            self.deterministic = self.affine = False


class Permutation(
        compact.Permutation, markov.Permutation, comarkov.Permutation, Box):
    """
    A permutation in a Frobenius diagram, i.e. a bijective :class:`Cospan`.

    >>> x, y = Ty('x'), Ty('y')
    >>> assert isinstance(Swap(x, y), Cospan)
    >>> assert Swap(x, y).legs == finset.Cospan([0, 1], [1, 0])
    """
    apex = property(lambda self: self.dom)
    legs = property(lambda self: finset.Cospan(
        list(range(len(self.dom))), list(self.perm), len(self.dom)))
    tensor, fuse = Cospan.tensor, Cospan.fuse


class Swap(Permutation, compact.Swap, markov.Swap, comarkov.Swap, Box):
    """
    A frobenius swap is a compact, Markov and comarkov swap in a frobenius
    diagram.

    Parameters:
        left (Ty) : The type on the top left and bottom right.
        right (Ty) : The type on the top right and bottom left.
    """

    def rotate(self, left=False):
        del left
        return self


class Spider(Box):
    """
    The spider with :code:`n_legs_in` and :code:`n_legs_out`
    on a given atomic type, with some optional phase as ``data``.

    Its flags are those of its legs and phase, by spider fusion: it is
    deterministic when it has a leg in and no phase, affine when it has
    exactly one leg in and no phase.

    Parameters:
        n_legs_in : The number of legs in.
        n_legs_out : The number of legs out.
        typ : The type of the spider.
        data : The phase of the spider.
        is_dagger : Whether the spider is marked as a dagger, which only
            a phased one is.

    Examples
    --------
    >>> x = Ty('x')
    >>> spider = Spider(1, 2, x)
    >>> assert spider.dom == x and spider.cod == x @ x
    >>> assert spider.is_cartesian and not spider.r.is_affine
    >>> assert not Spider(1, 2, x, 0.5).is_deterministic
    """
    draw_as_spider = True
    color = "black"
    apex = property(lambda self: self.typ)
    legs = property(lambda self: finset.Cospan.spiders(
        len(self.dom), len(self.cod), 1) if self.phase is None else None)
    tensor, __rmatmul__, fuse = Cospan.tensor, Cospan.__rmatmul__, Cospan.fuse

    def __init__(self, n_legs_in: int, n_legs_out: int, typ: Ty, data=None,
                 is_dagger=False, **params):
        assert_isatomic(typ)
        n_legs_in, n_legs_out = index(n_legs_in), index(n_legs_out)
        self.typ = typ
        is_dagger = is_dagger and bool(data)
        str_data = "" if data is None else f", {data}"
        n_in, n_out = (n_legs_out, n_legs_in) if is_dagger\
            else (n_legs_in, n_legs_out)
        name = type(self).__name__ + f"({n_in}, {n_out}, {typ}{str_data})"
        dom, cod = typ ** n_legs_in, typ ** n_legs_out
        Box.__init__(
            self, name, dom, cod, data=data, is_dagger=is_dagger,
            is_deterministic=n_legs_in > 0 and not data,
            is_affine=n_legs_in == 1 and not data, **params)
        self.drawing_name = "" if not data else str(data)

    def __setstate__(self, state):
        if "_name" in state and state["_name"] == type(self).__name__:
            phase = state.get("_data", None)
            str_data = "" if phase is None else f", {phase}"
            cod, dom = state['_dom'], state['_cod']
            state["_name"] = type(self).__name__\
                + f"({dom.n}, {cod.n}, {state['_typ']}{str_data})"
        super().__setstate__(state)
        if "affine" not in state:
            self.deterministic = len(self.dom) > 0 and not self.phase
            self.affine = len(self.dom) == 1 and not self.phase

    def __getnewargs__(self):
        """ The arguments of the constructor, which a pickle and a delay read.

        >>> x = Ty('x')
        >>> assert Spider(1, 2, x, .5).delay() == Spider(1, 2, x.delay(), .5)
        """
        return (
            len(self.dom), len(self.cod), self.typ, self.phase, self.is_dagger)

    @property
    def phase(self):
        """ The phase of the spider. """
        return self.data

    def __repr__(self):
        phase = "" if self.phase is None else f", {repr(self.phase)}"
        is_dagger = ", is_dagger=True" if self.is_dagger else ""
        return factory_name(type(self)) + f"({len(self.dom)}, {len(self.cod)}"\
            f", {repr(self.typ)}{phase}{is_dagger})"

    def dagger(self):
        """
        The dagger of a spider with no phase, i.e. ``None`` or zero as for its
        flags, is the spider with its legs swapped. A phased spider is a
        generator: its dagger keeps its phase and is marked as a dagger, so
        that any phase has one, a functor maps it to the dagger of its image,
        and negating a phase is left to :mod:`discopy.quantum.zx`, where
        phases form a group.

        >>> x = Ty('x')
        >>> spider = Spider(1, 2, x, "$\\\\phi$")
        >>> print(spider.dagger())
        Spider(1, 2, x, $\\phi$)[::-1]
        >>> assert spider.dagger().dagger() == spider
        >>> assert Spider(1, 2, x).dagger() == Spider(2, 1, x)
        >>> assert Spider(1, 2, x, 0).dagger() == Spider(2, 1, x, 0)
        >>> assert Spider(1, 2, x, is_dagger=True) == Spider(1, 2, x)
        """
        return type(self)(len(self.cod), len(self.dom), self.typ, self.phase,
                          is_dagger=not self.is_dagger)

    def to_tree(self) -> dict:
        """
        Serialise a spider by its legs, type and phase.

        >>> from discopy.utils import dumps, loads
        >>> x = Ty('x')
        >>> assert loads(dumps(Spider(1, 2, x))) == Spider(1, 2, x)
        """
        tree = {'factory': factory_name(type(self)),
                'n_legs_in': len(self.dom), 'n_legs_out': len(self.cod),
                'typ': self.typ.to_tree()}
        if self.phase is not None:
            tree['phase'] = self.phase
        if self.is_dagger:
            tree['is_dagger'] = True
        return tree

    @classmethod
    def from_tree(cls, tree: dict) -> Spider:
        return cls(tree['n_legs_in'], tree['n_legs_out'],
                   from_tree(tree['typ']), tree.get('phase', None),
                   is_dagger=tree.get('is_dagger', False))

    def rotate(self, left=False):
        del left
        return type(self)(len(self.cod), len(self.dom), self.typ, self.phase,
                          is_dagger=self.is_dagger)

    def unfuse(self) -> Diagram:
        if self.is_dagger:
            return self.dagger().unfuse().dagger()
        return coherence(self.ar, type(self))(
            len(self.dom), len(self.cod), self.typ, self.phase)


class Bubble(compact.Bubble, markov.Bubble, comarkov.Bubble, Box):
    """
    A Frobenius bubble is a compact, Markov and comarkov bubble and a
    frobenius box, which keeps its flags as it rotates.

    >>> x = Ty('x')
    >>> noisy = Box('f', x, x).bubble(is_deterministic=False)
    >>> assert noisy.r.l == noisy and not noisy.r.l.is_deterministic
    """
    serialised_attrs = compact.Bubble.serialised_attrs + (
        "is_deterministic", "is_affine")


class Sum(compact.Sum, markov.Sum, comarkov.Sum, Box):
    """
    A Frobenius sum is a sum of Frobenius diagrams.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(compact.Functor, markov.Functor, comarkov.Functor):
    """
    A hypergraph functor is a compact functor that preserves spiders.

    A spider with no phase is the Frobenius structure, mapped to the spiders
    of the codomain; a spider with a phase is a generator, mapped by
    ``ar_map`` like any box, e.g. the Z and X spiders of
    :mod:`discopy.quantum.zx`.

    Parameters:
        ob_map (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """

    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Spider) and other.phase is None:
            return self.cod.spiders(
                len(other.dom), len(other.cod), self(other.typ))
        if isinstance(other, Cospan)\
                and not isinstance(other, symmetric.Permutation):
            return self(other.to_spiders())
        if isinstance(other, (markov.Copy, Bubble)):
            return markov.Functor.__call__(self, other)
        if isinstance(other, comarkov.Merge):
            return comarkov.Functor.__call__(self, other)
        return compact.Functor.__call__(self, other)


def interleaving(cls: type, factory: Callable
                 ) -> Callable[..., Diagram]:
    """
    Take a ``factory`` for spiders of atomic types and extend it recursively.

    Parameters:
        cls : A diagram factory, e.g. :class:`Diagram`.
        factory : A factory for spiders of atomic types, e.g. :class:`Spider`.
    """
    def method(n_legs_in, n_legs_out, typ, phases=None):
        phases = phases or len(typ) * [None]
        if len(typ) == 1:
            return factory(n_legs_in, n_legs_out, typ, phases[0])
        result = cls.id().tensor(*[
            factory(n_legs_in, n_legs_out, x, p) for x, p in zip(typ, phases)])
        for i, t in enumerate(typ):
            for j in range(n_legs_in - 1):
                result <<= result.dom[:i * j + i + j] @ cls.swap(
                    t, result.dom[i * j + i + j:i * n_legs_in + j]
                ) @ result.dom[i * n_legs_in + j + 1:]
            for j in range(n_legs_out - 1):
                result >>= result.cod[:i * j + i + j] @ cls.swap(
                    result.cod[i * j + i + j:i * n_legs_out + j], t
                ) @ result.cod[i * n_legs_out + j + 1:]
        return result

    return method


def coherence(cls: type, factory: Callable
              ) -> Callable[..., Diagram]:
    """
    Take a ``factory`` for spiders with one or three legs of atomic types
    and extend it recursively to arbitrary spiders of atomic types.

    Parameters:
        cls : A diagram factory, e.g. :class:`Diagram`.
        factory : A factory for spiders of atomic types, e.g. :class:`Spider`.

    See Also
    --------
    This is called by :meth:`frobenius.Diagram.unfuse`.

    Note
    ----
    If the spider has a non-trivial phase then we also output a phase shifter.

    Example
    -------
    >>> print(Spider(2, 2, Ty('x'), 0.5).unfuse())
    Spider(2, 1, x) >> Spider(1, 1, x, 0.5) >> Spider(1, 2, x)
    """
    def method(a, b, x, phase=None):
        if phase is not None:  # Coherence for phase shifters.
            return method(a, 1, x)\
                >> factory(1, 1, x, phase)\
                >> method(1, b, x)
        if (a, b) in [(0, 1), (1, 0), (2, 1), (1, 2)]:
            return factory(a, b, x)
        if (a, b) == (1, 1):  # Speciality: one-to-one spiders are identity.
            return cls.id(x)
        if a < b:  # Cut the work in two.
            return method(b, a, x[::-1]).rotate()
        if b != 1:
            return method(a, 1, x) >> method(1, b, x)
        if a % 2:  # We can now assume a is odd and b == 1.
            return method(a - 1, 1, x) @ x >> factory(2, 1, x)
        # We can now assume a is even and b == 1.
        half_spiders = method(a // 2, 1, x)
        return half_spiders @ half_spiders >> factory(2, 1, x)

    return method


CMap = cmap.CMap[Diagram]

Feedback, Trace, FollowedBy, Head, Tail = feedback.generators(Diagram, Box)
Diagram.box_factory = Box
Diagram.functor_factory = Functor
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
Diagram.swap_factory = Swap
Diagram.spider_factory = Spider  # ty: ignore[invalid-assignment]
Diagram.permutation_factory = Permutation
Diagram.cospan_factory = Cospan
Diagram.bubble_factory = Bubble
Layer.plumbing = (Ty, Cospan)
Cospan.register(Permutation)
Diagram.sum_factory = Sum
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id


class Equation(compact.Equation):
    """ The :class:`compact.Equation` of Frobenius diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)


__getattr__ = deprecated_alias(__name__, {"Ob": "Wire", "PRO": "Nat"})
