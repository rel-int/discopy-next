# -*- coding: utf-8 -*-

"""
The free symmetric category with a supply of spiders, also known as special
commutative Frobenius algebras.

Diagrams in the free hypergraph category are faithfully encoded as
:class:`discopy.hypergraph.Hypergraph`, see :cite:t:`BonchiEtAl22`.

Their terms are those of the relational machine calculus of
:cite:t:`BarrettEtAl24`, see :class:`TermBase`: programs for a stack
machine that push and pop values, where a variable is a spider and each of
its pushes and pops one of its legs, so that a term has any number of
outputs and its dagger is its converse.

>>> x = Ty('x')
>>> assert Equation(x(lambda a: a.dagger() >> a >> a).eval(), Spider(2, 2, x))
>>> assert Equation((x @ x @ x)(lambda a, b, c: c >> a >> b).eval(),
...                 Diagram.permutation([2, 0, 1], x @ x @ x))

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
    Nat
    Dim
    Layer
    Diagram
    Box
    Cospan
    Cup
    Cap
    Permutation
    Swap
    Spider
    Bubble
    Sum
    Feedback
    Trace
    FollowedBy
    Head
    Tail
    Functor
    Equation
    TermBase
    Value
    Variable
    Application
    Push
    Pop
    Sequence
    Choice
    Star
    Exists
    Machine

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        interleaving
        coherence

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

from abc import abstractmethod
from collections.abc import Callable
from copy import copy
from functools import reduce
from inspect import signature
from itertools import product
from operator import index
from typing import TYPE_CHECKING, ClassVar, Self

from discopy import (
    monoidal, markov, comarkov, compact, pivotal, feedback, cmap,
    hypergraph, symmetric)
from discopy.abc import HypergraphCategory
from discopy.cat import factory
from discopy.python import finset
from discopy.testable import Equation as AbstractEquation, Serialisable, axiom
from discopy.utils import (
    assert_isatomic, assert_iscomposable, deprecated_alias, factory_key,
    factory_name, from_tree)

if TYPE_CHECKING:
    from hypothesis import strategies as st


class Wire(pivotal.Wire):
    """
    A frobenius object is a self-dual pivotal object: its adjoints are the
    object itself the other way round, from its codomain to its domain.

    It has one winding, so its winding number is read modulo one, i.e. it is
    always zero.

    Parameters:
        name : The name of the object.
        z : The winding number, zero up to its one winding.

    >>> from discopy.monoidal import Colour
    >>> x = Wire('x', dom=Colour('a'), cod=Colour('b'))
    >>> assert x.l == x.r and (x.l.dom, x.l.cod) == (x.cod, x.dom)
    >>> assert x.l.l == x and Wire('y').l == Wire('y')
    >>> assert Wire('y', 1) == Wire('y') and Wire('y', 1).l.l == Wire('y')
    """
    def __setstate__(self, state):
        super().__setstate__(state)
        self.z %= len(self.windings)

    def __init__(self, *args, **params):
        super().__init__(*args, **params)
        self.z %= len(self.windings)

    @property
    def l(self) -> Self:
        """ The object the other way round, from its codomain to its domain.
        """
        result = copy(self)
        result.dom, result.cod = self.cod, self.dom
        return result

    r = l
    windings = (0, )


HeadOb, TailOb = feedback.structural_wires(Wire)


@factory
class Ty(compact.Ty, markov.Ty, comarkov.Ty):
    """
    A frobenius type is a compact type with frobenius objects inside, a
    markov type and a comarkov one.

    Parameters:
        inside (frobenius.Wire) : The objects inside the type.
    """
    generator_factory = Wire

    def __call__(self, function: Callable | str) -> TermBase:
        """
        The kappa abstraction of a Python function from a variable for each
        wire of the type to a term of the relational machine calculus, see
        :class:`TermBase`: it binds the variables, see :meth:`exists`, then
        pops the stack into them, the last variable first, as the head of
        the stack is its last wire. A string is a constant, as in
        :class:`discopy.biclosed.Ty`.

        Parameters:
            function : The body, as a function of the variables.

        Example
        -------
        >>> x = Ty('x')
        >>> copy = x(lambda a: a >> a)
        >>> assert copy == x.exists(lambda a: a.dagger() >> a >> a)
        >>> assert Equation(copy.eval(), Spider(1, 2, x))
        """
        if isinstance(function, str):
            return super().__call__(function)
        return Exists.kappa(*self.bind(function))

    def exists(self, function: Callable) -> TermBase:
        """
        The binding of a fresh variable for each wire of the type in the
        term a Python function returns, the :math:`\\exists x. M` of the
        relational machine calculus, see :class:`Exists`.

        Parameters:
            function : The body, as a function of the variables.

        Example
        -------
        >>> x = Ty('x')
        >>> cap = x.exists(lambda a: a >> a)
        >>> assert Equation(cap.eval(), Spider(0, 2, x))
        """
        variables, body = self.bind(function)
        return Exists(variables, body) if variables else body

    def bind(self, function: Callable
             ) -> tuple[tuple[Variable, ...], TermBase]:
        """
        A variable for each wire of the type, named by the parameters of a
        Python function, and the term the function returns on them.
        """
        names = list(signature(function).parameters)
        if len(names) != len(self):
            raise ValueError(f"Expected {len(self)} variables for {self}, "
                             f"got {', '.join(names) or 'none'}.")
        variables = tuple(map(Variable, names, self))
        return variables, TermBase.cast(function(*variables))


@factory
class Nat(compact.Nat, markov.Nat, comarkov.Nat, Ty):
    """
    A ``Nat`` is a natural number ``n`` seen as a frobenius type with
    unnamed objects.

    Parameters
    ----------
    n : int
        The natural number.
    """

    l = r = property(lambda self: self)


Ty.nat_factory = Nat


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

    def converse(self) -> Diagram:
        """
        The relational converse of the diagram: its hypergraph with its
        domain and codomain exchanged, every box kept as it is, i.e. each
        wire of the boundary bent to the other side in the same order. It
        is the :meth:`TermBase.dagger` of the relational machine calculus,
        which keeps a box and pops what it pushed, so that the converse of
        a box is its rotation followed by the swap that restores the order
        of its wires, not its :meth:`dagger`.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> f = Box('f', x @ y, x)
        >>> assert Equation(f.converse(), f.rotate() >> Swap(y, x))
        >>> assert Equation(f.converse().converse(), f)
        """
        graph = self.to_hypergraph()
        return type(graph)(
            graph.cod, graph.dom, graph.boxes,
            (graph.cod_wires, graph.box_wires, graph.dom_wires),
            graph.spider_types, phases=graph.phases).to_diagram()

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
                f.unfuse() if isinstance(f, Spider)
                else f.to_spiders().unfuse() if isinstance(f, Cospan) else f,
            dom=self.factory, cod=self.factory)
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
        """ Whether the box is deterministic, never when it is rotated. """
        return super().is_deterministic and not self.z

    @property
    def is_affine(self) -> bool:
        """ Whether the box is affine, never when it is rotated. """
        return super().is_affine and not self.z

    @is_deterministic.setter
    def is_deterministic(self, value: bool):
        self.deterministic = value

    @is_affine.setter
    def is_affine(self, value: bool):
        self.affine = value

    def rotate(self, left=False):
        return self.flagged(super().rotate(left))

    def __call__(self, *args: Value) -> Value:
        """
        The box applied to values, a value of the relational machine
        calculus, see :class:`Application`, unless a class below calls a
        box otherwise, as :meth:`Diagram.from_callable` does while it runs.

        >>> x, y = Ty('x'), Ty('y')
        >>> f, a = Box('f', x, y), Variable('a', x)
        >>> assert f(a) == Application(f, (a, )) and f(a).cod == y
        """
        try:
            call = super().__call__
        except AttributeError:
            return Application(self, args)
        return call(*args)


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
        return factory_key(type(self)), self.apex, tuple(
            self.legs.left), tuple(
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
    deterministic = affine = True
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
            dom, cod, state["typ"] = (
                state['_dom'], state['_cod'], state.pop('_typ'))
            state["_name"] = type(self).__name__\
                + f"({dom.n}, {cod.n}, {state['typ']}{str_data})"
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

    @classmethod
    # pylint: disable-next=arguments-differ  # drawn with no boundary
    def strategy(cls, *, types=None):
        """
        Generate a spider on an atomic type with up to two legs a side,
        with or without a phase, and a phased one marked as a dagger or not.

        Parameters:
            types : The strategy to draw the types from, see
                :meth:`atomic_strategy`.
        """
        from hypothesis import strategies as st

        legs = st.sampled_from(((0, 1), (1, 0), (1, 2), (2, 1), (2, 2)))
        phases = st.sampled_from((None, 0.25, 0.5))
        return st.tuples(
            legs, cls.atomic_strategy(types), phases, st.booleans()).map(
                lambda args: cls(*args[0], *args[1:3], is_dagger=args[3]))

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
        result = cls.id(typ[:0]).tensor(*[
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

Feedback, Trace, FollowedBy, Head, Tail = feedback.structural_boxes(
    Diagram, Box)
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
Diagram.str_transparency = compact.Diagram.str_transparency.modulo(
    Diagram.to_hypergraph)
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id


class Equation(compact.Equation):
    """ The :class:`compact.Equation` of Frobenius diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)


DEFAULT = ""
""" The location of a push or a pop that names none: the default stack. """

type Typing = dict[str, tuple[Ty, Ty]]
""" The stack a term pops and the one it pushes, at each location. """


class TermBase(Serialisable):
    # pylint: disable=too-many-public-methods  # its laws are methods
    """
    A term of the relational machine calculus of :cite:t:`BarrettEtAl24`,
    a program for a machine with a stack of values at each location:

    - a :class:`Value`, i.e. a :class:`Variable` or an :class:`Application`
      of a box to values, used as a term is its push on the default stack,
      :math:`[t]`, and :meth:`Value.at` pushes it on another one,
      :math:`[t]a`, see :class:`Push`
    - its :meth:`dagger` is its pop :math:`\\langle t \\rangle`, which
      unifies the head of the stack with it, see :class:`Pop`
    - a :class:`Sequence` :math:`M ; N` is ``M >> N``, the empty one is
      skip :math:`\\star`, see :meth:`id`
    - an :class:`Exists` :math:`\\exists x. M` binds a fresh variable,
      see :meth:`Ty.exists`, and a type applied to a Python function is
      the kappa abstraction :math:`\\exists x. \\langle x \\rangle ; M`
    - a :class:`Choice` :math:`M + N` is ``M + N``, the empty one is failure
      :math:`0`, see :meth:`zero`, and a :class:`Star` :math:`M^*` is
      ``M.star()``

    The dagger of a term reverses it and exchanges push and pop. A term
    has a type at each location, see :attr:`typing`: the stack it pops and
    the one it pushes, the least of them, the head of each stack its last
    wire. A term evaluates to a diagram, see :meth:`eval`, by running the
    machine on spiders rather than values, see :class:`Machine`: the sum-free,
    star-free terms with no box are the free hypergraph category, Theorem
    6.5, except that a variable bound and never used is a spider with no
    leg, as in :class:`discopy.markov.TermBase` a discard is kept, and that
    the laws of unification hold of the values but not of the boxes they
    evaluate to. A diagram reads back as a term in normal form, see
    :meth:`from_diagram`, with any number of outputs.

    Attributes:
        typing (Typing) : The least type of the term at each location.
        dom (Ty) : The stacks it pops, see :meth:`boundary`.
        cod (Ty) : The stacks it pushes.

    Example
    -------
    >>> x = Ty('x')
    >>> frobenius = x(lambda a: a.dagger() >> a >> a)
    >>> print(frobenius)
    x(lambda a: a.dagger() >> a >> a)
    >>> assert frobenius.dom == x @ x and frobenius.cod == x @ x
    >>> assert Equation(frobenius.eval(), Spider(2, 2, x))
    >>> assert frobenius.dagger() == frobenius
    >>> print(x(lambda a: a >> a).dagger())
    x(lambda a: a.dagger() >> a)
    >>> print(Spider(1, 2, x).to_term())
    x(lambda x0: x0 >> x0)
    """
    typing: Typing
    dom: Ty
    cod: Ty
    serialised_attrs: ClassVar[tuple[str, ...]]

    @property
    def parts(self) -> tuple:
        """ What the term is built from, in the order of its arguments. """
        return tuple(getattr(self, attr) for attr in self.serialised_attrs)

    def __eq__(self, other):
        return type(self) is type(other) and self.parts == other.parts

    def __hash__(self):
        return hash((type(self).__name__, ) + self.parts)

    def __repr__(self):
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self.parts))})"

    def to_tree(self) -> dict:
        """ The tree of a term: its factory and the trees of its parts. """
        return {"factory": factory_name(type(self)), **{
            attr: [x.to_tree() for x in part] if isinstance(part, tuple)
            else part.to_tree() if hasattr(part, "to_tree") else part
            for attr, part in zip(self.serialised_attrs, self.parts)}}

    @classmethod
    def from_tree(cls, tree: dict) -> Self:
        """ The term of a tree, see :meth:`to_tree`. """
        return cls(*(
            tuple(map(from_tree, part)) if isinstance(part, list)
            else from_tree(part) if isinstance(part, dict) else part
            for part in (tree[attr] for attr in cls.serialised_attrs)))

    @staticmethod
    def cast(term: TermBase) -> TermBase:
        """ A term, raising :class:`TypeError` on anything else. """
        if not isinstance(term, TermBase):
            raise TypeError(f"Expected a term, got {term!r} instead.")
        return term

    @staticmethod
    def boundary(typing: Typing) -> tuple[Ty, Ty]:
        """
        The stacks a typing pops and pushes, one location after the other,
        the default first and then the others by name.
        """
        return (Ty().tensor(*(typing[key][0] for key in sorted(typing))),
                Ty().tensor(*(typing[key][1] for key in sorted(typing))))

    @staticmethod
    def compose(first: Typing, second: Typing) -> Typing:
        """
        The least typing of one term then another: at each location, what
        the second pops and the first did not push is popped first, below.

        >>> x, y = Ty('x'), Ty('y')
        >>> first, second = {DEFAULT: (x, x @ y)}, {DEFAULT: (x @ x @ y, y)}
        >>> assert TermBase.compose(first, second) == {DEFAULT: (x @ x, y)}
        """
        empty = (Ty(), Ty())
        typing = {}
        for location in sorted(set(first) | set(second)):
            (dom, mid), (top, cod) = (
                typing.get(location, empty) for typing in (first, second))
            n = min(len(mid), len(top))
            if mid[len(mid) - n:] != top[len(top) - n:]:
                raise TypeError(f"Expected {top} at location {location!r} "
                                f"to end as {mid} does.")
            typing[location] = (
                top[:len(top) - n] @ dom, mid[:len(mid) - n] @ cod)
        return typing

    @staticmethod
    def join(first: Typing, second: Typing) -> Typing:
        """
        The least typing of a choice between two terms: at each location,
        that of one of them, the other's with the same stack below.

        >>> x, y = Ty('x'), Ty('y')
        >>> first, second = {DEFAULT: (x @ y, x @ x)}, {DEFAULT: (y, x)}
        >>> assert TermBase.join(first, second) == first
        """
        empty = (Ty(), Ty())
        typing = {}
        for location in sorted(set(first) | set(second)):
            (dom, cod), (other_dom, other_cod) = sorted(
                (typing.get(location, empty) for typing in (first, second)),
                key=lambda pair: -len(pair[0]))
            below = dom[:len(dom) - len(other_dom)]
            if (dom, cod) != (below @ other_dom, below @ other_cod):
                raise TypeError(f"Expected {other_dom} -> {other_cod} to "
                                f"extend to {dom} -> {cod} at {location!r}.")
            typing[location] = (dom, cod)
        return typing

    def then(self, *others: TermBase) -> TermBase:
        """
        The sequence of the term then others, ``>>``, see :class:`Sequence`,
        the term itself when there is only one.
        """
        terms = [term for term in (self, *map(self.cast, others))
                 for term in (term.terms if isinstance(term, Sequence)
                              else (term, ))]
        return terms[0] if len(terms) == 1 else Sequence(tuple(terms))

    __rshift__ = then

    def __add__(self, other: TermBase) -> Choice:
        """ The choice between two terms, see :class:`Choice`. """
        return Choice(tuple(
            choice for term in (self, self.cast(other))
            for choice in (term.terms if isinstance(term, Choice)
                           and term.terms else (term, ))))

    def star(self) -> Star:
        """ The Kleene star of the term, see :class:`Star`. """
        return Star(self)

    @staticmethod
    def id() -> Sequence:
        """ Skip, the empty sequence. """
        return Sequence(())

    @staticmethod
    def zero() -> Choice:
        """ Failure, the empty choice. """
        return Choice(())

    @abstractmethod
    def dagger(self) -> TermBase:
        """
        The term reversed with its pushes and pops exchanged, an involution
        on syntax which evaluates to the :meth:`Diagram.converse`.
        """

    def atoms(self) -> list:
        """
        The generators the string of the term names: its free variables,
        its boxes and the wires of the types of its variables.
        """
        return []

    @property
    def freevars(self) -> list[Variable]:
        """ The free variables of the term, in order of occurrence. """
        return list(dict.fromkeys(
            atom for atom in self.atoms() if isinstance(atom, Variable)))

    def branches(self) -> list[TermBase]:
        """
        The sum-free terms whose choice is the term, raising
        :class:`ValueError` on a star, which has no finite one.
        """
        return [self]

    @abstractmethod
    def run(self, machine: Machine, scope: dict[Variable, int]) -> None:
        """
        Run the sum-free, star-free term on a machine, the spider of each
        variable bound around it in a scope.
        """

    def to_hypergraph(self, dom: dict[str, Ty] | None = None) -> Hypergraph:
        """
        The hypergraph the machine builds by running the sum-free term on a
        stack of spiders at each location, see :class:`Machine`, those of
        its own domain or of a larger one.
        """
        dom = dom or {location: typ for location, (typ, _) in sorted(
            self.typing.items())}
        machine = Machine(dom)
        self.run(machine, {})
        return machine.to_hypergraph()

    def eval(self, functor: Functor | None = None) -> Diagram:
        """
        The diagram of the term, a formal sum of those of its branches when
        it has several or none, under a functor if any. A free variable is a
        spider of no boundary: the term evaluates as if it bound it.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> f, g = Box('f', x, y), Box('g', y @ y, x)
        >>> term = x(lambda a: g(f(a), f(a)))
        >>> assert Equation(term.eval(), Spider(1, 2, x) >> f @ f >> g)
        >>> top = (x @ x).exists(lambda a, b: a.dagger() >> b)
        >>> assert Equation(top.eval(), Spider(1, 0, x) >> Spider(0, 1, x))
        >>> assert (term + top).eval() == Sum((term.eval(), top.eval()))
        """
        dom = {location: typ for location, (typ, _) in self.typing.items()}
        diagrams = [branch.to_hypergraph(dom).to_diagram()
                    for branch in self.branches()]
        diagram = diagrams[0] if len(diagrams) == 1\
            else Diagram.sum_factory(tuple(diagrams), self.dom, self.cod)
        return diagram if functor is None else functor(diagram)

    def draw(self, **params):
        """ Draw the diagram of the term, see :meth:`eval`. """
        return self.eval().draw(**params)

    @classmethod
    def from_hypergraph(cls, graph: Hypergraph) -> TermBase:
        """
        The term in normal form of a hypergraph, Corollary 4.9 with boxes,
        on the default stack: a variable ``x{i}`` for each spider ``i``,
        the pops of the domain, each box applied to its inputs and popped
        into its outputs, the pushes of the codomain. It is a kappa
        abstraction of the domain when no two of its wires share a spider.

        >>> x, y = Ty('x'), Ty('y')
        >>> print(TermBase.from_diagram(Box('f', x, y)))
        x(lambda x0: y.exists(lambda x1: f(x0) >> x1.dagger() >> x1))
        >>> print(TermBase.from_diagram(Spider(2, 0, x)))
        x(lambda x0: x0.dagger())
        """
        variables = tuple(
            Variable(f"x{i}", typ) for i, typ in enumerate(graph.spider_types))
        inputs = tuple(variables[spider] for spider in graph.dom_wires)
        body = tuple(term for box, (ins, outs) in zip(
            graph.boxes, graph.box_wires) for term in (
                Application(box, tuple(variables[spider] for spider in ins)),
                *(variables[spider].dagger() for spider in reversed(outs))))\
            + tuple(variables[spider] for spider in graph.cod_wires)
        bound = inputs if len(set(inputs)) == len(inputs) else ()
        rest = tuple(
            variable for variable in variables if variable not in bound)
        pops = () if bound else tuple(map(Pop, reversed(inputs)))
        inner = cls.id().then(*pops, *body)
        inner = Exists(rest, inner) if rest else inner
        return Exists.kappa(bound, inner) if bound else inner

    @classmethod
    def from_diagram(cls, diagram: Diagram) -> TermBase:
        """ The term of a diagram, see :meth:`from_hypergraph`. """
        return cls.from_hypergraph(diagram.to_hypergraph())

    def normal_form(self) -> TermBase:
        """
        The choice of the normal forms of the branches, on the default
        stack, see :meth:`from_hypergraph`: the term up to the laws of the
        free hypergraph category its diagram satisfies.
        """
        dom = {location: typ for location, (typ, _) in self.typing.items()}
        forms = [self.from_hypergraph(branch.to_hypergraph(dom))
                 for branch in self.branches()]
        return forms[0] if len(forms) == 1 else Choice(tuple(forms))

    def expand(self) -> TermBase:
        """
        The term then the identity on what it pushes, as a kappa
        abstraction at each location that pops and pushes it back.
        """
        identities = (Exists.kappa(variables, TermBase.id().then(*(
            variable.at(location) for variable in variables)), location)
            for location, (_, cod) in sorted(self.typing.items()) if cod
            for variables in [tuple(
                Variable(f"eta{i}", wire) for i, wire in enumerate(cod))])
        return self.then(*identities)

    @classmethod
    def strategy(cls, **params) -> st.SearchStrategy[TermBase]:
        """
        Generate closed terms with no choice and no star: an :class:`Exists`
        of variables of types ``x`` and ``y`` around pushes and pops of
        them and of boxes applied to them, on the default stack and on
        ``"a"``, each pop of what the stack holds.
        """
        # pylint: disable=unused-argument  # the matrix passes its params
        from hypothesis import strategies as st
        wires = (Ty('x'), Ty('y'))

        @st.composite
        def terms(draw):
            variables = tuple(Variable(f"v{i}", draw(st.sampled_from(wires)))
                              for i in range(draw(st.integers(1, 3))))
            stacks, body = {DEFAULT: [], "a": []}, []
            for i in range(draw(st.integers(0, 6))):
                location = draw(st.sampled_from(sorted(stacks)))
                stack = stacks[location]
                kind = draw(st.sampled_from(["push", "pop", "box", "unbox"]))
                if kind == "box" or kind == "unbox" and stack:
                    args = draw(st.lists(st.sampled_from(variables),
                                         max_size=2))
                    k = draw(st.integers(0, min(len(stack), 2)))
                    cod = stack[len(stack) - k:] if kind == "unbox" else draw(
                        st.lists(st.sampled_from(wires), max_size=2))
                    box = Box(f"f{i}", Ty().tensor(*(
                        arg.cod for arg in args)), Ty().tensor(*cod))
                    value = Application(box, tuple(args)).at(location)
                    if kind == "unbox":
                        del stack[len(stack) - k:]
                        value = value.dagger()
                    else:
                        stack.extend(cod)
                    body.append(value)
                    continue
                if kind in ("push", "unbox"):
                    variable = draw(st.sampled_from(variables))
                    body.append(variable.at(location))
                    stack.append(variable.cod)
                    continue
                variable = draw(st.sampled_from([
                    variable for variable in variables
                    if not stack or variable.cod == stack[-1]] or variables))
                if stack and variable.cod != stack[-1]:
                    continue
                body.append(variable.at(location).dagger())
                stack[-1:] = []
            return Exists(variables, cls.id().then(*body))
        return terms()

    @axiom
    def eta(cls, term: Self) -> AbstractEquation:
        """
        Eta, :math:`\\exists x. \\langle x \\rangle ; [x] = \\star` on what
        a term pushes: a term and its :meth:`expand` evaluate alike.
        """
        return Equation(term.expand().eval(), term.eval())

    @axiom
    def normal_form_soundness(cls, term: Self) -> AbstractEquation:
        """ A term and its normal form evaluate to equal diagrams. """
        return Equation(term.normal_form().eval(), term.eval())

    @axiom
    def normal_form_idempotence(cls, term: Self) -> AbstractEquation:
        """ A normal form is its own normal form, on the nose. """
        normal_form = term.normal_form()
        return AbstractEquation(normal_form.normal_form(), normal_form)

    @axiom
    def reading(cls, term: Self) -> AbstractEquation:
        """
        Reading is a section of evaluation: the diagram of a term reads as
        a term that evaluates back to it.
        """
        diagram = term.eval()
        return Equation(cls.from_diagram(diagram).eval(), diagram)

    @axiom
    def dagger_involution(cls, term: Self) -> AbstractEquation:
        """ The dagger of a term is an involution on the nose. """
        return AbstractEquation(term.dagger().dagger(), term)

    @axiom
    def dagger_soundness(cls, term: Self) -> AbstractEquation:
        """ The dagger of a term evaluates to the converse of its diagram. """
        return Equation(term.dagger().eval(), term.eval().converse())

    @axiom
    def functoriality(cls, term: Self) -> AbstractEquation:
        """
        Evaluation is a functor on closed terms, here a term then its
        dagger, which composes with it at every location.
        """
        return Equation((term >> term.dagger()).eval(),
                        term.eval() >> term.dagger().eval())


class Value(TermBase):
    """
    An algebraic term of the relational machine calculus, a value the
    machine pushes and pops: a :class:`Variable` or an :class:`Application`.
    Used as a term, a value is its push on the default stack.
    """

    @property
    def typing(self) -> Typing:
        """ A value pushes its type on the default stack. """
        return {DEFAULT: (Ty(), self.cod)}

    def at(self, location: str = DEFAULT) -> TermBase:
        """ The push of the value on a location, see :class:`Push`. """
        return self if location == DEFAULT else Push(self, location)

    def dagger(self) -> Pop:
        return Pop(self)

    def run(self, machine, scope):
        self.push(machine, DEFAULT, scope)

    @abstractmethod
    def push(self, machine: Machine, location: str, scope: dict) -> None:
        """ Push the spiders of the value on a stack of the machine. """

    @abstractmethod
    def pop(self, machine: Machine, location: str, scope: dict) -> None:
        """ Pop the spiders of the value from a stack of the machine. """


class Variable(Value):
    """
    A variable of a type of one wire, a spider when the machine runs.

    Parameters:
        name : The name of the variable.
        cod : Its type.
    """
    serialised_attrs = ("name", "cod")

    def __init__(self, name: str, cod: Ty):
        assert_isatomic(cod, Ty)
        self.name, self.dom, self.cod = name, Ty(), cod

    def __str__(self):
        return self.name

    def atoms(self):
        return [self]

    def push(self, machine, location, scope):
        machine.stack(location).append(machine.lookup(self, scope))

    def pop(self, machine, location, scope):
        machine.unify(
            machine.stack(location).pop(), machine.lookup(self, scope))


class Application(Value):
    """
    A box applied to values, an algebraic term :math:`f(t_1, \\dots, t_n)`
    with the box as function symbol, of any number of outputs: its push
    pushes the arguments then applies the box, its pop pops the outputs of
    the box and unifies its inputs with the arguments, i.e. applies its
    :meth:`Diagram.converse`.

    Parameters:
        box : The box.
        args : The values it is applied to, as many wires as it takes.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, a, b = Box('f', x @ y, x), Box('a', Ty(), x), Box('b', Ty(), y)
    >>> unify = (x @ y).exists(lambda u, v: f(u, b()) >> f(a(), v).dagger())
    >>> print(unify)
    (x @ y).exists(lambda u, v: f(u, b()) >> f(a(), v).dagger())
    >>> assert Equation(unify.eval(), Spider(0, 1, x) @ b >> f\\
    ...     >> f.converse() >> a.converse() @ Spider(1, 0, y))
    """
    serialised_attrs = ("box", "args")

    def __init__(self, box: Box, args: tuple[Value, ...] = ()):
        dom = Ty().tensor(*(arg.cod for arg in args))
        if dom != box.dom:
            raise TypeError(f"Expected arguments of type {box.dom} for "
                            f"{box}, got {dom} instead.")
        self.box, self.args = box, tuple(args)
        self.dom, self.cod = Ty(), box.cod

    def __str__(self):
        return f"{self.box}({', '.join(map(str, self.args))})"

    def atoms(self):
        return self.box.atoms() + [
            atom for arg in self.args for atom in arg.atoms()]

    def push(self, machine, location, scope):
        for arg in self.args:
            arg.push(machine, location, scope)
        machine.apply(self.box, location)

    def pop(self, machine, location, scope):
        machine.apply(self.box, location, converse=True)
        for arg in reversed(self.args):
            arg.pop(machine, location, scope)


class Push(TermBase):
    """
    The push :math:`[t]a` of a value on a location other than the default
    one, where a value used as a term is its own push, see :meth:`Value.at`.

    Parameters:
        value : The value pushed.
        location : The name of the stack.

    Example
    -------
    >>> x = Ty('x')
    >>> cells = (x @ x).exists(lambda u, v: u.at('a').dagger()
    ...     >> v.at('b').dagger() >> u.at('b') >> v.at('a'))
    >>> assert cells.dom == x @ x and Equation(cells.eval(), Swap(x, x))
    """
    serialised_attrs = ("value", "location")

    def __init__(self, value: Value, location: str):
        if location == DEFAULT:
            raise ValueError("Expected a location, a value pushes itself "
                             "on the default stack.")
        self.value, self.location = value, location
        self.dom, self.cod = self.boundary(self.typing)

    @property
    def typing(self) -> Typing:
        """ A push pushes the type of its value on its location. """
        return {self.location: (Ty(), self.value.cod)}

    def __str__(self):
        return f"{self.value}.at({self.location!r})"

    def atoms(self):
        return self.value.atoms()

    def dagger(self):
        return Pop(self.value, self.location)

    def run(self, machine, scope):
        self.value.push(machine, self.location, scope)


class Pop(TermBase):
    """
    The pop :math:`a \\langle t \\rangle` of a value from a location, the
    :meth:`dagger` of its push: the head of the stack is unified with it.

    Parameters:
        value : The value popped.
        location : The name of the stack, the default one if none.
    """
    serialised_attrs = ("value", "location")

    def __init__(self, value: Value, location: str = DEFAULT):
        self.value, self.location = value, location
        self.dom, self.cod = self.boundary(self.typing)

    @property
    def typing(self) -> Typing:
        """ A pop pops the type of its value from its location. """
        return {self.location: (self.value.cod, Ty())}

    def __str__(self):
        return f"{self.value.at(self.location)}.dagger()"

    def atoms(self):
        return self.value.atoms()

    def dagger(self):
        return self.value.at(self.location)

    def run(self, machine, scope):
        self.value.pop(machine, self.location, scope)


class Sequence(TermBase):
    """
    The sequence :math:`M_1 ; \\dots ; M_n` of terms, ``M1 >> ... >> Mn``,
    of none or at least two, flat: skip :math:`\\star` when empty.

    Parameters:
        terms : The terms in order.
    """
    serialised_attrs = ("terms", )

    def __init__(self, terms: tuple[TermBase, ...] = ()):
        if len(terms) == 1 or any(isinstance(t, Sequence) for t in terms):
            raise ValueError("Expected a flat sequence of none or several "
                             "terms, see TermBase.then.")
        self.terms = tuple(map(self.cast, terms))
        self.typing = reduce(
            self.compose, (term.typing for term in self.terms), {})
        self.dom, self.cod = self.boundary(self.typing)

    def __str__(self):
        return " >> ".join(
            f"({term})" if isinstance(term, Choice) and term.terms
            else str(term) for term in self.terms) or "Sequence()"

    def atoms(self):
        return [atom for term in self.terms for atom in term.atoms()]

    def dagger(self):
        return self.id().then(*(term.dagger() for term in self.terms[::-1]))

    def branches(self):
        return [self.id().then(*terms) for terms in product(
            *(term.branches() for term in self.terms))]

    def run(self, machine, scope):
        for term in self.terms:
            term.run(machine, scope)


class Choice(TermBase):
    """
    The choice :math:`M_1 + \\dots + M_n` between terms, ``M1 + ... + Mn``,
    of none or at least two, flat: failure :math:`0` when empty. It
    evaluates to the formal sum of the diagrams of its branches.

    Parameters:
        terms : The terms to choose from.
    """
    serialised_attrs = ("terms", )

    def __init__(self, terms: tuple[TermBase, ...] = ()):
        if len(terms) == 1 or any(
                isinstance(t, Choice) and t.terms for t in terms):
            raise ValueError("Expected a flat choice of none or several "
                             "terms, see TermBase.__add__.")
        self.terms = tuple(map(self.cast, terms))
        self.typing = reduce(
            self.join, (term.typing for term in self.terms), {})
        self.dom, self.cod = self.boundary(self.typing)

    def __str__(self):
        return " + ".join(
            f"({term})" if isinstance(term, Sequence) and term.terms
            else str(term) for term in self.terms) or "Choice()"

    def atoms(self):
        return [atom for term in self.terms for atom in term.atoms()]

    def dagger(self):
        return Choice(tuple(term.dagger() for term in self.terms))

    def branches(self):
        return [branch for term in self.terms for branch in term.branches()]

    def run(self, machine, scope):
        raise ValueError("Expected a term with no choice, see branches.")


class Star(TermBase):
    """
    The Kleene star :math:`M^*` of a term, ``M.star()``, any number of
    sequences of it: it pushes what it pops at each location, and has no
    diagram, the machine running it for ever.

    Parameters:
        body : The term repeated.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, x)
    >>> print(x(lambda a: f(a)).star())
    x(lambda a: f(a)).star()
    >>> f(Variable('a', x)).star()
    Traceback (most recent call last):
        ...
    TypeError: Expected the same type at '' in and out, got Ty() -> x.
    """
    serialised_attrs = ("body", )

    def __init__(self, body: TermBase):
        self.body = self.cast(body)
        for location, (dom, cod) in body.typing.items():
            if dom != cod:
                raise TypeError(f"Expected the same type at {location!r} "
                                f"in and out, got {dom} -> {cod}.")
        self.typing = body.typing
        self.dom, self.cod = self.boundary(self.typing)

    def __str__(self):
        return f"({self.body}).star()" if isinstance(
            self.body, (Sequence, Choice)) and self.body.terms\
            else f"{self.body}.star()"

    def atoms(self):
        return self.body.atoms()

    def dagger(self):
        return Star(self.body.dagger())

    def branches(self):
        raise ValueError(f"Expected a term with no star, got {self}.")

    def run(self, machine, scope):
        self.branches()


class Exists(TermBase):
    """
    The binding :math:`\\exists x_1 \\dots x_n. M` of fresh variables, as
    many spiders, in a term, see :meth:`Ty.exists`, a kappa abstraction
    when the term starts with their pops, see :meth:`kappa`.

    Parameters:
        variables : The variables bound, at least one.
        body : The term in which they are bound.

    Example
    -------
    >>> x = Ty('x')
    >>> top = (x @ x).exists(lambda a, b: a.dagger() >> b)
    >>> assert top == Exists((Variable('a', x), Variable('b', x)), top.body)
    >>> print(top, top.dagger(), sep='\\n')
    (x @ x).exists(lambda a, b: a.dagger() >> b)
    (x @ x).exists(lambda a, b: b.dagger() >> a)
    """
    serialised_attrs = ("variables", "body")

    def __init__(self, variables: tuple[Variable, ...], body: TermBase):
        if not variables:
            raise ValueError(f"Expected variables to bind in {body}.")
        self.variables, self.body = tuple(variables), self.cast(body)
        self.typing = body.typing
        self.dom, self.cod = self.boundary(self.typing)

    @staticmethod
    def kappa(variables: tuple[Variable, ...], body: TermBase,
              location: str = DEFAULT) -> TermBase:
        """
        The kappa abstraction of variables in a term, which pops them from
        a location, the last one first, then runs the term, the
        :math:`\\kappa x. M = \\exists x. \\langle x \\rangle ; M` of
        section 3.3, see :meth:`Ty.__call__`.
        """
        pops = (variable.at(location).dagger() for variable in variables)
        term = TermBase.id().then(*reversed(tuple(pops)), body)
        return Exists(tuple(variables), term) if variables else term

    def __str__(self):
        names = ", ".join(variable.name for variable in self.variables)
        dom = Ty().tensor(*(variable.cod for variable in self.variables))
        dom = str(dom) if len(dom) == 1 else f"({dom})"
        terms = self.body.terms if isinstance(self.body, Sequence)\
            else (self.body, )
        pops = tuple(map(Pop, reversed(self.variables)))
        if terms[:len(pops)] != pops:
            return f"{dom}.exists(lambda {names}: {self.body})"
        rest = terms[len(pops):]
        body = rest[0] if len(rest) == 1 else Sequence(rest)
        return f"{dom}(lambda {names}: {body})"

    def atoms(self):
        return [wire for variable in self.variables for wire in variable.cod]\
            + [atom for atom in self.body.atoms()
               if atom not in self.variables]

    def dagger(self):
        return Exists(self.variables, self.body.dagger())

    def branches(self):
        return [Exists(self.variables, branch)
                for branch in self.body.branches()]

    def run(self, machine, scope):
        self.body.run(machine, scope | {
            variable: machine.spider(variable.cod)
            for variable in self.variables})


type Term = Variable | Application | Push | Pop | Sequence | Choice | Star\
    | Exists


class Machine:
    """
    The relational machine of :cite:t:`BarrettEtAl24` on spiders: a stack
    of spiders at each location, where a push of a variable pushes its
    spider, a pop unifies the head of the stack with it, i.e. merges their
    spiders, a box pops its inputs and pushes a spider for each output, and
    its converse the other way round, so that what it builds is the
    :class:`Hypergraph` of the term it runs, see :meth:`TermBase.eval`.

    Parameters:
        dom : The stack each location starts with, one spider per wire.

    Attributes:
        parent (list[int]) : The spider each spider was merged into, if any.
        types (list[Ty]) : The type of each spider.
        boxes (list[Box]) : The boxes applied, in order.
        box_wires (list[tuple[list[int], list[int]]]) : Their spiders.
        stacks (dict[str, list[int]]) : The stack at each location.
        dom_wires (list[int]) : The spiders the stacks started with.
        free (dict[Variable, int]) : The spider of each free variable.
    """
    def __init__(self, dom: dict[str, Ty]):
        self.parent, self.types, self.boxes, self.box_wires = [], [], [], []
        self.free = {}
        self.stacks = {location: list(map(self.spider, typ))
                       for location, typ in sorted(dom.items())}
        self.dom_wires = [spider for location in sorted(self.stacks)
                          for spider in self.stacks[location]]

    def spider(self, typ: Ty) -> int:
        """ A fresh spider of a type of one wire. """
        self.parent.append(len(self.parent))
        self.types.append(typ)
        return len(self.parent) - 1

    def find(self, spider: int) -> int:
        """ The spider a spider was merged into. """
        while self.parent[spider] != spider:
            spider = self.parent[spider]
        return spider

    def unify(self, spider: int, other: int) -> None:
        """ Merge a spider into another. """
        self.parent[self.find(spider)] = self.find(other)

    def stack(self, location: str) -> list[int]:
        """ The stack of a location, empty when it is first used. """
        return self.stacks.setdefault(location, [])

    def lookup(self, variable: Variable, scope: dict) -> int:
        """ The spider of a variable, bound in a scope or else free. """
        if variable in scope:
            return scope[variable]
        if variable not in self.free:
            self.free[variable] = self.spider(variable.cod)
        return self.free[variable]

    def apply(self, box: Box, location: str, converse: bool = False):
        """ Apply a box, or its converse, to the head of a stack. """
        stack = self.stack(location)
        dom, cod = (box.cod, box.dom) if converse else (box.dom, box.cod)
        args = stack[len(stack) - len(dom):]
        del stack[len(stack) - len(dom):]
        outs = list(map(self.spider, cod))
        stack.extend(outs)
        self.boxes.append(box)
        self.box_wires.append((outs, args) if converse else (args, outs))

    def to_hypergraph(self) -> Hypergraph:
        """ The hypergraph built so far, from the start to the stacks. """
        find = self.find
        cod_wires = [spider for location in sorted(self.stacks)
                     for spider in self.stacks[location]]
        return Hypergraph(
            Ty().tensor(*(self.types[s] for s in self.dom_wires)),
            Ty().tensor(*(self.types[s] for s in cod_wires)),
            tuple(self.boxes), (
                tuple(map(find, self.dom_wires)), tuple(
                    (tuple(map(find, ins)), tuple(map(find, outs)))
                    for ins, outs in self.box_wires),
                tuple(map(find, cod_wires))),
            {find(s): typ for s, typ in enumerate(self.types)})


Diagram.term_factory = TermBase


__getattr__ = deprecated_alias(__name__, {"Ob": "Wire", "PRO": "Nat"})
