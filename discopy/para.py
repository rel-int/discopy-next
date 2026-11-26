# -*- coding: utf-8 -*-

"""
The category of parametric maps over a symmetric underlying `category`.

A parametric map from `x` to `y` with parameter space `p` is a morphism
`x @ p -> y` in the underlying category. Composition tensors the parameters
and tensor routes them to the right with a swap. Parametric maps first
appeared in the study of supervised learning :cite:p:`FongEtAl19`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Symmetric
    Reparam
    UpToPermutation
    isomorphic
    labelled
    profile
    permutations
    BOUND
    Traced
    Markov
    Comarkov
    Closed
    Feedback
    Compact
    Hypergraph

Axioms
------

Composition tensors the parameter spaces:

>>> from discopy.symmetric import Ty, Box, Diagram
>>> x, y, z, w, p, q = map(Ty, "xyzwpq")
>>> f = Symmetric(x, y, Box('f', x @ p, y), p)
>>> g = Symmetric(y, z, Box('g', y @ q, z), q)
>>> assert (f >> g).param == p @ q
>>> assert (f >> g).inside == f.inside @ q >> g.inside
>>> (f >> g).inside.draw(doctest="docs/_static/para/then.svg")

.. image:: /_static/para/then.svg
    :align: center

So does the tensor, with a swap routing the parameters to the right:

>>> h = Symmetric(z, w, Box('h', z @ q, w), q)
>>> assert (f @ h).param == p @ q
>>> assert (f @ h).inside\\
...     == x @ Diagram.swap(z, p) @ q >> f.inside @ h.inside
>>> (f @ h).inside.draw(doctest="docs/_static/para/tensor.svg")

.. image:: /_static/para/tensor.svg
    :align: center

Reparametrisation precomposes the parameters, contravariantly:

>>> p_, p__ = Ty("p'"), Ty("p''")
>>> r, s = Box('r', p_, p), Box('s', p__, p_)
>>> assert f.reparam(r).param == p_
>>> assert f.reparam(s >> r) == f.reparam(r).reparam(s)

A reparametrisation is a 2-cell, a :class:`Reparam` between parallel maps, and
the laws that fail on the nose hold up to the invertible ones, i.e. the
permutations of the parameters and of the coparameters, see
:class:`UpToPermutation`, decided by an isomorphism of hypergraphs, see
:func:`isomorphic`; the search of :class:`Reparam` for them is bounded by
``BOUND = 6`` parameter wires, see :func:`permutations`:

>>> alpha = Reparam.of(f, r)
>>> assert alpha.cod == f.reparam(r) and not alpha.is_invertible
>>> assert (alpha >> Reparam.id(g)).cod == f.reparam(r) >> g

The identity and swap of :class:`Symmetric` are those of the underlying
category, with the empty parameter space:

>>> assert Symmetric.id(x) == Symmetric.lift(Diagram.id(x))
>>> assert Symmetric.swap(x, y) == Symmetric.lift(Diagram.swap(x, y))
>>> t = Traced(x @ y, z @ y, Box('t', x @ y @ p, z @ y), p)
>>> assert t.trace().dom == x and t.trace().param == p

The construction preserves each level of the hierarchy below symmetric:
:class:`Traced`, :class:`Markov`, :class:`Comarkov`, :class:`Closed`,
:class:`Feedback`, :class:`Compact` and :class:`Hypergraph` lift the extra
structure of their underlying category with the empty parameter space, e.g.

>>> from discopy import frobenius
>>> X = frobenius.Ty('x')
>>> assert Hypergraph.spiders(1, 2, X)\\
...     == Hypergraph.lift(frobenius.Diagram.spiders(1, 2, X))

while the operations on morphisms swap the parameters out of the way,
the same as :meth:`Traced.trace`:

>>> from discopy import closed
>>> a, b, c, P = map(closed.Ty, "abcP")
>>> k = Closed(a @ b, c, closed.Box('k', a @ b @ P, c), P)
>>> assert k.curry(left=True).cod == c << b
>>> assert k.curry(left=False).cod == a >> c

A map may also carry a coparameter space `copar` on the codomain, i.e.
`inside : dom @ param -> cod @ copar`, empty by default — the type of one
time step of a stateful morphism sequence :cite:p:`DiLavoreEtAl22`, i.e. of
a :class:`Stream <discopy.stream.Stream>` with the delay forgotten.
Composition and tensor accumulate the hidden objects on both sides:

>>> m, n = Ty('m'), Ty('n')
>>> t = Symmetric(x, y, Box('t', x @ p, y @ m), p, m)
>>> u = Symmetric(y, z, Box('u', y @ q, z @ n), q, n)
>>> assert (t >> u).param == p @ q and (t >> u).copar == m @ n
>>> (t >> u).inside.draw(doctest="docs/_static/para/stateful-then.svg")

.. image:: /_static/para/stateful-then.svg
    :align: center

Coparametric maps, studied in categorical cybernetics
:cite:p:`CapucciEtAl21`, are the case of an empty `param`, composed by
accumulating the coparameters in forward order:

>>> f_ = Symmetric(x, y, Box("f'", x, y @ m), copar=m)
>>> g_ = Symmetric(y, z, Box("g'", y, z @ n), copar=n)
>>> assert (f_ >> g_).copar == m @ n
>>> (f_ >> g_).inside.draw(doctest="docs/_static/para/copara-then.svg")

.. image:: /_static/para/copara-then.svg
    :align: center

Recoparametrisation post-composes the coparameters, covariantly where
:meth:`Symmetric.reparam` is contravariant:

>>> assert t.recopar(Box('c', m, n)).copar == n

and the diagonal `param == copar` is closed under composition: it is the
free category with feedback of :cite:t:`KatisEtAl02`.

>>> s = Ty('s')
>>> v = Symmetric(x, y, Box('v', x @ s, y @ s), s, s)
>>> w = Symmetric(y, z, Box('w', y @ s, z @ s), s, s)
>>> assert (v >> w).param == (v >> w).copar == s @ s

This is not the state monad: para(copara(C)) is *not* the same as
State(C, S): in the first case the composition of two maps tensors their
co/parameters while in the second it chains them through. Each map above
brings its own hidden space and `v >> w` carries `s @ s`, where the state
category of maps `x @ s -> y @ s` in the underlying category chains one
fixed `s` through, `w` reading the state `v` wrote:

>>> state = v.inside >> w.inside
>>> assert state.dom == x @ s and state.cod == z @ s

Iterating a map on its own memory is the feedback operator of
:mod:`discopy.feedback`, which :mod:`discopy.stream` unrolls by chaining
the memory from one time step to the next.

Example
-------

Parametric maps compose like layers of a neural network, e.g. over
:class:`Function <discopy.python.Function>` with weight and bias parameters:

>>> from discopy.python import Function, Ty as PyTy
>>> layer = Symmetric[Function](PyTy(float), PyTy(float),
...     Function(lambda x, w, b: w * x + b, PyTy(float) ** 3, PyTy(float)),
...     param=PyTy(float, float))
>>> network = layer >> layer
>>> assert network.param == PyTy(float) ** 4
>>> network.inside(2., 3., 1., .5, 0.)
3.5
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

from networkx.algorithms.isomorphism import is_isomorphic

from discopy import (
    symmetric, markov, comarkov, closed, compact, frobenius, messages)
from discopy.testable import Equation, TestableGeneric, quotient
from discopy.abc import (
    ClosedCategory, ComarkovCategory, CompactCategory, FeedbackCategory,
    HypergraphCategory, MarkovCategory, SymmetricCategory, TracedCategory)
from discopy.utils import (
    AxiomError, assert_iscomposable, assert_isinstance, classproperty,
    factory_name, from_tree, get_origin, resolve_factory, unbiased)

BOUND = 6
""" The most parameter wires :func:`permutations` searches. """


def indices(dom: tuple, cod: tuple, free: tuple | None = None) -> Iterator:
    """
    The orderings `xs` of the positions of `dom` with ``dom[xs[i]] == cod[i]``,
    yielded lazily in lexicographic order, so the identity first if any.

    Parameters:
        dom : The wires to reorder.
        cod : The wires they have to land on.
        free : The positions of `dom` not yet used, all by default.

    Example
    -------
    >>> assert list(indices("aab", "aba")) == [(0, 2, 1), (1, 2, 0)]
    >>> assert not list(indices("ab", "aa")) and not list(indices("ab", "a"))
    """
    free = tuple(range(len(dom))) if free is None else free
    if not cod:
        if not free:
            yield ()
        return
    for i in free:
        rest = tuple(j for j in free if j != i)
        if dom[i] == cod[0]:
            yield from ((i, *xs) for xs in indices(dom, cod[1:], rest))


def permutations(category, dom, cod) -> Iterator:
    """
    The permutations `dom -> cod` of the wires of `dom` in a `category`, i.e.
    the invertible reparametrisations, yielded lazily and the identity first.
    At most ``BOUND`` (6) wires are searched: over longer types only the
    identity of ``dom == cod`` is yielded, so a permutation of 7 wires is not
    recognised, a documented limitation. The orderings that do not land on
    `cod` are discarded before any diagram is built; those of repeated wires
    are all kept, since they route the wires to different places.

    Parameters:
        category : The underlying category, with its swaps.
        dom : The type to permute.
        cod : The type the permutation has to land on.

    Example
    -------
    >>> from discopy.symmetric import Diagram, Ty
    >>> x, y = Ty('x'), Ty('y')
    >>> (perm, ) = permutations(Diagram, x @ y, y @ x)
    >>> assert perm.to_hypergraph() == Diagram.swap(x, y).to_hypergraph()
    >>> assert list(permutations(Diagram, x @ y, x @ x)) == []
    >>> assert next(permutations(Diagram, x @ x, x @ x)) == Diagram.id(x @ x)
    >>> assert len(list(permutations(Diagram, x @ x, x @ x))) == 2
    """
    if len(dom) > BOUND:
        if dom == cod:
            yield category.id(dom)
        return
    for xs in indices(tuple(dom), tuple(cod)):
        yield category.permutation(xs, dom)


def same(left: Symmetric, right: Symmetric) -> bool:
    """
    Whether two parametric maps are equal up to the equation of their
    underlying category, i.e. they have the same boundary, parameters and
    coparameters, and their insides have the same image in the quotient.

    Parameters:
        left : A parametric map.
        right : A parametric map of the same category.

    Example
    -------
    >>> from discopy.symmetric import Ty, Box
    >>> x, y, p = map(Ty, "xyp")
    >>> f = Symmetric(x, y, Box('f', x @ p, y), p)
    >>> assert same(f, f >> Symmetric.id(y))
    >>> assert not same(f, f.reparam(Box('r', y, p)))
    """
    category = type(left).category
    return (left.dom, left.cod, left.param, left.copar)\
        == (right.dom, right.cod, right.param, right.copar)\
        and quotient(category, left.inside) == quotient(category, right.inside)


def isomorphic(left, right, n_dom: int, n_cod: int) -> bool:
    """
    Whether two hypergraphs are isomorphic by an isomorphism that fixes the
    first `n_dom` ports of the domain and the first `n_cod` ports of the
    codomain in order, and is free on the others: these are the parameter and
    coparameter ports, which it may permute among those of the same type,
    since a port is typed by the spider it is attached to.

    Parameters:
        left : A hypergraph.
        right : A hypergraph.
        n_dom : The number of domain ports that stay in place.
        n_cod : The number of codomain ports that stay in place.

    Example
    -------
    >>> from discopy.symmetric import Ty, Box, Diagram
    >>> x, p, q = map(Ty, "xpq")
    >>> f = Box('f', x @ p @ q, x)
    >>> g = Diagram.id(x) @ Diagram.swap(q, p) >> f
    >>> assert isomorphic(f.to_hypergraph(), g.to_hypergraph(), 1, 1)
    >>> assert not isomorphic(f.to_hypergraph(), g.to_hypergraph(), 3, 1)
    """
    first, second = (labelled(left, n_dom, n_cod),
                     labelled(right, n_dom, n_cod))
    return profile(first) == profile(second) and is_isomorphic(
        first, second, node_match=lambda x, y: x == y)


def labelled(hypergraph, n_dom: int, n_cod: int):
    """
    The graph of a hypergraph, see :meth:`Hypergraph.to_graph`, where only
    the first `n_dom` ports of the domain and the first `n_cod` ports of the
    codomain keep their position, with the attributes of each node also
    written as one string, its ``label``.

    Parameters:
        hypergraph : A hypergraph.
        n_dom : The number of domain ports that keep their position.
        n_cod : The number of codomain ports that keep their position.

    Example
    -------
    >>> from discopy.symmetric import Ty, Box
    >>> x, p = map(Ty, "xp")
    >>> graph = labelled(Box('f', x @ p, x).to_hypergraph(), 1, 1)
    >>> kept = [graph.nodes[node]["i"] for node in graph
    ...         if node.kind == "input"]
    >>> assert sorted(kept, key=str) == [0, None]
    >>> assert all("label" in graph.nodes[node] for node in graph)
    """
    graph = hypergraph.to_graph()
    for node in graph:
        keep = {"input": n_dom, "output": n_cod}.get(node.kind)
        if keep is not None and graph.nodes[node]["i"] >= keep:
            graph.nodes[node]["i"] = None
    for node in graph:
        graph.nodes[node]["label"] = repr(
            sorted(graph.nodes[node].items(), key=repr))
    return graph


def profile(graph) -> list:
    """
    An invariant of labelled directed graphs under isomorphism: the sorted
    list of the label of each node with the sorted labels of its predecessors
    and of its successors, one round of colour refinement. Graphs with
    different profiles are not isomorphic, so it rejects most pairs before
    any search, e.g. those with a box that differs by name or data.

    Parameters:
        graph : A ``networkx.DiGraph`` with a ``label`` on each node.

    Example
    -------
    >>> from networkx import DiGraph
    >>> graph = DiGraph([(0, 1)])
    >>> graph.nodes[0]["label"], graph.nodes[1]["label"] = "a", "b"
    >>> profile(graph)
    [('a', [], ['b']), ('b', ['a'], [])]
    """
    label = graph.nodes.data("label")
    return sorted(
        (label[node],
         sorted(label[other] for other in graph.predecessors(node)),
         sorted(label[other] for other in graph.successors(node)))
        for node in graph)


@dataclass(frozen=True)
class UpToPermutation:
    """
    A parametric map compared up to the permutations of its parameter wires
    and of its coparameter wires, i.e. up to an invertible 2-cell, see
    :class:`Reparam`: a 2-cell acts on the parameters by precomposition, as
    :meth:`Symmetric.reparam`, and on the coparameters by postcomposition,
    as :meth:`Symmetric.recopar`. Two maps are equal when they have the same
    domain and codomain, the same parameter wires and the same coparameter
    wires as multisets, and the inside of one is, up to the equation of the
    underlying category, that of the other reparametrised by a permutation
    `sigma` and recoparametrised by a permutation `tau`, chosen independently.
    This is decided with no bound and no enumeration of the permutations, by
    an isomorphism of the hypergraphs of the insides (networkx's VF2, itself
    a backtracking search on the graphs) that fixes the boundary and may
    permute the parameter and coparameter ports of the same type, see
    :func:`isomorphic`;
    wires of different types, e.g. adjoint ones, are never exchanged.

    The underlying category has to be one of diagrams, e.g. ``Symmetric``,
    ``Markov`` or ``Compact``, where `permutation` and `quotient` make sense.
    Over a category of functions, equal maps are equal and comparing others
    may raise a ``TypeError``, since their parameters cannot be rerouted.

    Parameters:
        para : The parametric map to compare.

    Example
    -------
    >>> from discopy.symmetric import Ty, Box
    >>> x, y, p, q = map(Ty, "xypq")
    >>> f = Symmetric(x, y, Box('f', x @ p @ q, y), p @ q)
    >>> g = f.reparam(Symmetric.swap(q, p).inside)
    >>> assert f != g and UpToPermutation(f) == UpToPermutation(g)
    >>> h = f.reparam(Box('r', p @ q, p @ q))
    >>> assert UpToPermutation(f) != UpToPermutation(h)
    >>> m, n = Ty('m'), Ty('n')
    >>> t = Symmetric(x, y, Box('t', x @ p @ q, y @ m @ n), p @ q, m @ n)
    >>> v =t.recopar(Symmetric.swap(m, n).inside)
    >>> assert t != v and UpToPermutation(t) == UpToPermutation(v)
    """
    para: Symmetric

    @property
    def boundary(self) -> tuple:
        """ What a permutation of the parameters and coparameters keeps. """
        para = self.para
        return (para.dom, para.cod, *(
            frozenset(Counter(wires).items())
            for wires in (para.param, para.copar)))

    def matches(self, other: UpToPermutation) -> bool:
        """
        Whether the insides are isomorphic with free parameter and
        coparameter ports, see :func:`isomorphic`.

        Parameters:
            other : The map to compare with, of the same boundary.
        """
        left, right = self.para, other.para
        category = type(left).category
        return isomorphic(
            quotient(category, left.inside), quotient(category, right.inside),
            len(left.dom), len(left.cod))

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        if self.boundary != other.boundary:
            return False
        return self.para == other.para or self.matches(other)

    def __hash__(self):
        return hash(self.boundary)


@dataclass(unsafe_hash=True)
class Symmetric[category](
        SymmetricCategory[symmetric.Ty, "Symmetric"], TestableGeneric):
    """
    A parametric map from `dom` to `cod` with parameter space `param` is a
    morphism `inside : dom @ param -> cod` in an underlying `category`,
    optionally with a coparameter space `copar` on the codomain, i.e.
    `inside : dom @ param -> cod @ copar`.

    Parameters:
        dom (category.ob) : The domain of the parametric map.
        cod (category.ob) : The codomain of the parametric map.
        inside (category) : The morphism ``dom @ param -> cod @ copar``.
        param (category.ob) : The parameter space, empty by default.
        copar (category.ob) : The coparameter space, empty by default.

    .. admonition:: Summary

        .. autosummary::

            lift
            id
            then
            tensor
            swap
            reparam
            recopar
    """
    @classmethod
    def checked_params(cls) -> tuple:
        """ The default underlying category, for the class itself. """
        return (cls.category, )

    category = symmetric.Diagram
    ob = classproperty(lambda cls: cls.category.ob)

    dom: ob  # ty: ignore[invalid-type-form]
    cod: ob  # ty: ignore[invalid-type-form]
    inside: category
    param: ob = None  # ty: ignore[invalid-type-form]
    copar: ob = None  # ty: ignore[invalid-type-form]

    bifunctoriality = SymmetricCategory.bifunctoriality.modulo(
        UpToPermutation)
    scalar_sliding = SymmetricCategory.scalar_sliding.modulo(
        UpToPermutation)
    braid_naturality = SymmetricCategory.braid_naturality.modulo(
        UpToPermutation)

    @classmethod
    def equation_factory(cls, *terms: Symmetric) -> Equation:
        """
        The equation of a law, compared by the boundary and the hidden spaces
        of its terms and their inside up to the equation of the underlying
        category, e.g. its hypergraph, the parameters and coparameters being
        plain wires of it.
        """
        return Equation(*terms, up_to=lambda para: (
            para.dom, para.cod, para.param, para.copar,
            quotient(cls.category, para.inside)))

    @classmethod
    def strategy(cls, *, dom=None, cod=None, **params):
        """
        Generate parametric maps: a parameter and a coparameter space, then
        an inside from the strategy of the underlying category, with the
        boundary it needs, the parameters passed to it.
        """
        from hypothesis import strategies as st

        @st.composite
        def maps(draw):
            source = draw(cls.ob.strategy()) if dom is None else dom
            target = draw(cls.ob.strategy()) if cod is None else cod
            param, copar = (draw(cls.ob.strategy()) for _ in range(2))
            inside = draw(cls.category.strategy(
                dom=source @ param, cod=target @ copar, **params))
            return cls(source, target, inside, param, copar)

        return maps()

    def __post_init__(self):
        if self.param is None:
            self.param = self.ob()
        if self.copar is None:
            self.copar = self.ob()
        for obj in (self.dom, self.cod, self.param, self.copar):
            assert_isinstance(obj, self.ob)
        assert_isinstance(self.inside, self.category)
        assert_iscomposable(
            self.category.id(self.dom @ self.param), self.inside)
        assert_iscomposable(
            self.inside, self.category.id(self.cod @ self.copar))

    def to_tree(self) -> dict:
        """
        Serialise a parametric map, with the name of its category.

        >>> from discopy.utils import dumps, loads
        >>> from discopy.symmetric import Ty, Box
        >>> x, y, p = map(Ty, "xyp")
        >>> f = Symmetric(x, y, Box('f', x @ p, y), p)
        >>> assert loads(dumps(f)) == f
        """
        return {
            'factory': factory_name(get_origin(type(self))),
            'category': factory_name(self.category),
            'dom': self.dom.to_tree(),
            'cod': self.cod.to_tree(),
            'inside': self.inside.to_tree(),
            'param': self.param.to_tree(),
            'copar': self.copar.to_tree()}

    @classmethod
    def from_tree(cls, tree: dict) -> Symmetric:
        """ Decode a serialised parametric map, see :meth:`to_tree`. """
        factory = cls[resolve_factory(tree['category'])]
        return factory(*map(from_tree, (
            tree[key] for key in ('dom', 'cod', 'inside', 'param', 'copar'))))

    @classmethod
    def lift(cls, inside) -> Self:
        """
        A morphism of the underlying category as a parametric map with the
        empty parameter space, i.e. the injection functor from a category
        into its category of parametric maps.

        Parameters:
            inside : The morphism to lift.
        """
        return cls(inside.dom, inside.cod, inside)

    @classmethod
    def id(cls, dom=None) -> Symmetric:
        """
        The identity parametric map on `dom`, with empty parameter space.

        Parameters:
            dom : The domain of the identity, also its codomain.
        """
        return cls.lift(cls.category.id(cls.ob() if dom is None else dom))

    @unbiased
    def then(self, other: Symmetric) -> Symmetric:
        """
        Sequential composition tensors the hidden spaces on both sides,
        i.e. `(p, f) >> (q, g) == (p @ q, f @ q >> g)` for empty
        coparameters and the routing of :meth:`Stream.then
        <discopy.stream.Stream.then>` in general.

        Parameters:
            other : The parametric map to compose with.
        """
        assert_isinstance(other, type(self))
        assert_iscomposable(self, other)
        inside = self.inside @ other.param\
            >> self.cod @ self.category.swap(self.copar, other.param)\
            >> other.inside @ self.copar\
            >> other.cod @ self.category.swap(other.copar, self.copar)
        return type(self)(self.dom, other.cod, inside,
                          self.param @ other.param,
                          self.copar @ other.copar)

    @unbiased
    def tensor(self, other: Symmetric) -> Symmetric:
        """
        Parallel composition tensors the hidden spaces on both sides, with
        swaps routing the parameters to the right of the domains and the
        coparameters to the right of the codomains.

        Parameters:
            other : The parametric map to tensor with.
        """
        assert_isinstance(other, type(self))
        inside = self.dom @ self.category.swap(other.dom, self.param)\
            @ other.param >> self.inside @ other.inside >> self.cod\
            @ self.category.swap(self.copar, other.cod) @ other.copar
        return type(self)(self.dom @ other.dom, self.cod @ other.cod,
                          inside, self.param @ other.param,
                          self.copar @ other.copar)

    @classmethod
    def swap(cls, left, right) -> Symmetric:
        """
        The swap of the underlying category, with empty parameter space.

        Parameters:
            left : The object on the left of the swap.
            right : The object on the right of the swap.
        """
        return cls.lift(cls.category.swap(left, right))

    def reparam(self, other: category) -> Symmetric:
        """
        Precompose the parameters with `other : q -> param`; this is the
        1-cell part of the 2-cell :class:`Reparam`.

        Parameters:
            other : The reparametrisation, a morphism into ``param``.

        Example
        -------
        >>> from discopy.symmetric import Ty, Box
        >>> x, y, p, q = map(Ty, "xypq")
        >>> f = Symmetric(x, y, Box('f', x @ p, y), p)
        >>> r = Box('r', q, p)
        >>> f.reparam(r).inside.draw(doctest="docs/_static/para/reparam.svg")

        .. image:: /_static/para/reparam.svg
            :align: center
        """
        assert_iscomposable(self.dom @ other, self.inside)
        return type(self)(self.dom, self.cod,
                          self.dom @ other >> self.inside,
                          other.dom, self.copar)

    def recopar(self, other: category) -> Symmetric:
        """
        Post-compose the coparameter space with `other : copar -> q`,
        covariantly where :meth:`reparam` is contravariant.

        Parameters:
            other : The recoparametrisation, a morphism out of ``copar``.
        """
        assert_iscomposable(self.inside, self.cod @ other)
        return type(self)(self.dom, self.cod,
                          self.inside >> self.cod @ other,
                          self.param, other.cod)


@dataclass(frozen=True)
class Reparam:
    """
    A reparametrisation, i.e. a 2-cell `dom => cod` between parallel
    parametric maps: a morphism `inside : cod.param -> dom.param` of the
    underlying category such that ``cod == dom.reparam(inside)`` up to its
    equation, see :meth:`Symmetric.reparam`. Reparametrisations compose
    vertically with :meth:`then`, horizontally with ``>>`` and in parallel
    with ``@``, and those of :class:`Symmetric` with an invertible `inside`
    are the isomorphisms, up to which :class:`UpToPermutation` compares maps.
    They compose when their maps are equal up to the equation of the
    category, see :func:`same`, not only on the nose: the check is in the
    free category, so none relates two maps that agree only on the values of
    a model, e.g. two compiled networks. A reparametrisation covers the
    parameters only and needs parallel maps with the same coparameter: the
    invertible 2-cells of maps with coparameters, which also permute those,
    are only searched by :class:`UpToPermutation`. The invertible ones are
    searched among the permutations of at most ``BOUND`` (6) parameter wires,
    see :meth:`is_invertible`.

    Reparametrisations are frozen dataclasses, so that
    ``eval(repr(alpha)) == alpha`` holds whenever the names in the repr of
    the maps are in scope, e.g. in the test of this module, where they are
    imported from :mod:`discopy.symmetric`.

    Parameters:
        dom (Symmetric) : The parametric map to reparametrise.
        inside (category) : The morphism ``cod.param -> dom.param``.
        cod (Symmetric) : The reparametrised map, parallel to ``dom``.

    Example
    -------
    >>> from discopy.symmetric import Ty, Box
    >>> x, y, p, q = map(Ty, "xypq")
    >>> f = Symmetric(x, y, Box('f', x @ p, y), p)
    >>> alpha = Reparam.of(f, Box('r', q, p))
    >>> assert alpha.cod == f.reparam(Box('r', q, p))
    >>> assert Reparam.id(f).is_invertible and not alpha.is_invertible
    >>> assert Reparam.id(f).then(alpha) == alpha
    >>> assert alpha == alpha.then(Reparam.id(alpha.cod))
    >>> Reparam(f, Box('r', q, p), f)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    AxiomError
    """
    dom: Symmetric
    inside: symmetric.Diagram
    cod: Symmetric

    def __post_init__(self):
        left, right = self.dom, self.cod
        if (left.dom, left.cod, left.copar)\
                != (right.dom, right.cod, right.copar):
            raise AxiomError(f"{left} and {right} are not parallel.")
        assert_iscomposable(left.dom @ self.inside, left.inside)
        if self.inside.dom != right.param or self.inside.cod != left.param:
            raise AxiomError(f"{self.inside} is not {right.param} -> "
                             f"{left.param}.")
        if not same(left.reparam(self.inside), right):
            raise AxiomError(f"{right} is not {left} reparametrised.")

    @classmethod
    def of(cls, para: Symmetric, other) -> Reparam:
        """
        The reparametrisation of a parametric map, from the map to
        ``para.reparam(other)``.

        Parameters:
            para : The parametric map.
            other : The reparametrisation, a morphism into ``para.param``.
        """
        return cls(para, other, para.reparam(other))

    @classmethod
    def id(cls, para: Symmetric) -> Reparam:
        """
        The identity reparametrisation of a parametric map.

        Parameters:
            para : The parametric map.
        """
        return cls.of(para, type(para).category.id(para.param))

    def then(self, other: Reparam) -> Reparam:
        """
        Vertical composition: reparametrisations compose contravariantly,
        i.e. `alpha.then(beta).inside == beta.inside >> alpha.inside`. The
        codomain of this one and the domain of `other` need only be equal up
        to the equation of the underlying category, see :func:`same`, e.g.
        two maps with the same hypergraph.

        Parameters:
            other : The reparametrisation starting from the codomain of this
                one.

        Example
        -------
        >>> from discopy.symmetric import Ty, Box
        >>> x, y, p, q, r = map(Ty, "xypqr")
        >>> f = Symmetric(x, y, Box('f', x @ p, y), p)
        >>> alpha = Reparam.of(f, Box('a', q, p))
        >>> beta = Reparam.of(alpha.cod, Box('b', r, q))
        >>> assert alpha.then(beta).inside == Box('b', r, q) >> Box('a', q, p)
        >>> assert alpha.then(beta).cod == beta.cod
        """
        if not same(self.cod, other.dom):
            raise AxiomError(f"{self.cod} does not compose with {other.dom}.")
        return type(self)(self.dom, other.inside >> self.inside, other.cod)

    def __rshift__(self, other: Reparam) -> Reparam:
        """
        Horizontal composition along the composition of parametric maps, which
        tensors their parameters, i.e. ``inside @ inside``. The maps have to
        be composable, otherwise an :class:`AxiomError` is raised.

        Parameters:
            other : The reparametrisation of maps composable after those of
                this one.

        Example
        -------
        >>> from discopy.symmetric import Ty, Box
        >>> x, y, z, p, q, r = map(Ty, "xyzpqr")
        >>> f = Symmetric(x, y, Box('f', x @ p, y), p)
        >>> g = Symmetric(y, z, Box('g', y @ q, z), q)
        >>> alpha, beta = Reparam.of(f, Box('a', r, p)), Reparam.id(g)
        >>> assert (alpha >> beta).cod == f.reparam(Box('a', r, p)) >> g
        >>> assert (alpha >> beta).inside == Box('a', r, p) @ q
        """
        if self.dom.cod != other.dom.dom:
            raise AxiomError(f"{self.dom} does not compose with {other.dom}.")
        return type(self)(self.dom >> other.dom, self.inside @ other.inside,
                          self.cod >> other.cod)

    def __matmul__(self, other: Reparam) -> Reparam:
        """
        Horizontal composition along the tensor of parametric maps.

        Parameters:
            other : The reparametrisation to tensor with.

        Example
        -------
        >>> from discopy.symmetric import Ty, Box
        >>> x, y, z, p, q, r = map(Ty, "xyzpqr")
        >>> f = Symmetric(x, y, Box('f', x @ p, y), p)
        >>> g = Symmetric(y, z, Box('g', y @ q, z), q)
        >>> alpha = Reparam.of(f, Box('a', r, p))
        >>> assert (alpha @ Reparam.id(g)).cod == f.reparam(Box('a', r, p)) @ g
        >>> assert (alpha @ Reparam.id(g)).inside == Box('a', r, p) @ q
        """
        return type(self)(self.dom @ other.dom, self.inside @ other.inside,
                          self.cod @ other.cod)

    @property
    def is_invertible(self) -> bool:
        """
        Whether the reparametrisation is a permutation of the parameter wires
        up to the equation of the category. At most ``BOUND`` (6) wires are
        searched as :func:`permutations` does, so that a permutation of 7
        wires is not recognised: a documented limitation.
        """
        category = type(self.dom).category
        target = quotient(category, self.inside)
        return any(
            quotient(category, perm) == target for perm in permutations(
                category, self.inside.dom, self.inside.cod))


class Traced(Symmetric, TracedCategory):
    """
    Parametric maps over a traced symmetric underlying `category` form a
    traced category, with the parameters swapped out of the way.
    """
    trace_dinaturality_left = TracedCategory.trace_dinaturality_left.modulo(
        UpToPermutation)
    trace_dinaturality_right = TracedCategory.trace_dinaturality_right.modulo(
        UpToPermutation)
    sliding = TracedCategory.sliding.modulo(UpToPermutation)

    def trace(self, n: int = 1, left: bool = False) -> Traced:
        """
        The trace of a parametric map is the trace of the underlying
        morphism, with the parameters swapped out of the way.

        Parameters:
            n : The number of objects to trace over.
            left : Whether to trace the wires on the left or right.
        """
        if n == 0:
            return self
        if left:
            return type(self)(self.dom[n:], self.cod[n:],
                              self.inside.trace(n, left=True),
                              self.param, self.copar)
        inside = self.dom[:-n] @ self.category.swap(
            self.param, self.dom[-n:]) >> self.inside >> self.cod[:-n]\
            @ self.category.swap(self.cod[-n:], self.copar)
        return type(self)(self.dom[:-n], self.cod[:-n],
                          inside.trace(n), self.param, self.copar)


class Markov(Symmetric, MarkovCategory):
    """
    Parametric maps over a Markov underlying `category` form a Markov
    category, with the copy of the underlying category as :meth:`copy`.
    """
    category = markov.Diagram

    @classmethod
    def copy(cls, x: Symmetric.ob, n: int = 2) -> Markov:
        """
        The copy of the underlying category, with empty parameter space.

        Parameters:
            x : The object to copy.
            n : The number of copies.
        """
        return cls.lift(cls.category.copy(x, n))


class Comarkov(Symmetric, ComarkovCategory):
    """
    Parametric maps over a comarkov underlying `category` form a comarkov
    category, with the merge of the underlying category as :meth:`merge`.
    """
    category = comarkov.Diagram

    @classmethod
    def merge(cls, x: Symmetric.ob, n: int = 2) -> Comarkov:
        """
        The merge of the underlying category, with empty parameter space.

        Parameters:
            x : The object to merge.
            n : The number of copies.
        """
        return cls.lift(cls.category.merge(x, n))


class Closed(Markov, ClosedCategory):
    """
    Parametric maps over a closed underlying `category` form a closed
    category, currying with the parameters swapped out of the way.
    """
    curry_left = ClosedCategory.curry_left.failing(messages.PARA_CURRY)
    curry_right = ClosedCategory.curry_right.failing(messages.PARA_CURRY)
    currying_left = ClosedCategory.currying_left.failing(messages.FREE_CURRY)
    currying_right = ClosedCategory.currying_right.failing(messages.FREE_CURRY)
    currying_eta_left = ClosedCategory.currying_eta_left.failing(
        messages.FREE_CURRY)
    currying_eta_right = ClosedCategory.currying_eta_right.failing(
        messages.FREE_CURRY)
    currying_naturality_left = ClosedCategory.currying_naturality_left.failing(
        messages.FREE_CURRY)
    currying_naturality_right = ClosedCategory.currying_naturality_right\
        .failing(messages.FREE_CURRY)
    currying_symmetry = ClosedCategory.currying_symmetry.failing(
        messages.FREE_CURRY)
    category = closed.Diagram

    @classmethod
    def ev(cls, base: Symmetric.ob, exponent: Symmetric.ob, left: bool = True
           ) -> Closed:
        """
        The evaluation of the underlying category, with empty parameters.

        Parameters:
            base : The base of the exponential type.
            exponent : The exponent of the exponential type.
            left : Whether to take the left or right evaluation.
        """
        return cls.lift(cls.category.ev(base, exponent, left))

    def curry(self, n: int = 1, left: bool = True) -> Closed:
        """
        Curry the last `n` objects of the domain if `left` else the first,
        i.e. everything but the parameters, which a left currying swaps out
        of the way the same as :meth:`Traced.trace`.

        Parameters:
            n : The number of objects to curry.
            left : Whether to curry into a left or right exponential.

        Currying no object is the identity, as for every other curry.

        >>> from discopy import closed
        >>> a, P = closed.Ty('a'), closed.Ty('P')
        >>> k = Closed(a, a, closed.Box('k', a @ P, a), P)
        >>> assert k.curry(0) == k == k.curry(0, left=False)
        """
        if n < 0 or n > len(self.dom):
            raise ValueError(messages.NOT_SPLITTABLE.format(n, self.dom))
        if not n:
            return self
        if self.copar:
            raise ValueError(messages.PARA_CURRY_COPAR.format(self.copar))
        if not left:
            inside = self.inside.curry(n, left=False)
            return type(self)(self.dom[n:], inside.cod, inside, self.param)
        inside = self.dom[:-n] @ self.category.swap(
            self.param, self.dom[-n:]) >> self.inside
        inside = inside.curry(n, left=True)
        return type(self)(self.dom[:-n], inside.cod, inside, self.param)


class Feedback(Markov, FeedbackCategory):
    """
    Parametric maps over a feedback underlying `category` form a feedback
    category, with :meth:`delay` applied to all five components.

    The laws hold up to :class:`UpToPermutation`, as in every para category,
    since feedback never connects parameters to coparameters.
    """
    bifunctoriality = SymmetricCategory.bifunctoriality.modulo(
        UpToPermutation)
    scalar_sliding = SymmetricCategory.scalar_sliding.modulo(
        UpToPermutation)
    braid_naturality = SymmetricCategory.braid_naturality.modulo(
        UpToPermutation)
    joining = FeedbackCategory.joining.failing(
        messages.FREE_FEEDBACK)
    sliding = FeedbackCategory.sliding.failing(
        messages.FREE_FEEDBACK)
    strength = FeedbackCategory.strength.failing(
        messages.FREE_FEEDBACK)
    tightening = FeedbackCategory.tightening.failing(
        messages.FREE_FEEDBACK)
    category = markov.Diagram

    def delay(self, n_steps: int = 1) -> Feedback:
        """
        Delay a parametric map by delaying its underlying morphism together
        with its domain, codomain, parameter and coparameter spaces.

        Parameters:
            n_steps : The number of time steps to delay.
        """
        return type(self)(*(x.delay(n_steps) for x in (
            self.dom, self.cod, self.inside, self.param, self.copar)))

    def feedback(self, n: int | None = None, left: bool = False, *,
                 dom: Symmetric.ob = None, cod: Symmetric.ob = None,
                 mem: Symmetric.ob = None) -> Feedback:
        """
        The feedback of the underlying category, with the parameters
        swapped out of the way the same as :meth:`Traced.trace`, in either
        form of :meth:`discopy.abc.FeedbackCategory.feedback`.

        Parameters:
            n : The number of objects to feed back.
            left : Feeding back on the left, which is not implemented.
            dom : The domain of the feedback.
            cod : The codomain of the feedback.
            mem : The memory type to trace over.
        """
        if left:
            raise NotImplementedError
        dom, cod, mem = self.split_memory(
            n, left, dom=dom, cod=cod, mem=mem)
        inside = dom @ self.category.swap(self.param, mem.delay())\
            >> self.inside >> cod @ self.category.swap(mem, self.copar)
        return type(self)(
            dom, cod,
            inside.feedback(
                dom=dom @ self.param, cod=cod @ self.copar, mem=mem),
            self.param, self.copar)


class Compact(Traced, CompactCategory):
    """
    Parametric maps over a compact underlying `category` form a compact
    category, with the cups and caps of the underlying category.
    """
    adjunction = CompactCategory.adjunction.inapplicable(
        messages.NO_ROTATION)
    rotate_contravariance = CompactCategory.rotate_contravariance\
        .inapplicable(messages.NO_ROTATION)
    transpose_contravariance = CompactCategory.transpose_contravariance\
        .modulo(UpToPermutation)
    inverse_twisting = CompactCategory.inverse_twisting.inapplicable(
        messages.NO_DAGGER)
    category = compact.Diagram

    @classmethod
    def cups(cls, left: Symmetric.ob, right: Symmetric.ob) -> Compact:
        """
        The cups of the underlying category, with empty parameter space.

        Parameters:
            left : The left-hand side of the cups.
            right : Its adjoint, i.e. the right-hand side of the cups.
        """
        return cls.lift(cls.category.cups(left, right))

    @classmethod
    def caps(cls, left: Symmetric.ob, right: Symmetric.ob) -> Compact:
        """
        The caps of the underlying category, with empty parameter space.

        Parameters:
            left : The left-hand side of the caps.
            right : Its adjoint, i.e. the right-hand side of the caps.
        """
        return cls.lift(cls.category.caps(left, right))

    ev = classmethod(Closed.ev.__func__)


class Hypergraph(Compact, Markov, Comarkov, HypergraphCategory):
    """
    Parametric maps over a hypergraph underlying `category` form a
    hypergraph category, with the spiders of the underlying category.
    """
    category = frobenius.Diagram

    @classmethod
    def spiders(cls, n_legs_in: int, n_legs_out: int, typ: Symmetric.ob
                ) -> Hypergraph:
        """
        The spiders of the underlying category, with empty parameters.

        Parameters:
            n_legs_in : The number of legs in for each spider.
            n_legs_out : The number of legs out for each spider.
            typ : The type of the spiders.
        """
        return cls.lift(cls.category.spiders(n_legs_in, n_legs_out, typ))
