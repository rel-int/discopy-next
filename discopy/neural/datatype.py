# -*- coding: utf-8 -*-

"""
Inductive datatypes and their folds, the syntax of the recursive networks of
*Categorical Deep Learning* (arXiv:2402.15332).

A polynomial endofunctor :math:`F(S) = \\sum_i A_i \\otimes S^{\\otimes s_i}`
is given by its constructors
:math:`c_i : A_i \\otimes S^{\\otimes s_i} \\to S`, a :class:`Datatype`.
Its terms are the diagrams built from the constructors by composition and
tensor alone, with the data :math:`A_i` as inputs. An algebra
is a functor out of them, the fold of a term is its image, and without
parameters it is the unique one out of the terms, by functoriality: see
:class:`Algebra`.

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Datatype
    Interpretation
    Algebra
    LaxHomomorphism

.. autosummary::

    chain
    plug

The parameter of an algebra is copied into every constructor, a choice and
not forced (Appendix J), which makes the fold of a term a lax algebra of the
free monad of the datatype (*Categorical Deep Learning*, Appendix F). Every
coherence law is a :class:`discopy.para.Reparam`, checked by the equation of
its category: :meth:`Algebra.lift`, :meth:`Algebra.lift_reparam`,
:meth:`Algebra.unitality` and :meth:`Algebra.associativity`, and
:class:`LaxHomomorphism` for the folds that are lax and not tied. No box is
copyable or discardable, so a reparametrisation by a box is rejected by every
law that copies the parameter twice or discards it. The dual, the machines of
:mod:`discopy.neural.machine`, builds on :class:`Interpretation`.

Note
----
This module imports no tensor framework. Terms use composition and tensor
only, never swaps, copies or discards.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Any

from discopy import markov, para
from discopy.utils import assert_isinstance


class Datatype:
    """
    The constructors of a polynomial endofunctor, each a box with the same
    atomic codomain, the carrier :math:`S`, and a domain :math:`A_i \\otimes
    S^{\\otimes s_i}` where :math:`A_i` does not contain :math:`S`.

    Parameters:
        constructors : The boxes :math:`c_i`, with distinct names.

    Raises:
        ValueError : If the constructors are not of this shape.

    Example
    -------
    >>> from discopy.markov import Ty
    >>> A, S = Ty("A"), Ty("S")
    >>> L = Datatype.list(A, S)
    >>> nil, cons = L.constructors
    >>> assert L.carrier == S and L.data(cons) == A and L.arity(cons) == 1
    >>> assert L.term(chain(2)) == A @ A @ nil >> A @ cons >> cons
    >>> L.term(("Cons", ("Nil", ), ("Nil", )))
    Traceback (most recent call last):
    ...
    ValueError: Cons expects 1 children, got 2
    """

    def __init__(self, *constructors: markov.Box):
        if not constructors:
            raise ValueError("A datatype needs a constructor.")
        for box in constructors:
            assert_isinstance(box, markov.Box)
        names = [box.name for box in constructors]
        if len(set(names)) < len(names):
            raise ValueError(f"Constructor names must differ, got {names}.")
        if not constructors[0].cod.is_atomic:
            raise ValueError(
                f"The carrier {constructors[0].cod} is not atomic.")
        self.constructors = tuple(constructors)
        self.carrier = constructors[0].cod
        for box in self.constructors:
            self.check(box)

    def check(self, box: markov.Box):
        """
        Raise ``ValueError`` unless a box has the carrier as codomain and the
        carrier only as its last wires in the domain.

        Parameters:
            box : The constructor to check.
        """
        if box.cod != self.carrier:
            raise ValueError(f"The codomain {box.cod} of {box.name} is not "
                             f"the carrier {self.carrier}.")
        arity = self.arity(box)
        if box.dom[len(box.dom) - arity:] != self.carrier ** arity:
            raise ValueError(f"The domain {box.dom} of {box.name} has the "
                             f"carrier {self.carrier} before its last wires.")

    def arity(self, box: markov.Box) -> int:
        """ The number :math:`s_i` of recursive inputs of a constructor. """
        return box.dom.count(self.carrier)

    def data(self, box: markov.Box) -> markov.Ty:
        """ The type :math:`A_i` of the data of a constructor. """
        return box.dom[:len(box.dom) - self.arity(box)]

    def constructor(self, name: str) -> markov.Box:
        """
        The constructor of a given name.

        Parameters:
            name : The name of the box.
        """
        for box in self.constructors:
            if box.name == name:
                return box
        names = [box.name for box in self.constructors]
        raise ValueError(f"Unknown constructor {name!r}, expected {names}.")

    def term(self, value: tuple) -> markov.Diagram:
        """
        The diagram of a term ``(name, *children)``, with as many children
        as the arity of the constructor: the data of a node, then those of
        its children from left to right, are the inputs and the carrier is
        the output. The carrier appears in a constructor only as its last
        inputs, after its data, and never in the domain of the term.

        The recursion on the children bounds the depth of a term by the
        recursion limit of Python.

        Parameters:
            value : A nested tuple, the constructor name and its subterms.
        """
        box, children = self.unpack(value)
        return markov.Diagram.id(self.data(box)).tensor(
            *map(self.term, children)) >> box

    def unpack(self, value: tuple) -> tuple:
        """
        The constructor and the children of a term ``(name, *children)``.

        Parameters:
            value : A nested tuple, the constructor name and its subterms.

        Raises:
            ValueError : If the value is not a tuple, the constructor is
                unknown or the children are not as many as its arity.
        """
        if not isinstance(value, tuple) or not value:
            raise ValueError(f"Expected (name, *children), got {value!r}.")
        box, children = self.constructor(value[0]), value[1:]
        if len(children) != self.arity(box):
            raise ValueError(f"{box.name} expects {self.arity(box)} "
                             f"children, got {len(children)}")
        return box, children

    def substitute(self, term: markov.Diagram, *children: markov.Diagram
                   ) -> markov.Diagram:
        """
        The multiplication of the free monad of the datatype: a term with
        holes, i.e. a diagram with the carrier in its domain, with a diagram
        plugged into each hole.

        Parameters:
            term : A diagram of constructors.
            children : One diagram with the carrier as codomain per hole.

        Raises:
            ValueError : If there are not as many children as holes.

        Example
        -------
        >>> from discopy.markov import Diagram, Ty
        >>> A, S = Ty("A"), Ty("S")
        >>> L = Datatype.list(A, S)
        >>> nil, cons = L.constructors
        >>> assert L.substitute(cons, L.term(chain(0))) == L.term(chain(1))
        >>> assert L.substitute(Diagram.id(S), nil) == nil
        >>> L.substitute(cons)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Expected 1 maps ...
        """
        holes = plug(term.dom, self.carrier, children, markov.Diagram.id)
        return holes >> term

    @classmethod
    def list(cls, data: markov.Ty, carrier: markov.Ty) -> Datatype:
        """
        The lists of the endofunctor :math:`1 + A \\times -`, with
        constructors ``Nil : 1 -> S`` and ``Cons : A @ S -> S``.

        Parameters:
            data : The type :math:`A` of the elements.
            carrier : The atomic type :math:`S` of the lists.
        """
        return cls(markov.Box("Nil", markov.Ty(), carrier),
                   markov.Box("Cons", data @ carrier, carrier))

    @classmethod
    def tree(cls, data: markov.Ty, carrier: markov.Ty) -> Datatype:
        """
        The binary trees of the endofunctor :math:`A + (-)^2`, with
        constructors ``Leaf : A -> S`` and ``Node : S @ S -> S``.

        Parameters:
            data : The type :math:`A` of the leaves.
            carrier : The atomic type :math:`S` of the trees.
        """
        return cls(markov.Box("Leaf", data, carrier),
                   markov.Box("Node", carrier @ carrier, carrier))

    def __eq__(self, other):
        return isinstance(other, Datatype)\
            and self.constructors == other.constructors

    def __hash__(self):
        return hash(self.constructors)

    def __repr__(self):
        return f"Datatype({', '.join(map(repr, self.constructors))})"


def chain(n: int) -> tuple:
    """
    The term of a list of ``n`` elements, i.e. ``Cons`` applied ``n`` times
    to ``Nil``, to be given to :meth:`Datatype.term` of a list.

    Parameters:
        n : The length of the list.

    Example
    -------
    >>> chain(2)
    ('Cons', ('Cons', ('Nil',)))
    """
    if n < 0:
        raise ValueError(f"Expected a length, got {n}.")
    return reduce(lambda term, _: ("Cons", term), range(n), ("Nil", ))


def plug(dom, carrier, maps, identity) -> Any:
    """
    The tensor of ``maps`` on the wires of ``dom`` that are the carrier, in
    order, and of ``identity(atom)`` on the other atoms: the one place where
    the holes of a term are filled.

    Parameters:
        dom : The domain of the term, a type.
        carrier : The atomic type of the holes.
        maps : One map per hole.
        identity : The map to put on an atom that is not a hole.

    Raises:
        ValueError : If there are not as many maps as holes.

    Example
    -------
    >>> from discopy.markov import Box, Diagram, Ty
    >>> A, S, x = Ty("A"), Ty("S"), Box("x", Ty(), Ty("S"))
    >>> assert plug(A @ S, S, [x], Diagram.id) == A @ x
    >>> plug(A @ S, S, [], Diagram.id)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Expected 1 maps ...
    """
    maps = list(maps)
    if dom.count(carrier) != len(maps):
        raise ValueError(f"Expected {dom.count(carrier)} maps for the holes "
                         f"of {dom}, got {len(maps)}.")
    queue = iter(maps)
    atoms = [dom[i:i + 1] for i in range(len(dom))]
    parts = [next(queue) if atom == carrier else identity(atom)
             for atom in atoms]
    return parts[0].tensor(*parts[1:]) if parts else identity(dom)


def require_parametric(algebra):
    """
    Raise ``ValueError`` unless an algebra has a parameter.

    Parameters:
        algebra : The :class:`Algebra` to check.
    """
    if not algebra.parametric:
        raise ValueError(f"{algebra.cod} is not a parametric category.")


def check_param(images: dict, cod: type):
    """
    Raise ``ValueError`` unless the images are arrows of a category of
    parametric maps ``cod`` with the same parameter.

    Parameters:
        images : A map from a name to its image.
        cod : The category of parametric maps.

    Example
    -------
    >>> from discopy import python
    >>> Net, Int = para.Markov[python.Function], python.Ty(int)
    >>> f = Net(Int, Int, python.Function(lambda a, p: a, Int ** 2, Int),
    ...         param=Int)
    >>> check_param({"f": f, "g": f}, Net)
    >>> check_param({"f": f, "g": f @ f}, Net)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Expected one parameter, got ...
    """
    for name, image in images.items():
        if not issubclass(type(image), cod):
            raise ValueError(f"The image of {name} is not an arrow of {cod}.")
    params = {name: image.param for name, image in images.items()}
    if len(set(params.values())) > 1:
        raise ValueError(f"Expected one parameter, got {params}.")


class Interpretation:
    """
    A functor into a category ``cod``, whose images share one parameter,
    tied by one copy, when ``cod`` is a category of parametric maps. It
    reads the fields ``cod`` and ``param`` of the dataclass it is mixed in.

    Example
    -------
    >>> from discopy import python
    >>> class Plain(Interpretation):
    ...     cod = python.Function
    >>> class Net(Interpretation):
    ...     cod = para.Markov[python.Function]
    >>> assert Net().parametric and not Plain().parametric
    """
    cod: type
    param: Any

    @property
    def parametric(self) -> bool:
        """ Whether the codomain is a category of parametric maps. """
        return isinstance(self.cod, type)\
            and issubclass(self.cod, para.Symmetric)

    def tie(self, image, uses: int):
        """
        The image with its parameter copied ``uses`` times, if parametric.

        Parameters:
            image : The image of a term.
            uses : The number of parameters it uses.
        """
        if not self.parametric:
            return image
        return image.reparam(self.cod.category.copy(self.param, uses))


@dataclass
class Algebra(Interpretation):
    """
    An algebra of a datatype: one image for each constructor in a category
    ``cod``, i.e. a functor out of the free category on the constructors.
    Its fold sends a term to its image. Without parameters it is the unique
    functor out of the terms. With parameters, the tied fold is the
    homomorphism whose 2-cell is the copy of the parameters (App. J).

    When ``cod`` is a category of parametric maps, the algebra has one
    parameter :math:`P` for all its constructors, each of which uses it as
    it needs: a constructor that ignores :math:`P` must still declare it,
    by discarding it. The image of a term with :math:`k` constructors has
    the parameter :math:`P^{\\otimes k}` and the fold ties them by the copy
    :math:`P \\to P^{\\otimes k}`.

    The repr of an algebra is transparent only up to the maps it holds.

    Parameters:
        datatype : The datatype.
        ar_map : A map from each constructor to its image.
        ob_map : A map from the carrier and the atoms of the data to
            ``cod.ob``.
        cod : The codomain category.

    Raises:
        ValueError : If a constructor has no image, if an atom has no image,
            if an image in a parametric category is not an arrow of ``cod``
            or if the images do not have the same parameter.

    Example
    -------
    >>> from discopy import python
    >>> from discopy.markov import Ty
    >>> A, S = Ty("A"), Ty("S")
    >>> L = Datatype.list(A, S)
    >>> nil, cons = L.constructors
    >>> Int, Net = python.Ty(int), para.Markov[python.Function]
    >>> F = Algebra(L, {
    ...     nil: Net(python.Ty(), Int, python.Function(
    ...         lambda p: p, Int, Int), param=Int),
    ...     cons: Net(Int @ Int, Int, python.Function(
    ...         lambda a, s, p: a + p * s, Int ** 3, Int), param=Int)},
    ...     {A: Int, S: Int}, Net)
    >>> fold = F(L.term(chain(3)))
    >>> assert fold.param == Int
    >>> assert fold.inside(1, 2, 3, 10) == 1 + 10 * (2 + 10 * (3 + 10 * 10))
    >>> assert F.functor(L.term(chain(3))).param == Int ** 4
    """
    datatype: Datatype
    ar_map: dict
    ob_map: dict
    cod: type

    def __post_init__(self):
        self.check()

    def check(self):
        """
        Raise ``ValueError`` unless every constructor has an image, every
        atom of the carrier and the data has an image and, in a parametric
        category, the images are arrows of ``cod`` with the same parameter.
        """
        boxes = self.datatype.constructors
        for box in boxes:
            if box not in self.ar_map:
                raise ValueError(f"No image for the constructor {box.name}.")
        types = (self.datatype.carrier, ) + tuple(
            self.datatype.data(box) for box in boxes)
        missing = [ty[i:i + 1] for ty in types for i in range(len(ty))
                   if ty[i:i + 1] not in self.ob_map]
        if missing:
            raise ValueError(f"No image for the atom {missing[0]}.")
        if self.parametric:
            check_param({box.name: self.ar_map[box] for box in boxes},
                        self.cod)

    @property
    def param(self):
        """ The common parameter of the images, ``None`` if not parametric. """
        if not self.parametric:
            return None
        return self.ar_map[self.datatype.constructors[0]].param

    @property
    def functor(self) -> markov.Functor:
        """ The functor from the free category on the constructors. """
        return markov.Functor(self.ob_map, self.ar_map, cod=self.cod)

    def __call__(self, term: markov.Diagram) -> Any:
        """
        The fold of a term of the datatype.

        Parameters:
            term : The diagram of :meth:`Datatype.term`.
        """
        return self.tie(self.functor(term), len(term.boxes))

    def lift(self, term: markov.Diagram, *maps, param=None
             ) -> para.Symmetric:
        """
        The free monad on a map, i.e. the term with its holes filled by maps
        of this category: the image of the data is the identity of its image,
        and the parameter of the maps is copied into every hole, discarded if
        there are none (*Categorical Deep Learning*, Appendix G.8).

        Parameters:
            term : A diagram of constructors, with holes.
            maps : A parametric map for each hole, with one parameter.
            param : The parameter of the maps, needed when there are none.

        Raises:
            ValueError : If the algebra is not parametric, a map is a
                reparametrisation, there are no maps and no parameter, the
                parameters differ or the maps are not as many as the holes.

        Example
        -------
        >>> from discopy.markov import Box, Diagram, Ty
        >>> A, S, P = Ty("A"), Ty("S"), Ty("P")
        >>> L = Datatype.list(A, S)
        >>> nil, cons = L.constructors
        >>> Net = para.Markov[Diagram]
        >>> z = Net(Ty(), S, Box("z", P, S), param=P)
        >>> c = Net(A @ S, S, Box("c", A @ S @ P, S), param=P)
        >>> F = Algebra(L, {nil: z, cons: c}, {A: A, S: S}, Net)
        >>> lifted = F.lift(cons, F(nil))
        >>> assert (lifted.dom, lifted.cod, lifted.param) == (A, A @ S, P)
        >>> assert F.lift(nil, param=P).param == P
        """
        require_parametric(self)
        if any(isinstance(m, para.Reparam) for m in maps):
            raise ValueError("Expected maps, got a reparametrisation.")
        params = [m.param for m in maps] + ([] if param is None else [param])
        if not params:
            raise ValueError("Expected a param for a term without holes.")
        if any(p != params[0] for p in params):
            raise ValueError(f"Expected one parameter, got {params}.")
        inside = plug(term.dom, self.datatype.carrier, maps,
                      lambda atom: self.cod.id(self.functor(atom)))
        return inside.reparam(self.cod.category.copy(params[0], len(maps)))

    def lift_reparam(self, term: markov.Diagram, *reparams, param=None
                     ) -> para.Reparam:
        """
        The free monad on a reparametrisation: they have the same inside, so
        has the result, which goes from the lift of their domains to the lift
        of their codomains. Its equation rejects it unless the inside
        commutes with the copy.

        Parameters:
            term : A diagram of constructors, with holes.
            reparams : A reparametrisation for each hole, with one inside, at
                least one.
            param : The parameter of the domains, as in :meth:`lift`. The
                codomains have their own, that of the reparametrisations.

        Raises:
            ValueError : If the algebra is not parametric, there are no
                reparametrisations, one is a map, the insides differ or they
                are not as many as the holes.

        Example
        -------
        >>> from discopy.markov import Box, Diagram, Ty
        >>> A, S, P = Ty("A"), Ty("S"), Ty("P")
        >>> L = Datatype.list(A, S)
        >>> nil, cons = L.constructors
        >>> Net = para.Markov[Diagram]
        >>> z = Net(Ty(), S, Box("z", P, S), param=P)
        >>> c = Net(A @ S, S, Box("c", A @ S @ P, S), param=P)
        >>> F = Algebra(L, {nil: z, cons: c}, {A: A, S: S}, Net)
        >>> alpha = para.Reparam.id(F(nil))
        >>> assert F.lift_reparam(cons, alpha).inside == Diagram.id(P)
        """
        require_parametric(self)
        if not reparams:
            raise ValueError("Expected at least one reparametrisation.")
        if not all(isinstance(r, para.Reparam) for r in reparams):
            raise ValueError("Expected reparametrisations, got a map.")
        if any(r.inside != reparams[0].inside for r in reparams):
            raise ValueError(
                "Expected reparametrisations with the same inside.")
        return para.Reparam(
            self.lift(term, *(r.dom for r in reparams), param=param),
            reparams[0].inside,
            self.lift(term, *(r.cod for r in reparams)))

    def unitality(self) -> para.Reparam:
        """
        The unit of the lax algebra: a reparametrisation from the identity of
        the carrier to the fold of the hole, which discards the parameter
        (*Categorical Deep Learning*, Definition F.2).

        Raises:
            ValueError : If the algebra is not parametric.

        Example
        -------
        >>> from discopy.markov import Box, Diagram, Ty
        >>> A, S, P = Ty("A"), Ty("S"), Ty("P")
        >>> L = Datatype.list(A, S)
        >>> nil, cons = L.constructors
        >>> Net = para.Markov[Diagram]
        >>> z = Net(Ty(), S, Box("z", P, S), param=P)
        >>> c = Net(A @ S, S, Box("c", A @ S @ P, S), param=P)
        >>> F = Algebra(L, {nil: z, cons: c}, {A: A, S: S}, Net)
        >>> assert F.unitality().inside == Diagram.discard(P)
        >>> assert para.same(F.unitality().cod, F(Diagram.id(S)))
        """
        require_parametric(self)
        hole = self.cod.id(self.functor(self.datatype.carrier))
        return para.Reparam.of(hole, self.cod.category.copy(self.param, 0))

    def associativity(self, term: markov.Diagram, *children: markov.Diagram
                      ) -> para.Reparam:
        """
        The multiplication of the lax algebra at a term with holes and a
        child for each hole: a reparametrisation from the fold of the term of
        the folds of the children to the fold of the substituted term, which
        copies the parameter (*Categorical Deep Learning*, Definition F.2). At
        a constructor, it is the square of Appendix J.

        Parameters:
            term : A diagram of constructors, with holes.
            children : One diagram with the carrier as codomain per hole.

        Raises:
            ValueError : If the algebra is not parametric or the children are
                not as many as the holes.

        Example
        -------
        >>> from discopy.markov import Box, Diagram, Ty
        >>> A, S, P = Ty("A"), Ty("S"), Ty("P")
        >>> L = Datatype.list(A, S)
        >>> nil, cons = L.constructors
        >>> Net = para.Markov[Diagram]
        >>> z = Net(Ty(), S, Box("z", P, S), param=P)
        >>> c = Net(A @ S, S, Box("c", A @ S @ P, S), param=P)
        >>> F = Algebra(L, {nil: z, cons: c}, {A: A, S: S}, Net)
        >>> square = F.associativity(cons, L.term(chain(0)))
        >>> assert square.inside == Diagram.copy(P, 2)
        >>> assert square.cod == F(L.term(chain(1)))
        """
        require_parametric(self)
        folds = [self(child) for child in children]
        dom = self.lift(term, *folds, param=self.param) >> self(term)
        cod = self(self.datatype.substitute(term, *children))
        return para.Reparam(dom, self.cod.category.copy(self.param, 2), cod)


@dataclass
class LaxHomomorphism:
    """
    A lax homomorphism from the initial algebra of a datatype to an algebra
    with a parameter :math:`P` (*Categorical Deep Learning*, Definition F.3
    and Appendix J): a parameter :math:`Q`, a fold of each closed term with
    this parameter, and one map :math:`\\kappa : Q \\to Q \\otimes P` for all
    constructors, such that the fold of a constructor applied to terms is the
    fold of its children, then the image of the constructor, reparametrised
    by :math:`\\kappa`. The map is the copy of the parameter when it is tied,
    and reads another copy of :math:`Q` at each depth otherwise: the
    folds are the pairs :math:`(Q, \\kappa)`, so that not even the lax ones
    are tied. A strict homomorphism needs :math:`Q \\otimes P = Q`.

    The unit and multiplication of :math:`\\kappa` in :meth:`unitality` and
    :meth:`multiplication` are the axioms of Johnson and Yau, Definition
    6.5.9, of a lax morphism of algebras of the free monad. In a category
    where every map is copied, the unit forces :math:`\\kappa` to be a copy
    of :math:`Q` followed by a reparametrisation :math:`Q \\to P`, and the
    fold is the tied one. They are meant for the spiders: a box is neither
    copied nor discarded in the free category with copies and discards.
    The axioms are checked on diagram codomains: ``python.Function`` compares
    functions by identity, so they never hold there.

    Parameters:
        algebra : An algebra with a parameter.
        kappa : A diagram :math:`Q \\to Q \\otimes P` of the category of
            ``algebra.cod`` without parameters, e.g. a spider.

    Raises:
        ValueError : If the algebra is not parametric or the type of
            ``kappa`` is not :math:`Q \\to Q \\otimes P`.

    Example
    -------
    >>> from discopy.markov import Box, Diagram, Ty
    >>> A, S, P = Ty("A"), Ty("S"), Ty("P")
    >>> L = Datatype.list(A, S)
    >>> nil, cons = L.constructors
    >>> Net = para.Markov[Diagram]
    >>> z = Net(Ty(), S, Box("z", P, S), param=P)
    >>> c = Net(A @ S, S, Box("c", A @ S @ P, S), param=P)
    >>> F = Algebra(L, {nil: z, cons: c}, {A: A, S: S}, Net)
    >>> tied = LaxHomomorphism(F, Diagram.copy(P))
    >>> assert para.same(tied(chain(2)), F(L.term(chain(2))))
    >>> assert tied.unitality() and tied.multiplication()
    >>> pair = LaxHomomorphism(F, Diagram.from_function([1, 0, 0], P @ P))
    >>> assert not pair.unitality() and pair.param == P @ P
    >>> assert not para.same(pair(chain(2)), F(L.term(chain(2))))
    """
    algebra: Algebra
    kappa: markov.Diagram

    def __post_init__(self):
        require_parametric(self.algebra)
        if not isinstance(self.kappa, self.algebra.cod.category):
            raise ValueError(f"Expected a {self.algebra.cod.category}, "
                             f"got {self.kappa!r}.")
        if self.kappa.cod != self.param @ self.algebra.param:
            raise ValueError(f"Expected a map into {self.param} @ "
                             f"{self.algebra.param}, got {self.kappa.cod}.")

    @property
    def param(self):
        """ The parameter :math:`Q` of the folds, the domain of kappa. """
        return self.kappa.dom

    def top(self, value: tuple) -> para.Symmetric:
        """
        The map above the square of a closed term ``(name, *children)``: the
        constructor applied to the folds of the children.

        Parameters:
            value : A nested tuple, see :meth:`Datatype.term`.
        """
        box, children = self.algebra.datatype.unpack(value)
        folds = [self(child) for child in children]
        return self.algebra.lift(box, *folds, param=self.param)\
            >> self.algebra.ar_map[box]

    def __call__(self, value: tuple) -> para.Symmetric:
        """
        The fold of a closed term, by recursion on its constructors.

        Parameters:
            value : A nested tuple, see :meth:`Datatype.term`.
        """
        return self.top(value).reparam(self.kappa)

    def preservation(self, value: tuple) -> para.Reparam:
        """
        The square of a closed term: a reparametrisation from its :meth:`top`
        to its fold, which holds by construction.

        Parameters:
            value : A nested tuple, see :meth:`Datatype.term`.
        """
        top = self.top(value)
        return para.Reparam(top, self.kappa, self(value))

    def unitality(self) -> markov.Equation:
        """
        The unit axiom of :math:`\\kappa`: discarding the parameter it
        produces gives the identity of :math:`Q`.
        """
        category = self.algebra.cod.category
        return category.equation_factory(
            self.kappa >> self.param @ category.copy(self.algebra.param, 0),
            category.id(self.param))

    def multiplication(self) -> markov.Equation:
        """
        The multiplication axiom of :math:`\\kappa`: copying the parameter it
        produces is applying it twice.
        """
        category, param = self.algebra.cod.category, self.algebra.param
        return category.equation_factory(
            self.kappa >> self.param @ category.copy(param),
            self.kappa >> self.kappa @ param)
