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
    Algebra
    LaxHomomorphism
    Machine
    Coalgebra
    ToStream

.. autosummary::

    chain
    plug

The parameter of an algebra is copied into every constructor, which makes
the fold of a term a lax algebra of the free monad of the datatype, in the
2-category of parametric maps (*Categorical Deep Learning*, Appendix F and J).
The free monad is the free category of the terms with holes, so every
coherence law is a :class:`discopy.para.Cell` between maps of a given shape,
checked by the equation of its category: :meth:`Algebra.lift` is the free
monad on maps, :meth:`Algebra.lift_cell` on cells,
:meth:`Algebra.unitality` and :meth:`Algebra.associativity` are the cells of the lax algebra, copies of the parameter, and
:class:`LaxHomomorphism` the folds of a datatype that are lax and not tied.
The base is the free copy-discard category: no box is copyable or
discardable, so a reparametrisation by a box is rejected by every cell
where the parameter is copied twice or discarded.

Dually, a coalgebra :math:`S \\to F(S)` of :math:`F(S) = (I \\to O \\times S)`
(Mealy), :math:`O \\times (I \\to S)` (Moore) or :math:`O \\times S` (unfold)
is a cell from an input and the delayed state to an output and the state, a
:class:`Machine`. Its unfold is the feedback of the cell, the unique map into
the final coalgebra without parameters. With parameters, the tied map is the
homomorphism whose 2-cell is the copy of the parameters (App. J), a choice and
not forced. The semantics are the monoidal
streams of :mod:`discopy.stream`: the first steps are a finite diagram with
one box for each time step and one initial state as input. A
:class:`Coalgebra` is a functor out of them, sending every time step of a box
to one cell and the initial state to one ``start``, sampled once and copied by
the cell, and the unrolled image ties the cells by a copy of the parameter.

Note
----
This module imports no tensor framework. Terms use composition and tensor
only, never swaps, copies or discards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Any

from discopy import feedback, markov, para, stream
from discopy.utils import assert_isinstance

DIGITS = "0123456789"


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
        check_term(value)
        name, *children = value
        box = self.constructor(name)
        if len(children) != self.arity(box):
            raise ValueError(f"{name} expects {self.arity(box)} children, "
                             f"got {len(children)}")
        inputs = reduce(lambda left, right: left @ right,
                        map(self.term, children),
                        markov.Diagram.id(self.data(box)))
        return inputs >> box

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


def check_term(value: tuple):
    """
    Raise ``ValueError`` unless a value is a tuple ``(name, *children)``.

    Parameters:
        value : The candidate term.
    """
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"Expected (name, *children), got {value!r}.")


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
    if not parts:
        return identity(dom)
    return reduce(lambda left, right: left @ right, parts[1:], parts[0])


def require_parametric(algebra):
    """
    Raise ``ValueError`` unless an algebra has a parameter.

    Parameters:
        algebra : The :class:`Algebra` to check.
    """
    if not algebra.parametric:
        raise ValueError(f"{algebra.cod} is not a parametric category.")


def is_parametric(cod) -> bool:
    """
    Whether a codomain is a category of parametric maps.

    Parameters:
        cod : The codomain of an algebra or a coalgebra.

    Example
    -------
    >>> from discopy import python
    >>> assert is_parametric(para.Markov[python.Function])
    >>> assert not is_parametric(python.Function)
    """
    return isinstance(cod, type) and issubclass(cod, para.Symmetric)


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


@dataclass
class Algebra:
    """
    An algebra of a datatype: one image for each constructor in a category
    ``cod``, i.e. a functor out of the free category on the constructors.
    Its fold sends a term to its image. Without parameters it is the unique
    functor out of the terms. With parameters, the tied fold is the
    homomorphism whose 2-cell is the copy of the parameters (App. J), a choice
    and not forced: there is one for each choice of :math:`\\kappa`.

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
        for ty in types:
            for i in range(len(ty)):
                if ty[i:i + 1] not in self.ob_map:
                    raise ValueError(f"No image for the atom {ty[i:i + 1]}.")
        if self.parametric:
            check_param({box.name: self.ar_map[box] for box in boxes},
                        self.cod)

    @property
    def parametric(self) -> bool:
        """ Whether the codomain is a category of parametric maps. """
        return is_parametric(self.cod)

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
        image = self.functor(term)
        if not self.parametric:
            return image
        return image.reparam(self.cod.category.copy(
            self.param, len(term.boxes)))

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
            ValueError : If the algebra is not parametric, a map is a cell,
                there are no maps and no parameter, the parameters differ or
                the maps are not as many as the holes.

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
        if any(isinstance(m, para.Cell) for m in maps):
            raise ValueError("Expected maps, got a cell.")
        params = [m.param for m in maps] + ([] if param is None else [param])
        if not params:
            raise ValueError("Expected a param for a term without holes.")
        if any(p != params[0] for p in params):
            raise ValueError(f"Expected one parameter, got {params}.")
        inside = plug(term.dom, self.datatype.carrier, maps,
                      lambda atom: self.cod.id(self.functor(atom)))
        return inside.reparam(self.cod.category.copy(params[0], len(maps)))

    def lift_cell(self, term: markov.Diagram, *cells, param=None
                  ) -> para.Cell:
        """
        The free monad on a cell: the cells have the same inside, so has the
        result, which is a cell from the lift of their domains to the lift of
        their codomains. The equation of the cell rejects it unless the
        inside commutes with the copy.

        Parameters:
            term : A diagram of constructors, with holes.
            cells : A cell for each hole, with one inside, at least one.
            param : The parameter of the domains, as in :meth:`lift`. The
                codomains have their own, that of the cells.

        Raises:
            ValueError : If the algebra is not parametric, there are no cells,
                a cell is a map, the insides differ or the cells are not as
                many as the holes.

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
        >>> cell = para.Cell.id(F(nil))
        >>> assert F.lift_cell(cons, cell).inside == Diagram.id(P)
        """
        require_parametric(self)
        if not cells:
            raise ValueError("Expected at least one cell.")
        if not all(isinstance(c, para.Cell) for c in cells):
            raise ValueError("Expected cells, got a map.")
        if any(c.inside != cells[0].inside for c in cells):
            raise ValueError("Expected cells with the same inside.")
        return para.Cell(
            self.lift(term, *(c.dom for c in cells), param=param),
            self.lift(term, *(c.cod for c in cells)), cells[0].inside)

    def unitality(self) -> para.Cell:
        """
        The unit of the lax algebra: a cell from the identity of the carrier
        to the fold of the hole, which discards the parameter
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
        return para.Cell.reparam(
            hole, self.cod.category.copy(self.param, 0))

    def associativity(self, term: markov.Diagram, *children: markov.Diagram
                      ) -> para.Cell:
        """
        The multiplication of the lax algebra at a term with holes and a
        child for each hole: a cell from the fold of the term of the folds of
        the children to the fold of the substituted term, which copies the
        parameter (*Categorical Deep Learning*, Definition F.2). At a
        constructor, it is the square of Appendix J.

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
        return para.Cell(dom, cod, self.cod.category.copy(self.param, 2))


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
        check_term(value)
        box = self.algebra.datatype.constructor(value[0])
        folds = [self(child) for child in value[1:]]
        return self.algebra.lift(box, *folds, param=self.param)\
            >> self.algebra.ar_map[box]

    def __call__(self, value: tuple) -> para.Symmetric:
        """
        The fold of a closed term, by recursion on its constructors.

        Parameters:
            value : A nested tuple, see :meth:`Datatype.term`.
        """
        return self.top(value).reparam(self.kappa)

    def preservation(self, value: tuple) -> para.Cell:
        """
        The square of a closed term: a cell from its :meth:`top` to its fold,
        which holds by construction.

        Parameters:
            value : A nested tuple, see :meth:`Datatype.term`.
        """
        top = self.top(value)
        return para.Cell(top, self(value), self.kappa)

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


class ToStream(markov.Functor):
    """
    The functor from diagrams to monoidal streams, sending an atom :math:`x`
    to the sequence of types ``x0, x1, ...`` and every box to the sequence of
    boxes of the same name and time step. A delayed atom is sent to the delay
    of the sequence, preceded by the singleton of the atom itself: at the
    first time step, where the delay is empty, this is the initial state.

    Example
    -------
    >>> x = markov.Ty("x")
    >>> image = ToStream()(x @ x.delay())
    >>> print(image.now, "|", image.later.now)
    x0 @ x | x1 @ x0
    """
    def __init__(self):
        super().__init__(
            ob_map=self.atom, ar_map=self.box,
            cod=stream.Stream[markov.Diagram])

    @staticmethod
    def atom(atom: markov.Ty) -> stream.Ty:
        """ The sequence ``x0, x1, ...`` of an atom ``x``. """
        return stream.Ty.sequence(stream.Ty.base(str(atom)))

    def box(self, box: markov.Box) -> stream.Stream:
        """ The sequence of boxes ``f0, f1, ...`` of a box ``f``. """
        return stream.Stream[markov.Diagram].sequence(
            box.name, self(box.dom), self(box.cod))

    def __call__(self, other):
        if isinstance(other, feedback.Wire) and other.time_step:
            state = str(other.reset())
            initial = stream.Ty.singleton(stream.Ty.base(state))
            return initial @ super().__call__(other)
        return super().__call__(other)


@dataclass
class Machine:
    """
    A cell :math:`I \\otimes S^d \\to O \\otimes S` from an input and the
    delayed state to an output and the next state, the uncurried coalgebra of
    a Mealy, Moore or unfolding recurrent network. Its feedback is the
    recurrent network, the unfold into the final coalgebra, see
    `Categorical Deep Learning <https://arxiv.org/abs/2402.15332>`_, Figure 1.
    Without parameters it is the unique map into it; with parameters, the tied
    map has the copy of the parameters as 2-cell (App. J), a choice.

    The unrolled network takes the initial state :math:`S` once, as its first
    input, and the cell copies it where it needs to.

    The names of the boxes and of the atoms of the cell must not end with a
    digit: the atoms and boxes of the unrolling are named after their time
    step, and the functor to streams keeps no other record of it than that
    suffix, which :class:`Coalgebra` strips. A box reads the state only
    through its last wires, as many delayed copies of ``mem`` as it needs, and
    two boxes of the same name must have the same domain and codomain.

    Parameters:
        cell : The diagram from ``dom @ mem.delay()`` to ``cod @ mem``.
        dom : The input :math:`I`, empty for an unfold.
        cod : The output :math:`O`.
        mem : The state :math:`S`, not delayed.

    Raises:
        ValueError : If the cell is not of this shape.

    Example
    -------
    >>> from discopy.markov import Ty
    >>> I, O, S = Ty("I"), Ty("O"), Ty("S")
    >>> mealy = Machine.mealy(I, O, S)
    >>> assert mealy.cell.dom == I @ S.delay() and mealy.cell.cod == O @ S
    >>> unrolled = mealy.unroll(2)
    >>> print(unrolled.dom, "->", unrolled.cod)
    S @ I0 @ I1 @ I2 -> O0 @ O1 @ O2 @ S2
    >>> [box.name for box in unrolled.boxes if box.name.startswith("mealy")]
    ['mealy0', 'mealy1', 'mealy2']
    >>> moore = Machine.moore(I, O, S)
    >>> [box.name for box in moore.boxes]
    ['output', 'update']
    >>> assert moore.unroll(1).dom == Ty("S", "I0", "I1")
    >>> assert Machine.unfold(O, S).unroll(1).dom == S
    """
    cell: markov.Diagram
    dom: markov.Ty
    cod: markov.Ty
    mem: markov.Ty
    boxes: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        assert_isinstance(self.cell, markov.Diagram)
        if any(wire.time_step for wire in self.mem.inside):
            raise ValueError(f"The state {self.mem} must not be delayed.")
        shape = (self.dom @ self.mem.delay(), self.cod @ self.mem)
        if (self.cell.dom, self.cell.cod) != shape:
            raise ValueError(
                f"Expected a cell {shape[0]} -> {shape[1]}, got "
                f"{self.cell.dom} -> {self.cell.cod}.")
        self.boxes = self.generators()
        for box in self.boxes:
            self.reads(box)
        names = [box.name for box in self.boxes]
        names += [str(atom) for atom in self.atoms]
        for name in names:
            if name.endswith(tuple(DIGITS)):
                raise ValueError(f"The name {name} ends with a digit.")

    def generators(self) -> tuple:
        """
        The distinct generating boxes of the cell, i.e. without its copies,
        discards and permutations, by name: the attribute :attr:`boxes`.

        Raises:
            ValueError : If two boxes of the same name differ in their domain
                or codomain.
        """
        boxes = {}
        for box in self.cell.boxes:
            if isinstance(box, markov.Function):
                continue
            first = boxes.setdefault(box.name, box)
            if (first.dom, first.cod) != (box.dom, box.cod):
                raise ValueError(
                    f"The boxes named {box.name} have different types: "
                    f"{first.dom} -> {first.cod} and {box.dom} -> {box.cod}.")
        return tuple(boxes.values())

    @property
    def atoms(self) -> tuple:
        """ The distinct atoms of the types of the cell, never delayed. """
        types = (self.dom, self.cod, self.mem) + tuple(
            ty for box in self.boxes for ty in (box.dom, box.cod))
        return tuple(dict.fromkeys(
            markov.Ty(wire.name) for ty in types for wire in ty.inside))

    def box(self, name: str) -> markov.Box:
        """
        The generating box of a given name.

        Parameters:
            name : The name of the box.
        """
        for box in self.boxes:
            if box.name == name:
                return box
        names = [box.name for box in self.boxes]
        raise ValueError(f"Unknown box {name!r}, expected {names}.")

    def reads(self, box: markov.Box) -> int:
        """
        The number of times a box reads the delayed state, i.e. of copies of
        ``mem.delay()`` that end its domain.

        Parameters:
            box : A generating box of the cell.

        Raises:
            ValueError : If the box reads another delayed wire.
        """
        delayed = sum(1 for wire in box.dom.inside if wire.time_step)
        reads = delayed // len(self.mem) if self.mem else 0
        state = self.mem.delay() ** reads
        last = box.dom[len(box.dom) - len(state):]
        if delayed != len(state) or last != state:
            raise ValueError(
                f"The domain {box.dom} of {box.name} must end with "
                f"{reads} copies of {self.mem.delay()} and no other delayed "
                "wire.")
        return reads

    @classmethod
    def mealy(cls, dom: markov.Ty, cod: markov.Ty, mem: markov.Ty) -> Machine:
        """
        The Mealy machine :math:`S \\to (I \\to O \\times S)`: one box
        ``mealy`` from ``dom @ mem.delay()`` to ``cod @ mem``.

        Parameters:
            dom : The input :math:`I`.
            cod : The output :math:`O`.
            mem : The state :math:`S`.
        """
        cell = markov.Box("mealy", dom @ mem.delay(), cod @ mem)
        return cls(cell, dom, cod, mem)

    @classmethod
    def moore(cls, dom: markov.Ty, cod: markov.Ty, mem: markov.Ty) -> Machine:
        """
        The Moore machine :math:`S \\to O \\times (I \\to S)`: the box
        ``output`` from ``mem.delay()`` to ``cod`` and the box ``update`` from
        ``dom @ mem.delay()`` to ``mem`` read copies of the delayed state, so
        that the output at a time step does not depend on its input.

        Parameters:
            dom : The input :math:`I`.
            cod : The output :math:`O`.
            mem : The state :math:`S`.
        """
        state = mem.delay()
        output = markov.Box("output", state, cod)
        update = markov.Box("update", dom @ state, mem)
        cell = (dom @ markov.Diagram.copy(state) >> dom @ output @ state
                >> markov.Diagram.swap(dom, cod) @ state >> cod @ update)
        return cls(cell, dom, cod, mem)

    @classmethod
    def unfold(cls, cod: markov.Ty, mem: markov.Ty) -> Machine:
        """
        The unfolding machine :math:`S \\to O \\times S`: one box ``unfold``
        from ``mem.delay()`` to ``cod @ mem`` and no input.

        Parameters:
            cod : The output :math:`O`.
            mem : The state :math:`S`.
        """
        cell = markov.Box("unfold", mem.delay(), cod @ mem)
        return cls(cell, markov.Ty(), cod, mem)

    def feedback(self) -> markov.Diagram:
        """ The recurrent network: the delayed feedback of the cell. """
        return self.cell.feedback(dom=self.dom, cod=self.cod, mem=self.mem)

    def to_stream(self) -> ToStream:
        """ The functor to monoidal streams, see :class:`ToStream`. """
        return ToStream()

    def unroll(self, steps: int) -> markov.Diagram:
        """
        The first ``steps + 1`` time steps of the recurrent network, a finite
        diagram whose boxes carry their time step in their name. Its first
        input is the initial state, the state before the first step, which the
        boxes of the first step read through the copies of the cell.

        Parameters:
            steps : The number of steps after the first one.
        """
        functor = self.to_stream()
        initial = stream.Ty.singleton(
            stream.Ty.base(*(str(wire) for wire in self.mem.inside)))
        looped = functor(self.cell).feedback(
            dom=functor(self.dom) @ initial, cod=functor(self.cod),
            mem=functor(self.mem))
        unrolled = looped.unroll(steps).now
        size, state = len(self.dom), len(self.mem)
        if not size:
            return unrolled
        inputs = unrolled.dom
        swap = markov.Diagram.swap(inputs[size:size + state], inputs[:size])
        return swap @ inputs[size + state:] >> unrolled


@dataclass
class Coalgebra:
    """
    A coalgebra of a machine: one image for each box of its cell, sending
    every time step of a box to the same image, and the initial state
    ``start`` which a box reads at the first time step instead of the
    delayed state, a functor out of the unrolled machines.

    When ``cod`` is a category of parametric maps, the images and the start
    have the same parameter :math:`P`, which the unrolled image ties by one
    copy :math:`P \\to P^{\\otimes k}`, :math:`k` being the number of boxes of
    the unrolled diagram plus one for the start: the same cell with the same
    weights at every time step. The start is placed once, before the first
    step, and the copies of the cell, mapped to the copies of ``cod``, share
    it: for a stochastic start, one sample is read by every box that reads the
    state. The ``start`` is a field of its own, not an image of a box: a box
    may be named ``start``.

    The repr of a coalgebra is transparent only up to the maps it holds, as
    that of an :class:`Algebra`.

    Parameters:
        machine : The machine.
        ar_map : A map from the name of each box of the cell to its image,
            from the image of its domain to that of its codomain.
        ob_map : A map from the atoms of the cell to ``cod.ob``.
        cod : The codomain category, a Markov category.
        start : The arrow from the unit to the image of the state.

    Raises:
        ValueError : If a box or an atom has no image, if ``ar_map`` has a key
            that is not a box, if an image or the start has another boundary
            than expected or, in a parametric category, is not an arrow of
            ``cod``, if the parameters differ or if the parameter is empty.

    Example
    -------
    >>> from discopy import python
    >>> from discopy.markov import Ty
    >>> I, O, S = Ty("I"), Ty("O"), Ty("S")
    >>> Int = python.Ty(int)
    >>> F = Coalgebra(
    ...     Machine.mealy(I, O, S),
    ...     {"mealy": python.Function(
    ...         lambda a, s: (a + s, 2 * s + a), Int @ Int, Int @ Int)},
    ...     {I: Int, O: Int, S: Int}, python.Function,
    ...     python.Function(lambda: 1, python.Ty(), Int))
    >>> F(2)(1, 2, 3)
    (2, 5, 11, 19)
    """
    machine: Machine
    ar_map: dict
    ob_map: dict
    cod: type
    start: Any

    def __post_init__(self):
        self.check()

    def check(self):
        """
        Raise ``ValueError`` unless the boxes of the machine are exactly the
        keys of ``ar_map``, every atom of the machine has an image, in a
        parametric category the images and the start are arrows of ``cod``
        with the same non-empty parameter, and the boundaries of the images
        and of the start are the images of those of the boxes and of the
        state.
        """
        boxes = self.machine.boxes
        names = {box.name for box in boxes}
        for box in boxes:
            if box.name not in self.ar_map:
                raise ValueError(f"No image for the box {box.name}.")
        for name in self.ar_map:
            if name not in names:
                raise ValueError(f"No box {name} in the machine.")
        for atom in self.machine.atoms:
            if atom not in self.ob_map:
                raise ValueError(f"No image for the atom {atom}.")
        if self.parametric:
            images = {box: self.ar_map[box.name] for box in boxes}
            check_param(images | {"start": self.start}, self.cod)
            if not self.param:
                raise ValueError("Expected a non-empty parameter.")
        for box in boxes:
            self.check_boundary(box.name, self.ar_map[box.name], box.dom,
                                box.cod)
        self.check_boundary("start", self.start, markov.Ty(),
                            self.machine.mem)

    def ob(self, ty: markov.Ty):
        """
        The image of a type of the cell, delayed or not.

        Parameters:
            ty : A type of the machine.
        """
        return self.functor(markov.Ty(*(wire.name for wire in ty.inside)))

    def check_boundary(self, name: str, image, dom: markov.Ty, cod: markov.Ty):
        """
        Raise ``ValueError`` unless an image goes from the image of ``dom`` to
        the image of ``cod``.

        Parameters:
            name : The name of the box, for the error.
            image : The image of the box.
            dom : The domain of the box.
            cod : The codomain of the box.
        """
        if (image.dom, image.cod) != (self.ob(dom), self.ob(cod)):
            raise ValueError(
                f"The image of {name} goes from {image.dom} to {image.cod}, "
                f"expected {self.ob(dom)} -> {self.ob(cod)}.")

    @property
    def parametric(self) -> bool:
        """ Whether the codomain is a category of parametric maps. """
        return is_parametric(self.cod)

    @property
    def param(self):
        """ The common parameter of the images, ``None`` if not parametric. """
        return self.start.param if self.parametric else None

    def atom(self, atom: markov.Ty):
        """
        The image of an atom of an unrolled machine, i.e. of an atom of the
        cell at some time step.

        Parameters:
            atom : An atom of the cell followed by its time step.
        """
        return self.ob_map[markov.Ty(str(atom).rstrip(DIGITS))]

    def image(self, box: markov.Box):
        """
        The image of a box of an unrolled machine: that of the box of the cell
        of the same name, whatever its time step.

        Parameters:
            box : A box ``name + str(time_step)``.
        """
        return self.ar_map[box.name.rstrip(DIGITS)]

    @property
    def functor(self) -> markov.Functor:
        """ The functor from the unrolled machines. """
        return markov.Functor(
            ob_map=self.atom, ar_map=self.image, cod=self.cod)

    def __call__(self, steps: int):
        """
        The image of the unrolling of the machine to ``steps + 1`` time steps,
        with the ``start`` before the first step: one sample, which the cell
        copies. In a parametric category the copy of the parameter ties the
        images of the boxes of the unrolled diagram and the start.

        Parameters:
            steps : The number of steps after the first one.
        """
        unrolled = self.machine.unroll(steps)
        rest = self.cod.id(self.functor(unrolled.dom[len(self.machine.mem):]))
        image = self.start @ rest >> self.functor(unrolled)
        if not self.parametric:
            return image
        uses = sum(1 for box in unrolled.boxes
                   if not isinstance(box, markov.Function)) + 1
        return image.reparam(self.cod.category.copy(self.param, uses))
