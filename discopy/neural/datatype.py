# -*- coding: utf-8 -*-

"""
Inductive datatypes and their folds, the syntax of the recursive networks of
*Categorical Deep Learning* (arXiv:2402.15332).

A polynomial endofunctor :math:`F(S) = \\sum_i A_i \\otimes S^{\\otimes s_i}`
is given by its constructors
:math:`c_i : A_i \\otimes S^{\\otimes s_i} \\to S`, a :class:`Datatype`.
Its terms are the diagrams built from the constructors by composition and
tensor alone, with the data :math:`A_i` as inputs. An algebra
is a functor out of them, the fold of a term is its image, and the
uniqueness of the homomorphism is functoriality: see :class:`Algebra`.

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Datatype
    Algebra

.. autosummary::

    chain

Note
----
This module imports no tensor framework. Terms use composition and tensor
only, never swaps, copies or discards.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

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
        if not isinstance(value, tuple) or not value:
            raise ValueError(f"Expected (name, *children), got {value!r}.")
        name, *children = value
        box = self.constructor(name)
        if len(children) != self.arity(box):
            raise ValueError(f"{name} expects {self.arity(box)} children, "
                             f"got {len(children)}")
        inputs = reduce(lambda left, right: left @ right,
                        map(self.term, children),
                        markov.Diagram.id(self.data(box)))
        return inputs >> box

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


@dataclass
class Algebra:
    """
    An algebra of a datatype: one image for each constructor in a category
    ``cod``, i.e. a functor out of the free category on the constructors.
    Its fold sends a term to its image, which is the unique homomorphism.

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
            for box in boxes:
                if not isinstance(  # pylint: disable=W1116
                        self.ar_map[box], self.cod):
                    raise ValueError(
                        f"The image of {box.name} is not an arrow of "
                        f"{self.cod}.")
            params = {box.name: self.ar_map[box].param for box in boxes}
            if len(set(params.values())) > 1:
                raise ValueError(f"Expected one parameter, got {params}.")

    @property
    def parametric(self) -> bool:
        """ Whether the codomain is a category of parametric maps. """
        return isinstance(self.cod, type)\
            and issubclass(self.cod, para.Symmetric)

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

    def __call__(self, term: markov.Diagram):
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
