"""
Peirce's beta graphs: first-order logic as the calculus of neo-Peircean
relations (NPR) of :cite:t:`BonchiEtAl24`, two cartesian bicategories on the
same objects, white and black, as :class:`discopy.frobenius` diagrams with
every wire coloured by its region.

Peirce's lines of identity are the white spiders. The black structure is the
white one seen through a cut: the black copy is ``~copy``, the black
composition :meth:`alpha.Diagram.relative_sum` and the black tensor
:meth:`alpha.Diagram.par`. A :class:`Model` is a functor into
:class:`discopy.tensor.Tensor` of ``bool``.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Wire
    Ty
    Diagram
    Box
    Spider
    Swap
    Cut
    Functor
    Model

Example
-------
A relation ``R`` on a set of two elements, its complement, union and
intersection with its converse, its relative sum with itself:

>>> x = Ty('x')
>>> R = Box('R', x, x)
>>> M = Model(ob={'x': 2}, ar={'R': [[True, False], [True, True]]})
>>> M(R).array.astype(int).tolist()
[[1, 0], [1, 1]]
>>> M(~R).array.astype(int).tolist()
[[0, 1], [0, 0]]
>>> M(R & R[::-1]).array.astype(int).tolist()
[[1, 0], [0, 1]]
>>> M(R | R[::-1]).array.astype(int).tolist()
[[1, 1], [1, 1]]
>>> M(R.relative_sum(R)).array.astype(int).tolist()
[[1, 1], [1, 1]]

Each dunder is the relation of its definition, in every model of 1 to 3
elements, on all their relations for one and two elements, a sample for
three:

>>> from itertools import product
>>> import numpy as np
>>> S = Box('S', x, x)
>>> for n in (1, 2, 3):
...     relations = list(product((0, 1), repeat=n * n))[::1 if n < 3 else 17]
...     for r, s in zip(relations, reversed(relations)):
...         M = Model(ob={'x': n}, ar={'R': r, 'S': s})
...         rel = lambda d: M(d).array.reshape(n, n)
...         r, s = rel(R), rel(S)
...         assert (rel(~R) == ~r).all()
...         assert (rel(R & S) == (r & s)).all()
...         assert (rel(R | S) == (r | s)).all()
...         assert (rel(R >> S) == (r @ s > 0)).all()
...         assert (rel(R.relative_sum(S)) == ~(~r @ ~s > 0)).all()
...         assert (rel(R[::-1]) == r.T).all()

The double cut is the identity in every model of 1 to 3 elements, on every
relation of each:

>>> for n in (1, 2, 3):
...     for r in product((0, 1), repeat=n * n):
...         M = Model(ob={'x': n}, ar={'R': r})
...         assert M(~~R) == M(R)

Axioms
------
The axioms of NPR, Fig. 9 of :cite:t:`BonchiEtAl24`, are stated on
:class:`Model` as inclusions of relations, each one a cell of the property
matrix, checked on the boxes ``a``, ``b``, ``c``, ``d`` of type ``x → x``:
every relation of that type is the image of a box in some model, so
quantifying over models of the boxes is quantifying over the relations the
axioms of the paper quantify over, at the types ``n = m = 1``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from discopy import frobenius, markov, monoidal
from discopy.peirce import alpha
from discopy.peirce.alpha import white
from discopy.tensor import Tensor
from discopy.testable import (
    Axiom, Equation, Inequation, Testable, axiom, no_strategy)
from discopy.utils import factory


class Wire(frobenius.Wire):
    """
    A beta wire is a frobenius wire on one region, :data:`white` by default.

    >>> assert Wire('x').dom == Wire('x').cod == white
    """
    def __init__(self, name: str, z: int = 0, dom=white, cod=None,
                 **params):
        super().__init__(name, z, dom=dom, cod=dom if cod is None else cod,
                         **params)


@factory
class Ty(frobenius.Ty):
    """
    A beta type is a frobenius type of :class:`Wire`. It has no closed
    constants: a constant of a coloured type is not globular yet, #635.

    >>> assert Ty('x', 'y') == Ty(Wire('x'), Wire('y'))
    """
    generator_factory = Wire
    strategy = no_strategy
    constant_factory = None


@factory
class Diagram(alpha.Diagram, frobenius.Diagram):
    """
    A beta diagram is a frobenius diagram with coloured wires, the dunders
    of :class:`alpha.Diagram` lifted from propositions to relations.

    >>> x = Ty('x')
    >>> R, S = Box('R', x, x), Box('S', x, x)
    >>> assert R & S == Spider(1, 2, x) >> R @ S >> Spider(2, 1, x)
    >>> assert R | S == ~(~R & ~S) and R.relative_sum(S) == ~(~R >> ~S)
    """
    ob = Ty
    strategy = no_strategy
    trace_factory = classmethod(frobenius.Diagram.dual_trace.__func__)

    def bubble(self, *args, **params) -> Cut:
        """ The :class:`Cut` of a diagram, the bubble of this category. """
        return Cut(self, *args, **params)

    def conjunction(self, other: Diagram) -> Diagram:
        """
        The intersection of two parallel white relations: copy, then both,
        then co-copy.
        """
        return self.spiders(1, 2, self.dom) >> self @ other\
            >> self.spiders(2, 1, self.cod)


class Box(frobenius.Box, Diagram):
    """
    A beta box is a relation symbol, globular: its colour its region's. A
    relation need be neither total nor a function, so a box is neither
    deterministic nor affine unless it says so.

    >>> x = Ty('x')
    >>> assert not Box('R', x, x).is_deterministic
    >>> assert not Box('R', x, x).is_affine
    """
    def __init__(self, name: str, dom, cod, **params):
        params.setdefault("is_deterministic", False)
        params.setdefault("is_affine", False)
        super().__init__(name, dom, cod, **params)

    def flags(self) -> dict[str, bool]:
        """
        The flags the box declares beyond those of a relation, by name, as
        :meth:`__repr__` and :meth:`to_tree` write them down.

        >>> from discopy import peirce
        >>> f = Box('f', Ty('x'), Ty('x'), is_deterministic=True)
        >>> assert eval(repr(f)) == f
        >>> assert f.flags() == {'is_deterministic': True}
        """
        return {name: True for name, value in [
            ("is_deterministic", self.deterministic),
            ("is_affine", self.affine)] if value}

    def __repr__(self):
        flags = "".join(f", {name}=True" for name in self.flags())
        result = super(markov.Box, self).__repr__()
        if self.is_dagger or not flags:
            return result
        return result[:-1] + flags + ")"


class Spider(frobenius.Spider, Box):
    """ A beta spider is a frobenius spider on a coloured wire. """


class Permutation(frobenius.Permutation, Box):
    """ A beta permutation is a frobenius permutation of coloured wires. """


class Cup(frobenius.Cup, Box):
    """
    A beta cup is a frobenius cup on wires of one region.

    >>> x = Ty('x')
    >>> assert isinstance(Diagram.cups(x, x), Diagram)
    """


class Cap(frobenius.Cap, Box):
    """ A beta cap is a frobenius cap on wires of one region. """


class Swap(Permutation, frobenius.Swap, Box):
    """ A beta swap is a frobenius swap of two wires of one region. """


class Cut(alpha.Cut, Box):
    """
    A beta cut is an :class:`alpha.Cut` around a frobenius diagram.

    >>> x = Ty('x')
    >>> assert (~Box('R', x, x)).arg == Box('R', x, x).negative()
    >>> assert not (~Box('R', x, x)).is_deterministic
    """
    def __init__(self, arg, dom=None, cod=None, **params):
        params.setdefault("is_deterministic", False)
        params.setdefault("is_affine", False)
        super().__init__(arg, dom=dom, cod=cod, **params)

    serialised_attrs = ("*args", ) + tuple(
        attr for attr in Box.serialised_attrs
        if attr not in ("*", "args", "method"))

    def to_tree(self) -> dict:
        """
        The tree of the bubble with both of its flags.

        >>> from discopy.utils import dumps, loads
        >>> cut = ~Box('R', Ty('x'), Ty('x'))
        >>> assert loads(dumps(cut)) == cut
        """
        return {**super().to_tree(), "is_deterministic": self.deterministic,
                "is_affine": self.affine}

    def __repr__(self):
        """
        The repr of the bubble with the flags it declares.

        >>> from discopy import peirce
        >>> R = Box('R', Ty('x'), Ty('x'))
        >>> cut = Cut(R.negative(), is_deterministic=True)
        >>> for cut in (~R, cut, cut.dagger()):
        ...     assert eval(repr(cut)) == cut
        """
        flags = "".join(f", {name}=True" for name in self.flags())
        result = alpha.Cut.__repr__(self)
        return result[:-1] + flags + ")" if flags else result

    @classmethod
    def from_tree(cls, tree: dict):
        result = super().from_tree(tree)
        result.deterministic = tree.get("is_deterministic", False)
        result.affine = tree.get("is_affine", False)
        return result


class Sum(alpha.Sum, frobenius.Sum, Box):
    """ A formal sum of beta diagrams with the same boundary. """


class Functor(alpha.Functor, frobenius.Functor):
    """ A beta functor is a frobenius functor that sends cuts as alpha. """
    dom = cod = Diagram


Diagram.box_factory = Box
Diagram.spider_factory = Spider  # ty: ignore[invalid-assignment]
Diagram.swap_factory = Swap
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
Diagram.permutation_factory = Permutation
Diagram.sum_factory = Sum
Diagram.functor_factory = Functor
Id = Diagram.id

x = Ty('x')
""" The sort the axioms are stated on. """

unit = Ty.id(white)
""" The unit type of the white region. """

a, b, c, d = (Box(name, x, x) for name in "abcd")
""" The boxes the axioms are stated on. """

copy, discard = Spider(1, 2, x), Spider(1, 0, x)
merge, start = Spider(2, 1, x), Spider(0, 1, x)
swap = Swap(x, x)


def inclusion(sides: Callable[[], tuple]) -> Axiom:
    """
    The axiom that the relation of one diagram is included in that of the
    other in every model.
    """
    @axiom
    def law(cls,
            model: Self,  # ty: ignore[invalid-type-form]
            ) -> Inequation[Tensor]:
        """ An inclusion of NPR, Fig. 9 of :cite:t:`BonchiEtAl24`. """
        # pylint: disable=unused-argument  # the law quantifies over models
        return Inequation(*map(model, sides()))
    return law


def equality(sides: Callable[[], tuple]) -> Axiom:
    """ The axiom that two diagrams are the same relation in every model. """
    @axiom
    def law(cls,
            model: Self,  # ty: ignore[invalid-type-form]
            ) -> Equation[Tensor]:
        """ An equation of NPR, Fig. 9 of :cite:t:`BonchiEtAl24`. """
        # pylint: disable=unused-argument  # the law quantifies over models
        return Equation(*map(model, sides()))
    return law


def black_id(typ: monoidal.Ty) -> alpha.Diagram:
    """ The black identity, the difference relation. """
    return ~Id(typ)


def linear_adjoint(diagram: alpha.Diagram) -> alpha.Diagram:
    """ The linear adjoint of a relation, the complement of its converse. """
    return ~diagram[::-1]


class Model(alpha.Model, Testable):
    """
    A beta model is a model of the sort ``x`` of 1 to 3 elements, each box a
    relation, the axioms of NPR its laws.

    >>> M = Model(ob={'x': 2}, ar=dict.fromkeys('abcd', [1, 0, 1, 1]))
    >>> from discopy import peirce
    >>> assert eval(repr(M)) == M
    >>> assert Model.copy_naturality(M) and Model.linear_distributivity_left(M)
    """
    diagram = Diagram

    @classmethod
    def strategy(cls, sizes: tuple[int, ...] = (1, 2, 3)):
        """
        Generate a model of the sort ``x`` of a size in ``sizes``, each of
        the boxes ``a``, ``b``, ``c``, ``d`` a relation of ``x``.
        """
        from hypothesis import strategies as st

        def relations(n):
            relation = st.lists(st.booleans(), min_size=n * n, max_size=n * n)
            return st.fixed_dictionaries(
                {name: relation for name in "abcd"}).map(
                    lambda ar: cls(ob={'x': n}, ar=ar))
        return st.sampled_from(sizes).flatmap(relations)

    # Cartesian bicategory, white (Fig. 2).
    copy_associativity = equality(lambda: (
        copy >> x @ copy, copy >> copy @ x))
    copy_unitality = equality(lambda: (copy >> x @ discard, Id(x)))
    copy_commutativity = equality(lambda: (copy >> swap, copy))
    merge_associativity = equality(lambda: (
        x @ merge >> merge, merge @ x >> merge))
    merge_unitality = equality(lambda: (x @ start >> merge, Id(x)))
    merge_commutativity = equality(lambda: (swap >> merge, merge))
    frobenius_law = equality(lambda: (
        copy @ x >> x @ merge, x @ copy >> merge @ x))
    speciality = equality(lambda: (copy >> merge, Id(x)))
    discard_counit = inclusion(lambda: (start >> discard, Id(unit)))
    copy_counit = inclusion(lambda: (merge >> copy, Id(x @ x)))
    discard_unit = inclusion(lambda: (Id(x), discard >> start))
    copy_unit = inclusion(lambda: (Id(x), copy >> merge))
    copy_naturality = inclusion(lambda: (c >> copy, copy >> c @ c))
    discard_naturality = inclusion(lambda: (c >> discard, discard))

    # Cocartesian bicategory, black (Fig. 3): the white laws through a cut.
    black_copy_associativity = equality(lambda: (
        (~copy).relative_sum(black_id(x).par(~copy)),
        (~copy).relative_sum((~copy).par(black_id(x)))))
    black_copy_unitality = equality(lambda: (
        (~copy).relative_sum(black_id(x).par(~discard)), black_id(x)))
    black_copy_commutativity = equality(lambda: (
        (~copy).relative_sum(~swap), ~copy))
    black_merge_associativity = equality(lambda: (
        black_id(x).par(~merge).relative_sum(~merge),
        (~merge).par(black_id(x)).relative_sum(~merge)))
    black_merge_unitality = equality(lambda: (
        black_id(x).par(~start).relative_sum(~merge), black_id(x)))
    black_merge_commutativity = equality(lambda: (
        (~swap).relative_sum(~merge), ~merge))
    black_frobenius_law = equality(lambda: (
        (~copy).par(black_id(x)).relative_sum(black_id(x).par(~merge)),
        black_id(x).par(~copy).relative_sum((~merge).par(black_id(x)))))
    black_speciality = equality(lambda: (
        (~copy).relative_sum(~merge), black_id(x)))
    black_discard_counit = inclusion(lambda: (
        (~discard).relative_sum(~start), black_id(x)))
    black_copy_counit = inclusion(lambda: (
        (~copy).relative_sum(~merge), black_id(x)))
    black_discard_unit = inclusion(lambda: (
        black_id(unit), (~start).relative_sum(~discard)))
    black_copy_unit = inclusion(lambda: (
        black_id(x).par(black_id(x)), (~merge).relative_sum(~copy)))
    black_copy_naturality = inclusion(lambda: (
        (~copy).relative_sum(c.par(c)), c.relative_sum(~copy)))
    black_discard_naturality = inclusion(lambda: (
        ~discard, c.relative_sum(~discard)))

    # Linear bicategory (Fig. 4).
    linear_distributivity_left = inclusion(lambda: (
        a >> b.relative_sum(c), (a >> b).relative_sum(c)))
    linear_distributivity_right = inclusion(lambda: (
        a.relative_sum(b) >> c, a.relative_sum(b >> c)))
    swap_white_unit = inclusion(lambda: (
        Id(x @ x), swap.relative_sum(~swap)))
    swap_white_counit = inclusion(lambda: (
        ~swap >> swap, black_id(x @ x)))
    swap_black_unit = inclusion(lambda: (
        Id(x @ x), (~swap).relative_sum(swap)))
    swap_black_counit = inclusion(lambda: (
        swap >> ~swap, black_id(x @ x)))
    box_white_unit = inclusion(lambda: (
        Id(x), c.relative_sum(linear_adjoint(c))))
    box_white_counit = inclusion(lambda: (
        linear_adjoint(c) >> c, black_id(x)))
    box_black_unit = inclusion(lambda: (
        Id(x), (~c).relative_sum(linear_adjoint(~c))))
    box_black_counit = inclusion(lambda: (
        linear_adjoint(~c) >> ~c, black_id(x)))
    white_par_identity = inclusion(lambda: (
        Id(x @ x), Id(x).par(Id(x))))
    black_tensor_identity = inclusion(lambda: (
        black_id(x) @ black_id(x), black_id(x @ x)))
    white_linear_strength_left = inclusion(lambda: (
        a.relative_sum(b) @ c.relative_sum(d),
        (a @ c).relative_sum(b.par(d))))
    black_linear_strength_left = inclusion(lambda: (
        a.par(c) >> b @ d, (a >> b).par(c >> d)))
    white_linear_strength_right = inclusion(lambda: (
        a.relative_sum(b) @ c.relative_sum(d),
        a.par(c).relative_sum(b @ d)))
    black_linear_strength_right = inclusion(lambda: (
        a @ c >> b.par(d), (a >> b).par(c >> d)))

    # First order bicategory: linear adjunctions and Frobenius (Fig. 5).
    copy_black_merge_unit = inclusion(lambda: (
        Id(x), copy.relative_sum(~merge)))
    copy_black_merge_counit = inclusion(lambda: (
        ~merge >> copy, black_id(x @ x)))
    discard_black_start_unit = inclusion(lambda: (
        Id(x), discard.relative_sum(~start)))
    discard_black_start_counit = inclusion(lambda: (
        ~start >> discard, black_id(unit)))
    black_copy_merge_unit = inclusion(lambda: (
        Id(x), (~copy).relative_sum(merge)))
    black_copy_merge_counit = inclusion(lambda: (
        merge >> ~copy, black_id(x @ x)))
    black_discard_start_unit = inclusion(lambda: (
        Id(x), (~discard).relative_sum(start)))
    black_discard_start_counit = inclusion(lambda: (
        start >> ~discard, black_id(unit)))
    merge_black_copy_unit = inclusion(lambda: (
        Id(x @ x), merge.relative_sum(~copy)))
    merge_black_copy_counit = inclusion(lambda: (
        ~copy >> merge, black_id(x)))
    start_black_discard_unit = inclusion(lambda: (
        Id(unit), start.relative_sum(~discard)))
    start_black_discard_counit = inclusion(lambda: (
        ~discard >> start, black_id(x)))
    black_merge_copy_unit = inclusion(lambda: (
        Id(x @ x), (~merge).relative_sum(copy)))
    black_merge_copy_counit = inclusion(lambda: (
        copy >> ~merge, black_id(x)))
    black_start_discard_unit = inclusion(lambda: (
        Id(unit), (~start).relative_sum(discard)))
    black_start_discard_counit = inclusion(lambda: (
        discard >> ~start, black_id(x)))
    linear_frobenius_black_white = equality(lambda: (
        (~copy) @ x >> x @ merge, x @ copy >> (~merge) @ x))
    linear_frobenius_white_black = equality(lambda: (
        copy @ x >> x @ ~merge, x @ ~copy >> merge @ x))
    black_linear_frobenius_white_black = equality(lambda: (
        copy.par(black_id(x)).relative_sum(black_id(x).par(~merge)),
        black_id(x).par(~copy).relative_sum(merge.par(black_id(x)))))
    black_linear_frobenius_black_white = equality(lambda: (
        (~copy).par(black_id(x)).relative_sum(black_id(x).par(merge)),
        black_id(x).par(copy).relative_sum((~merge).par(black_id(x)))))

    # De Morgan through the cut.
    double_cut = equality(lambda: (~~c, c))
    de_morgan = equality(lambda: (~(a & b), ~a | ~b))
    cut_complement = equality(lambda: (~c, c.negative()))
