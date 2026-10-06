""" The sequent patterns, their collection and their matching. """

from typing import Annotated

from pytest import raises

from discopy.abc import (
    Category, ColouredMonoid, Pregroup, ResiduatedMonoid)
from discopy.monoidal import Ty
from discopy import pattern
from discopy.pattern import (
    UNIT, Adjoint, Atom, Count, Exp, Hom, L, Ob, Over, R, Repeat,
    Sort, Tensor, Under, Unit, parse, sort_of)


x, y, z = map(Ty, "xyz")
A, B = (Ob(name, Sort(bound=ColouredMonoid)) for name in "AB")
M = Ob("M", Sort(atomic=True, bound=ColouredMonoid))
X = Ob("X", Sort(atomic=True, bound=Pregroup))
E = Ob("E", Sort(bound=ResiduatedMonoid))
N = Ob("N", Sort("Count"))
ONE = Unit(A.sort)


def test_operators():
    """ The operators of the objects build the compound patterns. """
    assert A @ B == Tensor(A, B) and A @ B @ M == Tensor(A, B, M)
    assert X.l == Adjoint(X, "l") and X.r == Adjoint(X, "r")
    assert (E << E) == Exp("<<", E, E) and (E >> E) == Exp(">>", E, E)
    assert M ** N == Repeat(M, N)
    assert str(Hom(X @ X.r, ONE)) == "C1[X @ X.r, Unit[C0]]"
    assert str(M ** N) == "M ** N"
    assert str((X @ X).l) == "(X @ X).l"
    assert str((E << E) @ E) == "(E << E) @ E"

    with raises(TypeError):
        Tensor(A)
    with raises(TypeError):
        Repeat(A, 2)
    with raises(TypeError, match="atomic"):
        Repeat(A @ B, N)

    def cups[V: Atom](cls): ...
    def spiders[K: Count](cls): ...
    V, K = cups.__type_params__ + spiders.__type_params__
    assert Ob(V).sort.atomic and Ob(K).sort == Sort("Count")
    assert Hom(V, "W") == Hom(Ob(V), Ob("W"))  # Hom lifts a bare side.
    assert Hom([V, "W"], ()) == Hom(Ob(V) @ Ob("W"), UNIT)
    with raises(TypeError):
        Hom(42, "W")
    assert sort_of(Pregroup).bound is Pregroup
    with raises(TypeError):
        sort_of(int)


def test_formers():
    """ Subscripting a pattern class builds the pattern a bound states. """
    assert Atom["C0"] == Sort("C0", atomic=True)
    assert Unit["C0"] == UNIT and Tensor[A, B] == A @ B
    assert L[X] == X.l and R[X] == X.r
    assert Over[E, E] == (E << E) and Under[E, E] == (E >> E)
    assert Repeat[M, N] == M ** N


def test_alias():
    """ The aliases of abc state the premises and the conclusion. """
    from discopy import abc
    C0, C1 = "C0", "C1"

    def cups[V: Atom](
            cls, left: abc.Ob[Ty, V], right: abc.Ob[Ty, R[V]]
    ) -> abc.Hom[C1, Tensor[V, R[V]], Unit[C0]]:
        ...
    assert str(parse(cups))\
        == "V: Atom[C0] | left: V, right: V.r ⊢ C1[V @ V.r, Unit[C0]]"

    def unary[V: Atom](cls, f: abc.Hom[C1, V]) -> None:
        ...
    with raises(TypeError):
        parse(unary, conclusion=False)

    def bare[V: Atom](cls, x: abc.Ob[Ty]) -> None:
        ...
    with raises(TypeError):
        parse(bare, conclusion=False)


def test_parse():
    def then[A, B, C](
            self: Annotated[str, Hom(A, B)],
            other: Annotated[str, Hom(B, C)]
    ) -> Annotated[str, Hom(A, C)]:
        ...
    sequent = parse(then)
    assert list(sequent.variables) == ["A", "B", "C"]
    assert list(sequent.premises) == ["self", "other"]
    assert str(sequent.conclusion) == "C1[A, C]"
    assert parse(then, conclusion=False).conclusion is None

    def law[X](cls, x: Annotated[Ty, Ob(X)], n: int = 1, *args, **kwargs):
        ...
    assert list(parse(law, conclusion=False).premises) == ["x"]
    assert str(parse(lambda cls: None, conclusion=False)) == ""
    with raises(TypeError):
        parse(law)
    namespace = {}
    exec(compile(  # A module deferring its annotations states strings.
        "from __future__ import annotations\ndef eager(cls, f: int): ...",
        "<deferred>", "exec", dont_inherit=True), namespace)
    with raises(TypeError, match="__future__"):
        parse(namespace["eager"], conclusion=False)

    def unannotated(cls, f):
        ...
    with raises(TypeError, match="states no pattern"):
        parse(unannotated, conclusion=False)

    def two[V, W](cls) -> Annotated[str, Ob(V), Ob(W)]:
        ...
    with raises(TypeError, match="exactly one pattern"):
        parse(two)

    def unstated[X: Atom, N: Count](  # The conclusion's N has no premise.
            cls, x: Annotated[Ty, Ob(X)]
    ) -> Annotated[Ty, Hom(X, Ob(X) ** Ob(N))]:
        ...
    with raises(TypeError, match="no premise states"):
        parse(unstated)


def test_level():
    """ A pattern needs the level its known objects are bounded by. """
    assert Ob.level() is Category and Tensor.level() is ColouredMonoid
    assert Adjoint.level() is Pregroup
    assert Exp.level() is ResiduatedMonoid and Unit.level() is ColouredMonoid
    assert X.l.r.bound is Pregroup and Unit(X.sort).bound is Pregroup
    with raises(TypeError, match="needs a"):
        Adjoint(A, "r")
    unknown = Ob("U", Sort())
    assert (unknown @ A).bound is None  # Checked by parse with owner.
    with raises(AttributeError):
        A.z
