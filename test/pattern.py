""" The sequent patterns, their collection and their matching. """

from typing import Annotated

from pytest import raises

from discopy.abc import (
    Category, ColouredMonoid, DelayedMonoid, FeedbackCategory, Pregroup,
    ResiduatedMonoid)
from discopy.monoidal import Ty
from discopy import pattern
from discopy.pattern import (
    UNIT, Adjoint, Atom, Count, Delay, Exp, Hom, L, Over, R, Repeat, Var,
    Sort, Tensor, Under, Unit, parse, sort_of)


x, y, z = map(Ty, "xyz")
A, B = (Var(name, Sort(bound=ColouredMonoid)) for name in "AB")
M = Var("M", Sort(atomic=True, bound=ColouredMonoid))
X = Var("X", Sort(atomic=True, bound=Pregroup))
D = Var("D", Sort(bound=DelayedMonoid))
E = Var("E", Sort(bound=ResiduatedMonoid))
N = Var("N", Count())
ONE = Unit(A.sort)


def test_operators():
    """ The operators of the objects build the compound patterns. """
    assert A @ B == Tensor(A, B) and A @ B @ M == Tensor(A, B, M)
    assert X.l == Adjoint(X, "l") and X.r == Adjoint(X, "r")
    assert D.d == Delay(D)
    assert (E << E) == Exp("<<", E, E) and (E >> E) == Exp(">>", E, E)
    assert M ** N == Repeat(M, N)
    assert str(Hom(X @ X.r, ONE)) == "C1[X @ X.r, Unit[C0]]"
    assert str(M ** N) == "M ** N"
    assert str((X @ X).l) == "(X @ X).l"
    assert str(D.d) == "D.d" and str((E << E) @ E) == "(E << E) @ E"

    with raises(TypeError):
        Tensor(A)
    with raises(TypeError):
        Repeat(A, 2)
    with raises(TypeError, match="atomic"):
        Repeat(A @ B, N)

    def cups[V: Atom](cls): ...
    def spiders[K: Count](cls): ...
    V, K = cups.__type_params__ + spiders.__type_params__
    assert Var(V).sort.atomic and Var(K).sort == Count()
    assert Hom(V, "W") == Hom(Var(V), Var("W"))  # Hom lifts a bare side.
    assert Hom([V, "W"], ()) == Hom(Var(V) @ Var("W"), UNIT)
    with raises(TypeError):
        Hom(42, "W")
    assert sort_of(Pregroup).bound is Pregroup
    with raises(TypeError):
        sort_of(int)


def test_repr():
    """ A pattern reads back from its representation. """
    for value in (A, X, A @ X @ A, Hom(A, A @ X), X.l, D.d, E << E,
                  X ** N, UNIT, Sort("C0", atomic=True)):
        assert eval(repr(value), vars(pattern)) == value


def test_formers():
    """ Subscripting a pattern class builds the pattern a bound states. """
    assert Atom["C0"] == Sort("C0", atomic=True)
    assert Unit["C0"] == UNIT and Tensor[A, B] == A @ B
    assert L[X] == X.l and R[X] == X.r and pattern.D[D] == D.d
    assert Over[E, E] == (E << E) and Under[E, E] == (E >> E)
    assert Repeat[M, N] == M ** N


def test_alias():
    """ The aliases of abc state the premises and the conclusion. """
    from discopy import abc
    C0, C1 = "C0", "C1"

    def cups[V: Atom](
            cls, left: abc.Obj[Ty, V], right: abc.Obj[Ty, R[V]]
    ) -> abc.Hom[C1, Tensor[V, R[V]], Unit[C0]]:
        ...
    assert str(parse(cups))\
        == "V: Atom[C0] | left: V, right: V.r ⊢ C1[V @ V.r, Unit[C0]]"

    def unary[V: Atom](cls, f: abc.Hom[C1, V]) -> None:
        ...
    with raises(TypeError):
        parse(unary, conclusion=False)

    def bare[V: Atom](cls, x: abc.Obj[Ty]) -> None:
        ...
    with raises(TypeError):
        parse(bare, conclusion=False)


def test_overloads():
    """ The overloads of a helper taking ``left`` restate the sequents of
    its two rules, and those of ``uncurry`` state ``curry`` upside down. """
    from typing import get_args, get_overloads
    import inspect

    from discopy.abc import BiclosedCategory, TracedCategory

    def sides(helper, owner):
        stubs = {}
        for stub in get_overloads(helper):
            function = getattr(stub, "__func__", stub)
            annotation = inspect.signature(
                function).parameters["left"].annotation
            stubs[get_args(annotation)[0]] = parse(function, owner=owner)
        return stubs[True], stubs[False]

    for owner, helper, left_rule, right_rule in (
            (TracedCategory, "trace", "trace_left", "trace_right"),
            (BiclosedCategory, "ev", "ev_left", "ev_right"),
            (BiclosedCategory, "curry", "curry_left", "curry_right"),
            (FeedbackCategory, "feedback", "feedback_left",
             "feedback_right")):
        left, right = sides(getattr(owner, helper), owner)
        assert str(left) == str(getattr(owner, left_rule).sequent)
        assert str(right) == str(getattr(owner, right_rule).sequent)

    for uncurried, rule in zip(
            sides(BiclosedCategory.uncurry, BiclosedCategory),
            ("curry_left", "curry_right")):
        curried = getattr(BiclosedCategory, rule).sequent
        assert uncurried.premises["self"] == curried.conclusion
        assert uncurried.conclusion == curried.premises["self"]


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

    def law[X](cls, x: Annotated[Ty, Var(X)], n: int = 1, *args, **kwargs):
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

    def two[V, W](cls) -> Annotated[str, Var(V), Var(W)]:
        ...
    with raises(TypeError, match="exactly one pattern"):
        parse(two)

    def unstated[X: Atom, N: Count](  # The conclusion's N has no premise.
            cls, x: Annotated[Ty, Var(X)]
    ) -> Annotated[Ty, Hom(X, Var(X) ** Var(N))]:
        ...
    with raises(TypeError, match="no premise states"):
        parse(unstated)
    assert str(FeedbackCategory.feedback_right.sequent) == (
        "A: C0, B: C0, M: Atom[C0] | self: C1[A @ M.d, B @ M] ⊢ C1[A, B]")
    assert FeedbackCategory.feedback_left.sequent.variables["M"].bound\
        is DelayedMonoid


def test_exp_unify():
    """ An exponential pattern decomposes a single exponential object
    its base and exponent rebuild, matches nothing else, and keeps the
    residual of a pregroup, whose exponentials are adjoint atoms. """
    from discopy import biclosed, rigid

    a, b = biclosed.Ty("a"), biclosed.Ty("b")
    Z = Var("Z", Sort(bound=ResiduatedMonoid))
    Y = Var("Y", Sort(bound=ResiduatedMonoid))
    ((subst, residuals),) = (Z << Y).match(b << a)
    assert subst == {"Z": b, "Y": a} and not residuals
    assert not list((Z >> Y).match(b << a))  # The symbols disagree.
    assert not list((Z << Y).match(b @ a))
    ((_, residual),) = (Z << Y).match(rigid.Ty("b") << rigid.Ty("a"))
    assert residual  # A pregroup exponential is two adjoint atoms.


def test_delay_unify():
    """ A delay pattern matches the delayed objects, undelayed. """
    from discopy import feedback

    x, y = feedback.Ty("x"), feedback.Ty("y")
    assert list(D.d.match(x.d @ y.d)) == [({"D": x @ y}, ())]
    assert not list(D.d.match(x.d @ y))
    assert list((D.d @ D.d).match((x @ x).d)) == [({"D": x}, ())]


def test_level():
    """ A pattern needs the level its known objects are bounded by. """
    assert Var.level() is Category and Tensor.level() is ColouredMonoid
    assert Adjoint.level() is Pregroup and Delay.level() is DelayedMonoid
    assert Exp.level() is ResiduatedMonoid and Unit.level() is ColouredMonoid
    assert (X @ D).l == Adjoint(Tensor(X, D), "l")
    assert (D @ X).l == Adjoint(Tensor(D, X), "l")  # Whatever the order.
    assert X.l.r.bound is Pregroup and Unit(X.sort).bound is Pregroup
    for build in (lambda: X.d, lambda: D.r, lambda: D >> D,
                  lambda: (A @ X).d, lambda: Adjoint(A, "r")):
        with raises(TypeError, match="needs a"):
            build()
    unknown = Var("U", Sort())
    assert (unknown @ A).bound is None  # Checked by parse with owner.

    def snake[U: Atom](cls, u: Annotated[Ty, Var(U)]) -> Annotated[
            Ty, Hom(Var(U) @ Var(U).r, UNIT)]:
        ...
    from discopy.abc import MonoidalCategory, RigidCategory
    with raises(TypeError, match="needs a"):
        parse(snake, owner=MonoidalCategory)
    assert parse(snake, owner=RigidCategory).conclusion is not None
    with raises(AttributeError):
        A.z


def test_errors_and_inverses():
    """ A head is a name, a bound names one, an annotation carries a
    pattern, and an adjoint or a repetition reads back what it built. """
    from discopy import abc, rigid
    with raises(TypeError, match="Expected a head"):
        Atom[42]
    with raises(TypeError, match="one head"):
        sort_of(abc.Obj["C0", "C1"])

    def bad(cls, f: Annotated[Ty, 42]): ...
    with raises(TypeError, match="pattern or a sort"):
        parse(bad, conclusion=False)

    with raises(KeyError, match="T is no head of this scope, which names C0"):
        Sort("T").resolve({"C0": Ty})

    def conflicting(cls, a: Annotated[Ty, Var("A", Sort(atomic=True))],
                    b: Annotated[Ty, Var("A")]): ...
    with raises(TypeError, match="both as Atom\\[C0\\] and as C0"):
        parse(conflicting, conclusion=False)
    assert list(X.r.match(rigid.Ty('x').r)) == [({"X": rigid.Ty('x')}, ())]
    assert (M ** N).instantiate({"M": x, "N": 2}, Ty) == x @ x
