""" The sequents read off signatures, and their matching. """

from typing import TypeVar, get_args, get_origin

from discopy.abc import (
    BiclosedCategory, Category, FeedbackCategory, MonoidalCategory,
    TracedCategory)
from discopy.pattern import Atom, Count, D, Hom, Over, R, Repeat, Tensor
from discopy.sequent import Declaration, Sort, match


A, B = TypeVar("A"), TypeVar("B")


def test_read_off():
    """ Variables, premises and conclusion are what Python evaluates. """
    then = Category.then
    assert list(then.variables) == ["A", "B", "C"]
    assert all(sort == Sort() for sort in then.variables.values())
    assert list(then.premises) == ["self", "other"]
    assert str(then.conclusion) == "Hom[C1, A, C]"
    _, dom, _ = get_args(MonoidalCategory.tensor.conclusion)
    assert get_origin(dom) is Tensor
    assert [variable.__name__ for variable in get_args(dom)] == ["A", "C"]

    def spiders[X: Atom, N: Count](): ...
    X, N = spiders.__type_params__
    assert Sort.of(X).atomic and Sort.of(N).count


def test_atom():
    """ ``Atom`` has the interface of ``Obj``: a coarse type, and an
    optional pattern unifying it with the other annotations. """
    from hypothesis import find
    from discopy.axioms import Axiom
    from discopy.monoidal import Diagram, Ty

    def bound[X: Atom[Ty]](): ...
    assert str(Sort.of(bound.__type_params__[0])) == "Atom[Ty]"

    def law[X](cls, x: Atom[Ty, X], y: Atom[Ty]):
        return cls.Equation(cls.id(x), cls.id(x))
    law = Axiom(law).bind(Diagram)
    assert law.variables["X"].atomic and law.premises["y"].atomic
    assert len(find(law.strategy(), lambda _: True).terms[0].dom) == 1
    x, y = Ty("x"), Ty("y")
    assert [s for s, _ in match(Atom[Ty, A], x)] == [{"A": x}]
    assert not list(match(Atom[Ty, A], x @ y))


def test_match():
    from discopy import rigid
    x, y = rigid.Ty("x"), rigid.Ty("y")
    X = TypeVar("X", bound=Atom)
    N = TypeVar("N", bound=Count)
    assert [s["X"] for s, _ in match(Tensor[X, R[X]], x @ x.r)] == [x]
    assert not list(match(Tensor[X, R[X]], x @ y.r))
    assert list(match(Hom[None, A, B], (x, None))) == [({"A": x}, ())]
    assert [s for s, _ in match(Repeat[X, N], x @ x @ x)]\
        == [{"N": 3, "X": x}]


def test_exp_unify():
    """ An exponential pattern decomposes a single exponential object
    its base and exponent rebuild, matches nothing else, and keeps the
    residual of a pregroup, whose exponentials are adjoint atoms. """
    from discopy import biclosed, rigid

    a, b = biclosed.Ty("a"), biclosed.Ty("b")
    ((subst, residuals),) = match(Over[A, B], b << a)
    assert subst == {"A": b, "B": a} and not residuals
    assert not list(match(Over[A, B], b @ a))
    ((_, residual),) = match(Over[A, B], rigid.Ty("b") << rigid.Ty("a"))
    assert residual


def test_delay_unify():
    from discopy import feedback

    x, y = feedback.Ty("x"), feedback.Ty("y")
    assert list(match(D[A], x.d @ y.d)) == [({"A": x @ y}, ())]
    assert not list(match(D[A], x.d @ y))


def test_sides():
    """ A side ``S: bool`` orients ``TensorDir``, ``ExpDir`` and
    ``AdjDir``: matching tries both sides unless the side is bound. """
    from discopy import biclosed, rigid
    from discopy.pattern import AdjDir, ExpDir, TensorDir
    from discopy.sequent import instantiate

    S = TypeVar("S", bound=bool)
    x, y = biclosed.Ty("x"), biclosed.Ty("y")
    assert [(s["S"], s["A"], s["B"]) for s, _ in match(
        TensorDir[A, B, S], x @ y) if len(s["A"]) == 1]\
        == [(True, x, y), (False, y, x)]
    assert [s["S"] for s, _ in match(ExpDir[A, B, S], x << y)] == [True]
    assert [s["S"] for s, _ in match(ExpDir[A, B, S], y >> x)] == [False]
    assert instantiate(TensorDir[A, B, S], {"A": x, "B": y, "S": False},
                       biclosed.Ty) == y @ x
    r = rigid.Ty("r")
    assert [s["S"] for s, _ in match(AdjDir[A, S], r.l)] == [True, False]
    assert Sort.of(S) == Sort("bool", side=True)


def test_trace():
    """ The n-ary trace is one rule on both sides: ``M`` is of size
    ``n``, the side ``S`` is ``left``. """
    from discopy import traced

    trace = TracedCategory.trace
    assert trace.variables["M"] == Sort(size="N")
    assert list(trace.premises) == ["self", "n", "left"]
    x, y, a, b = map(traced.Ty, "xyab")
    found = [(s["S"], s["N"], s["M"]) for s, _ in match(
        trace.premises["self"], (x @ y @ a, x @ y @ b))]
    assert found == [  # The right ends a and b share no wire.
        (True, 0, traced.Ty()), (True, 1, x), (True, 2, x @ y),
        (False, 0, traced.Ty())]
    assert str(Sort(size="N")) == "Obj[C0, N]"
    canonical = traced.Diagram.trace_iteration.canonical()
    assert str(canonical.terms[0]).count("Trace") == 2


def test_curry_and_uncurry():
    """ The curry concludes on the exponential of all ``n`` objects at
    once, as ``curry(n, left)`` builds it, and the uncurry states it
    upside down. """
    from discopy import biclosed
    from discopy.sequent import instantiate

    curry, uncurry = BiclosedCategory.curry, Declaration(
        BiclosedCategory.uncurry)
    assert str(uncurry.premises["self"]) == str(curry.conclusion)
    assert str(uncurry.conclusion) == str(curry.premises["self"])
    x, y, z, w = map(biclosed.Ty, "xyzw")
    f = biclosed.Box("f", x @ y @ z, w)
    for subst, _ in match(curry.premises["self"], (f.dom, f.cod)):
        curried = f.curry(subst["N"], left=subst["S"])
        assert instantiate(curry.conclusion, subst, biclosed.Ty)\
            == (curried.dom, curried.cod)
    for left in (True, False):
        g = biclosed.Box("g", x, w << y @ z if left else y @ z >> w)
        (subst, _), = match(uncurry.premises["self"], (g.dom, g.cod))
        assert (subst["S"], subst["N"], subst["Y"]) == (left, 2, y @ z)
        uncurried = g.uncurry(2, left)
        assert instantiate(uncurry.conclusion, subst, biclosed.Ty)\
            == (uncurried.dom, uncurried.cod)


def test_ev_and_feedback():
    """ The evaluation and the feedback are one rule each, on both
    sides; the memory of a feedback is any object. """
    from discopy import biclosed
    from discopy.sequent import instantiate

    ev = BiclosedCategory.ev
    assert list(ev.premises) == ["base", "exponent", "left"]
    x, y = biclosed.Ty("x"), biclosed.Ty("y")
    for left in (True, False):
        subst = {"Y": y, "E": x, "S": left}
        built = biclosed.Diagram.ev(y, x, left)
        assert instantiate(ev.conclusion, subst, biclosed.Ty)\
            == (built.dom, built.cod)
    feedback = FeedbackCategory.feedback
    assert feedback.variables["M"] == Sort()
    assert str(feedback.premises["mem"]) == "Obj[C0 | None, M]"
