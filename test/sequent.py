""" The sequents read off signatures, and their matching. """

from typing import TypeVar, get_args, get_origin, get_overloads

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


def test_overloads():
    """ The overloads of a helper taking ``left`` restate the sequents of
    its two rules, read off the same way. """
    for owner, helper, left_rule, right_rule in (
            (TracedCategory, "trace", "trace_left", "trace_right"),
            (BiclosedCategory, "ev", "ev_left", "ev_right"),
            (BiclosedCategory, "curry", "curry_left", "curry_right"),
            (FeedbackCategory, "feedback", "feedback_left",
             "feedback_right")):
        stubs = {}
        for stub in get_overloads(getattr(owner, helper)):
            stub = Declaration(getattr(stub, "__func__", stub))
            left = stub.__signature__.parameters["left"].annotation
            stubs[get_args(left)[0]] = stub
        for left, name in ((True, left_rule), (False, right_rule)):
            rule = getattr(owner, name)
            assert str(stubs[left].conclusion) == str(rule.conclusion)
            assert list(map(str, stubs[left].premises.values()))\
                == list(map(str, rule.premises.values()))
