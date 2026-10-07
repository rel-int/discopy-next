""" The patterns read off signatures, their matching, and the search
for the terms of a category by its rules. """

from typing import TypeVar, get_args, get_origin

from hypothesis import find
from hypothesis import strategies as st
from pytest import raises

from discopy import braided, cat, rigid
from discopy.abc import (
    BiclosedCategory, Category, ColouredMonoid, FeedbackCategory,
    MonoidalCategory, TracedCategory)
from discopy.monoidal import Box, Diagram, Ty
from discopy.pattern import (
    Atom, Count, D, Declaration, Hom, Obj, Over, R, Repeat, Rule, Sort,
    Tensor, Unit, Var, match, rule, search)
from discopy.utils import AxiomError


A, B = TypeVar("A"), TypeVar("B")
x, y = Ty("x"), Ty("y")


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


def test_sorts():
    """ ``Obj``, ``Atom``, ``Unit``, ``Count`` and ``bool`` are sorts
    with no variable, bounding one or sampled as a premise; ``Var``
    refers to a variable, or a pattern over them, bound elsewhere. """
    from discopy.axioms import Axiom

    def bound[X: Atom[Ty], N: Count, M: Obj[Ty, N]](): ...
    sorts = [Sort.of(variable) for variable in bound.__type_params__]
    assert list(map(str, sorts)) == ["Atom[Ty]", "Count", "Obj[Ty, N]"]

    def law[X: Atom[Ty]](cls, x: Var[Ty, X], y: Atom[Ty]):
        return cls.Equation(cls.id(x), cls.id(y))
    law = Axiom(law).bind(Diagram)
    assert law.variables["X"].atomic and law.premises["y"].atomic
    equation = find(law.strategy(), lambda _: True)
    assert len(equation.terms[0].dom) == len(equation.terms[1].dom) == 1
    assert [s for s, _ in match(Var[Ty, A], x)] == [{"A": x}]
    assert list(match(Atom[Ty], x)) == [({}, ())]
    assert not list(match(Atom[Ty], x @ y))
    assert list(match(Unit[Ty], Ty())) and not list(match(Unit[Ty], x))


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


def test_delay_and_image():
    """ A delay matches every number of steps the value is the delay of,
    and the image of a functor is checked once it is bound, a residual
    until then. """
    from typing import Literal
    from discopy import feedback
    from discopy.pattern import Image, instantiate

    x = feedback.Ty("x")
    X, F = TypeVar("X"), TypeVar("F")
    N = TypeVar("N", bound=Count)
    assert [s for s, _ in match(D[X, N], x.delay(2))] == [
        {"N": 0, "X": x.delay(2)}, {"N": 1, "X": x.d}, {"N": 2, "X": x}]
    assert [s for s, _ in match(D[X], x.d)] == [{"X": x}]
    assert instantiate(D[X, Literal[2]], {"X": x}, feedback.Ty) == x.delay(2)
    relabel = feedback.Functor({x: x.d}, {})
    assert instantiate(Image[F, X], {"F": relabel, "X": x}, None) == x.d
    ((_, residuals), ) = match(Image[F, X], x.d)
    assert residuals == ((Image[F, X], x.d), )
    assert list(match(Image[F, X], x.d, {"F": relabel, "X": x}))
    assert not list(match(Image[F, X], x, {"F": relabel, "X": x}))


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
    from discopy.pattern import instantiate

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
    from discopy.pattern import instantiate

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
    from discopy.pattern import instantiate

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
    assert str(feedback.premises["mem"]) == "Var[C0 | None, M]"


def test_rule():
    assert repr(Rule(Category.then.function)) == "Rule(then)"
    assert repr(Category.then) == "abc.Category.then"
    assert cat.Arrow.then is not None and cat.Arrow.rules["then"].category\
        is cat.Arrow
    assert hash(Category.then) == hash(Category.then.bind(Category))
    assert Category.then.__isabstractmethod__
    assert cat.Arrow.rules["then"].owner is cat.Arrow  # The latest wins.
    assert str(cat.Arrow.rules["then"].conclusion).endswith(", A, C]")\
        and str(Category.then.conclusion).endswith(", A, C]")
    with raises(TypeError):
        Rule(Category.then.function).scope
    with raises(TypeError):
        rule(classmethod(lambda cls: None))

    class Wrapped(Diagram):
        @rule
        def twice[A](self: Hom[Diagram, A, A]) -> Hom[Diagram, A, A]:
            """ A rule declared and implemented in one place. """
            return self >> self

    f = Box("f", x, x)
    assert Wrapped.twice(f) == f >> f == Wrapped(f.inside, x, x).twice()
    assert list(Wrapped.rules) == ["id", "tensor", "cut", "dagger", "twice"]
    assert str(Wrapped.rules["twice"]) == "twice(self: Hom[discopy."\
        "monoidal.Diagram, A, A]) -> Hom[discopy.monoidal.Diagram, A, A]"
    found = find(Wrapped.strategy(dom=x, cod=x, types=st.just(x)),
                 lambda value: len(value.boxes) == 2
                 and len(set(value.boxes)) == 1)
    assert found.boxes[0] == found.boxes[1]


def test_generator():
    assert Category.then.recursive
    assert not rigid.Diagram.rules["cups"].recursive
    assert "cups" in rigid.Diagram.generators
    assert rigid.Diagram.generators["cups"].category is rigid.Diagram
    braid = braided.Diagram.generators["braid"]
    assert [str(sort) for sort in braid.variables.values()]\
        == ["Atom[C0]", "Atom[C0]"]
    assert list(braid.premises) == ["left", "right"]

    class Lying(Diagram):
        @classmethod
        @rule
        def wrong[A: ColouredMonoid](
                cls, dom: Var[Ty, A]) -> Hom[Diagram, A, Unit[Ty]]:
            """ A generator whose conclusion lies. """
            return cls.id(dom)

    with raises(AxiomError):
        find(Lying.strategy(dom=x, cod=Ty()), lambda value: True)


def test_cut():
    """ The search composes in context by cut, while the methods the
    rule derives from stay callable outside of it. """
    z = Ty("z")
    f, g = Box("f", y, y @ y), Box("g", x @ y @ y @ z, z)
    assert f.cut(g, x, z) == x @ f @ z >> g
    assert list(Diagram.rules["cut"].variables) == list("ABCXY")
    assert list(Diagram.rules) == ["id", "tensor", "cut", "dagger"]
    assert Diagram.rules["cut"].recursive
    assert "then" in cat.Arrow.rules  # A mere category composes by then,
    assert f.then(Box("h", y @ y, z)).cod == z  # a monoidal one cuts.
    assert (f @ g).dom == f.dom @ g.dom


def test_calculus():
    """ The recursive rules in action at each level of the tower, the
    admissible and inapplicable ones curated out: ``then`` is a cut
    with empty contexts everywhere, the rigid curries are caps
    compositions, and a method an admissible rule leaves behind still
    runs. """
    from discopy import (
        balanced, biclosed, closed, compact, feedback, frobenius, markov,
        monoidal, pivotal, ribbon, symmetric, traced)

    composition, trace, dagger = ["tensor", "cut"], ["trace"], ["dagger"]
    for module, calculus in (
            (monoidal, composition + dagger),
            (braided, composition + dagger),
            (rigid, composition),
            (balanced, composition + trace + dagger),
            (symmetric, composition + trace + dagger),
            (traced, composition + trace + dagger),
            (markov, composition + trace + dagger),
            (pivotal, composition + trace + dagger),
            (ribbon, composition + trace + dagger),
            (compact, composition + trace + dagger),
            (frobenius, composition + trace + dagger),
            (biclosed, composition + ["curry"]),
            (closed, composition + trace + ["curry"]),
            (feedback, composition + trace
             + ["delay", "feedback"])):
        assert [name for name, found in module.Diagram.rules.items()
                if found.recursive] == calculus, module.__name__

    assert monoidal.Diagram.__dict__["then"].__admissible__
    assert "twist" not in compact.Diagram.generators
    twist = compact.Diagram.twist(compact.Ty("x"))
    assert twist.dom == twist.cod and not twist.inside


def test_declarations():
    class Hidden(Diagram):
        unitality = None
        cut = None

    assert "unitality" not in Hidden.axioms
    assert "cut" not in Hidden.rules
    assert list(Category.generators) == ["id"]

    assert list(Ty.rules) == ["id"]  # Composing objects is a method,
    # not a rule: Ty.then shadows the rule a mere category declares.


def test_focusing():
    """ A goal commits to the rule it applies deterministically, with an
    ``epsilon`` chance of escaping back to the full search. """
    from discopy import biclosed, rigid
    from discopy.pattern import focused

    a, b = biclosed.Ty("a"), biclosed.Ty("b")

    def rules_in_focus(cls, dom, cod):
        return [found.name for found, _ in focused([
            (r, list(r.match(dom, cod))) for r in cls.rules.values()],
            dom, cod, type(dom))]

    assert rules_in_focus(biclosed.Diagram, a, b << a) == ["curry"]
    assert [name for name, r in rigid.Diagram.rules.items()
            if r.recursive] == ["tensor", "cut"]
    x, y, z = map(rigid.Ty, "xyz")
    curry = rule(rigid.Diagram.curry).bind(rigid.Diagram)
    assert not focused([(curry, list(curry.match(x, y @ z.l)))],
                       x, y @ z.l, rigid.Ty)  # z is no subformula of z.l.
    # A rigid curry is derived — caps and cut reach every transpose —
    # and self-dual types would let it focus on every goal.

    curried = find(search(
        biclosed.Diagram, biclosed.Box.strategy,
        dom=a, cod=b << a, epsilon=0), bool)
    assert isinstance(curried.boxes[-1], biclosed.Curry)
    for epsilon in (0.5, 1e-4):  # Support survives any epsilon > 0.
        escaped = find(search(
            biclosed.Diagram, biclosed.Box.strategy,
            dom=a, cod=b << a, epsilon=epsilon),
            lambda term: not any(
                isinstance(box, biclosed.Curry) for box in term.boxes))
        assert escaped.cod == b << a


def test_goal_patterns():
    """ The sides of a goal are patterns under a shared substitution. """
    from typing import TypeVar

    from discopy import markov

    A = TypeVar("A")
    loop = find(search(Diagram, Box.strategy, dom=A, cod=A),
                lambda term: len(term.boxes) == 1)
    assert loop.dom == loop.cod
    copy = find(
        search(markov.Diagram, markov.Box.strategy,
               dom=A, cod=Tensor[A, A]),
        lambda term: bool(term.boxes)
        and all(isinstance(box, markov.Copy) for box in term.boxes))
    assert copy.cod == copy.dom @ copy.dom


def test_constant():
    """ A word of a vocabulary is a rule with no premise. """
    from discopy.grammar import pregroup
    word = pregroup.Word('Alice', pregroup.Ty('n'))
    constant = Rule.constant(word)
    assert not constant.recursive and not constant.premises
    assert constant.__doc__ == "The constant Alice."
    assert constant.apply({}) == word
