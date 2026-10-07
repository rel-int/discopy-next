""" The patterns read off signatures, their matching, and the search
for the terms of a category by its rules. """

from typing import TypeVar, get_args, get_origin

from hypothesis import find
from hypothesis import strategies as st
from pytest import raises

from discopy import braided, cat, rigid
from discopy.abc import (
    BiclosedCategory, Category, ColouredMonoid, FeedbackCategory,
    MonoidalCategory, RigidCategory, TracedCategory)
from discopy.monoidal import Box, Diagram, Ty
from discopy.pattern import (
    Atom, Count, Counts, D, Declaration, Hom, Obj, Objects, Over, R, Repeat,
    Rule, Sort, Tensor, Terms, Unit, Var, match, rule)
from discopy.utils import AxiomError


A, B = TypeVar("A"), TypeVar("B")
x, y = Ty("x"), Ty("y")


def test_read_off():
    """ Variables, premises and conclusion are what Python evaluates. """
    cut = Category.cut
    assert list(cut.variables) == ["A", "B", "C"]
    assert all(isinstance(sort, Objects) and sort.size is None
               for sort in cut.variables.values())
    assert list(cut.premises) == ["self", "other"]
    assert str(cut.conclusion) == "Hom[C1, A, C]"
    _, dom, _ = get_args(MonoidalCategory.mix.conclusion)
    assert get_origin(dom) is Tensor
    assert [variable.__name__ for variable in get_args(dom)] == ["A", "C"]

    def spiders[X: Atom, N: Count](): ...
    X, N = spiders.__type_params__
    assert Sort.of(X).atomic and Sort.of(N) == Counts()


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
    assert [s for s, _ in match(D[X], x.d)] == [{"X": x}]
    assert not list(match(D[X], x))
    assert instantiate(D[D[X]], {"X": x}, feedback.Ty) == x.delay(2)
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


def test_adjoints():
    """ An adjoint pattern inverts to the adjoint on the other side, an
    exponential one matches the exponential of its own side only. """
    from discopy import biclosed, rigid
    from discopy.pattern import L, Under, instantiate

    r = rigid.Ty("r")
    assert [s for s, _ in match(L[A], r.l)] == [{"A": r}]
    assert [s for s, _ in match(R[A], r.r)] == [{"A": r}]
    x, y = biclosed.Ty("x"), biclosed.Ty("y")
    assert [s for s, _ in match(Under[B, A], y >> x)] == [{"A": x, "B": y}]
    assert not list(match(Under[B, A], x << y))
    assert instantiate(Under[B, A], {"A": x, "B": y}, biclosed.Ty) == y >> x


def test_trace():
    """ A trace is a rule per side forwarding to the one method
    :meth:`trace`, of a memory partitioning the boundary as that of a
    feedback does. """
    from discopy import traced

    left, right = TracedCategory.trace_left, TracedCategory.trace_right
    assert left.variables["M"].size is None
    assert list(left.premises) == ["self", "dom", "cod", "mem"]
    x, y, a, b = map(traced.Ty, "xyab")
    goal = (x @ y @ a, x @ y @ b)
    assert [s["M"] for s, _ in match(left.premises["self"], goal)]\
        == [traced.Ty(), x, x @ y]
    assert [s["M"] for s, _ in match(right.premises["self"], goal)]\
        == [traced.Ty()]  # The right ends a and b share no wire.
    f = traced.Box("f", *goal)
    assert f.trace_left(mem=x @ y) == f.trace(dom=a, cod=b, left=True)\
        == f.trace(dom=y @ a, cod=y @ b, left=True).trace(left=True)
    with raises(AxiomError):
        f.trace(mem=x, cod=b, left=True)
    assert str(Objects(size="N")) == "Obj[_, N]"
    canonical = traced.Diagram.trace_iteration_left.canonical()
    assert canonical and str(canonical.terms[0]).count("Trace") == 1


def test_curry_and_uncurry():
    """ A curry is a rule per side, concluding on the exponential of its
    whole exponent, which the uncurry evaluates back. """
    from discopy import biclosed
    from discopy.pattern import instantiate

    x, y, z, w = map(biclosed.Ty, "xyzw")
    f = biclosed.Box("f", x @ y @ z, w)
    for curry, left in ((BiclosedCategory.curry_left, True),
                        (BiclosedCategory.curry_right, False)):
        assert list(curry.premises) == [
            "self", "context", "base", "exponent"]
        for subst, _ in match(curry.premises["self"], (f.dom, f.cod)):
            curried = f.curry(exponent=subst["Y"], left=left)
            assert instantiate(curry.conclusion, subst, biclosed.Ty)\
                == (curried.dom, curried.cod)
            if subst["Y"]:
                uncurried = curried.uncurry(left=left)
                assert (uncurried.dom, uncurried.cod) == (f.dom, f.cod)


def test_ev_and_feedback():
    """ The evaluation and the feedback are a rule per side each, the
    memory of a feedback any object. """
    from discopy import biclosed
    from discopy.pattern import instantiate

    x, y = biclosed.Ty("x"), biclosed.Ty("y")
    for ev, left in ((BiclosedCategory.ev_left, True),
                     (BiclosedCategory.ev_right, False)):
        assert list(ev.premises) == ["base", "exponent"]
        built = biclosed.Diagram.ev(y, x, left)
        assert instantiate(ev.conclusion, {"Y": y, "E": x}, biclosed.Ty)\
            == (built.dom, built.cod)
    for feedback in (FeedbackCategory.feedback_left,
                     FeedbackCategory.feedback_right):
        assert feedback.variables["M"].size is None
        assert str(feedback.premises["mem"]) == "Var[C0 | None, M]"


def test_rule():
    assert repr(Rule(Category.cut.function)) == "Rule(cut)"
    assert repr(Category.cut) == "abc.Category.cut"
    assert cat.Arrow.rules["cut"].category is cat.Arrow
    assert hash(Category.cut) == hash(Category.cut.bind(Category))
    assert Category.then.__isabstractmethod__  # A method, not a rule.
    assert not isinstance(vars(Category)["then"], Rule)
    assert Diagram.rules["cut"].owner is MonoidalCategory  # The latest wins.
    assert str(cat.Arrow.rules["cut"].conclusion).endswith(", A, C]")
    with raises(TypeError):
        Rule(Category.cut.function).unit
    with raises(TypeError):
        rule(classmethod(lambda cls: None))

    class Wrapped(Diagram):
        @rule
        def twice[A](self: Hom[Diagram, A, A]) -> Hom[Diagram, A, A]:
            """ A rule declared and implemented in one place. """
            return self >> self

    f = Box("f", x, x)
    assert Wrapped.twice(f) == f >> f == Wrapped(f.inside, x, x).twice()
    assert list(Wrapped.rules) == ["ax", "cut", "mix", "dagger", "twice"]
    assert str(Wrapped.rules["twice"]) == "twice(self: Hom[discopy."\
        "monoidal.Diagram, A, A]) -> Hom[discopy.monoidal.Diagram, A, A]"
    found = find(Wrapped.strategy(dom=x, cod=x),
                 lambda value: len(value.boxes) == 2
                 and len(set(value.boxes)) == 1)
    assert found.boxes[0] == found.boxes[1]


def test_generator():
    assert Category.cut.recursive
    assert not rigid.Diagram.rules["cups"].recursive
    assert "cups" in rigid.Diagram.generators
    assert rigid.Diagram.generators["cups"].category is rigid.Diagram
    braid = braided.Diagram.generators["braid"]
    assert [str(sort) for sort in braid.variables.values()]\
        == ["Atom", "Atom"]
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
    assert list(Diagram.rules) == ["ax", "cut", "mix", "dagger"]
    assert Diagram.rules["cut"].recursive
    assert list(cat.Arrow.rules["cut"].premises) == ["self", "other"]
    assert f.then(Box("h", y @ y, z)).cod == z  # A method, n-ary.
    assert (f @ g).dom == f.dom @ g.dom


def test_calculus():
    """ The recursive rules in action at each level of the tower, the
    admissible and inapplicable ones curated out: the rigid curries are
    caps compositions, and a method an admissible rule leaves behind
    still runs. """
    from discopy import (
        balanced, biclosed, closed, compact, feedback, frobenius, markov,
        monoidal, pivotal, ribbon, symmetric, traced)

    composition, dagger = ["cut", "mix"], ["dagger"]
    trace = ["trace_left", "trace_right"]
    curry = ["curry_left", "curry_right"]
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
            (biclosed, composition + curry),
            (closed, composition + trace + curry),
            (feedback, composition + trace
             + ["delay", "feedback_left", "feedback_right"])):
        assert [name for name, found in module.Diagram.rules.items()
                if found.recursive] == calculus, module.__name__

    assert "twist" not in compact.Diagram.generators
    twist = compact.Diagram.twist(compact.Ty("x"))
    assert twist.dom == twist.cod and not twist.inside


def test_declarations():
    class Hidden(Diagram):
        unitality = None
        cut = None

    assert "unitality" not in Hidden.axioms
    assert "cut" not in Hidden.rules
    assert list(Category.generators) == ["ax"]
    assert list(Ty.rules) == ["ax"]  # Objects compose by their tensor.


def test_focusing():
    """ A goal commits to the rule it applies deterministically, with an
    ``epsilon`` chance of escaping back to the full search: only a
    biclosed category has invertible rules to focus on. """
    from discopy import biclosed, rigid, symmetric

    a, b = biclosed.Ty("a"), biclosed.Ty("b")

    def rules_in_focus(cls, dom, cod):
        return [found.name for found, _ in cls.focus(
            cls.branches(dom, cod), dom, cod)]

    assert rules_in_focus(biclosed.Diagram, a, b << a) == ["curry_left"]
    assert [name for name, r in rigid.Diagram.rules.items()
            if r.recursive] == ["cut", "mix"]
    x, y, z = map(rigid.Ty, "xyz")
    curry = vars(RigidCategory)["curry_left"].bind(rigid.Diagram)
    assert not rigid.Diagram.focus(
        [(curry, list(curry.match(x, y @ z.l)))],
        x, y @ z.l)  # z is no subformula of z.l.
    # A rigid curry is derived — caps and cut reach every transpose —
    # and self-dual types would let it focus on every goal.
    s = symmetric.Ty("s")
    assert symmetric.Diagram.focus(
        symmetric.Diagram.branches(s, s), s, s) == []

    curried = find(biclosed.Diagram.search(
        dom=a, cod=b << a, epsilon=0), bool)
    assert isinstance(curried.boxes[-1], biclosed.Curry)
    for epsilon in (0.5, 1e-4):  # Support survives any epsilon > 0.
        escaped = find(biclosed.Diagram.search(
            dom=a, cod=b << a, epsilon=epsilon),
            lambda term: not any(
                isinstance(box, biclosed.Curry) for box in term.boxes))
        assert escaped.cod == b << a


def test_goal_patterns():
    """ The sides of a goal are patterns under a shared substitution. """
    from typing import TypeVar

    from discopy import markov

    A = TypeVar("A")
    loop = find(Diagram.search(dom=A, cod=A),
                lambda term: len(term.boxes) == 1)
    assert loop.dom == loop.cod
    copy = find(
        markov.Diagram.search(dom=A, cod=Tensor[A, A]),
        lambda term: bool(term.boxes)
        and all(isinstance(box, markov.Copy) for box in term.boxes))
    assert copy.cod == copy.dom @ copy.dom


def test_contexts():
    """ Each doctrine rewires a side of a goal by its own plumbing before
    a split of it puts a term in context: none for a planar category, a
    permutation for a symmetric one, copies and discards on the domain of
    a Markov one, cups and caps for a compact one, spiders for a
    hypergraph one. """
    from hypothesis import given, settings, strategies as st
    from discopy import compact, frobenius, markov, monoidal, symmetric

    def plumbing(level, dom, side, other):
        @st.composite
        def rewirings(draw):
            return level.Diagram.rewire(draw, dom, side, other)
        return rewirings()

    x, y = Ty("x"), Ty("y")
    assert find(plumbing(monoidal, x @ y, True, None), bool) == (
        x @ y, None)

    for level, kinds in (
            (symmetric, ("Swap", "Permutation")),
            (markov, ("Copy", "Discard")),
            (compact, ("Cup", "Cap")),
            (frobenius, ("Spider", ))):
        a, b = level.Ty("a"), level.Ty("b")
        for side in (True, False):
            if level is markov and not side:
                kinds = ("Swap", "Permutation")

            @settings(max_examples=30, database=None)
            @given(plumbing(level, a @ b @ a, side, b))
            def wired(rewiring):
                value, wiring = rewiring
                if wiring is not None:
                    assert (wiring.dom, wiring.cod) == (
                        (a @ b @ a, value) if side else (value, a @ b @ a))
            wired()
            found = find(plumbing(level, a @ b @ a, side, b),
                         lambda rewiring: rewiring[1] is not None and any(
                             type(box).__name__ in kinds
                             for box in rewiring[1].boxes))
            assert found[1] is not None, (level, side)


def test_constant():
    """ A word of a vocabulary is a rule with no premise. """
    from discopy.grammar import pregroup
    word = pregroup.Word('Alice', pregroup.Ty('n'))
    constant = Rule.constant(word)
    assert not constant.recursive and not constant.premises
    assert constant.__doc__ == "The constant Alice."
    assert constant.apply({}) == word
