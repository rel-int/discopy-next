""" The patterns read off signatures, their matching, and the sampling
of the terms of a category by its rules. """

from hypothesis import find
from pytest import raises

from discopy import abc, braided, cat, rigid
from discopy.abc import (
    BiclosedCategory, Category, ColouredMonoid, FeedbackCategory,
    MonoidalCategory, TracedCategory)
from discopy.monoidal import Box, Diagram, Ty
from discopy.pattern import (
    Count, D, Hom, Image, L, Obj, Over, Pattern, R, Repeat, Rule, rule,
    Substitution, Tensor, Under, Var)
from discopy.utils import AxiomError


A, B = Var("A"), Var("B")
x, y = Ty("x"), Ty("y")


def match(pattern, value, subst=None):
    """ Every substitution under which a pattern stands for a value. """
    return list(Substitution(subst or {}).unify(pattern, value))


def instantiate(pattern, subst, ob):
    """ The value a pattern stands for under a substitution. """
    return Substitution(subst).instantiate(pattern, ob)


def test_read_off():
    """ Variables, premises and conclusion are the patterns the
    annotations Python evaluates read as. """
    cut = Category.cut
    assert list(cut.variables) == ["A", "B", "C"]
    assert all(isinstance(sort, Obj) and sort.size is None
               for sort in cut.variables.values())
    assert list(cut.premises) == ["self", "other"]
    assert str(cut.conclusion) == "Hom[C1, A, C]"
    assert isinstance(MonoidalCategory.mix.conclusion.dom, Tensor)
    assert str(MonoidalCategory.mix.conclusion.dom) == "Tensor[A, C]"

    def spiders[X: abc.Atom, N: Count](): ...
    X, N = spiders.__type_params__
    assert Pattern.read(X).sort == Obj(size=1)
    assert Pattern.read(N) == Var("N", Count())


def test_sorts():
    """ ``Obj``, ``Atom``, ``Unit`` and ``Count`` read as sorts with no
    variable, bounding one or sampled as a premise; ``Var`` refers to a
    variable, or a pattern over them, bound elsewhere. """
    from discopy.axioms import Axiom

    def bound[X: abc.Atom[Ty], N: Count, M: abc.Obj[Ty, N]](): ...
    sorts = [Pattern.read(variable).sort
             for variable in bound.__type_params__]
    assert list(map(str, sorts)) == ["Atom[Ty]", "Count", "Obj[Ty, N]"]

    def law[X: abc.Atom[Ty]](cls, x: abc.Var[Ty, X], y: abc.Atom[Ty]):
        return cls.Equation(cls.id(x), cls.id(y))
    law = Axiom(law).bind(Diagram)
    assert law.variables["X"].size == law.premises["y"].size == 1
    equation = find(law.strategy(), lambda _: True)
    assert len(equation.terms[0].dom) == len(equation.terms[1].dom) == 1
    assert [dict(s) for s in match(A, x)] == [{"A": x}]
    assert match(Obj(size=1), x) == [{}]
    assert not match(Obj(size=1), x @ y)
    assert match(Obj(size=0), Ty()) and not match(Obj(size=0), x)
    assert str(Obj(size="N")) == "Obj[_, N]"


def test_match():
    x, y = rigid.Ty("x"), rigid.Ty("y")
    X, N = Var("X", Obj(size=1)), Var("N", Count())
    assert [s["X"] for s in match(Tensor(X, R(X)), x @ x.r)] == [x]
    assert not match(Tensor(X, R(X)), x @ y.r)
    assert match(Hom(A, B), Hom(x, None)) == [{"A": x}]
    assert [dict(s) for s in match(Repeat(X, N), x @ x @ x)]\
        == [{"N": 3, "X": x}]
    assert Tensor(X, R(X)).variables == ("X", "X")


def test_delay_and_image():
    """ A delay matches every number of steps the value is the delay of,
    and the image of a functor is checked once it is bound, a residual
    until then. """
    from discopy import feedback

    x = feedback.Ty("x")
    X, F = Var("X"), Var("F")
    assert [dict(s) for s in match(D(X), x.d)] == [{"X": x}]
    assert not match(D(X), x)
    assert instantiate(D(D(X)), {"X": x}, feedback.Ty) == x.delay(2)
    relabel = feedback.Functor({x: x.d}, {})
    assert instantiate(Image(F, X), {"F": relabel, "X": x}, None) == x.d
    (found, ) = match(Image(F, X), x.d)
    assert found.residuals == ((Image(F, X), x.d), )
    assert match(Image(F, X), x.d, {"F": relabel, "X": x})
    assert not match(Image(F, X), x, {"F": relabel, "X": x})


def test_exp_unify():
    """ An exponential pattern decomposes a single exponential object
    its base and exponent rebuild, matches nothing else, and keeps the
    residual of a pregroup, whose exponentials are adjoint atoms. """
    from discopy import biclosed

    a, b = biclosed.Ty("a"), biclosed.Ty("b")
    (subst, ) = match(Over(A, B), b << a)
    assert subst == {"A": b, "B": a} and not subst.residuals
    assert not match(Over(A, B), b @ a)
    (subst, ) = match(Over(A, B), rigid.Ty("b") << rigid.Ty("a"))
    assert subst.residuals


def test_delay_unify():
    from discopy import feedback

    x, y = feedback.Ty("x"), feedback.Ty("y")
    assert match(D(A), x.d @ y.d) == [{"A": x @ y}]
    assert not match(D(A), x.d @ y)


def test_adjoints():
    """ An adjoint pattern inverts to the adjoint on the other side, an
    exponential one matches the exponential of its own side only. """
    from discopy import biclosed

    r = rigid.Ty("r")
    assert [dict(s) for s in match(L(A), r.l)] == [{"A": r}]
    assert [dict(s) for s in match(R(A), r.r)] == [{"A": r}]
    x, y = biclosed.Ty("x"), biclosed.Ty("y")
    assert [dict(s) for s in match(Under(B, A), y >> x)] == [{"A": x, "B": y}]
    assert not match(Under(B, A), x << y)
    assert instantiate(Under(B, A), {"A": x, "B": y}, biclosed.Ty) == y >> x


def test_trace():
    """ A trace is a rule per side forwarding to the one method
    :meth:`trace`, of a memory partitioning the boundary as that of a
    feedback does. """
    from discopy import traced

    left, right = TracedCategory.trace_left, TracedCategory.trace_right
    assert left.variables["M"].size is None
    assert list(left.premises) == ["self", "dom", "cod", "mem"]
    x, y, a, b = map(traced.Ty, "xyab")
    goal = Hom(x @ y @ a, x @ y @ b)
    assert [s["M"] for s in match(left.premises["self"], goal)]\
        == [traced.Ty(), x, x @ y]
    assert [s["M"] for s in match(right.premises["self"], goal)]\
        == [traced.Ty()]  # The right ends a and b share no wire.
    f = traced.Box("f", goal.dom, goal.cod)
    assert f.trace_left(mem=x @ y) == f.trace(dom=a, cod=b, left=True)\
        == f.trace(dom=y @ a, cod=y @ b, left=True).trace(left=True)
    with raises(AxiomError):
        f.trace(mem=x, cod=b, left=True)
    canonical = traced.Diagram.trace_iteration_left.canonical()
    assert canonical and str(canonical.terms[0]).count("Trace") == 1


def test_curry_and_uncurry():
    """ A curry is a rule per side, concluding on the exponential of its
    whole exponent, which the uncurry evaluates back. """
    from discopy import biclosed

    x, y, z, w = map(biclosed.Ty, "xyzw")
    f = biclosed.Box("f", x @ y @ z, w)
    for curry, left in ((BiclosedCategory.curry_left, True),
                        (BiclosedCategory.curry_right, False)):
        assert list(curry.premises) == [
            "self", "context", "base", "exponent"]
        for subst in match(curry.premises["self"], Hom(f.dom, f.cod)):
            curried = f.curry(exponent=subst["Y"], left=left)
            assert instantiate(curry.conclusion, subst, biclosed.Ty)\
                == Hom(curried.dom, curried.cod, head=curry.conclusion.head)
            if subst["Y"]:
                uncurried = curried.uncurry(left=left)
                assert (uncurried.dom, uncurried.cod) == (f.dom, f.cod)


def test_ev_and_feedback():
    """ The evaluation and the feedback are a rule per side each, the
    memory of a feedback any object. """
    from discopy import biclosed

    x, y = biclosed.Ty("x"), biclosed.Ty("y")
    for ev, left in ((BiclosedCategory.ev_left, True),
                     (BiclosedCategory.ev_right, False)):
        assert list(ev.premises) == ["base", "exponent"]
        built = biclosed.Diagram.ev(y, x, left)
        assert instantiate(ev.conclusion, {"Y": y, "E": x}, biclosed.Ty)\
            == Hom(built.dom, built.cod, head=ev.conclusion.head)
    for feedback in (FeedbackCategory.feedback_left,
                     FeedbackCategory.feedback_right):
        assert feedback.variables["M"].size is None
        assert feedback.premises["mem"].name == "M"


def test_rule():
    assert repr(Rule(Category.cut.function)) == "Rule(cut)"
    assert repr(Category.cut) == "abc.Category.cut"
    assert Rule.inherited(cat.Arrow)["cut"].category is cat.Arrow
    assert hash(Category.cut) == hash(Category.cut.bind(Category))
    assert Category.then.__isabstractmethod__  # A method, not a rule.
    assert not isinstance(vars(Category)["then"], Rule)
    # The latest wins.
    assert Rule.inherited(Diagram)["cut"].owner is MonoidalCategory
    assert str(Rule.inherited(cat.Arrow)["cut"].conclusion).endswith(", A, C]")
    with raises(TypeError):
        Rule(Category.cut.function).bound
    with raises(TypeError):
        rule(classmethod(lambda cls: None))

    class Wrapped(Diagram):
        @rule
        def twice[A](self: abc.Hom[Diagram, A, A]
                     ) -> abc.Hom[Diagram, A, A]:
            """ A rule declared and implemented in one place. """
            return self >> self

    f = Box("f", x, x)
    assert Wrapped.twice(f) == f >> f == Wrapped(f.inside, x, x).twice()
    assert list(Rule.inherited(Wrapped))\
        == ["ax", "cut", "mix", "dagger", "twice"]
    assert str(Rule.inherited(Wrapped)["twice"].conclusion)\
        == "Hom[Diagram, A, A]"
    found = find(Wrapped.strategy(Hom(x, x)),
                 lambda value: len(value.boxes) == 2
                 and len(set(value.boxes)) == 1)
    assert found.boxes[0] == found.boxes[1]


def test_generator():
    assert Category.cut.recursive
    cups = Rule.inherited(rigid.Diagram)["cups"]
    assert not cups.recursive and cups.category is rigid.Diagram
    braid = Rule.inherited(braided.Diagram)["braid"]
    assert [str(sort) for sort in braid.variables.values()]\
        == ["Atom", "Atom"]
    assert list(braid.premises) == ["left", "right"]

    class Lying(Diagram):
        @classmethod
        @rule
        def wrong[A: ColouredMonoid](
                cls, dom: abc.Var[Ty, A]
        ) -> abc.Hom[Diagram, A, abc.Unit[Ty]]:
            """ A generator whose conclusion lies. """
            return cls.id(dom)

    with raises(AxiomError):
        find(Lying.strategy(Hom(x, Ty())), lambda value: False)


def test_cut():
    """ The search composes in context by cut, while the methods the
    rule derives from stay callable outside of it. """
    z = Ty("z")
    f, g = Box("f", y, y @ y), Box("g", x @ y @ y @ z, z)
    assert f.cut(g, x, z) == x @ f @ z >> g
    assert list(Rule.inherited(Diagram)["cut"].variables) == list("ABCXY")
    assert list(Rule.inherited(Diagram)) == ["ax", "cut", "mix", "dagger"]
    assert Rule.inherited(Diagram)["cut"].recursive
    assert list(Rule.inherited(cat.Arrow)["cut"].premises) == ["self", "other"]
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
        assert [name for name, found in Rule.inherited(module.Diagram).items()
                if found.recursive] == calculus, module.__name__

    assert "twist" not in Rule.inherited(compact.Diagram)
    twist = compact.Diagram.twist(compact.Ty("x"))
    assert twist.dom == twist.cod and not twist.inside


def test_declarations():
    class Hidden(Diagram):
        unitality = None
        cut = None

    assert "unitality" not in Hidden.axioms
    assert "cut" not in Rule.inherited(Hidden)
    assert list(Rule.inherited(Category)) == ["ax", "cut"]

