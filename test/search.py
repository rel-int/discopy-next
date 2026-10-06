""" The rules and generators of a category, and the search by them. """


from hypothesis import find
from hypothesis import strategies as st
from pytest import raises

from discopy import braided, cat, rigid
from discopy.abc import Category, ColouredMonoid
from discopy.monoidal import Box, Diagram, Ty
from discopy.pattern import Hom, Obj, Tensor, Unit
from discopy.search import Rule, rule, search
from discopy.utils import AxiomError


x, y = Ty("x"), Ty("y")


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
    assert list(Wrapped.rules) == ["id", "tensor", "cut", "twice"]
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
                cls, dom: Obj[Ty, A]) -> Hom[Diagram, A, Unit[Ty]]:
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
    assert list(Diagram.rules) == ["id", "tensor", "cut"]
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

    composition, trace = ["tensor", "cut"], ["trace"]
    for module, calculus in (
            (monoidal, composition),
            (braided, composition),
            (rigid, composition),
            (balanced, composition + trace),
            (symmetric, composition + trace),
            (traced, composition + trace),
            (markov, composition + trace),
            (pivotal, composition + trace),
            (ribbon, composition + trace),
            (compact, composition + trace),
            (frobenius, composition + trace),
            (biclosed, composition + ["curry"]),
            (closed, composition + trace + ["curry"]),
            (feedback, composition + trace
             + ["feedback"])):
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
    from discopy.search import focused

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
