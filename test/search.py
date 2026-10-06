""" The rules and generators of a category, and the search by them. """


from typing import Annotated

from hypothesis import find
from hypothesis import strategies as st
from pytest import raises

from discopy import braided, cat, rigid
from discopy.abc import Category, ColouredMonoid
from discopy.monoidal import Box, Diagram, Ty
from discopy.pattern import Hom, Ob, UNIT
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
    assert cat.Arrow.rules["then"].sequent.conclusion\
        == Category.then.sequent.conclusion
    with raises(TypeError):
        Rule(Category.then.function).scope
    with raises(TypeError):
        rule(classmethod(lambda cls: None))

    class Wrapped(Diagram):
        @rule
        def twice[A](self: Annotated[Diagram, Hom(A, A)]
                     ) -> Annotated[Diagram, Hom(A, A)]:
            """ A rule declared and implemented in one place. """
            return self >> self

    f = Box("f", x, x)
    assert Wrapped.twice(f) == f >> f == Wrapped(f.inside, x, x).twice()
    assert list(Wrapped.rules) == ["id", "tensor", "cut", "twice"]
    assert str(Wrapped.rules["twice"])\
        == "twice: A: C0 | self: C1[A, A] ⊢ C1[A, A]"
    found = find(Wrapped.strategy(dom=x, cod=x, types=st.just(x)),
                 lambda value: len(value.boxes) == 2
                 and len(set(value.boxes)) == 1)
    assert found.boxes[0] == found.boxes[1]


def test_generator():
    assert Category.then.recursive
    assert not rigid.Diagram.rules["cups"].recursive
    assert "cups" in rigid.Diagram.generators
    assert rigid.Diagram.generators["cups"].category is rigid.Diagram
    assert str(braided.Diagram.generators["braid"]) == (
        "braid: X: Atom[C0], Y: Atom[C0] | left: X, right: Y"
        " ⊢ C1[X @ Y, Y @ X]")

    class Lying(Diagram):
        @classmethod
        @rule
        def wrong[A: ColouredMonoid](
                cls, dom: Annotated[Ty, Ob(A)]
        ) -> Annotated[Diagram, Hom(A, UNIT)]:
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
    assert str(Diagram.rules["cut"]) == (
        "cut: A: C0, B: C0, C: C0, X: C0, Y: C0 | self: C1[B, A], "
        "other: C1[X @ A @ Y, C], left: X, right: Y ⊢ C1[X @ B @ Y, C]")
    assert list(Diagram.rules) == ["id", "tensor", "cut"]
    assert Diagram.rules["cut"].recursive
    assert "then" in cat.Arrow.rules  # A mere category composes by then,
    assert f.then(Box("h", y @ y, z)).cod == z  # a monoidal one cuts.
    assert (f @ g).dom == f.dom @ g.dom


def test_declarations():
    class Hidden(Diagram):
        unitality = None
        cut = None

    assert "unitality" not in Hidden.axioms
    assert "cut" not in Hidden.rules
    assert list(Category.generators) == ["id"]

    assert list(Ty.rules) == ["id"]  # Composing objects is a method,
    # not a rule: Ty.then shadows the rule a mere category declares.


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
               dom=A, cod=Ob(A) @ Ob(A)),
        lambda term: bool(term.boxes)
        and all(isinstance(box, markov.Copy) for box in term.boxes))
    assert copy.cod == copy.dom @ copy.dom
