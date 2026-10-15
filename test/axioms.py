""" DisCoPy's property-testing module in action. """

from typing import Self

from hypothesis import find
from hypothesis.errors import NoSuchExample
from pytest import raises

from discopy import braided, cat, monoidal, rigid
from discopy.abc import Category, ColouredMonoid, MonoidalCategory
from hypothesis import strategies as st

from discopy.axioms import (
    L, R, Atom, Axiom, AxiomFailure, Count, Equation, Equivalence, Hom, Obj,
    Over, Related, Repeat, Tensor, Testable, Under, Unit, Var,
    axiom, inhabits)
from discopy.monoidal import Ty
from discopy.cat import Arrow, Box, Ob


class ParityEquation(Equation):
    """ An equation between integers which holds up to their parity. """

    up_to = staticmethod(lambda n: n % 2)


class Parity(int, Equivalence):
    """ An integer up to parity, an equivalence relation besides equality. """

    equivalence_factory = ParityEquation

    @classmethod
    def strategy(cls, **params):
        """ Generate an integer, no strategy being enrolled from a test. """
        return st.integers().map(cls)

    @classmethod
    def related(cls, **params):
        """ Generate an integer, another of the same parity and any third. """
        return st.tuples(st.integers(), st.integers(), st.integers()).map(
            lambda triple: (
                cls(triple[0]), cls(triple[0] + 2 * triple[1]),
                cls(triple[2])))


def test_Equivalence():
    assert Parity.equivalent(Parity(1), Parity(3), Parity(-5))
    assert not Parity.equivalent(Parity(1), Parity(2))
    assert Parity.reflexivity(Parity(2))
    assert Parity.symmetry(Related[Parity]((1, 3, 4)))
    assert Parity.transitivity(Related[Parity]((1, 3, 5)))
    assert Parity.transitivity(Related[Parity]((1, 3, 4)))
    assert not Parity.transitivity(Related[Parity]((1, 2, 3)))
    assert find(Related[Parity].strategy(), lambda t: t[2] % 2 == 1)[2] % 2
    with raises(ValueError):
        Related[Parity]((1, 3))
    arguments = find(
        Parity.transitivity.arguments(), lambda arguments: True)
    assert isinstance(arguments["terms"], Related[Parity])


def test_failing_when_the_terms_do_not_build():
    class Refusing(Arrow):
        @axiom
        def refusing(cls, f: Self) -> Equation:
            raise ValueError("No terms.")
        refusing = refusing.failing("Declared broken.")

    with raises(AxiomFailure) as error:
        Refusing.refusing(Arrow.id(Ob('x')))
    assert isinstance(error.value.equation, ValueError)


def test_strategy():
    x, y = Ob('x'), Ob('y')
    find(Ob.strategy(), lambda ob: ob.name == "a")
    arrow = find(Arrow.search(dom=x, cod=y), lambda f: len(f) > 1)
    assert (arrow.dom, arrow.cod) == (x, y)
    identity = find(
        Arrow.search(fuel=0, within=Category), lambda f: not f.inside)
    assert identity == Arrow.id(identity.dom)
    assert find(Box.strategy(dom=x), lambda _: True).dom == x


def test_equation():
    f = Box('f', Ob('x'), Ob('x'))
    assert Equation(f, f, f).symbols == ("=", "=")
    with raises(ValueError):
        Equation(f, f, symbols=("=", "="))


def test_axiom_binding():
    assert repr(Axiom(lambda cls: NotImplemented)) == "Axiom(<lambda>)"
    assert eval(repr(Arrow.unitality)) == cat.Arrow.unitality
    assert hash(Arrow.unitality) == hash(eval(repr(Arrow.unitality)))
    for method in ("__call__", "falsify"):
        with raises(TypeError):
            getattr(Axiom(lambda cls: NotImplemented), method)()
    assert axiom(lambda cls: NotImplemented).bind(Arrow)() is NotImplemented
    assert Arrow.unitality(f=Box('f', Ob('x'), Ob('y')))


def test_string_annotations():
    """ A law whose annotations are strings is refused. """
    namespace = {}
    exec("from __future__ import annotations\n"
         "def deferred(cls, f: int): return NotImplemented", namespace)
    with raises(TypeError, match="__future__"):
        axiom(namespace["deferred"])


def test_sequents():
    @axiom
    def law[X: Atom[Ty], Y: Obj[Ty], N: Count](
            cls, x: Var[Ty, X], f: Hom[Self, X, Y], term: Self,
            n: Var[int, N]
    ) -> Equation[tuple[int, Var[Ty, X], bool, bool]]:
        """ Quantify over an atom, a morphism out of it, a term and a
        number. """
        return Equation(
            (len(x), f.dom, isinstance(term, cls), n <= 3),
            (1, x, True, True))

    bound = law.bind(monoidal.Diagram)
    scope, arguments = find(bound.sequents(), lambda _: True)
    assert set(arguments) == {"x", "f", "term", "n"} and bound(**arguments)
    assert inhabits(bound(**arguments), bound.output(scope))
    assert bound.canonical()


def test_formers():
    x, y = rigid.Ty('x'), rigid.Ty('y')
    assert Tensor[x, y] == x @ y and Unit[rigid.Ty] == rigid.Ty()
    assert (L[x], R[x]) == (x.l, x.r) and Repeat[x, 3] == x @ x @ x
    assert (Over[x, y], Under[y, x]) == (x << y, y >> x)
    f = rigid.Box('f', x, y)
    assert inhabits(f, Hom[rigid.Diagram, x, y])
    assert not inhabits(f, Hom[rigid.Diagram, y, x])
    assert not inhabits(Equation(f), Equation[Hom[rigid.Diagram, x, x]])


def test_classifiers():
    f = Box('f', Ob('x'), Ob('y'))
    broken = Arrow.unitality.failing("Never holds.")
    with raises(AxiomFailure) as failure:
        broken(f=f)
    assert broken.broken and failure.value.equation
    law = Arrow.unitality.inapplicable("No identities to cancel.")
    assert law() is NotImplemented
    assert law.__doc__ == "No identities to cancel."
    assert Arrow.unitality.modulo(lambda term: term.dom)(f=f)


def test_rules():
    x, y = Ob('x'), Ob('y')
    f = Box('f', x, y)
    assert sorted(Arrow.rules) == ["ax", "cut"]
    assert Arrow.cut(f, f[::-1]) == f >> f[::-1] and Arrow.ax(x) == Arrow.id(x)
    assert Arrow.cut.is_recursive and not Arrow.ax.is_recursive
    assert monoidal.Diagram.mix.conclusion == (None, None)
    assert Arrow.ax.given(x, y) == {"X": y}
    assert braided.Diagram.braiding.given(braided.Ty('x', 'y')) == {}


def test_weaken():
    law = rigid.Diagram.dagger_involution_monoidal
    assert law.params == {"within": MonoidalCategory}
    args = find(law.strategy(), lambda _: True)
    assert not any(
        isinstance(box, (rigid.Cup, rigid.Cap)) for box in args["f"].boxes)


def test_falsify():
    @axiom
    def trivial[A: Obj[Ob], B: Obj[Ob]](
            cls, f: Hom[Self, A, B]) -> Equation[Self]:
        """ Every arrow is an identity, which a box refutes. """
        return Equation(f, cls.id(f.dom))

    assert trivial.bind(Arrow).falsify()["f"].inside
    assert Arrow.unitality.failing("Never holds.").falsify()
    with raises(NoSuchExample):
        Arrow.associativity.falsify()


def test_canonical():
    assert str(Arrow.associativity.canonical()) \
        == "Equation(f >> g >> h, f >> g >> h)"
    equation = rigid.Diagram.dagger_involution.canonical()
    assert equation.terms[1] == rigid.Box('f', rigid.Ty('A'), rigid.Ty('B'))


def test_axioms_of_category():
    class Broken(Arrow):
        """ A category declaring an inherited law broken. """
        unitality = Arrow.unitality.failing("Never holds.")

    assert Broken.axioms["unitality"].broken
    assert not Arrow.axioms["unitality"].broken

    class Hidden(Arrow):
        """ Assigning a non-axiom over an inherited law drops it. """
        unitality = None

    assert "unitality" not in Hidden.axioms


def test_canonical_unnamed_objects():
    from discopy.neural.network import Network
    with raises(TypeError, match="no canonical instance"):
        Network.associativity.draw()


def test_matrix():
    assert monoidal.Diagram in Testable.enrolled()
    assert monoidal.Ty in Testable.enrolled()
    assert cat.FreeCategory not in Testable.enrolled()
    once = Testable.matrix(once=True)
    assert rigid.Diagram.dagger_involution in once
    assert cat.Arrow.dagger_involution in once
    assert rigid.Diagram.unitality not in once
    assert set(once) <= set(Testable.matrix())


def test_canonical_instances():
    """ Every law of a free category holds on its canonical instance unless
    declared broken, types included, so that a law that crashes on its own
    statement fails here before the property matrix ever searches for a
    counterexample. """
    for category in Testable.enrolled():
        box = getattr(category, "box_factory", None)
        if box is None or not (issubclass(box, category)
                               or issubclass(category, ColouredMonoid)):
            continue  # a map, a hypergraph, a type of one colour or a test
        try:
            category.ob("A")
        except TypeError:
            continue  # objects that take no name, e.g. the shapes of networks
        for law in category.axioms.values():
            if law.broken:
                continue
            equation = law.canonical()
            assert equation is NotImplemented or equation, (category, law.name)
