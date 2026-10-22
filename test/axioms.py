""" DisCoPy's property-testing module in action. """

from typing import Self, TypeVar

from hypothesis import find, settings
from hypothesis import strategies as st
from hypothesis.errors import NoSuchExample
from pytest import raises

from discopy import cat, feedback, rigid
from discopy.abc import Category, Monoid
from discopy.axioms import (
    L, R, Axiom, AxiomFailure, Count, Delay, Equation, Equivalence,
    Hom, Obj, Over, Related, Repeat, Tensor, Testable, Under, Unit, Var,
    assert_axioms, axiom, inhabits, is_open, position, substitute)
from discopy.cat import Arrow, Box, Functor, Ob
from discopy.monoidal import Ty


class ParityEquation(Equation):
    """ An equation between integers which holds up to their parity. """

    up_to = staticmethod(lambda n: n % 2)


class Parity(int, Equivalence):
    """ An integer up to parity, an equivalence relation besides equality. """

    equivalence_factory = ParityEquation
    serialisation = Testable.serialisation.inapplicable(
        "An integer has no tree.")

    @classmethod
    def strategy(cls, **params):
        """Generate an integer."""
        return st.integers().map(cls)

    @classmethod
    def related(cls, **params):
        """Generate an integer, another of the same parity and any third."""
        return st.tuples(st.integers(), st.integers(), st.integers()).map(
            lambda triple: (
                cls(triple[0]), cls(triple[0] + 2 * triple[1]),
                cls(triple[2])))


def test_axioms():
    assert_axioms(Arrow)

    class Classified(Arrow):
        """ A category with a broken law and an inapplicable one. """
        unitality = Arrow.unitality.failing("Never holds.")
        dagger_involution = Arrow.dagger_involution.inapplicable("No dagger.")

    assert_axioms(Classified)


def test_Equivalence():
    assert_axioms(Parity)
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


def test_failing_when_the_terms_do_not_build():
    class Refusing(Arrow):
        @axiom
        def refusing(cls, f: Self) -> Equation:
            raise ValueError("No terms.")
        refusing = refusing.failing("Declared broken.")

    with raises(AxiomFailure) as error:
        Refusing.refusing(f=Arrow.id(Ob('x')))
    assert isinstance(error.value.equation, ValueError)
    assert_axioms(Refusing)


def test_strategy():
    x, y = Ob('x'), Ob('y')
    find(Ob.strategy(), lambda ob: ob.name == "a")
    assert find(Arrow.strategy(dom=x, cod=x), lambda _: True) == Arrow.id(x)
    assert find(Arrow.strategy(dom=x, cod=y), lambda _: True).cod == y
    assert find(Arrow.strategy(dom=x), lambda _: True).dom == x
    assert find(Arrow.strategy(cod=y), lambda _: True).cod == y
    assert find(Box.strategy(dom=x), lambda _: True).dom == x


def test_sequents():
    @axiom
    def law[X: Obj[Ob], Y: Obj[Ob], N: Count](
            cls, x: Var[Ob, X], f: Hom[Self, X, Y], term: Self,
            n: Var[int, N]
    ) -> Equation[tuple[Var[Ob, X], bool, bool]]:
        """ Quantify over an object, a morphism out of it, a term and a
        number. """
        return Equation(
            (f.dom, isinstance(term, cls), n <= 3), (x, True, True))

    bound = law.bind(Arrow)
    scope, arguments = find(bound.sequents(), lambda _: True)
    assert set(arguments) == {"x", "f", "term", "n"} and bound(**arguments)
    assert inhabits(bound(**arguments), bound.output(scope))
    _, given = find(bound.sequents(given={"Y": Ob('y')}), lambda _: True)
    assert given["f"].cod == Ob('y')


def test_constructors():
    x, y = rigid.Ty('x'), rigid.Ty('y')
    assert Tensor[x, y] == x @ y and Unit[rigid.Ty] == rigid.Ty()
    assert (L[x], R[x]) == (x.l, x.r) and Repeat[x, 3] == x @ x @ x
    assert (Over[x, y], Under[y, x]) == (x << y, y >> x)
    z = feedback.Ty('z')
    assert Delay[feedback.Diagram, z] == z.delay()
    f = rigid.Box('f', x, y)
    assert inhabits(f, Hom[rigid.Diagram, x, y])
    assert not inhabits(f, Hom[rigid.Diagram, y, x])
    assert inhabits(Equation(f), Equation[Hom[rigid.Diagram, x, y]])
    assert not inhabits(Equation(f), Equation[Hom[rigid.Diagram, x, x]])
    assert inhabits(Equation(f), Equation) and not inhabits(f, Equation)
    assert inhabits((f, x), tuple[Hom[rigid.Diagram, x, y], Var[Ty, x]])
    assert inhabits((x, x), tuple[Var[Ty, x], ...])
    assert not inhabits(f, tuple[Hom[rigid.Diagram, x, y]])
    assert inhabits(Arrow, type[Category]) and inhabits(1, int)


def test_substitute():
    A, B = TypeVar("A"), TypeVar("B")
    x, y = Ty('x'), Ty('y')
    assert is_open(Tensor[A, B]) and not is_open(Tensor[x, y])
    assert substitute(Tensor[A, B], {A: x, B: y}) == x @ y
    assert substitute(int, {A: x}) is int
    assert substitute(Equation, {A: x}) is Equation
    assert substitute(Equation[Hom[Arrow, A, B]], {A: x, B: y}).__args__ \
        == (Hom[Arrow, x, y], )


def test_position():
    assert position(Monoid.__type_params__[0], Monoid) == 1
    assert position(Category.__type_params__[0], Category) == 0
    assert position(TypeVar("T"), Category) is None


def test_axiom_binding():
    assert repr(Axiom(lambda cls: NotImplemented)) == "Axiom(<lambda>)"
    assert eval(repr(Arrow.unitality)) == cat.Arrow.unitality
    assert hash(Arrow.unitality) == hash(eval(repr(Arrow.unitality)))
    assert Arrow.unitality != Functor.unitality
    with raises(TypeError):
        Axiom(lambda cls: NotImplemented)()
    with raises(TypeError):
        Axiom(lambda cls: NotImplemented).falsify()
    assert axiom(lambda cls: NotImplemented).bind(Arrow)() is NotImplemented
    box = Box('f', Ob('x'), Ob('y'))
    assert Arrow.unitality(f=box)
    broken = Arrow.unitality.weaken(max_leaves=1).failing("Never holds.")
    assert broken.params == {"max_leaves": 1}
    with raises(AxiomFailure) as failure:
        broken(f=box)
    assert failure.value.equation


def test_string_annotations():
    """ A law whose annotations are strings reads them as objects. """
    namespace = {}
    exec("from __future__ import annotations\n"
         "from discopy.axioms import Equation, Hom, Obj\n"
         "def deferred[A: Obj[int]](cls, f: Hom[int, A, A])"
         " -> Equation[Hom[int, A, A]]: return NotImplemented", namespace)
    law = axiom(namespace["deferred"]).bind(Arrow)
    (A, ), annotations = law.function.__type_params__, law.annotations()
    assert annotations["f"] == Hom[int, A, A]
    assert annotations["return"] == Equation[Hom[int, A, A]]


def test_inapplicable():
    law = Arrow.unitality.inapplicable("No identities to cancel.")
    assert law.name == "unitality"
    assert law.__doc__ == "No identities to cancel."
    assert law() is NotImplemented


def test_modulo():
    law = Arrow.unitality.modulo(lambda term: term.dom).bind(Arrow)
    assert law(f=Box('f', Ob('x'), Ob('y')))


def test_equation_symbols():
    x = Ob('x')
    f, g = Box('f', x, x), Box('g', x, x)
    assert Equation(f).symbols == ()
    assert Equation(f, g, symbol="<").symbols == ("<", )
    assert Equation(f, f, g).symbols == ("=", "=")
    with raises(ValueError):
        Equation(f, g, symbols=("=", "="))


def test_weaken():
    class Recording(Arrow):
        """ Arrows recording the parameters their strategy is given. """
        given = []

        @classmethod
        def strategy(cls, **params):
            cls.given.append(params.pop("tag", None))
            return Arrow.strategy.__func__(cls, **params)

    law = Recording.unitality.weaken(tag="loop")
    assert law.modulo(lambda term: term).params == law.params == {
        "tag": "loop"}
    assert law(**find(law.strategy(), lambda _: True))
    assert Recording.given and set(Recording.given) == {"loop"}


def test_self_annotation():
    @axiom
    def absorbing(cls, f: Self) -> Equation:
        """ The identity on the domain absorbs into any arrow. """
        return Equation(cls.id(f.dom) >> f, f)

    law = absorbing.bind(Arrow)
    arguments = find(law.strategy(), lambda _: True)
    assert isinstance(arguments["f"], Arrow) and law(**arguments)


def test_weaken_shape_of_self():
    from discopy.biclosed import TermBase, Ty as Type
    x = Type("x")
    for law in (TermBase.symmetry, TermBase.alpha_renaming,
                TermBase.alpha_completeness):
        with raises(NoSuchExample):
            find(law.weaken(cod=x).strategy(),
                 lambda arguments: any(
                     term.cod != x for term in arguments["terms"]),
                 settings=settings(max_examples=50, database=None))


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


def test_copying():
    x, y = Ob('x'), Ob('y')
    assert Arrow.copying(term=Box('f', x, y))
    assert "copying" in Arrow.axioms and "copying" in Ob.axioms
