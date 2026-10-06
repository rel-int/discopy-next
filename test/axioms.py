""" DisCoPy's property-testing module in action. """

from typing import Annotated, Any, Self

from hypothesis import find
from hypothesis.errors import NoSuchExample
from pytest import raises

from discopy import braided, cat, feedback, rigid
from discopy.abc import MonoidalCategory
from discopy.axioms import (
    Axiom, AxiomFailure, Equation, Hom, assert_axioms, axiom)
from discopy.cat import Arrow, Box, Functor, Ob
from discopy.monoidal import Diagram


def test_axioms():
    assert_axioms(Arrow)

    class Classified(Arrow):
        """ A category with a broken law and an inapplicable one. """
        unitality = Arrow.unitality.failing("Never holds.")
        dagger_involution = Arrow.dagger_involution.inapplicable("No dagger.")

    assert_axioms(Classified)


def test_axiom_binding():
    assert repr(Axiom(lambda cls: NotImplemented)) == "Axiom(<lambda>)"
    assert eval(repr(Arrow.unitality)) == cat.Arrow.unitality
    assert hash(Arrow.unitality) == hash(eval(repr(Arrow.unitality)))
    assert Arrow.unitality != Functor.unitality
    with raises(TypeError):
        Axiom(lambda cls: NotImplemented)()
    with raises(TypeError):
        Axiom(lambda cls: NotImplemented).falsify()
    with raises(TypeError):
        Axiom(lambda cls: NotImplemented).strategy()
    assert axiom(lambda cls: NotImplemented).bind(Arrow)() is NotImplemented
    box = Box('f', Ob('x'), Ob('y'))
    assert Arrow.unitality(box)
    broken = Arrow.unitality.weaken(max_leaves=1).failing("Never holds.")
    assert broken.params == {"max_leaves": 1}
    with raises(AxiomFailure) as failure:
        broken(box)
    assert failure.value.equation


def test_inapplicable():
    law = Arrow.unitality.inapplicable("No identities to cancel.")
    assert law.name == "unitality"
    assert law.__doc__ == "No identities to cancel."
    assert law() is NotImplemented


def test_modulo():
    law = Arrow.unitality.modulo(lambda term: term.dom).bind(Arrow)
    assert law(Box('f', Ob('x'), Ob('y')))


def test_weaken():
    law = Arrow.unitality.weaken(max_leaves=1).bind(Arrow)
    assert law.modulo(lambda term: term).params == law.params
    equation = find(law.strategy(), lambda _: True)
    assert equation and all(len(term.inside) <= 1 for term in equation.terms)


def test_self_annotation():
    @axiom
    def absorbing(cls, f: Self) -> Equation:
        """ The identity on the domain absorbs into any arrow. """
        return Equation(cls.id(f.dom) >> f, f)

    law = absorbing.bind(Arrow)
    equation = find(law.strategy(), lambda _: True)
    assert isinstance(equation.terms[1], Arrow) and equation


def test_falsify():
    @axiom
    def trivial[A, B](cls, f: Annotated[Any, Hom(A, B)]) -> Equation:
        """ Every arrow is an identity, which a box refutes. """
        return Equation(f, cls.id(f.dom))

    equation = trivial.bind(Arrow).falsify()
    assert not equation and equation.terms[0].inside
    for law in (Arrow.associativity, Arrow.unitality.failing("Declared.")):
        with raises(NoSuchExample):
            law.falsify()


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


def test_axiom():
    assert MonoidalCategory.bifunctoriality.parameters[0].name == "f"
    assert str(MonoidalCategory.bifunctoriality.sequent).startswith(
        "A: C0, B: C0, C: C0, D: C0, U: C0, V: C0 | f: C1[A, B]")
    assert MonoidalCategory.tensor.sequent.conclusion is not None
    assert Axiom.concludes is False
    @axiom
    def unannotated(cls, f):
        """ An unannotated premise has no pattern. """
    with raises(TypeError, match="states no pattern"):
        unannotated.bind(Diagram).sequent


def test_equation_types():
    """ The Equation subscript of an axiom types its canonical equation. """
    import inspect
    from discopy import (
        balanced, closed, compact, pivotal, ribbon, symmetric, traced)
    from discopy.pattern import cell, expand
    (parameter, ) = Equation.__type_params__
    levels = (Diagram, braided.Diagram, traced.Diagram, balanced.Diagram,
              symmetric.Diagram, closed.Diagram, rigid.Diagram,
              pivotal.Diagram, ribbon.Diagram, compact.Diagram,
              feedback.Diagram)
    checked = set()
    for category in levels:
        for law in category.axioms.values():
            returns = inspect.signature(law.function).return_annotation
            pattern = expand(getattr(returns, parameter.__name__, None))
            if pattern is None\
                    or (equation := law.canonical()) is NotImplemented:
                continue
            subst = {
                name: 2 if sort.head == "Count"
                else cell(sort.resolve(law.scope), name)
                for name, sort in law.sequent.variables.items()}
            value = pattern.instantiate(subst, law.unit)
            boundary = (lambda term: (term.dom, term.cod))\
                if isinstance(pattern, Hom) else (lambda term: term)
            assert all(
                boundary(term) == value for term in equation.terms), law
            checked.add(law.name)
    assert len(checked) > 30


def test_canonical():
    equation = Diagram.bifunctoriality.canonical()
    assert equation and str(equation.terms[0])\
        == "f @ C >> B @ g >> h @ D >> U @ k"
    assert str(Arrow.associativity.canonical())\
        == "Equation(f >> g >> h, f >> g >> h)"
    assert str(Diagram.tensor_unitality.canonical())\
        == "Equation(Id(X @ Y), Id(X @ Y))"
    assert Arrow.unitality.failing("Declared.").canonical()
    assert not braided.Diagram.braid_naturality.canonical()
    inapplicable = Arrow.unitality.inapplicable("No identities.")
    assert inapplicable.canonical() is NotImplemented
    with raises(TypeError, match="nothing to draw"):
        inapplicable.draw()
    assert str(rigid.Diagram.snake_equations.canonical().terms[1]) == "Id(X)"
    cups = rigid.Diagram.generators["cups"].canonical()
    assert cups == {"left": rigid.Ty('X'), "right": rigid.Ty('X').r}
    assert feedback.Diagram.feedback_joining.canonical()
