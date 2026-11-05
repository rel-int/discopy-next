from pytest import mark, raises

from discopy import abc, monoidal
from discopy.abc import Category, Groupoid, Nat
from discopy.drawing.bracketing import (
    Bracketed, equation, free_category, stated_axioms)
from discopy.testable import Axiom, Equation, Hom, Obj, Repeat, Testable

STATED = [
    axiom for cls in vars(abc).values()
    if isinstance(cls, type) and cls.__module__ == abc.__name__
    for axiom in stated_axioms(cls).values()]


@mark.parametrize("axiom", STATED, ids=map(repr, STATED))
def test_every_axiom_draws(axiom, tmp_path):
    if free_category(axiom) is None:
        with raises(ValueError):
            equation(axiom)
        return
    result = equation(axiom)
    assert len(result.terms) >= 2
    result.draw(path=str(tmp_path / f"{axiom.name}.svg"))


def test_brackets_tell_the_sides_apart():
    lhs, rhs = equation(Category.associativity).terms
    assert lhs.value == rhs.value
    assert lhs.to_drawing() != rhs.to_drawing()
    assert lhs.drawing.bracketing and not lhs.to_drawing().bracketing


def test_no_free_groupoid():
    assert free_category(Groupoid.invertibility) is None


def test_generic_needs_names():
    with raises(TypeError):
        Testable.transparency.bind(monoidal.Diagram).generic()


def test_generic_number():
    def repeated[A: Obj[monoidal.Ty], N: Nat](
            cls, f: Hom[monoidal.Diagram, Repeat[A, N], A]) -> Equation:
        return Equation(f, f)
    scope, arguments = Axiom(repeated).bind(monoidal.Diagram).generic()
    x = monoidal.Ty('A')
    assert arguments['f'].dom == x @ x
    assert Nat(2) in scope.values()


def test_bracket_tikz(tmp_path):
    x = monoidal.Ty('x')
    f = monoidal.Box('f', x, x).to_drawing()
    path = tmp_path / "bracket.tikz"
    (f >> f).bracket("b").draw(to_tikz=True, path=str(path))
    assert "dotted" in path.read_text()


def test_dagger_swaps_the_boundary():
    x, y = monoidal.Ty('x'), monoidal.Ty('y')
    f = Bracketed[monoidal.Diagram](monoidal.Box('f', x, y))
    assert (f.dagger().dom, f.dagger().cod) == (y, x)
    assert (f.dagger().drawing.dom, f.dagger().drawing.cod) == (y, x)
    assert f.dagger().dagger().value == f.value


def test_trace_operand_is_bracketed():
    from discopy import symmetric
    x = symmetric.Ty('x')
    f = Bracketed[symmetric.Diagram](symmetric.Box('f', x @ x, x @ x))
    traced = (f >> f).trace()
    assert traced.drawing.bracketing
    assert any(box.draw_as_bracket for box in traced.to_drawing().boxes)
    assert (traced.dom, traced.cod) == (x, x)

