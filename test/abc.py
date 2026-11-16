from pytest import raises

from discopy.abc import Nat
from discopy.symmetric import Diagram, Equation, Ty


def test_abc_permutation():
    from itertools import permutations
    from discopy.abc import SymmetricCategory

    abc_permutation = SymmetricCategory.permutation.__func__
    for n in range(1, 5):
        doms = [Ty(name) for name in "abcd"[:n]]
        for xs in permutations(range(n)):
            result = abc_permutation(Diagram, list(xs), doms)
            assert Equation(result, Diagram.permutation(list(xs), doms))
    with raises(ValueError):
        abc_permutation(Diagram, [0, 0], [Ty('a'), Ty('b')])


def test_Monoid_refuses_non_elements_on_either_side():
    from discopy import abc, group, monoidal

    for element in (abc.Nat(2), monoidal.Nat(2), monoidal.List[int](2),
                    group.Cyclic[3](1)):
        assert type(element)() @ element == element
        for other in ("x", 1, None, monoidal.Ty('y')):
            with raises(TypeError):
                other @ element
            with raises(TypeError):
                element @ other


def test_Nat_str_transparency():
    assert str(Nat(2)) == "Nat(2)"
    for n in range(4):
        assert eval(str(Nat(n)), {"Nat": Nat}) == Nat(n)


def test_LinearCategory_states_the_laws_of_a_linear_category():
    from discopy import biclosed, rigid
    from discopy.abc import LinearCategory
    from discopy.testable import Bang

    assert issubclass(biclosed.Diagram, LinearCategory)
    assert len(LinearCategory.axioms) - len(
        biclosed.BiclosedCategory.axioms) == 27
    x = biclosed.Ty('x')
    assert Bang[biclosed.Diagram, x] == x.bang()
    for law in set(LinearCategory.axioms) - set(
            biclosed.BiclosedCategory.axioms):
        assert getattr(biclosed.Diagram, law).broken
        assert getattr(rigid.Diagram, law).broken
    with raises(TypeError):
        type("Free", (LinearCategory, ), {})()
