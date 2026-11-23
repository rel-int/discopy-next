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
    from discopy import bang
    from discopy.abc import LinearCategory, MonoidalCategory
    from discopy.testable import Bang

    assert issubclass(bang.Diagram, LinearCategory)
    laws = set(LinearCategory.axioms) - set(MonoidalCategory.axioms)
    assert len(laws) == 28
    x = bang.Ty('x')
    assert Bang[bang.Diagram, x] == x.bang()
    for law in laws:
        assert getattr(bang.Diagram, law).broken
    with raises(TypeError):
        type("Free", (LinearCategory, ), {})()


def test_BraidedCategory_braid_inverse():
    from discopy import braided
    from discopy.abc import BraidedCategory

    x, y = braided.Ty('x'), braided.Ty('y')
    braid = braided.Diagram.braid(x, y)
    assert braided.Diagram.braid_inverse(x, y) == braid.dagger()

    class Inverted(braided.Diagram):
        @classmethod
        def braid_inverse(cls, left, right):
            return cls.braid(right, left)

    other = Inverted.braid(y, x)
    assert BraidedCategory.inverse_braiding.function(Inverted, x, y) == other
    for law, terms in {
            "braid_then_inverse": (braid >> other, braid.id(x @ y)),
            "inverse_then_braid": (other >> braid, braid.id(y @ x))}.items():
        equation = getattr(BraidedCategory, law).function(Inverted, x, y)
        assert equation.terms == terms


def test_SymmetricCategory_braid_inverse_needs_no_dagger():
    from discopy.python.multiplicative import Function

    inverse = Function.braid_inverse(int, str)
    assert not hasattr(Function, "dagger")
    assert inverse.dom == Function.ob(str, int)
    assert inverse.cod == Function.ob(int, str)
    assert inverse("a", 1) == (1, "a")


def test_dagger_monoidality_needs_a_dagger_and_a_tensor():
    from discopy import cat, group, matrix, monoidal
    from discopy.abc import (
        DaggerCategory, DaggerMonoidalCategory, Groupoid, MonoidalCategory)
    from discopy.python import finset

    for cls in (DaggerCategory, Groupoid, cat.Arrow, group.Cyclic[3]):
        assert "dagger_monoidality" not in cls.axioms
    for cls in (monoidal.Diagram, matrix.Matrix[int], finset.Permutation):
        assert issubclass(cls, DaggerMonoidalCategory)
        assert "dagger_monoidality" in cls.axioms
    assert not issubclass(MonoidalCategory, DaggerCategory)


def test_DaggerMonoidalCategory_keeps_its_type_parameters():
    from discopy.abc import DaggerMonoidalCategory
    from discopy.testable import position

    assert [position(param, DaggerMonoidalCategory)
            for param in DaggerMonoidalCategory.__type_params__] == [0, 1]
