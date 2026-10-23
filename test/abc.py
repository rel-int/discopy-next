from pytest import raises

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
