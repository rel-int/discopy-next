from itertools import permutations

import numpy as np
import pytest
from pytest import raises

from discopy import axioms
from discopy.abc import DaggerCategory, Nat
from discopy.matrix import Matrix, backend
from discopy.symmetric import Ty, Permutation, Swap, Functor
from discopy.tensor import Tensor, Dim
from discopy.utils import AxiomError, dumps, loads


def test_Matrix_trace_left():
    f = Matrix[bool]([[1, 1], [0, 0]], 2, 2)
    assert f.trace() == Matrix[bool]([[1]], 1, 1)
    assert f.trace(left=True) == Matrix[bool]([[0]], 1, 1)
    assert Matrix[bool].swap(1, 1).trace(left=True) == Matrix[bool].id(1)


def test_bad_composition():
    m = Matrix([1, 2, 3, 4, 5, 6], 2, 3)

    with raises(TypeError):
        m >> 1
    with raises(AxiomError):
        m >> m


def test_matrix_tensor():
    m = Matrix([1], 1, 1)
    assert (m.tensor(m, m).array == np.eye(3)).all()
    with raises(TypeError):
        m @ "bla"


def test_matrix_add():
    m = Matrix([1, 2, 3, 4, 5, 6], 2, 3)
    assert 0 + m == m
    with raises(TypeError):
        m + 123
    with raises(AxiomError):
        m + m.dagger()
    assert isinstance(m, DaggerCategory)


def test_repeat():
    with raises(TypeError):
        Matrix[int](0, 1, 1, 0).repeat()


def test_autotyping():
    pytest.importorskip("jax")
    torch = pytest.importorskip("torch")
    assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == np.float64
    assert Matrix([0.5j], dom=1, cod=1).dtype == np.complex128
    with backend('jax'):
        assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == np.float32
    with backend('pytorch'):
        assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == torch.float32



def test_permutation():
    """ The native permutation agrees with a composition of swaps. """
    def swaps(xs, doms):
        result, done, cur = Matrix.id(sum(doms)), 0, list(doms)
        xs = list(xs)
        while xs != list(range(len(xs))):
            i = xs[0]
            left = sum(cur[:i])
            result >>= Matrix.id(done) @ Matrix.swap(left, cur[i])\
                @ Matrix.id(sum(cur[i + 1:]))
            done, cur = done + cur[i], cur[:i] + cur[i + 1:]
            xs = [x - 1 if x > i else x for x in xs[1:]]
        return result

    assert Matrix.permutation([1, 0], [1, 1]) == Matrix.swap(1, 1)
    for xs in permutations(range(4)):
        doms = [1 + x % 3 for x in xs]
        assert Matrix.permutation(xs, doms) == swaps(xs, doms)
    with raises(ValueError):
        Matrix.permutation([0, 0], [1, 1])


def test_copy_defaults_to_two():
    """ Matrix is a MarkovCategory, whose copy has n=2 by default. """
    assert Matrix.copy(2) == Matrix.copy(2, 2)
    assert Tensor.copy(Dim(2)) == Tensor.copy(Dim(2), 2)


def test_permutation_doms_optional():
    """ One dimension per block is the default. """
    for size in range(1, 6):
        for xs in permutations(range(size)):
            assert Matrix.permutation(xs) == Matrix.permutation(xs, size * [1])
    assert Matrix.permutation([1, 0]) == Matrix.swap(1, 1)
    assert Matrix.permutation([0, 1]) == Matrix.id(2)
    with raises(ValueError):
        Matrix.permutation([0, 2])
    with raises(ValueError):
        Matrix.permutation([0], [-1])


def test_permutation_on_immutable_backend():
    """ A block permutation is built without writing into an array. """
    pytest.importorskip("jax")
    with backend("jax"):
        swap = Matrix.permutation([1, 0], [1, 2])
    assert np.array_equal(
        np.asarray(swap.array), [[0, 0, 1], [1, 0, 0], [0, 1, 0]])


def test_matrix_valued_functor_on_permutation():
    """ Matrix is a MarkovCategory, so it is a valid symmetric codomain. """
    x, y = Ty('x'), Ty('y')
    perm = Permutation(x @ y @ x, [1, 2, 0])
    F = Functor(ob_map={x: 2, y: 3}, ar_map={}, cod=Matrix)
    assert F(perm) == Matrix.permutation([1, 2, 0], [2, 3, 2])


def test_swap_on_immutable_backend():
    """ A swap is a block permutation, built without writing into an array. """
    pytest.importorskip("jax")
    with backend("jax"):
        swap = Matrix.swap(1, 2)
        assert swap == Matrix.permutation([1, 0], [1, 2])
    assert np.array_equal(
        np.asarray(swap.array), [[0, 0, 1], [1, 0, 0], [0, 1, 0]])


def test_composition_promotes_dtype():
    """ The integer structural matrices compose with floats as units. """
    m = Matrix([[.5, .5], [.25, .75]], 2, 2)
    assert Matrix.id(2) >> m == m == m >> Matrix.id(2)
    assert Matrix.zero(2, 2) + m == m
    assert (Matrix.id(1) @ m).array.tolist()\
        == [[1, 0, 0], [0, .5, .5], [0, .25, .75]]
    c = Matrix([1j], 1, 1)
    assert Matrix.id(1) >> c == c == c >> Matrix.id(1)
    assert Matrix[float].id(2) >> Matrix.id(2) == Matrix[float].id(2)
    one_plus_one = Matrix[bool].copy(1) >> Matrix[bool].merge(1, 2)
    assert one_plus_one == Matrix[bool].id(1)


@pytest.mark.parametrize('name', ['pytorch', 'jax'])
def test_tensor_of_many_on_backend(name):
    """ The direct sum of three matrices is built on a backend that takes
    the dtypes of two arrays at a time, and on one whose arrays are
    immutable. """
    pytest.importorskip({'pytorch': 'torch'}.get(name, name))
    with backend(name):
        one, two = Matrix([1.], 1, 1), Matrix([2., 2.], 1, 2)
        result = one.tensor(two, one)
    assert np.array_equal(np.asarray(result.array), [
        [1, 0, 0, 0], [0, 2, 2, 0], [0, 0, 0, 1]])


def test_tensor_of_many_promotes_dtype():
    """ The tensor of ``n`` matrices is built in the class the ``dtype`` of
    all of them promotes to, wherever the float one comes. """
    m, v = Matrix([.5], 1, 1), Tensor([.5, .5], Dim(1), Dim(2))
    i, j = Matrix.id(1), Tensor.id(Dim(2))
    assert i.tensor(i, m) == i @ i @ m and i.tensor(i, m).array[2, 2] == .5
    assert j.tensor(v, j) == j @ v @ j and type(j.tensor(v, j)) is type(v)


def test_Nat_strategy():
    """ The objects of a matrix generate, as the search asks of Matrix.ob. """
    from hypothesis import find
    assert find(Nat.strategy(), lambda n: n == Nat(2)) == Nat(2)


def test_Nat_serialisation():
    assert loads(dumps(Nat(2))) == Nat(2)



def test_Matrix_strategy():
    """ The bare class generates Matrix[int], a subscript its own members. """
    from hypothesis import find
    matrix = find(Matrix.strategy(dom=2, cod=3), lambda m: m.array.any())
    assert type(matrix) is Matrix[int]
    assert (matrix.dom, matrix.cod) == (Nat(2), Nat(3))
    assert type(find(Matrix[bool].strategy(), lambda _: True)) is Matrix[bool]


def test_Matrix_axioms():
    """ A dry run of the property matrix on the carrier it enrols. """
    axioms.assert_axioms(Matrix[int])


def test_Matrix_copy():
    """ The copy of the direct sum is the identities side by side. """
    assert Matrix.copy(3, 2) == Matrix(
        [[1, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 1]], 3, 6)
    for x in range(4):
        for n in range(4):
            copy = Matrix.copy(x, n)
            assert (copy.dom, copy.cod) == (Nat(x), Nat(n * x))
            assert copy.dtype == Matrix.id(x).dtype
            assert copy.array.tolist() == [
                [int(j % x == i) for j in range(n * x)] for i in range(x)]
    assert Matrix.discard(2).array.shape == (2, 0)
    assert Matrix[bool].copy(2).dtype == bool
    assert Matrix.permutation([], []).dtype == Matrix.id(0).dtype


def test_Matrix_comonoid():
    """ Copy and discard are a commutative comonoid under the direct sum. """
    for x in map(Nat, range(4)):
        assert Matrix.copy_coassociativity(x)
        assert Matrix.copy_cocommutativity(x)
        assert Matrix.copy_counitality(x)
        assert Matrix.copy_coherence(x, Nat(2))
        assert Matrix.ones(x) @ Matrix.id(x) >> Matrix.merge(x)\
            == Matrix.id(x)


def test_Matrix_monoid():
    """ Merge and noise are a commutative monoid under the direct sum. """
    for x in map(Nat, range(4)):
        assert Matrix.merge_associativity(x)
        assert Matrix.merge_commutativity(x)
        assert Matrix.merge_unitality(x)
        assert Matrix.merge_coherence(x, Nat(2))
        assert Matrix.noise(x) == Matrix.ones(x) == Matrix.discard(x).dagger()
        assert Matrix.noise(x @ Nat(2)) == Matrix.noise(x) @ Matrix.noise(2)
    assert Matrix.merge(2) == Matrix([[1, 0], [0, 1], [1, 0], [0, 1]], 4, 2)
    assert {"merge_associativity", "merge_unitality"} <= Matrix[int].axioms.keys()


def test_Matrix_ar():
    """ A subscript of Matrix is its own factory, where functors build swaps. """
    assert Matrix[complex].ar is Matrix[complex]
    x, y = Ty('x'), Ty('y')
    F = Functor(ob_map={x: 2, y: 3}, ar_map={}, cod=Matrix[complex])
    assert F(Swap(x, y)).dtype == complex
    assert F(Permutation(x @ y @ x, [1, 2, 0])).dtype == complex


def test_Matrix_copy_dtype():
    for x in range(4):
        dtype = Matrix.id(x).dtype
        assert Matrix.copy(x, 2).dtype == dtype
        assert Matrix.discard(x).dtype == Matrix.ones(x).dtype == dtype
    for dtype in (bool, int, float, complex):
        assert Matrix[dtype].copy(2, 2).dtype == dtype

def test_trace_takes_the_keyword_form():
    f = Matrix[bool]([[1, 1], [0, 0]], 2, 2)
    assert f.trace() == f.trace(mem=1) == f.trace(cod=1)\
        == f.trace(dom=1, cod=1, mem=1) != f.trace(left=True)
    assert f.trace(left=True) == f.trace(mem=Nat(1), left=True)
    with raises(TypeError):
        f.trace(1, mem=1)

def test_Matrix_to_tree():
    # A matrix had no tree at all, see #781.
    for m in (Matrix[float]([1], 1, 1), Matrix([0, 1, 1, 0], 2, 2),
              Matrix[bool]([1, 0], 1, 2), Matrix[int].zero(2, 3)):
        assert loads(dumps(m)) == m
    # The dtype is part of the matrix, so it has to travel with it.
    assert Matrix[float]([1], 1, 1) != Matrix([1], 1, 1)
    assert loads(dumps(Matrix[float]([1], 1, 1))).dtype == float
