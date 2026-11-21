from itertools import permutations

import numpy as np
import pytest
from pytest import raises

from discopy.abc import DaggerCategory, Nat, TracedCategory
from discopy.matrix import (
    Matrix, asarray, backend, is_complex, is_field)
from discopy.symmetric import Ty, Permutation, Swap, Functor
from discopy.tensor import Tensor, Dim
from discopy.utils import AxiomError, Subscript, dumps, loads


def test_Matrix_order():
    R, S = Matrix[bool]([1, 0, 0, 0], 2, 2), Matrix[bool].id(2)
    assert R <= S and S >= R and R <= R and not S <= R
    assert R >> R <= S >> S and R @ R <= S @ S
    assert R.transpose() <= S.transpose()
    assert not R <= Matrix[bool]([1, 1], 1, 2)
    assert Matrix([1, 1], 1, 2) <= Matrix([2, 1], 1, 2)
    assert not Matrix([1, 2], 1, 2) <= Matrix([2, 1], 1, 2)
    assert Matrix([.5], 1, 1) <= Matrix([1.5], 1, 1)
    full = Tensor[bool]([1, 1, 1, 1], Dim(2), Dim(2))
    assert Tensor[bool].id(Dim(2)) <= full and not full <= full.id(Dim(2))
    with raises(TypeError):
        R <= 1
    with raises(TypeError):
        R <= R.cast(int)
    with raises(TypeError):
        R < S
    for dtype in (complex, np.complex64):
        with raises(TypeError):
            Matrix[dtype]([1j], 1, 1) <= Matrix[dtype]([2j], 1, 1)
    assert is_complex(complex) and is_complex(np.dtype("complex64"))
    assert not is_complex(float) and not is_complex(bool)


def test_Matrix_trace_left():
    f = Matrix[bool]([[1, 1], [0, 0]], 2, 2)
    assert f.trace() == Matrix[bool]([[1]], 1, 1)
    assert f.trace(left=True) == Matrix[bool]([[0]], 1, 1)
    assert Matrix[bool].swap(1, 1).trace(left=True) == Matrix[bool].id(1)


def test_trace_over_a_field_is_the_sum_over_unrollings_when_acyclic():
    rng = np.random.default_rng(0)
    for dtype in (float, complex):
        for dom, cod, n in [(1, 1, 1), (2, 3, 2), (0, 2, 3), (3, 1, 4)]:
            array = rng.integers(-3, 4, (dom + n, cod + n)).astype(dtype)
            array[dom:, cod:] = np.triu(array[dom:, cod:], k=1)
            f = Matrix[dtype](array, dom + n, cod + n)
            A, B = array[:dom, :cod], array[:dom, cod:]
            C, D = array[dom:, :cod], array[dom:, cod:]
            unrolled = A + sum(
                B @ np.linalg.matrix_power(D, k) @ C for k in range(n + 1))
            assert f.trace(n).is_close(Matrix[dtype](unrolled, dom, cod))
            assert f.trace(n).dtype == dtype


def test_trace_over_a_field_is_partial():
    f = Matrix[float]([[0.5, 1], [1, 1]], 2, 2)
    with raises(ValueError, match="partial"):
        f.trace()
    with raises(ValueError, match="partial"):
        Matrix[complex].id(2).trace(left=True)
    assert Matrix[float].id(2).trace(0) == Matrix[float].id(2)
    ill_conditioned = Matrix[float]([[0, 1e8], [0, 0]], 2, 2)
    assert ill_conditioned.resolvent()\
        == Matrix[float]([[1, 1e8], [0, 1]], 2, 2)


def test_trace_needs_booleans_or_a_field():
    f = Matrix[int]([[0, 1], [1, 0]], 2, 2)
    with raises(TypeError):
        f.trace()
    assert f.cast(float).trace() == Matrix[float]([[1]], 1, 1)


def test_trace_over_a_field_satisfies_the_traced_laws():
    # Matrix is not enrolled as a TracedCategory, so the property matrix has
    # no traced cell for it: the laws are stated on arrays here, with entries
    # small enough for every trace they take to be defined.
    rng, M = np.random.default_rng(0), Matrix[float]

    def matrix(dom, cod):
        return M(rng.uniform(0, .2, (dom, cod)), dom, cod)

    def holds(equation):
        first, *others = equation.terms
        return all(first.is_close(other) for other in others)

    laws = {name: getattr(TracedCategory, name).function for name in (
        "yanking", "trace_vanishing", "trace_iteration_left",
        "trace_iteration_right", "trace_superposing_left",
        "trace_superposing_right", "trace_naturality_left",
        "trace_naturality_right", "trace_dinaturality_left",
        "trace_dinaturality_right")}
    assert holds(laws["yanking"](M, Nat(1)))
    for _ in range(20):
        a, b, m, s, t = (int(x) for x in rng.integers(0, 3, 5))
        assert holds(laws["trace_vanishing"](M, matrix(a, b)))
        for side in ("left", "right"):
            f = matrix(m + a, m + b)
            assert holds(laws[f"trace_iteration_{side}"](M, f, Nat(m)))
            f = matrix(1 + a, 1 + b)
            assert holds(laws[f"trace_superposing_{side}"](M, f, Nat(s)))
            f, g = matrix(1 + s + b, 1 + s + a), matrix(a, b)
            assert holds(laws[f"trace_naturality_{side}"](M, Nat(1 + s), f, g))
            f, g = matrix(1 + s + a, 1 + t + b), matrix(1 + t, 1 + s)
            if side == "right":
                f = matrix(a + 1 + s, b + 1 + t)
            assert holds(laws[f"trace_dinaturality_{side}"](M, f, g))


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


def test_repeat_takes_logarithmically_many_products(monkeypatch):
    products, then = [], Matrix.then
    monkeypatch.setattr(Matrix, "then", lambda self, *others: (
        products.append(others), then(self, *others))[1])
    for n in (64, 3, 2, 1, 0):
        path = Matrix[bool](np.eye(n, k=1, dtype=bool), n, n)
        closure = Matrix[bool](np.triu(np.ones((n, n), dtype=bool)), n, n)
        products.clear()
        assert path.repeat() == closure
        assert len(products) <= n.bit_length()


def test_autotyping():
    pytest.importorskip("jax")
    torch = pytest.importorskip("torch")
    assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == float
    assert Matrix([0.5j], dom=1, cod=1).dtype == complex
    with backend('jax'):
        assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == np.float32
    with backend('pytorch'):
        assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == torch.float32



def test_trace_over_a_field_in_jax():
    pytest.importorskip("jax")
    with backend("jax"):
        f = Matrix([[0.5, 1.], [1., 0.25]], 2, 2)
        assert f.trace().is_close(Matrix([[0.5 + 4 / 3]], 1, 1), rtol=1e-5)
        with raises(ValueError, match="partial"):
            Matrix([[0.5, 1.], [1., 1.]], 2, 2).trace()

    def trace(array):
        with backend("jax"):
            return Matrix(array, 2, 2).trace().array

    jax = pytest.importorskip("jax")
    traced = jax.jit(trace)(np.array([[0.5, 1.], [1., 0.25]]))
    assert np.allclose(traced, [[0.5 + 4 / 3]], rtol=1e-5)


@pytest.mark.parametrize('name', ['pytorch', 'jax'])
def test_trace_and_composition_on_backend(name):
    """ The identity takes the dtype of the backend, so a matrix inferred
    there traces, and composes with the integer structural matrices on a
    backend whose product does not promote. """
    pytest.importorskip({'pytorch': 'torch'}.get(name, name))
    with backend(name):
        f = Matrix([[0.5, 1.], [1., 0.25]], 2, 2)
        assert type(f).id(2).dtype == f.dtype
        assert f.trace().is_close(Matrix([[0.5 + 4 / 3]], 1, 1), rtol=1e-5)
        assert (f >> Matrix.id(2)).dtype == f.dtype\
            == (Matrix.id(2) >> f).dtype
        assert (Matrix.id(2) >> f).is_close(f)


def test_is_field():
    assert is_field(float) and is_field(complex) and not is_field(int)
    assert is_field(np.float32) and not is_field(object) and not is_field(bool)
    assert is_field(np.dtype("float64")) and not is_field(np.dtype("int64"))
    torch = pytest.importorskip("torch")
    assert is_field(torch.float32) and is_field(torch.complex64)
    assert not is_field(torch.int64) and not is_field(torch.bool)


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


@pytest.mark.parametrize('name', ['pytorch', 'jax'])
def test_dagger_and_round_on_backend(name):
    """ Transposing, conjugating and rounding call the functions of the
    active backend, whose arrays need not have the methods numpy's have. """
    pytest.importorskip({'pytorch': 'torch'}.get(name, name))
    m = lambda: Matrix([1j, 2, 3, 4, 5, 6], 2, 3)
    run = lambda: [f.array.tolist() for f in (
        m().transpose(), m().conjugate(), m().dagger(), Matrix.merge(2),
        Matrix([1.2, 2.7], 1, 2).round())]
    expected = run()
    with backend(name):
        assert run() == expected


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


@pytest.mark.parametrize('cls', [Matrix, Tensor])
def test_strategy_max_entries_on_pytorch(cls):
    """ Entries are counted from the dimensions: a torch ``size`` is a
    method. """
    torch = pytest.importorskip("torch")
    from hypothesis import find
    strategy = cls.strategy(max_entries=4)
    with backend('pytorch'):
        drawn = find(strategy, lambda x: x.array.numel() > 3)
    assert type(drawn.array) is torch.Tensor and drawn.array.numel() == 4



def test_Matrix_of_backend_scalars_keeps_their_gradient():
    """ #652: a list of entries the backend already owns is stacked, where
    converting them one by one reads each as a scalar and detaches it. """
    torch = pytest.importorskip("torch")
    t = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    for entries, dom, cod, expected in (
            ([t], 1, 1, 1.), ([0, t], 1, 2, 1.), ([[t, t], [t, t]], 2, 2, 4.)):
        t.grad = None
        with backend('pytorch'):
            Matrix(entries, dom, cod).array.sum().backward()
        assert t.grad == expected
    t.grad = None
    with backend('pytorch'):
        Matrix[complex]([t], 1, 1).array.real.sum().backward()
    assert t.grad == 1.


def test_asarray_leaves_an_array_alone():
    """ #652: only nested entries are stacked, so an array of the right
    dtype is still the one the caller gave. """
    array = np.arange(4, dtype=float)
    assert asarray(array) is array
    assert (asarray([0.5, 0.5]) == np.array([0.5, 0.5])).all()
    assert asarray([], dtype=float).shape == (0, )
    jax = pytest.importorskip("jax")
    with backend('jax'):
        assert jax.grad(lambda x: asarray([x, x]).sum())(.5) == 2.


def test_Matrix_dtype_is_the_ring_of_entries():
    """ numpy's default precisions are the Python types they stand for, so
    the class a matrix infers is the one a user names, see #88. """
    for dtype in (bool, int, float, complex):
        assert Matrix[np.dtype(dtype).type] is Matrix[dtype]
        assert Tensor[np.dtype(dtype).type] is Tensor[dtype]
    assert Matrix[np.float32] is not Matrix[float]
    assert Matrix[int].id(2) == Matrix.id(2)
    assert type(Matrix.id(2) >> Matrix[int].id(2)) is Matrix[int]
    m = Matrix([1, 2], 1, 2)
    assert eval(repr(m), {"Matrix": Matrix, "Nat": Nat}) == m
    assert loads('{"factory": "matrix.Matrix[int64]", "array": [1, 2], '
                 '"dom": {"factory": "abc.Nat", "n": 1}, '
                 '"cod": {"factory": "abc.Nat", "n": 2}}') == m
    assert Subscript.decode(Subscript(Matrix, (np.int64, ))) is Matrix[int]


def test_Matrix_copy():
    """ The copy of the direct sum is the identities side by side. """
    assert Matrix.copy(3, 2) == Matrix(
        [[1, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 1]], 3, 6)
    assert Matrix.copy(3, Nat(2)) == Matrix.copy(3, 2)
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


def test_repr_leaves_numpy_printoptions_alone():
    threshold = np.get_printoptions()['threshold']
    with np.printoptions(threshold=1000):
        repr(Matrix(list(range(25)), 5, 5))
        assert np.get_printoptions()['threshold'] == 1000
    assert np.get_printoptions()['threshold'] == threshold


def test_Matrix_map_widens():
    """ A map with no dtype widens to its entries, see B9 of #31. """
    m = Matrix([1, 2], 1, 2)
    assert m.map(lambda x: x + 0.5) == Matrix[float]([1.5, 2.5], 1, 2)
    assert m.map(lambda x: 2 * x) == Matrix([2, 4], 1, 2)
    assert m.map(lambda x: x + 0.5, dtype=int) == m
    assert Matrix[complex]([1j], 1, 1).map(abs).dtype == complex
    assert Matrix.zero(0, 0).map(float) == Matrix.zero(0, 0)


def test_Matrix_basis_dtype():
    for x in range(4):
        assert Matrix.basis(x, 0).dtype == Matrix.id(x).dtype
    for dtype in (bool, int, float, complex):
        assert Matrix[dtype].basis(0, 0).dtype == dtype


def test_symbolic_matrix_composes_with_permutations_and_copies():
    """ Permutations and copies hold the integers zero and one, as id does:
    sympy does not multiply an expression by a boolean. """
    pytest.importorskip("sympy")
    from sympy import Expr
    from sympy.abc import psi
    M = Matrix[Expr]
    f, swap = M([psi], 1, 1), M.swap(1, 1)
    assert f @ M.id(1) >> swap == swap >> M.id(1) @ f
    assert f >> M.permutation([0]) == f
    assert f >> M.copy(1) == M([psi, psi], 1, 2)


def test_Matrix_shares_a_read_only_view():
    """ #8: an array of the right dtype is not copied, and the matrix
    stays immutable without freezing the caller's array. """
    array = np.arange(4, dtype=float).reshape((2, 2))
    m = Matrix[float](array, 2, 2)
    assert np.shares_memory(m.array, array)
    assert not m.array.flags.writeable and array.flags.writeable
    with raises(ValueError):
        m.array[0, 0] = 1
    array[0, 0] = 1
    assert array[0, 0] == 1


def test_Matrix_lambdify_modules():
    """ #445: a ``modules`` given to ``lambdify`` replaces the backend's,
    rather than being passed to sympy a second time. """
    pytest.importorskip("sympy")
    from sympy import Expr, sin
    from sympy.abc import psi
    f = Matrix[Expr]([sin(psi)], 1, 1)
    modules = [{"sin": lambda _: 42.}, "math"]
    assert f.lambdify(psi, dtype=float, modules=modules)(3)\
        == Matrix[float]([42.], 1, 1) != f.lambdify(psi, dtype=float)(3)


@pytest.mark.parametrize("matrix", [
    Matrix([np.nan, 1., -np.inf], 1, 3),
    Matrix[complex]([complex(1, np.nan), complex(np.nan, -np.inf)], 1, 2),
    Tensor([np.nan, np.inf], Dim(1), Dim(2))])
def test_repr_nan_inf(matrix):
    other = eval(repr(matrix))
    assert type(other) is type(matrix)
    for part in ("real", "imag"):
        assert np.array_equal(getattr(other.array, part),
                              getattr(matrix.array, part), equal_nan=True)


def test_repr_symbol_named_inf():
    sympy = pytest.importorskip("sympy")
    inf, nan = sympy.Symbol('inf'), sympy.Symbol('nan')
    matrix = Matrix[sympy.Expr]([inf, nan + 1, float('nan')], 1, 3)
    assert "float('nan')" in repr(matrix)
    env = {'Matrix': Matrix, 'Expr': sympy.Expr, 'Nat': Nat,
           'inf': inf, 'nan': nan}
    other = eval(repr(matrix), env)
    entries = other.array.reshape(-1)
    assert entries[:2].tolist() == [inf, nan + 1]
    assert np.isnan(float(entries[2]))
