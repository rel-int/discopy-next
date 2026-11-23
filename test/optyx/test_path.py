import numpy as np
import pytest
import sympy

from discopy.abc import Nat
from discopy.optyx import photonic, qmode
from discopy.optyx.channel import Diagram
from discopy.optyx.core import zw
from discopy.optyx.core.path import Matrix, Sum, npperm
from discopy.utils import AxiomError, dumps, loads


def beam_splitter():
    """The balanced beam splitter as a QPath matrix."""
    amplitudes = [2 ** -.5, 2 ** -.5, 2 ** -.5, -2 ** -.5]
    return (zw.Split(2) @ zw.Split(2) >> zw.Id(1) @ zw.SWAP @ zw.Id(1)
            >> zw.Id(0).tensor(*map(zw.Endo, amplitudes))
            >> zw.Merge(2) @ zw.Merge(2)).to_path()


def test_repr_is_transparent():
    matrix = Matrix(np.eye(3), 2, 2, creations=(1, ), selections=(2, ))
    assert eval(repr(matrix), Matrix.environment()) == matrix


def test_the_tree_of_a_matrix_names_its_whole_profile():
    """Every field the repr names travels through the tree, see #696."""
    matrix = Matrix(np.eye(3), 2, 2, creations=(1, ), selections=(2, ),
                    normalisation=.5, scalar=2j)
    assert sorted(matrix.to_tree()) == [
        'array', 'cod', 'creations', 'dom', 'factory', 'normalisation',
        'scalar', 'selections']
    assert loads(dumps(matrix)) == matrix
    assert loads(dumps(Matrix[float]([[1.]], 1, 1))).dtype is float


@pytest.mark.parametrize("field", [
    {"creations": (2, )}, {"selections": (2, )},
    {"normalisation": 2}, {"scalar": 2}])
def test_equality_reads_every_field_of_the_repr(field):
    fields = {"creations": (1, ), "selections": (1, )}
    matrix = Matrix(np.eye(2), 1, 1, **fields)
    assert matrix == Matrix(np.eye(2), 1, 1, **fields)
    assert matrix != Matrix(np.eye(2), 1, 1, **fields | field)


def test_matrices_with_different_creations_are_not_equal():
    assert zw.Create(1).to_path() != zw.Create(2).to_path()
    assert Matrix([[1], [1]], 0, 1, creations=(1, 1)) \
        != Matrix([[1]], 0, 1, creations=(1, ))
    assert Matrix([[1], [1], [1]], 0, 1, creations=(1, 1, 1)) \
        != Matrix([[1], [1]], 0, 1, creations=(1, 1))


def test_selecting_more_photons_than_there_are():
    with pytest.raises(ValueError, match="positive number of photons"):
        zw.Select(2).to_path().eval(1)


@pytest.mark.parametrize("with_perceval", [False, True])
def test_probabilities_as_a_tensor(with_perceval):
    """The Born rule squares the amplitudes, whoever computes them."""
    matrix = beam_splitter()
    probabilities = matrix.prob(1, with_perceval=with_perceval, as_tensor=True)
    amplitudes = matrix.eval(1, as_tensor=True)
    assert np.allclose(probabilities.array, np.abs(amplitudes.array) ** 2)


def test_permanent_of_the_empty_matrix_is_one():
    assert npperm(np.zeros((0, 0))) == 1


def test_identity_with_no_photons_evaluates():
    assert np.allclose(zw.Id(1).to_path().eval().array, [[1]])


def test_dagger_conjugates_the_scalar():
    matrix = Matrix(np.eye(1), 1, 1, normalisation=2j, scalar=1j)
    dagger = matrix.dagger()
    assert (dagger.normalisation, dagger.scalar) == (-2j, -1j)
    assert np.allclose(dagger.eval(1).array, matrix.eval(1).array.conj().T)


def test_dilate_keeps_the_scalar_and_normalisation():
    matrix = Matrix(beam_splitter().array, 2, 2, normalisation=2, scalar=1j)
    for n_photons in range(3):
        assert np.allclose(matrix.dilate().eval(n_photons).array,
                           matrix.eval(n_photons).array)


def selected_phase(scalar):
    """A matrix whose profile makes it unaddable as an array, see #544."""
    return Matrix(np.array([[1, 1], [1, 0]]), 1, 1,
                  creations=(1, ), selections=(1, ), scalar=scalar)


def test_amplitudes_of_a_sum_add():
    a, b = selected_phase(2j), selected_phase(5j)
    assert np.allclose(a.eval(2).array, 4j)
    assert np.allclose(b.eval(2).array, 10j)
    assert np.allclose((a + b).eval(2).array, 14j)
    assert np.allclose((a + b).prob(2).array, abs(14j) ** 2)
    assert np.allclose((a + b).prob(2, as_tensor=True).array,
                       np.abs((a + b).eval(2, as_tensor=True).array) ** 2)


def test_zero_is_the_empty_sum():
    zero = Matrix.zero(1, 1)
    assert zero.terms == () and (zero.dom, zero.cod) == (Nat(1), Nat(1))
    assert np.allclose(zero.eval(2).array, [[0]])
    tensor = zero.eval(2, as_tensor=True)
    term = selected_phase(2j).eval(2, as_tensor=True)
    assert (tensor.dom, tensor.cod) == (term.dom, term.cod)
    assert np.allclose(tensor.array, np.zeros(term.array.shape))
    assert zero + selected_phase(2j) == Sum((selected_phase(2j), ), 1, 1)


def test_a_sum_is_neither_a_unitary_nor_hashable():
    a, b = selected_phase(2j), selected_phase(5j)
    with pytest.raises(NotImplementedError, match="one unitary"):
        (a + b).prob(2, with_perceval=True)
    with pytest.raises(TypeError, match="unhashable"):
        hash(a + b)


def test_sum_of_different_photon_sectors_has_none_in_common():
    created = Matrix(np.ones((2, 2)), 1, 2, creations=(1, ))
    assert created.eval(1).cod != zw.Split(2).to_path().eval(1).cod
    with pytest.raises(AxiomError, match="same number of photons"):
        (created + zw.Split(2).to_path()).eval(1)


def test_one_mode_out_hides_no_sector():
    kept = Matrix(np.eye(1), 1, 1)
    created = Matrix(np.ones((2, 1)), 1, 1, creations=(1, ))
    assert (kept.net_photons, created.net_photons) == (0, 1)
    assert kept.eval(1).cod == created.eval(1).cod == Nat(1)
    with pytest.raises(AxiomError, match=r"same number of photons, got \[0, 1\]"):
        (kept + created).eval(1)
    assert np.allclose((kept + kept).eval(1).array, 2)


def test_the_ring_of_a_sum_survives_every_operation():
    zero = Matrix[float].zero(1, 1)
    for sum_ in (zero.tensor(zero), zero.dagger(), zero + zero,
                 zero.then(zero)):
        assert sum_.terms == () and sum_.dtype is float
    typed = Matrix[float]([1.], 1, 1)
    assert (typed + typed).dtype is float


def test_repr_of_a_sum_is_transparent():
    for matrix in (Matrix.zero(1, 1), Matrix[float].zero(1, 1),
                   selected_phase(2j) + selected_phase(5j)):
        assert eval(repr(matrix), Matrix.environment()) == matrix


def test_the_tree_of_an_empty_sum_names_its_ring():
    for matrix in (Matrix.zero(1, 1), Matrix[float].zero(1, 1),
                   selected_phase(2j) + selected_phase(5j)):
        assert loads(dumps(matrix)) == matrix


def test_a_sum_evaluates_over_the_ring_of_its_terms():
    assert Matrix[float].zero(1, 1) != Matrix[complex].zero(1, 1)
    assert Matrix[float].zero(1, 1).eval(1).dtype is float
    mixed = Matrix[int](np.array([[1]]), 1, 1) \
        + Matrix[complex](np.array([[.5 + 0j]]), 1, 1)
    assert mixed.dtype is int and mixed.eval(1).dtype is complex
    assert np.allclose(mixed.prob(1).array, abs(1.5) ** 2)


def test_sums_compose_and_tensor_termwise():
    a, b = selected_phase(2j), selected_phase(5j)
    assert (a + b).then(a) == a.then(a) + b.then(a) == a.then(a + b)
    assert (a + b).tensor(a) == a.tensor(a) + b.tensor(a) == a.tensor(a + b)
    assert (a + b).dagger() == a.dagger() + b.dagger()


@pytest.mark.parametrize("diagram, expected", [
    (lambda phase: phase, [[1]]),
    (lambda phase: phase @ Diagram.id(qmode), [[1, 0], [0, 0]]),
    (lambda phase: phase >> photonic.Phase(.2), [[np.exp(.4j * np.pi)]])])
def test_grad_of_a_composite_phase(diagram, expected):
    """The three cases of #544: the expected amplitudes on one photon are
    those of the derivative of a phase, `2i pi n e^{2i pi psi n}`."""
    psi = sympy.Symbol("psi")
    amplitudes = diagram(photonic.Phase(psi)).grad(psi).subs(
        (psi, .1)).to_path().eval(1)
    assert np.allclose(
        amplitudes.array,
        2j * np.pi * np.exp(.2j * np.pi) * np.array(expected))
