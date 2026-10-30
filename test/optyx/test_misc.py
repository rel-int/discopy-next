import numpy as np
import pytest

from discopy.optyx.channel import Channel
from discopy.optyx.core import diagram, zw, zx
from discopy.optyx.utils.misc import (
    basis_vector_from_kets, explode_channel, filter_occupation_numbers,
    matrix_to_zw, occupation_numbers, tensor_2_amplitudes)


def test_no_photons_fit_in_no_modes():
    assert occupation_numbers(0, 0) == [()]
    with pytest.raises(ValueError, match="zero modes"):
        occupation_numbers(1, 0)


def test_filter_occupation_numbers_by_dimension():
    configurations = [[0, 1], [2, 0], [1, 1]]
    assert filter_occupation_numbers(configurations, [1, 1]) \
        == [[0, 1], [1, 1]]


def test_amplitudes_past_the_truncation_read_zero():
    with pytest.warns(UserWarning) as caught:
        amplitudes = tensor_2_amplitudes(zw.Create(1).to_tensor(), 3)
    assert np.allclose(amplitudes, [0])
    messages = [str(warning.message) for warning in caught]
    assert any("truncated" in message for message in messages)
    assert any("out of bounds" in message for message in messages)


def test_explode_channel_keeps_the_kraus_map():
    kraus = zx.Z(1, 2) >> zx.X(2, 1)
    assert explode_channel(kraus).get_kraus() == kraus
    identity = diagram.Diagram.id(diagram.bit)
    assert explode_channel(identity) == Channel("Id", identity)


@pytest.mark.parametrize("array", [
    [[1, 2j, 3]],
    [[1, 0, 2], [0, 3, 4j]],
    [[1, 0], [2, 3], [0, 4j]],
    [[0, 1], [1, 0]],
])
def test_matrix_to_zw_of_a_rectangular_matrix(array):
    array = np.array(array, dtype=complex)
    assert np.allclose(matrix_to_zw(array).to_path().array, array)


def test_matrix_to_zw_of_a_symbolic_matrix_with_zeros():
    from sympy import Symbol
    x = Symbol("x")
    array = np.array([[x, 0], [2 * x, 1 - x]], dtype=object)
    result = matrix_to_zw(array).lambdify(x)(.5).to_path().array
    assert np.allclose(result, [[.5, 0], [1, .5]])


def test_basis_vector_rejects_negative_indices():
    assert basis_vector_from_kets([1, 2], [2, 3]) == 5
    with pytest.raises(ValueError, match="non-negative"):
        basis_vector_from_kets([-1], [3])
