import numpy as np
import pytest

from discopy.optyx.core import zw
from discopy.optyx.core.path import Matrix


def beam_splitter():
    """The balanced beam splitter as a QPath matrix."""
    amplitudes = [2 ** -.5, 2 ** -.5, 2 ** -.5, -2 ** -.5]
    return (zw.Split(2) @ zw.Split(2) >> zw.Id(1) @ zw.SWAP @ zw.Id(1)
            >> zw.Id(0).tensor(*map(zw.Endo, amplitudes))
            >> zw.Merge(2) @ zw.Merge(2)).to_path()


def test_repr_is_transparent():
    matrix = Matrix(np.eye(3), 2, 2, creations=(1, ), selections=(2, ))
    assert eval(repr(matrix)) == matrix


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
