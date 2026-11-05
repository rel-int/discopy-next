from discopy.optyx.core.zx import *
from discopy.optyx.core.zx import ZBox
from discopy.optyx.core.diagram import Scalar
import itertools
import numpy as np
import pytest

n_legs_in = list(range(1, 3))
n_legs_out = list(range(0, 3))
phase = [0, 0.5, 1]

combinations = list(itertools.product(n_legs_in, n_legs_out, phase))


@pytest.mark.parametrize("n_legs_in, n_legs_out, phase", combinations)
def test_Z(n_legs_in, n_legs_out, phase):
    z = Z(n_legs_in, n_legs_out, phase)
    assert np.allclose(z.to_tensor().eval().array.flatten(), z.to_pyzx().to_tensor().flatten())

@pytest.mark.parametrize("n_legs_in, n_legs_out, phase", combinations)
def test_Z_tensor(n_legs_in, n_legs_out, phase):
    z = Z(n_legs_in, n_legs_out, phase)
    assert np.allclose(z.to_tensor().eval().array.flatten(),
                       z.to_pyzx().to_tensor().flatten())

@pytest.mark.parametrize("n_legs_in, n_legs_out, phase", combinations)
def test_X(n_legs_in, n_legs_out, phase):
    x = X(n_legs_in, n_legs_out, phase)
    assert np.allclose(x.to_tensor().eval().array.flatten(), x.to_pyzx().to_tensor().flatten())

@pytest.mark.parametrize("n_legs_in, n_legs_out, phase", combinations)
def test_X_tensor(n_legs_in, n_legs_out, phase):
    x = X(n_legs_in, n_legs_out, phase)
    assert np.allclose(x.to_tensor().eval().array.flatten(),
                       x.to_pyzx().to_tensor().flatten())

scalar = [0, 0.5, 1]


@pytest.mark.parametrize("scalar", scalar)
def test_scalar(scalar):
    s = Scalar(scalar)
    assert np.allclose(s.to_tensor().eval().array.flatten(), s.to_pyzx().to_tensor().flatten())


amplitudes = [0, 0.5, 3, -1j]
legs = [legs for legs in itertools.product(range(3), repeat=2) if any(legs)]


@pytest.mark.parametrize("legs", legs)
@pytest.mark.parametrize("amplitude", amplitudes)
def test_ZBox_is_a_pyzx_z_box(legs, amplitude):
    box = ZBox(*legs, amplitude)
    assert np.allclose(box.to_tensor().eval().array.flatten(),
                       box.to_pyzx().to_tensor().flatten())
