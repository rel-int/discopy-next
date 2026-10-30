import math

from discopy.optyx.channel import *
from discopy.optyx.core import zx
from discopy.optyx import photonic
import pytest
import numpy as np


@pytest.mark.parametrize("box", [photonic.Phase, photonic.BBS, photonic.TBS])
def test_conjugation_LO(box):
    gate = box(0.27)
    dom = gate.dom
    lhs = Discard(dom)
    rhs = gate >> Discard(dom)
    lhs_tensor = lhs.double().to_tensor().eval().array
    rhs_tensor = rhs.double().to_tensor().eval().array
    assert np.allclose(lhs_tensor, rhs_tensor)

def test_conjugation_MZI():
    gate = photonic.MZI(0.27, 0.76)
    dom = qmode ** 2
    lhs = Discard(dom)
    rhs = gate >> Discard(dom)
    lhs_tensor = lhs.double().to_tensor().eval().array
    rhs_tensor = rhs.double().to_tensor().eval().array
    assert np.allclose(lhs_tensor, rhs_tensor)


def test_conjugation_LO_Gate():
    hbs_array = (1 / 2) ** (1 / 2) * np.array([[1, 1j], [1j, 1]])
    gate = photonic.Gate(hbs_array, 2, 2, "HBS").get_kraus()
    assert np.allclose(gate.conjugate().to_path().array,
                       hbs_array.conjugate())

@pytest.mark.parametrize("box", [photonic.Phase, photonic.BBS, photonic.TBS])
def test_conjugation_LO_channel(box):
    gate = box(0.27)
    hbs_array = gate.array
    assert np.allclose(gate.conjugate().to_path().array,
                       hbs_array.conjugate())


def test_conjugation_MZI_channel():
    gate = photonic.MZI(0.27, 0.76)
    hbs_array = gate.array
    assert np.allclose(gate.conjugate().to_path().array,
                       hbs_array.conjugate())


@pytest.mark.parametrize("box", [zx.Z, zx.X])
def test_conjugation_ZX(box):
    gate = box(1, 1, 0.69)
    lhs = Discard(qubit)
    rhs = Channel("Gate", gate) >> Discard(qubit)
    lhs_tensor = lhs.double().to_tensor().eval().array
    rhs_tensor = rhs.double().to_tensor().eval().array
    assert np.allclose(lhs_tensor, rhs_tensor)


def test_conjugation_ZW():
    diagram = photonic.TBS(0.24, 0.86).get_kraus()
    lhs = Discard(qmode ** 2)
    rhs = Channel('diagram', diagram) >> Discard(qmode ** 2)
    lhs_tensor = lhs.double().to_tensor().eval().array
    rhs_tensor = rhs.double().to_tensor().eval().array
    assert np.allclose(lhs_tensor, rhs_tensor)


GATES = [
    photonic.TBS(0.15),
    photonic.TBS(0.15, is_conj=True),
    photonic.TBS(0.15, is_gate_dagger=True),
    photonic.TBS(0.15, is_gate_dagger=True, is_conj=True),
    photonic.MZI(0.15, 0.3),
    photonic.MZI(0.15, 0.3, is_conj=True),
    photonic.MZI(0.15, 0.3, is_gate_dagger=True),
    photonic.MZI(0.15, 0.3, is_gate_dagger=True, is_conj=True),
]


def path_array(gate):
    return np.asarray(gate.to_path().array, dtype=complex)


# next#314: the global phase ORed the two flags, so a conjugated gate was the
# daggered one, and decomp ignored is_conj, so the gradient of a conjugate was
# the gradient.
@pytest.mark.parametrize("gate", GATES)
def test_a_gate_decomposes_into_itself(gate):
    assert np.allclose(path_array(gate), path_array(gate.decomp()))


@pytest.mark.parametrize("gate", GATES)
def test_a_gate_undoes_its_dagger(gate):
    assert np.allclose(
        path_array(gate >> gate.dagger()), path_array(Id(qmode ** 2)))


@pytest.mark.parametrize("gate", GATES)
def test_the_conjugate_of_a_gate_conjugates_its_matrix(gate):
    assert np.allclose(
        path_array(gate.conjugate()), np.conjugate(path_array(gate)))

