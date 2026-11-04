import numpy as np
import pytest
from sympy.abc import x, y

from discopy.optyx import qubits
from discopy.optyx.channel import Diagram, Ty
from discopy.optyx.core import zw
from discopy.optyx.photonic import (
    BBS, BS, Create, DualRail, PhotonLoss, Scalar, Select, Swap,
    XMeasurementDR, ZMeasurementDR)


def outcomes(diagram):
    return {key: value for key, value in diagram.eval().prob_dist().items()
            if not np.isclose(value, 0)}


def test_select_to_path():
    assert Select(1, 2).to_path() == zw.Select(1, 2).to_path()


def test_scalar_grad_and_lambdify():
    assert Scalar(2 * x).grad(x).data == 2
    assert Scalar(2 * x).grad(y) == Diagram.sum_factory((), Ty(), Ty())
    assert Scalar(2 * x).lambdify(x)(3) == Scalar(6)
    assert Scalar(x ** 2).grad(x).data == 2 * x


@pytest.mark.parametrize("outcome", [0, 1])
def test_dual_rail_measurements(outcome):
    """With no phase, the Z measurement reads the X basis of a dual-rail
    qubit and the X measurement, a Hadamard before it, the Z basis."""
    sign = "+-"[outcome]
    assert outcomes(qubits.Ket(sign) >> DualRail(1) >> ZMeasurementDR(0)) \
        == {(outcome, ): 1}
    assert outcomes(qubits.Ket(outcome) >> DualRail(1) >> XMeasurementDR(0)) \
        == {(outcome, ): 1}


def test_swap():
    assert outcomes(Create(2, 0) >> Swap(1, 1)) == {(0, 2): 1}
    assert outcomes(Create(1, 2, 3) >> Swap(2, 1)) == {(3, 1, 2): 1}
    assert outcomes(Create(1, 2, 3) >> Swap(1, 2)) == {(2, 3, 1): 1}
    assert outcomes(Create(1, 2, 3) >> Swap(2, 1) >> Swap(2, 1).dagger()) \
        == {(1, 2, 3): 1}


def test_prob_dist_closed():
    hom = Create(1, 1) >> BS >> Select(2, 0)
    assert np.isclose(hom.eval().prob_dist(normalise=False)[()], 0.5)
    lossy = Create(1) >> PhotonLoss(0.5) >> Select(1)
    assert np.isclose(lossy.eval().prob_dist(normalise=False)[()], 0.5)
    assert lossy.eval().prob_dist() == {(): 1}


@pytest.mark.parametrize("is_conj", [False, True])
def test_BBS_dagger_and_lambdify_keep_is_conj(is_conj):
    box = BBS(0.3, is_conj=is_conj)
    assert np.allclose(box.dagger().array, box.array.conj().T)
    assert BBS(x, is_conj=is_conj).lambdify(x)(0.3).is_conj == is_conj


def test_select_and_create_default_to_one_photon():
    """Both default to one photon on one mode, like the ZW boxes they
    wrap, rather than to no mode at all."""
    assert Select() == Select(1) and Create() == Create(1)
    assert Select().photons == Create().photons == (1,)
    assert Select().to_path() == zw.Select().to_path()
    assert outcomes(Create() >> Select()) == {(): 1}
