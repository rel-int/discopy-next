import numpy as np
import pytest
from sympy.abc import x, y

from discopy.optyx import qubits
from discopy.optyx.channel import Diagram, Ty
from discopy.optyx.core import zw
from discopy.optyx.photonic import (
    Create, DualRail, Scalar, Select, Swap, XMeasurementDR, ZMeasurementDR)


def outcomes(diagram):
    return {key: value for key, value in diagram.eval().prob_dist().items()
            if not np.isclose(value, 0)}


def test_select_to_path():
    assert Select(1, 2).to_path() == zw.Select(1, 2).to_path()


def test_scalar_grad_and_lambdify():
    assert Scalar(2 * x).grad(x).data == 2
    assert Scalar(2 * x).grad(y) == Diagram.sum_factory((), Ty(), Ty())
    assert Scalar(2 * x).lambdify(x)(3) == Scalar(6)


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
