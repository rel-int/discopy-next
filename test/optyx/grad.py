from discopy.optyx.core.zw import Id
from discopy.optyx.photonic import Phase, BS, TBS, MZI
from discopy.optyx.channel import qmode, Diagram
from sympy.abc import psi, theta
import numpy as np

from itertools import product

import pytest

param_circuits = [
    Phase(psi),
    BS >> Phase(psi) @ qmode >> BS,
    Phase(3 * psi ** 3),
    TBS(psi),
    MZI(psi, 0.123),
    MZI(0.123, psi),
    # next#314: a conjugated gate's dagger was not its inverse, and its
    # gradient was the unconjugated gate's.
    TBS(psi, is_conj=True),
    TBS(psi, is_gate_dagger=True, is_conj=True),
    MZI(psi, 0.123, is_conj=True),
    MZI(psi, 0.123, is_gate_dagger=True, is_conj=True),
]

values = [x * 0.123 for x in range(10)]


@pytest.mark.parametrize("circ, value", product(param_circuits, values))
def test_daggers_cancel(circ, value):
    d = circ >> circ.dagger()
    out = d.grad(psi).subs((psi, value)).eval().tensor.array
    assert np.allclose(out, np.zeros(shape=out.shape))


@pytest.mark.parametrize("circ", param_circuits)
def test_zero_grad(circ):
    assert circ.grad(theta) == Diagram.sum_factory((), circ.dom, circ.cod)


@pytest.mark.parametrize("circ, value", product(param_circuits, values[1:]))
def test_grad_matches_central_differences(circ, value):
    """The gradient of a parametrised gate against the difference quotient of
    the gate itself, which is what pins the gradient of a conjugate."""
    step = 1e-6
    gradient = np.asarray(
        circ.grad(psi).subs((psi, value)).eval().tensor.array, dtype=complex)
    quotient = (
        np.asarray(circ.subs((psi, value + step)).eval().tensor.array,
                   dtype=complex)
        - np.asarray(circ.subs((psi, value - step)).eval().tensor.array,
                     dtype=complex)) / (2 * step)
    # The gradient diagram is longer, so it is evaluated at a higher
    # truncation: the two agree on the dimensions they share.
    shared = tuple(
        slice(0, min(i, j)) for i, j in zip(gradient.shape, quotient.shape))
    assert np.allclose(gradient[shared], quotient[shared], atol=1e-4)
