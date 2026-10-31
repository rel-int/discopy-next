import pytest
import numpy as np
import copy as cp
import itertools

from discopy.optyx.core.control import *
from discopy.optyx.core.control import BinaryMatrixBox, ClassicalFunctionBox
from discopy.optyx.core.diagram import MAX_DIM
from discopy.optyx.core.zw import Create, W, ZBox
from discopy.optyx.core.diagram import PhotonThresholdDetector, Mode, Swap, Bit, Id
from discopy.optyx.photonic import Phase, BS, MZI
from discopy.optyx.utils.misc import matrix_to_zw
from discopy.optyx.classical import BitControlledGate, BitControlledPhaseShift
from discopy.optyx.channel import Channel, qmode
from discopy.optyx import photonic
from discopy.optyx import classical

# Helper Functions and Data

def assert_allclose_with_shape_mismatch(res_1, res_2):
    """
    Assert that res_1 and res_2 are elementwise close, allowing for shape mismatches
    by comparing only on the overlapping slice.
    """
    if res_1.shape != res_2.shape:
        min_shape = [min(r1, r2) for r1, r2 in zip(res_1.shape, res_2.shape)]
        slices = tuple(slice(0, s) for s in min_shape)
        assert np.allclose(res_1[slices], res_2[slices])
    else:
        assert np.allclose(res_1, res_2)



# Boolean classical functions

@pytest.mark.skip(reason="Helper function for testing")
def xor_2bits(x):
    """Return [x[0] ^ x[1]]."""
    return [x[0] ^ x[1]]

@pytest.mark.skip(reason="Helper function for testing")
def duplicate_bit(x):
    """Return [x[0], x[0]]."""
    return [x[0], x[0]]

@pytest.mark.skip(reason="Helper function for testing")
def swap_and_xor(x):
    """Return [x[0] ^ x[1], x[0]]."""
    return [x[0] ^ x[1], x[0]]

@pytest.mark.skip(reason="Helper function for testing")
def xor_both_bits(x):
    """Return [x[0] ^ x[1], x[0] ^ x[1]]."""
    return [x[0] ^ x[1], x[0] ^ x[1]]

@pytest.mark.skip(reason="Helper function for testing")
def complicated_xor(x):
    """
    Swap bits 2 and 3, then compute a few XORs. If xor_1 == 0 or xor_3 == 0,
    return 0; otherwise return [xor_2].
    """
    bitstring = cp.deepcopy(x)
    bitstring[2] = x[3]
    bitstring[3] = x[2]

    xor_1 = bitstring[0] ^ bitstring[1]
    xor_2 = bitstring[1] ^ bitstring[2]
    xor_3 = bitstring[2] ^ bitstring[3]

    if xor_1 == 0:
        return 0
    if xor_3 == 0:
        return 0
    else:
        return [xor_2]


# Real-valued classical functions (for Phase shifts)

@pytest.mark.skip(reason="Helper function for testing")
def real_f_1(x):
    x_val = x[0]
    return [x_val*0.234, x_val*0.912, x_val*0.184]

@pytest.mark.skip(reason="Helper function for testing")
def real_f_2(x):
    return [0.23*x[0]]

@pytest.mark.skip(reason="Helper function for testing")
def real_f_3(x):
    _ = x[0]
    return [0.8362, 0.193, 0.654]


# Parametrized data sets

CIRCUITS_TO_TEST = [
    (Phase(0.1).get_kraus(), None),
    (Phase(0.456).get_kraus(), Phase(0.8765).get_kraus()),
    (BS.get_kraus(), None),
    (BS.get_kraus(), MZI(0.324, 0.9875).get_kraus()),
    (W(2).dagger() >> W(2), MZI(0.324, 0.9875).get_kraus()),
]

CLASSICAL_FUNCTIONS_TO_TEST_MATRIX = (
    (xor_2bits, [1, 1]),
    (duplicate_bit, [[1],
                     [1]]),
    (swap_and_xor, [[1, 1],
                    [1, 0]]),
    (xor_both_bits, [[1, 1],
                     [1, 1]]),
)

REAL_FUNCS = [real_f_1, real_f_2, real_f_3]


# Test Classes

class TestBinaryControlledBox:
    """
    Tests for the BitControlledBox, verifying correct action vs. default
    based on a created photon (1) or not (0).
    """

    @pytest.mark.parametrize("action, default", CIRCUITS_TO_TEST)
    def test_binary_controlled_box(self, action, default):
        action_result = action.to_tensor().eval().array
        default_result = default.to_tensor().eval().array if default is not None else None

        if default is None:
            action_test = (
                (Create(1) >> PhotonThresholdDetector()) @ Mode(len(action.cod))
                >> BitControlledBox(action)
            ).to_tensor().eval().array
            default_test = (
                (Create(0) >> PhotonThresholdDetector()) @ Mode(len(action.cod))
                >> BitControlledBox(action)
            ).to_tensor().eval().array
        else:
            action_test = (
                (Create(1) >> PhotonThresholdDetector()) @ Mode(len(action.cod))
                >> BitControlledBox(action, default)
            ).to_tensor().eval().array
            default_test = (
                (Create(0) >> PhotonThresholdDetector()) @ Mode(len(action.cod))
                >> BitControlledBox(action, default)
            ).to_tensor().eval().array

        assert np.allclose(action_result, action_test)
        if default is not None:
            assert np.allclose(default_result, default_test)

    @pytest.mark.parametrize("action, default", CIRCUITS_TO_TEST)
    def test_bit_controlled_gate(self, action, default):

        action = Channel("action", action)
        default = Channel("default", default) if default is not None else None

        action_result = action.double().to_tensor().eval().array
        default_result = default.double().to_tensor().eval().array if default is not None else None

        if default is None:
            action_test = (
                (photonic.Create(1) >> photonic.PhotonThresholdMeasurement()) @ photonic.Id(len(action.cod))
                >> BitControlledGate(action)
            ).double().to_tensor().eval().array
            default_test = (
                (photonic.Create(0) >> photonic.PhotonThresholdMeasurement()) @ photonic.Id(len(action.cod))
                >> BitControlledGate(action)
            ).double().to_tensor().eval().array
        else:
            action_test = (
                (photonic.Create(1) >> photonic.PhotonThresholdMeasurement()) @ photonic.Id(len(action.cod))
                >> BitControlledGate(action, default)
            ).double().to_tensor().eval().array
            default_test = (
                (photonic.Create(0) >> photonic.PhotonThresholdMeasurement()) @ photonic.Id(len(action.cod))
                >> BitControlledGate(action, default)
            ).double().to_tensor().eval().array

        assert np.allclose(action_result, action_test)
        if default is not None:
            assert np.allclose(default_result, default_test)

    @pytest.mark.parametrize("action, default", CIRCUITS_TO_TEST)
    def test_binary_controlled_box_dagger(self, action, default):
        res_1 = BitControlledBox(action, default).to_tensor().dagger().eval().array
        res_2 = BitControlledBox(action, default).dagger().to_tensor().eval().array
        assert_allclose_with_shape_mismatch(res_1, res_2)


class TestClassicalFunctionBox:
    """
    Tests that a ClassicalFunctionBox(f, dom, cod) yields the same matrix
    as the known symbolic circuit or matrix box.
    """

    @pytest.mark.parametrize("function, m", CLASSICAL_FUNCTIONS_TO_TEST_MATRIX)
    def test_logical_matrix_box(self, function, m):
        m = np.array(m)
        if len(m.shape) == 1:
            m = m.reshape(1, -1)
        cod = Bit(len(m))
        dom = Bit(len(m[0]))

        f_arr = ClassicalFunctionBox(function, dom, cod).to_tensor().eval().array
        circ_arr = BinaryMatrixBox(m).to_tensor().eval().array
        assert np.allclose(f_arr, circ_arr)

    @pytest.mark.parametrize("function, m", CLASSICAL_FUNCTIONS_TO_TEST_MATRIX)
    def test_logical_matrix_box_dagger(self, function, m):
        """
        Verify dagger consistency for BinaryMatrixBox.
        """
        m = np.array(m)
        if len(m.shape) == 1:
            m = m.reshape(1, -1)
        cod = Bit(len(m))
        dom = Bit(len(m[0]))

        res_1 = BinaryMatrixBox(m).to_tensor().dagger().eval().array
        res_2 = BinaryMatrixBox(m).dagger().to_tensor().eval().array
        assert_allclose_with_shape_mismatch(res_1, res_2)


class TestControlledPhaseShift:
    """
    Tests involving real-valued classical functions for phase shifts and
    verifying dagger consistency.
    """

    @pytest.mark.parametrize("f", REAL_FUNCS)
    @pytest.mark.parametrize("diagram_creator", [
        lambda f: ControlledPhaseShift(f, len(f([0]))),
        lambda f: Mode(len(f([0]))) @ Create(4) >> ControlledPhaseShift(f, len(f([0])))
    ])
    def test_dagger_controlled_phase_shift(self, f, diagram_creator):
        """
        Confirm that taking the dagger at the diagram level is consistent
        with converting the diagram to a tensor, then taking dagger.
        """
        diagram = diagram_creator(f)
        res_1 = diagram.to_tensor().dagger().eval().array
        res_2 = diagram.dagger().to_tensor().eval().array
        assert_allclose_with_shape_mismatch(res_1, res_2)

    @pytest.mark.parametrize("f", REAL_FUNCS)
    @pytest.mark.parametrize("diagram_creator", [
        lambda f: BitControlledPhaseShift(f, len(f([0]))),
        lambda f: qmode**len(f([0])) @ classical.Digit(4) >> BitControlledPhaseShift(f, len(f([0])))
    ])
    def test_dagger_controlled_phase_shift_channel(self, f, diagram_creator):
        """
        Confirm that taking the dagger at the diagram level is consistent
        with converting the diagram to a tensor, then taking dagger.
        """
        diagram = diagram_creator(f)
        res_1 = diagram.double().to_tensor().dagger().eval().array
        res_2 = diagram.dagger().double().to_tensor().eval().array
        assert_allclose_with_shape_mismatch(res_1, res_2)

    diagrams_to_test_2 = [
        ControlledPhaseShift(lambda x: [x[0]], 1),
        ControlledPhaseShift(lambda x: [0.23*x[0], 0.456*x[0], 0.876*x[0], 0.654*x[0]], 4),
        ControlledPhaseShift(lambda x: [0.23*x[0]], 1),
        ControlledPhaseShift(lambda x: [0.23*x[0], 0.456*x[0], 0.876*x[0]], 3),
    ]

    @pytest.mark.parametrize("diagram", diagrams_to_test_2)
    def test_dagger_controlled_phase_shift_2(self, diagram):
        """
        Additional tests for ControlledPhaseShift, verifying that dagger
        consistency holds.
        """
        res_1 = diagram.to_tensor().dagger().eval().array
        res_2 = diagram.dagger().to_tensor().eval().array
        assert_allclose_with_shape_mismatch(res_1, res_2)

    xs = range(5)

    @pytest.mark.parametrize("f", REAL_FUNCS)
    @pytest.mark.parametrize("x", xs)
    def test_controlled_phase_shift_numeric(self, f, x):
        print(f"Testing ControlledPhaseShift with x: {x}")
        """
        For each real-valued function f and input x, build a circuit with
        ControlledPhaseShift, compare to a ZBox with the matching phase exponent.
        """
        n = len(f([0]))
        diag = Create(x) @ Mode(n) >> ControlledPhaseShift(f, n)

        # Build the "manual" ZBox product
        zbox = Id(Mode(0))
        for y in f([x]):
            zbox @= ZBox(1, 1, lambda i, y=y: np.exp(2 * np.pi * 1j * y) ** i)

        # Check forward and dagger equivalences
        assert np.allclose(zbox.to_tensor().eval().array, diag.to_tensor().eval().array)
        assert np.allclose(zbox.dagger().to_tensor().eval().array, diag.dagger().to_tensor().eval().array)
        assert np.allclose(
            zbox.to_tensor().dagger().eval().array,
            zbox.dagger().to_tensor().eval().array
        )


class TestPhotonThresholdDetector:
    """
    Tests verifying that PhotonThresholdDetector's dagger
    is consistent with the array representation.
    """

    circuits_to_test = [
        PhotonThresholdDetector()
    ]

    @pytest.mark.parametrize("circ", circuits_to_test)
    def test_photon_threshold_detector_dagger(self, circ):
        res_1 = circ.to_tensor().dagger().eval().array
        res_2 = circ.dagger().to_tensor().eval().array
        assert_allclose_with_shape_mismatch(res_1, res_2)


class TestMatrixToZW:
    """
    A sanity check that matrix_to_zw(...) yields correct transformations.
    """

    def test_matrix_to_zw(self):
        U = np.array([[1, 1], [1, 1]])
        diagram1 = matrix_to_zw(U)

        diagram2 = (
            W(2) @ W(2) >>
            Mode(1) @ Swap(Mode(1), Mode(1)) @ Mode(1) >>
            W(2).dagger() @ W(2).dagger()
        )

        assert np.allclose(diagram1.to_tensor().eval().array,
                           diagram2.to_tensor().eval().array)


def test_classical_function_keeps_the_outputs_it_can_reach():
    """No transition leaves an input the function sends to `None`, or to a
    value past the truncation; the dagger is the transposed relation."""
    def add(x):
        return None if x[0] == x[1] == 1 else x[0] + x[1]

    box = ClassicalFunctionBox(add, Mode(2), Mode(1))
    assert box.determine_output_dimensions([3, 3]) == [MAX_DIM]
    expected = np.zeros((3, 3, 3))
    for a, b in itertools.product(range(3), repeat=2):
        if (a, b) != (1, 1) and a + b < 3:
            expected[a, b, a + b] = 1
    array = box.truncation([3, 3], [3]).eval().array
    assert np.allclose(array, expected)
    assert np.allclose(box.dagger().truncation([3], [3, 3]).eval().array,
                       np.transpose(array, (2, 0, 1)))
    assert box.conjugate() == box


def test_binary_matrix_from_a_row():
    row = BinaryMatrixBox([1, 1])
    assert np.allclose(row.to_tensor().eval().array,
                       BinaryMatrixBox([[1, 1]]).to_tensor().eval().array)
    assert row.conjugate() == row


@pytest.mark.parametrize("controls", list(itertools.product(range(3), repeat=2)))
def test_controlled_phase_shift_two_control_modes(controls):
    """
    With two or more control modes the phase is the one the function reads off
    the whole control configuration: the box used to index its array along the
    first axis only, so the per-configuration cell was the wrong shape.
    """
    f = lambda xs: [0.23 * xs[0] + 0.456 * xs[1], 0.876 * xs[1]]
    n = len(f([0, 0]))
    diag = (Create(*controls) @ Mode(n)
            >> ControlledPhaseShift(f, n, n_control_modes=2))

    zbox = Id(Mode(0))
    for y in f(list(controls)):
        zbox @= ZBox(1, 1, lambda i, y=y: np.exp(2 * np.pi * 1j * y) ** i)

    assert_allclose_with_shape_mismatch(
        zbox.to_tensor().eval().array, diag.to_tensor().eval().array)
    assert_allclose_with_shape_mismatch(
        zbox.dagger().to_tensor().eval().array,
        diag.dagger().to_tensor().eval().array)


@pytest.mark.parametrize("n_control_modes, controls", [
    (1, (0,)), (1, (2,)), (1, (3,)),
    (2, (0, 0)), (2, (1, 2)), (2, (3, 1)),
])
def test_controlled_phase_shift_conjugate_negates_the_phases(
        n_control_modes, controls):
    """
    `conjugate()` used to be given the arguments of `self`, so the conjugate
    was `self`. Complex conjugation sends `exp(2*pi*1j*y)` to
    `exp(-2*pi*1j*y)`, so the oracle is the class docstring's own with every
    phase negated: the product of `ZBox(1, 1, exp(-2*pi*1j*y) ** i)` over
    `y in function(controls)`. Each function has a constant term, so no
    configuration leaves the phases at zero where the conjugate would be the
    original for a second reason.
    """
    f = lambda xs: [0.23 * sum(xs) + 0.1, 0.876 * xs[-1] + 0.3]
    n = len(f(list(controls)))
    box = ControlledPhaseShift(f, n, n_control_modes=n_control_modes)
    state = Create(*controls) @ Mode(n)
    original = (state >> box).to_tensor().eval().array
    conjugate = (state >> box.conjugate()).to_tensor().eval().array

    zbox = Id(Mode(0))
    for y in f(list(controls)):
        zbox @= ZBox(1, 1, lambda i, y=y: np.exp(-2 * np.pi * 1j * y) ** i)
    assert_allclose_with_shape_mismatch(
        zbox.to_tensor().eval().array, conjugate)

    # The oracle is not the original: a test that only reads the two as
    # different passes for a wrong answer.
    assert not np.allclose(original, conjugate)
    assert_allclose_with_shape_mismatch(np.conjugate(original), conjugate)

    # Conjugation is an involution, it keeps the boundary, and the dagger of
    # the conjugate is still the dagger of its truncation.
    assert_allclose_with_shape_mismatch(
        original,
        (state >> box.conjugate().conjugate()).to_tensor().eval().array)
    assert (box.conjugate().dom, box.conjugate().cod,
            box.conjugate().is_dagger) == (box.dom, box.cod, box.is_dagger)
    assert_allclose_with_shape_mismatch(
        (state >> box.conjugate()).to_tensor().dagger().eval().array,
        (state >> box.conjugate()).dagger().to_tensor().eval().array)
