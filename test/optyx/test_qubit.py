import pytest
import pyzx
from discopy.optyx import qubits
from pytket import Circuit
from pytket.extensions.qiskit import AerBackend
from pytket.utils import probs_from_counts
from discopy.optyx import channel
from discopy.optyx.core import zx
import numpy as np

def _extract(graph):
    graph = graph.copy()
    pyzx.simplify.full_reduce(graph)
    return pyzx.extract_circuit(graph)


def test_pyzx():
    c = pyzx.Circuit(3)
    c.add_gate("TOF", 0, 1, 2)
    g = c.to_basic_gates().to_graph()
    c1 = _extract(g)
    c2 = _extract(qubits.Circuit(g).to_pyzx())
    assert c1.verify_equality(c2)

    c1 = _extract(g)
    c2 = _extract(channel.Diagram.from_pyzx(g).to_pyzx())
    assert c1.verify_equality(c2)

def test_graphix():
    import graphix
    from graphix.simulator import PatternSimulator
    for _ in range(5):
        rng = np.random.default_rng()
        theta = rng.random(4)

        circuit = graphix.Circuit(2)
        circuit.rz(0, theta[0])
        circuit.rz(1, theta[1])
        circuit.cnot(0, 1)
        circuit.s(0)
        circuit.cnot(1, 0)
        circuit.rz(1, theta[2])
        circuit.cnot(1, 0)
        circuit.rz(0, theta[3])
        pattern = circuit.transpile().pattern
        optyx_res = (qubits.Ket("+")**2 >> qubits.Circuit(pattern)).eval().amplitudes()

        simulator = PatternSimulator(pattern, backend="statevector")
        simulator.run()
        state = simulator.backend.state
        graphix_result = state.psi.reshape((2, ) * state.nqubit).conj()
        # an open graph defines its state up to a scalar, and graphix 0.4
        # absorbs Cliffords into measurements, so compare up to a phase
        largest = max(optyx_res, key=lambda keys: abs(optyx_res[keys]))
        phase = graphix_result[largest] / optyx_res[largest]
        assert np.isclose(abs(phase), 1, atol=1e-6)
        for keys in optyx_res.keys():
            assert np.isclose(
                phase * optyx_res[keys], graphix_result[keys], atol=1e-6)


# def test_tket_discopy():
#     from discopy.optyx import classical, bit
#     ghz_circ = Circuit(3)
#     ghz_circ.H(0)
#     ghz_circ.CX(0, 1)
#     ghz_circ.CX(1, 2)
#     ghz_circ.measure_all()

#     backend = AerBackend()
#     compiled_circ = backend.get_compiled_circuit(ghz_circ)
#     handle = backend.process_circuit(compiled_circ, n_shots=200000)
#     counts = backend.get_result(handle).get_counts()
#     tket_probs = probs_from_counts({key: np.round(v, 2) for key, v in probs_from_counts(counts).items()})

#     circ = qubits.Circuit(ghz_circ)
#     circ = qubits.Ket(0)**3 @ classical.Bit(0)**3 >> circ >> qubits.Discard(3) @ bit**3

#     res = (circ.double().to_tensor().to_quimb()^...).data

#     rounded_result = np.round(res, 6)

#     non_zero_dict = {idx: val for idx, val in np.ndenumerate(rounded_result) if val != 0}

#     assert tket_probs == non_zero_dict

#     circ = channel.Diagram.from_tket(ghz_circ)
#     circ = qubits.Ket(0)**3 @ classical.Bit(0)**3 >> circ >> qubits.Discard(3) @ bit**3

#     res = (circ.double().to_tensor().to_quimb()^...).data

#     rounded_result = np.round(res, 6)

#     non_zero_dict = {idx: val for idx, val in np.ndenumerate(rounded_result) if val != 0}

#     assert tket_probs == non_zero_dict

def test_zx():
    circuit = qubits.Z(1, 2) >> qubits.H() @ qubits.H()
    assert qubits.Circuit(circuit) == circuit

def test_pure_double_kraus():
    c = pyzx.Circuit(3)
    c.add_gate("TOF", 0, 1, 2)
    g = c.to_basic_gates().to_graph()
    assert qubits.Circuit(g).double() == qubits.Circuit(g).double()

    assert qubits.Circuit(g).is_pure == qubits.Circuit(g).is_pure

    assert qubits.Circuit(g).get_kraus() == qubits.Circuit(g).get_kraus()

# def test_to_dual_rail():
#     circuit = qubits.Z(1, 2) >> qubits.H() @ qubits.H()
#     dr_1 = qubits.Circuit(circuit).to_dual_rail().get_kraus()
#     dr_2 = zx.zx2path(circuit.get_kraus())
#     assert dr_1 == dr_2

def test_discard_qubits():
    a = (qubits.Discard(2).double().to_tensor().to_quimb() ^ ...).data
    b = ((qubits.Z(1, 1)**2 >> qubits.Discard(2)).double().to_tensor().to_quimb() ^ ...).data
    assert np.allclose(a, b)

def test_bit_flip_error():
    prob = 0.43
    a = (qubits.Z(0, 1) >> qubits.BitFlipError(prob)).eval().tensor.array
    assert np.allclose(a, np.array([[-.14, -0.14], [-0.14, -0.14]]))

def test_dephasingerror():
    prob = 0.43
    a = (qubits.Z(0, 1) >> qubits.DephasingError(prob)).eval().tensor.array
    assert np.allclose(a,  np.array([[-.14, 1.], [1, -0.14]]))

def test_ket():
    from discopy.optyx.core import diagram
    a = (qubits.Ket(1).get_kraus().to_tensor().to_quimb() ^ ...).data
    b = zx.X(0, 1, 0.5) @ diagram.Scalar(1 / np.sqrt(2))

    b = (b.to_tensor().to_quimb() ^ ...).data
    assert np.allclose(a, b)

def test_bra():
    from discopy.optyx.core import diagram
    a = (qubits.Bra(1).get_kraus().to_tensor().to_quimb() ^ ...).data
    b = zx.X(1, 0, 0.5) @ diagram.Scalar(1 / np.sqrt(2))
    b = (b.to_tensor().to_quimb() ^ ...).data
    assert np.allclose(a, b)


def test_ket_bra_accept_string_digits():
    assert qubits.Ket("0") == qubits.Ket(0)
    assert qubits.Ket("1") == qubits.Ket(1)
    assert qubits.Bra("0") == qubits.Bra(0)
    assert qubits.Bra("1") == qubits.Bra(1)


def test_ket_bra_reject_invalid_value():
    with pytest.raises(ValueError):
        qubits.Ket("2")
    with pytest.raises(ValueError):
        qubits.Bra("2")


def test_graphix_phase_gadget():
    """A YZ-plane measurement becomes a phase gadget, and the state agrees
    with graphix's own simulation up to a global phase, whichever outcome
    the correction undoes."""
    from graphix.command import E, M, N, Z
    from graphix.measurements import Measurement
    from graphix.pattern import Pattern
    from graphix.simulator import PatternSimulator
    pattern = Pattern(input_nodes=[0], cmds=[
        N(1), E((0, 1)), M(1, Measurement.YZ(0.25)), Z(0, {1})],
        output_nodes=[0])
    result = (qubits.Ket("+") >> channel.Diagram.from_graphix(pattern)
              ).eval().amplitudes()
    amplitudes = np.array([result.get((i, ), 0) for i in range(2)])
    for seed in range(4):
        simulator = PatternSimulator(pattern, backend="statevector")
        simulator.run(rng=np.random.default_rng(seed))
        expected = simulator.backend.state.psi.reshape(-1)
        assert np.isclose(abs(np.vdot(expected, amplitudes)), 1)


def test_graphix_output_cliffords_are_not_supported():
    from graphix.clifford import Clifford
    from graphix.command import C
    from graphix.pattern import Pattern
    pattern = Pattern(
        input_nodes=[0], cmds=[C(0, Clifford.H)], output_nodes=[0])
    with pytest.raises(NotImplementedError, match="Clifford"):
        qubits.Circuit(pattern)


def test_discopy_circuits_with_bits():
    from discopy.quantum import circuit, gates
    assert qubits.Circuit(gates.H @ circuit.bit) == qubits.H() @ channel.bit
    assert channel.Diagram.from_discopy(gates.H) == qubits.H()
    with pytest.raises(TypeError, match="Unsupported object type"):
        qubits.Circuit(circuit.Id(circuit.Ty(circuit.Qudit(3))))
    with pytest.raises(NotImplementedError, match="Encode"):
        qubits.Circuit(circuit.Encode())


def test_encode_is_the_ket_of_its_bit():
    from discopy.optyx import classical
    encoded = classical.Bit(1) >> qubits.Encode(1)
    assert np.allclose(
        encoded.double().to_tensor().eval().array,
        qubits.Ket(1).double().to_tensor().eval().array)


def test_spiders_lambdify_and_dagger():
    from sympy.abc import x
    assert qubits.Z(1, 2, x).lambdify(x)(.25) == qubits.Z(1, 2, .25)
    assert qubits.X(2, 1, x).lambdify(x)(.25) == qubits.X(2, 1, .25)
    assert qubits.Z(1, 2, .25).dagger() == qubits.Z(2, 1, -.25)
    assert qubits.X(1, 2, .25).dagger() == qubits.X(2, 1, -.25)
    assert qubits.H().dagger() == qubits.H()


def test_effect_decomposition():
    effect = qubits.Z(1, 0, .5)
    assert np.allclose(
        effect.decomp().double().to_tensor().eval().array,
        effect.double().to_tensor().eval().array)


def probabilities(diagram):
    return {key: value for key, value in diagram.eval().prob_dist().items()
            if not np.isclose(value, 0)}


def test_spiders_to_dual_rail():
    """Encoding then applying a spider is applying its dual-rail
    translation to the encoding; a fusion keeps equal inputs."""
    from discopy.optyx.photonic import DualRail
    assert probabilities(qubits.Z(0, 1).to_dual_rail()) \
        == probabilities(qubits.Z(0, 1) >> DualRail(1))
    for i in range(2):
        state = qubits.Ket(i) @ qubits.Ket(i)
        assert probabilities(
            state >> DualRail(2) >> qubits.Z(2, 1).to_dual_rail()) \
            == probabilities(state >> qubits.Z(2, 1) >> DualRail(1))


def test_scalar():
    from discopy.optyx import photonic
    assert qubits.Scalar(2).decomp() == qubits.Scalar(2)
    assert qubits.Scalar(2).to_dual_rail() == photonic.Scalar(2)
    assert np.isclose(qubits.Scalar(1j).dagger().get_kraus().to_tensor()
                      .eval().array, -1j)


@pytest.mark.parametrize("error", [qubits.BitFlipError, qubits.DephasingError])
def test_pauli_errors_are_self_adjoint(error):
    """A mixture of Hermitian Kraus maps is its own adjoint."""
    superoperator = error(.1).double().to_tensor().eval().array.reshape(4, 4)
    assert np.allclose(superoperator, superoperator.conjugate().T)
    assert error(.1).dagger() == error(.1)

# def test_to_tket():
#     circ = qubits.X(1, 2) @ channel.qubit >> channel.qubit @ qubits.Z(2, 1) @ qubits.Scalar(2**0.5)

#     tket_circ = circ.to_tket()
#     tket_circ.measure_all()
#     backend = AerBackend()
#     compiled_circ = backend.get_compiled_circuit(tket_circ)
#     handle = backend.process_circuit(compiled_circ, n_shots=200000)
#     counts = backend.get_result(handle).get_counts()
#     tket_probs = probs_from_counts({key: np.round(v, 2) for key, v in probs_from_counts(counts).items()})

#     circ_meas_prep = qubits.Ket(0) @ qubits.Ket(0) >> circ >> qubits.Measure(2)

#     res = ((circ_meas_prep.double().to_tensor().to_quimb()^...).data)

#     rounded_result = np.round(res, 6)

#     non_zero_dict = {idx: val for idx, val in np.ndenumerate(rounded_result) if val != 0}

#     assert tket_probs == non_zero_dict