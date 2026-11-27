import pytest
import pyzx
from discopy.optyx import qubits
from pytket import Circuit
from discopy.optyx import channel
from discopy.optyx.core import diagram, zx
from discopy.quantum.gates import (
    CRx, CRz, CU1, CX, CY, CZ, S, SWAP, T, U1, Controlled, Rx, Ry, Rz)
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
        optyx_res = (
            qubits.Ket("+")**2 >> qubits.Circuit(pattern)
        ).eval().amplitudes()

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


def _implicit_swap(circuit):
    circuit.replace_SWAPs()
    assert circuit.implicit_qubit_permutation() != {
        q: q for q in circuit.qubits}
    return circuit


@pytest.mark.parametrize("build", [
    lambda: Circuit(1).H(0),
    lambda: Circuit(1).X(0),
    lambda: Circuit(1).Y(0),
    lambda: Circuit(1).Z(0),
    lambda: Circuit(1).S(0),
    lambda: Circuit(1).T(0),
    lambda: Circuit(1).Sdg(0),
    lambda: Circuit(1).Tdg(0),
    lambda: Circuit(1).Rx(0.3, 0),
    lambda: Circuit(1).Ry(0.4, 0),
    lambda: Circuit(1).Rz(0.5, 0),
    lambda: Circuit(1).U1(0.6, 0),
    lambda: Circuit(3).CX(0, 2),
    lambda: Circuit(3).CX(2, 0),
    lambda: Circuit(3).CZ(0, 2),
    lambda: Circuit(3).CY(2, 0),
    lambda: Circuit(3).CRz(0.3, 0, 2),
    lambda: Circuit(3).CRx(0.3, 2, 0),
    lambda: Circuit(3).CRy(0.3, 0, 2),
    lambda: Circuit(3).CU1(0.3, 2, 0),
    lambda: Circuit(3).SWAP(0, 2),
    lambda: _implicit_swap(Circuit(3).SWAP(0, 2).CX(0, 1)),
    lambda: Circuit(2).H(0).CX(0, 1).Rz(0.3, 1).add_barrier([0, 1]).add_phase(
        0.37),
])
def test_from_tket_unitary(build):
    tk_circ = build()
    n = tk_circ.n_qubits
    diagram = channel.Diagram.from_tket(tk_circ)
    assert qubits.Circuit(tk_circ) == diagram
    assert (diagram.dom, diagram.cod) == (
        channel.qubit ** n, channel.qubit ** n)
    unitary = tk_circ.get_unitary()
    assert np.allclose(diagram.to_pyzx().to_matrix(), unitary)
    plus = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    psi = unitary @ plus
    expected_dm = np.outer(psi, psi.conj()).reshape((2,) * (2 * n))
    axes = [i + j * n for i in range(n) for j in (0, 1)]
    actual_dm = np.asarray(
        (qubits.Ket("+") ** n >> diagram).double().to_tensor().eval().array)
    assert np.allclose(actual_dm, np.transpose(expected_dm, axes))


def test_from_tket_measure():
    from discopy.optyx import classical
    ghz = Circuit(3).H(0).CX(0, 1).CX(1, 2)
    ghz.measure_all()
    diagram = channel.Diagram.from_tket(ghz)
    dom = channel.qubit ** 3 @ channel.bit ** 3
    assert (diagram.dom, diagram.cod) == (dom, dom)
    evaluated = (
        qubits.Ket(0) ** 3 @ classical.Bit(0, 0, 0)
        >> diagram
        >> qubits.Discard(3) @ channel.bit ** 3
    ).double().to_tensor().eval().array
    expected = np.zeros((2, 2, 2))
    expected[0, 0, 0] = expected[1, 1, 1] = 0.5
    assert np.allclose(evaluated, expected)

    partial = Circuit(3, 2).X(2).H(0).CX(0, 1).Measure(2, 0).Measure(0, 1)
    partial_diag = qubits.Circuit(partial)
    partial_dom = channel.qubit ** 3 @ channel.bit ** 2
    assert (partial_diag.dom, partial_diag.cod) == (partial_dom, partial_dom)
    probs = (
        qubits.Ket(0) ** 3 @ classical.Bit(1, 0)
        >> partial_diag
        >> qubits.Discard(3) @ channel.bit ** 2
    ).double().to_tensor().eval().array
    assert np.allclose(probs, [[0.0, 0.0], [0.5, 0.5]])


def test_from_tket_unsupported():
    from discopy.quantum import circuit as qcirc, gates as qgates, tk as qtk
    with pytest.raises(NotImplementedError, match="Post-selection"):
        channel.Diagram.from_tket(
            qtk.Circuit(1, 1).Measure(0, 0).post_select({0: 0}))
    with pytest.raises(NotImplementedError, match="post-processing"):
        channel.Diagram.from_tket(
            qtk.Circuit(1, 1).Measure(0, 0).post_process(
                qgates.Discard(qcirc.bit)))
    with pytest.raises(NotImplementedError, match="Reset"):
        channel.Diagram.from_tket(Circuit(1).Reset(0))
    with pytest.raises(NotImplementedError, match="CCX"):
        channel.Diagram.from_tket(Circuit(3).CCX(0, 1, 2))

    from sympy.abc import x
    sym = channel.Diagram.from_tket(Circuit(1).Rz(x, 0))
    assert np.allclose(
        sym.lambdify(x)(0.5).to_pyzx().to_matrix(),
        Circuit(1).Rz(0.5, 0).get_unitary(),
    )


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


def test_discard_qubits():
    a = (qubits.Discard(2).double().to_tensor().to_quimb() ^ ...).data
    b = ((qubits.Z(1, 1)**2 >> qubits.Discard(2))
         .double().to_tensor().to_quimb() ^ ...).data
    assert np.allclose(a, b)


def test_bit_flip_error():
    prob = 0.43
    a = (qubits.Z(0, 1) >> qubits.BitFlipError(prob)).eval().tensor.array
    assert np.allclose(a, np.array([[-.14, -0.14], [-0.14, -0.14]]))


def test_dephasingerror():
    prob = 0.43
    a = (qubits.Z(0, 1) >> qubits.DephasingError(prob)).eval().tensor.array
    assert np.allclose(a, np.array([[-.14, 1.], [1, -0.14]]))


def test_ket():
    a = (qubits.Ket(1).get_kraus().to_tensor().to_quimb() ^ ...).data
    b = zx.X(0, 1, 0.5) @ diagram.Scalar(1 / np.sqrt(2))

    b = (b.to_tensor().to_quimb() ^ ...).data
    assert np.allclose(a, b)


def test_bra():
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
    assert qubits.Circuit(gates.Discard(circuit.qubit)) == qubits.Discard(1)
    assert qubits.Circuit(gates.Discard(circuit.bit)) == channel.Discard(
        channel.bit)
    assert qubits.Circuit(gates.Measure(2)) == qubits.Measure(2)
    with pytest.raises(TypeError, match="Unsupported object type"):
        qubits.Circuit(circuit.Id(circuit.Ty(circuit.Qudit(3))))
    with pytest.raises(NotImplementedError, match="Encode"):
        qubits.Circuit(circuit.Encode())
    with pytest.raises(NotImplementedError, match="Unsupported gate"):
        qubits.Circuit(gates.CCX)


@pytest.mark.parametrize("gate", [
    CX, CY, CZ, SWAP, Rz(.3), Rx(.3), Ry(.3), U1(.3),
    CRz(.3), CRx(.3), CU1(.3), Controlled(Ry(.3)),
    CRz(.3, distance=-1), CRx(.3, distance=-1), CU1(.3, distance=-1),
    CRz(.3, distance=2), CU1(.3, distance=2), Controlled(U1(.3), distance=2),
    S.dagger(), T.dagger(),
], ids=str)
def test_discopy_gates_keep_their_unitary_up_to_a_global_phase(gate):
    n = 2 ** len(gate.dom)
    expected = np.asarray(gate.eval().array, dtype=complex).reshape(n, n)
    actual = np.asarray(
        qubits.Circuit(gate).get_kraus().to_tensor().eval().array,
        dtype=complex).reshape(n, n)
    phase = np.vdot(actual, expected) / n
    assert np.isclose(abs(phase), 1)
    assert np.allclose(phase * actual, expected)


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


@pytest.mark.parametrize("spider", [qubits.Z, qubits.X])
@pytest.mark.parametrize("phase", [0, .25])
def test_scalar_spider_decomposition(spider, phase):
    """A spider with no legs is a scalar, and so is its decomposition:
    by fusion it is the state on one wire followed by the effect on it."""
    decomp = spider(0, 0, phase).decomp()
    assert decomp.dom == decomp.cod == qubits.qubit ** 0
    assert np.allclose(
        decomp.double().to_tensor().eval().array,
        (spider(0, 1, phase) >> spider(1, 0)
         ).double().to_tensor().eval().array)


@pytest.mark.parametrize("spider", [qubits.Z, qubits.X])
@pytest.mark.parametrize("phase", [0, .25, .5])
def test_scalar_spider_evaluates(spider, phase):
    """A spider with no legs is a scalar, so it evaluates to shape () and to
    the same number as its decomposition and as the state-effect pair it
    fuses from. Its truncation used to raise `cannot reshape array of size 2
    into shape ()`, the legless box carrying both amplitudes."""
    box = spider(0, 0, phase)
    value = box.double().to_tensor().eval().array
    assert value.shape == ()
    assert np.allclose(value, box.decomp().double().to_tensor().eval().array)
    assert np.allclose(value, (
        spider(0, 1, phase) >> spider(1, 0)
    ).double().to_tensor().eval().array)


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


@pytest.mark.parametrize("n, m, phase", [
    (0, 1, 0), (0, 1, .5), (1, 0, 0), (1, 0, -.5),
    (1, 1, .25), (1, 1, .75), (1, 1, -.25)])
def test_x_dual_rail_commutes_with_the_encoding(n, m, phase):
    """Encoding then applying the dual-rail translation of an X spider is
    applying the spider then encoding, scalars included."""
    from discopy.optyx.photonic import DualRail
    spider = qubits.X(n, m, phase)
    left = DualRail(n) >> spider.dual_rail() if n else spider.dual_rail()
    right = spider >> DualRail(m) if m else spider
    assert np.allclose(left.get_kraus().to_tensor().eval().array,
                       right.get_kraus().to_tensor().eval().array)


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


@pytest.mark.parametrize("box", [
    qubits.Measure(2),
    qubits.Discard(2),
    qubits.Encode(2),
    qubits.Z(1, 2),
    qubits.Z(1, 2, 0.25),
    qubits.X(2, 1, 0.5),
    qubits.H(),
    qubits.Scalar(0.5),
    qubits.Scalar(0.5j),
    qubits.BitFlipError(0.1),
    qubits.DephasingError(0.2),
    qubits.Ket(0),
    qubits.Ket(1),
    qubits.Ket("+"),
    qubits.Ket("-"),
    qubits.Ket(0, channel.bit),
    qubits.Bra(0),
    qubits.Bra(1),
    qubits.Bra("+"),
    qubits.Bra("-"),
    qubits.Bra(1, channel.bit),
    qubits.QubitChannel(
        "custom", diagram.Swap(diagram.bit, diagram.bit),
        channel.qubit ** 2, channel.qubit ** 2),
    qubits.QubitChannel(
        "env_ch", diagram.Swap(diagram.bit, diagram.bit),
        channel.qubit ** 2, channel.qubit,
        env=diagram.Ty(diagram.Ob("bit"))),
    pytest.param(
        qubits.QubitChannel(
            "custom", zx.Z(1, 1, 0.25), channel.qubit, channel.qubit),
        marks=pytest.mark.skipif(
            "from_tree" not in zx.Spider.__dict__,
            reason="waits on zx.Spider.from_tree (#731)")),
])
def test_qubits_boxes_load_their_own_tree(box):
    from discopy.utils import dumps, loads
    loaded = loads(dumps(box))
    assert loaded == box
    assert loaded.to_tree() == box.to_tree()
    assert loaded.double() == box.double()
    assert loads(dumps(box @ box)) == box @ box
