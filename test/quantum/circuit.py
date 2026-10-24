# -*- coding: utf-8 -*-

import pytest

for backend in ("sympy", "tensornetwork"):
    pytest.importorskip(backend)

import numpy as np
import sympy
import tensornetwork as tn
from pytest import raises

from discopy.quantum.channel import *
from discopy.quantum.circuit import *
from discopy.quantum.gates import *
from discopy.quantum.ansatze import IQPansatz
from discopy import rigid


def test_index2bitstring():
    with raises(ValueError):
        index2bitstring(1, 0)
    assert index2bitstring(42, 8) == (0, 0, 1, 0, 1, 0, 1, 0)


def test_bitstring2index():
    assert bitstring2index((0, 0, 1, 0, 1, 0, 1, 0)) == 42


def test_Circuit_eval():
    with raises(ValueError):
        Box('f', qubit, qubit).eval()
    assert MixedState().eval() == Discard().eval().dagger()


def test_Circuit_cups_and_caps():
    assert Circuit.cups(bit, bit) == Match() >> Discard(bit)
    assert Circuit.caps(bit, bit) == MixedState(bit) >> Copy()


def test_Circuit_spiders():
    assert Circuit.spiders(123, 456, qubit ** 0) == Id()
    assert Circuit.spiders(0, 0, qubit) == (sqrt(2) >> Ket(0)
                                            >> H >> H
                                            >> Bra(0) >> sqrt(2))
    assert Circuit.spiders(1, 1, qubit) == Id(qubit)
    assert Circuit.spiders(0, 1, qubit ** 2) == ((sqrt(2) >> Ket(0) >> H)
                                                 @ (sqrt(2) >> Ket(0) >> H))

    swap = Swap(qubit, qubit)  # spiders plumb their wires, they do not SWAP
    assert Circuit.spiders(2, 1, qubit ** 2) == Circuit.decode(
        qubit @ qubit @ qubit @ qubit,
        zip([swap, CX, Bra(0), CX, Bra(0)], [1, 0, 1, 1, 2]))

    boxes = [
        sqrt(2), Ket(0), H, Ket(0), CX, Ket(0), CX,
        sqrt(2), Ket(0), H, Ket(0), CX, Ket(0), CX,
        swap, swap, swap]
    offsets = [0, 0, 0, 1, 0, 1, 0, 3, 3, 3, 4, 3, 4, 3, 2, 1, 3]
    ghz2 = Circuit.decode(Ty(), zip(boxes, offsets))
    assert Circuit.spiders(0, 3, qubit ** 2).eval() == ghz2.eval()

    assert np.isclose(np.abs(Circuit.spiders(0, 0, qubit).eval().array), 2)

    combos = [(2, 3), (5, 4), (0, 1)]
    for n_legs_in, n_legs_out in combos:
        flat_tensor = np.abs(Circuit.spiders(n_legs_in, n_legs_out, qubit)
                             .eval().array.flatten())
        assert np.isclose(flat_tensor[0], 1) and np.isclose(flat_tensor[-1], 1)


def test_bra_ket_inputs():
    bad_inputs = ['0', '1', 2]
    for box in [Bra, Ket]:
        for bad_input in bad_inputs:
            with raises(Exception):
                box(bad_input)


def test_Circuit_get_counts():
    assert Id(qubit).get_counts() == {(): 1.0}


def test_Circuit_conjugate():
    assert (Rz(0.1) >> H).conjugate() == Rz(-0.1) >> H


def test_Circuit_measure():
    assert Id().measure() == 1
    assert all(Bits(0).measure(mixed=True) == np.array([1, 0]))
    assert np.allclose(Copy().measure(), [[1, 0], [0, 0]])
    circuit = Ket(0, 0, 0) >> H @ H @ H
    assert np.allclose(circuit.measure(), np.ones((2, 2, 2)) / 8)
    assert np.allclose(
        circuit.measure(), np.absolute(circuit.eval().array) ** 2)


def test_Box():
    with raises(TypeError):
        Box('f', rigid.Ty('x'), bit)
    with raises(TypeError):
        Box('f', bit, rigid.Ty('x'))


def test_pure_Box():
    with raises(ValueError):
        Box('f', bit, qubit, is_mixed=False)


def test_pure_Box_stays_pure():
    quantum = Box('f', qubit ** 2, qubit, data=list(range(8)), is_mixed=False)
    classical = Box('g', bit, bit, data=[1, 2, 3, 4], is_mixed=False)
    for box in (quantum, classical):
        for rebuilt in (box.dagger(), box.rotate(), box.delay()):
            assert not rebuilt.is_mixed
        assert box.dagger().eval().is_close(box.eval().dagger())
        assert box.rotate().eval().is_close(box.eval().r)
        assert box.rotate(left=True) == box.rotate()
        assert box.rotate(left=True).rotate() == box


def test_gate_hash():
    """
    Quantum gates carry their matrix as data, which numpy makes unhashable.
    They must nevertheless be hashable and usable as functor keys, with a hash
    that does not depend on hypergraph equality, see
    https://github.com/discopy/discopy/pull/387
    """
    assert hash(Rx(0.25)) == hash(Rx(0.25)) and Rx(0.25) == Rx(0.25)
    # A data-carrying gate can be stored and looked up in a dictionary.
    assert {Rx(0.25): 42}[Rx(0.25)] == 42
    assert X in {X} and hash(X) == hash(X)


def test_Swap():
    assert Swap(bit, qubit).is_mixed
    assert Swap(bit, bit).eval(mixed=True) == Channel.swap(C(Dim(2)), C(Dim(2)))


def test_Discard():
    assert Discard().dagger() == MixedState()
    assert MixedState().dagger() == Discard()
    import discopy
    for box in [Discard(), Discard(bit @ qubit), MixedState(bit)]:
        assert eval(repr(box), {'quantum': discopy.quantum}) == box


def test_Measure():
    assert Measure(destructive=False, override_bits=True).dagger()\
        == Encode(constructive=False, reset_bits=True)
    assert Encode().dagger() == Measure()
    import discopy
    for box in [Measure(), Measure(2, destructive=False, override_bits=True),
                Encode(), Encode(2, constructive=False, reset_bits=True),
                Measure(2).delay(), Encode(reset_bits=True).delay(2)]:
        assert eval(repr(box), {'quantum': discopy.quantum}) == box


def test_ClassicalGate():
    f = ClassicalGate('f', bit, bit, [0, 1, 1, 0])
    assert repr(f.dagger())\
        == "quantum.gates.ClassicalGate('f', "\
           "quantum.circuit.Ty(quantum.circuit.Digit(2)), "\
           "quantum.circuit.Ty(quantum.circuit.Digit(2)), "\
           "data=[0, 1, 1, 0]).dagger()"


def test_Digits():
    with raises(TypeError):
        Digits()
    d = Digits(0, 1, 2, dim=3)
    assert d.digits == [0, 1, 2]
    assert d.dagger().dagger() == d


def test_Bits():
    assert repr(Bits(0).dagger()) == "quantum.gates.Bits(0).dagger()"
    assert Bits(0).dagger().dagger() == Bits(0)


def test_Rx():
    assert repr(Rx(0.4)) == "quantum.gates.Rx(0.4)"
    assert Rx(0).eval() == Rx(0).dagger().eval() == Id(qubit).eval()


def test_Ry():
    assert Ry(0).eval() == Id(qubit).eval()


def test_Rz():
    assert Rz(0).eval() == Id(qubit).eval()


def test_CRz():
    assert CRz(0).eval() == Id(qubit ** 2).eval()


def test_CU1():
    assert CU1(0).eval() == Id(qubit ** 2).eval()


def test_CRx():
    assert CRx(0).eval() == Id(qubit ** 2).eval()


def test_Sum():
    zero, one = Sum((), qubit, qubit), Id(qubit)
    assert zero.eval() + one.eval() == (zero + one).eval() == one.eval()
    assert Sum([Id()]).get_counts() == Id().get_counts()
    assert (Id() + Id()).get_counts()[()] == 2
    assert (Id() + Id()).eval() == Id().eval() + Id().eval()
    assert not (X + X).is_mixed and (X >> Bra(0) + Discard()).is_mixed
    assert (Discard() + Discard()).eval()\
        == Discard().eval() + Discard().eval()
    assert Sum([Id(qubit)]).eval() == Id(qubit).eval()


def test_subs():
    from sympy.abc import phi
    assert (Rz(phi) + Rz(phi + 1)).subs(phi, 1) == Rz(1) + Rz(2)
    circuit = sqrt(2) @ Ket(0, 0) >> H @ Rx(phi) >> CX >> Bra(0, 1)
    assert circuit.subs(phi, 0.5)\
        == sqrt(2) @ Ket(0, 0) >> H @ Rx(0.5) >> CX >> Bra(0, 1)


def test_lambdify():
    from sympy.abc import phi
    assert list(Rz(phi).lambdify(phi)(0).array.flatten()) == [1, 0, 0, 1]


def _to_square_mat(m):
    m = np.asarray(m).flatten()
    return m.reshape(2 * (int(np.sqrt(len(m))), ))


def test_grad_basic():
    from sympy.abc import phi
    for constant in (Rz(0), CU1(1), CRz(0), CRx(1), scalar(1.23), Measure()):
        zero = constant.grad(phi)
        assert isinstance(zero, Sum) and not zero.terms
        assert (zero.dom, zero.cod) == (constant.dom, constant.cod)

    assert scalar(2 * phi).grad(phi).eval().array == 2
    assert (scalar(2 * phi) + scalar(3 * phi)).grad(phi).eval().array == 5

    with raises(NotImplementedError):
        Box("dummy box", qubit, qubit, data=phi).grad(phi)


def _assert_is_close_to_iden(m):
    m = _to_square_mat(m)
    assert np.isclose(np.linalg.norm(m - np.eye(len(m))), 0)


def _assert_is_close_to_0(m):
    if isinstance(m, Circuit):
        assert m.dom == m.cod
        m = m.eval().array
    m = np.asarray(m).flatten()
    assert np.isclose(np.linalg.norm(m), 0)


def test_testing_utils():
    for k in range(1, 4):
        _assert_is_close_to_iden(np.eye(k))
        _assert_is_close_to_iden(Id(qubit ** k))
        _assert_is_close_to_0(np.zeros(k))
        _assert_is_close_to_0(Ket(*([0] * k)) >> Bra(*([1] * k)))


def test_rot_grad():
    from sympy import pi
    from sympy.abc import phi
    x = 0.7
    crz_diff = CRz(phi).grad(phi, mixed=False).lambdify(phi)(x).eval()
    crz_res = (
        (CRz(phi) >> Z @ Z @ scalar(0.5j * pi))
        + (CRz(phi) >> Id(qubit) @ Z @ scalar(-0.5j * pi))
    ).lambdify(phi)(x).eval()
    assert np.allclose(crz_diff.array, crz_res.array)

    crx_diff = CRx(phi).grad(phi, mixed=False).lambdify(phi)(x).eval()
    crx_res = (
        (CRx(phi) >> Z @ X @ scalar(0.5j * pi))
        + (CRx(phi) >> Id(qubit) @ X @ scalar(-0.5j * pi))
    ).lambdify(phi)(x).eval()
    assert np.allclose(crx_diff.array, crx_res.array)


def test_rot_grad_NotImplemented():
    from sympy.abc import z
    with raises(NotImplementedError):
        CU1(z).grad(z, mixed=True)


def test_Copy_Match():
    assert Match().dagger() == Copy() and Copy().dagger() == Match()


def test_Sum_get_counts():
    assert Sum([], qubit, qubit).get_counts() == {}


def test_Controlled():
    with raises(TypeError):
        Controlled(None)
    with raises(ValueError):
        Controlled(X, distance=0)
    assert np.allclose(CU1(0.5, distance=2).eval().array,
                       Controlled(U1(0.5), distance=2).eval().array)
    assert np.allclose(CRz(0.7, distance=-2).eval().array,
                       Controlled(Rz(0.7), distance=-2).eval().array)


def test_adjoint():
    n, s = map(rigid.Ty, 'ns')
    Bob = rigid.Box('Bob', rigid.Ty(), n)
    eats = rigid.Box('eats', rigid.Ty(), n.r @ s)
    diagram = Bob @ eats >> rigid.Cup(n, n.r) @ rigid.Id(s)

    func_ob = {n: qubit, s: qubit}
    func_ar = {Bob: Ket(0), eats: Ket(1, 1)}
    F = rigid.Functor(ob_map=func_ob, ar_map=func_ar, cod=Circuit)

    assert F(diagram.transpose_box(0, left=True).normal_form())\
        == Circuit.decode(Ty(), zip([Ket(1, 1), Bra(0)], [0, 0]))

    gates = [
        Bra(0), Ket(0, 0), Rx(0.1), Ry(0.2), Rz(0.3),
        CU1(0.4), CRx(0.5), CRz(0.7), Scalar(1 + 2j), CX,
        Swap(bit, qubit), Copy(), Match()
    ]

    gates_conj = [
        Bra(0), Ket(0, 0), Rx(-0.1), Ry(0.2), Rz(-0.3),
        Controlled(U1(-0.4), distance=-1),
        Controlled(Rx(-0.5), distance=-1),
        Controlled(Rz(-0.7), distance=-1),
        Scalar(1 - 2j),
        Controlled(X, distance=-1),
        Swap(qubit, bit), Copy(), Match()
    ]

    for g, g_conj in zip(gates, gates_conj):
        assert g.conjugate() == g_conj


def test_causal_cx():
    assert np.allclose((CX >> Discard(2)).eval().array,
                       Discard(2).eval().array)


def test_eval_no_qutrits():
    qutrit = Ty(Qudit(3))
    with raises(Exception):
        Box('qutrit box', qutrit, qutrit).to_tn(mixed=True)


def test_grad_unknown_controlled():
    from sympy.abc import phi
    unknown = QuantumGate('gate', qubit, qubit, data=phi)
    with raises(NotImplementedError):
        Controlled(unknown).grad(phi)


def test_controlled_subs():
    from sympy.abc import phi, psi
    assert CRz(phi).subs(phi, 0.1) == CRz(0.1)
    assert CRx(psi).l.subs((phi, 0.1), (psi, 0.2)) == CRx(0.2).l


def test_circuit_chaining():
    circuit = Id(qubit ** 5).CX(0, 2).X(4).CRz(0.2, 4, 2).H(2)
    boxes = [Controlled(X, distance=2), X, Controlled(Rz(0.2), distance=-2), H]
    offsets = [0, 4, 2, 2]
    expected_circuit = Circuit.decode(qubit ** 5, zip(boxes, offsets))
    assert circuit == expected_circuit

    circuit = Id(qubit ** 3).CZ(0, 1).CX(2, 0).CCX(1, 0, 2).CCZ(2, 0, 1).X(0).Y(1).Z(2)
    boxes = [
        CZ, Controlled(X, distance=-2), Controlled(CX, distance=1),
        Controlled(CZ, distance=-1), X, Y, Z]
    offsets = [0, 0, 0, 0, 0, 1, 2]
    assert circuit == Circuit.decode(qubit ** 3, zip(boxes, offsets))

    circuit = Id(qubit).Rx(0.1, 0).Ry(0.2, 0).Rz(0.3, 0)
    expected_circuit = Rx(0.1) >> Ry(0.2) >> Rz(0.3)
    assert circuit == expected_circuit

    assert Id(qubit).S(0) == S

    with raises(ValueError):
        Id(qubit ** 3).CY(0, 0)
    with raises(IndexError):
        Id(qubit).CRx(0.7, 1, 0)
    with raises(IndexError):
        Id(qubit ** 2).X(999)


@pytest.mark.parametrize('x,y', [(0, 1), (0, 2), (1, 0), (2, 0), (5, 0)])
def test_CX_decompose(x, y):
    n = abs(x - y) + 1
    binary_mat = np.eye(n, dtype=int)
    binary_mat[y] = np.bitwise_xor(binary_mat[x], binary_mat[y])

    N = 1 << n
    unitary_mat = np.zeros(shape=(N, N))
    for i in range(N):
        bits = index2bitstring(i, n)
        v = bitstring2index(binary_mat @ bits % 2)
        unitary_mat[i][v] = 1

    # take transpose because tensor axes follow diagrammatic order
    out = Id(n).CX(x, y).eval().array.reshape(N, N).T
    # but CX matrices are self transpose
    assert (out == out.T).all()
    assert (out == unitary_mat).all()


@pytest.mark.parametrize('x,y, z', [(0, 1, 2), (0, 2, 4),
                                    (0, 4, 2), (4, 2, 0),
                                    (0, 4, 1), (4, 0, 1)])
def test_CCX_decompose(x, y, z):

    n = max(x, y, z) - min(x, y, z) + 1
    N = 1 << n

    unitary_mat = np.zeros(shape=(N, N))

    for i in range(N):
        bits = list(index2bitstring(i, n))
        bits[z] = (bits[x] & bits[y]) ^ bits[z]
        v = bitstring2index(bits)
        unitary_mat[i][v] = 1

    # take transpose because tensor axes follow diagrammatic order
    out = Id(n).CCX(x, y, z).eval().array.reshape(N, N).T

    assert (out == unitary_mat).all()


def test_loads_dumps():
    from discopy.utils import loads, dumps
    assert loads(dumps(Rx(1))) == Rx(1)


mixed_circuits = [
    (Copy() >> Encode(2) >> CX >> Rx(0.3) @ Rz(0.3)
        >> SWAP >> Measure() @ Discard()),
    (Copy() @ Id(bit) >> Id(bit @ bit) @ Copy() >> Encode(4)
        >> H @ qubit ** 3 >> qubit @ CX @ qubit >> qubit ** 3 @ Rz(0.4)
        >> CX @ qubit ** 2 >> qubit ** 2 @ CX >> Rx(0.3) @ qubit ** 3
        >> qubit @ CX @ qubit >> qubit ** 3 @ H >> Measure(4)),
    Ket(0, 0, 0, 0) >> Discard(2) @ Measure(2) @ sqrt(2),
    Circuit.swap(bit, bit) @ (MixedState(2) >> SWAP),
    Measure() >> Copy() >> Match() >> Encode(),
    Bits(1, 0) >> Discard(bit) @ bit >> MixedState(bit) @ bit,
    H @ Ry(0.3) >> Measure(2, destructive=False),
    Ket(0) @ Bits(1) >> Measure(override_bits=True),
    Encode(2, constructive=False, reset_bits=True)
]

pure_circuits = [
    H >> X >> Y >> Z,
    CX >> H @ Rz(0.5),
    CRz(0.123) >> Z @ Z,
    CX >> H @ qubit >> Bra(0, 0),
    IQPansatz(3, [[0.1, 0.2], [0.3, 0.4]]),
    IQPansatz(3, [[0.1, 0.2], [0.3, 0.4]]).l.dagger(),
    Circuit.permutation([1, 2, 0])
]

contractor = tn.contractors.auto


def is_close_smallno(a, b):
    return a.is_close(b, rtol=1.e-15, atol=1.e-15)


@pytest.mark.parametrize('c', pure_circuits + mixed_circuits)
def test_mixed_eval(c):
    assert is_close_smallno(c.eval(contractor=contractor), c.eval())


@pytest.mark.parametrize('c', pure_circuits)
def test_consistent_eval(c):
    pure_result = c.eval(mixed=False, contractor=contractor)
    mixed_result = c.eval(mixed=True, contractor=contractor)

    doubled_result = (pure_result
                      @ pure_result.conjugate(diagrammatic=False))
    assert is_close_smallno(doubled_result, mixed_result.to_tensor())


def test_mixed_eval_foliated():
    circuit = Ket(0, 0, 1) >> H @ X @ X >> Measure() @ Discard(qubit ** 2)
    assert [len(layer.boxes) for layer in circuit.foliation().inside]\
        == [1, 3, 2]
    assert circuit.foliation().eval(mixed=True, contractor=contractor)\
        .is_close(circuit.eval(mixed=True))


@pytest.mark.parametrize('c', mixed_circuits)
def test_pytorch_mixed_eval(c):
    pytest.importorskip("torch")
    with tn.DefaultBackend('pytorch'):
        assert is_close_smallno(c.eval(contractor=contractor), c.eval())


@pytest.mark.parametrize('c', pure_circuits)
def test_pytorch_pure_eval(c):
    pytest.importorskip("torch")
    with tn.DefaultBackend('pytorch'):
        assert is_close_smallno(c.eval(contractor=contractor), c.eval())


@pytest.mark.parametrize('c', pure_circuits)
def test_pytorch_consistent_eval(c):
    pytest.importorskip("torch")
    with tn.DefaultBackend('pytorch'):
        pure_result = c.eval(mixed=False, contractor=contractor)
        mixed_result = c.eval(mixed=True, contractor=contractor)

        doubled_result = (
            pure_result
            @ pure_result.conjugate(diagrammatic=False))
        assert is_close_smallno(doubled_result, mixed_result.to_tensor())


@pytest.mark.parametrize('c', pure_circuits)
def test_quimb_pure_eval(c):
    pytest.importorskip("quimb")
    t = c.to_quimb().contract()
    t = t.data.transpose(*np.argsort(t.inds))

    assert np.allclose(t, c.eval().array), f"{t} != {c.eval().array}"


def test_logical_vs_physical_swap():
    logical, physical = Circuit.swap(qubit, qubit), SWAP

    assert logical != physical
    assert np.allclose(logical.eval().array, physical.eval().array)

    # A logical swap is plumbing: it coalesces with the wires beside it.
    assert str(qubit @ logical)\
        == "Permutation(qubit @ qubit @ qubit, [0, 2, 1])"

    # The gate is a box: it stays whiskered and it is drawn as a crossing.
    assert str(qubit @ physical) == "qubit @ SWAP"
    assert physical.to_drawing().boxes[0].is_crossing


def test_feedback_is_the_trace():
    array = CX.eval().array
    assert np.allclose(CX.feedback().eval().array,
                       np.einsum('aibi->ab', array))
    assert np.allclose(CX.feedback(left=True).eval().array,
                       np.einsum('iaib->ab', array))
    assert CX.feedback(mem=qubit) == CX.trace(mem=qubit)


def test_delay():
    from discopy.quantum import gates
    from discopy.utils import loads, dumps
    env = Circuit.environment() | vars(gates)
    for typ in (qubit, bit, qubit @ bit.delay()):
        delayed = typ.delay()
        assert delayed != typ and delayed.delay(-1) == typ
        assert eval(repr(delayed), env) == delayed
        assert loads(dumps(delayed)) == delayed
    for gate in (CX, CCX, CRz(.5), CRx(.25, distance=-1), Rz(.5), Ry(.25),
                 Controlled(H, distance=-2), Controlled(X, distance=2)):
        delayed = gate.delay()
        assert (delayed.dom, delayed.cod) == (
            gate.dom.delay(), gate.cod.delay())
        assert delayed != gate and delayed.delay(-1) == gate
        assert delayed.delay(2) == gate.delay(3)
        assert eval(repr(delayed), env) == delayed
        assert loads(dumps(delayed)) == delayed
        assert np.allclose(delayed.eval().array, gate.eval().array)
        assert delayed.dagger() == gate.dagger().delay()
    assert CX.delay() == Controlled(X.delay())
    assert CRz(.5).delay() == Controlled(Rz(.5).delay())
    assert loads(dumps(Rz(.5).delay())) == Rz(.5).delay()
    phi = sympy.Symbol('phi')
    for gate in (Rz(phi), CRz(phi), CRx(phi, distance=2)):
        delayed = gate.delay()
        assert delayed.subs(phi, .5) == gate.subs(phi, .5).delay()
        assert delayed.lambdify(phi)(.5) == gate.lambdify(phi)(.5).delay()
        grad = delayed.grad(phi)
        assert grad.dom == gate.dom.delay() and np.allclose(
            grad.subs(phi, .5).eval().array,
            gate.grad(phi).subs(phi, .5).eval().array)
    circuit = CX >> Rz(.5) @ H
    assert np.allclose(
        circuit.delay().eval().array, circuit.eval().array)


def test_Bubble_is_mixed_and_grad():
    phi = sympy.Symbol('phi')
    assert X.bubble(cod=bit).is_mixed and not X.bubble().is_mixed
    assert X.bubble().grad(phi) == Sum((), qubit, qubit)
    with raises(NotImplementedError):
        Rx(phi).bubble().grad(phi)
