# -*- coding: utf-8 -*-

from unittest.mock import Mock

import pytest

pytest.importorskip("pytket")

import numpy as np
from pytest import raises

from discopy.quantum import tk
from discopy.quantum.circuit import Swap
from discopy.quantum.gates import *
from discopy.tensor import Tensor, Dim


def test_Circuit_to_tk():
    bell_state = Circuit.caps(qubit, qubit)
    bell_effect = bell_state[::-1]
    snake = (bell_state @ qubit >> qubit @ bell_effect)[::-1]
    tk_circ = snake.to_tk()
    assert repr(tk_circ) ==\
        'tk.Circuit(3, 2)'\
        '.H(1)'\
        '.CX(1, 2)'\
        '.CX(0, 1)'\
        '.Measure(1, 1)'\
        '.H(0)'\
        '.Measure(0, 0)'\
        '.post_select({0: 0, 1: 0})'\
        '.scale(4)'
    assert repr((CX >> Measure(2) >> Swap(bit, bit)).to_tk())\
        == "tk.Circuit(2, 2).CX(0, 1).Measure(1, 0).Measure(0, 1)"
    assert repr((Bits(0) >> Id(bit) @ Bits(0)).to_tk())\
        == "tk.Circuit(0, 2)"
    assert "Swap" in repr((Bra(0) @ Bits(0) >> Bits(0) @ Id(bit)).to_tk())


def test_Sum_from_tk():
    assert Circuit.from_tk(*(X + X).to_tk()) == (X + X).init_and_discard()
    assert Circuit.from_tk() == Sum([], qubit ** 0, qubit ** 0)


def test_tk_err():
    with raises(TypeError):
        Circuit.from_tk("foo")
    with raises(NotImplementedError):
        QuantumGate("foo", qubit, qubit, [1, 2, 3, 4]).to_tk()
    with raises(NotImplementedError):
        Bits(1).to_tk()
    with raises(NotImplementedError):
        Circuit.from_tk(tk.Circuit(3).CSWAP(0, 1, 2))



def test_Circuit_from_tk():
    def back_n_forth(f):
        return Circuit.from_tk(f.to_tk())

    m = Measure(1, destructive=False, override_bits=True)
    assert back_n_forth(m) == m.init_and_discard()
    assert back_n_forth(CRx(0.5)) ==\
        Ket(0) @ Ket(0) >> CRx(0.5) >> Discard() @ Discard()
    assert back_n_forth(CRz(0.5)) ==\
        Ket(0) @ Ket(0) >> CRz(0.5) >> Discard() @ Discard()
    assert Id(qubit @ bit).init_and_discard()\
        == back_n_forth(Swap(qubit, bit)) == back_n_forth(Swap(bit, qubit))
    c = (T >> T.dagger()).init_and_discard()
    assert c == back_n_forth(c)


def test_controlled_gates_to_tk():
    gates = [CY, CCX, U1(.25), CU1(.25), Controlled(Ry(.25))]
    for gate in gates:
        assert Circuit.from_tk(gate.to_tk()) == gate.init_and_discard()
    gates += [
        Controlled(CX, distance=-1),
        Controlled(Controlled(X, distance=-1), distance=2)]
    for gate in gates:
        unitary = gate.to_tk().get_unitary()
        assert np.allclose(unitary, gate.eval().array.reshape(unitary.shape).T)
    with raises(NotImplementedError):
        CCZ.to_tk()


def test_tk_round_trip_is_numerical():
    circuits = [
        Ket(0, 0) >> Ry(.2) @ Rx(.7) >> CRz(.3) >> CRx(.4) >> H @ U1(.1)
        >> CU1(.6) >> H @ H >> Measure(2),
        Ket(0, 0, 0) >> Ry(.2) @ Rx(.7) @ H >> Circuit.permutation([2, 0, 1])
        >> Controlled(Ry(.4), distance=2) >> qubit @ CU1(.3, distance=-1)
        >> H @ H @ H >> Measure(3)]
    for circuit in circuits:
        assert np.allclose(
            Circuit.from_tk(circuit.to_tk()).eval().array,
            circuit.eval().array)


def test_mixed_permutation_to_tk():
    measure = Ket(1, 0, 0) >> Measure() @ qubit @ Measure()
    swap_bits = Circuit.permutation([2, 1, 0], bit @ qubit @ bit)
    swap_qubits = Circuit.permutation([2, 1, 0], qubit @ bit @ qubit)
    assert repr(measure.to_tk())\
        == "tk.Circuit(3, 2).Measure(2, 1).X(0).Measure(0, 0)"
    assert repr((measure >> swap_bits).to_tk())\
        == "tk.Circuit(3, 2).Measure(2, 0).X(0).Measure(0, 1)"
    assert repr((Ket(1, 0, 0) >> qubit @ Measure() @ qubit >> swap_qubits
                 >> X @ bit @ qubit).to_tk())\
        == "tk.Circuit(3, 1).Measure(1, 0).X(0).X(2)"
    post = ClassicalGate('post', bit ** 2, bit ** 2, np.eye(4).flatten())
    circuit = Ket(0, 0, 0) >> H @ qubit ** 2 >> Measure(2) @ qubit\
        >> post @ qubit >> Swap(bit, bit) @ qubit
    backend = tk.mockBackend({(0, 0): 512, (1, 0): 512})
    assert np.allclose(
        circuit.eval(backend=backend).array,
        circuit.init_and_discard().eval().array.real)


def test_Discard_bit_to_tk():
    circuit = Ket(1, 0) >> Measure(2) >> Discard(bit) @ bit
    assert circuit.to_tk().post_processing == Discard(bit) @ bit
    backend = tk.mockBackend({(1, 0): 1024})
    assert circuit.eval(backend=backend)\
        == Tensor[float]([1, 0], Dim(1), Dim(2))
    backend = tk.mockBackend({(1, 0): 1024})
    assert (circuit >> Bits(0).dagger()).eval(backend=backend)\
        == Tensor[float]([1], Dim(1), Dim(1))


def test_ClassicalGate_to_tk():
    post = ClassicalGate('post', bit ** 2, bit ** 0, data=[0, 0, 0, 1])
    assert (post[::-1] >> Swap(bit, bit)).to_tk().post_processing\
        == post[::-1] >> Swap(bit, bit)
    circuit = sqrt(2) @ Ket(0, 0) >> H @ Rx(0) >> CX >> Measure(2) >> post
    assert Circuit.from_tk(circuit.to_tk())[-1] == post


def test_tk_dagger():
    assert S.dagger().to_tk() == tk.Circuit(1).Sdg(0)
    assert T.dagger().to_tk() == tk.Circuit(1).Tdg(0)


def test_Circuit_get_counts_snake():
    compilation = Mock()
    compilation.apply = lambda x: x
    backend = tk.mockBackend({
        (0, 0): 240, (0, 1): 242, (1, 0): 271, (1, 1): 271})
    scaled_bell = Circuit.caps(qubit, qubit)
    snake = scaled_bell @ qubit >> qubit @ scaled_bell[::-1]
    result = np.round(snake.eval(
        backend=backend, compilation=compilation, measure_all=True).array)
    assert result == 1


def test_Circuit_get_counts_empty():
    assert not Id(qubit).get_counts(backend=tk.mockBackend({}))



def test_Bra_and_Measure_to_tk():
    boxes = [
        Ket(0), Rx(0.552), Rz(0.512), Rx(0.917), Ket(0, 0, 0), H, H, H,
        CRz(0.18), CRz(0.847), CX, H, sqrt(2), Bra(0, 0), Ket(0),
        Rx(0.446), Rz(0.256), Rx(0.177), CX, H, sqrt(2), Bra(0, 0), Measure()]
    offsets=[
        0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 2, 2, 3, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    c = Circuit.decode(qubit ** 0, zip(boxes, offsets))
    assert repr(c.to_tk()) ==\
        "tk.Circuit(5, 5)"\
        ".Rx(0.892, 0)"\
        ".H(1)"\
        ".H(2)"\
        ".H(3)"\
        ".Rx(1.104, 4)"\
        ".Rz(0.512, 0)"\
        ".CRz(0.36, 1, 2)"\
        ".Rz(1.024, 4)"\
        ".Rx(0.354, 0)"\
        ".CRz(1.694, 2, 3)"\
        ".Rx(1.834, 4)"\
        ".Measure(2, 4)"\
        ".CX(0, 1)"\
        ".CX(3, 4)"\
        ".Measure(4, 1)"\
        ".Measure(1, 3)"\
        ".H(0)"\
        ".H(3)"\
        ".Measure(3, 0)"\
        ".Measure(0, 2)"\
        ".post_select({0: 0, 1: 0, 2: 0, 3: 0})"\
        ".scale(4)"


def test_ClassicalGate_eval():
    backend = tk.mockBackend({
        (0, 0): 256, (0, 1): 256, (1, 0): 256, (1, 1): 256})
    post = ClassicalGate('post', bit ** 2, bit ** 0, [1, 0, 0, 0])
    assert post.eval(backend=backend) == Tensor[float]([0.25], Dim(1), Dim(1))
