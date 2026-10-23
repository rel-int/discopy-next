import numpy as np
import pytest

from discopy.optyx import qubits
from discopy.optyx.channel import bit, qubit
from discopy.optyx.classical import *
from discopy.optyx.classical import (
    Bit, BitControlledGate, Id, PostselectBit, PostselectDigit)
from discopy.optyx.utils.misc import compare_arrays_of_different_sizes

def test_addn():
    d_1 = Digit(2, 3) >> Add(2)
    d_2 = Digit(5)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                       d_2.double().to_tensor().eval().array)

def test_subn():
    d_1 = Digit(2, 5) >> Sub()
    d_2 = Digit(3)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                       d_2.double().to_tensor().eval().array)

def test_multiplyn():
    d_1 = Digit(2, 3) >> Multiply()
    d_2 = Digit(6)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                          d_2.double().to_tensor().eval().array)

def test_dividen():
    d_1 = Digit(6, 2) >> Divide()
    d_2 = Digit(3)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                          d_2.double().to_tensor().eval().array)

def test_mod2():
    d_1 = Digit(5) >> Mod2()
    d_3 = Bit(1)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_3.double().to_tensor().eval().array)

def test_copyn():
    d_1 = Digit(2) >> CopyN(3)
    d_2 = Digit(2, 2, 2)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_swapn():
    d_1 = Digit(2, 3) >> SwapN()
    d_2 = Digit(3, 2)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_postselectbit():
    d_1 = Bit(1, 0) >> PostselectBit(1, 0)
    d_2 = Scalar(1)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_postselectdigit():
    d_1 = Digit(2, 3, 4) >> PostselectDigit(2, 3, 4)
    d_2 = Scalar(1)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_notbit():
    d_1 = Bit(1) >> Not()
    d_2 = Bit(0)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_xorbit():
    d_1 = Bit(1, 0) >> Xor()
    d_2 = Bit(1)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_andbit():
    d_1 = Bit(1, 0) >> And()
    d_2 = Bit(0)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_copybit():
    d_1 = Bit(1) >> CopyBit(3)
    d_2 = Bit(1, 1, 1)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_swapbit():
    d_1 = Bit(1, 0) >> SwapBit()
    d_2 = Bit(0, 1)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_orbit():
    d_1 = Bit(1, 0) >> Or()
    d_2 = Bit(1)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_zxh():
    d_1 = X(0, 1) >> H()
    d_2 = Z(0, 1)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)


def test_bit_controlled_diagram():
    """Copying and merging the target are applied whatever the bit, the
    phase between them only when the bit is one."""
    phase = qubits.Z(1, 2) >> qubits.Z(1, 1, .25) @ qubit >> qubits.Z(2, 1)
    controlled = BitControlledGate(phase)
    for value, expected in [(0, qubits.Z(1, 1)), (1, qubits.Z(1, 1, .25))]:
        assert np.allclose(
            (Bit(value) @ qubits.Ket("+") >> controlled).double()
            .to_tensor().eval().array,
            (qubits.Ket("+") >> expected).double().to_tensor().eval().array)
    assert BitControlledGate(phase, is_dagger=True) == controlled.dagger()


def test_classical_guards():
    with pytest.raises(ValueError, match="0s and 1s"):
        PostselectBit(2)
    with pytest.raises(ValueError, match="integers"):
        PostselectDigit(1.5)
    with pytest.raises(TypeError, match="channel.Ty"):
        Id(2)
    assert Bit(1) >> Id(bit) == Bit(1)
