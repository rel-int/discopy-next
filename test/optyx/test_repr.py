"""
The transparency law of STYLE.md, i.e. :meth:`discopy.testable.transparency`,
on every box of :mod:`discopy.optyx`, the core and the tower.

``zw.ZBox`` and ``zw.IndexableAmplitudes`` are not quantified over: the
amplitudes may be lazy or infinite, so a representation cannot print what it
cannot enumerate, and which way that goes waits on
rel-int/discopy-next#472.

A cell of :func:`test_transparency` is only as strong as the equality it is
checked by, and the boxes whose semantics live outside ``setoid`` —
``BBS``, ``TBS``, ``MZI``, ``Create``, ``DualRail`` and every ``CQMap`` —
would pass a representation that dropped one, see
rel-int/discopy-next#386. :func:`test_what_equality_ignores` is their cell:
it reads back what ``==`` does not.

A box holding a Python function — ``channel.Observable`` with a
``function``, ``classical.ClassicalFunction``,
``classical.BitControlledPhaseShift`` — prints that function as Python
does, so its representation reads back only if the function's does, as for
``python.Function``. They are not quantified over here.
"""

import sys

import numpy as np
import pytest

import sympy

import discopy
from discopy import optyx
from discopy.optyx import channel, classical, photonic, qubits
from discopy.optyx.channel import bit, mode, qubit, qmode
from discopy.optyx.core import diagram, zw, zx


def environment(module):
    """The namespace a representation of ``module`` reads back in."""
    public = lambda namespace: {
        name: value for name, value in namespace.items()
        if not name.startswith("_")}
    return {**public(vars(discopy)), "optyx": optyx, **public(vars(module))}


CHANNEL = [
    channel.Channel("f", diagram.Id(diagram.mode)),
    channel.CQMap("g", diagram.Id(mode.double()), mode, mode),
    channel.Measure(qubit @ qmode),
    channel.Encode(bit @ mode),
    channel.Encode(mode, internal_states=([1, 0],)),
    channel.Discard(qubit),
    channel.Permutation(qubit @ bit @ mode, [2, 0, 1]),
    channel.Swap(qubit, bit),
    channel.Observable(operator=qubits.Z(1, 1, 0.25)),
]

QUBITS = [
    qubits.Z(1, 2), qubits.Z(1, 1, 0.25), qubits.Z(1, 1, 0.0),
    qubits.X(2, 1), qubits.X(0, 1, 0.5), qubits.X(1, 1, 0.0),
    qubits.H(), qubits.Scalar(2), qubits.Scalar(0.5j),
    qubits.Ket(0), qubits.Ket("+"), qubits.Bra(1), qubits.Bra("-"),
    qubits.Measure(2), qubits.Encode(3), qubits.Discard(1),
    qubits.BitFlipError(0.1), qubits.DephasingError(0.2),
]

PHOTONIC = [
    photonic.Create(1, 0, 2), photonic.Select(1, 1),
    photonic.Scalar(2), photonic.Scalar(0.5),
    photonic.Encode(2), photonic.Discard(2),
    photonic.PhotonThresholdMeasurement(2),
    photonic.NumberResolvingMeasurement(2),
    photonic.Gate([[0, 1], [1, 0]], 2, 2, "flip"),
    photonic.Phase(0.25), photonic.NumOp(), photonic.HadamardBS(),
    photonic.BBS(0), photonic.BBS(0.5, is_conj=True),
    photonic.TBS(0.25), photonic.TBS(0.25, is_gate_dagger=True),
    photonic.MZI(0.25, 0.5), photonic.MZI(0.25, 0.5, is_conj=True),
    photonic.DualRail(2), photonic.PhaseShiftDR(0.25),
    photonic.Swap(1, 1), photonic.PhotonLoss(0.5),
]

CLASSICAL = [
    classical.Digit(1, 2), classical.PostselectDigit(1),
    classical.PostselectBit(0, 1),
    classical.Add(2), classical.CopyN(2), classical.CopyBit(3),
    classical.And(2), classical.Or(2), classical.Xor(2),
    classical.Not(), classical.Mod2(), classical.Divide(),
    classical.Multiply(), classical.Sub(), classical.H(),
    classical.SwapBit(), classical.SwapN(),
    classical.Scalar(2), classical.Z(1, 2), classical.Z(1, 1, 0.0),
    classical.X(2, 1, 0.5), classical.X(1, 1, 0.0),
    classical.BinaryMatrix([[1, 0], [0, 1]]),
]


CORE = [
    zw.W(1), zw.W(2), zw.W(2).dagger(),
    zw.Create(), zw.Create(1, 0, 2), zw.Select(), zw.Select(2),
    zw.Endo(0.5), zw.Endo(2j),
    zw.Add(2), zw.Add(3).dagger(), zw.Multiply(), zw.Multiply().dagger(),
    zw.Divide(), zw.Divide().dagger(), zw.Mod2(), zw.Mod2().dagger(),
    zx.Spider(1, 1), zx.Z(1, 2), zx.X(0, 1, 0.5), zx.ZBox(1, 1),
    zx.And(), zx.And(3), zx.And().dagger(), zx.Or(), zx.Or(3).dagger(),
    diagram.Scalar(2), diagram.Scalar(0.5j),
    diagram.DualRail(), diagram.DualRail(True),
    diagram.PhotonThresholdDetector(),
    diagram.PhotonThresholdDetector(True),
    diagram.EmbeddingTensor(1, 2), diagram.EmbeddingTensor(3, 2),
    diagram.Box("f", diagram.Mode(1), diagram.Mode(1)),
    diagram.Spider(1, 2, diagram.Mode(1)),
    diagram.Swap(diagram.Mode(1), diagram.Bit(1)),
]


@pytest.mark.parametrize("box", CHANNEL + QUBITS + PHOTONIC + CLASSICAL + CORE,
                         ids=lambda box: type(box).__name__)
def test_transparency(box):
    module = sys.modules[type(box).__module__]
    assert eval(repr(box), environment(module)) == box


IGNORED = [
    (zw.Create(1, 1, internal_states=(np.array([1, 0]), [0, 1])),
     "internal_states"),
    (diagram.DualRail(internal_state=[1, 0]), "internal_state"),
    (photonic.BBS(0.5, is_conj=True), "is_conj"),
    (photonic.TBS(0.25, is_gate_dagger=True), "is_gate_dagger"),
    (photonic.MZI(0.25, 0.5, is_conj=True), "is_conj"),
    (photonic.Create(1, 1, internal_states=(np.array([1, 0]), [0, 1])),
     "internal_states"),
    (photonic.DualRail(1, internal_states=([1, 0],)), "internal_states"),
    (classical.BinaryMatrix([[0, 1], [1, 0]]), "density_matrix"),
]


@pytest.mark.parametrize("box, attribute", IGNORED,
                         ids=lambda x: x if isinstance(x, str) else "")
def test_what_equality_ignores(box, attribute):
    module = sys.modules[type(box).__module__]
    other = eval(repr(box), environment(module))
    assert other == box
    if attribute == "internal_state":
        assert np.array_equal(other.internal_state, box.internal_state)
    elif attribute == "density_matrix":
        assert np.array_equal(other.density_matrix.matrix,
                              box.density_matrix.matrix)
    elif attribute == "internal_states":
        assert len(other.internal_states) == len(box.internal_states)
        assert all(np.array_equal(x, y) for x, y in zip(
            other.internal_states, box.internal_states))
    else:
        assert getattr(other, attribute) == getattr(box, attribute)
        assert np.allclose(other.array, box.array)


DIAGRAMS = [
    qubits.Z(1, 2) @ qubit >> qubit @ qubits.X(2, 1),
    photonic.Create(1, 1) >> photonic.BBS(0),
    classical.Digit(1) >> classical.Mod2(),
    channel.Discard(qmode) @ channel.Measure(bit),
]


@pytest.mark.parametrize("term", DIAGRAMS, ids=str)
def test_transparency_of_diagrams(term):
    assert eval(repr(term), environment(channel)) == term


PSI = sympy.Symbol("psi")

SYMBOLIC = [zw.Endo(PSI ** 2), diagram.Scalar(2 * PSI), zx.Z(1, 1, PSI)]


@pytest.mark.parametrize("box", SYMBOLIC, ids=lambda box: type(box).__name__)
def test_transparency_of_a_symbol(box):
    """A symbolic phase reads back where its symbol is named, as a float
    would not: the representation prints the number ``==`` compares."""
    module = sys.modules[type(box).__module__]
    assert eval(repr(box), {**environment(module), "psi": PSI}) == box


CORE_DIAGRAMS = [
    zw.Create(1) >> zw.W(2) >> zw.Endo(0.5) @ diagram.Mode(1),
    zw.Create(2) >> zw.Mod2() >> zx.Z(1, 2),
    diagram.Scalar(2) @ zx.And() >> zx.Z(1, 1),
    zw.Add(2).dagger() >> diagram.PhotonThresholdDetector() @ zw.Select(),
]


@pytest.mark.parametrize("term", CORE_DIAGRAMS, ids=str)
def test_transparency_of_core_diagrams(term):
    assert eval(repr(term), environment(zw)) == term
