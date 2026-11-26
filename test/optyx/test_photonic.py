import numpy as np
import pytest
from sympy.abc import x, y

from discopy import cat
from discopy.matrix import backend
from discopy.optyx import photonic, qubits
from discopy.optyx.channel import Diagram, Ty, qmode
from discopy.optyx.core import zw
from discopy.optyx.photonic import (
    BBS, BS, MZI, TBS, AbstractGate, Create, Discard, DualRail, Encode, Gate,
    HadamardBS, Id, NumOp, NumberResolvingMeasurement, Phase, PhaseShiftDR,
    PhotonLoss, PhotonThresholdMeasurement, Scalar, Select, Swap,
    XMeasurementDR, ZMeasurementDR, ansatz)
from discopy.python import finset
from discopy.utils import dumps, from_tree, loads


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


@pytest.mark.parametrize("swap, perm", [
    (Swap(), [1, 0]),
    (Swap(2, 1), [2, 0, 1]),
    (Swap(0, 2), [0, 1]),
])
def test_swap_tree_and_transparency(swap, perm):
    import pickle
    assert swap.perm == finset.Permutation(perm, len(perm))
    loaded = loads(dumps(swap))
    assert type(loaded) is Swap and loaded == swap
    assert loaded.perm == swap.perm
    assert loaded.to_tree() == swap.to_tree()
    assert from_tree(swap.to_tree()) == swap
    assert loaded.double() == swap.double()
    assert eval(repr(swap), Swap.environment()) == swap
    pickled = pickle.loads(pickle.dumps(swap))
    assert type(pickled) is Swap and pickled == swap


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


@pytest.mark.parametrize("gate", [
    lambda t: MZI(t, 0.2), lambda t: MZI(0.1, t, is_gate_dagger=True),
    lambda t: TBS(t, is_conj=True), BBS, lambda t: Phase(t) @ qmode])
def test_jax_traces_gate_parameters(gate):
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    def loss(diagram, module):
        return module.abs(diagram.to_path().array[0, 0]) ** 2

    def traced(t):
        with backend("jax"):
            return loss(step.lambdify(x, modules="jax")(t), jnp)

    step = Create(1) @ qmode >> gate(x)
    epsilon = 1e-5
    expected = (loss(step.lambdify(x)(.3 + epsilon), np)
                - loss(step.lambdify(x)(.3 - epsilon), np)) / (2 * epsilon)
    assert np.isclose(traced(jnp.array(.3)), loss(step.lambdify(x)(.3), np))
    assert np.isclose(jax.grad(traced)(.3), expected, atol=1e-4)


def test_jax_traces_feedback_parameters():
    jax = pytest.importorskip("jax")
    loop = (Create(1) @ qmode >> MZI(x, 0.2)).feedback(state=Create(0))

    def traced(t):
        step = loop.lambdify(x, modules="jax")(t).one_step()
        with backend("jax"):
            return jax.numpy.abs(step.dilate().to_path().array[0, 0]) ** 2

    assert np.isclose(jax.grad(traced)(.3), -np.pi * np.sin(.6 * np.pi),
                      atol=1e-4)


@pytest.mark.parametrize("tensor, dtype", [
    ("theta", "float64"), ("phi", "float64"), ("phi", "int64")])
def test_torch_parameter_beside_a_float(tensor, dtype):
    torch = pytest.importorskip("torch")
    params = {"theta": 0.1, "phi": 0.2 if dtype == "float64" else 0}
    expected = MZI(**params).array
    params[tensor] = torch.tensor(params[tensor], dtype=getattr(torch, dtype))
    assert np.allclose(MZI(**params).array.numpy(), expected)


def test_jax_gradient_where_an_entry_vanishes():
    jax = pytest.importorskip("jax")

    def entry(t):
        with backend("jax"):
            return TBS(t).to_path().array[0, 0].imag

    assert np.isclose(jax.grad(entry)(0.), jax.grad(
        lambda t: TBS(t).array[0, 0].imag)(0.))
    assert np.isclose(jax.grad(entry)(0.), np.pi)


@pytest.mark.parametrize("gate", [
    lambda t: MZI(t, 0.2), lambda t: MZI(0.1, t, is_gate_dagger=True),
    lambda t: TBS(t, is_conj=True), BBS, Phase])
def test_torch_gradient_of_gate_parameters(gate):
    torch = pytest.importorskip("torch")

    def loss(array):
        return sum((i + 1) * (x.real + 2 * x.imag)
                   for i, x in enumerate(array.reshape(-1)))

    t = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    loss(gate(t).array).backward()
    epsilon = 1e-5
    expected = (loss(gate(.3 + epsilon).array)
                - loss(gate(.3 - epsilon).array)) / (2 * epsilon)
    assert np.isclose(t.grad.item(), expected, atol=1e-4)


def test_torch_gates_compare():
    torch = pytest.importorskip("torch")
    t = torch.tensor(.3, dtype=torch.float64)
    assert TBS(t) == TBS(t) and MZI(t, 0.2) == MZI(t, 0.2)
    assert TBS(t) != TBS(t + 1) and len({BBS(t), BBS(t)}) == 1


def test_torch_gradient_survives_to_path():
    """#652: the zw path of a gate is built from the entries of its array,
    so it keeps the gradient the array has."""
    torch = pytest.importorskip("torch")
    t = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    with backend("pytorch"):
        array = MZI(t, 0.2).to_path().array
        assert array.requires_grad
        array.abs().sum().backward()
    assert t.grad is not None


PHOTONIC_BOXES = [
    Select(),
    Select(1, 0, 2),
    Scalar(0.5),
    Scalar(1j),
    Scalar(2 * x),
    Encode(2),
    Discard(2),
    PhotonThresholdMeasurement(),
    PhotonThresholdMeasurement(2),
    NumberResolvingMeasurement(2),
    Create(),
    Create(1, 0, 2),
    Create(1, 0, 1, internal_states=([1, 0], [0, 1])),
    Create(1, 1, internal_states=(np.array([1, 0]), [0, 1j])),
    Gate([[0, 1], [1, 0]], 2, 2, "flip"),
    Gate([[1j, 0], [0, 1]], 2, 2, "P", data=(0.1, 0.2)),
    Gate([[1j, 0], [0, 1]], 2, 2, "P", data=(0.1, 0.2)).dagger(),
    Gate([[1j, 0], [0, 1]], 2, 2, "P").conjugate(),
    Gate([[x, 0], [0, 1]], 2, 2, "sym", data=x),
    Phase(0.25),
    Phase(x),
    NumOp(),
    BS,
    BBS(0.2, is_conj=True),
    TBS(0.3),
    TBS(0.3, is_gate_dagger=True, is_conj=True),
    MZI(0.1, 0.2),
    MZI(0.1, 0.2, is_gate_dagger=True, is_conj=True),
    HadamardBS(),
    HadamardBS().dagger(),
    HadamardBS().conjugate(),
    DualRail(2),
    DualRail(2, internal_states=([1, 0], [0, 1])),
    DualRail(1, internal_states=(np.array([1, 0]),)),
    PhaseShiftDR(0.25),
    PhaseShiftDR(x),
    Swap(),
    Swap(2, 1),
    PhotonLoss(0.8),
    PhotonLoss(x),
]


def test_every_photonic_box_class_is_covered():
    defined = {
        cls for cls in vars(photonic).values()
        if isinstance(cls, type) and issubclass(cls, cat.Box)
        and cls.__module__ == photonic.__name__ and cls is not AbstractGate
    }
    assert defined == {type(box) for box in PHOTONIC_BOXES}


@pytest.mark.parametrize("box", PHOTONIC_BOXES)
def test_photonic_box_tree_roundtrip(box):
    loaded = loads(dumps(box))
    assert type(loaded) is type(box) and loaded == box
    assert loaded.to_tree() == box.to_tree()
    assert from_tree(box.to_tree()) == box
    diagram = box @ Id(1)
    assert loads(dumps(diagram)) == diagram


def test_photonic_diagram_tree_roundtrip():
    diagram = (
        Create(1, 0, 1, internal_states=([1, 0], [0, 1]))
        >> HadamardBS() @ Phase(0.25)
        >> Swap(2, 1)
        >> Id(1) @ PhotonLoss(0.8) @ NumOp()
        >> Select(1, 0) @ PhotonThresholdMeasurement()
    )
    assert loads(dumps(diagram)) == diagram
    assert loads(dumps(diagram)).to_tree() == diagram.to_tree()
    assert loads(dumps(ansatz(4, 2))) == ansatz(4, 2)
