import pytest

from discopy.optyx.channel import *
from sympy.abc import x, y
from discopy.optyx.channel import (
    CQMap, Channel, Diagram, Encode, Measure, Permutation,
    bit, mode, qmode, qubit)
from discopy.optyx import photonic
from discopy.optyx.core import diagram, zw, zx
from discopy import tensor
import numpy as np

bell_density_re = np.array([
    [0.012, 0.014, 0.014, 0.000],
    [0.014, 0.508, 0.475, 0.008],
    [0.014, 0.475, 0.479, 0.009],
    [0.000, 0.008, 0.009, 0.000]
])
bell_density_im = np.exp(1j * np.pi * np.array([
    [0.000, -1.850, -1.825, -0.985],
    [1.850, 0.000, -0.002, -0.902],
    [1.825, 0.002, 0.000, -0.931],
    [0.985, 0.902, 0.931, 0.000]
]))
bell_density = np.multiply(bell_density_re, bell_density_im)


def test_box_conjugate_preserves_shape():
    array = np.random.default_rng(0).normal(size=(4, 8))
    box = diagram.Box("f", diagram.bit ** 2, diagram.bit ** 3, array=array)

    conjugated = box.conjugate()

    assert (conjugated.dom, conjugated.cod) == (box.dom, box.cod)
    assert np.allclose(conjugated.array, array.conjugate())


def test_channel_double_non_square_kraus():
    # Box.conjugate() used to swap dom and cod, so this raised AxiomError
    # when composing self.kraus @ self.kraus.conjugate(), see optyx#51.
    array = np.random.default_rng(0).normal(size=(4, 8))
    kraus = diagram.Box("f", diagram.bit ** 2, diagram.bit ** 3, array=array)

    Channel("f", kraus, bit ** 2, bit ** 3).double()


def test_channel_double_non_square_kraus_evaluates():
    # Box.determine_output_dimensions returned len(dom) dimensions instead
    # of len(cod), desyncing to_tensor's wire-count bookkeeping for any
    # array box with dom != cod; raised AxiomError on eval, see optyx#51.
    array = np.random.default_rng(0).normal(size=(4, 8))
    kraus = diagram.Box("f", diagram.bit ** 2, diagram.bit ** 3, array=array)

    value = Channel("f", kraus, bit ** 2, bit ** 3).double().to_tensor().eval()

    assert np.allclose(np.asarray(value.array).reshape(4, 8), array ** 2)


def test_box_to_tensor_non_square_array():
    # Same determine_output_dimensions bug as above, surfacing outside
    # Channel.double() entirely: a non-square array box followed by an
    # idle parallel wire desyncs layer_dims for every box downstream,
    # since it's a plain python list, not just the wire *count* tracked
    # separately. See optyx#51.
    f_array = np.random.default_rng(0).normal(size=(4, 8))
    h_array = np.random.default_rng(1).normal(size=(4, 2))
    f = diagram.Box("f", diagram.bit ** 2, diagram.bit ** 3, array=f_array)
    h = diagram.Box("h", diagram.bit ** 2, diagram.bit, array=h_array)
    d = ((f @ diagram.Id(diagram.bit))
         >> (diagram.Id(diagram.bit) @ h @ diagram.Id(diagram.bit)))

    value = d.to_tensor().eval()

    assert value.dom == tensor.Dim(2, 2, 2)
    assert value.cod == tensor.Dim(2, 2, 2)


def test_CQMap():
    X = Channel("X", zx.X(1, 1, 0.5))
    bell = diagram.Box(name="Bell", dom=diagram.bit ** 2, cod=diagram.bit ** 2, array=bell_density)
    bell = diagram.Spider(0, 2, typ=diagram.bit) >> diagram.Id(diagram.bit) @ diagram.Spider(0, 2, typ=diagram.bit) @ diagram.Id(diagram.bit) >> diagram.Diagram.permutation([0,1,3,2], diagram.bit**4) >> diagram.Id(diagram.bit ** 2) @ bell >> diagram.Diagram.permutation([0,2,1,3], diagram.bit**4)

    Noisy_bell = CQMap('Physical Bell', bell @ diagram.Scalar(1/0.999), dom=Ty(), cod=qubit ** 2)
    Perfect_Bell_Effect = Channel("Perfect Bell Effect", diagram.Spider(2,0,typ=diagram.bit) @ diagram.Scalar(1 / np.sqrt(2)))

    CALCULATED_FIDELITY = (Noisy_bell >> Diagram.id(qubit) @ X >> Perfect_Bell_Effect).double().to_tensor().eval().array.real
    REAL_FIDELITY = .96898
    print(CALCULATED_FIDELITY)
    print(REAL_FIDELITY)
    assert np.isclose(CALCULATED_FIDELITY, REAL_FIDELITY, rtol=1e-3)


def test_from_bosonic_op():
    from discopy.optyx.channel import Diagram

    matrix = [
        [0, 1],
        [1, 0]
    ]

    n = len(matrix)
    assert len(matrix[0]) == n

    terms = []

    for i in range(n):
        for j in range(n):
            term = Diagram.from_bosonic_operator(
                n_modes=n,
                operators=[(i, False), (j, True)],
                scalar=matrix[i][j]
            )
            terms.append(term)

    hamiltonian = Diagram.sum_factory(terms)

    sum_1 = np.sign(hamiltonian.get_kraus().to_tensor().eval().array)
    sum_2 = np.sign(hamiltonian.eval().tensor.array) # bug in discopy.tensor.Tensor.eval() for Sums (?)

    assert np.allclose(sum_2, sum_1)


def test_observables_agree_with_their_samples():
    from discopy.optyx.channel import Observable, Discard, Diagram, qmode, qubit
    from discopy.optyx import photonic, qubits
    hom = photonic.Create(1, 1) >> photonic.BS
    number = Observable(Diagram.from_bosonic_operator(
        2, ((0, False), (0, True))))
    counted = Observable(function=lambda n: n[0], dom=qmode @ qmode)
    assert np.isclose(number.expectation(hom), counted.expectation(hom))
    assert np.isclose(number.expectation(hom), 1)
    samples = (hom >> photonic.NumberResolvingMeasurement(2)).sample(
        burn_in=0, shots=500, seed=0)
    assert abs(np.mean([counted.function(list(sample[0]))
                        for sample in samples]) - 1) < .15
    trace = Observable(function=lambda n: 1, dom=qmode @ qmode)
    assert np.isclose(trace.expectation(hom), (hom >> Discard(
        qmode @ qmode)).double().to_tensor().eval().array)
    rotated = qubits.Ket("+") >> qubits.Z(1, 1, .1)
    pauli_x = Observable(qubits.X(1, 1, .5))
    assert np.isclose(pauli_x.expectation(rotated), np.cos(.2 * np.pi))


def test_observable_errors():
    from discopy.optyx.channel import Observable, Diagram, qmode, bit
    from discopy.optyx import photonic
    with pytest.raises(ValueError):
        Observable()
    with pytest.raises(ValueError):
        Observable(photonic.Create(1))
    with pytest.raises(ValueError):
        Observable(Diagram.id(bit))
    with pytest.raises(ValueError):
        Observable(function=lambda n: n[0], dom=qmode).expectation(
            photonic.Create(1, 1))


def test_bosonic_operator_indices_are_modes():
    for index in (-1, 2):
        with pytest.raises(ValueError, match="out of bounds"):
            Diagram.from_bosonic_operator(2, [(index, True)])


def test_decomposition_is_for_zx_channels():
    with pytest.raises(NotImplementedError, match="ZX channels"):
        Channel("create", zw.Create(1)).decomp()
    with pytest.raises(TypeError, match="ZX channels"):
        photonic.TBS(.25).to_dual_rail()


def test_sums_double_and_differentiate_term_by_term():
    dims = [2] * 4
    assert np.allclose(
        (photonic.BS + photonic.BS).double().to_tensor(dims).eval().array,
        2 * photonic.BS.double().to_tensor(dims).eval().array)
    phases = photonic.Phase(x) + photonic.Phase(2 * x)
    assert phases.grad(y) == Diagram.sum_factory((), qmode, qmode)
    one_photon = phases.grad(x).subs((x, .1)).eval().tensor.array[1, 1]
    assert np.isclose(one_photon, 2j * np.pi * (
        np.exp(.2j * np.pi) + 2 * np.exp(.4j * np.pi)))


def test_cq_map_power_is_a_tensor_power():
    measure = CQMap("measure", diagram.Spider(2, 1, diagram.bit), qubit, bit)
    assert measure ** 1 == measure
    assert measure ** 3 == measure @ measure @ measure


def test_permutation_dagger_is_its_inverse():
    perm = Permutation(qmode ** 3, [1, 2, 0])
    assert (perm.dagger().dom, perm.dagger().cod) == (perm.cod, perm.dom)
    assert np.allclose((perm >> perm.dagger()).to_path().array, np.eye(3))


def test_inflation_only_touches_optical_wires():
    assert Measure(qubit).inflate(2) == Measure(qubit)
    assert Encode(bit).inflate(2) == Encode(bit)
    assert Encode(mode, internal_states=[1, 0]).internal_states == ([1, 0], )


def test_measure_inflate_mixed_types():
    dom = qubit @ bit @ qmode @ mode
    assert Measure(dom).inflate(2).dom == dom.inflate(2)


def test_encode_inflate_mixed_types():
    dom = qubit @ bit @ qmode @ mode
    encode = Encode(dom, internal_states=([1, 0],))
    assert encode.inflate(2).dom == dom.inflate(2)


def test_dual_rail_only_for_zx_channels():
    with pytest.raises(TypeError, match="Only ZX channels"):
        Channel("id", diagram.Id(diagram.mode)).dual_rail()
