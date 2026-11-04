import random

import numpy as np
from pyzx import VertexType
from pytest import mark, raises, fixture

from discopy.quantum.gates import CRz, CRx, CU1, Ket, Rx, H
from discopy import abc, quantum
from discopy.optyx.core.zx import *
from discopy.optyx.core.zx import And, H, Or, Spider, X, Z, ZBox, ZXBox
from discopy.optyx.core import diagram
from discopy.optyx import qubits
from discopy.optyx.core.diagram import Diagram, Bit

@fixture
def random_had_cnot_diagram():
    def _random_had_cnot_diagram(qubits_, depth, p_had=0.5):
        random.seed(0)
        c = Ket(*(qubits_ * [0]))
        for _ in range(depth):
            r = random.random()
            if r > p_had:
                c = c.H(random.randrange(qubits_))
            else:
                tgt = random.randrange(qubits_)
                while True:
                    ctrl = random.randrange(qubits_)
                    if ctrl != tgt:
                        break
                c = c.CX(tgt, ctrl)
        return c
    return _random_had_cnot_diagram


def test_Diagram():
    bialgebra = Z(1, 2) @ Z(1, 2) >> Bit(1) @ SWAP @ Bit(1) >> X(2, 1) @ X(2, 1)
    assert str(bialgebra) == "Z(1, 2) @ bit >> bit @ bit @ Z(1, 2) " \
                             ">> Permutation(bit @ bit @ bit @ bit, " \
                             "[0, 2, 1, 3]) " \
                             ">> X(2, 1) @ bit @ bit >> bit @ X(2, 1)"


def test_Spider():
    assert str(Z(1, 2, 3)) == "Z(1, 2, 3)"
    assert str(X(1, 2, 3)) == "X(1, 2, 3)"
    for spider in [Z, X]:
        assert spider(1, 2, 3).phase == 3
        assert spider(1, 2, 3j).dagger() == spider(2, 1, -3j)
        assert spider(abc.Nat(2), 1, .5).conjugate() == spider(2, 1, -.5)


def test_H():
    assert str(H) == "H"
    assert np.allclose(H[::-1].to_tensor().eval().array, H.to_tensor().eval().array)


def test_Sum():
    assert Z(1, 1) + Z(1, 1) >> Z(1, 1) == sum(2 * [Z(1, 1) >> Z(1, 1)])


def test_scalar():
    assert scalar(1j)[::-1] == scalar(-1j)


def test_subs():
    from sympy.abc import phi, psi
    assert Z(3, 2, phi).subs(phi, 1) == Z(3, 2, 1)
    assert scalar(phi).subs(phi, psi) == scalar(psi)


def test_grad():
    from sympy.abc import phi, psi
    from math import pi
    assert not scalar(phi).grad(psi) and scalar(phi).grad(phi) == scalar(1)
    assert not Z(1, 1, phi).grad(psi)
    assert Z(1, 1, phi).grad(phi) == scalar(1j * pi) @ Z(1, 1, phi)\
        + scalar(-1j * pi) @ Z(1, 1, phi + .5)
    composite = Z(1, 1, phi / 2) >> Z(1, 1, phi + 1)
    assert len((composite.grad(phi)).terms) == 4  # two per spider


def test_ZXBox_equals_its_diagram():
    for box in [Z(0, 2), X(2, 1), Z(1, 1, 0.25), And(2), Or(2), H]:
        diagram = box >> Diagram.id(box.cod)
        assert box == diagram and diagram == box
        assert hash(box) == hash(diagram)
    assert Z(1, 1, 0.25) != Z(1, 1, 0.5) and Z(1, 1) != X(1, 1)


def test_to_pyzx_errors():
    with raises(NotImplementedError):
        Diagram.to_pyzx(quantum.H)


def test_to_pyzx():
    assert ZXDiagram.from_pyzx(Z(0, 2).to_pyzx()) == Z(0, 2)
    assert hash(ZXDiagram.from_pyzx(Z(0, 2).to_pyzx())) == hash(Z(0, 2))


def test_to_pyzx_scalar():
    # Test that a scalar is translated to the corresponding pyzx object.
    k = np.exp(np.pi / 4 * 1j)
    m = (scalar(k) @ scalar(k) @ Bit(1)).to_pyzx().to_matrix()
    m = np.linalg.norm(m / 1j - np.eye(2))
    assert np.isclose(m, 0)


def test_from_pyzx_errors():
    bialgebra = Z(1, 2) @ Z(1, 2) >> Bit(1) @ SWAP @ Bit(1) >> X(2, 1) @ X(2, 1)
    graph = bialgebra.to_pyzx()
    graph.set_inputs(())
    graph.set_outputs(())
    with raises(ValueError):  # missing_boundary
        ZXDiagram.from_pyzx(graph)
    graph.auto_detect_io()
    graph.set_inputs(graph.inputs() + graph.outputs())
    with raises(ValueError):  # duplicate_boundary
        ZXDiagram.from_pyzx(graph)


def test_from_pyzx_boundaries_in_odd_orders():
    from pyzx import compare_tensors
    from fixtures.zx_graphs import boundaries_in_odd_orders
    graph = boundaries_in_odd_orders()
    assert compare_tensors(ZXDiagram.from_pyzx(graph).to_pyzx(), graph)


def _std_basis_v(*c):
    v = np.zeros(2**len(c), dtype=complex)
    v[np.sum((np.array(c) != 0) * 2**np.arange(len(c)))] = 1
    return np.expand_dims(v, -1)


def test_circuit2zx():
    circuit = Ket(0, 0) >> quantum.H @ Rx(0) >> CRz(0) >> CRx(0) >> CU1(0)
    assert qubits.Circuit(circuit).get_kraus() == Diagram.decode(
        dom=Bit(0), boxes_and_offsets=zip([
            X(0, 1), X(0, 1), scalar(0.5), H, X(1, 1),
            Z(1, 2), Z(1, 2), X(2, 1), Z(1, 0), scalar(2 ** 0.5),
            X(1, 2), X(1, 2), Z(2, 1), X(1, 0), scalar(2 ** 0.5),
            Z(1, 2), Z(1, 2), X(2, 1), Z(1, 0)],
            [0, 1, 2, 0, 1, 0, 2, 1, 1, 2, 0, 2, 1, 1, 2, 0, 2, 1, 1]))

    # Verify XYZ=iI
    circuit = quantum.Z >> quantum.Y >> quantum.X
    t = qubits.Circuit(circuit).get_kraus()
    t = t.to_pyzx().to_matrix() - 1j * np.eye(2)
    assert np.isclose(np.linalg.norm(t), 0)

    # Check scalar translation
    circuit = quantum.X >> quantum.X @ quantum.scalar(1j)
    t = qubits.Circuit(circuit).to_pyzx().to_matrix()
    assert np.isclose(np.linalg.norm(t - 1j * np.eye(2)), 0)

    # with raises(NotImplementedError):
    #     circuit2zx(quantum.scalar(1, is_mixed=True))

    circuit = Ket(0)
    t = qubits.Circuit(circuit).to_pyzx().to_matrix() - _std_basis_v(0)
    assert np.isclose(np.linalg.norm(t), 0)
    circuit = Ket(0, 0)
    t = qubits.Circuit(circuit).to_pyzx().to_matrix() - _std_basis_v(0, 0)
    assert np.isclose(np.linalg.norm(t), 0)

    circuit = quantum.Id(3).CX(0, 2)
    assert (qubits.Circuit(circuit).get_kraus()
            == Diagram.decode(
                dom=Bit(3),
                boxes_and_offsets=zip(
                    [SWAP, Z(1, 2), X(2, 1), scalar(2 ** 0.5), SWAP],
                    [1, 0, 1, 2, 1])))


def test_to_pyzx_reads_any_spider_as_a_z_vertex():
    graph = diagram.Spider(1, 2, diagram.bit).to_pyzx()
    boundary = graph.inputs() + graph.outputs()
    spiders = [v for v in graph.vertices() if v not in boundary]
    assert [graph.type(v) for v in spiders] == [VertexType.Z]
    assert len(graph.neighbors(spiders[0])) == 3
    with raises(NotImplementedError):
        diagram.Box("f", diagram.bit, diagram.bit).to_pyzx()


def test_spiders_rotate_and_inflate_to_themselves():
    assert Z(1, 2, .25).rotate() == Z(2, 1, .25)
    assert X(2, 0, .5).rotate(left=True) == X(0, 2, .5)
    assert Z(1, 2, .25).inflate(3) == Z(1, 2, .25)
    assert H.inflate(3) == H
    with raises(NotImplementedError):
        Spider(1, 1).truncation()


def test_classical_gates_are_real():
    assert And().conjugate() == And()
    assert Or(3).conjugate() == Or(3)
    with raises(NotImplementedError):
        ZXBox("f", 1, 1).conjugate()


def test_classical_gates_dagger():
    for cls in (And, Or):
        for n in (2, 3):
            box, flipped = cls(n), cls(n).dagger()
            assert (box.dom, box.cod) == (Bit(n), Bit(1))
            assert flipped.is_dagger
            assert (flipped.dom, flipped.cod) == (Bit(1), Bit(n))
            assert flipped.dagger() == box
            assert cls(n, is_dagger=True) == flipped
            array = box.to_tensor(input_dims=[2] * n).eval().array
            daggered = flipped.to_tensor(input_dims=[2]).eval().array
            assert np.allclose(daggered.reshape(2, 2 ** n),
                               array.reshape(2 ** n, 2).conj().T)


def test_SWAP_evaluates():
    assert np.allclose(
        SWAP.to_tensor().eval().array.reshape(4, 4),
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


@mark.parametrize("spider", [Z, X])
@mark.parametrize("phase", [0, .25, .5])
def test_legless_spider_is_the_sum_of_its_amplitudes(spider, phase):
    """A spider with no legs has no index left to carry its amplitudes, so it
    is their sum `1 + exp(2i pi phase)`. pyzx is the oracle this module's
    docstring names for the array of a `Z` or an `X`; spider fusion is the
    other, and both agree."""
    box = spider(0, 0, phase)
    value = box.to_tensor().eval().array
    assert value.shape == ()
    assert np.allclose(value, 1 + np.exp(2j * np.pi * phase))
    assert np.allclose(value, box.to_pyzx().to_tensor())
    assert np.allclose(value, (
        spider(0, 1, phase) >> spider(1, 0)).to_tensor().eval().array)


@mark.parametrize("spider", [Z, X, ZBox])
@mark.parametrize("legs", [(1, 1), (1, 0), (0, 1), (2, 1), (0, 0)])
def test_grad_matches_central_differences(spider, legs):
    """A structural assertion cannot tell a right gradient formula from a
    wrong one, so the gradient is checked against the difference quotient of
    the spider's own array, as `test/optyx/grad.py` checks the photonic
    gates. `Spider.grad` used to return the half-turn shift alone, which
    carries a spurious value on the all-zero entry of a spider with legs, and
    a `ZBox`'s parameter is an amplitude rather than an angle, so the two
    gradients are not the same formula."""
    from sympy.abc import phi
    n_legs_in, n_legs_out = legs
    step, dims = 1e-6, [2] * n_legs_in

    def array(box):
        return np.asarray(
            box.to_tensor(input_dims=dims).eval().array, dtype=complex
        ).reshape(-1)

    for value in [0., .123, .25, .5]:
        gradient = array(spider(*legs, phi).grad(phi).subs((phi, value)))
        quotient = (array(spider(*legs, value + step))
                    - array(spider(*legs, value - step))) / (2 * step)
        assert np.allclose(gradient, quotient, atol=1e-4)


def test_grad_of_a_composite_matches_central_differences():
    """The product rule over the corrected gradient, against the difference
    quotient of the composite's own array."""
    from sympy.abc import phi
    step = 1e-6

    def composite(phase):
        return Z(1, 1, phase / 2) >> Z(1, 1, phase + 1)

    def array(box):
        return np.asarray(box.to_tensor(input_dims=[2]).eval().array,
                          dtype=complex).reshape(-1)

    for value in [0., .123, .25]:
        gradient = array(composite(phi).grad(phi).subs((phi, value)))
        quotient = (array(composite(value + step))
                    - array(composite(value - step))) / (2 * step)
        assert np.allclose(gradient, quotient, atol=1e-4)
