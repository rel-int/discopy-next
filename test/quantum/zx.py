import random

import pytest

import numpy as np
from pytest import raises, fixture

from discopy.quantum.gates import CRz, CRx, CU1, Controlled, Rz, Rx
from discopy.quantum.zx import *
from discopy import frobenius
from discopy.tensor import Dim, Tensor


@fixture
def random_had_cnot_diagram():
    def _random_had_cnot_diagram(qubits, depth, p_had=0.5):
        random.seed(0)
        c = Ket(*(qubits * [0]))
        for _ in range(depth):
            r = random.random()
            if r > p_had:
                c = c.H(random.randrange(qubits))
            else:
                tgt = random.randrange(qubits)
                while True:
                    ctrl = random.randrange(qubits)
                    if ctrl != tgt:
                        break
                c = c.CX(tgt, ctrl)
        return c
    return _random_had_cnot_diagram


def test_Diagram():
    bialgebra = Z(1, 2) @ Z(1, 2) >> Nat(1) @ SWAP @ Nat(1) >> X(2, 1) @ X(2, 1)
    assert str(bialgebra) == "Z(1, 2) @ Nat(1) >> Nat(2) @ Z(1, 2) " \
                             ">> Permutation(Nat(4), [0, 2, 1, 3]) " \
                             ">> X(2, 1) @ Nat(2) >> Nat(1) @ X(2, 1)"


def test_Spider():
    assert str(Z(1, 2, 3)) == "Z(1, 2, 3)"
    assert str(Y(1, 2, 3)) == "Y(1, 2, 3)"
    assert str(X(1, 2, 3)) == "X(1, 2, 3)"
    for spider in [Z, Y, X]:
        assert spider(1, 2, 3).phase == 3
        assert spider(1, 2, 3j).dagger() == spider(2, 1, -3j)


def test_H():
    assert str(H) == "H"
    assert H[::-1] == H


def test_Sum():
    assert Z(1, 1) + Z(1, 1) >> Z(1, 1) == sum(2 * [Z(1, 1) >> Z(1, 1)])


def test_scalar():
    assert scalar(1j)[::-1] == scalar(-1j)


def test_Functor():
    x = frobenius.Ty('x')
    f = frobenius.Box('f', x, x)
    F = frobenius.Functor(
        ob_map=lambda _: Nat(1),
        ar_map=lambda f: Z(len(f.dom), len(f.cod)),
        cod=Diagram)
    assert F(f) == Z(1, 1)
    assert F(frobenius.Swap(x, x)) == Diagram.permutation([1, 0]) == SWAP
    assert F(frobenius.Cup(x.l, x)) == Z(2, 0)
    assert F(frobenius.Cap(x.r, x)) == Z(0, 2)
    assert F(f + f) == Z(1, 1) + Z(1, 1)


def test_subs():
    pytest.importorskip("pyzx")
    from sympy.abc import phi, psi
    assert Z(3, 2, phi).subs(phi, 1) == Z(3, 2, 1)
    assert scalar(phi).subs(phi, psi) == scalar(psi)


def test_grad():
    pytest.importorskip("sympy")
    from sympy.abc import phi, psi
    from math import pi
    assert not scalar(phi).grad(psi) and scalar(phi).grad(phi) == scalar(1)
    assert not Z(1, 1, phi).grad(psi)
    assert Z(1, 1, phi).grad(phi) == scalar(1j * pi) @ Z(1, 1, phi)\
        + scalar(-1j * pi) @ Z(1, 1, phi + .5)
    assert Y(2, 1, phi / 2).grad(phi) == scalar(.5j * pi) @ Y(2, 1, phi / 2)\
        + scalar(-.5j * pi) @ Y(2, 1, phi / 2 + .5)


def test_grad_numeric():
    """ The gradient is the derivative of :meth:`Diagram.eval`. """
    pytest.importorskip("sympy")
    from sympy.abc import phi
    diagrams = [
        Z(1, 1, phi), X(1, 1, phi), Y(1, 1, phi), Z(2, 1, phi), X(1, 2, phi),
        Y(2, 2, phi), Z(0, 0, phi), Z(1, 1, phi / 2) >> X(1, 1, phi + 1)]
    for diagram in diagrams:
        gradient = diagram.grad(phi).subs(phi, .3).eval().array
        finite_difference = (
            diagram.subs(phi, .3 + 1e-6).eval().array
            - diagram.subs(phi, .3 - 1e-6).eval().array) / 2e-6
        assert np.allclose(gradient, finite_difference)


def test_to_pyzx_errors():
    pytest.importorskip("pyzx")
    with raises(NotImplementedError):
        Box('f', Nat(1), Nat(1)).to_pyzx()


def test_to_pyzx():
    pytest.importorskip("pyzx")
    assert Diagram.from_pyzx(Z(0, 2).to_pyzx()) == Z(0, 2) >> SWAP


def test_to_pyzx_parallel_wires():
    pytest.importorskip("pyzx")
    diagram = X(1, 2, .75) >> Z(2, 1, 1 / 3)
    graph = diagram.to_pyzx()
    assert graph.num_edges() == 2 and np.isclose(graph.scalar.to_number(), .5)
    assert np.allclose(matrix(diagram), graph.to_matrix())
    assert proportional(matrix(Diagram.from_pyzx(graph)), matrix(diagram))
    diagrams = [
        Z(1, 2) >> Z(2, 1, .3),
        Z(1, 2, .25) >> H @ H >> Z(2, 1),
        Z(1, 2) >> Nat(1) @ H >> Z(2, 1),
        Z(1, 2, .25) >> Nat(1) @ H >> X(2, 1),
        Z(1, 3, .25) >> X(3, 1, .5),
        Z(1, 3, .25) >> H @ Nat(2) >> X(3, 1, .5)]
    for diagram in diagrams:
        assert np.allclose(matrix(diagram), diagram.to_pyzx().to_matrix())


def test_pyzx_scalars():
    pytest.importorskip("pyzx")
    for diagram in [
            scalar(2) @ Nat(1), X(1, 2, .75) >> Z(2, 1, 1 / 3),
            Z(0, 0, .3), X(0, 2) >> Z(2, 0, .5), X(0, 2, .25) >> Z(2, 0, 1 / 3),
            Z(1, 1, .25) @ (X(0, 1) >> Z(1, 0, .5))]:
        graph = diagram.to_pyzx()
        assert np.allclose(matrix(diagram), graph.to_matrix())
        assert Diagram.from_pyzx(graph).eval().is_close(diagram.eval())


def test_to_pyzx_Y():
    pytest.importorskip("pyzx")
    for diagram in [
            Y(1, 1, .5), Y(2, 1, 1 / 3), Y(0, 2, -.125), Y(1, 0) @ Nat(1),
            H >> Y(1, 2, .3) >> H @ Nat(1), Y(1, 2, .75) >> Y(2, 1, .3)]:
        graph = diagram.to_pyzx()
        assert np.allclose(matrix(diagram), graph.to_matrix())
        assert Diagram.from_pyzx(graph).eval().is_close(diagram.eval())


def test_to_pyzx_scalar():
    pytest.importorskip("pyzx")
    # Test that a scalar is translated to the corresponding pyzx object.
    k = np.exp(np.pi / 4 * 1j)
    m = (scalar(k) @ scalar(k) @ Nat(1)).to_pyzx().to_matrix()
    m = np.linalg.norm(m / 1j - np.eye(2))
    assert np.isclose(m, 0)


def test_from_pyzx_errors():
    pytest.importorskip("pyzx")
    bialgebra = Z(1, 2) @ Z(1, 2) >> Nat(1) @ SWAP @ Nat(1) >> X(2, 1) @ X(2, 1)
    graph = bialgebra.to_pyzx()
    graph.set_inputs(())
    graph.set_outputs(())
    with raises(ValueError):  # missing_boundary
        Diagram.from_pyzx(graph)
    graph.auto_detect_io()
    graph.set_inputs(graph.inputs() + graph.outputs())
    with raises(ValueError):  # duplicate_boundary
        Diagram.from_pyzx(graph)


def test_backnforth_pyzx_1():
    pytest.importorskip("pyzx")
    from pyzx import Graph
    path = 'test/fixtures/zx-graph.json'
    graph = Graph.from_json(open(path).read())
    diagram = Diagram.from_pyzx(graph)
    backnforth = lambda diagram: Diagram.from_pyzx(diagram.to_pyzx())
    assert backnforth(diagram) == diagram


def test_backnforth_pyzx_2(random_had_cnot_diagram):
    pytest.importorskip("pyzx")
    from pyzx import compare_tensors
    backnforth = lambda diagram: Diagram.from_pyzx(diagram.to_pyzx())
    c = random_had_cnot_diagram(3, 25)
    diagram = circuit2zx(c)
    new_diagram = backnforth(diagram)
    assert compare_tensors(
        new_diagram.to_pyzx(), diagram.to_pyzx(), preserve_scalar=True
    )


def _std_basis_v(*c):
    v = np.zeros(2**len(c), dtype=complex)
    v[np.sum((np.array(c) != 0) * 2**np.arange(len(c)))] = 1
    return np.expand_dims(v, -1)


def test_circuit2zx():
    pytest.importorskip("pyzx")
    circuit = Ket(0, 0) >> quantum.H @ Rx(0) >> CRz(0) >> CRx(0) >> CU1(0)
    gadget = [Z(1, 2), Z(1, 2), X(2, 1), Z(1, 0), scalar(2 ** 0.5)]
    assert circuit2zx(circuit) == Diagram.decode(
        dom=Nat(0), boxes_and_offsets=zip(
            [X(0, 1), X(0, 1), scalar(0.5), H, X(1, 1)]
            + gadget + [H] + gadget + [H] + gadget,
            [0, 1, 2, 0, 1]
            + [0, 2, 1, 1, 2] + [1] + [0, 2, 1, 1, 2] + [1] + [0, 2, 1, 1, 2]))

    # Verify XYZ=iI
    t = circuit2zx(quantum.Z >> quantum.Y >> quantum.X)
    t = t.to_pyzx().to_matrix() - 1j * np.eye(2)
    assert np.isclose(np.linalg.norm(t), 0)

    # Check scalar translation
    t = circuit2zx(
        quantum.X >> quantum.X @ quantum.scalar(1j)).to_pyzx().to_matrix()
    assert np.isclose(np.linalg.norm(t - 1j * np.eye(2)), 0)

    with raises(NotImplementedError):
        circuit2zx(quantum.scalar(1, is_mixed=True))

    t = circuit2zx(Ket(0)).to_pyzx().to_matrix() - _std_basis_v(0)
    assert np.isclose(np.linalg.norm(t), 0)
    t = circuit2zx(Ket(0, 0)).to_pyzx().to_matrix() - _std_basis_v(0, 0)
    assert np.isclose(np.linalg.norm(t), 0)

    assert (circuit2zx(quantum.Id(3).CX(0, 2))
            == Diagram.decode(
                dom=Nat(3),
                boxes_and_offsets=zip(
                    [SWAP, Z(1, 2), X(2, 1), scalar(2 ** 0.5), SWAP],
                    [1, 0, 1, 2, 1])))


def matrix(diagram):
    """ The matrix of a diagram with outputs as rows, as pyzx has it. """
    n, m = len(diagram.dom), len(diagram.cod)
    return diagram.eval().array.reshape(2 ** n, 2 ** m).T


def proportional(left, right):
    left, right = np.asarray(left).flatten(), np.asarray(right).flatten()
    i = np.argmax(abs(right))
    return abs(left[i]) > 1e-9\
        and np.allclose(left * right[i], right * left[i])


def test_dim():
    assert Diagram.dim(Nat(3)) == Dim(2, 2, 2) and Diagram.dim(Nat(0)) == Dim()


def test_eval():
    for spider, pauli in [(Z, quantum.Z), (X, quantum.X), (Y, quantum.Y)]:
        assert spider(1, 1, .5).eval().is_close(pauli.eval())
    hadamard = H.eval()
    assert hadamard.dom == hadamard.cod == Dim(2)
    assert not np.isnan(hadamard.array).any()
    assert (hadamard >> hadamard.dagger()).is_close(Tensor[complex].id(Dim(2)))
    assert (H >> H).eval().is_close(Tensor[complex].id(Dim(2)))
    for phase in (0, .25, 1 / 3):
        assert X(1, 2, phase).eval().is_close(
            (H >> Z(1, 2, phase) >> H @ H).eval())
    assert Z(1, 2).eval() == Tensor[complex].spiders(1, 2, Dim(2))
    assert Z(0, 0, .5).eval().is_close(Tensor[complex]([0], Dim(1), Dim(1)))
    assert scalar(1j).eval() == Tensor[complex]([1j], Dim(1), Dim(1))
    assert np.allclose(SWAP.eval().array, Tensor.swap(Dim(2), Dim(2)).array)
    assert np.allclose(
        Diagram.permutation([2, 0, 1]).eval().array,
        Tensor.permutation([2, 0, 1], 3 * [Dim(2)]).array)
    assert (Z(1, 2, .25) >> X(2, 1, .5)).eval().is_close(
        Z(1, 2, .25).eval() >> X(2, 1, .5).eval())


def test_eval_symbolic():
    pytest.importorskip("sympy")
    from sympy import Expr, exp, I, pi
    from sympy.abc import phi
    assert Z(1, 1, phi).eval(dtype=Expr).array[1, 1] == exp(2 * I * pi * phi)


def random_diagram(n_wires, depth, phases=(0, .25, .5, .75, 1 / 3, -.125)):
    """
    A random diagram of spiders of the three colours, Hadamards, swaps and
    scalars, with parallel wires between spiders.
    """
    diagram = Id(n_wires)
    for _ in range(depth):
        offset, choice = random.randrange(len(diagram.cod)), random.random()
        if choice < .2:
            box = H
        elif choice < .3 and len(diagram.cod) - offset >= 2:
            box = SWAP
        elif choice < .4:
            box = scalar(random.choice([2 ** .5, 1j, .5]))
        else:
            n_legs_in = random.randint(0, min(2, len(diagram.cod) - offset))
            n_legs_out = 1 if not n_legs_in else random.randint(0, 2)
            box = random.choice([Z, X, Y])(
                n_legs_in, n_legs_out, random.choice(phases))
        diagram = diagram >> Id(offset) @ box\
            @ Id(len(diagram.cod) - offset - len(box.dom))
        if not diagram.cod:
            diagram = diagram @ Z(0, 1, random.choice(phases))
    return diagram


def test_eval_pyzx():
    pytest.importorskip("pyzx")
    random.seed(0)
    for _ in range(50):
        diagram = random_diagram(random.randint(1, 3), random.randint(1, 8))
        assert np.allclose(matrix(diagram), diagram.to_pyzx().to_matrix())


def test_eval_circuit2zx(random_had_cnot_diagram):
    controlled = [
        gate(.3, distance=distance) for gate in (CU1, CRz, CRx)
        for distance in (-2, -1, 1, 2, 3)]
    gates = [
        quantum.H, quantum.X, quantum.Y, quantum.Z, quantum.S, quantum.T,
        quantum.CX, quantum.CZ, quantum.SWAP, Ket(0, 1), Bra(1, 0),
        quantum.scalar(1j), CU1(.3).dagger(), CRz(.3).dagger(),
        CRx(.3).dagger(), Controlled(quantum.X, distance=-2),
        Controlled(quantum.Z, distance=2), random_had_cnot_diagram(3, 20)]
    for gate in controlled + gates:
        diagram = circuit2zx(gate)
        assert (diagram.dom, diagram.cod)\
            == (Nat(len(gate.dom)), Nat(len(gate.cod)))
        assert diagram.eval().is_close(gate.eval())
    for rotation in (Rz(.3), Rx(.3)):
        assert proportional(circuit2zx(rotation).eval().array,
                            rotation.eval().array)


def test_to_hypergraph():
    diagram = Z(1, 2, .25) >> X(2, 1)
    assert diagram.to_hypergraph().to_diagram() == diagram
    assert diagram.simplify() == diagram
    left = Z(1, 1, .25) @ X(1, 1, .5)
    right = Nat(1) @ X(1, 1, .5) >> Z(1, 1, .25) @ Nat(1)
    assert left != right and frobenius.Equation(left, right)
    assert not frobenius.Equation(left, left >> H @ Nat(1))


def test_to_quimb_and_to_tn():
    pytest.importorskip("quimb")
    tn = pytest.importorskip("tensornetwork")
    diagram = Z(1, 2, .25) @ X(0, 1, .5) >> Id(1) @ SWAP >> X(2, 1, .3) @ H
    quimb = diagram.to_quimb().contract()
    assert np.allclose(
        quimb.data.transpose(*np.argsort(quimb.inds)), diagram.eval().array)
    nodes, edges = diagram.to_tn()
    assert np.allclose(
        tn.contractors.auto(nodes, edges).tensor, diagram.eval().array)


def test_Spider_to_tree():
    from discopy.utils import dumps, loads
    for spider in (Z(1, 1, .5), X(2, 1), Y(1, 2, .25), Z(0, 3)):
        assert loads(dumps(spider)) == spider
    diagram = Z(1, 2, .5) >> X(1, 1) @ H
    assert loads(dumps(diagram)) == diagram
    assert loads(dumps(diagram.to_map())) == diagram.to_map()


def test_delay_is_the_identity():
    for diagram in (Z(1, 1, .5), H, SWAP, Z(1, 2) >> X(2, 1, .25) >> H):
        assert diagram.delay() == diagram == diagram.delay(2)


def test_feedback_is_the_trace():
    f = Z(2, 2, .25)
    assert f.feedback() == f.trace() == f.feedback(mem=Nat(1))
    assert f.feedback(left=True) == f.trace(left=True)


def test_functor_into_zx_rebuilds_spiders():
    x = frobenius.Ty('x')
    F = frobenius.Functor(ob_map={x: Nat(1)}, ar_map={}, cod=Diagram)
    assert F(frobenius.Spider(1, 2, x)) == Z(1, 2)
    assert F(frobenius.Diagram.spiders(2, 1, x @ x)).eval().is_close(
        Diagram.spiders(2, 1, Nat(2)).eval())
    with raises(ValueError):
        Diagram.spider_factory(1, 1, Nat(2))
