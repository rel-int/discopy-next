import random
import tomllib
from fractions import Fraction
from pathlib import Path

import pytest

import numpy as np
from pytest import raises, fixture

from discopy.quantum.gates import CRz, CRx, CU1, Controlled, Rz, Rx
from discopy.quantum.zx import *
from discopy import frobenius, tensor
from discopy.tensor import Dim, Tensor

with open(Path(__file__).parents[2] / "pyproject.toml", "rb") as file:
    QUANTUM = tomllib.load(file)["project"]["optional-dependencies"]["quantum"]
PYZX = next(req.split("==")[1].strip() for req in QUANTUM if "pyzx" in req)


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
        assert spider(Nat(2), 1, .5) == spider(2, 1, .5)
        assert spider(Nat(2), 1, .5).dagger().dagger() == spider(2, 1, .5)


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


def test_to_hypergraph_keeps_colour_and_phase():
    assert Z(1, 1, 0).to_hypergraph() != X(1, 1, 0).to_hypergraph()
    assert Z(1, 1, 0).to_hypergraph() != Z(1, 1, .5).to_hypergraph()
    assert Z(1, 2, .5).to_hypergraph().to_diagram() == Z(1, 2, .5)
    assert tensor.Equation(
        Z(1, 1, .25) @ X(1, 1, .5) >> SWAP, SWAP >> X(1, 1, .5) @ Z(1, 1, .25))


def test_subs():
    pytest.importorskip("pyzx", minversion=PYZX)
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
    pytest.importorskip("pyzx", minversion=PYZX)
    with raises(NotImplementedError):
        Box('f', Nat(1), Nat(1)).to_pyzx()


def test_to_pyzx():
    pytest.importorskip("pyzx", minversion=PYZX)
    assert Diagram.from_pyzx(Z(0, 2).to_pyzx()) == Z(0, 2)


def test_to_pyzx_parallel_wires():
    pytest.importorskip("pyzx", minversion=PYZX)
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
    pytest.importorskip("pyzx", minversion=PYZX)
    for diagram in [
            scalar(2) @ Nat(1), X(1, 2, .75) >> Z(2, 1, 1 / 3),
            Z(0, 0, .3), X(0, 2) >> Z(2, 0, .5), X(0, 2, .25) >> Z(2, 0, 1 / 3),
            Z(1, 1, .25) @ (X(0, 1) >> Z(1, 0, .5))]:
        graph = diagram.to_pyzx()
        assert np.allclose(matrix(diagram), graph.to_matrix())
        assert Diagram.from_pyzx(graph).eval().is_close(diagram.eval())


def test_to_pyzx_Y():
    pytest.importorskip("pyzx", minversion=PYZX)
    for diagram in [
            Y(1, 1, .5), Y(2, 1, 1 / 3), Y(0, 2, -.125), Y(1, 0) @ Nat(1),
            H >> Y(1, 2, .3) >> H @ Nat(1), Y(1, 2, .75) >> Y(2, 1, .3)]:
        graph = diagram.to_pyzx()
        assert np.allclose(matrix(diagram), graph.to_matrix())
        assert Diagram.from_pyzx(graph).eval().is_close(diagram.eval())


def test_to_pyzx_scalar():
    pytest.importorskip("pyzx", minversion=PYZX)
    # Test that a scalar is translated to the corresponding pyzx object.
    k = np.exp(np.pi / 4 * 1j)
    m = (scalar(k) @ scalar(k) @ Nat(1)).to_pyzx().to_matrix()
    m = np.linalg.norm(m / 1j - np.eye(2))
    assert np.isclose(m, 0)


def test_from_pyzx_spider_with_several_outputs():
    pyzx = pytest.importorskip("pyzx", minversion=PYZX)
    graph = pyzx.Graph()
    outputs = [graph.add_vertex(pyzx.VertexType.BOUNDARY, qubit=q, row=3)
               for q in range(3)]
    green = graph.add_vertex(pyzx.VertexType.Z, qubit=0, row=1)
    red = graph.add_vertex(
        pyzx.VertexType.X, qubit=1, row=1, phase=Fraction(1))
    graph.set_inputs(())
    graph.set_outputs(tuple(outputs))
    for spider, output in [
            (green, outputs[0]), (red, outputs[1]), (green, outputs[2])]:
        graph.add_edge((spider, output), pyzx.EdgeType.SIMPLE)
    assert proportional(matrix(Diagram.from_pyzx(graph)), graph.to_matrix())


def test_from_pyzx_boundaries_in_odd_orders():
    pyzx = pytest.importorskip("pyzx", minversion=PYZX)
    from fixtures.zx_graphs import boundaries_in_odd_orders
    graph = boundaries_in_odd_orders()
    assert proportional(matrix(Diagram.from_pyzx(graph)), graph.to_matrix())


def test_from_pyzx_errors():
    pytest.importorskip("pyzx", minversion=PYZX)
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
    pytest.importorskip("pyzx", minversion=PYZX)
    from pyzx import Graph
    path = 'test/fixtures/zx-graph.json'
    graph = Graph.from_json(open(path).read())
    diagram = Diagram.from_pyzx(graph)
    backnforth = lambda diagram: Diagram.from_pyzx(diagram.to_pyzx())
    assert backnforth(diagram) == diagram


def test_backnforth_pyzx_2(random_had_cnot_diagram):
    pytest.importorskip("pyzx", minversion=PYZX)
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
    pytest.importorskip("pyzx", minversion=PYZX)
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
    pytest.importorskip("pyzx", minversion=PYZX)
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
    graph = diagram.to_hypergraph()
    assert str(graph.to_diagram()) == "Z(1, 2, 0.25) >> H @ H >> Z(2, 1) >> H"
    assert graph.to_diagram().to_hypergraph() == graph
    assert graph.to_diagram().eval().is_close(diagram.eval())
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


def test_Bubble_grad():
    pytest.importorskip("sympy")
    from sympy.abc import phi
    bubble = Z(1, 1, phi).bubble()
    assert bubble.grad(phi).arg == bubble
    assert not Z(1, 1).bubble().grad(phi).eval().array.any()


def layers_of(qubits: int, depth: int, seed: int) -> list[Diagram]:
    """ The layers of a random circuit of Hadamards, CNOTs and phases. """
    rng = random.Random(seed)
    layers = []
    for _ in range(depth):
        i = rng.randrange(qubits)
        gate = rng.choice(["H", "phase", "CNOT"]) if qubits > 1\
            else rng.choice(["H", "phase"])
        if gate == "H":
            layer = Diagram.id(i) @ H @ Diagram.id(qubits - i - 1)
        elif gate == "phase":
            phase = rng.randrange(1, 8) / 8
            layer = Diagram.id(i) @ Z(1, 1, phase)\
                @ Diagram.id(qubits - i - 1)
        else:
            i = rng.randrange(qubits - 1)
            layer = Diagram.id(i) @ (Z(1, 2) @ Diagram.id(1)
                                     >> Diagram.id(1) @ X(2, 1))\
                @ Diagram.id(qubits - i - 2)
        layers.append(layer)
    return layers


def circuit_matrix(layers: list[Diagram]) -> np.ndarray:
    """ The matrix of a circuit, the product of those of its layers. """
    size = 2 ** len(layers[0].dom)
    result = np.eye(size)
    for layer in layers:
        result = result @ np.asarray(layer.eval().array).reshape(size, size)
    return result


def contract(graph: Hypergraph) -> np.ndarray:
    """
    The tensor of a graph-like hypergraph, an index for each spider with its
    phase: a Hadamard is a matrix on two, a scalar a number, and a boundary
    port an identity to a fresh index.
    """
    spiders = graph.n_spiders
    operands = [np.ones(()), []]
    for box, (dom, cod) in zip(graph.boxes, graph.box_wires):
        if box == H:
            operands += [np.array([[1, 1], [1, -1]]), list(dom + cod)]
        else:
            operands += [np.array(box.data), []]
    ports = graph.dom_wires + graph.cod_wires
    for i, spider in enumerate(ports):
        operands += [np.eye(2), [spider, spiders + i]]
    for spider, phase in enumerate(graph.phases):
        operands += [np.array([1, np.exp(2j * np.pi * float(phase))]),
                     [spider]]
    return np.einsum(*operands, list(range(spiders, spiders + len(ports))),
                     optimize=True)


def proportional(left, right) -> bool:
    left, right = (np.asarray(x).flatten() for x in (left, right))
    k = np.argmax(abs(left))
    return abs(right[k]) > 1e-9 and np.allclose(
        left / left[k], right / right[k])


def checked(rules):
    """ The rules, each step checked as a match and as an equation. """
    def wrap(rule):
        def result(rewriter, spider):
            found = rule(rewriter, spider)
            if found is None:
                return None
            left, right, match = found
            spiders, boxes = match
            assert left.is_parallel(right)
            assert rewriter.glues(left, match)
            for i, j in enumerate(boxes):
                assert rewriter.boxes[j] == left.boxes[i]
                assert rewriter.box_wires[j] == tuple(tuple(
                    spiders[s] for s in wires)
                    for wires in left.box_wires[i])
            if len(left.dom) <= 12:
                assert proportional(contract(left), contract(right))
            return found
        return result
    return [([wrap(rule) for rule in level], anchors)
            for level, anchors in rules]


def random_graph(seed: int) -> Hypergraph:
    """ A random graph-like hypergraph: Hadamard edges, phases and ports. """
    rng = random.Random(seed)
    n = rng.randint(2, 7)
    edges = [(a, b) for a in range(n) for b in range(a + 1, n)
             if rng.random() < .35]
    phases = [Fraction(rng.choice([0, 0, 4, 2, 1, 3]), 8) for _ in range(n)]
    ports = rng.sample(range(n), rng.randint(0, min(3, n)))
    dom, cod = ports[:len(ports) // 2], ports[len(ports) // 2:]
    return Hypergraph(
        Nat(len(dom)), Nat(len(cod)), len(edges) * (H, ),
        (tuple(dom), tuple(((a, ), (b, )) for a, b in edges), tuple(cod)),
        n * [Nat(1)], phases=phases)


def reduced(graph: Hypergraph) -> Hypergraph:
    """ The checked reduction of a graph, where no rule applies anymore. """
    rewriter = Rewriter(graph)
    rewriter.reduce(checked(RULES))
    result = rewriter.to_hypergraph()
    rewriter = Rewriter(result)
    for level, _ in RULES:
        for rule in level:
            for spider in range(result.n_spiders):
                assert rule(rewriter, spider) is None
    return result


def test_Hypergraph_phases():
    graph = Z(1, 2, .25).to_hypergraph() >> Z(2, 1, .75).to_hypergraph()
    assert graph.phases == (0, ) and not graph.boxes
    assert graph == Diagram.id(1).to_hypergraph()
    assert Z(0, 0, .5).to_hypergraph().phases == (.5, )
    assert Y(1, 1, .5).to_hypergraph() == Y(1, 1, .5).to_x().to_hypergraph()
    assert X(1, 1, .5).to_hypergraph().boxes == (H, H)
    assert Hypergraph.reduce_phase(1.25) == .25


def test_Hypergraph_phases_symbolic():
    sympy = pytest.importorskip("sympy")
    phi = sympy.Symbol("phi")
    graph = Z(1, 1, phi).to_hypergraph() >> Z(1, 1, phi).to_hypergraph()
    assert graph == Z(1, 1, 2 * phi).to_hypergraph()


def test_Hypergraph_to_tree():
    from discopy.utils import dumps, loads
    graph = (Z(1, 2, .25) >> X(2, 1, .5)).to_hypergraph()
    assert loads(dumps(graph)) == graph


def test_Hypergraph_eval():
    for diagram in (Z(1, 2, .25) >> X(2, 1, .5), Y(2, 1, .3) @ H,
                    Z(0, 0, .125) @ scalar(2) @ X(1, 0)):
        graph = diagram.to_hypergraph()
        assert graph.to_diagram().eval().is_close(diagram.eval())
        assert proportional(contract(graph), diagram.eval().array)


@pytest.mark.parametrize("seed", range(24))
def test_full_reduce_is_sound(seed):
    layers = layers_of(2 + seed % 3, 8 + seed, seed)
    graph = Diagram.id(len(layers[0].dom)).then(*layers).to_hypergraph()
    assert proportional(contract(graph), circuit_matrix(layers))
    assert proportional(contract(reduced(graph)), circuit_matrix(layers))


@pytest.mark.parametrize("seed", range(24))
def test_full_reduce_reaches_fixpoint(seed):
    graph = random_graph(seed)
    result, expected = contract(reduced(graph)), contract(graph)
    assert np.allclose(expected, 0) or proportional(result, expected)


def test_full_reduce_example():
    diagram = Z(1, 1, .125) >> H >> H >> Z(1, 1, .125)
    assert str(diagram.to_hypergraph().full_reduce()) == "Z(1, 1, 1/4)"


def test_scalar_spiders_size():
    for spider in (Z(0, 0, .125), X(0, 0, .125)):
        graph = spider.to_hypergraph()
        assert graph.size() == (1, 0, 1)
        assert proportional(contract(graph), spider.eval().array)


def test_size_counts_hadamard_edges():
    zx = pytest.importorskip("pyzx", minversion=PYZX)
    graph = zx.Graph()
    a, b, c = (graph.add_vertex(zx.VertexType.Z) for _ in range(3))
    graph.add_edge(graph.edge(a, b), zx.EdgeType.SIMPLE)
    graph.add_edge(graph.edge(b, c), zx.EdgeType.HADAMARD)
    assert Hypergraph.from_pyzx(graph).size() == (2, 1, 0)


def test_size_hadamard_loops_and_parallel_edges():
    loop = (Z(1, 3) >> Nat(1) @ H @ Nat(1) >> Nat(1) @ Z(2, 0)
            ).to_hypergraph()
    parallel = (Z(1, 2) >> H @ H >> Z(2, 1)).to_hypergraph()
    assert loop.size() == (1, 0, 0) and parallel.size() == (2, 0, 0)
    pytest.importorskip("pyzx", minversion=PYZX)
    for graph in (loop, parallel):
        assert Hypergraph.from_pyzx(graph.to_pyzx()).size() == graph.size()


def test_Hypergraph_to_pyzx_errors():
    pytest.importorskip("pyzx", minversion=PYZX)
    with raises(ValueError):
        Box('f', Nat(1), Nat(1)).to_hypergraph().to_pyzx()


@pytest.mark.parametrize("seed", range(6))
def test_full_reduce_against_pyzx(seed):
    zx = pytest.importorskip("pyzx", minversion=PYZX)
    random.seed(seed)
    graph = zx.generate.CNOT_HAD_PHASE_circuit(4, 60).to_graph()
    ours = Hypergraph.from_pyzx(graph)
    assert zx.compare_tensors(ours.to_pyzx(), graph, preserve_scalar=False)
    result = ours.full_reduce()
    assert zx.compare_tensors(result.to_pyzx(), graph, False)
    zx.full_reduce(graph)
    assert result.size()[2] == Hypergraph.from_pyzx(graph).size()[2]
    again = result.to_pyzx()
    zx.full_reduce(again)
    assert Hypergraph.from_pyzx(again).size() == result.size()


def test_copy_and_supplement():
    legs = Z(1, 3) @ Z(1, 3) >> Diagram.permutation([0, 3, 1, 4, 2, 5], Nat(6))
    examples = {
        "copy": (Z(0, 1) >> H) @ H >> Z(2, 1, .125) >> H,
        "supplement": legs >> Nat(2) @ H @ H @ H @ H
        >> Nat(2) @ Z(2, 0, .125) @ Z(2, 0, .375)}
    for name, diagram in examples.items():
        graph, fired = diagram.to_hypergraph(), []

        def counted(rule):
            def result(rewriter, spider):
                found = rule(rewriter, spider)
                fired.extend([rule.__name__] if found else [])
                return found
            return result
        rewriter = Rewriter(graph)
        rewriter.reduce(checked([
            ([counted(rule) for rule in level], anchors)
            for level, anchors in RULES]))
        result = rewriter.to_hypergraph()
        assert fired == [name] and result.size() == (2, 0, 0)
        assert proportional(contract(result), contract(graph))
