import random
from fractions import Fraction

import numpy as np
import pytest

from discopy.quantum import graphlike as G
from discopy.quantum.zx import Z, X, H, Diagram
from discopy.rigid import Nat


def circuit(qubits: int, depth: int, seed: int) -> list[Diagram]:
    """ The layers of a random circuit of Hadamards, CNOTs and Ts. """
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


def matrix(layers: list[Diagram]) -> np.ndarray:
    """ The matrix of a circuit, the product of those of its layers. """
    size = 2 ** len(layers[0].dom)
    result = np.eye(size)
    for layer in layers:
        result = result @ np.asarray(layer.eval().array).reshape(size, size)
    return result


def tensor(graph) -> np.ndarray:
    """
    The tensor of a graph-like hypergraph, an index for each spider: a
    Hadamard is a matrix on two, a phase a vector on one, and a boundary
    port an identity to a fresh index.
    """
    spiders = graph.n_spiders
    operands = [np.ones(()), []]
    for box, (dom, cod) in zip(graph.boxes, graph.box_wires):
        if box is H:
            operands += [np.array([[1, 1], [1, -1]]), list(dom + cod)]
        else:
            operands += [np.array([1, np.exp(2j * np.pi * box.phase)]),
                         list(dom)]
    ports = graph.dom_wires + graph.cod_wires
    for i, spider in enumerate(ports):
        operands += [np.eye(2), [spider, spiders + i]]
    for spider in range(spiders):
        operands += [np.ones(2), [spider]]
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
                assert proportional(tensor(left), tensor(right))
            return found
        return result
    return [([wrap(rule) for rule in level], anchors)
            for level, anchors in rules]


def random_graph(seed: int) -> G.Hypergraph:
    """ A random graph-like hypergraph: Hadamard edges, phases and ports. """
    rng = random.Random(seed)
    n = rng.randint(2, 7)
    edges = [(a, b) for a in range(n) for b in range(a + 1, n)
             if rng.random() < .35]
    phases = [(s, Fraction(rng.choice([0, 0, 4, 2, 1, 3]), 8))
              for s in range(n)]
    phases = [(s, phase) for s, phase in phases if phase]
    ports = rng.sample(range(n), rng.randint(0, min(3, n)))
    dom, cod = ports[:len(ports) // 2], ports[len(ports) // 2:]
    boxes = [H for _ in edges] + [G.phase_box(phase) for _, phase in phases]
    wires = [((a, ), (b, )) for a, b in edges] + [
        ((s, ), ()) for s, _ in phases]
    return G.Hypergraph(
        Nat(len(dom)), Nat(len(cod)), tuple(boxes),
        (tuple(dom), tuple(wires), tuple(cod)),
        dict.fromkeys(range(n), G.qubit))


def reduced(graph: G.Hypergraph) -> G.Hypergraph:
    """ The checked reduction of a graph, where no rule applies anymore. """
    rewriter = G.Simplifier(graph)
    G.reduce(rewriter, checked(G.RULES))
    result = rewriter.to_hypergraph()
    rewriter = G.Simplifier(result)
    for level, _ in G.RULES:
        for rule in level:
            for spider in range(result.n_spiders):
                assert rule(rewriter, spider) is None
    return result


@pytest.mark.parametrize("seed", range(24))
def test_full_reduce_is_sound(seed):
    layers = circuit(2 + seed % 3, 8 + seed, seed)
    graph = G.from_diagram(Diagram.id(len(layers[0].dom)).then(*layers))
    assert proportional(tensor(graph), matrix(layers))
    assert proportional(tensor(reduced(graph)), matrix(layers))


@pytest.mark.parametrize("seed", range(24))
def test_full_reduce_reaches_fixpoint(seed):
    graph = random_graph(seed)
    result, expected = tensor(reduced(graph)), tensor(graph)
    assert np.allclose(expected, 0) or proportional(result, expected)


def test_full_reduce_example():
    diagram = Z(1, 1, .125) >> H >> H >> Z(1, 1, .125)
    assert str(G.full_reduce(G.from_diagram(diagram)))\
        == "Z(1, 2) >> Z(1, 0, 1/4) @ Nat(1)"


def test_from_diagram_scalar_spiders():
    for spider in (Z(0, 0, .125), X(0, 0, .125)):
        graph = G.from_diagram(spider)
        assert G.size(graph) == (1, 0, 1)
        assert proportional(tensor(graph), spider.eval().array)


def test_size_counts_hadamard_edges():
    zx = pytest.importorskip("pyzx")
    graph = zx.Graph()
    a, b, c = (graph.add_vertex(zx.VertexType.Z) for _ in range(3))
    graph.add_edge(graph.edge(a, b), zx.EdgeType.SIMPLE)
    graph.add_edge(graph.edge(b, c), zx.EdgeType.HADAMARD)
    assert G.size(graph) == (3, 1, 0)


def test_from_diagram_errors():
    from discopy.quantum.zx import Y
    with pytest.raises(ValueError):
        G.from_diagram(Y(1, 1, .5))


@pytest.mark.parametrize("seed", range(6))
def test_pyzx(seed):
    zx = pytest.importorskip("pyzx")
    random.seed(seed)
    graph = zx.generate.CNOT_HAD_PHASE_circuit(4, 60).to_graph()
    ours = G.from_pyzx(graph)
    assert zx.compare_tensors(G.to_pyzx(ours), graph, preserve_scalar=False)
    result = G.full_reduce(ours)
    assert zx.compare_tensors(G.to_pyzx(result), graph, False)
    zx.full_reduce(graph)
    assert G.size(result)[2] == G.size(graph)[2]
    again = G.to_pyzx(result)
    zx.full_reduce(again)
    assert G.size(again) == G.size(result)


def test_copy_and_supplement():
    legs = Z(1, 3) @ Z(1, 3) >> Diagram.permutation([0, 3, 1, 4, 2, 5], Nat(6))
    examples = {
        "copy": (Z(0, 1) >> H) @ H >> Z(2, 1, .125) >> H,
        "supplement": legs >> Nat(2) @ H @ H @ H @ H
        >> Nat(2) @ Z(2, 0, .125) @ Z(2, 0, .375)}
    for name, diagram in examples.items():
        graph, fired = G.from_diagram(diagram), []

        def counted(rule):
            def result(rewriter, spider):
                found = rule(rewriter, spider)
                fired.extend([rule.__name__] if found else [])
                return found
            return result
        rewriter = G.Simplifier(graph)
        G.reduce(rewriter, checked([
            ([counted(rule) for rule in level], anchors)
            for level, anchors in G.RULES]))
        result = rewriter.to_hypergraph()
        assert fired == [name] and G.size(result) == (2, 0, 0)
        assert proportional(tensor(result), tensor(graph))
