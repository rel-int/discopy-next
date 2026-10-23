import networkx as nx
import pytest

from discopy.optyx.compiler.mbqc import Measurement, OpenGraph
from discopy.optyx.compiler.semm import (
    compile_to_fusion_network, compile_to_semm, decompile_from_semm)


# Generate many random graphs and confirm all of them can be compiled and
# reconstructed correctly.
# NOTE: This fails when the graph only contains one node since our graph
# structure only stores a dictionary of edges so it doesn't contain any
# information in this case. This would require refactoring the graph
# datastructure
@pytest.mark.parametrize("num_vertices", range(2, 5))
def test_fuzz_semm_compiler(num_vertices: int):
    with open(f"test/optyx/graph_data/graph{num_vertices}c.g6", "rb") as f:
        lines = f.readlines()

    graphs = nx.read_graph6(lines)
    meas = {i: Measurement(i, "XY") for i in range(num_vertices - 1)}

    # This choice of inputs and outputs is completely arbitary.
    # Should write more tests with different inputs and output combinations
    inputs = {0}
    outputs = {num_vertices - 1}

    # For some reason nx.read_graph6 returns a list of graphs if there are many
    # graphs, and the actual graph if there is only one graph, so we need to
    # convert it back into a list here
    graphs = graphs if isinstance(graphs, list) else [graphs]

    for graph in graphs:
        og = OpenGraph(graph, meas, inputs, outputs)

        # If the graph does not have gflow, then we omit this case as
        # compilation will fail.
        if og.find_gflow() is None:
            continue

        ins = compile_to_semm(og)
        og_reconstructed = decompile_from_semm(ins, inputs, outputs)

        assert og.isclose(og_reconstructed)


def test_a_graph_without_gflow_does_not_compile():
    triangle = nx.Graph([(0, 1), (1, 2), (0, 2)])
    meas = {i: Measurement(0, "XY") for i in range(2)}
    og = OpenGraph(triangle, meas, {0}, {2})
    assert og.find_gflow() is None
    with pytest.raises(ValueError, match="gflow"):
        compile_to_semm(og)


def test_fusion_network_joins_the_paths_of_its_cover():
    """A claw has no Hamiltonian path: its cover has two paths, joined by
    one new vertex, and the edge the path misses becomes a fusion."""
    claw = nx.Graph([(0, 1), (0, 2), (0, 3)])
    meas = {i: Measurement(0, "XY") for i in range(2)}
    network = compile_to_fusion_network(OpenGraph(claw, meas, {1}, {2, 3}))
    assert sorted(network.path) == [0, 1, 2, 3, 4]
    assert 4 in network.measurements
    path_edges = set(zip(network.path, network.path[1:]))
    fusion_edges = {(f.node1, f.node2) for f in network.fusions}
    for edge in claw.edges:
        assert {edge, edge[::-1]} & (path_edges | fusion_edges)
