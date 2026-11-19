from fractions import Fraction


def boundaries_in_odd_orders():
    """
    Spiders whose inputs and outputs are listed out of the order of their
    nodes, so that :meth:`from_pyzx` has to move wires in both of its loops.
    pyzx is imported here, not at the top: pytest's ``testpaths`` glob reaches
    this file, and ``--skip-extra`` must collect it without pyzx installed.
    """
    import pyzx  # pylint: disable=import-outside-toplevel
    graph = pyzx.Graph()
    green = graph.add_vertex(pyzx.VertexType.Z)
    red = graph.add_vertex(pyzx.VertexType.X, phase=Fraction(1, 2))
    i0, i1, i2, o0, o1, o2 = (
        graph.add_vertex(pyzx.VertexType.BOUNDARY) for _ in range(6))
    graph.set_inputs((i2, i0, i1))
    graph.set_outputs((o2, o0, o1))
    for edge, etype in [
            ((i0, green), pyzx.EdgeType.SIMPLE),
            ((green, red), pyzx.EdgeType.SIMPLE),
            ((green, o2), pyzx.EdgeType.HADAMARD),
            ((i2, red), pyzx.EdgeType.SIMPLE),
            ((i1, red), pyzx.EdgeType.HADAMARD),
            ((red, o0), pyzx.EdgeType.SIMPLE),
            ((red, o1), pyzx.EdgeType.SIMPLE)]:
        graph.add_edge(edge, etype)
    return graph
