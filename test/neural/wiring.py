# -*- coding: utf-8 -*-

"""
The tables a recursive network gathers along: the primal graph and the
routing of a factor table, and the factor table read back off a drawn map.
"""

from pytest import raises

from discopy.frobenius import Box, CMap, Ty
from discopy.neural.signature import (
    Orbit, Signature, Sym, from_incidence, from_relation)
from discopy.neural.wiring import (
    Routing, from_map, memberships, primal, routing, table_of)

MESSAGE, STATE = Ty("message"), Ty("state")


def sudoku(side: int = 4, box: int = 2) -> tuple:
    """ The rows, the columns and the boxes of a sudoku, each increasing. """
    cells = side * side
    rows = [tuple(range(side * r, side * r + side)) for r in range(side)]
    columns = [tuple(range(c, cells, side)) for c in range(side)]
    boxes = [tuple(sorted(
        side * (box * (b // (side // box)) + i) + box * (b % (side // box)) + j
        for i in range(box) for j in range(box))) for b in range(side)]
    return tuple(rows + columns + boxes)


def test_table_of():
    assert table_of([[0, 1], (1, 2)]) == ((0, 1), (1, 2))
    assert table_of(()) == ()
    with raises(ValueError, match="factor 1 is not sorted and distinct"):
        table_of(((0, 1), (2, 2)))


def test_memberships():
    assert memberships(((0, 2), (1, 2))) == ((0, ), (1, ), (0, 1))
    assert memberships(()) == ()
    with raises(ValueError, match="factor 0 holds node 3"):
        memberships(((1, 3), ), 3)
    with raises(ValueError, match="node 0 belongs to no factor"):
        memberships(((1, 2), ))


def test_primal():
    units = sudoku()
    peers = primal(units)
    assert len(peers) == 16 and {len(p) for p in peers} == {7}
    assert peers[0] == (1, 2, 3, 4, 5, 8, 12)
    for node, others in enumerate(peers):
        assert node not in others and list(others) == sorted(others)
        assert all(node in peers[other] for other in others)
    assert primal(((0, 1), ), 3 - 1) == ((1, ), (0, ))


def test_routing_regular():
    units = sudoku()
    graph = routing(units)
    assert isinstance(graph, Routing)
    assert (graph.degree, graph.arity, graph.valid) == (3, 4, None)
    assert graph.degrees == (3, ) * 16
    assert graph.to_factors == sum(units, ())
    assert graph.to_nodes[:6] == (0, 16, 32, 1, 20, 33)
    for node in range(16):
        for port in graph.to_nodes[3 * node:3 * node + 3]:
            assert graph.to_factors[port] == node


def test_routing_padded():
    graph = routing(((0, 1, 2), (2, 3, 4)))
    assert graph.degrees == (1, 1, 2, 1, 1) and graph.degree == 2
    assert graph.to_nodes == (0, 0, 1, 1, 2, 3, 4, 4, 5, 5)
    assert graph.valid == ((1., 0.), (1., 0.), (1., 1.), (1., 0.), (1., 0.))
    assert routing(((0, 1), ), 2).valid is None
    with raises(ValueError, match="unequal arity"):
        routing(((0, 1), (1, 2, 3)))
    with raises(ValueError, match="not sorted and distinct"):
        routing(((1, 0), ))
    with raises(ValueError, match="belongs to no factor"):
        routing(((0, 1), ), 3)


def test_from_map():
    units = sudoku()
    node = Signature((Orbit(MESSAGE, 3, Sym.PERM), Orbit(STATE, traced=True)))
    unit = Signature((Orbit(MESSAGE, 4, Sym.PERM), ))
    drawn = from_incidence(memberships(units), node, unit)
    assert from_map(drawn) == units
    readout = from_incidence(
        ((0, 1), (0, 1), (1, )), node, unit, "cell", ("unit", "readout"))
    assert from_map(readout) == ((0, 1), (0, 1, 2))
    clique = from_relation(primal(units), node)
    assert from_map(clique) == primal(units)
    renamed = from_relation(primal(units), node, "site")
    assert from_map(renamed, "site") == primal(units)
    with raises(ValueError, match="not closed"):
        from_map(CMap.from_box(Box("cell", Ty(), MESSAGE)))
    factor = Box("unit", Ty(), MESSAGE)
    with raises(ValueError, match="box 0 is wired to a box that is not"):
        from_map(CMap.from_wiring((factor, factor), [((0, 0), (1, 0))]))
