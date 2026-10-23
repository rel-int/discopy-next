# -*- coding: utf-8 -*-

"""
The port layout of one generator, and the wirings drawn out of a family's
combinatorics: the builders draw frobenius maps whose loops are traces, and
every layout question has one answer.
"""

from pytest import raises

from discopy import frobenius
from discopy.frobenius import CMap, Ty
from discopy.neural import (
    Orbit, Signature, Sym, from_incidence, from_relation)
from discopy.neural.signature import leg_generators


MESSAGE, PEER, STATE = Ty("message"), Ty("peer"), Ty("state")
HIDDEN, MEMORY = Ty("hidden"), Ty("memory")
CLUE, ANSWER = Ty("clue"), Ty("answer")


def cell(degree: int = 3) -> Signature:
    """
    A cell of a factor graph: one message port per factor it belongs to, plus
    a state, a clue and an answer loop.
    """
    return Signature((
        Orbit(MESSAGE, degree, Sym.PERM), Orbit(STATE, traced=True),
        Orbit(CLUE, traced=True), Orbit(ANSWER, traced=True)))


def peer_cell(peers: int = 20) -> Signature:
    """
    A cell of a clique: one port per peer, plus the two traced states of an
    ``LSTMCell`` and one traced clue.
    """
    return Signature((
        Orbit(PEER, peers, Sym.PERM), Orbit(HIDDEN, traced=True),
        Orbit(MEMORY, traced=True), Orbit(CLUE, traced=True)))


def test_positions_and_loops():
    """
    The outgoing copies of a traced orbit come before the incoming ones, so
    the ``i``-th and the ``arity + i``-th positions of its role are the two
    ends of one loop; a role in two orbits sits in both.
    """
    assert len(cell(3).cod) == 9 and cell(3).positions(MESSAGE) == (0, 1, 2)
    assert cell(3).positions(STATE) == (3, 4)
    assert cell(3).loops() == ((3, 4), (5, 6), (7, 8))
    assert peer_cell(2).loops() == ((2, 3), (4, 5), (6, 7))
    three = Signature((Orbit(STATE, 3, Sym.PERM, traced=True), ))
    assert three.loops() == ((0, 3), (1, 4), (2, 5))
    twice = Signature((Orbit(STATE, traced=True), Orbit(STATE, traced=True)))
    assert twice.positions(STATE) == (0, 1, 2, 3)
    assert twice.loops() == ((0, 1), (2, 3))
    assert cell(0).positions(MESSAGE) == () and len(cell(0).cod) == 6


def test_both_builders_draw_frobenius_maps():
    """
    A triangle from a relation and a square with two factors from an
    incidence are closed maps in ``frobenius``, the one category where a
    wire between two codomain ports type checks, with every port wired
    exactly once; each is the map of a frobenius diagram, its wires the cups
    of the host category.
    """
    node = Signature((Orbit(PEER, 2, Sym.PERM), Orbit(STATE, traced=True)))
    triangle = from_relation(((1, 2), (0, 2), (0, 1)), node)
    assert [len(box.cod) for box in triangle.boxes] == [4, 4, 4]
    factor = Signature((Orbit(PEER, 3, Sym.PERM), ))
    square = from_incidence(((0, 1), ) * 4, node, factor)
    assert [len(box.cod) for box in square.boxes] == [4] * 6
    assert [box.name for box in square.boxes] == ["cell"] * 4 + ["factor"] * 2
    for shape, n_ports in ((triangle, 12), (square, 24)):
        assert isinstance(shape, frobenius.CMap)
        assert shape.n_ports == n_ports
        assert not len(shape.dom) and not len(shape.cod)
        assert shape.edges.is_fixpoint_free_involution()
        assert shape.to_diagram().to_map() == shape
    assert isinstance(factor.box("factor"), frobenius.Box)


def test_the_relation_must_be_symmetric_irreflexive_and_in_range():
    node = Signature((Orbit(PEER, 1), Orbit(STATE, traced=True)))
    factor = Signature((Orbit(PEER, 1), ))
    with raises(ValueError, match="not symmetric"):
        from_relation(((1, ), (0, ), (0, )), node)
    with raises(ValueError, match="related to itself"):
        from_relation(((0, 1), (0, )), node)
    with raises(ValueError, match="more than once"):
        from_relation(((1, 1), (0, 0)), node)
    for relation in (((1, ), (0, 3)), ((-1, ), (0, )), ((1.0, ), (0, ))):
        with raises(ValueError, match="does not exist"):
            from_relation(relation, node)
    for incidence in (((0, -1), ), ((0.0, ), )):
        with raises(ValueError, match="does not exist"):
            from_incidence(incidence, node, factor)


def test_incidence_names_and_signatures():
    """
    Factors may carry one signature each, under one name each; the names
    must be one per factor; and a factor nobody belongs to is a box
    with no ports, a scalar of the map.
    """
    node = Signature((Orbit(MESSAGE, 2, Sym.PERM), Orbit(STATE, traced=True)))
    factor = Signature((Orbit(MESSAGE, 3, Sym.PERM), ))
    readout = Signature((Orbit(MESSAGE, 1), Orbit(CLUE, traced=True)))
    shape = from_incidence(
        ((0, 1), (0, 1)), node, {"factor": factor, "readout": readout},
        factor_name=("factor", "readout"))
    assert [(box.name, len(box.cod)) for box in shape.boxes] \
        == [("cell", 4), ("cell", 4), ("factor", 2), ("readout", 4)]
    with raises(ValueError, match="2 names for 1 factors"):
        from_incidence(((0, ), ), node, factor, factor_name=("a", "b"))
    memberless = from_incidence(((1, ), ), node, factor)
    assert [len(box.cod) for box in memberless.boxes] == [3, 0, 1]


def test_resize_and_what_an_orbit_refuses():
    """
    An orbit is resized by its position, since a role may play in several
    orbits, and a position with no orbit is an error, not a no-op, so a
    builder resizing the wrong orbit hears about it; a leg carries one
    atomic role.
    """
    clique = peer_cell(2)
    assert clique.resize(0, 5).orbits[0].arity == 5
    assert clique.resize(1, 1) == clique
    assert clique.resize(1, 2).positions(HIDDEN) == (2, 3, 4, 5)
    twice = Signature((Orbit(STATE, traced=True), Orbit(STATE, traced=True)))
    assert twice.resize(1, 2).loops() == ((0, 1), (2, 4), (3, 5))
    for index in (4, -1):
        with raises(IndexError):
            clique.resize(index, 2)
    with raises(ValueError):
        Orbit(PEER, -1)
    for role in (HIDDEN @ MEMORY, Ty()):
        with raises(ValueError):
            Orbit(role, traced=True)
    with raises(TypeError):
        Orbit("peer")


def test_the_builders_resize_the_first_orbit_only():
    """
    A node of degree ``d`` gets its first orbit resized to ``d`` and no
    other, even one playing the same role, as does a factor of ``m``
    members.
    """
    node = Signature((Orbit(MESSAGE, 2, Sym.PERM), Orbit(MESSAGE, traced=True)))
    path = from_relation(((1, ), (0, 2), (1, )), node)
    assert [len(box.cod) for box in path.boxes] == [3, 4, 3]
    factor = Signature((Orbit(MESSAGE, 3, Sym.PERM), Orbit(MESSAGE, traced=True)))
    star = from_incidence(((0, ), ) * 4, node, factor)
    assert [len(box.cod) for box in star.boxes] == [3] * 4 + [6]


def test_generators_act_on_legs_and_on_every_copy():
    """
    A leg permutation acts on every copy of the orbit alike, the identity
    on the other orbits: a cyclic orbit has one generator, a
    permutation-symmetric one two, a single leg none.
    """
    cyclic = Signature((Orbit(PEER, 3, Sym.CYCLIC), Orbit(STATE, traced=True)))
    assert [tuple(p.inside) for p in cyclic.generators()] == [(1, 2, 0, 3, 4)]
    traced = Signature((
        Orbit(CLUE, 1), Orbit(STATE, 3, Sym.PERM, traced=True),
        Orbit(PEER, 1)))
    assert [tuple(p.inside) for p in traced.generators()] \
        == [(0, 2, 1, 3, 5, 4, 6, 7), (0, 2, 3, 1, 5, 6, 4, 7)]
    assert leg_generators(Sym.CYCLIC, 2) == [(1, 0)]
    assert cell(1).generators() == [] == leg_generators(Sym.PERM, 0)


def test_repr_round_trips():
    orbit = Orbit(PEER, 3, Sym.PERM, traced=True)
    scope = {"Orbit": Orbit, "Signature": Signature, "Sym": Sym,
             "frobenius": frobenius}
    assert eval(repr(orbit), scope) == orbit
    assert eval(repr(cell(2)), scope) == cell(2)
    assert eval(repr(Sym.CYCLIC), scope) is Sym.CYCLIC
    assert repr(Sym.NONE) == "Sym.NONE" and str(Sym.NONE) == "none"


def test_loop_wires_are_the_traces_of_a_box():
    node = Signature((Orbit(PEER, 1), Orbit(STATE, traced=True)))
    assert node.loop_wires(2) == [((2, 1), (2, 2))]
    assert peer_cell(2).loop_wires(0) \
        == [((0, 2), (0, 3)), ((0, 4), (0, 5)), ((0, 6), (0, 7))]
    box = node.box("cell")
    assert from_relation(((1, ), (0, )), node) == CMap.from_wiring(
        (box, box),
        [((0, 0), (1, 0))] + node.loop_wires(0) + node.loop_wires(1))
