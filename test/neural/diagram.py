# -*- coding: utf-8 -*-

"""
The diagram of a recursive network: its round is the round of the table, its
trace unrolls to the rounds of a cycle, its weights are the network's own.
"""

import pytest

WIDTHS = (4, 6, 8, 5)
TRIANGLE = ((0, 1, 2), )
PATH = ((0, 1), (1, 2))
STAR = ((0, 1), (1, 2), (1, 3))
SUDOKU = (
    tuple(tuple(4 * r + c for c in range(4)) for r in range(4))
    + tuple(tuple(4 * r + c for r in range(4)) for c in range(4))
    + tuple(tuple(4 * (2 * i + r) + 2 * j + c for r in range(2)
                  for c in range(2)) for i in range(2) for j in range(2)))
TOL = dict(rtol=1e-12, atol=1e-12)

CASES = [("peer", TRIANGLE, False), ("peer", SUDOKU, False),
         ("factor", TRIANGLE, False), ("factor", TRIANGLE, True),
         ("factor", PATH, False), ("factor", PATH, True),
         ("factor", STAR, False),
         ("factor", STAR, True),
         ("factor", SUDOKU, False), ("factor", SUDOKU, True)]


def build(kind, units, split=False, halt=False, cell_split=False, gru=False):
    torch = pytest.importorskip("torch")
    from discopy.neural.recursion import FactorNet, PeerNet
    from discopy.neural.wiring import primal
    torch.manual_seed(0)
    if kind == "peer":
        net = PeerNet(WIDTHS, primal(units), 3, 2, 3, 1, halt=halt)
    else:
        net = FactorNet(WIDTHS, units, 3, 2, 3, 1, halt=halt)
        net.split = split
    net.cell.split, net.cell.explicit_gru = cell_split, gru or halt
    return net.double()


def clue(torch, units, seed=1):
    cells = 1 + max(max(unit) for unit in units)
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(2, cells, 3, generator=g) > 0.5).double()


def uniform(torch, state, seed=2):
    g = torch.Generator().manual_seed(seed)
    return type(state)(*(
        2 * torch.rand(t.shape, generator=g, dtype=t.dtype) - 1
        for t in state))


@pytest.mark.parametrize("kind, units, split", CASES)
@pytest.mark.parametrize("cell_split", [False, True])
@pytest.mark.parametrize("gru", [False, True])
def test_round(kind, units, split, cell_split, gru):
    torch = pytest.importorskip("torch")
    from discopy.neural.diagram import legs, of
    net = build(kind, units, split, cell_split=cell_split, gru=gru)
    s = uniform(torch, net.initial(clue(torch, units)))
    given, memory = legs(net, s)
    out = of(net).arg.to_function(torch.Tensor)(*given, *memory)
    expected = legs(net, net.round(s))[1]
    assert len(out) == 2 * len(expected) == 2 * len(memory)
    for got, want in zip(out, expected + expected):
        torch.testing.assert_close(got, want, **TOL)


@pytest.mark.parametrize("kind, units", [("peer", TRIANGLE),
                                         ("factor", PATH)])
def test_halt_adds_no_parameter(kind, units):
    torch = pytest.importorskip("torch")
    from discopy.neural.diagram import legs, of
    from discopy.neural.torch import Module
    net = build(kind, units, halt=True)
    s = net.initial(clue(torch, units))
    found = {id(p) for p in Module(of(net).arg).parameters()}
    assert not found & {id(p) for p in net.halt.parameters()}
    plain = build(kind, units)
    shapes = [[leg.shape for leg in part] for part in legs(net, s)]
    assert shapes == [[leg.shape for leg in part] for part in legs(
        plain, plain.initial(clue(torch, units)))]
    assert len(of(net).mem) == len(of(plain).mem)


@pytest.mark.parametrize("kind, units, split", CASES)
def test_trace(kind, units, split):
    torch = pytest.importorskip("torch")
    from discopy.neural.diagram import legs, of
    net = build(kind, units, split)
    trace = of(net)
    s = net.initial(clue(torch, units))
    given = legs(net, s)[0]
    for rounds in range(1, net.rounds + 2):
        s = net.round(s)
        out = trace.to_function(torch.Tensor, rounds=rounds)(*given)
        for got, want in zip(out, legs(net, s)[1], strict=True):
            torch.testing.assert_close(got, want, **TOL)


@pytest.mark.parametrize("kind, units", [("peer", TRIANGLE),
                                         ("factor", PATH)])
def test_parameters_are_the_nets(kind, units):
    torch = pytest.importorskip("torch")
    from discopy.neural.diagram import of
    from discopy.neural.torch import Module
    net = build(kind, units)
    children = [name for name, _ in net.named_children()]
    keys, state = list(net.state_dict()), torch.get_rng_state()
    module = Module(of(net).arg)
    assert torch.equal(state, torch.get_rng_state())
    assert children == [name for name, _ in net.named_children()]
    assert keys == list(net.state_dict())
    shared = [net.cell] + [getattr(net, name) for name in net.factors]
    expected = [id(p) for unit in shared for p in unit.parameters()]
    found = [id(p) for p in module.parameters()]
    assert sorted(found) == sorted(expected)
    assert len(set(found)) == len(found)


@pytest.mark.parametrize("kind, units, split", CASES)
def test_unrolled_gradients(kind, units, split):
    torch = pytest.importorskip("torch")
    from discopy.neural.diagram import legs, of
    net = build(kind, units, split)
    leaf = clue(torch, units).requires_grad_()
    s = net.initial(leaf)
    given = legs(net, s)[0]
    for _ in range(3):
        s = net.round(s)
    expected = legs(net, s)[1]
    g = torch.Generator().manual_seed(3)
    weights = [torch.rand(t.shape, generator=g, dtype=t.dtype)
               for t in expected]
    out = of(net).to_function(torch.Tensor, rounds=3)(*given)
    image = sum((w * t).sum() for w, t in zip(weights, out, strict=True))
    table = sum((w * t).sum() for w, t in zip(weights, expected))
    wrt = list(net.parameters()) + [leaf]
    got = torch.autograd.grad(image, wrt, retain_graph=True,
                              allow_unused=True)
    want = torch.autograd.grad(table, wrt, allow_unused=True)
    assert any(x is not None for x in want)
    for x, y in zip(got, want, strict=True):
        assert (x is None) == (y is None)
        if x is not None:
            torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-10)


def test_a_state_may_not_be_a_message():
    from discopy.frobenius import Box, CMap, Ty
    from discopy.neural.diagram import MESSAGE, interpret
    from discopy.neural.network import Dims
    box = Box("node", Ty(), MESSAGE @ MESSAGE)
    looped = CMap.from_wiring((box, ), [((0, 0), (0, 1))])
    with pytest.raises(ValueError, match="not a state"):
        interpret(looped, {MESSAGE: Dims(1)}, {"node": abs})


def test_the_peers_must_be_symmetric():
    pytest.importorskip("torch")
    from discopy.neural.diagram import closed, of
    from discopy.neural.recursion import PeerNet
    net = PeerNet(WIDTHS, ((1, 2), (0, 3), (3, 1), (2, 0)), 3, 2, 3, 1)
    for function in (closed, of):
        with pytest.raises(ValueError, match="symmetric"):
            function(net)


def test_factor_named_as_nodes():
    pytest.importorskip("torch")
    from discopy.neural.diagram import of
    net = build("factor", PATH)
    net.factor = "cell"
    with pytest.raises(ValueError, match="named 'cell'"):
        of(net)


@pytest.mark.parametrize("kind, units, split", CASES)
def test_gradients(kind, units, split):
    torch = pytest.importorskip("torch")
    from discopy.neural.diagram import legs, of
    net = build(kind, units, split)
    s = uniform(torch, net.initial(clue(torch, units)))
    given, memory = legs(net, s)
    expected = legs(net, net.round(s))[1]
    g = torch.Generator().manual_seed(3)
    weights = [torch.rand(t.shape, generator=g, dtype=t.dtype)
               for t in expected]
    params = list(net.cell.parameters()) + [
        p for name in net.factors for p in getattr(net, name).parameters()]
    out = of(net).arg.to_function(torch.Tensor)(*given, *memory)
    image = sum((w * t).sum() for w, t in zip(weights, out))
    table = sum((w * t).sum() for w, t in zip(weights, expected))
    for got, want in zip(torch.autograd.grad(image, params),
                         torch.autograd.grad(table, params), strict=True):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("kind, units, split", CASES)
def test_wiring_is_the_table(kind, units, split):
    from discopy.neural.diagram import closed
    from discopy.neural.wiring import from_map, primal
    net = build(kind, units, split)
    assert from_map(closed(net)) == (
        primal(units) if kind == "peer" else units)
