# -*- coding: utf-8 -*-

"""
The diagram of a recursive network: its round is the round of the table, its
trace unrolls to the rounds of a cycle, its weights are the network's own.
"""

from collections import Counter
from itertools import combinations

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


K4 = tuple(combinations(range(4), 2))
K33 = tuple((i, 3 + j) for i in range(3) for j in range(3))
ONE = ((0, 1), )
DISTINCT = ((0, 4, 5), (1, 4, 5), (2, 4, 5), (3, 4, 5), (2, 3, 5),
            (1, 3, 5), (2, 3, 4))
TWO = ((0, 1), (1, 2), (1, 3), (2, 3))
STACKED = [("peer", TRIANGLE), ("peer", SUDOKU), ("factor", TRIANGLE),
           ("factor", SUDOKU), ("factor", PATH), ("factor", STAR),
           ("factor", K4), ("factor", K33), ("factor", ONE),
           ("factor", DISTINCT), ("factor", TWO)]
CELLS = pytest.mark.parametrize("cell_split, gru, halt", [
    (False, False, False), (True, False, False), (False, True, False),
    (True, True, False), (False, False, True)])


def regular(kind, units):
    """ Whether the nodes have one degree and the factors one arity. """
    degrees = Counter(n for unit in units for n in unit)
    return kind == "factor" and len(set(degrees.values())) == 1\
        and len({len(unit) for unit in units}) == 1


def shaped(out, image):
    """ The shapes of the legs of an image: those of its types. """
    return [tuple(t.shape) for t in out] == [
        (2, *dim.inside) for dim in image.mem.inside]


def grid(n):
    """ The rows and the columns of a square of ``n * n`` nodes. """
    rows = tuple(tuple(n * r + c for c in range(n)) for r in range(n))
    return rows + tuple(tuple(n * r + c for r in range(n))
                        for c in range(n))


def named(trace):
    return [box.name for box in trace.arg.boxes if box.module is not None]


@pytest.mark.parametrize("kind, units", STACKED)
@CELLS
def test_stack_round(kind, units, cell_split, gru, halt):
    torch = pytest.importorskip("torch")
    from discopy.neural.diagram import legs, of, pack, stack
    net = build(kind, units, halt=halt, cell_split=cell_split, gru=gru)
    s = uniform(torch, net.initial(clue(torch, units)))
    given, memory = legs(net, s)
    trace, image = of(net), stack(of(net))
    packed = pack(trace, given, memory)
    out = image.arg.to_function(torch.Tensor)(*packed[0], *packed[1])
    want = pack(trace, given, legs(net, net.round(s))[1])[1]
    assert len(out) == 2 * len(want)
    assert shaped(out[:len(want)], image)
    for got, expected in zip(out, want + want, strict=True):
        if regular(kind, units):
            assert torch.equal(got, expected)
        else:
            torch.testing.assert_close(got, expected, **TOL)


@pytest.mark.parametrize("units", [SUDOKU, K4, K33])
def test_stack_regular_layout(units):
    torch = pytest.importorskip("torch")
    from discopy.neural.diagram import legs, of, pack
    net = build("factor", units)
    s = uniform(torch, net.initial(clue(torch, units)))
    memory = pack(of(net), *legs(net, s))[1]
    assert len(memory) == net.degree + 1 + net.arity
    for k in range(net.degree):
        assert torch.equal(memory[k], s.cells[:, :, k])
    assert torch.equal(memory[net.degree], s.state)
    for j in range(net.arity):
        assert torch.equal(memory[net.degree + 1 + j], s.units[:, :, j])


@pytest.mark.parametrize("kind, units", STACKED)
@CELLS
def test_stack_trace(kind, units, cell_split, gru, halt):
    torch = pytest.importorskip("torch")
    from discopy.neural.diagram import legs, of, pack, stack
    net = build(kind, units, halt=halt, cell_split=cell_split, gru=gru)
    trace, image = of(net), stack(of(net))
    s = net.initial(clue(torch, units))
    given, memory = legs(net, s)
    packed = pack(trace, given, memory)[0]
    for rounds in range(1, net.rounds + 2):
        s = net.round(s)
        out = image.to_function(torch.Tensor, rounds=rounds)(*packed)
        want = pack(trace, given, legs(net, s)[1])[1]
        assert shaped(out, image)
        for got, expected in zip(out, want, strict=True):
            torch.testing.assert_close(got, expected, **TOL)


def weighted(torch, tensors, seed=3):
    g = torch.Generator().manual_seed(seed)
    return sum((torch.rand(t.shape, generator=g, dtype=t.dtype) * t).sum()
               for t in tensors)


@pytest.mark.parametrize("kind, units", STACKED)
@pytest.mark.parametrize("rounds", [1, 3])
def test_stack_gradients(kind, units, rounds):
    torch = pytest.importorskip("torch")
    from discopy.neural.diagram import legs, of, pack, stack
    net = build(kind, units)
    trace, image = of(net), stack(of(net))
    given, memory = legs(net, net.initial(clue(torch, units)))
    leaves = [t.detach().requires_grad_() for t in given]
    packed = pack(trace, leaves, memory)[0]
    out = image.to_function(torch.Tensor, rounds=rounds)(*packed)
    reference = trace.to_function(torch.Tensor, rounds=rounds)(*leaves)
    want = pack(trace, leaves, reference)[1]
    assert shaped(out, image)
    wrt = [*net.parameters(), *leaves]
    got = torch.autograd.grad(weighted(torch, out), wrt, retain_graph=True,
                              allow_unused=True)
    expected = torch.autograd.grad(weighted(torch, want), wrt,
                                   allow_unused=True)
    assert expected[-1] is not None
    for a, b in zip(got, expected, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("units, degrees", [
    (grid(3), 1), (grid(6), 1), (STAR, 2), (DISTINCT, 6), (K4, 1)])
@pytest.mark.parametrize("rounds", [1, 3])
def test_stack_calls_each_module_once_per_bucket(units, degrees, rounds):
    torch = pytest.importorskip("torch")
    from discopy.neural.diagram import legs, of, pack, stack
    net = build("factor", units)
    trace = of(net)
    s = net.initial(clue(torch, units))
    packed = pack(trace, *legs(net, s))[0]
    image = stack(trace)
    modules = {box.name: box.module for box in image.arg.boxes
               if box.module is not None}
    calls = {"cell": 0, "unit": 0}
    hooks = [modules[name].register_forward_hook(
        lambda *_, name=name: calls.update({name: calls[name] + 1}))
        for name in calls]
    image.to_function(torch.Tensor, rounds=rounds)(*packed)
    for hook in hooks:
        hook.remove()
    assert calls == {"cell": degrees * rounds, "unit": rounds}


def test_stack_boxes_do_not_grow_with_the_graph():
    pytest.importorskip("torch")
    from discopy.neural.diagram import of, stack
    sizes = [(len(named(of(net))), named(stack(of(net)))) for net in (
        build("factor", grid(n)) for n in (3, 6))]
    assert sizes == [(15, ["cell", "unit", "gather"]),
                     (48, ["cell", "unit", "gather"])]


def test_stack_has_the_parameters_of_the_net():
    pytest.importorskip("torch")
    from discopy.neural.diagram import of, stack
    from discopy.neural.torch import Module
    net = build("factor", PATH)
    found = [id(p) for p in Module(stack(of(net)).arg).parameters()]
    shared = [net.cell] + [getattr(net, name) for name in net.factors]
    assert sorted(found) == sorted(
        id(p) for unit in shared for p in unit.parameters())


def test_stack_mixed_degrees_have_a_box_each():
    pytest.importorskip("torch")
    from discopy.neural.diagram import buckets, of, stack
    trace = of(build("factor", STAR))
    assert [len(group) for group in buckets(trace)[1]] == [3, 1, 3]
    assert named(stack(trace)) == ["cell", "cell", "unit", "gather"]


def test_stack_buckets_are_nodes_then_factors():
    pytest.importorskip("torch")
    from discopy.neural.diagram import buckets, of
    parts, groups = buckets(of(build("factor", K4)))
    assert [len(group) for group in groups] == [4, 6]
    assert [parts[group[0]].name for group in groups] == ["cell", "unit"]


def test_stack_one_node_has_no_node_axis():
    pytest.importorskip("torch")
    from discopy.neural.diagram import fold, of, stack
    box = [b for b in of(build("factor", ONE)).arg.boxes if b.module][-1]
    assert fold(box, 1) is box
    assert fold(box, 2).dom != box.dom
    assert all(len(dim.inside) == 1
               for dim in stack(of(build("factor", ONE))).mem.inside[2:])


def test_stack_same_name_other_module_is_another_bucket():
    pytest.importorskip("torch")
    from discopy.neural.diagram import buckets
    from discopy.neural.network import Box, Dims, Function, Trace
    boxes = [Box("f", Dims(2), Dims(2), module=module)
             for module in (abs, abs, round)]
    body = Function(Dims(2, 2, 2), [0, 1, 2]) >> boxes[0] @ boxes[1] @ boxes[2]
    assert buckets(Trace(body, mem=Dims(2, 2, 2)))[1] == [[0, 1], [2]]


def test_wires_need_a_trace_drawn_by_interpret():
    pytest.importorskip("torch")
    from discopy.neural.diagram import wires
    from discopy.neural.network import Box, Dims, Trace
    with pytest.raises(ValueError, match="interpret"):
        wires(Trace(Box("f", Dims(2), Dims(2), module=abs), mem=Dims(2)))


@pytest.mark.parametrize("cell_split", [False, True])
@pytest.mark.parametrize("gru", [False, True])
def test_fire_on_stacked_nodes_is_the_loop(cell_split, gru):
    torch = pytest.importorskip("torch")
    from discopy.neural.diagram import Fire
    net = build("factor", PATH, cell_split=cell_split, gru=gru)
    fire, g = Fire(net.cell), torch.Generator().manual_seed(4)
    rand = lambda *shape: torch.rand(*shape, generator=g, dtype=torch.double)
    clues, answers, state = rand(2, 3, 4), rand(2, 3, 5), rand(2, 3, 6)
    messages = [rand(2, 3, 4) for _ in range(2)]
    stacked = fire(clues, answers, *messages, state)
    for i in range(3):
        each = fire(clues[:, i], answers[:, i], *(m[:, i] for m in messages),
                    state[:, i])
        for got, want in zip(stacked, each, strict=True):
            torch.testing.assert_close(got[:, i], want, **TOL)


def test_reply_on_stacked_factors_is_the_loop():
    torch = pytest.importorskip("torch")
    from discopy.neural.diagram import Reply
    net = build("factor", PATH)
    reply, g = Reply(getattr(net, net.factor)), torch.Generator().manual_seed(5)
    beliefs = [torch.rand(2, 3, 4, generator=g, dtype=torch.double)
               for _ in range(2)]
    stacked = reply(*beliefs)
    for i in range(3):
        each = reply(*(b[:, i] for b in beliefs))
        for got, want in zip(stacked, each, strict=True):
            assert torch.equal(got[:, i], want)


def test_gather_is_a_module_with_a_public_wiring():
    torch = pytest.importorskip("torch")
    from discopy.neural.diagram import Gather, legs, of, pack, stack
    from discopy.neural.torch import Module
    gather = Gather((((0, 1), (2, 1)), ((1,), (0, )), ((0,), (0, 1))), (2, 1))
    assert eval(repr(gather), {"Gather": Gather}).wired == gather.wired
    assert gather.index.tolist() == [2, 1, 0, 0, 1]
    assert gather.offsets == (0, 2, 3, 5)
    a, b = torch.rand(2, 2, 3), torch.rand(2, 3)
    first, second, third = gather(a, b)
    assert second is b and third is a
    assert torch.equal(first, torch.cat([a, b[:, None]], 1)[:, [2, 1]])
    net = build("factor", PATH)
    s = net.initial(clue(torch, PATH))
    trace = of(net)
    packed = pack(trace, *legs(net, s))
    module = Module(stack(trace).arg)
    want = module(*packed[0], *packed[1])
    module.to(torch.float32)
    got = module(*(t.float() for t in (*packed[0], *packed[1])))
    assert all(g.dtype == torch.float32 for g in got)
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(g.double(), w, rtol=1e-4, atol=1e-5)
    if torch.cuda.is_available():
        module.to("cuda")
        out = module(*(t.float().cuda() for t in (*packed[0], *packed[1])))
        assert all(t.is_cuda for t in out)


@pytest.mark.parametrize("kind, units, split", CASES)
def test_wiring_is_the_table(kind, units, split):
    from discopy.neural.diagram import closed
    from discopy.neural.wiring import from_map, primal
    net = build(kind, units, split)
    assert from_map(closed(net)) == (
        primal(units) if kind == "peer" else units)
