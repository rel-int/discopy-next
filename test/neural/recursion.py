# -*- coding: utf-8 -*-

"""
Recursive networks: what they build and in which order, their two wirings
and two factor layouts, the supervision step and the segment of adaptive
computation time.
"""

from pytest import importorskip, raises

WIDTHS = (4, 6, 8, 5)
TRIANGLE = ((0, 1, 2), )
PATH = ((0, 1), (1, 2))


def lattice(torch, rows=2, cells=3, channels=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(rows, cells, channels, generator=g) > 0.4).float()


def test_build_order():
    torch = importorskip("torch")
    from discopy.neural.recursion import FactorNet, PeerNet
    torch.manual_seed(0)
    net = PeerNet(WIDTHS, ((1, 2), (0, 2), (0, 1)), 3, 2, 3, 1, halt=True)
    names = [name for name, _ in net.named_parameters()]
    assert names[0] == "y0" and names[-2:] == ["halt.weight", "halt.bias"]
    assert [name for name, _ in net.named_children()] == [
        "cell", "refresh", "encoder", "bce", "sm", "conf", "halt"]
    assert net.cell.explicit_gru and net.refresh.explicit_gru
    assert list(dict(net.named_buffers())) == ["peers"]
    assert "peers" not in net.state_dict()
    assert net.shape() == {"widths": dict(zip(
        ("dim", "state_dim", "hidden", "y_dim"), WIDTHS)),
        "rounds": 2, "cycles": 3, "detached": 1}
    factor = FactorNet(WIDTHS, PATH, 3, 2, 3, 1, factor="line")
    assert [name for name, _ in factor.named_children()] == [
        "cell", "line", "refresh", "encoder", "bce", "sm", "conf"]
    assert not factor.cell.explicit_gru and factor.shape()["graph"] == "fg"
    assert list(dict(factor.named_buffers())) == [
        "to_units", "to_cells", "port_valid", "real_degree"]
    regular = FactorNet(WIDTHS, TRIANGLE, 3, 2, 3, 1)
    assert list(dict(regular.named_buffers())) == ["to_units", "to_cells"]
    assert regular.port_valid is None and regular.real_degree is None


def test_same_draws_as_the_boxes():
    torch = importorskip("torch")
    from discopy.neural.cell import DeepSet, Refresh, Site, Widths
    from discopy.neural.recursion import FactorNet
    torch.manual_seed(0)
    net = FactorNet(WIDTHS, PATH, 3, 2, 3, 1, halt=True)
    torch.manual_seed(0)
    w = Widths(*WIDTHS)
    boxes = [Site(w), DeepSet(w), Refresh(w), torch.nn.Linear(3, 4),
             torch.nn.Linear(5, 3), torch.nn.Linear(5, 3),
             torch.nn.Linear(5, 1)]
    torch.nn.Linear(5, 1)
    expected = [p for box in boxes for p in box.parameters()]
    assert all(torch.equal(p, q) for p, q in zip(
        list(net.parameters())[1:-2], expected, strict=True))
    assert torch.equal(torch.rand(3), (torch.manual_seed(0), FactorNet(
        WIDTHS, PATH, 3, 2, 3, 1, halt=True), torch.rand(3))[2])


def test_abstract():
    importorskip("torch")
    from discopy.neural.recursion import Recursion
    net = Recursion(WIDTHS, 3, 2, 3, 1)
    with raises(NotImplementedError):
        net.initial(None)
    with raises(NotImplementedError):
        net.round(None)


def test_initial():
    torch = importorskip("torch")
    from discopy.neural.recursion import FactorNet, PeerNet
    from discopy.neural.wiring import primal
    torch.manual_seed(0)
    peer = PeerNet(WIDTHS, primal(TRIANGLE), 3, 2, 3, 1)
    x = lattice(torch)
    s = peer.initial(x)
    assert type(s).__name__ == "State" and s.units.stride() == (0, 0, 0)
    assert torch.equal(s.clue, peer.encoder(x))
    after = peer.cycle(s)
    assert after.units.shape == (2, 3, 4) and after.answer.shape == (2, 3, 5)
    factor = FactorNet(WIDTHS, TRIANGLE, 3, 2, 3, 1)
    s = factor.initial(x)
    assert s.units.shape == (2, 1, 3, 4) and s.cells.shape == (2, 3, 1, 4)


def test_factor_layouts():
    torch = importorskip("torch")
    from discopy.neural.recursion import FactorNet
    torch.manual_seed(0)
    net = FactorNet(WIDTHS, PATH, 3, 1, 2, 0)
    x = lattice(torch)
    reference = net.cycle(net.cycle(net.initial(x)))
    net.split = True
    s = net.initial(x)
    assert s.units.shape == (2, 3, 4)
    split = net.cycle(net.cycle(s))
    assert torch.allclose(split.answer, reference.answer, atol=1e-5)
    net.pool = lambda embedded: torch.stack([
        embedded[:, :2].sum(1), embedded[:, 1:].sum(1)], 1)
    pooled = net.cycle(net.cycle(net.initial(x)))
    assert torch.allclose(pooled.answer, split.answer, atol=1e-6)


def test_act_step():
    torch = importorskip("torch")
    from discopy.neural.recursion import FactorNet
    torch.manual_seed(0)
    net = FactorNet(WIDTHS, PATH, 3, 2, 4, 2, halt=True)
    x = lattice(torch)
    s, every, halt, final = net.act_step(net.initial(x), x)
    assert len(every) == 2 and final is every[-1]
    assert torch.equal(halt, torch.full((2, 3), -5.))
    sum(e.sum() for e in every).backward()
    assert net.encoder.weight.grad is not None and net.y0.grad is None
    assert net.halt.weight.grad is None
    with torch.no_grad():
        s, every, halt, final = net.act_step(net.initial(x), x)
    assert not final.requires_grad
    s, every, halt, final = net.act_step(net.initial(x), x, grad=False)
    assert not s.state.requires_grad
    net.deep, net.halt_detach = False, False
    net.zero_grad(set_to_none=True)
    s, every, halt, final = net.act_step(net.initial(x), x)
    assert len(every) == 1
    halt.sum().backward()
    assert net.halt.weight.grad is not None
    assert net.cell.emit.weight.grad is not None


def test_window_takes_a_cycle():
    torch = importorskip("torch")
    from discopy.neural.recursion import PeerNet
    torch.manual_seed(0)
    net = PeerNet(WIDTHS, ((1, 2), (0, 2), (0, 1)), 3, 2, 3, 1, halt=True)
    x = lattice(torch)
    calls = []

    def cycle(s):
        calls.append(s)
        return net.cycle(s)
    s, every, _ = net.window(net.prefix(net.initial(x)), x, cycle)
    assert len(calls) == len(every) == 2
    assert all(map(torch.equal, s, net.act_step(net.initial(x), x)[0]))


def test_run():
    torch = importorskip("torch")
    from discopy.neural.lattice import Heads, Projection
    from discopy.neural.recursion import PeerNet
    torch.manual_seed(0)
    net = PeerNet(WIDTHS, ((1, 2), (0, 2), (0, 1)), 3, 2, 3, 1)
    x = lattice(torch)
    state, heads, every = net.run(x)
    assert type(state).__name__ == "State" and state.answer.requires_grad
    assert isinstance(heads, Heads) and len(every) == 2 and heads is every[-1]
    assert heads.bce.shape == heads.sm.shape == (2, 3, 3)
    assert heads.conf.shape == (2, 3, 1)
    sum(h.sum() for h in every[-1]).backward()
    assert net.encoder.weight.grad is not None and net.y0.grad is None
    _, again, _ = net.run(x, grad=False)
    assert not again.sm.requires_grad and torch.equal(again.sm, heads.sm)
    assert torch.equal(net(x, grad=False)[1].sm, heads.sm)
    net.deep = False
    _, last, every = net.run(x)
    assert every == () and torch.equal(last.sm, heads.sm)
    found = net.step(x, x.sum(-1) == 1, torch.Generator().manual_seed(0))
    assert isinstance(found, Projection) and found.x.shape == x.shape


def test_compile():
    torch = importorskip("torch")
    from discopy.neural.recursion import PeerNet
    torch.manual_seed(0)
    net = PeerNet(WIDTHS, ((1, 2), (0, 2), (0, 1)), 3, 2, 3, 1, halt=True)
    x = lattice(torch)
    eager = net.act_step(net.initial(x), x, grad=False)
    assert net.compile(unroll=False, backend="eager") is net
    assert net.cycler is not net.cycle
    compiled = net.act_step(net.initial(x), x, grad=False)
    assert all(map(torch.equal, eager[1], compiled[1]))
    torch.manual_seed(0)
    net = PeerNet(WIDTHS, ((1, 2), (0, 2), (0, 1)), 3, 2, 3, 1)
    keys = list(net.state_dict())
    net.compile(backend="eager")
    assert net.unroller is not net.unroll and list(net.state_dict()) == keys
    assert torch.equal(net.run(x)[1].sm, net.unroll(x)[1].sm)


def test_step_shape_and_compile_options(monkeypatch):
    torch = importorskip("torch")
    from discopy.neural.lattice import project
    from discopy.neural.recursion import PeerNet
    torch.manual_seed(0)
    net = PeerNet(WIDTHS, ((1, 2), (0, 2), (0, 1)), 3, 2, 3, 1)
    x, given = lattice(torch, rows=4), torch.zeros(4, 3, dtype=torch.bool)
    x[0] = torch.eye(3)
    _, heads, _ = net.run(x, grad=False)
    for fixed in (False, True):
        net.fixed_shape = fixed
        mine = net.step(x, given, torch.Generator().manual_seed(1))
        theirs = project(x, given, heads, generator=torch.Generator(
        ).manual_seed(1), fixed_shape=fixed)
        assert all(map(torch.equal, mine, theirs))
    with raises(AttributeError, match="no halting head"):
        net.act_step(net.initial(x), x)
    seen = []
    monkeypatch.setattr(torch, "compile", lambda f, **kw: seen.append(kw) or f)
    net.compile_options["dynamic"] = False
    assert PeerNet.compile_options == {}
    net.compile()
    net.compile(unroll=False, dynamic=True)
    assert seen == [{"dynamic": False, "fullgraph": True}, {"dynamic": True}]


def test_no_differentiated_cycle():
    torch = importorskip("torch")
    from discopy.neural.recursion import PeerNet
    torch.manual_seed(0)
    net = PeerNet(WIDTHS, ((1, 2), (0, 2), (0, 1)), 3, 2, 2, 2, halt=True)
    x = lattice(torch)
    _, heads, every = net.run(x)
    assert every == () and heads.sm.shape == (2, 3, 3)
    _, every, halt, final = net.act_step(net.initial(x), x)
    assert len(every) == 1 and final is every[-1]
