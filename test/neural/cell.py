# -*- coding: utf-8 -*-

"""
The boxes of a recursive network: their parameters and draws, and the two
operation orders of a site, which agree up to rounding.
"""

from pytest import importorskip, raises

SITE_KEYS = [
    "encode.0.weight", "encode.0.bias", "encode.2.weight", "encode.2.bias",
    "update.weight_ih", "update.weight_hh", "update.bias_ih",
    "update.bias_hh", "norm.weight", "norm.bias", "emit.weight", "emit.bias"]


def inputs(torch, rows=2, sites=3, ports=4, w=(4, 6, 8, 5), seed=0):
    g = torch.Generator().manual_seed(seed)
    dim, state_dim, _, y_dim = w
    return [torch.randn(rows, sites, *shape, generator=g) for shape in (
        (ports, dim), (state_dim, ), (dim, ), (y_dim, ))]


def test_widths():
    importorskip("torch")
    from discopy.neural.cell import Widths, as_widths
    w = Widths(15, 39, 76, 20)
    assert as_widths(w) is not w and as_widths(w) == w
    assert as_widths(w._asdict()) == as_widths(tuple(w)) == w
    with raises(TypeError):
        as_widths((1, 2, 3))


def test_gru_cell():
    torch = importorskip("torch")
    from discopy.neural.cell import gru_cell
    cell = torch.nn.GRUCell(5, 3)
    inp, hidden = torch.randn(7, 5), torch.randn(7, 3)
    assert torch.allclose(gru_cell(cell, inp, hidden), cell(inp, hidden),
                          atol=1e-6)
    batched = torch.func.vmap(lambda i, h: gru_cell(cell, i, h))
    assert torch.allclose(batched(inp, hidden), cell(inp, hidden), atol=1e-6)


def test_site():
    torch = importorskip("torch")
    from discopy.neural.cell import Site, Widths
    torch.manual_seed(0)
    site = Site(Widths(4, 6, 8, 5))
    assert list(site.state_dict()) == SITE_KEYS
    assert site.encode[0].weight.shape == (8, 6 + 4)
    assert site.update.weight_ih.shape == (3 * 6, 8 + 4 + 5)
    reference = site(*inputs(torch))
    site.split = True
    split = site(*inputs(torch))
    assert all(torch.allclose(a, b, atol=1e-6)
               for a, b in zip(reference, split))
    site.explicit_gru = True
    explicit = site(*inputs(torch))
    assert all(torch.allclose(a, b, atol=1e-5)
               for a, b in zip(explicit, split))
    site.split = False
    assert all(map(torch.equal, site(*inputs(torch)),
                   Site.forward(site, *inputs(torch))))


def test_site_pool():
    torch = importorskip("torch")
    from discopy.neural.cell import Site, Widths
    torch.manual_seed(0)
    site = Site(Widths(4, 6, 8, 5))
    messages, state, clue, answer = inputs(torch, ports=2)
    valid = torch.tensor([[1., 1.], [1., 0.], [1., 1.]])
    degree = valid.sum(1)
    padded = messages.clone()
    padded[:, 1, 1] = padded[:, 1, 0]
    masked = site(padded, state, clue, answer, valid, degree)
    one_port = site(padded[:, 1:2, :1], state[:, 1:2], clue[:, 1:2],
                    answer[:, 1:2])
    assert torch.allclose(masked[0][:, 1:2], one_port[0], atol=1e-6)
    site.split = True
    assert torch.allclose(
        site(padded, state, clue, answer, valid, degree)[0], masked[0],
        atol=1e-6)
    full = torch.ones(3, 2)
    assert torch.equal(
        Site.pool(messages.reshape(6, 2, 4), 2, 3, full, full.sum(1)) * 2,
        messages.reshape(6, 2, 4).sum(1))


def test_peer_site():
    torch = importorskip("torch")
    from discopy.neural.cell import PeerSite, Site, Widths
    torch.manual_seed(0)
    site = PeerSite(Widths(4, 6, 8, 5))
    assert list(site.state_dict()) == SITE_KEYS
    peers = torch.tensor([[1, 2], [0, 2], [0, 1]])
    _, state, clue, answer = inputs(torch)
    units = torch.randn(2, 3, 4)
    belief, updated = site(units, state, clue, answer, peers)
    split = Site(Widths(4, 6, 8, 5))
    split.load_state_dict(site.state_dict())
    split.split = True
    expected = split(units[:, peers], state, clue, answer)
    assert torch.allclose(belief, expected[0], atol=1e-6)
    assert torch.allclose(updated, expected[1], atol=1e-6)


def test_deep_set():
    torch = importorskip("torch")
    from discopy.neural.cell import DeepSet, Widths
    torch.manual_seed(0)
    unit = DeepSet(Widths(4, 6, 8, 5))
    assert list(unit.state_dict()) == [
        "phi.0.weight", "phi.0.bias", "rho.0.weight", "rho.0.bias",
        "rho.2.weight", "rho.2.bias"]
    beliefs = torch.randn(2, 3, 5, 4)
    replies = unit(beliefs)
    assert replies.shape == beliefs.shape
    shuffled = unit(beliefs[:, :, [4, 0, 3, 1, 2]])
    assert torch.allclose(shuffled, replies[:, :, [4, 0, 3, 1, 2]],
                          atol=1e-6)
    embedded, member = unit.parts(beliefs.reshape(-1, 4))
    assert embedded.shape == member.shape == (30, 8)
    assert torch.allclose(embedded, unit.phi(beliefs.reshape(-1, 4)),
                          atol=1e-6)
    pooled = unit.pool(embedded.view(6, 5, 8).sum(1)).view(6, 1, 8)
    split = unit.rho[2](torch.relu(member.view(6, 5, 8) + pooled))
    assert torch.allclose(split.view(replies.shape), replies, atol=1e-6)


def test_refresh_and_halt():
    torch = importorskip("torch")
    from discopy.neural.cell import HaltHead, Refresh, Widths
    torch.manual_seed(0)
    refresh = Refresh(Widths(4, 6, 8, 5))
    assert list(refresh.state_dict()) == [
        "module.weight_ih", "module.weight_hh", "module.bias_ih",
        "module.bias_hh", "norm.weight", "norm.bias"]
    state, answer = torch.randn(2, 3, 6), torch.randn(2, 3, 5)
    fused = refresh(state, answer)
    refresh.explicit_gru = True
    assert torch.allclose(refresh(state, answer), fused, atol=1e-5)
    torch.manual_seed(0)
    halt = HaltHead(5)
    torch.manual_seed(0)
    torch.nn.Linear(5, 1)
    assert torch.equal(torch.rand(3), (torch.manual_seed(0), HaltHead(5),
                                       torch.rand(3))[2])
    assert torch.equal(halt.weight, torch.zeros(1, 5))
    assert torch.equal(halt.bias, torch.tensor([-5.]))
    assert halt.read(answer).shape == (2, 3)


def test_count_and_widths_for():
    torch = importorskip("torch")
    from discopy.neural.cell import Site, Widths, count, widths_for
    site = Site(Widths(4, 6, 8, 5))
    site.emit.bias.requires_grad_(False)
    assert count(site) == sum(p.numel() for p in site.parameters()) - 4
    built = []

    def build(widths):
        built.append(widths)
        return Site(widths)
    widths = widths_for(5000, build)
    sizes = [count(Site(w)) for w in built]
    assert sizes[-1] > 2 * 5000 >= sizes[-2] and built[0].dim == 4
    distance = [abs(size - 5000) for size in sizes]
    assert widths == built[distance.index(min(distance))]
    assert built[1] == Widths(5, 13, 26, 7)
    assert widths_for(10 ** 9, lambda w: torch.nn.Linear(1, 1), low=3,
                      high=5) == Widths(3, 8, 16, 4)
    assert widths_for(10 ** 9, lambda w: torch.nn.Linear(1, 1), (2, 4, 8, 2),
                      low=3, high=3) == Widths(3, 6, 12, 3)
