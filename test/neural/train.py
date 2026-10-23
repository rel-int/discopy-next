# -*- coding: utf-8 -*-

from pytest import importorskip, raises


def test_seed_adamw_schedule():
    torch = importorskip("torch")
    from discopy.neural.train import adamw, cosine_schedule, seed_everything
    seed_everything(3)
    first = torch.nn.Linear(3, 2)
    seed_everything(3)
    second = torch.nn.Linear(3, 2)
    assert torch.equal(first.weight, second.weight)
    optimizer = adamw(first, lr=1., weight_decay=.5, betas=(.8, .9))
    decay, rest = optimizer.param_groups
    assert decay["params"] == [first.weight] and rest["params"] == [first.bias]
    assert (decay["weight_decay"], rest["weight_decay"]) == (.5, 0.)
    assert decay["betas"] == (.8, .9) and not decay["fused"]
    scheduler = cosine_schedule(optimizer, warmup=0, total=2, floor=.1)
    rates = []
    for _ in range(4):
        rates.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    assert rates[0] == 1. and abs(rates[1] - .55) < 1e-12
    assert rates[2] == rates[3] == .1


def test_ema():
    torch = importorskip("torch")
    from discopy.neural.train import EMA
    net = torch.nn.Linear(2, 2)
    net.register_buffer("steps", torch.zeros((), dtype=torch.long))
    ema = EMA(net, decay=.9)
    assert "steps" not in ema.shadow
    assert all(torch.equal(ema.weights()[key], value)
               for key, value in net.state_dict().items() if key in ema.shadow)
    live = {key: value.clone() for key, value in net.state_dict().items()}
    with torch.no_grad():
        net.weight.add_(1.)
    for _ in range(3):
        ema.update(net)
    assert ema.t == 3
    assert torch.allclose(ema.weights()["weight"], live["weight"] + 1.)
    with ema.averaged(net):
        assert torch.allclose(net.weight, live["weight"] + 1.)
        with torch.no_grad():
            net.weight.zero_()
    assert torch.equal(net.weight, live["weight"] + 1.)
    ema.decay = 1.
    assert torch.equal(ema.weights()["weight"], ema.shadow["weight"])


def test_reference_and_scales():
    torch = importorskip("torch")
    from discopy.neural.train import (
        positive, ratios, rescale, scales, statistics, update_reference)
    with raises(ValueError):
        positive(-1.)
    assert update_reference(2., 0., .5) == 2.
    assert update_reference(4., 1., 0.) == 1.
    norms = torch.tensor([1., 3., float("nan"), 1.])
    losses = torch.tensor([1., 1., 1., 0.])
    r = ratios(norms, losses)
    assert r[3] == 1. / torch.tensor(torch.finfo(torch.float32).tiny)
    scale = scales(norms, 1., losses)
    assert scale[:3].tolist() == [1., 1., 0.] and 0 < scale[3] < 1e-30
    assert statistics(norms, losses, scale) == {
        "median": 3., "clipped_frac": .5}
    row = torch.ones(4, 2, dtype=torch.float64)
    assert rescale(row, scale).dtype == torch.float64
    assert rescale(row, scale)[2].tolist() == [0., 0.]


def test_clip_rows():
    torch = importorskip("torch")
    from discopy.neural.train import clip_rows
    rows = (torch.tensor([[3.], [0.], [6.]], dtype=torch.float64),
            torch.tensor([[4.], [1.], [8.]], dtype=torch.float64))
    (first, second), stats = clip_rows(
        1., *rows, ref=torch.ones(3), reference=5.)
    assert first.dtype == torch.float64
    assert first.squeeze(-1).tolist() == [3., 0., 3.]
    assert second.squeeze(-1).tolist() == [4., 1., 4.]
    assert stats == {"median": 5., "clipped_frac": 1 / 3}


def test_per_board_backward():
    torch = importorskip("torch")
    from discopy.neural.train import per_board_backward

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Linear(3, 2)
            self.halt = torch.nn.Linear(2, 1)
            self.unused = torch.nn.Linear(2, 2)

    def board_loss(net, x, y):
        hidden = net.body(x)
        loss = (hidden - y).pow(2).mean() + net.halt(hidden.detach()).mean()
        return loss, (hidden.detach(), )

    torch.manual_seed(0)
    net = Net()
    x, y = torch.randn(5, 3), torch.randn(5, 2)
    (hidden, ), stats = per_board_backward(net, board_loss, (x, y), k=1e9)
    assert hidden.shape == (5, 1, 2) and stats["clipped_frac"] == 0.
    assert net.unused.weight.grad is None
    grads = torch.autograd.grad(
        sum(board_loss(net, x[b:b + 1], y[b:b + 1])[0] for b in range(5)) / 5,
        [net.body.weight, net.halt.weight])
    assert torch.allclose(net.body.weight.grad, grads[0])
    assert torch.allclose(net.halt.weight.grad, grads[1])
    net.zero_grad(set_to_none=True)
    net.body.bias.requires_grad_(False)
    per_board_backward(net, board_loss, (x, y), k=1e9)
    assert net.body.bias.grad is None
    assert torch.allclose(net.body.weight.grad, grads[0])
    net.body.bias.requires_grad_(True)
    before = net.halt.weight.grad.clone()
    _, stats = per_board_backward(
        net, board_loss, (x, y), k=1e-6, reference=1.)
    assert stats["clipped_frac"] == 1.
    assert torch.allclose(net.halt.weight.grad, 2 * before)
    assert torch.allclose(net.body.weight.grad, grads[0], atol=1e-5)
