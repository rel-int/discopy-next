# -*- coding: utf-8 -*-

from collections import namedtuple

from pytest import importorskip

State = namedtuple("State", "state clue answer")


def toy():
    """ A recursive net with the interface the pool and the rollout read. """
    torch = importorskip("torch")

    class Net(torch.nn.Module):
        def __init__(self, cycles=3, detached=2):
            super().__init__()
            self.cycles, self.detached = cycles, detached
            self.y0 = torch.nn.Parameter(torch.zeros(4))
            self.cell = torch.nn.Linear(4 + 4 + 4, 4)
            self.encoder = torch.nn.Linear(2, 4)
            self.bce = torch.nn.Linear(4, 2)
            self.sm = torch.nn.Linear(4, 2)
            self.halt = torch.nn.Linear(4, 1)

        @torch.no_grad()
        def initial(self, x):
            rows, cells = x.shape[:2]
            zeros = torch.zeros((), dtype=x.dtype)
            return State(zeros.expand(rows, cells, 4), self.encoder(x),
                         self.y0.expand(rows, cells, 4))

        def cycle(self, s):
            state = torch.tanh(self.cell(torch.cat(s, -1)))
            return s._replace(state=state, answer=s.answer + state)

        def prefix(self, s):
            with torch.no_grad():
                for _ in range(self.detached):
                    s = self.cycle(s)
            return s

        def window(self, s, x, cycle=None):
            s = s._replace(clue=self.encoder(x))
            every = []
            for _ in range(self.cycles - self.detached):
                s = (cycle or self.cycle)(s)
                every.append(self.sm(s.answer))
            return s, tuple(every), self.halt(s.answer.detach()).squeeze(-1)

        def act_step(self, s, x, grad=True):
            with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
                s, every, halt = self.window(self.prefix(s), x)
            return s, every, halt, every[-1]

    class Stream:
        def __init__(self, seed):
            self.generator = torch.Generator().manual_seed(seed)

        def take(self, n):
            y = torch.randint(2, (n, 3), generator=self.generator)
            hot = torch.nn.functional.one_hot(y, 2).float()
            blank = torch.rand(n, 3, 1, generator=self.generator) < .5
            return torch.where(blank, torch.ones_like(hot), hot), hot

    return Net, Stream


def test_terms():
    torch = importorskip("torch")
    from discopy.neural.act import nearest_terms, terms
    torch.manual_seed(0)
    every = tuple(torch.randn(4, 3, 2) for _ in range(2))
    halt = torch.randn(4, 3)
    x = torch.ones(4, 3, 2)
    x[0, 0] = torch.tensor([0., 1.])
    given, target = x.sum(-1) == 1, torch.randint(2, (4, 3))
    target[0, 0] = 1
    single = terms(every, halt, every[-1], x, target, given, .5, 2.)
    nearest = nearest_terms(every, halt, every[-1], x, target.unsqueeze(1),
                            torch.ones(4, 1, dtype=torch.bool), given, .5, 2.)
    assert all(torch.equal(a, b) for a, b in zip(single, nearest))
    loss, ce, q, cell_ok, board = single
    assert bool(cell_ok[0, 0])
    assert torch.allclose(loss, ce + .5 * q)
    assert torch.allclose(board.mean(), loss)
    loss, ce, q, cell_ok, board = terms(
        every, halt, every[-1], x, target, given, board_weight=0.)
    assert torch.allclose(board.mean(), loss)
    wrong = (1 - target).unsqueeze(1)
    sols = torch.cat([wrong, target.unsqueeze(1), wrong], 1)
    mask = torch.tensor([True, True, False]).expand(4, 3)
    assert torch.equal(
        nearest_terms(every, halt, every[-1], x, sols, mask, given)[3],
        nearest_terms(every, halt, every[-1], x, sols[:, :2], mask[:, :2],
                      given)[3])


def test_act_pool():
    torch = importorskip("torch")
    from discopy.neural.act import ActConfig, ActPool
    from discopy.neural.train import EMA, adamw, cosine_schedule
    Net, Stream = toy()
    torch.manual_seed(0)
    net = Net()
    cfg = ActConfig(steps=4, batch_size=6, n_sup=2, slot_clip=5,
                    skip_spike=None)
    pool = ActPool(net, Stream(0), cfg)
    assert net.slot_clip == 5. and net.slot_ref_running is None
    assert pool.depth.tolist() == [0, 1, 0, 1, 0, 1]
    optimizer = adamw(net, cfg.lr, cfg.weight_decay)
    scheduler = cosine_schedule(optimizer, 1, cfg.steps)
    ema = EMA(net, cfg.ema_decay)
    stats = pool.iterate(optimizer, scheduler, ema)
    assert int(stats["halted"]) >= 3 and stats["spike"] == 0.
    assert stats["clip"]["median"] == net.slot_ref_running
    assert net.bce.weight.grad is None and net.y0.grad is None
    assert net.halt.weight.grad is not None
    assert type(pool.state) is State and pool.state.answer.shape == (6, 3, 4)
    assert torch.equal(pool.given, pool.x.sum(-1) == 1)
    assert torch.equal(pool.target, pool.y.argmax(-1))
    pool.cfg.skip_spike = 0.
    weight, reference = net.cell.weight.clone(), net.slot_ref_running
    stats = pool.iterate(optimizer, scheduler, ema)
    assert stats["spike"] == 1. and net.cell.weight.grad is None
    assert torch.equal(net.cell.weight, weight)
    assert net.slot_ref_running == reference


def test_act_pool_unstaggered():
    torch = importorskip("torch")
    from discopy.neural.act import ActConfig, ActPool
    from discopy.neural.train import EMA
    Net, Stream = toy()
    torch.manual_seed(1)
    net = Net()
    cfg = ActConfig(batch_size=4, n_sup=3, stagger_pool=False,
                    halt_threshold=1e9, slot_ref_decay=0.)
    pool = ActPool(net, Stream(1), cfg)
    optimizer = torch.optim.SGD(net.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    x = pool.x
    for depth in (1, 2, 0):
        stats = pool.iterate(optimizer, scheduler, EMA(net, .9))
        assert pool.depth.tolist() == [depth] * 4
    assert int(stats["capped"]) == 4 and net.slot_ref_running is None
    assert not torch.equal(pool.x, x)


def test_rollout():
    torch = importorskip("torch")
    from discopy.neural.act import rollout
    Net, Stream = toy()
    torch.manual_seed(2)
    net = Net()
    x, y = Stream(2).take(5)
    target, given = y.argmax(-1), x.sum(-1) == 1
    plain = rollout(net, x, target, given, (1, 3))
    assert sorted(plain) == [1, 3] and plain[3]["depth"].shape == (5, )
    noisy = [rollout(net, x, target, given, (2, ), sigma=.5,
                     generator=torch.Generator().manual_seed(0))
             for _ in range(2)]
    assert all(torch.equal(noisy[0][2][key], noisy[1][2][key])
               for key in noisy[0][2])
    assert not torch.equal(noisy[0][2]["best_q"], plain[3]["best_q"])
    always = rollout(net, x, target, given, (2, ), threshold=-1e9)
    assert always[2]["depth"].tolist() == [1.] * 5


def test_act_pool_peer_net():
    torch = importorskip("torch")
    from discopy.neural.act import ActConfig, ActPool, rollout
    from discopy.neural.recursion import PeerNet, State
    from discopy.neural.train import EMA, adamw, cosine_schedule
    _, Stream = toy()
    torch.manual_seed(3)
    peers = ((1, 2), (0, 2), (0, 1))
    net = PeerNet((4, 6, 8, 5), peers, 2, 2, 3, 1, halt=True)
    cfg = ActConfig(batch_size=4, n_sup=2, slot_clip=.5)
    pool = ActPool(net, Stream(3), cfg)
    optimizer = adamw(net, cfg.lr, cfg.weight_decay)
    scheduler = cosine_schedule(optimizer, 1, 3)
    ema = EMA(net, cfg.ema_decay)
    for _ in range(3):
        stats = pool.iterate(optimizer, scheduler, ema)
    assert type(pool.state) is State and stats["clip"]["clipped_frac"] > 0
    assert net.bce.weight.grad is None and net.sm.weight.grad is not None
    x, y = Stream(4).take(3)
    out = rollout(net, x, y.argmax(-1), x.sum(-1) == 1, (2, ))
    assert out[2]["cell"].shape == (3, )


def test_act_pool_without_clip():
    torch = importorskip("torch")
    from discopy.neural.act import ActConfig, ActPool
    from discopy.neural.train import EMA, adamw, cosine_schedule
    Net, Stream = toy()
    torch.manual_seed(0)
    net = Net()
    cfg = ActConfig(steps=4, batch_size=6, n_sup=2, slot_clip=None)
    pool = ActPool(net, Stream(0), cfg)
    assert net.slot_clip is None
    optimizer = adamw(net, cfg.lr, cfg.weight_decay)
    stats = pool.iterate(optimizer, cosine_schedule(optimizer, 1, 4),
                         EMA(net, cfg.ema_decay))
    assert stats["clip"] is None and net.slot_ref_running is None
    assert net.cell.weight.grad is not None
    assert not pool.state.answer.requires_grad


def test_act_pool_window_loss():
    torch = importorskip("torch")
    from discopy.neural.act import ActConfig, ActPool, terms
    Net, Stream = toy()

    class Wrong(ActPool):
        def window_loss(self, every, halt, x, target, given):
            return terms(every, halt, every[-1], x, target, given,
                         self.cfg.halt_weight, self.cfg.halt_board_weight)[
                :3] + (torch.ones_like(target, dtype=torch.bool), None)

    for clip in (5, None):
        torch.manual_seed(0)
        cfg = ActConfig(steps=4, batch_size=6, n_sup=2, slot_clip=clip)
        pool = Wrong(Net(), Stream(0), cfg)
        assert pool.segment()[5].all()
        assert not ActPool(Net(), Stream(0), cfg).segment()[5].all()
