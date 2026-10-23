# -*- coding: utf-8 -*-

from pytest import importorskip


def toy():
    """ A lattice net with ``run`` and ``step``, its heads off the lattice. """
    torch = importorskip("torch")
    from discopy.neural.lattice import Heads, project

    class Net(torch.nn.Module):
        def __init__(self, cycles=2):
            super().__init__()
            self.cycles = cycles
            self.body = torch.nn.Linear(3, 3)
            self.bce = torch.nn.Linear(3, 3)
            self.sm = torch.nn.Linear(3, 3)
            self.conf = torch.nn.Linear(3, 1)

        def run(self, x, grad=True):
            with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
                every, answer = [], x
                for _ in range(self.cycles):
                    answer = torch.tanh(self.body(answer))
                    every.append(Heads(
                        self.bce(answer), self.sm(answer), self.conf(answer)))
            return answer, every[-1], tuple(every)

        @torch.no_grad()
        def step(self, x, given, generator=None, theta=.1, theta_cls=.5,
                 temp=1.5):
            _, heads, _ = self.run(x, grad=False)
            return project(x, given, heads, theta, theta_cls, temp, generator)

    class Stream:
        def __init__(self, seed):
            self.generator = torch.Generator().manual_seed(seed)

        def take(self, n):
            y = torch.randint(3, (n, 4), generator=self.generator)
            hot = torch.nn.functional.one_hot(y, 3).float()
            blank = torch.rand(n, 4, 1, generator=self.generator) < .7
            return torch.where(blank, torch.ones_like(hot), hot), hot

    return Net, Stream


def test_decide_and_project():
    torch = importorskip("torch")
    from discopy.neural.lattice import Heads, conflicts, decide, project
    x = torch.tensor([[[1., 0.], [0., 1.]], [[1., 1.], [1., 1.]]])
    sm = torch.tensor([[[0., 5.], [5., 0.]], [[0., 5.], [5., 0.]]])
    generator = torch.Generator().manual_seed(0)
    pinned = decide(x.clone(), sm, torch.tensor([True, False]), 0., generator)
    assert torch.equal(pinned, x)
    heads = Heads(torch.full((2, 2, 2), 5.), sm, torch.full((2, 2, 1), -9.))
    given = x.sum(-1) == 1
    fixed = project(x, given, heads, temp=0., generator=generator,
                    fixed_shape=True)
    assert fixed.solved.tolist() == [True, False]
    assert fixed.decided.tolist() == [False, True]
    assert int(fixed.x[1].sum()) == 3 and not fixed.kill.any()
    found = project(x, given, heads, temp=0., generator=generator)
    assert torch.equal(found.x[0], x[0]) and int(found.x[1].sum()) == 3
    sampled = project(x, given, heads, generator=generator)
    assert int(sampled.x[1].sum()) == 3
    unsure = heads._replace(bce=torch.full((2, 2, 2), -5.))
    killed = project(x, given, unsure)
    assert killed.conflict.tolist() == [False, True]
    assert killed.kill[1].all() and not killed.decided.any()
    assert conflicts(killed.x, x)[1].tolist() == [False, True]


def test_losses():
    torch = importorskip("torch")
    from discopy.neural.lattice import W_CELL, losses
    Net, Stream = toy()
    torch.manual_seed(0)
    x, y = Stream(0).take(5)
    _, _, every = Net().run(x)
    given = x.sum(-1) == 1
    plain, parts = losses(every, x, y, given)
    total, more = losses(every, x, y, given, cell_conf=True)
    assert "cell_conf" not in parts and set(more) == set(parts) | {
        "cell_conf"}
    assert torch.allclose(total, plain + W_CELL * more["cell_conf"])
    plain.backward()


def test_pool():
    torch = importorskip("torch")
    from discopy.neural.lattice import Config, Pool
    from discopy.neural.train import EMA
    Net, Stream = toy()
    torch.manual_seed(1)
    net = Net()
    cfg = Config(batch_size=6, max_age=2, compile=False)
    pool = Pool(net, Stream(1), cfg)
    assert pool.amp_dtype is None
    optimizer = torch.optim.SGD(net.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    generator = torch.Generator().manual_seed(1)
    ema = EMA(net, .9)
    first = pool.iterate(optimizer, scheduler, ema, generator)
    assert set(Pool.COUNTERS) <= set(first) and ema.t == 1
    second = pool.iterate(optimizer, scheduler, None, generator)
    assert second["discarded"] == 6
    assert pool.age.tolist() == [0] * 6 and ema.t == 1
    assert torch.equal(pool.x, pool.x0)
    assert Pool(net, Stream(1), Config(batch_size=2)).losses is not None


def test_solve():
    torch = importorskip("torch")
    from discopy.neural.lattice import SolveConfig, describe, evaluate, solve
    Net, Stream = toy()
    torch.manual_seed(2)
    net = Net()
    x, y = Stream(2).take(5)
    cfg = SolveConfig(max_rounds=3, n_chains=2, batch_size=4, theta_cls=.9)
    out = solve(net, x, y, cfg)
    again = solve(net, x, y, cfg)
    assert out["bucket"] == again["bucket"]
    assert torch.equal(out["puzzle_calls"], again["puzzle_calls"])
    assert len(out["bucket"]) == 5 and (out["puzzle_calls"] > 0).all()
    seen = []

    def prefer(lattices, label):
        seen.append(label.clone())
        return label > 2

    preferred = solve(net, x, y, SolveConfig(
        max_rounds=3, n_chains=2, batch_size=4, sync_every=2),
        grade=lambda board, truth: "graded", prefer=prefer,
        label=torch.arange(5))
    assert set(preferred["bucket"]) <= {"graded", "timeout"}
    assert seen and seen[0].tolist() == [0, 0, 1, 1]
    score = evaluate(net, x, y, cfg)
    assert score["n"] == 5 and describe("test", score).startswith("test: ")
    never = evaluate(net, x[:2], y[:2], SolveConfig(
        max_rounds=2, n_chains=2, batch_size=2, theta_cls=0.))
    assert never["timeout"] == 2 and never["rounds_p50"] == -1.
    assert never["model_calls"] == 4 == 2 * never["calls_p50"]


def test_pool_and_solve_peer_net():
    torch = importorskip("torch")
    from discopy.neural.lattice import Config, Pool, SolveConfig, solve
    from discopy.neural.recursion import PeerNet
    from discopy.neural.train import EMA, adamw, cosine_schedule
    _, Stream = toy()
    torch.manual_seed(4)
    peers = tuple(tuple(j for j in range(4) if j != i) for i in range(4))
    net = PeerNet((4, 6, 8, 5), peers, 3, 2, 3, 1)
    cfg = Config(batch_size=5, max_age=2, compile=False)
    pool = Pool(net, Stream(4), cfg)
    optimizer = adamw(net, cfg.lr, cfg.weight_decay)
    scheduler = cosine_schedule(optimizer, 1, 3)
    ema, generator = EMA(net, cfg.ema_decay), torch.Generator().manual_seed(4)
    stats = [pool.iterate(optimizer, scheduler, ema, generator)
             for _ in range(3)]
    assert sum(step["discarded"] for step in stats) >= 5
    x, y = Stream(5).take(3)
    out = solve(net, x, y, SolveConfig(max_rounds=3, n_chains=2,
                                       batch_size=4))
    assert len(out["bucket"]) == 3 and out["model_calls"] >= 3
