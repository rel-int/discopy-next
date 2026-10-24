# -*- coding: utf-8 -*-

"""
Recursive networks run along the blocks of a group action: the block
operators, and every wiring computing what the reference network does.
"""

from pytest import importorskip, raises

from discopy.group import Product, Symmetric, Wreath

WIDTHS = (4, 6, 8, 5)
LINES = Wreath[Symmetric[2], Symmetric[2]]


def sudoku():
    from discopy.neural.wiring import Block
    return sum((Block((2, 2, 2, 2), keep).members()
                for keep in ((0, 1), (2, 3), (0, 2))), ())


def same(torch, s, t):
    assert type(s) is type(t)
    for a, b in zip(s, t):
        assert a.shape == b.shape
        assert torch.allclose(a, b, rtol=0, atol=1e-12)


def test_operators():
    torch = importorskip("torch")
    from discopy.neural.geometric import (
        Shift, pairs, pool, spread, unview, view)
    from discopy.neural.wiring import Block
    x = torch.rand(2, 24, 3, dtype=torch.float64)
    for keep in ((), (0, ), (1, 2), (0, 2), (0, 1, 2)):
        block = Block((2, 3, 4), keep)
        assert torch.equal(unview(block, view(block, x)), x)
        means = pool(block, x, mean=True)
        assert torch.allclose(means * block.size, pool(block, x))
        for m, members in enumerate(block.members()):
            for v in members:
                assert torch.allclose(spread(block, means)[:, v],
                                      means[:, m])
                assert torch.allclose(pairs(block, x, x)[:, v], sum(
                    torch.relu(x[:, v] + x[:, w]) for w in members))
    down, right = Shift((2, 3, 4), 0, 1), Shift((2, 3, 4), 2, -2)
    grid = x.view(2, 2, 3, 4, 3)
    assert torch.equal(down(x).view(2, 2, 3, 4, 3)[:, 0], grid[:, 1])
    assert not down(x).view(2, 2, 3, 4, 3)[:, 1].any()
    assert torch.equal(right(x).view(2, 2, 3, 4, 3)[..., 2:, :],
                       grid[..., :2, :])
    assert not right(x).view(2, 2, 3, 4, 3)[..., :2, :].any()


def test_peer_and_factor():
    torch = importorskip("torch")
    from discopy.neural.geometric import Factors, Geometric, Peers
    from discopy.neural.recursion import FactorNet, PeerNet
    from discopy.neural.wiring import primal
    units, group = sudoku(), Product[LINES, LINES]

    class GeometricPeerNet(Geometric, PeerNet):
        group = Product[LINES, LINES]

    class GeometricFactorNet(Geometric, FactorNet):
        geometry = Factors.of(units, group)

    torch.manual_seed(0)
    x = torch.rand(3, 16, 4, dtype=torch.float64).round()
    for reference, geometric, wiring in (
            (PeerNet, GeometricPeerNet, primal(units)),
            (FactorNet, GeometricFactorNet, units)):
        for halt in (False, True):
            ref = reference(WIDTHS, wiring, 4, 2, 4, 2, halt=halt).double()
            geo = geometric(WIDTHS, wiring, 4, 2, 4, 2, halt=halt).double()
            geo.load_state_dict(ref.state_dict())
            ref.split = True
            s, t = ref.initial(x), geo.initial(x)
            for _ in range(2):
                s, t = ref.cycle(s), geo.cycle(t)
            same(torch, s, t)
            out, mine = ref.run(x), geo.run(x)
            sum(h.sm.sum() + h.bce.sum() for h in out[2]).backward()
            sum(h.sm.sum() + h.bce.sum() for h in mine[2]).backward()
            for p, q in zip(ref.parameters(), geo.parameters()):
                assert (p.grad is None) == (q.grad is None)
                assert p.grad is None or torch.allclose(
                    p.grad, q.grad, rtol=0, atol=1e-12)
    geo = GeometricFactorNet(WIDTHS, units, 4, 2, 4, 2).double()
    s = geo.initial(x)
    assert torch.equal(FactorNet.round(geo, s).units, geo.round(s).units)
    assert GeometricPeerNet(WIDTHS, primal(units), 4, 2, 4, 2).geometry \
        == Peers.of(primal(units), group)
    with raises(ValueError, match="as many peers"):
        Peers.of(((1, ), (0, 2), (1, )), Symmetric[3])
    for wrong in (((1, 1), (0, 2), (1, 0)), ((1, 3), (0, 2), (1, 0))):
        with raises(ValueError, match="distinct points"):
            Peers.of(wrong, Symmetric[3])

    class Padded(Geometric, FactorNet):
        group = Symmetric[3]

    with raises(ValueError, match="padded"):
        Padded(WIDTHS, ((0, 1), (1, 2)), 4, 2, 4, 2)


def test_grid():
    torch = importorskip("torch")
    from discopy.neural.geometric import Factors, Geometric, Port, Shift
    from discopy.neural.recursion import PortState, Recursion
    from discopy.neural.wiring import Block

    side, shape = 3, (3, 3)
    shifts = tuple(Shift(shape, axis, step)
                   for axis in (0, 1) for step in (-1, 1))
    rows, columns, grid = (Block(shape, keep) for keep in ((0, ), (1, ), ()))

    class Grid(Recursion):
        """ A grid: its neighbours, and its rows, columns and the whole
        grid pooled by a mean, the round written out by hand. """
        def __init__(self, *args):
            super().__init__(*args, factors=("line", "readout"), halt=True)

        @torch.no_grad()
        def initial(self, x):
            rows_, cells, w = len(x), side * side, self.widths
            zeros = torch.zeros((), dtype=x.dtype)
            return PortState(zeros.expand(rows_, cells, 7, w.dim),
                             zeros.expand(rows_, cells, w.dim),
                             zeros.expand(rows_, cells, w.state_dim),
                             self.encoder(x),
                             self.y0.expand(rows_, cells, w.y_dim))

        def reply(self, module, block, beliefs):
            embedded, member = module.parts(beliefs.reshape(-1, 4))
            embedded = embedded.view(len(beliefs), -1, 8)
            member = member.view(len(beliefs), -1, 8)
            pooled = module.pool(view_mean(block, embedded))
            return module.rho[2](torch.relu(member + spread(block, pooled)))

        def round(self, s):
            belief, state = self.cell(s.cells, s.state, s.clue, s.answer)
            replies = [self.reply(self.line, rows, s.units),
                       self.reply(self.line, columns, s.units),
                       self.reply(self.readout, grid, s.units)]
            cells = torch.stack([f(belief) for f in shifts] + replies, 2)
            return PortState(cells, belief, state, s.clue, s.answer)

    from discopy.neural.geometric import pool, spread

    def view_mean(block, x):
        return pool(block, x, mean=True)

    class GeometricGrid(Geometric, Grid):
        geometry = Factors(shifts + tuple(
            Port(name, block, True) for name, block in (
                ("line", rows), ("line", columns), ("readout", grid))))

    torch.manual_seed(1)
    ref = Grid(WIDTHS, 5, 3, 4, 2).double()
    geo = GeometricGrid(WIDTHS, 5, 3, 4, 2).double()
    geo.load_state_dict(ref.state_dict())
    assert GeometricGrid.geometry.modules == ("line", "readout")
    x = torch.rand(2, 9, 5, dtype=torch.float64).round()
    s, t = ref.initial(x), geo.initial(x)
    for _ in range(3):
        s, t = ref.cycle(s), geo.cycle(t)
    same(torch, s, t)
    a, b = ref.act_step(ref.initial(x), x), geo.act_step(geo.initial(x), x)
    assert all(torch.allclose(e, f, rtol=0, atol=1e-12)
               for e, f in zip(a[1], b[1]))

    class Neighbours(Grid):
        """ The grid with its neighbours only, no factor. """
        def __init__(self, *args):
            Recursion.__init__(self, *args, halt=True)

        @torch.no_grad()
        def initial(self, x):
            s = super().initial(x)
            return s._replace(cells=s.cells[:, :, :len(shifts)])

        def round(self, s):
            belief, state = self.cell(s.cells, s.state, s.clue, s.answer)
            cells = torch.stack([f(belief) for f in shifts], 2)
            return PortState(cells, belief, state, s.clue, s.answer)

    class GeometricNeighbours(Geometric, Neighbours):
        geometry = Factors(shifts)

    ref = Neighbours(WIDTHS, 5, 3, 4, 2).double()
    geo = GeometricNeighbours(WIDTHS, 5, 3, 4, 2).double()
    geo.load_state_dict(ref.state_dict())
    s, t = ref.initial(x), geo.initial(x)
    for _ in range(3):
        s, t = ref.cycle(s), geo.cycle(t)
    same(torch, s, t)
