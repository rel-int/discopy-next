---
title: Constraint Satisfaction
marimo-version: 0.23.14
---

```python {.marimo}
import marimo as mo
```

# Constraint satisfaction with Tiny Recursive Message Passing

[*Tiny Recursive Message Passing*](https://arxiv.org/abs/2610.18649) solves constraint satisfaction problems with tiny weight-shared message-passing networks trained on the problem's own graph.
In DisCoPy, the problem and the network are two readings of the same diagram:

* [`discopy.csp`](../../_api/discopy.csp.html) writes an instance as a [`frobenius.Hypergraph`](../../_api/discopy.frobenius.Hypergraph.html) with one spider per variable and one box per constraint, and interprets its [`Language`](../../_api/discopy.csp.Language.html) as a tensor functor;
* [`discopy.neural`](../../_api/discopy.neural.html) turns each variable into a stateful site and each constraint symbol into one shared module, so that the parameters are indexed by the signature rather than by the size of the instance.

## 1. A problem is a diagram

An instance of [`discopy.csp`](../../_api/discopy.csp.html) is an arrow of the free hypergraph category on a relational signature: every variable is a spider of its sort, and every constraint is a box plugged onto the tuple of variables in its scope.
Four families from the paper come built in:

* [`csp.sudoku(n)`](../../_api/discopy.csp.sudoku.html) has $n^4$ spiders of sort `digit` (values $0, \dots, n^2 - 1$) and $3 n^2$ `AllDifferent` boxes on the rows, columns and $n \times n$ blocks;
* [`csp.queens(n)`](../../_api/discopy.csp.queens.html) has $n^2$ spiders of sort `bit`, `ExactlyOne` on each row and column, and `AtMostOne` on each diagonal of at least two squares;
* [`csp.cnf(clauses)`](../../_api/discopy.csp.cnf.html) has one `bit` per Boolean variable, one `Or` box per clause, and a fresh `bit` wired through `Not` for every negated literal so that `Or` is one symmetric symbol at every arity;
* [`csp.maze(grid)`](../../_api/discopy.csp.maze.html) has one `bit` per walkable edge saying whether the path takes it, `ExactlyOne` at the start `S` and goal `G`, and `ZeroOrTwo` at every other walkable cell `.`.

```python {.marimo}
from discopy import csp

sudoku4_graph, sudoku4_language = csp.sudoku(2)
queens5_graph, queens5_language = csp.queens(5)
cnf_graph, cnf_language = csp.cnf([[1, -2], [2, 3], [-1, -3]])

MAZE = [
    "S..#.",
    "##.#.",
    ".....",
    ".###.",
    "..G#.",
]
maze_graph, maze_language = csp.maze(MAZE)

problems = [
    ("4×4 Sudoku", sudoku4_graph, sudoku4_language),
    ("5-Queens", queens5_graph, queens5_language),
    ("CNF", cnf_graph, cnf_language),
    ("5×5 Maze", maze_graph, maze_language),
]
assert [(len(_g.spider_types), len(_g.boxes)) for _, _g, _ in problems] == [
    (16, 12), (25, 24), (6, 6), (16, 17)
]

_rows = "\n".join(
    f"| {_name} | {len(_g.spider_types)} | {len(_g.boxes)} "
    f"| {', '.join(f'`{k}`' for k in _lang.predicates)} |"
    for _name, _g, _lang in problems
)
mo.md(f"""
| instance | variables (spiders) | constraints (boxes) | symbols |
|---|---:|---:|---|
{_rows}
""")
```

Because a hypergraph category has cups and caps, a planar diagram whose boxes have both inputs and outputs represents the same instance once converted with `Hypergraph.from_diagram`: [`csp.scopes`](../../_api/discopy.csp.scopes.html) reads each box's inputs followed by its outputs.
[`csp.maze_diagram`](../../_api/discopy.csp.maze_diagram.html) draws the maze directly in the plane, one layer per row of the grid, a vertical wire to the row below and a cup for each horizontal edge:

```python {.marimo}
csp.maze_diagram(MAZE).draw(wire_labels=False, figsize=(5, 5))
```

## 2. Its language is a functor

A [`csp.Language`](../../_api/discopy.csp.Language.html) sends each sort to a finite set `range(size)` and each constraint symbol to its relation.
`Language.functor` turns that interpretation into a [`tensor.Functor`](../../_api/discopy.tensor.Functor.html): over `bool`, an open instance evaluates to the projection of its solutions onto its free output wires; over `int`, a closed instance contracts to the number of its solutions without enumerating them.

```python {.marimo}
counts = {
    _name: _lang.functor(int)(_g).array.item()
    for _name, _g, _lang in problems
}
assert counts == {
    "4×4 Sudoku": 288,
    "5-Queens": 10,
    "CNF": 2,
    "5×5 Maze": 1,
}
mo.md(", ".join(f"**{_k}**: {_v} solution(s)" for _k, _v in counts.items()))
```

Before searching or learning, [`Language.propagate`](../../_api/discopy.csp.Language.html#discopy.csp.Language.propagate) runs arc consistency: each variable starts with a set of candidate values, and each round keeps only the values supported by an assignment of every incident scope.
It never discards a value that some solution takes, and [`Language.holds`](../../_api/discopy.csp.Language.html#discopy.csp.Language.holds) checks whether a complete assignment satisfies every constraint of the instance.

On the tree-shaped maze, `propagate` narrows all 16 edges to singletons — the unique path of length 10 from `S` to `G`.
On a $4 \times 4$ Sudoku with five clues, `propagate` likewise deduces the whole board; drop one clue to restore the symmetry between digits `2` and `3` (values `1` and `2` in code), and arc consistency stops with pairs of candidates on the eight unfixed cells.

```python {.marimo}
maze_domains = maze_language.propagate(maze_graph)
assert all(len(_d) == 1 for _d in maze_domains)
maze_solution = tuple(next(iter(_d)) for _d in maze_domains)
assert sum(maze_solution) == 10
assert maze_language.holds(maze_graph, maze_solution)

_clues = {0: 0, 1: 1, 5: 3, 10: 3, 15: 0}
sudoku4_domains = sudoku4_language.propagate(
    sudoku4_graph,
    [{_clues[_i]} if _i in _clues else set(range(4)) for _i in range(16)],
)
assert all(len(_d) == 1 for _d in sudoku4_domains)
sudoku4_solution = tuple(next(iter(_d)) for _d in sudoku4_domains)
assert sudoku4_language.holds(sudoku4_graph, sudoku4_solution)

_underconstrained = {0: 0, 5: 3, 10: 3, 15: 0}
_ambiguous = sudoku4_language.propagate(
    sudoku4_graph,
    [{_underconstrained[_i]} if _i in _underconstrained else set(range(4))
     for _i in range(16)],
)
assert sum(len(_d) > 1 for _d in _ambiguous) == 8

_grid_rows = "\n".join(
    "    " + " ".join(str(sudoku4_solution[4 * _r + _c] + 1) for _c in range(4))
    for _r in range(4)
)
mo.md(f"""
Maze path edges active: `{sum(maze_solution)} / {len(maze_solution)}` (`holds = True`).

Solved $4 \\times 4$ Sudoku from 5 clues (`holds = True`):

{_grid_rows}
""")
```

## 3. A network is a functor on the same diagram, read with the variable as a box

Where `Language.functor` sends a variable spider to tensor contraction and a constraint box to a fixed relation table, recursive message passing replaces each variable spider with a stateful `cell` box and assigns **one PyTorch module per box name**.
Because every cell shares one [`Site`](../../_api/discopy.neural.cell.Site.html) (or [`PeerSite`](../../_api/discopy.neural.cell.PeerSite.html)) and every constraint of the same symbol shares one permutation-equivariant [`DeepSet`](../../_api/discopy.neural.cell.DeepSet.html), the parameter count is fixed by the signature and the widths `(dim, state_dim, hidden, y_dim)` rather than by the size of the board.

[`discopy.neural.signature`](../../_api/discopy.neural.signature.html) draws both wirings as closed maps in `discopy.frobenius`, and [`discopy.neural.wiring`](../../_api/discopy.neural.wiring.html) compiles the scopes of `csp.sudoku(3)` directly into the gather tables of [`FactorNet`](../../_api/discopy.neural.recursion.FactorNet.html) (on the 27 scopes of 9 cells) and [`PeerNet`](../../_api/discopy.neural.recursion.PeerNet.html) (on their primal graph, 20 peers per cell):

```python {.marimo}
import torch

from discopy import frobenius
from discopy.neural import cell, lattice, recursion, signature, wiring

sudoku_graph, sudoku_language = csp.sudoku(3)
units = [scope for _, scope in csp.scopes(sudoku_graph)]
peers = wiring.primal(units)
assert len(sudoku_graph.spider_types) == 81 and len(units) == 27
assert len(peers) == 81 and {len(_p) for _p in peers} == {20}

_message, _state = frobenius.Ty("message"), frobenius.Ty("state")
_node_sig = signature.Signature((
    signature.Orbit(_message, 3, signature.Sym.PERM),
    signature.Orbit(_state, traced=True),
))
_factor_sig = signature.Signature((
    signature.Orbit(_message, 9, signature.Sym.PERM),
))
_factor_map = signature.from_incidence(
    wiring.memberships(units), _node_sig, _factor_sig
)
_peer_map = signature.from_relation(peers, _node_sig)
assert wiring.from_map(_factor_map) == tuple(units)
assert wiring.from_map(_peer_map) == peers

torch.manual_seed(0)
peer_net = recursion.PeerNet(
    (10, 27, 55, 14), peers, 9, rounds=5, cycles=10, detached=8
)
factor_net = recursion.FactorNet(
    (9, 24, 48, 12), units, 9, rounds=5, cycles=10, detached=8
)

assert sum(_p.numel() for _p in peer_net.parameters()) == 16_485
assert sum(_p.numel() for _p in factor_net.parameters()) == 16_543
assert cell.count(peer_net) == 16_485 and cell.count(factor_net) == 16_543

mo.md(f"""
| network | wiring | boxes in map | widths `(dim, state, hidden, y)` | parameters |
|---|---|---:|---|---:|
| `PeerNet` | `wiring.primal(units)` (20 peers/cell) | {len(_peer_map.boxes)} | `{tuple(peer_net.widths)}` | `{cell.count(peer_net):,}` |
| `FactorNet` | `units` (27 scopes of 9) | {len(_factor_map.boxes)} | `{tuple(factor_net.widths)}` | `{cell.count(factor_net):,}` |
""")
```

In [`discopy.neural.lattice`](../../_api/discopy.neural.lattice.html), a $9 \times 9$ Sudoku puzzle is carried as a candidate lattice `x` in $\{0, 1\}^{B \times 81 \times 9}$: each given clue is one-hot and each blank cell starts with all nine digits alive.
One supervision step (`factor_net.step`) runs 10 cycles of 5 message-passing rounds, reads the keep/kill (`bce`), symbol (`sm`) and conflict (`conf`) heads off the answer trace, kills candidates below threshold (never touching a given), and pins one multi-candidate cell when no conflict fires.

At test time, [`lattice.solve`](../../_api/discopy.neural.lattice.solve.html) steps parallel chains of candidate lattices, resetting chains that enter conflict, and grades each accepted board against its stored solution with `grade`, `lattice.exact` by default.
`sudoku_language.holds(sudoku_graph, assignment)` checks a board against the constraints themselves instead, with no stored solution.
Below is one untrained step on a 30-clue $9 \times 9$ board, showing the candidate-set sizes ($1$–$9$) per cell before and after projection:

```python {.marimo}
import time

PUZZLE = (
    "530070000"
    "600195000"
    "098000060"
    "800060003"
    "400803001"
    "700020006"
    "060000280"
    "000419005"
    "000080079"
)
x0 = lattice.lattice_of([[int(_c) for _c in PUZZLE]])
given = x0.sum(-1) == 1
assert x0.shape == (1, 81, 9) and int(given.sum()) == 30

_t0 = time.perf_counter()
projection = factor_net.step(
    x0.clone(), given, generator=torch.Generator().manual_seed(0)
)
step_ms = (time.perf_counter() - _t0) * 1e3

assert torch.equal(projection.x[given], x0[given])
_untrained_guess = projection.x[0].argmax(-1).tolist()
assert not sudoku_language.holds(sudoku_graph, _untrained_guess)


def _format_counts(tensor):
    _alive = tensor[0].sum(-1).to(torch.int64).tolist()
    return "\n".join(
        "    " + " ".join(str(_alive[9 * _r + _c]) for _c in range(9))
        for _r in range(9)
    )


mo.md(f"""
One untrained `FactorNet` step (10 cycles × 5 rounds): `{step_ms:.1f} ms` on CPU —
alive candidates `{int(x0.sum())} -> {int(projection.x.sum())}`,
killed `{int(projection.kill.sum())}`,
decided `{bool(projection.decided[0])}`,
conflict `{bool(projection.conflict[0])}`,
`holds = {sudoku_language.holds(sudoku_graph, _untrained_guess)}`.

Candidate counts per cell before the step (`1` = given clue, `9` = blank):

{_format_counts(x0)}

Candidate counts per cell after one untrained step:

{_format_counts(projection.x)}
""")
```
