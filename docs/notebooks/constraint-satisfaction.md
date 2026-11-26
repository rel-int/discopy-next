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

Before searching or learning, [`Language.propagate`](../../_api/discopy.csp.Language.propagate.html) runs arc consistency: each variable starts with a set of candidate values, and each round keeps only the values supported by an assignment of every incident scope.
It never discards a value that some solution takes, and [`Language.holds`](../../_api/discopy.csp.Language.holds.html) checks whether a complete assignment satisfies every constraint of the instance.

On the tree-shaped maze, `propagate` narrows all 16 edges to singletons — the unique path of length 10 from `S` to `G`.
On a $4 \times 4$ Sudoku with five clues, `propagate` likewise deduces the whole board; drop one clue to restore the symmetry between digits `1` and `2`, and arc consistency stops with pairs of candidates on the unfixed cells.

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
