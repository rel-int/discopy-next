The notebook `docs/notebooks/constraint-satisfaction.md` presents *Tiny
Recursive Message Passing* (arXiv:2610.18649) on `discopy.csp` and
`discopy.neural`: `csp.sudoku`, `csp.queens`, `csp.cnf` and `csp.maze` as
hypergraph diagrams with the maze drawn in the plane, solution counting by
`Language.functor(int)`, arc consistency by `Language.propagate` and solution
checking by `Language.holds`, and the 9x9 Sudoku `PeerNet` (16,485
parameters) and `FactorNet` (16,543 parameters) wired from `csp.scopes` with
one untrained lattice step ([next#600](https://github.com/rel-int/discopy-next/issues/600)).
