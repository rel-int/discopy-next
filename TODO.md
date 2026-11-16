> toumix, live in a session on 2026-11-16 (brief: #553):
>
> > one main change is we are making the presentation more focused on constraint satisfaction problems in general rather than Sudoku-Maze-NQueens-SAT each with their own graph. It would make sense to define a discopy module for CSP, help me brainstorm what it would look like.
>
> > for the maze, is it possible to express it as a CSP where each edge of the maze can be either in the shortest path or not?
>
> > write the brief for discopy.csp

- [WIP] @session_01MEJxy5FM3MT3hk7p67LLmW-2026-11-16 15:25 slice 1 of #553: `tensor.Functor` contracts a `Hypergraph`, one index per spider, one array per box, the boundary as free indices, a spider on no port counts its sort
- [ ] the law `F(graph) == F(graph.to_diagram())` on small random frobenius hypergraphs with a boundary, each dtype
- [ ] the 4×4 sudoku counted as 288 by `F(graph)` (through `to_diagram()` it raises `iterator is too large` on main)
- [ ] changelog; next-checks --full and the notebooks on the merged tree; slice 1 opened
