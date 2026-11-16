> toumix, live in a session on 2026-11-16 (brief: #553):
>
> > one main change is we are making the presentation more focused on constraint satisfaction problems in general rather than Sudoku-Maze-NQueens-SAT each with their own graph. It would make sense to define a discopy module for CSP, help me brainstorm what it would look like.
>
> > for the maze, is it possible to express it as a CSP where each edge of the maze can be either in the shortest path or not?
>
> > write the brief for discopy.csp

- [x] slice 2 of #553: `csp.instance`, `csp.scopes`, the round trip
- [x] `Language`: `holds`, `solutions` (backtracking), `functor` (a `tensor.Functor`), the counting law and the projection law on random instances
- [ ] the references checked and in `docs/discopy.bib`; `discopy.csp` in `docs/api/semantics.rst` beside `discopy.tensor`, the import list, one README bullet, the changelog; slice 2 opened
