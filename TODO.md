> toumix, live in a session on 2026-11-16 (brief: #553):
>
> > one main change is we are making the presentation more focused on constraint satisfaction problems in general rather than Sudoku-Maze-NQueens-SAT each with their own graph. It would make sense to define a discopy module for CSP, help me brainstorm what it would look like.
>
> > for the maze, is it possible to express it as a CSP where each edge of the maze can be either in the shortest path or not?
>
> > write the brief for discopy.csp

- [x] slice 3 of #553: the predicates and `sudoku`, `queens`, `cnf`, `maze`, their sizes and counts (288; 2, 10, 4; a random formula's truth table; 1), the 9×9 sudoku's 810 primal edges, the maze drawn as a doctest image
- [WIP] @session_01MEJxy5FM3MT3hk7p67LLmW-2026-11-16 17:17 the test with `negation` as equality, which breaks the count; the maze with a cycle; changelog; slice 3 opened
