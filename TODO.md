> toumix, live in a session on 2026-11-16 (brief: #553):
>
> > one main change is we are making the presentation more focused on constraint satisfaction problems in general rather than Sudoku-Maze-NQueens-SAT each with their own graph. It would make sense to define a discopy module for CSP, help me brainstorm what it would look like.
>
> > for the maze, is it possible to express it as a CSP where each edge of the maze can be either in the shortest path or not?
>
> > write the brief for discopy.csp

- [x] slice 4 of #553: `Language.propagate`, arc consistency to its fixed point; soundness (no value a solution takes is removed) on random instances
- [x] the tree law: on random instances whose incidence graph is a tree, the fixed point is the projection of the solutions; one round alone fails it, a test shows an instance
- [WIP] @session_01MEJxy5FM3MT3hk7p67LLmW-2026-11-16 17:52 on the maze it alone leaves one value per variable; Freuder (JACM 1982) and Dechter–Mateescu (UAI 2003) in docs/discopy.bib; changelog; slice 4 opened
