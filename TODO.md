toumix, 11-05:

> I got some spare tokens to spend: let’s use them to polish all things discopy-next. Make a deep review of the code and the infra around it, summarising your findings and suggestions in an issue, fixing any bugs that you can fix without requiring a design decisions from my side.

toumix, 11-06:

> use the questions interface so that I can take the decisions from here

The readings toumix picked for streams, drawing, quantum, optyx and the infra, recorded on #445 (comment 6681874391). Each is one commit with a regression test that fails on its parent where one can fail. S10-20 and S16-1 keep what is there and need no change.

- [ ] S8-10: `Head(f).dagger() == Head(f.dagger())`, and the same for `Tail`
- [ ] S8-13: `sample` sorts outcomes by a canonical key that sorts the elements of sets first
- [ ] S9-9: labels stay at 10pt, and the docs and `DRAWING_DEFAULT` say 10
- [ ] S9-17: TikZ spider styles are keyed by name, fill and shape
- [ ] S10-2: `Parametrized` compares its phase exactly; `repr` keeps rounding for display
- [ ] S14-14: `BitControlledGate` reprs through its public constructor and keeps its gates in `data`
- [ ] S2-11: the `shared` profile reads its repo from an environment variable defaulting to discopy/discopy
- [ ] S16-2, S17-9: the `docs` extra takes `discopy[optyx]`, and egglog joins the `dev` group
- [ ] S16-3: `benchmark_comment.py` skips with a `::notice::` and exits 0 when the base moved
- [ ] S16-9: `project.license` is `BSD-3-Clause AND Apache-2.0`, with all three license files
