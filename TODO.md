> No human prompt: [#536](https://github.com/rel-int/discopy-next/issues/536) is a bug in this
> repo whose reading 1 chooses nothing, so `AGENTS.md`'s *Next repos* has it fixed without asking
> (the ask was withdrawn in place, [#536 comment](https://github.com/rel-int/discopy-next/issues/536#issuecomment-6844880792)).
> Reading 1, verbatim from the issue:
>
> > **Build the traced graph in one constructor call.** `explicit_trace(n)` knows its answer
> > combinatorially — *n* caps, the graph's boxes, *n* cups, and a wiring that is the current wiring
> > plus 2*n* spiders — so it can hand `__init__` the finished `boxes`, `wires` and `spider_types`
> > instead of composing 2*n* intermediate graphs. One scan instead of *O*(*k*). This is the local
> > fix and it stays inside `hypergraph.py`.
>
> Reading 2 (incremental `scan_types`) is out of scope: it changes a core invariant and waits on a
> word.

- [WIP] @session_01C7YfAPBzDhC42HUCSUu4td-2026-11-14 01:21 explicit_trace builds the cup-and-cap trace in one Hypergraph constructor call, equal term by term to the n one-wire nest
- [ ] a test that the one-call result equals the nest, on every pivotal category the branch is reachable from, left and right
- [ ] measure make_causal on frobenius before and after, at the sizes #536 reports
- [ ] CHANGELOG.md and every check of CONTRIBUTING.md's Run the unit tests
