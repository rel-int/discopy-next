> No human prompt: the first half of #541 says in its own body *"No decision in it"*, so
> `AGENTS.md`'s *Next repos* has it taken without asking. Verbatim from the issue:
>
> > ### 1. The pivotal `CMap` onion — chooses nothing, takeable
> >
> > `compact` `make_causal` on a map with *n* cut wires, which is the class-method branch of
> > `CMap.explicit_trace`:
> >
> > | *n* | 16 | 32 | 64 | 128 |
> > |---|---|---|---|---|
> > | `make_causal` | 0.060 | 0.213 | 0.802 | **3.298** |
> >
> > ×4 per doubling, unchanged by #540. `cProfile` at *n*=64: 64 calls of `pivotal.trace_factory`,
> > each two `CMap.then` (3.6 s of 5.0) and a `tensor` (1.2 s), with
> > `python.finset.Function.__init__` (3,970 calls, 2.0 s) and the `ports` cached property (257
> > recomputations, 1.75 s) underneath. Every composition rebuilds a permutation over a port set
> > that grows with *n*.
> >
> > This is exactly #536's reading 1 in the other representation: the onion is known
> > combinatorially — *n* caps, the map's boxes, *n* cups, and one permutation — so
> > `explicit_trace(n)` can build the edge permutation itself instead of composing 2*n* maps.

- [x] Measure the pivotal branch before the change: compact make_causal over n=16..128, with a profile of where the time goes
- [x] Build the onion in one constructor call in CMap.explicit_trace, the way Hypergraph.explicit_trace does, keeping the nest for a category that overrides trace_factory
- [x] Pin the map, not just the morphism: explicit_trace(n) against the n-fold nest on the pivotal kind, n=1..3, both sides, and the count of CMap.then calls
- [ ] Measure after, same table, and say what the ratio is
- [ ] CHANGELOG.md bullet and the gates of CONTRIBUTING.md
