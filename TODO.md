> No human prompt: [#537](https://github.com/rel-int/discopy-next/issues/537) is a bug in this repo
> whose own body says *"No decision in it"*, so `AGENTS.md`'s *Next repos* has it fixed without
> asking. Verbatim from the issue:
>
> > - `CMap.explicit_trace(left)` (`discopy/cmap.py:1651`) branches on exactly the same two kinds of
> >   `trace_factory` as `Hypergraph.explicit_trace` did, with the same docstring note, and traces
> >   one wire.
> > - `CMap.make_causal` (`discopy/cmap.py:1748`) computes every cut combinatorially and then ends
> >   `for _ in range(n_traces): result = result.explicit_trace()`.
> > - `CMap.trace` (`discopy/cmap.py:1490`) loops `for _ in range(n): result =
> >   result.explicit_trace(left)` on a traced category that is not symmetric.
> >
> > So the change is the same two lines plus the two call sites. What is **not** carried over is the
> > measurement: #478's table is `Hypergraph`'s, and #534's is too. Before changing anything here,
> > the thing worth knowing is which of the two kinds of `trace_factory` a `CMap` meets in practice
> > and what the loop costs on each.

- [ ] measure both kinds of trace_factory on a CMap, and say which call sites reach which
- [ ] explicit_trace takes n and converts once on the class-factory branch; trace and make_causal stop looping
- [ ] a test that the n-wire trace is the n nested ones on a CMap, both kinds, and that the conversion happens once
- [ ] CHANGELOG.md and every check of CONTRIBUTING.md's Run the unit tests
