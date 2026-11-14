> cubic's review of `f1748033`, two [P2] findings, verbatim:
>
> > **test/hypergraph.py:1255** — This counter never sees the constructor call. `module.Hypergraph`
> > is a `Hypergraph[Diagram]` alias, so patching its `__init__` does not intercept
> > `Hypergraph(...)` calls on the origin class. Patch `module.Hypergraph.__origin__` instead, and
> > take `init` from the same origin, so the counter wraps the real constructor.
>
> > **discopy/hypergraph.py:856** — `explicit_trace` now bypasses `self.category.trace_factory` for
> > every pivotal category, so a pivotal category that overrides the classmethod gets the
> > hard-coded cap/cup onion instead of its requested trace. Keep the optimized path limited to the
> > standard pivotal trace implementation and retain the factory loop for overrides.

- [x] the onion is built only for pivotal's own trace_factory; an override keeps the loop, with a test
- [ ] answer the counter finding with the measurement: 40 calls on the merge base, 1 on the head
