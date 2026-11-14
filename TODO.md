> cubic's review of `431a3f37`, two [P2] findings, verbatim:
>
> > **discopy/cmap.py:1648** — The signature silently breaks existing positional calls:
> > `explicit_trace(True)` used to mean `left=True`, but now means one right-side trace. Preserve
> > the boolean positional form or provide an explicit compatibility path before adopting `n` as the
> > first argument.
>
> > **discopy/cmap.py:1685** — `explicit_trace` silently accepts invalid wire counts: with a pivotal
> > trace factory, `explicit_trace(-1)` returns the untraced map because `range(-1)` is empty. Use
> > an exact zero check and validate `n` with `assert_istraceable` before dispatch, matching
> > `Hypergraph.explicit_trace`.

- [x] a negative count raises rather than returning the map, as on Hypergraph, with a doctest
- [ ] say in the changelog that the first argument is the count now, and answer the compatibility finding
