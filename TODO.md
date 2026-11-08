# TODO

> ### What this asks for
>
> Two separate pieces of work, neither needing a decision:
>
> - `make_monogamous` in one pass, carrying `leg_order` through the scan.
> - `trace_backward_wires` as a loop rather than a recursion (it raises `RecursionError` on a graph with 1,000 backward wires, as the three did), and `explicit_trace` measured on its own, since that is where `make_causal`'s time is.

— [#471](https://github.com/rel-int/discopy-next/issues/471)

- [ ] Build the fingerprint corpus on the unmodified tree: `dom`, `cod`, `boxes`, `wires`,
      `spider_types` and `offsets` of `make_monogamous`, `make_causal`, `make_cocausal` and
      `trace_backward_wires` over hypergraphs from the strategies of `frobenius`, `markov`,
      `comarkov`, `compact` and `symmetric`, errors included.
- [ ] Measure `main` at n = 32, 64, 128 for both methods, and `explicit_trace` on its own.
- [ ] `make_monogamous` in one pass, through `insert_boxes` and `shift_depths`.
- [ ] Measure whether `leg_order` is invariant through the pass, rather than assume it.
- [ ] `trace_backward_wires` as a loop, the cut of one wire its own method.
- [ ] Assert the rewrite agrees with the recursion on every fingerprint.
- [ ] Re-measure n = 32, 64, 128 and check no `RecursionError` at n = 1000.
- [ ] Tests in `test/hypergraph.py`, one failing on the parent for each piece.
- [ ] `CHANGELOG.md` bullet under `[Unreleased]` with a `[next#471]` reference link.
- [ ] Local gates through `uv run`, each as a delta against `origin/main`.
