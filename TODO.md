> Review round on #534 (cubic-dev-ai[bot]), head 7ec79ec4:
>
> 1. discopy/hypergraph.py:799 (thread 4558790982)
> [P2] This parameter reorder silently breaks existing positional calls: `explicit_trace(True)` no longer means a left trace, and `explicit_trace(False)` now returns the identity. Preserve the old boolean positional form or provide a compatibility shim while adding the wire-count argument.
>
> 2. discopy/hypergraph.py:848 (thread 4558790995)
> [P2] `explicit_trace(-1)` silently returns the untraced graph for classmethod trace factories because `range(-1)` is empty. Validate `n` with `assert_istraceable` before the loop so negative and oversized counts are rejected consistently.
>
> ```suggestion
>         assert_istraceable(self, n, left)
>         result = self
> ```

- [x] discopy/hypergraph.py:799 This parameter reorder silently breaks existing positional calls: `explicit_trace(True)` no longer means a le…
- [WIP] @session_01C7YfAQNzY28EVbk6YirCHJ-2026-11-13 02:16 discopy/hypergraph.py:848 `explicit_trace(-1)` silently returns the untraced graph for classmethod trace factories because `range(-1)` …
