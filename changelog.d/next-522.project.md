The Lean model in `lean/` covers the pure `EGraph`: `merge` is reflexive,
symmetric and transitive, `add` returns the same outputs when a
deterministic box is added again, and `rebuild` ends on the least
congruent equivalence containing the classes, hence idempotent and
confluent; the differential test compares the fold of a history with
`History.fold()` ([next#522](https://github.com/rel-int/discopy-next/pull/522)).
