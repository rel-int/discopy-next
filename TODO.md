can we reintroduce separate rules for left and right variants and completely remove ExpDir, TensorDir and AdjDir? i want {curry,trace,feedback,ev}_{left,right} as rules forwarding to a {curry,trace,feedback} method containing the implementation (this way child classes just have one entry point to override). similarly, axioms should state two different laws for left and right, except when the two sides collapse down to the same hom (e.g. the two snakes give identity so we can declare it as a single rule returning a single chained equation of length 3)

- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 18:20 `pattern`: `Over`, `Under`, `L` and `R` on their own, `ExpDir`, `TensorDir`, `AdjDir` and `sides` go
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 18:20 `abc`: `{trace,curry,feedback,ev}_{left,right}` rules forwarding to the `trace`, `curry`, `feedback` and `ev` methods, which are no longer rules
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 18:20 implementations: one plain entry point each, the marks moved onto the per-side rules
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 18:20 axioms: a law per side, one chained equation where both sides give the same hom
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 18:20 tests, docs, checks, changelog
