see how feedback takes dom cod and mem instead of a count n? it describes an already partitioned context so that we don't need to split at a given index, which is more theoretically aligned. make trace like feedback, and also curry, which could take a context, a base and an exponent. this should remove all uses of Count besides the typing for spiders/copy.

- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 16:31 `trace(dom, cod, mem, left)` on `TracedCategory` and every implementation, with one helper partitioning the boundary
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 16:31 `curry(context, base, exponent, left)` and `uncurry(base, exponent, left)` on `BiclosedCategory`, rigid and every implementation
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 16:31 the trace and currying axioms and the call sites, tests and docs
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 16:31 `Count` left only for spiders, copies and their kin
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 16:31 checks, changelog
