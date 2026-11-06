> toumix, 2026-11-06: "upstream 817 and 818 have become kind of stale, let's start by incorporating all the comments I had on them into next"
>
> toumix, 2026-11-06, choosing among the routes proposed: "Small PR from main", i.e. the laws of discopy/discopy#818 on neither next `main` nor #1, plus its measured fixes next lacks; #1 stays the vehicle for the rest.

The comments (quoted from the `TODO.md` history of discopy/discopy#817 and #818): `is_monochrome` goes (next#390), the argument shapes live in `testable` and the laws in `abc` (next `main` since #214), `axioms` is `testable` (next#403), and "There should also be two other reidemeister 1 for left and right trace": that one is not on next.

- [ ] `yanking_left`: tracing a braid of an object with itself on the left gives its twist, beside `yanking` at every level that states or declares it, so that Reidemeister 1 for the right trace is `yanking` on a compact category, for the left `yanking_left`, and discopy/discopy#818's `twist_as_trace` is the two together
- [ ] Each free category's declaration of `yanking_left` measured on the matrix
- [ ] #818's fixes next lacks, checked one by one on next `main`: `feedback.Trace.delay`, `CMap.is_boundary_connected`, the trees of the structural boxes (#817)
