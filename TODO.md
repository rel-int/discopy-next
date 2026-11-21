> cubic-dev-ai review on #688 (https://github.com/rel-int/discopy-next/pull/688#pullrequestreview-6005346975):
>
> > 1. discopy/drawing/drawing.py:433 [P2] `dagger()` does not map `wire` nodes, so `mapping[n]` raises KeyError on any drawing with waypoints.
> > 2. discopy/drawing/layout.py:392 [P2] Reject negative weights before building the QP: a negative `square` makes its objective non-convex, while negative `bend` or `width` makes the minimization unbounded.
> > 3. discopy/drawing/backend.py:992 [P3] The waypoint branch ignores the frame-boundary zero-width rule that the branch below applies.

- [x] drawing.py dagger maps wire nodes
- [x] layout.py solve rejects negative weights
- [x] backend.py waypoint branch and frame boundaries
