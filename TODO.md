> cubic-dev-ai review on #688 (https://github.com/rel-int/discopy-next/pull/688#pullrequestreview-6091818158):
>
> > discopy/drawing/optimiser.py:1009 [P2] `is_valid` checks only eight subdivisions of each Bezier, so a curve can cross another wire or pass through a box between samples and still be accepted. Use continuous or adaptive curve-intersection and distance checks before accepting a layout.
> > discopy/drawing/optimiser.py:407 [P2] `to_pivotal_drawing` collapses every nonempty identity wire to a point because this branch creates a zero-height identity, and the waypoint backend skips its zero-length edges. Preserve a positive height for boxless drawings with wires.

- [WIP] @T-01a1fdff-2026-11-27 18:00 optimiser.py:1009 is_valid samples each curve finely enough to see a crossing
- [WIP] @T-01a1fdff-2026-11-27 18:00 optimiser.py:407 a diagram of wires and no box has height one
