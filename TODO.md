> cubic-dev-ai review on #688 (https://github.com/rel-int/discopy-next/pull/688#pullrequestreview-6091470640):
>
> > 6. discopy/drawing/optimiser.py:1171 [P2] `to_pivotal_drawing` crashes on the scalar identity: it has no curves or boxes, so this concatenation receives an empty list. Handle the empty layout before computing its shift.
> > 7. discopy/drawing/optimiser.py:1029 [P2] `is_valid` checks crossings only between distinct wires, so a self-crossing wire can pass the line search. Also reject intersections between non-adjacent segments of each wire.

- [WIP] @T-01a1fdff-2026-11-27 17:21 optimiser.py:1171 to_pivotal_drawing on the scalar identity
- [WIP] @T-01a1fdff-2026-11-27 17:21 optimiser.py:1029 is_valid rejects a wire crossing itself
