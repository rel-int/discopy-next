`Diagram.structural_boxes`, the box classes among the factories of a free
category that draw themselves, a `strategy` for each one (braids,
twists, feedback and traces, swaps and permutations, copies, discards,
merges, noise, spiders, cups, caps, evaluations and coevaluations) on
atoms drawn by `Box.atomic_strategy`, and
`testable.assert_structural_boxes`, which checks each one by the laws of
`Serialisable`; it found that the string of a closed `Eval` or `Coeval`
on the right read back on the left, now fixed ([next#1](https://github.com/rel-int/discopy-next/pull/1)).
