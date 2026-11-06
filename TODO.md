> toumix, 2026-11-06: "add the new strategies you propose to nightmare's branch then merge it, it's been open long enough"
>
> toumix, 2026-11-06, on the clash of `generators`: "structural_boxes sounds much better indeed, they don't really count as generators, same for the feedback ones they are just structural boxes" and "\"generator_strategy\" also sounds good let's gooo"

- [x] Merge `main` (next#458's `yanking_left` folded into `yanking` on either side)
- [x] `generators(*, dom, cod)` is `generator_strategy(*, dom, cod)`: a strategy of the generators, `generators` keeping its meaning in `group`
- [ ] `Diagram.structural_boxes`, the structural box classes among a level's factories, each drawing itself with its own `strategy` (discopy/discopy#817): Braid, Twist, Trace, Copy, Merge, Spider with phases, Feedback, checked by `assert_structural_boxes` through their own serialisation laws
- [ ] Merge next#459 (`feedback.structural_boxes`) once it lands
