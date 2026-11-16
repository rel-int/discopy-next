> the triangles are boxes, not spiders: we need a triangle option for discopy boxes, write the brief
>
> yes approve on all these, spawn Opus subagents as needed
>
> (USER, 2026-11-16; brief #559, slice 1)

- [x] `shape="triangle"` on any box, drawn by `Backend.draw_box` for every backend: apex on the side with fewer wires, at that wire when there is one
- [x] dagger flips it: `Box.dagger` keeps the drawing attributes
- [x] test drawing `Box("t", x ** 2, x, shape="triangle")` against a baseline, fails on main
- [ ] doctest drawing `docs/_static/drawing/triangle.svg`: a box `m -> 1` and its dagger
- [ ] changelog entry, next-checks
