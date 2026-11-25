> in parallel let's open a thread on implementing isotopy as gif, first for planar isotopy (just sequences of interchangers) then braided and symmetric (i.e. with naturality) and compact (reidemeister)

- [x] Design: rule on the questions of the pull request description (frames, API, search, how non-bijective moves tween)
- [x] Planar: the shortest sequence of interchangers between two diagrams equal up to interchange
- [x] Planar: a gif of that sequence, tweened by blending the positions of the drawing of each step
- [x] Braided and symmetric: add naturality (sliding a box through a crossing) and the braid/swap laws to the search, and tween them
- [ ] Compact: add the Reidemeister moves (snakes and their braided versions) to the search, and tween them
- [ ] Docs, tests, changelog entry and the checks of CONTRIBUTING.md
- [WIP] @T-e8a5da2e-2026-11-25 11:01 Rewrite: `discopy.rewrite` with `Rewrite`, a sesquicategory of diagrams, its `Layer` of steps rewritten at once, `Step` and its `Interchange` and `Slide`, and an ADR
