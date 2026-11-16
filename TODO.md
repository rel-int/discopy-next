> once the whole thing is ready plug it together into a draft paper

USER, 2026-11-16, on the tool paper of #564. The paper's LaTeX figures are drawn by the TikZ backend, and #574 is two bugs of it found drawing them: a bug of a next repo is fixed here without asking.

- [x] TikZ.draw_polygon honours edgecolor, so the background of a drawing has no black frame (#574, 1)
- [x] a TikZ spider keeps its label at full size, nodesize sets the size of the node (#574, 2)
- [x] tests, the tikz baselines regenerated, changelog entry, next-checks
