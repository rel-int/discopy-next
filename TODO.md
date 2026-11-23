> in parallel let's open a thread on implementing isotopy as gif, first for planar isotopy (just sequences of interchangers) then braided and symmetric (i.e. with naturality) and compact (reidemeister)

- [ ] Design: rule on the questions of the pull request description (frames, API, search, how non-bijective moves tween)
- [ ] Planar: the shortest sequence of interchangers between two diagrams equal up to interchange
- [ ] Planar: a gif of that sequence, tweened by blending the positions of the drawing of each step
- [ ] Braided and symmetric: add naturality (sliding a box through a crossing) and the braid/swap laws to the search, and tween them
- [ ] Compact: add the Reidemeister moves (snakes and their braided versions) to the search, and tween them
- [ ] Docs, tests, changelog entry and the checks of CONTRIBUTING.md
