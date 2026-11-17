> cubic P3: the long-label assertion is vacuous, a single box has no pair. cubic P2: `label_overlap` pairs boxes by bottom edge, so a tall box and a short one in the same row are never compared. cubic P3: the changelog says `hand_score` is the weighted sum, it is minus it. cubic P3: no test of the tie-break. Codex P2: a LaTeX label is measured by its source length, not its rendered width (`utils.text_width`).

- [x] rectangles measure the label with utils.text_width; label_overlap pairs boxes whose rectangles share a height range; tests for both and for the tie-break; changelog sign
