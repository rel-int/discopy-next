> toumix, in https://ampcode.com/threads/T-48aec84b-09cc-47fa-bf08-e9067aece821:
>
> > just drop the left and right distinction since they're the same anyway and draw the four examples in a square rather than a line
> >
> > the example you labeled left-to-right is in fact top to bottom and vice-versa you messed up the labels
> >
> > the first example looks awesome but the others look weird: why are the diagrams inside squished? why is the chopped corner missing?

- [x] Cut the variant's corner off a frame drawn top to bottom around its args, with its label in the corner opposite, as for a box (draft in the commit after this one)
- [ ] Regenerate the baselines the frame corners and labels change, and test the cut corner of each variant of a frame and its reflection by Drawing.dagger
- [ ] Show toumix W, W[::-1], W.r and W[::-1].r in a 2x2 square for args on top of one another and side by side, each drawn at its own size
- [ ] Fall back for an arg without rotate in rigid.Box.to_drawing (Codex, discussion_r4712123856)
- [ ] Reserve the width of a frame's label when it moves to the corner (Codex, discussion_r4712123881)
