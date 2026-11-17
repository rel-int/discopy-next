> let's refactor so that box labels are computed by a smaller OCR model after the first GGN computed the boxes, maybe we can add another prediction for wire labels with its own OCR step too
>
> (USER, 2026-11-17, on brief #559; this branch is its slices 5 and 6: the networks in drawing.vision and drawing.from_picture)

- [x] vision.py and from_picture from the pipeline measured at 852c2c1, with their tests
- [x] docs/notebooks/from_picture.md with the numbers of slices 4 to 6 and run 7
- [ ] changelog entry, next-checks
