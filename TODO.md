> let's refactor so that box labels are computed by a smaller OCR model after the first GGN computed the boxes, maybe we can add another prediction for wire labels with its own OCR step too / drop the idea of predicting modules, just keep prediction of structural primitives and let a symbolic scan decide ie if there are no swaps then output a monoidal diagram etc
>
> (USER, 2026-11-17, on brief #559; this branch is its slice 4, the symbolic half: components, scan and decode)

- [x] decode, regions, attach, refine, scan, read from the pipeline measured at 852c2c1, with their tests
- [ ] gate: true masks read back at 990 to 1,000 of 1,000 per module from seed 7
- [ ] changelog entry, next-checks
