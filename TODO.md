> 🌙 Evening, 2026-11-20. Autonomous round: the next repo's open bugs that need no
> decision, oldest first (DESIRE_REPO `EVENING.md` step 5).
>
> rel-int/discopy-next#653, `toumix-agents`, 2026-11-19 15:53Z, verbatim:
>
> > **`test_train_ocr_lowers_its_loss` is flaky: OCR built before the seed**
> >
> > `test/drawing/vision.py::test_train_ocr_lowers_its_loss` (the OCR training test of
> > `discopy.drawing.vision`) failed on #646 (spider shapes in the drawing backends,
> > which touches no vision code), `test (3.14)` on `24ad225`:
> >
> > ```
> > assert losses[-1] < .8 * losses[0]
> > E   assert 48.8929 < (0.8 * 60.1725)
> > ```
> >
> > Passes 6 of 6 alone locally. Why: the test makes `vision.OCR(8)` before `train_ocr`
> > calls `torch.manual_seed(seed)`, so the network's init comes from global torch RNG,
> > whatever tests ran before it on that xdist worker.
> >
> > Fix (default): seed before building, e.g. `torch.manual_seed(0)` in the test, or
> > `train_ocr` builds the `OCR` itself when none given. Small, no design.

- [WIP] @session_01C7YfAP4hAgDkBVZMLuMoYv-2026-11-20 01:54 Reproduce: the loss the test starts from moves with the global torch RNG, so the ratio is a coin
- [ ] Seed before the OCR is built, train_ocr's own seed being too late for a network it is given
- [ ] Check it holds under whatever ran before it, and run the gates
