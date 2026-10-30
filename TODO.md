# TODO

cubic on #409, verbatim:

> P2: This gradient test only asserts that a conjugate's gradient differs from the original's, so a
> wrong gradient would still pass. It also covers only default flags, although the PR description
> says gradients are checked across flag combinations. Compare against an expected gradient, or
> parametrize over the flag combinations in `GATES`.

Correct on both counts.

- [x] drop the inequality assertion and pin the gradient against the difference quotient of the
      gate itself, in `test/optyx/grad.py` where gradients are tested, not in `test_conjugation.py`
- [x] carry the four conjugated gates through the existing `param_circuits`, so `test_zero_grad`
      and `test_daggers_cancel` cover them too
