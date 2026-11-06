> I got some spare tokens to spend: let’s use them to polish all things discopy-next. Make a deep review of the code and the infra around it, summarising your findings and suggestions in an issue, fixing any bugs that you can fix without requiring a design decisions from my side.

cubic's review of #453 at 5d045108:

- [x] **C1** `discopy/quantum/reservoir.py:346`: `Readout.__eq__` assumes every field has `.shape`, so list-valued `weights` or `intercept` raise AttributeError instead of comparing. Compare with `np.shape` and `np.all` so plain sequences work too.
- [x] **C2** `discopy/quantum/zx.py:555`: Under the PyTorch backend, this passes a plain Python complex to `torch.exp`, which requires a Tensor, so `circuit2zx(Rz(.3))` should raise TypeError. Wrap the phase in `np.array` under `backend()` the way `Rz.array` does.
- [x] **C3** `discopy/grammar/cfg.py:97`: `assert_iscomposable` dereferences `other.cod` before validating that each argument is a `Tree`, so `f(42)` raises an incidental AttributeError instead of the API's normal TypeError. Validate the arguments before reading their codomains.
- [x] **C4** `discopy/quantum/circuit.py:232`: `init_and_discard` treats every `Digit` as a binary bit, so circuits over `Digit(4)` cannot be initialized. Use `Digits(0, dim=x.dim)` for non-binary digits and keep `Bits(0)` only for dimension-2 digits.
- [ ] **C5** `discopy/quantum/contextuality.py:73`: The docstring says the dual is always non-unique on the non-contextual polytope, but a valid one-context model with strictly positive probabilities has a unique dual optimizer (`M = I`, `y_i = 1`). Say it may be non-unique, or scope the claim to the multi-context case.
- [ ] **C6** `discopy/quantum/gates.py:935`: `Parametrized.lambdify` now raises TypeError whenever callers pass the supported `modules=` option because it supplies `modules` twice. Pop `modules` from `kwargs` first, using the backend module as the default.
- [ ] **C7** `CHANGELOG.md:717`: Add the missing verb before `Controlled`: the bullet reads "and `Controlled` that a zero distance raises" and is grammatically incomplete.
