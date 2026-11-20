> Review round on #654 (cubic-dev-ai[bot]), head e37c3eb6:
>
> 1. test/optyx/test_zw.py:660 (thread 4637567414)
> [P2] This assertion does not prove that the autograd graph still reaches `t`; it only checks the output flag. Backpropagate through `endo.to_path().array` and assert `t.grad is not None` so a detached-but-grad-enabled tensor cannot pass the regression test.
>
> ```suggestion
>         array = endo.to_path().array
>         assert array.requires_grad
>         array.abs().sum().backward()
>         assert t.grad is not None
> ```
>
> 2. test/matrix.py:369 (thread 4637567435)
> [P2] These assertions do not prove that the autograd graph reaches `t`; a newly detached tensor can also have `requires_grad=True`. Backpropagate a scalar derived from each constructed matrix and assert that `t.grad` is populated with the expected value.
>
> 3. discopy/matrix.py:824 (thread 4637567440)
> [P2] `asarray` checks only the first nested leaf, so inputs such as `Matrix([0, t], 1, 2)` bypass stacking when a later entry is a gradient-carrying backend tensor. Detect backend-owned leaves recursively across all entries before falling back to `np.asarray`.
>
> 4. discopy/optyx/core/zw.py:670 (thread 4637567447)
> [P2] Preserving the backend tensor in `Endo.scalar` makes `Endo`'s equality/hash contract invalid: `setoid()` compares tensor values, but `Arrow.__hash__` hashes the tensor object. Equal backend-backed endomorphisms can therefore have different hashes or fail when hashed; make the setoid use a backend-aware stable hashable representation.

- [ ] test/optyx/test_zw.py:660 This assertion does not prove that the autograd graph still reaches `t`; it only checks the output flag.
- [ ] test/matrix.py:369 These assertions do not prove that the autograd graph reaches `t`; a newly detached tensor can also have `req…
- [x] discopy/matrix.py:824 `asarray` checks only the first nested leaf, so inputs such as `Matrix([0, t], 1, 2)` bypass stacking when a …
- [x] discopy/optyx/core/zw.py:670 Preserving the backend tensor in `Endo.scalar` makes `Endo`'s equality/hash contract invalid: `setoid()` comp…
