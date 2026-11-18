> Review round on #630 (cubic-dev-ai[bot]), head 27fd168c:
>
> 1. CHANGELOG.md:416 (thread 4610852913)
> [P3] The new entry makes the optyx dagger keep the box name, so the `optyx.core.zx.H` Fixed entry's claim that the optyx dagger 'returns a differently named copy' is now stale. Reword that Fixed entry to give the reason without that clause, or drop the clause.
>
> 2. discopy/optyx/core/diagram.py:792 (thread 4610852924)
> [P3] `Box.__repr__` emits bare `nan`/`inf` tokens for non-finite array entries, so `eval(repr(box))` raises `NameError` instead of reconstructing the box. Serialize non-finite values as valid expressions such as `float('nan')` and `float('inf')`.
>
> 3. discopy/optyx/core/diagram.py:781 (thread 4610877357)
> [P2] `array_adjoint` reverses the wire order for a two-wire 0→2 or 2→0 array because every rank-2 array is forced through `.T`. Use the per-wire axis permutation whenever `ndim == dom + cod`, including rank-2 tensors.
>
> ```suggestion
>         if self.array.ndim != dom + cod:
> ```
>
> 4. discopy/optyx/core/diagram.py:783 (thread 4610877374)
> [P2] `array_adjoint` dispatches through the active matrix backend instead of the backend of `self.array`, so differentiable Torch-backed boxes can fail or lose autograd when daggered. Dispatch movement and conjugation through the stored array's backend, or require a matching backend explicitly.
>
> 5. test/optyx/test_diagram.py:146 (thread 4610877384)
> [P3] This test never checks the daggered array's axis order for n-d arrays, which is the bug this PR fixes. Add an assertion that `dagger.array` equals the conjugate of the moved axes, so a return to `.T` fails here too.
>
> ```suggestion
>     assert dagger[::-1] == box and (dagger[::-1].array == box.array).all()
>     cod = len(box.cod)
>     axes = list(range(len(box.dom @ box.cod)))
>     assert (dagger.array == np.conj(np.moveaxis(
>         box.array, axes, [*range(cod, len(axes)), *range(cod)]))).all()
> ```
>
> 6. discopy/optyx/core/zx.py:291 (thread 4610890528)
> [P1] The outer `.dagger()` on the `tensor.Box` may only swap `dom`/`cod` and flip `is_dagger` without reordering the array axes. If so, the truncation of a dagger ZX box would have an array whose axes do not match its dom/cod. Check that the returned tensor box's array is in (dom, cod) axis order, and add a test comparing `f[::-1].truncation()` with `f.truncation()[::-1]` on a non-symmetric array.
>
> 7. discopy/optyx/core/diagram.py:819 (thread 4610890536)
> [P2] Daggered array boxes lose inferred dimensions when `input_dims` is omitted: this branch converts `None` to a mode cutoff of 2, so daggering a state/effect with another inferred cutoff raises a size-mismatch error. Preserve `None` here so the undaggered box can infer its dimensions from the array.
>
> ```suggestion
>                 None if input_dims is None else self.array_wire_dims(self.dom, input_dims)).dagger()
> ```
>
> 8. discopy/optyx/core/zx.py:291 (thread 4610890552)
> [P2] This dagger branch duplicates the rule already in `Box.array_truncation`. Drop the override's own logic and delegate to `self.array_truncation(input_dims, output_dims)` so the dagger handling and the array size check live in one place.
>
> 9. CHANGELOG.md:423 (thread 4610890561)
> [P3] The sentence "its truncation is the dagger of the box's" is ambiguous. Say that truncating a dagger gives the dagger of the truncated box, and name the old behaviour explicitly in the last clause.
>
> ```suggestion
> - The dagger of an `optyx.core` box defined by its array keeps its name and
>   flips `is_dagger`, as `cat.Box.dagger` does, so `f[::-1][::-1] == f`: it
>   prints `f[::-1]`, not `f.dagger()`. Truncating a dagger gives the dagger of
>   the truncated box, and its repr prints the array, so it reads back. An array
>   with an axis per wire is daggered wire by wire, where it used to reverse
>   their order ([next#630]).
> ```

- [WIP] @session_01MEJxy45CJtENv2bizg14LE-2026-11-18 11:32 CHANGELOG.md:416 The new entry makes the optyx dagger keep the box name, so the `optyx.core.zx.H` Fixed entry's claim that the…
- [WIP] @session_01MEJxy45CJtENv2bizg14LE-2026-11-18 11:32 discopy/optyx/core/diagram.py:792 `Box.__repr__` emits bare `nan`/`inf` tokens for non-finite array entries, so `eval(repr(box))` raises `NameE…
- [WIP] @session_01MEJxy45CJtENv2bizg14LE-2026-11-18 11:32 discopy/optyx/core/diagram.py:781 `array_adjoint` reverses the wire order for a two-wire 0→2 or 2→0 array because every rank-2 array is forced …
- [WIP] @session_01MEJxy45CJtENv2bizg14LE-2026-11-18 11:33 discopy/optyx/core/diagram.py:783 `array_adjoint` dispatches through the active matrix backend instead of the backend of `self.array`, so diffe…
- [WIP] @session_01MEJxy45CJtENv2bizg14LE-2026-11-18 11:33 test/optyx/test_diagram.py:146 This test never checks the daggered array's axis order for n-d arrays, which is the bug this PR fixes.
- [WIP] @session_01MEJxy45CJtENv2bizg14LE-2026-11-18 11:33 discopy/optyx/core/zx.py:291 The outer `.dagger()` on the `tensor.Box` may only swap `dom`/`cod` and flip `is_dagger` without reordering t…
- [ ] discopy/optyx/core/diagram.py:819 Daggered array boxes lose inferred dimensions when `input_dims` is omitted: this branch converts `None` to a …
- [ ] discopy/optyx/core/zx.py:291 This dagger branch duplicates the rule already in `Box.array_truncation`.
- [ ] CHANGELOG.md:423 The sentence "its truncation is the dagger of the box's" is ambiguous.
