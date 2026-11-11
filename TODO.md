> Review round on #495 (cubic-dev-ai[bot]), head 942b5d47:
>
> 1. test/quantum/circuit.py:606 (thread 4532374042)
> [P2] This test fails whenever the optional `quimb` package is not installed. Add `pytest.importorskip("quimb")` before constructing `Quimb()`, matching the adjacent Quimb test.
>
> ```suggestion
>     pytest.importorskip("quimb")
>     assert c.eval(contractor=Quimb()).is_close(c.eval())
> ```
>
> 2. README.md:24 (thread 4532374064)
> [P2] This new backend list leaves the later tensor-valued-functor section stale: it still describes `TensorNetwork` as the evaluation backend and omits `opt_einsum` and `Quimb`. Update that paragraph to describe the current `contractor` API so users do not receive conflicting guidance.
>
> 3. test/tensor.py:710 (thread 4532374070)
> [P2] This optional autodiff test fails the tensor test suite when PyTorch is not installed. Use `pytest.importorskip('torch')` so environments testing Quimb without PyTorch skip this case.
>
> ```suggestion
>     torch = pytest.importorskip('torch')
> ```
>
> 4. discopy/quantum/circuit.py:326 (thread 4532374076)
> [P2] Plain matrix backends such as `NumPy()` still enter the legacy callable path here and raise `TypeError` instead of being used as the `einsum` engine. Route `Backend` instances alongside `Contractor` instances.
>
> 5. pyproject.toml:80 (thread 4532374081)
> [P2] This extra does not enforce the `autoray` version required for the new `Quimb` backend to preserve gradients. Add `autoray >= 0.9` so an existing 0.8 installation cannot satisfy Quimb while silently falling back to NumPy for cached contraction trees.
>
> ```suggestion
>     "quimb",
>     "autoray >= 0.9",
> ```
>
> 6. discopy/matrix.py:842 (thread 4532374087)
> [P2] `Quimb.network` uses the backend captured when the contractor was constructed for trace preprocessing, so `Quimb()` created before a PyTorch/JAX context can run NumPy `einsum` on those arrays and break gradients. Resolve the active backend at contraction time when `base` is not explicit.
>
> ```suggestion
>                 array = (self.base or get_backend()).module.einsum(
> ```
>
> 7. discopy/tensor.py:711 (thread 4532374094)
> [P2] `to_quimb` rejects every tensor sum because `CMap.from_diagram` cannot encode `Sum` boxes. Handle sums before this conversion, or explicitly preserve their terms when building the Quimb network, so valid tensor diagrams do not fail unexpectedly.
>
> 8. discopy/matrix.py:776 (thread 4532374103)
> [P3] `Contractor.__repr__` is not round-trippable when `base` is supplied: `Quimb(NumPy())` includes the default address-based `Backend` repr and cannot be evaluated back. Give backends deterministic representations and value equality, or serialize the base by backend type.
>
> 9. pyproject.toml:63 (thread 4532374108)
> [P3] The new `tensor` dependency group is not documented alongside the other groups, so users cannot discover `uv sync --group tensor` or `pip install -e '.[tensor]'`. Add `tensor` to the group list and optional-dependency list in `CONTRIBUTING.md`.
>
> 10. discopy/tensor.py:566 (thread 4532374120)
> [P3] A contractor-only functor now raises a serialization error that says only einsum `params` prevent a tree, which misidentifies the cause. Update the no-tree documentation/message to mention non-serializable contractors as well.

- [ ] test/quantum/circuit.py:606 This test fails whenever the optional `quimb` package is not installed.
- [WIP] @session_01MEJxy29LoL954iS6TzLhbH-2026-11-11 00:38 README.md:24 This new backend list leaves the later tensor-valued-functor section stale: it still describes `TensorNetwork…
- [WIP] @session_01MEJxy29LoL954iS6TzLhbH-2026-11-11 00:39 test/tensor.py:710 This optional autodiff test fails the tensor test suite when PyTorch is not installed.
- [WIP] @session_01MEJxy29LoL954iS6TzLhbH-2026-11-11 00:39 discopy/quantum/circuit.py:326 Plain matrix backends such as `NumPy()` still enter the legacy callable path here and raise `TypeError` inste…
- [ ] pyproject.toml:80 This extra does not enforce the `autoray` version required for the new `Quimb` backend to preserve gradients.
- [ ] discopy/matrix.py:842 `Quimb.network` uses the backend captured when the contractor was constructed for trace preprocessing, so `Qu…
- [ ] discopy/tensor.py:711 `to_quimb` rejects every tensor sum because `CMap.from_diagram` cannot encode `Sum` boxes.
- [ ] discopy/matrix.py:776 `Contractor.__repr__` is not round-trippable when `base` is supplied: `Quimb(NumPy())` includes the default a…
- [ ] pyproject.toml:63 The new `tensor` dependency group is not documented alongside the other groups, so users cannot discover `uv …
- [ ] discopy/tensor.py:566 A contractor-only functor now raises a serialization error that says only einsum `params` prevent a tree, whi…
