> Prompt: rel-int/discopy-next#551, quoted
>
> > on `main` (`b6765a3e2`), with torch installed:
> >
> > ```python
> > >>> from discopy.matrix import Matrix, backend
> > >>> with backend("pytorch"):
> > ...     Matrix([[0.5, 1.], [1., 0.25]], 2, 2).trace()
> > TypeError: Cannot interpret 'torch.float32' as a data type
> > ```
> >
> > `Matrix.id` builds its array with numpy's `identity(..., dtype=cls.dtype)`, and under the pytorch backend an inferred `cls.dtype` is `torch.float32`, which numpy cannot read. so `trace` and anything else that calls `id`, `ones` or `discard` on such a matrix fails, whatever the dtype. found while doing #550.
> >
> > recommended: build the identity with the active backend, or cast the numpy array after.
>
> The maths: `id(n)` is the matrix of the Kronecker delta, the same in every ring, so it is built
> once over the integers and cast into the ring of `cls.dtype` by the active backend, as `swap`,
> `copy` and `basis` already are. A composite `f >> g` lives over the ring both promote to, so
> both arrays are cast there before the product, as `tensor` already does.

- [WIP] @session_01MEJxy4kqJ6YhajqkcZRUXm-2026-11-17 00:18 Matrix.id builds the integer identity and casts it with the active backend
- [ ] Matrix.then casts both arrays to the promoted dtype before matmul (torch does not promote)
- [ ] test in test/matrix.py: trace and mixed composition on pytorch and jax
- [ ] CHANGELOG.md bullet
