> Prompt: rel-int/discopy-next#584, quoted.
>
> > on `main` (`89467e62`), with torch installed:
> >
> > ```python
> > >>> from discopy.tensor import Tensor, Dim
> > >>> from discopy.matrix import backend
> > >>> with backend("pytorch"):
> > ...     t = Tensor([0.5, 1., 1., .25], Dim(2), Dim(2))
> > ...     t >> Tensor.id(Dim(2))
> > RuntimeError: both inputs should have same dtype
> > ```
> >
> > `Tensor.then` calls `np.tensordot(self.array, other.array, ...)`, and torch's `tensordot` does not promote a float32 and an int64 array. `promote` already knows the dtype of the composite; casting each factor to it first fixes it. #583 does this for `Matrix.then` (found while doing it, #551); `Tensor.then` overrides it, so it needs the same.
>
> Maths: the composite of f: A -> B and g: B -> C over dtypes s and t lives over the join s v t that `promote` computes; each factor is first sent along the inclusion s -> s v t (the identity when s = s v t), then contracted. Same for the tensor product.

- [x] Matrix.cast is the identity on a matrix that already has the dtype; Matrix.then uses it
- [x] Tensor.then and Tensor.tensor cast each factor to the promoted dtype before tensordot
- [ ] test in test/tensor.py on pytorch and jax
- [ ] CHANGELOG.md bullet
