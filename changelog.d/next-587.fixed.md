`Tensor.then` and `Tensor.tensor` cast each factor to the dtype they
promote to, as `Matrix.then` does, so a float tensor composes and tensors
with the integer identity on the pytorch backend; `Matrix.cast` returns
the matrix itself when it already has the dtype ([next#587](https://github.com/rel-int/discopy-next/pull/587)).
