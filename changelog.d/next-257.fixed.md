`optyx.photonic`'s gates `MZI`, `TBS`, `BBS` and `Phase` compute their
array in the backend of their parameters, so a diagram lambdified with
`modules="jax"` evaluates on a JAX tracer and `jax.grad` reaches its
parameters through `to_path().array` under `matrix.backend("jax")`;
a PyTorch parameter keeps its gradient in the gate's array, and a float
beside it in an `MZI` is cast to its dtype ([next#257](https://github.com/rel-int/discopy-next/issues/257)).
