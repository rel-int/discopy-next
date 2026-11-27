`lambdify` on a parametrised quantum gate or scalar evaluates its phase in
the current backend, whatever `modules` it is given, and passes its other
keyword arguments to `sympy.lambdify`, so torch gradients flow through
nonlinear phases and `jax.grad` works through them ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
