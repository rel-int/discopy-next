Task: fix https://github.com/rel-int/discopy-next/issues/257, "Diagram parameters cannot be traced by JAX" (optyx is `discopy.optyx` on next; its bugs are fixed here). Branch `claude/257-jax-trace`, worktree /home/user/wt/257.
Issue (ported from rel-int/optyx#81, paths now under `discopy.optyx`):
```python
import sympy, jax.numpy as jnp
from discopy.optyx.photonic import Create, MZI
from discopy.optyx.channel import qmode
x = sympy.Symbol("x")
step = Create(1) @ qmode >> MZI(x, 0.2)
step.lambdify(x, modules="jax")(jnp.array(0.3))
# TypeError: iteration over a 0-d array (photonic.AbstractGate._contains_expr)
```
- `AbstractGate._contains_expr` iterates over its argument, which fails on a 0-d JAX array.
- `AbstractGate.array` returns `np.asarray(self._compute_array())`, which turns a JAX tracer into a NumPy array.
- `lambdify` on a diagram with a `Feedback` box fails (`'NoneType' object has no attribute 'replace'`).
What closes it: a `jax.grad` through the diagram's evaluation after `lambdify(x, modules="jax")` (the issue names `Interferometer.from_diagram(...)` from optyx.sampling — check what exists on main now, and use `to_path().array` or the equivalent if not). Reproduce first on today's main; parts may be fixed. Test with `pytest.importorskip("jax")`. If making the array backend-generic needs a choice (e.g. which array namespace a gate computes in), stop and report it.

- [x] `_contains_expr` does not iterate a 0-d array
- [x] gates compute their array in the backend of their parameters
- [x] `lambdify` on a `Feedback` diagram: already works on main, tested with JAX
- [x] a test of `jax.grad` through `to_path().array`
- [x] `Interferometer.from_diagram` traceable: a design question (which completion to differentiate), left open on #257
