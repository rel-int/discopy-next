# TODO

Round of review feedback from cubic on `5f73bb5d`, quoted verbatim:

> **P2, `discopy/optyx/core/zw.py:392`** — This prefix comparison makes `ZBox` equality
> non-transitive, which is unsafe now that `ZBox` is usable as a dictionary key. A dict or set
> containing different truncations can return different results depending on which
> equivalent-looking box was inserted first; use an equivalence-compatible representation for
> `__eq__` or keep truncation comparison out of key equality.

> **P2, `discopy/optyx/core/zw.py:392`** — This comparison treats NaN amplitudes as equal to any
> matching-prefix values. Reject NaNs before accepting equality, or make the helper use an
> explicit `equal_nan=False` policy.

> **P3, `discopy/optyx/core/zw.py:399`** — The hash docstring gives a false equality example:
> `ZBox(1, 1, [1, 2])` also equals `ZBox(1, 1, [1, 2, 4])` under the comparator above. Correct the
> example so the documented hash rationale matches the implementation.

- [x] the first P2: drop `ZBox.__hash__`, keep `W.__hash__`. Box 24 states two defects, both of
      `W` — `hash(zw.W(2))` and a `Functor` with `W` in a dict `ar_map` — so `ZBox` need not be
      hashable to close it, and a non-transitive `__eq__` admits no consistent hash at all. What
      `ZBox` equality means is box 22's open question and stays untouched; #314's own preamble
      calls the prefix comparison "Not a bug"
- [x] the second P2: reject NaN amplitudes in `ZBox.__eq__`. Real, and a regression of this head:
      `compare_arrays_of_different_sizes` reads `abs(x - y) > tol`, which is False for a NaN
      difference, so a NaN compared equal to *anything*, where the old `np.allclose` gave False
- [x] the P3: the false example goes with the docstring that carried it
- [x] tests: `ZBox` is unhashable with its reason, NaN amplitudes are unequal, `W`'s hash and the
      `Functor` test stay
- [x] `CHANGELOG.md`: the third bullet says what landed
- [x] checks: `pflake8`, `pylint`, `ruff`, `ty`, `pytest test/optyx`, the #314 probe
