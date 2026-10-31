# TODO

🌙 Evening's prompt for the night of 2026-10-31, verbatim:

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

Three boxes of rel-int/discopy-next#314 in `discopy/optyx/core/`, all reproduced on `main`
`adcd06b6`: a `ControlledPhaseShift` that raises on reshape with two or more control modes, a
`ZBox.__eq__` that raises on amplitude lists of different lengths, and a `W` and `ZBox` that
override `__eq__` without a `__hash__`, so neither is hashable. None picks between readings:
each module states its own oracle.

- [x] `control.ControlledPhaseShift.truncation` wrote each control configuration's cell as
      `array[i, :]`, which indexes the first axis with the whole configuration `i`, so with two
      control modes the cell was `(2, 2, 2)` where the phase is `(2, 2)` and the reshape raised.
      It is `array[tuple(i)]` now, the box's own suggested fix, which is the same cell as before
      for one control mode.
- [x] `zw.ZBox.__eq__` compared amplitudes with `np.allclose`, which raises on lists of different
      lengths. It is `misc.compare_arrays_of_different_sizes` now, which is what this module says
      equality of two truncations of the same infinite amplitude list is — its docstring states
      the ZW axioms with it and `test/optyx/test_zw.py` checks every one of them that way.
- [x] `zw.W` and `zw.ZBox` override `__eq__`, so Python set their `__hash__` to `None`:
      `hash(W(2))` raised and a `Functor` could not look either box up in a dict `ar_map`. Both
      have a `__hash__` now, built like `core.diagram.Ty`'s from a tuple of the class and the
      fields `__eq__` reads. A `ZBox` hashes on its legs alone: its amplitudes are equal up to
      truncation and up to a tolerance, so no hash of them can keep equal objects hashing equal.
      What equality of two callable amplitudes means — box 22 of the same issue — is untouched.
