# TODO

🌙 Evening's prompt for the night of 2026-10-31, verbatim:

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

Box 7 of next#314: `ControlledPhaseShift.conjugate` returned
`ControlledPhaseShift(self.function, self.n_modes, self.n_control_modes,
self.is_dagger)` — the arguments of `self` — so the conjugate *was* `self` and
the truncation of `conjugate()` equalled the original's. Complex conjugation
sends `exp(2*pi*1j*y)` to `exp(-2*pi*1j*y)`, so the conjugate is the same box
with every phase negated. Cut on top of #418 because both are in
`discopy/optyx/core/control.py`.

- [x] `conjugate()` negates the phases, by conjugating its data the way
      `BitControlledBox.conjugate` next door conjugates its boxes, rather than
      carrying an `is_conj` flag: `function` stays the one source of truth for
      the phases the box applies, and `dagger()` and `conjugate()` commute for
      free. Checked against an independent oracle — for each control
      configuration, the product of `exp(-2*pi*1j*y) ** x` over
      `y in function(configuration)` — which reproduces the *original* box's
      truncation exactly, so it is an oracle and not a tautology.
