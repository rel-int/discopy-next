# TODO

cubic's P3 on #423, `discopy/optyx/core/control.py:327`, review 5720816803 on
`721bae59`, verbatim:

> P3: `conjugate()` now stores a local lambda, making the conjugated box
> unpicklable even when the original `function` is picklable. Use a
> module-level phase-negation callable (for example with
> `functools.partial`) so conjugation preserves the existing serialization
> behavior.

- [x] Verified, both halves, with a *module-level* function (a lambda is
      unpicklable whoever holds it): on `origin/main` the original and its
      conjugate both pickle to 788 bytes and the truncation survives the
      round trip — the conjugate trivially, since it *was* `self` — and on
      `721bae59` the original still pickles while `conjugate()` raises
      `AttributeError: Can't get local object
      'ControlledPhaseShift.conjugate.<locals>.<lambda>'`. So the head
      narrowed what the class supports, and the finding lands.
- [x] `conjugate()` wraps with `partial(negate_phases, self.function)`,
      `negate_phases` a module-level function, so the conjugate pickles
      whenever the original does, nested partials included. The negation
      stays in the data rather than a flag: that part of the shape is what
      `control.py`'s own `conjugate`s do.
- [x] Pinned with a test that pickles and unpickles the conjugate and the
      double conjugate of a box built from a module-level function and
      checks the truncation against the oracle after the round trip.
