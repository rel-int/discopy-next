cubic-dev-ai's review of 66100597, 11-06:

> **5 issues found** across 32 files
>
> - `test/monoidal.py:908` P2: This loop only compares each type with itself, since all six `types` are distinct, so `F(a) == F(b)` can never fail. Build two separately constructed equal types and compare their images, e.g. `Ty(Wire('x', red, green))` against `c`.
> - `discopy/tensor.py:1240` P2: `Bubble.setoid` drops the `time_step` term that `feedback.Bubble.setoid` adds, so bubbles that differ only in time step compare and hash equal. Include `self.time_step` in the tuple.
> - `discopy/cat.py:962` P2: `Bubble.setoid` omits `is_dagger`, unlike `Box.setoid`, so a daggered bubble can compare equal to its undaggered counterpart. Add `is_dagger` to the attribute tuple so equality and hashing track it.
> - `discopy/cat.py:971` P2: `Bubble.__str__` now prints `data` but still omits `is_dagger`, so string round-tripping a daggered bubble silently loses its dagger. Include `is_dagger=True` whenever the bubble is daggered.
> - `discopy/cat.py:1385` P2: `Transformation.id` now stores `identity` as components, so `components[x]` returns the object `x` rather than the identity arrow on `dom(x)`. Only `__call__` special-cases this. Consider keeping a callable that returns the identity arrow, or documenting that `components` must be read through `__call__`.

Each finding is checked against the code first: a real one is fixed with a test that fails on 66100597, and the others are answered on their thread.

- [ ] S1-6: the test compares the images of separately built equal wires
- [ ] S1-7: a bubble's identity and string count `is_dagger`, and a tensor bubble's its `time_step`
- [ ] S1-19: the components of `Transformation.id` read as identity arrows, or say they do not
