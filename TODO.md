# TODO

🌙 Evening's prompt for the night of 2026-10-31, verbatim:

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

One box of #314: `**` with any `n <= 0` recursed forever — `zw.W(2) ** 0`, `photonic.BS ** 0` —
because three classes each carried the same floorless `def __pow__(self, n)`, on
`optyx.core.diagram.Box`, on `channel.Diagram` and, verbatim, on `channel.CQMap`, which already
inherited the second. The fix chooses nothing: the package states what `**` does at `n <= 0` in
`discopy/monoidal.py`'s `Ty.__pow__` — the identity on an endomorphism, an assertion otherwise —
so the three copies are deleted and that one clause is reused, as `discopy/stream.py` already
reuses it.

- [x] `optyx.core.diagram.Diagram` and `channel.Diagram` reuse `monoidal.Ty.__pow__`, the three
      floorless copies are gone, `channel.CQMap` inherits its base's, and a composite
      `optyx.core.diagram.Diagram` has a power at all, where only its boxes did.
