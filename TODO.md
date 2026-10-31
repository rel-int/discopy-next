# TODO

🌙 Evening's prompt for the night of 2026-10-31, verbatim:

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

This head takes the "constructors at zero" cluster of rel-int/discopy-next#314: five boxes whose
degenerate case — no photon, no bit, no leg — raises or lies about its own type. All five were
reproduced on `main` `adcd06b6` before the work started. Four are the fix their own module, sibling
box or protocol already states; the fifth, the two error channels at `prob` 0, turned out to need a
design decision and is left alone, with what it needs recorded below.

- [x] `photonic.Select()` and `photonic.Create()` stored `()` where the `zw` boxes they wrap
  default to `(1,)`, so `Select()`'s path matrix had domain 0 and no selection while the channel
  was typed on `qmode`: both now default to one photon on one mode, the `zw` docstrings' own
  `assert Select() == Select(1)` carried over.
- [x] `qubits.Z(0, 0).decomp()` was `Ty() -> qubit`, the `n == 0` branch assuming a leg out: the
  legs in and the legs out are now read independently, `X(0, 1) >> H()` for none in and
  `H() >> X(1, 0)` for none out, which leaves the three non-degenerate cases exactly as they were
  and makes the scalar spider decompose into a scalar. `qubits.X(0, 0).decomp()`, which goes
  through it, is fixed with it.
- [x] `classical.PostselectDigit()` and `classical.Digit()` declared `mode ** 0` while their
  `zw.Select()`/`zw.Create()` kraus defaults to one mode, and failed the arity assertion: they now
  declare the mode the kraus has.
- [x] `classical.PostselectBit()` raised `IndexError` reading `bits[0]` to start the fold: the fold
  now starts from `zx.Id(0)`, so the box is the tensor of one effect per bit and postselecting on
  no bit is the empty product, the scalar one. The non-empty diagrams are unchanged, box for box.
- [x] `qubits.BitFlipError(0)` and `DephasingError(0)`: **left alone, it needs a decision**, filed
  in this head's report rather than here. The `ZeroDivisionError` is `sqrt((1 - prob) / prob)`, and
  the environment bit's weight is `diag(sqrt(2 * prob), sqrt(2 * (1 - prob)))` — writing the two
  amplitudes instead of their ratio is division-free and reproduces every published value of the
  kraus, but **not of the channel**: `zx.Spider.conjugate` negates the phase, and a `ZBox`'s phase
  is an amplitude, so `ZBox(1, 1, 3).conjugate()` is `diag(1, -3)`. That sign is where the
  published `BitFlipError(p)` gets the minus in `p·I⊗I − (1 − p)·X⊗X`, i.e. its non-positivity, and
  any rewrite that changes how many `ZBox`es carry the weight moves it. So the box cannot be
  settled without ruling on what `conjugate` means for a `ZBox` — which is #314's own box 22 and
  next#410.
