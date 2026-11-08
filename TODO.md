# TODO

## The prompt

From [next#472](https://github.com/rel-int/discopy-next/issues/472), the section
*"Why it needs no decision, and the one part that might"*, verbatim:

> `STYLE.md` states the oracle on the nose — *"`eval(repr(x)) == x` should always be true"* — and #445's own rulings already settled this exact shape twice, for `quantum.gates.Parametrized` (`S10-2`) and `optyx.qubits.BitControlledGate` (`S14-14`), both landed: **print through the public constructor, keep the data where `==` reads it**. Applying the same to these fifteen chooses nothing.
>
> The sixteenth might: `zw.ZBox`'s amplitudes are an `IndexableAmplitudes`, which may be lazy or infinite, and a `repr` cannot print what it cannot enumerate. Either `IndexableAmplitudes` gains an evaluable `repr` of its own for the finite case and `ZBox` delegates to it, or `ZBox` is declared outside the law with the docstring saying so. One word on that.
>
> This is also the `optyx` half of **#168**, our own serialization audit, whose `eval(repr(x)) == x` section lists twelve more violations in the rest of the library, open 19 days.

From [next#468](https://github.com/rel-int/discopy-next/issues/468), the whole body, verbatim:

> Found while checking cubic's findings on #464 (toumix's #445 calls for the core and the tower). It is on `main` at 72fec684.
>
> ```python
> >>> from discopy.quantum import zx
> >>> repr(zx.scalar(2))
> "quantum.zx.Scalar('scalar', rigid.Nat(0), rigid.Nat(0), data=2)"
> >>> eval(repr(zx.scalar(2)))
> TypeError: Scalar.__init__() got multiple values for argument 'data'
> ```
>
> `zx.Scalar.__init__` (`discopy/quantum/zx.py`, ~515–521) takes only `data`, while its `repr` comes from the generic box `repr`, which prints name, dom and cod first. That breaks transparency (`eval(repr(x)) == x`, STYLE.md). The fix is probably a `__repr__` that prints `quantum.zx.Scalar(2)`, with a test that it reads back.

## The work

One `__repr__` per class, printing the call to the public constructor, keeping the
data where `==` reads it — the shape `S10-2` and `S14-14` landed.

### `discopy/optyx/core/zw.py`

- [ ] `zw.W` — prints `W(1)` unqualified, and `dagger=True` is not the parameter name
- [ ] `zw.Create` — reads back to a box whose photons are its own name, dom and cod
- [ ] `zw.Select` — the same
- [ ] `zw.Endo` — prints a `data=` keyword its constructor does not take
- [ ] `zw.Add` — prints four positional arguments, takes two
- [ ] `zw.Multiply` — prints four, takes one
- [ ] `zw.Divide` — prints four, takes one
- [ ] `zw.Mod2` — prints four, takes one

### `discopy/optyx/core/zx.py`

- [ ] `zx.And` — prints four positional arguments, takes two
- [ ] `zx.Or` — prints four, takes two

### `discopy/optyx/core/diagram.py`

- [ ] `diagram.Scalar` — prints a `data=` keyword its constructor does not take
- [ ] `diagram.DualRail` — prints four positional arguments, takes two
- [ ] `diagram.PhotonThresholdDetector` — prints four, takes one
- [ ] `diagram.EmbeddingTensor` — prints `Dim(1)` unqualified and its array, not its dimensions

### `discopy/quantum/zx.py`

- [ ] `quantum.zx.Scalar` — next#468: prints name, dom and cod before the `data=` it takes
- [ ] `quantum.zx.Swap` — next#168's row: measure it, and put the law in the suite

### The suite and the changelog

- [ ] Parametrised `eval(repr(x)) == x` round-trip tests in `test/optyx/test_core_zw.py`,
      `test_core_zx.py`, `test_core_diagram.py` and `test/quantum/zx.py`
- [ ] A `CHANGELOG.md` bullet under `[Unreleased]` with its `[next#N]` reference links
- [ ] Every gate of `CONTRIBUTING.md` run through `uv run`, each verdict a delta against `main`

### Left alone, with a word asked

- [ ] `zw.ZBox` and `zw.IndexableAmplitudes` — the sixteenth and the one that is not a
      box, both waiting on the pick #472 quotes above. One 🚀-able comment on #472,
      not a line of code here.
