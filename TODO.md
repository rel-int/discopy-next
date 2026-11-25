> Review round 1, cubic-dev-ai on f4915fca (https://github.com/rel-int/discopy-next/pull/729#pullrequestreview-6061258625), 17 findings, in short:
>
> > 1. egraph.py:261 [P0] `Join(...)` passes `time_step` into `markov.Box` ... every join construction raises `TypeError`.
> > 2, 8, 12. notebooks: `Cut` cannot be daggered or deserialised, its constructor accepts only `arg`.
> > 3, 4. optyx `Feedback` (core and channel): `loads(dumps(feedback))` fails, its constructor cannot consume `name`, `args`, `method`.
> > 5. test/feedback.py:523: keep the `module.Bubble` alias assertion.
> > 6, 9. CHANGELOG, ADR 0004: `Bubble` is `Elementwise` in tensor and quantum, not `Box.from_args`.
> > 7, 11, 15. cat/monoidal/ribbon: trees dumped with a `*.Bubble` factory no longer load.
> > 10, 13, 17. tensor/quantum: `Bubble` is still the `Elementwise` class, alias `Elementwise.from_args`.
> > 14. peirce/alpha.py:217: `Cut` accepts a `method` it does not serialise.
> > 16. tensor.py:1429: `Elementwise.__repr__` emits raw arrays for `data`.

- [WIP] @T-14209019-2026-11-25 16:36 1: Join with time_step (reproduce first)
- [WIP] @T-14209019-2026-11-25 16:36 2, 8, 12: notebook Cuts rebuild and dagger
- [WIP] @T-14209019-2026-11-25 16:36 3, 4: optyx Feedback round-trips its tree
- [ ] 7, 11, 15: legacy Bubble trees load
- [ ] 10, 13, 17: tensor and quantum Bubble is Elementwise.from_args; 16: Elementwise repr of data
- [ ] 5, 6, 9, 14: alias test, changelog and ADR wording, Cut method
