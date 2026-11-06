# TODO

> Go ahead with the implementation in a new optyx notebook. We've moved the optyx work to discopy-next so you can push it as a PR there, or move everything you need from optyx if it's still open

The chip, from the conversation: a unitary U on M + M modes, a source giving a state psi on the first M inputs, the first M outputs fed back to the second M inputs, the other M outputs counted, and a fast switch on the loop modes set by a fixed function of the counts. The question: can this fixed nonlinearity make the chip universal, and can it be chosen so that the chip is trainable by equilibrium propagation (EP)? The plan: a fixed fast switch (a parity of counts controlling a rail swap), slow electronic units setting the preparation of psi at each tick, and EP on the slow units, whose energy is an expectation value of the chip.

- [WIP] @session_01XmqyhQuUzn8CQGX7CBCMw5-2026-11-06 01:30 a marimo notebook `docs/notebooks/optyx/feedforward_ep.md`: the chip as an optyx diagram with feedback, a classical function and a bit-controlled gate; its shot-by-shot samples; its exact averages, unrolled; the switch against no switch
- [ ] the slow units: an energy from the exact average, its gradient by parameter shifts, checked against `eval`
- [ ] EP on the slow units, checked against finite differences, then trained on a small task
- [ ] the notebook checked with `docs/export_notebooks.py --check`, a changelog entry
