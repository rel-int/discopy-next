# TODO

🌙 Evening's prompt for the night of 2026-10-30, verbatim:

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

Two more boxes of [#314](https://github.com/rel-int/discopy-next/issues/314), both about the
`is_conj` flag of `photonic.TBS` and `MZI`, both reproduced on `main` `13d68a38`, and the second is
why the first is worth fixing in the same pull request: the gradient reads the decomposition.

- [x] the global phase ORs `is_gate_dagger` and `is_conj`, so a conjugated gate carries the
      daggered gate's phase and `g >> g.dagger()` is not the identity — XOR them
- [x] `decomp` drops `is_conj`, so a conjugated gate decomposes into the unconjugated circuit and
      `grad`, which is `decomp().grad`, cannot see the flag either — conjugate the beam splitters
      and the phase shifters
