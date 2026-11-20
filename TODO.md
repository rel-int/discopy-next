> Review round on #665 (chatgpt-codex-connector and cubic-dev-ai):
>
> > Preserve optyx wire semantics in heads and tails: `qmode.head.single()` raises `KeyError`, `qmode.head.double()` returns one plain wire named `qmode.head`.
> > Match structural input types when initializing circuits: `Circuit.id(qubit.head).init_and_discard()` builds `Ket(0)` with codomain `qubit`, raising `AxiomError`; the discard and measurement branches likewise.
> > Recognize mixed circuits containing structural wires: `Circuit.id(bit.head @ qubit.head).is_mixed` is false.
> > `Box.__init__` still checks `isinstance(x, Digit/Qudit)`: a pure box on head or tail wires is rejected at construction.
> > CHANGELOG: the sentence is missing `is`.

- [x] optyx `Ob.single`, `Ob.double` and `is_classical` read a head through its argument
- [WIP] @T-43eb8dbf-2026-11-20 15:46 `Circuit.is_mixed` and `Box.__init__` read the kind of a wire with `is_quantum`
- [ ] `init_and_discard` and `measure` on head wires: need gates on structural types, i.e. #662
- [WIP] @T-43eb8dbf-2026-11-20 15:46 Changelog wording
