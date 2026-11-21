> Let's brainstorm with Giovanni on how to deduplicate the features that are in both discopy and optyx e.g. channels and doubling. [...] yeah draft it and show us some diagrams to check it works. [...] don't leave cpstar.py uncommitted. Merge origin/main, commit it, push a branch and open a draft PR with a TODO.md listing the open items (generic CPM/CPStar classes, box names, the conjugate convention question for Giovanni).

- [ ] Generic `CPM[category]` and `CPStar[category]` classes, stacked as in `para`, with their laws in `testable` (double is a functor, discard is natural, Kraus channels are causal) enrolled in the proptest matrix
- [ ] Box names: discopy.quantum's `Measure(destructive, override_bits)`, `Encode(constructive)`, `MixedState` against optyx's `Measure`, `Encode`, `CQMap`; settle with Giovanni
- [ ] Conjugate convention, question for Giovanni: discopy's pivotal conjugate (mirrors wires, swap-free environment cup, used by the draft) or optyx's box-wise one; pivotal conjugate on optyx boxes raises (#410)
- [ ] optyx's `core.diagram.Diagram` needs a cup of its own (#661) so that `double` works on optyx without the stand-in
- [ ] Unit tests in `test/cpstar.py` against `discopy.quantum` mixed eval and optyx's `double`, docs page and changelog bullet
