# TODO

> remove the sentence generation for pregroup, it should generate arbitrary diagrams like the others. cur down the code and make everything uniform

- [x] pregroup and circuit sample arbitrary diagrams by `Testable.sample` like every other level: their `generators`, `strategy` overrides, `VOCABULARY` and the marks that only held for sentences or gate sets go
- [x] `pattern.Constant`, `Rule.constant` and the vocabulary branch of `Testable.sample` and `Testable.enrolled` go
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-08 16:00 the matrix cells the two levels gain are checked, and declared where they break
- [ ] tests, docs, CHANGELOG and checks

> what about making it a fuel: int = MAX_FUEL argument  on strategy?
>
> next, can we generate Ty's the same way as diagrams, from rules? I think all Ty's of discopy already are categories themselves, so it should work out of the box. continue cutting down features from the exoratory development phase of this branch until the bare minimum has a good architecture and matches the code quality of a 10x engineer

- [x] `strategy(cls, pattern=None, fuel=MAX_FUEL)`: a premise of any category is sampled by its strategy with the fuel left, so `Testable.max_depth` and the recursion special case go
- [x] a free category samples its terms by its rules, types included: `FreeCategory.strategy` samples, `FreeCategory.generators` draws its terms of length one, the strategies of `Ty`, `Box` and the circuit `Ty` go, an object of a monoidal category is a term on the unit colour
- [x] objects have no size but zero or one: `Obj[C0, N]`, size variables and the counts-first ordering go
- [ ] survey the rest of the branch for exploratory leftovers and cut them
