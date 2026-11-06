toumix, 11-05:

> I got some spare tokens to spend: let’s use them to polish all things discopy-next. Make a deep review of the code and the infra around it, summarising your findings and suggestions in an issue, fixing any bugs that you can fix without requiring a design decisions from my side.

toumix, 11-06:

> use the questions interface so that I can take the decisions from here

The readings toumix picked for the core and the tower, recorded on #445 (comment 6681874391). Each is one commit with a regression test that fails on its parent.

- [ ] S1-5: the composite of two `DualRail`s is the level's plain `Functor`
- [ ] S1-6: `Wire` equality and hash count `is_dagger`
- [ ] S1-7: a bubble's `data` is part of its identity, as for `Box`
- [ ] S1-19: `Transformation` mirrors `Functor`: a shared identity and dict components, so the category laws hold
- [ ] S1-24: `Equation ==` compares type, terms, symbols, space and `up_to` (a callable `up_to` by identity); `repr` prints the symbols
- [ ] S2-1: `failing` raises only when the equation is false, so a fixed law XPASSes
- [ ] S2-5: a non-element on either side of a monoid's `@` raises `TypeError`
- [ ] S2-6: `str(Nat(2))` reads `Nat(2)`, so `str_transparency` holds on the nose
- [ ] S3-5, S8-3: a delayed bubble carries its own `time_step`, with a delayed boundary, and stays opaque
- [ ] S6-2: `CMap.loops` are stored sorted and compare as a multiset
