> toumix, in the Amp thread of this pull request: "mmh I think you're getting stuck on just monoidal when you should be implementing a Term class for every module, in fact let's make it go all the way to categories where terms are just sequences"
>
> toumix: "yeah i don't think kappa should be its own module, every module should have its own Term definition and all the ones strictly below biclosed are known as variants on the kappa calculus"

- [x] cat: the term machinery defined once (alpha-equivalence moves down from biclosed) and the terms of a category, sequences: cat.Term, Arrow.to_term, Ob.kappa
- [x] monoidal: the planar linear calculus moves from discopy.kappa into monoidal as subclasses of the terms of cat, and discopy.kappa is removed
- [ ] symmetric: terms with exchange, a variable used once in any order
- [ ] markov: terms with copy and discard, a variable used any number of times
- [ ] the modules in between (braided, traced, rigid, compact, ...): their own Term definition
- [ ] docs, changelog and README say the terms of each module below biclosed are variants of the kappa calculus
