> another crazy idea we've been discussing for while with @Joseph_Fourment is to implement discopy.frobenius.Term based on the relational machine calculus https://arxiv.org/abs/2405.10801 — we could even restrict it to compact, pivotal, etc.

> discussing with @user_01M60ZNDDTYM6A278J514WBXK5, i think we want to preserve the fact that `A(lambda x: e)` means lambda-abstraction, while other sorts of binders would require an extra method on `Ty`, like `A.exists(lambda x: e)`. this way terms from one level would also work on higher levels of the hierarchy, but now as syntactic sugar. i also think we want locations from the beginning, so that we can more easily implement compilation to frobenius terms.

- [x] Read the paper and agree the design and the syntax with toumix and Joseph Fourment
- [x] `frobenius.TermBase`: values, push, pop, sequence, choice, star, exists, locations
- [x] Evaluation by the machine on spiders, `from_diagram`, `normal_form`, `Diagram.converse`
- [x] Laws in the property matrix: eta, normal forms, reading, dagger, functoriality
- [x] Changelog, ADR 12, the paper's examples as a test
- [ ] State β and ν as laws, which needs substitution on terms
- [ ] Draw choices into the strategy, and compare formal sums up to their hypergraphs
- [ ] The restrictions, one pull request each: compact, symmetric, markov, comarkov; pivotal and rigid as a hypothesis
- [ ] Review of the syntax and the design by toumix and Joseph Fourment
