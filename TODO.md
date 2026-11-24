> it's not surprising that LLM fuzzing on a small set of axioms would not find bugs but maybe we can let the LLM pick the axiom that it thinks most likely to fail?
>
> Go ahead with 30 picks and the random control

- [x] Rank the live cells and publish 30 picks, each with its reason and predicted counterexample
- [ ] Hunt the picks: explore-budget Hypothesis and hand-written candidates
- [ ] Control: the same explore budget on 30 random live cells
- [ ] File each counterexample as an issue, declare its law failing, add its entry to the corpus
- [ ] Record the result on #43
