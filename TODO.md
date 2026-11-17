> let's reproduce lambeq's bobcat as discopy.lambek.categorial.Parser
>
> also we're not porting lambeq directly but Tommaso's reproduction of it

- [x] lambek.neural: Encoder and ChartClassifier, the model of the Bobcat reproduction
- [x] categorial.Parser.from_pretrained, decode, __call__: strict parse then missing spans, as the reproduction's command line
- [ ] same supertags and trees as the reproduction's pipeline on WSJ 23 with the real weights
- [ ] tests on a tiny random model, every check of Run the unit tests
- [ ] changelog entry
