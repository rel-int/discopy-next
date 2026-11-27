`lambek.categorial.Parser`, the chart parser of the reproduction of Bobcat
(Clark, 2021) on CCGrebank: a CKY chart over the binary and unary rules
seen in the treebank and the schemata of `schemata` at a penalty, scored by
the log-probabilities of a supertagger and a span classifier, each cell the
best `Derivation` of each category in a beam. A derivation is a categorial
diagram by `tree2diagram`. `structure`, `render` and `normalise` read the
categories of the treebank with their features ([next#603](https://github.com/rel-int/discopy-next/pull/603)).
