> start a subthread to add spindle to discopy.lambek.abstract, also i just noticed the docs has bobcat stuff in neural which doesn't make much sense, let's make abstract a sub-subpackage of lambek with proofnet and neural as modules, then add a module spindle and also make a bobcat submodule of categorial in the same way

- [x] make discopy.lambek.abstract a subpackage with proofnet and neural as its modules
- [x] make discopy.lambek.categorial a subpackage with a bobcat module: the chart parser, Encoder and ChartClassifier
- [x] follow the imports in tests, docs, notebooks, the paper and the Lean statements
- [x] add discopy.lambek.abstract.spindle: SPINDLE from rel-int/spindle-ccg-deberta-v3-large, read as a ProofNet (its own pull request, design asked of toumix), filed as #673
