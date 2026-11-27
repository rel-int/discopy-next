`proptest/test_corpus.py` replays a committed corpus of terms written by hand
or by a language model, `proptest/corpus/`, against the laws of the property
matrix, deterministically and offline, with a first corpus for
`symmetric.Diagram`: shapes the strategies draw rarely or never, e.g. boxes
that share a name or carry `data`, bubbles and multi-wire feedback. A pilot
on `symmetric`, `compact` and `Int` found no counterexample beyond those
already declared ([next#707](https://github.com/rel-int/discopy-next/pull/707)).
