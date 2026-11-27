`closed.Hypergraph` and `lambek.abstract.Hypergraph` declare
`dagger_involution`, `dagger_contravariance` and `dagger_monoidality`
failing, as their categories do: a curry has no dagger. The strategy draws
no curry, so `proptest/corpus/hypergraph.py` gives one to each
([next#722](https://github.com/rel-int/discopy-next/issues/722)).
