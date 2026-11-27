`linear.monoidal.Diagram` declares `dagger_involution` and `dagger_contravariance`
failing, since every structural box but the exchange has no dagger, and
`proptest/corpus/linear.py` replays them on the structural boxes its strategy
does not draw ([next#721](https://github.com/rel-int/discopy-next/pull/721)).
