`benchmark/test_egraph.py` replays those traces through `egraph.Builder`
and the bookkeeping of the search that grew them, the small ones through
the pure `EGraph`, and times a rebuild and a freeze ([next#503](https://github.com/rel-int/discopy-next/pull/503)); what it
measured against the targets it was given is in [next#504](https://github.com/rel-int/discopy-next/issues/504).
