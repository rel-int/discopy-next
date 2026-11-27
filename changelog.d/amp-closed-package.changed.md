`discopy.closed` is a package, see
`docs/adr/2026-11-25-closed-is-a-package.md`: `closed.monoidal` is the free
biclosed category, `discopy.biclosed` before; `closed.symmetric` the free
symmetric closed category, new, with no copy or discard; `closed.markov`
the free closed Markov category, the module `discopy.closed` before, with
its terms, tuples, projections, `let` and Böhm trees. `abc.ClosedCategory`
is symmetric monoidal closed, no longer Markov, and
`abc.ClosedMarkovCategory` adds copy and discard. `abc.CompactCategory` and
`compact.Diagram` are closed, with `x ⊸ y = x* @ y`: evaluation and currying
stay cups and caps. `feedback.closed` is a package too, its
`feedback.closed.markov` the `feedback.closed` module before, over
`closed.markov`. A `lambek.modality.Functor` needs a biclosed codomain and
raises when constructed into any other, instead of sending exponentials
through `ob_map` at call time: into `Tensor`, compact hence closed, `N >> S`
is the internal hom and `Eval` a cup, and the strict and sloppy notebook,
which wants exponentials as generators, sends the underlying monoidal
diagram with a `monoidal.Functor`. The old names keep no alias, and a tree
or pickle saved under them does not load ([next#743](https://github.com/rel-int/discopy-next/pull/743)).
