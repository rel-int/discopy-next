# TODO

> nah let's drop that PR for now and open a fresh draft in discopy-next, reuse the property-based testing machinery from daydream's PR as much as possible so you can start by writing down the axioms that ehypergraphs should respect

Context: "that PR" is [discopy#730](https://github.com/discopy/discopy/pull/730), the 1255-line
`discopy.ehypergraph`; "daydream's PR" is [rel-int/discopy-next#1](https://github.com/rel-int/discopy-next/pull/1),
which this branch is stacked on. Earlier in the same session it was agreed that an e-hypergraph can be a
`hypergraph.Hypergraph` with an equivalence relation on its spiders, checked on #730's own example.

- [x] State the axioms of `ehypergraph.EHypergraph` in the sequent DSL of `discopy.axioms`
- [x] Implement them on top of `Hypergraph`, small enough to be the specification: `merge`, `rebuild`,
      `sections`, `extract` by brute force, `then`, `tensor`, `dagger`, equality up to isomorphism
- [x] `EHypergraph.strategy`, so that the symmetric, compact, Markov and Frobenius aliases enrol in the
      matrix: 101 laws pass, 2 skipped, 10 declared broken with their reasons
- [x] Settle with the human what `s ~ t` means over a linear category. @toumix, verbatim:
      > Yeah it doesn't make sense to equate two wires in an existing symmetric diagram but what does make
      > sense is a ehypergraph which holds a monogamous acyclic hypergraph and then applies a rewrite rule to
      > equate one subdiagram to a given RHS, adding that RHS to the ehypergraph and declaring equality of its
      > domain and codomain
- [x] `rewrite(region, dom, cod, rhs)` records the
      region and the right-hand side as alternatives; a section chooses a side of each, so that extraction holds
      over linear categories and `LINEAR_SECTIONS` goes

The linear e-hypergraphs above, their alternatives, `fits` and `Rewrites` (at `a763134`), and everything
traced (coinduction on feedback loops, `is_strict`, `markov.Square` and uniformity, the #91 port) moved to
[#103](https://github.com/rel-int/discopy-next/pull/103), which targets this one, as round 3 below asks.

## Round 2

> monoidal egraphs go brrr

@toumix's prompt for the session that picked this pull request up from a handoff of the session that opened
it, whose next steps are the boxes above and below. It worked on a pull request into this branch, #92, which
@toumix asked to fold back here: "keep only one PR open".

- [x] Fix [#91](https://github.com/rel-int/discopy-next/issues/91), the multiplicative trace raising on any
      loop: ported from next `main`'s #84, moved with the traced rewrites
- [x] `extraction_rewrite` over symmetric and compact categories: `rewrite` refuses what does not `fits`,
      moved with the linear e-hypergraphs

## Round 3

> let's keep traced rewrites for a next PR targeting this one, it's too much theory to guess at once
>
> let's focus on the case that's known best: cartesian diagrams are equivalent to terms so ehypergraphs over them should correspond precisely to egraphs. use egglog-python as a reference to stress case your implementation, add the example of egraphs from the standard docs (screenshot attached)
>
> then make a similar example with two primitives for sampling and conditioning that are not copyable and not discardable respectively

The screenshot is egg's: `(a * 2) / 2` under `x * 2 = x << 1`, `(x * y) / z = x * (y / z)`, `x / x = 1` and
`x * 1 = x`, four e-graphs representing more and more ways to write it, extracting `a`.

- [x] Move the traced rewrites to [#103](https://github.com/rel-int/discopy-next/pull/103), which targets this one
- [x] E-hypergraphs over copy-discard diagrams: a box is
      deterministic and affine unless it says otherwise, as @toumix asked on discopy#730 ("`is_deterministic`
      for the copy-naturality alone", "shouldn't be a ClassVar"); a section chooses one producer per class,
      so that over cartesian diagrams the e-hypergraph is an e-graph
- [x] Matching and saturation, the rules of egg's example
      applied until nothing changes, and extraction at least tree cost, as egg's
- [x] egglog-python as the reference: egg's example, and a
      differential stress test of the relation and the extracted cost
- [WIP] @session_01MY1Vs5CEhXgYwJcDnNwx6i-2026-10-17 18:40 The same with `sample`, not deterministic, and
      `condition`, not affine
