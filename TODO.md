# TODO

Re-cut of #103 (traced rewrites) from `main`, its default taken 11-21:
https://github.com/rel-int/discopy-next/pull/103#issuecomment-7055058857

> let's keep traced rewrites for a next PR targeting this one, it's too much theory to guess at once

> can we add an extra boolean attribute "strict" which characterises for which maps the uniform axiom holds?

> I disagree with “no way to state an implication”, this is what shapes do! eg if f g h are composable then associativity
> try to do the same with this uniformity axiom

> Using coinduction would be awesome! I expect that the rebuild should work fine as an iterator so you
> could unroll a feedback loop as many times as needed. Not sure if it makes sense to use coinduction to
> "prove" anything about Python though?

> decision A is fine

@toumix on #90 and #103. The picks A, C, B below were asked live in Daylight 11-09 and recorded at
https://github.com/rel-int/discopy-next/pull/103#issuecomment-6749985578; "linear first" is USER's word of
11-11 in Daylight.

- [ ] Port #103's slice at `1d965fd` onto `main`: `is_strict` on `feedback.Diagram` (a free box is not, a
      trace is when its argument is), strict `Permutation`, `Copy`, `Discard` and `comarkov.Merge`;
      `markov.Square[factory]` as a `TestableGeneric` shape built so that it commutes, like
      `biclosed.Reducible`; `trace_uniformity` on `markov.Diagram` declared failing (`NOT_UNIFORM`) and
      `trace_uniformity_written` holding; both inapplicable on `frobenius` (`NO_SQUARE`); uniformity of
      `python.Function`'s trace along a copy, and not along a discard
- [ ] Coinduction on loops, pick A: `bisimulation() -> Self`
      (options: https://github.com/rel-int/discopy-next/pull/103#issuecomment-6242891748)
- [ ] Strictness of a merge, pick C: strict means keeping ⊥, a spider is strict when it has an input
      (options: https://github.com/rel-int/discopy-next/pull/103#issuecomment-6242894146)
- [ ] The flags of a trace, pick B: deterministic from its argument, never affine
      (options: https://github.com/rel-int/discopy-next/pull/103#issuecomment-6246345485)
- [ ] E-hypergraphs over linear and hypergraph categories, pick A of 10-19: nested `Join`s inside the
      e-hypergraphs on `main`, linear first. Waits on USER's read of wiki#129's prototype (bundles, classes
      per wire). The four flaws #103 found at `a763134` are listed in its `TODO.md`
