# TODO

> let's keep traced rewrites for a next PR targeting this one, it's too much theory to guess at once

@toumix on [#90](https://github.com/rel-int/discopy-next/pull/90), which this pull request targets, while #90
turns to cartesian diagrams, whose e-hypergraphs are e-graphs. What moved here from #90 is done but for the
questions. #90's e-hypergraphs over copy-discard diagrams are merged in at `71fcecc`: they are acyclic, refusing a
feedback loop (`NOT_SHARING`), so what is left here is what they need to take traces, and linear or hypergraph
categories.

- [x] Mark the maps the trace is uniform along. @toumix, verbatim:
      > can we add an extra boolean attribute "strict" which characterises for which maps the uniform axiom holds?
      `is_strict` on traced diagrams and `trace_uniformity` along copy, declared failing on the free
      categories, `NOT_UNIFORM`
- [x] State the premise of uniformity as a shape. @toumix, verbatim:
      > I disagree with “no way to state an implication”, this is what shapes do! eg if f g h are composable then associativity
      > try to do the same with this uniformity axiom

      `markov.Square[C]` draws commuting squares by construction; `trace_uniformity_written` holds on
      every free traced Markov category but the hypergraph ones, where no square is drawn, `NO_SQUARE`
- [x] Fix [#91](https://github.com/rel-int/discopy-next/issues/91), the multiplicative trace raising on any
      loop: ported from next `main`'s #84, identical; uniformity along copy holds on `python.Function`,
      checked over `markov.Square` mapped into monotone Boolean functions; it fails along a discard, whose
      loop need not terminate
- [ ] Settle with the human whether congruence on a feedback loop should be coinductive, for e-hypergraphs
      over traced diagrams: #90's refuse a cycle, and the least fixpoint it declared broken on loops went with
      its linear e-hypergraphs. @toumix, verbatim:
      > Using coinduction would be awesome! I expect that the rebuild should work fine as an iterator so you
      > could unroll a feedback loop as many times as needed. Not sure if it makes sense to use coinduction to
      > "prove" anything about Python though?

      The API of `bisimulation`, [asked](https://github.com/rel-int/discopy-next/pull/103#issuecomment-6242891748)
      as options A, B and C
- [ ] Ask the human what strictness should mean for a merge, which `is_strict` declares strict and #90 flags
      deterministic but not affine, and for its unit `Merge(x, 0)` and the frobenius spiders, which say they are
      not strict: [asked](https://github.com/rel-int/discopy-next/pull/103#issuecomment-6242894146) as options A
      to D, the premise "not natural along a copy" being wrong, as the question says
- [ ] The flags of a trace: `markov.Trace` keeps the defaults of a box, so the trace of a sample says it is
      deterministic and that of a condition that it is affine. The trace of a non-deterministic or non-affine map
      is neither, which reading the flags off its argument gives; whether the trace of a total function is
      total, i.e. affine, is a question for the human, a loop needing not terminate
- [ ] E-hypergraphs over linear and hypergraph categories, as #90 had them at `a763134`: alternatives between
      regions and right-hand sides, `fits` and `Rewrites`. Found against them, to fix when they come back:
      `rebuild` merges the outputs of two congruent boxes that coexist, e.g. `a @ a >> m`, so no section is
      valid; the spiders of a right-hand side on no box leak into the sections that drop it; `fits` accepts a
      right-hand side that drops a wire the region passes through; and matching needs the region to lie in
      one section and its alternatives to be nested, which loses overlapping redexes
