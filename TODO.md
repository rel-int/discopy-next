> I got some spare tokens to spend: let’s use them to polish all things discopy-next. Make a deep review of the code and the infra around it, summarising your findings and suggestions in an issue, fixing any bugs that you can fix without requiring a design decisions from my side.
>
> go ahead with the fixes once workflow A lands

The fixes of rel-int/discopy-next#445 in `hypergraph`, `frobenius`, `cmap`, `ehypergraph`, `group`: bugs the review confirmed that need no design decision, one commit and one regression test each.

- [x] **S5-1** Hypergraph.to_graph keys each boundary edge's spider node by the boundary object (e.g.
- [x] **S5-2** Hypergraph.__hash__'s fast path hashes repr(box), so equal hypergraphs whose boxes are equal but print differently (data=1 vs data=1.0, 0.5 vs …
- [x] **S5-4** frobenius.Diagram.unfuse leaves the Cospan plumbing fused:
- [x] **S5-5** Hypergraph.from_callable deletes the category's own __call__ in its finally, e.g.
- [x] **S5-7** deferred to #445: needs a one-pass rewrite of five methods, past 40 lines
- [x] **S5-10** Hypergraph.__repr__ drops the offsets that to_tree keeps:
- [x] **S5-11** Hypergraph.tensor copies each factor's offsets unshifted, so even x @ s.to_hypergraph() cannot be decoded in a category without swaps, though (x @ …
- [x] **S5-12** Hypergraph.permutation wires a list of compound types as if each were one wire:
- [x] **S5-14** Hypergraph.simplify's docstring says it takes quadratic time, but each pass decodes all n^2 interchanges (Omega(n^3), 31 s for an already minimal …
- [x] **S5-15** frobenius.Spider.__setstate__ names a legacy pickled spider with dom and cod swapped (and leaves typ unset, so its repr raises)
- [x] **S5-16** The shuffle half of test_Hypergraph_eq_fast_path_trace interchanges box 0 with itself (its trace has one box), and the fallback test compares only …
- [x] **S5-17** The docstring of Hypergraph.is_monogamous counts box inputs with the domain and box outputs with the domain again, so its formula is false on every …
- [x] **S5-18** relabeling sorts the labels of scalar (portless) spiders, so Hypergraph.__init__ raises TypeError when two scalar spiders have labels of incomparable …
- [x] **S6-1** CMap.copy/merge/spiders wrap whatever the category returns with from_box.
- [x] **S6-3** CMap.match runs complement on every one of the n^k straight-wire assignments that homomorphisms enumerates, so it is exponential in the pattern's …
- [x] **S6-4** CMap.leg_order hands Hypergraph.leg_order the legs in CMap port order, where a box's outputs are reversed.
- [x] **S6-5** Wreath rebuilds top.action() in degree() and permutation() on every element construction, multiply and dagger.
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S6-6** CMap.eval drops the dtype of its category.
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S6-7** CMap in a planar host reads a cup or cap between two outputs of one box in its clockwise port order (right to left).
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S6-8** test/cmap.py:924: the cup entry of test_from_glued_agrees_with_folding is dead.
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 full suite and lint gates on the branch
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 one `CHANGELOG.md` line per user-facing fix
