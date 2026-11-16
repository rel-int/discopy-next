> Review round on #567 (chatgpt-codex-connector[bot], cubic-dev-ai[bot]), head dad2fb15:
>
> 1. test/monoidal.py:1149 (thread 4591895708)
> [P2] This states/effects test only verifies that returned foliations contain the same box names, so it can pass for an invalid ordering or mergeable adjacent layers. Check each result against the source diagram's normal form and assert adjacent layers cannot merge, as the preceding foliation test does.
>
> ```suggestion
>     diagram = s @ s >> f @ x >> e @ e
>     normal = lambda d: d.to_staircases().normal_form(
>         normalize=Diagram.normalize)
>     for foliation in diagram.foliations():
>         assert normal(foliation) == normal(diagram)
>         assert sorted(map(str, foliation.boxes)) == ["e", "e", "f", "s", "s"]
>         for upper, lower in zip(foliation.inside, foliation.inside[1:]):
>             with raises(AxiomError):
>                 upper.merge(lower)
> ```
>
> 2. discopy/monoidal.py:1919 (thread 4591895730)
> [P2] `limit=0` does not return zero foliations: the method appends the first result before checking equality, then continues because `1 != 0`. Handle nonpositive limits before enumeration and use a bounded check for positive limits.
>
> 3. discopy/monoidal.py:1911 (thread 4591895740)
> [P2] `foliations()` omits valid results when one foliation needs both left and right interchangers, because each candidate uses one global `left` flag for every move. Enumerate per-interchange choices or narrow the API contract so it does not claim to list all foliations.
>
> 4. CHANGELOG.md:12 (thread 4591895753)
> [P3] This changelog entry is hard to parse: the last clause 'interchangers then `Layer.merge`' has no verb, and the sentence chains four helper methods. Rewrite it as a short user-facing sentence, e.g. 'Diagram.foliations(limit), the foliations of a diagram: its equal-up-to-interchange diagrams whose adjacent layers cannot merge, largest layers first; Diagram.foliation() gives the first one ([next#567]).' Drop the internal helper names unless they are part of the public API.
>
> 5. discopy/monoidal.py:1912 (thread 4591895770)
> [P3] `itertools.product` materializes `self.layerings(...)` in full before the loop starts, so `limit` cannot stop the layering enumeration early. Iterate lazily, for example with a generator expression.
>
> ```suggestion
>         for layering, left in (
>                 (layering, left) for layering in self.layerings(
>                     staircase.dependencies()) for left in (False, True)):
> ```
>
> 6. discopy/monoidal.py:1912 (thread 4591897301)
> **  Consume layerings lazily before applying the result limit**
>
> For diagrams with many possible layerings, even `foliations(limit=1)` enumerates and stores every layering before constructing its first result: `itertools.product` eagerly consumes its input iterables. For two rows of six independent unary boxes, the layering generator already produces 44,823 candidates, although the first candidate suffices for this limit; larger diagrams incur rapidly growing runtime and memory usage. Iterate over layerings directly and use an inner loop over `(False, True)` so reaching the requested limit stops enumeration.
>
> Useful? React with 👍 / 👎.
>
> 7. discopy/monoidal.py:1797 (thread 4591897312)
> **  Preserve enclosing dependencies through intervening wire updates**
>
> When an intervening box updates one of the wires surrounding a state, comparing only their immediate producers loses the state's enclosing dependency. For `cup: I → x @ x`, `s: I → x`, and unary `f, g`, the diagram `cup >> f @ x >> x @ s @ x >> x @ g @ x` gets dependencies `((), (0,), (), (2,))`, although `s` must remain below `cup`. Consequently, `layerings` excludes `((0,), (2,), (1, 3))`, and `foliations()` misses the valid foliation `cup >> x @ s @ x >> f @ g @ x`, whose adjacent layers cannot merge. This requires only an unambiguous interchange of `f` and `s`, not the documented unsupported mixture of left and right interchangers. Track enclosing dependencies beyond immediate wire producers so this layering remains available.
>
> Useful? React with 👍 / 👎.
>
> 8. discopy/monoidal.py:1918 (thread 4591897323)
> **  Account for subclass merge rules when returning foliations**
>
> For symmetric diagrams containing consecutive native permutations, a dependency between layers does not guarantee that they cannot merge. For example, with `p = symmetric.Permutation(symmetric.Ty('x') ** 3, [1, 2, 0])`, `(p >> p).foliations()` returns the unchanged two-layer diagram because its dependencies permit only `((0,), (1,))`. However, `symmetric.Layer.merge` explicitly fuses these two layers into the nonidentity permutation `[2, 0, 1]`, so the returned diagram violates the documented requirement that adjacent layers cannot merge. The existing `(p >> p).foliation()` also produces that single fused layer. Account for the subclass's actual merge behavior rather than accepting every dependency-maximal layering as a foliation.
>
> Useful? React with 👍 / 👎.

- [ ] test/monoidal.py:1149 This states/effects test only verifies that returned foliations contain the same box names, so it can pass fo…
- [ ] discopy/monoidal.py:1919 `limit=0` does not return zero foliations: the method appends the first result before checking equality, then…
- [ ] discopy/monoidal.py:1911 `foliations()` omits valid results when one foliation needs both left and right interchangers, because each c…
- [ ] CHANGELOG.md:12 This changelog entry is hard to parse: the last clause 'interchangers then `Layer.merge`' has no verb, and th…
- [ ] discopy/monoidal.py:1912 `itertools.product` materializes `self.layerings(...)` in full before the loop starts, so `limit` cannot stop…
- [ ] discopy/monoidal.py:1912 ** Consume layerings lazily before applying the result limit** For diagrams with many possible layerings, eve…
- [ ] discopy/monoidal.py:1797 ** Preserve enclosing dependencies through intervening wire updates** When an intervening box updates one of …
- [ ] discopy/monoidal.py:1918 ** Account for subclass merge rules when returning foliations** For symmetric diagrams containing consecutive…
