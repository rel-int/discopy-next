> Review round on #567 (chatgpt-codex-connector[bot], cubic-dev-ai[bot]), head 5ee7e0df:
>
> 1. discopy/monoidal.py:1893 (thread 4592553687)
> **  Preserve left-interchanger choices when merging a layer**
>
> For `e: x → I`, `s: I → x`, and `f: x → x`, `(e >> s >> f).foliations()` omits the valid foliation `Diagram((Layer(e, s), Layer(f)), x, x)`, printed as `e @ s >> f`. This diagram has the same interchange normal form and is unchanged by `foliation()`. Both `left` attempts keep the original box order within the first layer, and this merge always uses `Layer.merge`'s right-biased normalization, producing `s @ e` instead. This requires only a left interchanger, not the documented unsupported mixture of directions. Preserve the directional choice during layer construction so this foliation is included.
>
> Useful? React with 👍 / 👎.
>
> 2. discopy/monoidal.py:1842 (thread 4592553695)
> **  Prune impossible continuations before enumerating their subsets**
>
> For independent boxes, there is exactly one layering, but exhausting this generator still explores every proper first-layer subset and then enumerates every subset of its remaining boxes, only to reject them all because none depends on the previous layer. Running the extracted method with `((),) * 14` took about 15 seconds to return its single result, versus 1.4 seconds at 12 boxes. Thus a modest-width tensor of independent boxes incurs roughly exponential search in `foliations()` despite having only one result; requesting a limit greater than one has the same problem. Check whether any free box depends on the previous layer before generating its combinations, and prune impossible continuations early.
>
> Useful? React with 👍 / 👎.
>
> 3. discopy/monoidal.py:1854 (thread 4592632243)
> [P2] `layerings` removes recursion but still copies and retains the full `layering` and `placed` state in every stack frame. A sequential diagram of N boxes therefore uses O(N²) memory, so sufficiently large diagrams can still exhaust memory; keep one mutable path/set and undo changes when backtracking.
>
> 4. discopy/monoidal.py:1843 (thread 4592632254)
> [P2] Check whether any free box depends on `previous` before generating combinations. For independent boxes, this generator explores exponentially many subsets only to reject every continuation, despite there being just one layering.
>
> ```suggestion
>             if previous and not any(
>                     set(dependencies[i]) & set(previous) for i in free):
>                 return iter(())
>             return (layer for size in range(len(free), 0, -1)
>                     for layer in itertools.combinations(free, size)
>                     if not previous or any(
>                         set(dependencies[i]) & set(previous) for i in layer))
> ```

- [x] discopy/monoidal.py:1893 ** Preserve left-interchanger choices when merging a layer** For `e: x → I`, `s: I → x`, and `f: x → x`, `(e …
- [x] discopy/monoidal.py:1842 ** Prune impossible continuations before enumerating their subsets** For independent boxes, there is exactly …
- [x] discopy/monoidal.py:1854 `layerings` removes recursion but still copies and retains the full `layering` and `placed` state in every st…
- [x] discopy/monoidal.py:1843 Check whether any free box depends on `previous` before generating combinations.
