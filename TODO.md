> Review round on #567 (chatgpt-codex-connector[bot]), head 05ae15a9:
>
> 1. discopy/monoidal.py:1804 (thread 4592274590)
> **  Preserve common enclosing ancestors when an effect joins gaps**
>
> When an effect joins gaps with different immediate enclosers, intersecting their indices loses shared enclosing ancestors. For `a: I → x @ x`, `b: x → x @ x`, `e: x → I`, `s: I → x`, and `g: x → x`, consider `a >> b @ x >> x @ e @ x >> x @ s @ x >> x @ g @ x`. The joined gaps have enclosers `b` and `a`, so their intersection is empty and `s` incorrectly receives no dependency on `a`. Consequently, `foliations()` omits `a >> x @ s @ x >> b @ g @ x >> x @ e @ x @ x`, although `foliate(((0,), (3,), (1, 4), (2,)), left=True)` constructs it and every adjacent pair is unmergeable. The earlier unary-wire example is fixed; this new example exposes dependency loss specifically when an effect closes a gap. Preserve enclosing ancestry before intersecting the joined gaps.
>
> Useful? React with 👍 / 👎.
>
> 2. discopy/monoidal.py:1842 (thread 4592274599)
> **  Enumerate deep layerings without recursive stack growth**
>
> For a sequential diagram of 1,000 unary boxes, `foliations(limit=1)` raises `RecursionError` under Python's default recursion limit instead of returning its single foliation. Each layer adds another recursive generator frame, and the first result cannot be yielded until the entire dependency chain has been traversed, so the result limit cannot prevent this failure. This affects deep circuits even when there are no alternative layerings to enumerate. Use an explicit traversal stack so supported diagram depth is not bounded by Python's call-stack limit.
>
> Useful? React with 👍 / 👎.

- [x] discopy/monoidal.py:1804 ** Preserve common enclosing ancestors when an effect joins gaps** When an effect joins gaps with different i…
- [x] discopy/monoidal.py:1842 ** Enumerate deep layerings without recursive stack growth** For a sequential diagram of 1,000 unary boxes, `…
