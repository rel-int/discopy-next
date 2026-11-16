> Review round on #567 (chatgpt-codex-connector[bot], cubic-dev-ai[bot]), head 9954f8ee:
>
> 1. discopy/monoidal.py:1815 (thread 4592884402)
> **  Avoid retaining unconditional effect dependencies across states**
>
> When a state is inserted into a gap joined by an effect, copying that gap's effect history onto both sides introduces dependencies that need not hold after interchange. For `e: x → I`, `s: I → x`, `h: x @ x → x`, and `d = x @ e >> x @ s >> h`, `dependencies()` returns `((), (), (0, 1))`, incorrectly requiring `h` to remain below `e`. Consequently, `d.foliations()` omits `x @ s @ x >> h @ e`, although `d.foliate(((1,), (0, 2)), left=False)` constructs this valid foliation using only right interchangers. Account for the state's possible placement relative to the effect rather than preserving the effect as an unconditional dependency on both new gaps.
>
> Useful? React with 👍 / 👎.
>
> 2. discopy/monoidal.py:1814 (thread 4592947364)
> [P2] Track the state’s placement relative to an effect instead of copying the gap’s effect history to both sides. This makes `h` depend on `e` and causes `foliations()` to omit the valid `x @ s @ x >> h @ e` foliation.

- [ ] discopy/monoidal.py:1815 ** Avoid retaining unconditional effect dependencies across states** When a state is inserted into a gap join…
- [ ] discopy/monoidal.py:1814 Track the state’s placement relative to an effect instead of copying the gap’s effect history to both sides.
