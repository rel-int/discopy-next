> I got some spare tokens to spend: let’s use them to polish all things discopy-next. Make a deep review of the code and the infra around it, summarising your findings and suggestions in an issue, fixing any bugs that you can fix without requiring a design decisions from my side.
>
> go ahead with the fixes once workflow A lands

The fixes of rel-int/discopy-next#445 in `neural`: bugs the review confirmed that need no design decision, one commit and one regression test each.

- [x] **S12-1** neural.torch.Module keeps a compiled python.Function whose closures capture the original boxes' bound forward methods.
- [x] **S12-2** per_board_backward adds the excluded (halt.) rows of a board whose r_b is not finite unmasked, so with the default exclude a NaN board poisons …
- [x] **S12-3** neural.act's losses cast with .float() / .to(torch.float32), which narrows a float64 network's cross-entropy, board BCE and board losses to float32.
- [x] **S12-5** Geometric.cycle with rounds=0 differs from Recursion.cycle:
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S12-6** Shift with |step| > shape[axis] raises RuntimeError in narrow() instead of returning the all-zero translation its docstring defines
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S12-7** from_incidence and from_relation (and CMap.from_wiring) reject numpy integer indices, a tuple of np.int64 included, with a false 'factor/node 0 does …
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 full suite and lint gates on the branch
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 one `CHANGELOG.md` line per user-facing fix
