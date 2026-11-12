> Let's go with this neural redesign https://github.com/rel-int/discopy-next/issues/93
> Make sure you test we get parity on TRMP (without re-running the whole experiment)

- [x] slice 3 of the brief on #93: `Box.vjp`, `Box.backward` and `Network.to_lens`, the reverse derivative as a functor into the lenses of functions on tensors
- [x] tests that fail on `main`: `to_lens().put` equals `torch.autograd.grad` of `to_function` on random networks, and fails if the copy drops a branch
- [WIP] @session_01MEJxy4n4hekCBMayiVejgd-2026-11-12 10:14 CHANGELOG bullet, gates of next-checks, CI green with `main` merged in
