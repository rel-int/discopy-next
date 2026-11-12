> Let's go with this neural redesign https://github.com/rel-int/discopy-next/issues/93
> Make sure you test we get parity on TRMP (without re-running the whole experiment)

- [x] slice 5 of the brief on #93: `Network.unroll`, a trace as a number of rounds, with `rounds` for `to_function` and `to_lens`
- [x] the law, tested or refuted: the reverse derivative of `n` unrolled rounds equals autograd through the same rounds, and the rounds of the reverse derivative of the body, for `n` in 1, 2, 5
- [WIP] @session_01MEJxy4n4hekCBMayiVejgd-2026-11-12 11:07 CHANGELOG bullet, gates of next-checks, CI green with `main` merged in
