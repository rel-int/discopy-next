> Let's go with this neural redesign https://github.com/rel-int/discopy-next/issues/93
> Make sure you test we get parity on TRMP (without re-running the whole experiment)
>
> And, live the same day, on the brief of that issue:
>
> add the belief propagation law to the brief, we'll check this on the TRMP repo
> this is an important theoretical direction: neural GoI to tackle constraint satisfaction problems

- [x] slice 6 of the brief on #93: `neural.propagation`, one round of sum-product on a factor table as a `Network` whose trace is the message state
- [WIP] @session_01MEJxy4n4hekCBMayiVejgd-2026-11-12 11:39 the law on trees: after `diameter` rounds the beliefs equal the marginals of the contraction, on random factor tables with random positive tensors; the test fails if a factor's message includes the node it is sent to
- [ ] the 4×4 sudoku, which has cycles: nothing asserted, the gap per round reported in the pull request
- [ ] CHANGELOG bullet, gates of next-checks, CI green with `main` merged in
