> Let's go with this neural redesign https://github.com/rel-int/discopy-next/issues/93
> Make sure you test we get parity on TRMP (without re-running the whole experiment)

- [x] slice 4 of the brief on #93: `Learner`, the parametric lenses of functions on tensors, with gradient descent as a reparametrisation and weight tying as `reparam` by the copy
- [x] `neural.torch.Module` holds the weights of a parametric network as its parameters (#124)
- [ ] tests that fail on `main`: one step of the learner equals one step of `torch.optim.SGD`, tied weights included
- [ ] CHANGELOG bullet, gates of next-checks, CI green with `main` merged in
