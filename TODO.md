# TODO

> Can you solve the library bug, and the 'still open' parts? You can delegate it to a smart opus subagent

`Legs.forward` on an unbatched input whose output is one point with one channel called `reshape()` with no shape,
since every unit dimension is dropped: `Module(layer(Symmetric[5].action(), Action.trivial(Symmetric[5])),
torch.randn(()))(torch.randn(5))` raised `TypeError` where the batch `(1, 5)` gave `(1, )`.

- [x] `Legs.forward` and `Legs.restore` pass the shape to `reshape` as one tuple, so an empty shape is a 0-d tensor
- [x] a regression test: unbatched equals batched without its batch axis, invariant and equivariant, with and
      without a bias
