# TODO

> Perfect. Start this, and use a swarm of sonnet subagents to help you step by step. You are the orchestrator. At every step, where the code should run on GPUs, have an opus subagent decide speedups, implementations, etc. Work independently for a very long time, make many tests, make sure that what you do is correct.

> Actually, create a pr that will go into rel-int (discpy-next) that will explain the changes and what happened.

("This" in the first prompt: implement items 10.1 to 10.7, what `discopy.neural` lacks to be faithful to arXiv:2402.15332.)

- [x] 10.1 equivariant layers: `neural.equivariant`, fast realisations in `neural.factorised` (five slices)
- [x] 10.1b augmentation and symmetrisation: `neural.augment`
- [x] 10.2 datatypes and tied folds: `neural.datatype`
- [x] 10.2b state machines and tied unfolds: `Machine`, `Coalgebra`
- [x] 10.3 traces read as streams: `Network.unroll(sequence=True)`
- [x] 10.4 2-cells and laws up to them: `para.Cell`, `UpToPermutation`
- [x] 10.4 lax algebras and lax homomorphisms (App. J): `LaxHomomorphism`
- [x] 10.5 sampled check of strict homomorphisms: `strictness`
- [x] 10.6 `Para` over an actegory: `actegory`
- [x] 10.7 message passing as one diagram: `neural.diagram`, `stack`
- [x] land the fix of `Legs.forward` on an unbatched input into one point
- [x] land the fix of `stream.Ty` powers and `Stream.copy(Y, 0)` (branch `tommaso/stream-ty-power`)
- [ ] re-measure `fft_from` with cuDNN on
- [ ] merge `origin/main` and get CI green
- [ ] split into one PR per slice (< 1000 lines each), in the order of the description, and close this one
