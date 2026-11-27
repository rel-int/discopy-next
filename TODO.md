> Every functor class declares the abc its codomain must subclass, and the constructor raises when it does not (toumix, approving the inventory: "we already decided to move the stream primitives to a new subpackage discopy.dataflow, otherwise all your propostions make sense"; then, for Array, Tensor and Channel while Hypergraph and CMap are figured out in their own thread: "yes let's go").

- [ ] matrix.Array: the array plumbing, sums and delay, declaring no category; Matrix and Tensor are siblings on it
- [ ] Tensor declares HypergraphCategory and DaggerMonoidalCategory, with its own braid, merge and basis
- [ ] quantum.Channel on Array, dagger compact with copy and discard on classical wires, to_tensor an explicit doubling functor
- [ ] peirce.alpha tests for Array rather than Matrix
- [ ] requires on every functor class, checked at construction, call-time fallbacks deleted
- [ ] Hypergraph and CMap, after the design thread T-d55d90a2 reports
- [ ] merge amp/dataflow once it is pushed
- [ ] changelog, ADR, checks
