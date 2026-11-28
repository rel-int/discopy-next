> > yes they should declare their level but the main issue is the contravariance: monogamous acyclic hypergraphs are a subclass of hypergraphs but the symmetric monoidal categories they represent are a superclass of hypergraph categories... how do we implement that faithfully in Python?
>
> > we do have planar hypergraphs implemented if i remember correctly, with each box coming with its offset? for boundary connected the offset can be computed from the structure alone and to_diagram should respect it already
>
> > Fold both into one PR stacked on #744
>
> — Alexis (toumix), in https://ampcode.com/threads/T-d55d90a2-06b7-4f25-afbc-a18d59e963db, on the proposal there: `Hypergraph[C]` and `CMap[C]` declare the abcs of the level `C` they represent, a subscript checks the wiring of its level when it is built, and `to_diagram` computes the offset of a state from the structure. #744 merged into `main` before work started, so this branches from `main`.

- [x] `Hypergraph[C]` and `CMap[C]` declare the abcs of `C` up to `HypergraphCategory`, set when the subscript is built
- [x] `from_diagram` and `to_compact` take the highest functor whose `requires` the subscript satisfies; `requires` is checked on the subscript, not its origin
- [x] a subscript refuses at construction a wiring that is not a term of its level (producers and consumers per spider, cycles, planarity)
- [x] `to_diagram` places a state from the wire it feeds, so the offsets of a boundary-connected hypergraph are not needed
- [ ] property matrix: the new Hypergraph and CMap cells, currying declared failing as on the free closed diagrams
- [ ] docs, changelog, ADR, `ABC.lean` regenerated, every check of CONTRIBUTING.md
