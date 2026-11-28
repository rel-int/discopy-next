> we need a better drawing for the feedback.cat entry of the hierarchy diagram, with the delay morphism from x to x.d drawn as a small loop together with an axiom at the feedback.symmetric level equating this to the feedback of the swap
>
> the delay morphism itself should be formulated as a natural transformation from identity to delay

- [x] abc: the wait, a natural transformation from the identity to the delay, `DelayCategory.wait(x): x -> x.d`, the identity by default, with its naturality axiom `f >> wait(B) == wait(A) >> f.d`
- [x] feedback.cat: a primitive `Wait` box, drawn as a small loop, with naturality declared on the free levels
- [ ] feedback.symmetric: the axiom `wait(x) == Swap(x, x.d).feedback()`, and `Diagram.wait` the primitive
- [ ] docs/architecture.py: feedback.cat shows naturality, feedback.symmetric the axiom; regenerate the SVGs and ABC.lean
- [ ] changelog fragment, tests, lint, proptest cells; reply to T-01a221f6 with a screenshot of the two cells
