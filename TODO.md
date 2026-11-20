> we agreed on the following design: we perform introspection on the abc module to extract all definitions of categories with their axioms then compile them to Lean definition, then for each module we prove the corresponding coherence theorem i.e. that diagrams are the arrows of the free category with that structure

- [x] state the laws of whiskering on `abc.MonoidalCategory`, so that its theory without bifunctoriality is that of strict premonoidal categories (#669)
- [x] compile a `Functor` structure for each class of `abc`, the maps preserving its operations
- [x] prove the coherence of `monoidal` against the compiled definitions: Diagram(S) is the free model of `Discolean.ABC.monoidal` (#669)
- [ ] check the new laws on every category of the property matrix
- [ ] compile the rest of `abc` and prove one coherence theorem per module, filed as #670
