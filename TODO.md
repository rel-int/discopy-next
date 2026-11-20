> yes let's go: the lean folder should be renamed to discolean
>
> we agreed on the following design: we perform introspection on the abc module to extract all definitions of categories with their axioms then compile them to Lean definition, then for each module we prove the corresponding coherence theorem i.e. that diagrams are the arrows of the free category with that structure

- [x] rename lean/ to discolean/ and update every path that points at it (Python, CI, docs)
- [x] make discolean a Lake package that requires Mathlib, built and audited in CI with Mathlib's cache
- [x] compile the categories of discopy.abc and their axioms to Lean definitions, from introspection, checked in and tested to be up to date
- [ ] state the coherence theorem of monoidal against the compiled definitions, from the existing proof that diagrams are the free premonoidal category
- [ ] file the coherence theorems of the other modules as issues
