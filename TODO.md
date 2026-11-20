> Asked in the Amp thread of this pull request: "the error for a wrong or missing offset does not mention the offset [...] Do you want me to fix that message and tighten `normal_form_soundness` to compare up to interchange before #658 is merged, or merge it as it stands?"
>
> toumix: "yep fix it"

- [x] Kappa says what is wrong with an offset: missing, or not the one the variables or the stack ask for
- [x] normal_form_soundness compares diagrams up to interchange, not up to the hypergraph, which forgets the side of a scalar
