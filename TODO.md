> Asked in the Amp thread of this pull request: "the error for a wrong or missing offset does not mention the offset [...] Do you want me to fix that message and tighten `normal_form_soundness` to compare up to interchange before #658 is merged, or merge it as it stands?"
>
> toumix: "yep fix it"

- [WIP] @T-0729b388-26a3-4a3d-ac11-e727fdb06d74-2026-11-20 16:19 Kappa says what is wrong with an offset: missing, or not the one the variables or the stack ask for
- [ ] normal_form_soundness compares diagrams up to interchange, not up to the hypergraph, which forgets the side of a scalar
