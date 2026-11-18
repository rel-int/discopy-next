> chatgpt-codex-connector[bot], review of `45f250f`, two P2 findings:
>
> > Preserve beta diagrams and cuts when unfusing spiders (`discopy/peirce/beta.py:128`)
>
> > Default a beta model's functor to the beta category (`discopy/peirce/alpha.py:264`)

- [ ] frobenius.Diagram.unfuse builds its functor on the diagram's own category, so a beta cut survives it
- [ ] Model.functor() defaults to the category of the model
