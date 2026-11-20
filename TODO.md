> for the terms we need to implement kappa calculus so we can go all the way up to monoidal
>
> A key design point: the translation between diagrams and terms is only faithful for diagrams with exactly one output, that should be explained clearly in the tool paper. Of course we also have Product types and Pack boxes so that every diagram can be turned into one with exactly one output.

- [ ] discopy.kappa: planar linear kappa terms with one output (Variable, Constant, Application, Kappa), eval to a diagram of any monoidal category
- [ ] monoidal.Diagram.to_term: one let per box, refusing a diagram without exactly one output; normal_form inlines the lets
- [ ] laws on kappa.TermBase in the property matrix: alpha, beta (normal form), eta, the section to_term of eval, serialisation
- [ ] test/kappa.py, docs/api, CHANGELOG
