> for the terms we need to implement kappa calculus so we can go all the way up to monoidal
>
> A key design point: the translation between diagrams and terms is only faithful for diagrams with exactly one output, that should be explained clearly in the tool paper. Of course we also have Product types and Pack boxes so that every diagram can be turned into one with exactly one output.

- [x] discopy.kappa: planar linear kappa terms with one output (Variable, Constant, Application, Kappa), eval to a diagram of any monoidal category
- [x] monoidal.Diagram.to_term: one let per box, refusing a diagram without exactly one output; normal_form inlines the lets
- [x] laws on kappa.TermBase in the property matrix: alpha, beta (normal form), eta, the section to_term of eval, serialisation
- [x] test/kappa.py, docs/api, CHANGELOG
- [x] the tool paper explains one output: paragraph and listing drafted on rel-int/wiki#195 (issuecomment-7039254628), to land there once this is on main, filed as rel-int/wiki#195
