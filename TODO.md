> Review round on #556 (cubic-dev-ai[bot]), head a3e99c4b:
>
> 1. test/csp.py:3 (thread 4588771358)
> [P2] This module-level skip hides all CSP tests in installations without the optional `opt_einsum` package, including tests that do not need tensor contraction and functor cases that fit NumPy’s backend. Remove the global skip and isolate any optional-backend requirement to tests that explicitly need it.
>
> 2. README.md:36 (thread 4588808611)
> [P3] `Language.functor(bool)` returns only the projection of solutions onto `graph.cod_wires`, but this bullet says it returns the instance's solutions. State that it returns their output projection.
>
> ```suggestion
> * constraint satisfaction problems (`discopy.csp`): an instance is a `Hypergraph` with a variable as a spider and a constraint as a box on its scope, a language is a functor to `Tensor[bool]` that sends it to the projection of its solutions on its outputs, and its `int` interpretation counts them
> ```
>
> 3. discopy/csp.py:106 (thread 4588808642)
> [P3] The `scopes` round-trip claim is false for the supported bent-box case: `instance(scopes(graph), ...)` always rebuilds each constraint with an empty domain, losing its original input/output partition. Qualify the round-trip claim for no-input boxes or document that only the constraint scope, not the graph, is preserved.
>
> ```suggestion
>     The symbol and the scope of each constraint, so that an instance with no-input boxes can be reconstructed as
> ```
>
> 4. discopy/csp.py:180 (thread 4588819896)
> [P2] `holds` accepts values outside the language domains and assignments with the wrong length, so it can report an invalid assignment as satisfying an instance. Check the assignment length and each value against its variable's `domain` before evaluating predicates.
>
> ```suggestion
>         return (
>             len(assignment) == len(graph.spider_types)
>             and all(value in self.domain(sort)
>                     for value, sort in zip(assignment, graph.spider_types))
>             and all(
>                 self.predicates[name](*(assignment[i] for i in scope))
>                 for name, scope in scopes(graph)))
> ```

- [ ] test/csp.py:3 This module-level skip hides all CSP tests in installations without the optional `opt_einsum` package, includ…
- [ ] README.md:36 `Language.functor(bool)` returns only the projection of solutions onto `graph.cod_wires`, but this bullet say…
- [ ] discopy/csp.py:106 The `scopes` round-trip claim is false for the supported bent-box case: `instance(scopes(graph), ...)` always…
- [ ] discopy/csp.py:180 `holds` accepts values outside the language domains and assignments with the wrong length, so it can report a…
