> Review round on #558 (chatgpt-codex-connector[bot], cubic-dev-ai[bot]), head e9689535:
>
> 1. discopy/csp.py:318 (thread 4589543875)
> **  Enforce equal values for repeated variables in a scope**
>
> When a constraint mentions the same variable more than once, the Cartesian product assigns its occurrences independently, so tuples that are not valid assignments can count as supports. For example, with domain `{0, 1}`, `instance([("neq", (0, 0))], [x])` and predicate `a != b`, propagation returns `[{0, 1}]` even though neither value has support: the predicate must receive `(a, a)`. Repeated scopes are already supported by `instance` and exercised in `test_instance_scopes_roundtrip`. Enumerate assignments over distinct scope variables and expand them into argument positions before evaluating the predicate.
>
> Useful? React with 👍 / 👎.
>
> 2. discopy/csp.py:318 (thread 4589543894)
> **  Accumulate supported values without retaining every tuple**
>
> For high-arity constraints with many satisfying tuples, this list consumes exponential memory even when propagation removes nothing. For example, the single positive clause produced by `cnf([list(range(1, 21))])` retains 1,048,575 twenty-element tuples; running the extracted method on that constraint reached approximately 227 MiB peak RSS. A 25-literal clause requires several GiB, making otherwise small CNF inputs liable to exhaust memory. Accumulate a set of supported values per scope position while iterating instead of retaining all satisfying tuples, and stop scanning once every candidate has support.
>
> Useful? React with 👍 / 👎.
>
> 3. discopy/csp.py:318 (thread 4589544953)
> [P1] `propagate` treats repeated variables in a constraint scope as independent values. For example, `neq` on `(0, 0)` incorrectly keeps both candidates although `holds` and `solutions` correctly evaluate both positions from the same assignment value. Enumerate distinct scope variables and reuse each value for every occurrence.
>
> ```suggestion
>                 variables = tuple(dict.fromkeys(scope))
>                 supports = [
>                     tuple(values[variables.index(i)] for i in scope)
>                     for values in product(*(candidates[i] for i in variables))
>                     if predicate(*(values[variables.index(i)] for i in scope))]
> ```
>
> 4. discopy/csp.py:319 (thread 4589544996)
> [P2] A false nullary constraint is silently ignored because the support loop has no variables to prune. Clear the candidate sets (or otherwise record inconsistency) when a nullary predicate is false, so propagation does not report full domains for an unsatisfiable instance.
>
> ```suggestion
>                 if not scope and not predicate():
>                     if any(candidates):
>                         candidates = [set() for _ in candidates]
>                         changed = True
>                     continue
>                 for k, variable in enumerate(scope):
> ```
>
> 5. test/csp.py:231 (thread 4589545014)
> [P3] The new tests never pass `domains` to `propagate`. Add a case that restricts the candidates, and check that the caller's sets are not mutated.

- [WIP] @session_01MEJxy5FM3MT3hk7p67LLmW-2026-11-16 18:06 discopy/csp.py:318 ** Enforce equal values for repeated variables in a scope** When a constraint mentions the same variable more…
- [WIP] @session_01MEJxy5FM3MT3hk7p67LLmW-2026-11-16 18:06 discopy/csp.py:318 ** Accumulate supported values without retaining every tuple** For high-arity constraints with many satisfyin…
- [WIP] @session_01MEJxy5FM3MT3hk7p67LLmW-2026-11-16 18:06 discopy/csp.py:318 `propagate` treats repeated variables in a constraint scope as independent values.
- [WIP] @session_01MEJxy5FM3MT3hk7p67LLmW-2026-11-16 18:06 discopy/csp.py:319 A false nullary constraint is silently ignored because the support loop has no variables to prune.
- [ ] test/csp.py:231 The new tests never pass `domains` to `propagate`.
