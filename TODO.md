> Review round on #566 (chatgpt-codex-connector[bot], cubic-dev-ai[bot]), head abd0a0d7:
>
> 1. discopy/testable.py:249 (thread 4591787950)
> **  Handle parameterized inequations in the property matrix**
>
> For an axiom annotated with `Inequation[Self]`, as in the new test, the property matrix raises `TypeError` before checking whether the inclusion holds. `proptest/test_axioms.py` calls `inhabits(verdict, law.output(scope))`, but `inhabits` only handles `origin is Equation`; an `Inequation[T]` falls through to `isinstance(value, kind)`, which rejects subscripted generics. Extend that handling to inequations and add a test exercising the matrix's output-validation path, rather than only calling the bound axiom directly.
>
> Useful? React with 👍 / 👎.
>
> 2. discopy/matrix.py:317 (thread 4591809807)
> [P1] Different matrix data types currently compare as `False` instead of raising `TypeError`, contradicting the documented contract and the added `R <= R.cast(int)` test. Check `self.dtype != other.dtype` and raise before evaluating the pointwise comparison.
>
> ```suggestion
>         if self.dtype != other.dtype:
>             raise TypeError(
>                 f"Matrices over {self.dtype} and {other.dtype} cannot be compared.")
>         return (self.dom, self.cod) == (other.dom, other.cod)\
>             and (self.array <= other.array).all()
> ```
>
> 3. test/testable.py:379 (thread 4591809854)
> [P2] The property matrix cannot validate this annotated output: `inhabits` recognizes parameterized `Equation` but not `Inequation`, so it falls through to `isinstance` with `Inequation[Self]` and raises `TypeError`. Extend `inhabits` to handle `Inequation` and test the output-validation path.

- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:26 discopy/testable.py:249 ** Handle parameterized inequations in the property matrix** For an axiom annotated with `Inequation[Self]`, …
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:26 discopy/matrix.py:317 Different matrix data types currently compare as `False` instead of raising `TypeError`, contradicting the do…
- [ ] test/testable.py:379 The property matrix cannot validate this annotated output: `inhabits` recognizes parameterized `Equation` but…
