> Review round on #571 (cubic-dev-ai[bot]), head dc7c5cde:
>
> 1. discopy/abc.py:1479 (thread 4593131410)
> [P2] This law proves only a right inverse for `exchange`, not invertibility. Because `is_dagger=True` is just an argument to an abstract method, an implementation can satisfy every declared law while `exchange(..., True) >> exchange(...)` is not the identity; add a separate axiom enforcing the reverse composition as well.
>
> 2. discopy/testable.py:534 (thread 4593131419)
> [P3] The `discopy.closed` example is incorrect: its `Ty` and `Diagram` inherit the banged-object implementation from `biclosed`, so this bang is not the identity. Remove that example or replace it with a category that actually defines bang as identity.
>
> ```suggestion
>     identity, reads each object as its own bang.
> ```
>
> 3. CHANGELOG.md:16 (thread 4593131455)
> [P3] `x.bang()` is the containing type, not the `BangOb` wire itself. Describe `BangOb` as the wire holding the type inside `x.bang()` to match the API.
>
> ```suggestion
>   the free one: `BangOb`, the wire holding a type inside `x.bang()`, `Bang`, the
> ```
>
> 4. CHANGELOG.md:19 (thread 4593131469)
> [P3] This sentence has an unfinished possessive and obscures the dispatch rule. Name the codomain's bang support and state that the functor uses ordinary maps otherwise.
>
> ```suggestion
>   codomain's bang support, or through its ordinary maps otherwise. The laws
> ```

- [WIP] @session_01MEJxy2QJCwia83KHppaJ1H-2026-11-16 23:29 discopy/abc.py:1479 This law proves only a right inverse for `exchange`, not invertibility.
- [WIP] @session_01MEJxy2QJCwia83KHppaJ1H-2026-11-16 23:29 discopy/testable.py:534 The `discopy.closed` example is incorrect: its `Ty` and `Diagram` inherit the banged-object implementation fr…
- [WIP] @session_01MEJxy2QJCwia83KHppaJ1H-2026-11-16 23:29 CHANGELOG.md:16 `x.bang()` is the containing type, not the `BangOb` wire itself.
- [ ] CHANGELOG.md:19 This sentence has an unfinished possessive and obscures the dispatch rule.
