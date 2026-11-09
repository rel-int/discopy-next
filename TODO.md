> Review round on #1 (cubic-dev-ai[bot]), head a606e069:
>
> 1. discopy/testable.py:801 (thread 4508528258)
> [P2] The new `Atom` premise path assumes every object type implements `generator_strategy`, but `cat.Ob` only provides `strategy`; declarations over an `Ob`-based category therefore fail while drawing atomic premises. Fall back to the ordinary object strategy when no generator strategy exists, or provide the required method on `Ob`.
>
> ```suggestion
>                 return draw(
>                     ob.generator_strategy()
>                     if get_origin(sort) is Atom
>                     and hasattr(ob, "generator_strategy") else ob.strategy(),
>                     label=name)
> ```
>
> 2. test/group.py:56 (thread 4508654637)
> [P2] This deletion leaves five group instances without any axiom checks. Keep this parametrized test, or add these instances to the matrix’s checked parameters so edge cases such as orders 0 and 1 remain covered.

- [WIP] @session_01MEJxy65KogNCEukoKHtHe1-2026-11-09 01:30 discopy/testable.py:801 Atom premise over an object type with no generator_strategy
- [x] test/group.py:56 five groups the matrix does not enrol lose their law checks
