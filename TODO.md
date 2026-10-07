# TODO

> In `discopy/python/finset.py` at lines 195-198:
> ```python
>     @classmethod
>     @rule
>     def id[A](cls, dom: Var[int | Nat, A] = 0
>               ) -> Hom[Self, A, A]:
> ```
> similarly, make a rule called `ax` forwarding to the id method
>
> In `discopy/python/finset.py` at line 266:
> ```python
>     ) -> Hom[Self, A, C]:  # ty: ignore[invalid-type-form]
> ```
> make then a regular method, variadic by contract (no need to annotate it with a pattern as its a regular method), and a cut @rule that forwards to it. let me know every similar change i could make for the other rules, whenever it helps for simplifying the pattern EDSL, the implementation, code quality, more back compat, etc...
>
> In `discopy/python/finset.py` at line 275:
> ```python
>     ) -> Hom[Self, B, A]:  # ty: ignore[invalid-type-form]
> ```
> resolve the # ty: ignore[invalid-type-form] by simply using Nat here instead of Self, similar in the other cases
>
> In `discopy/monoidal.py` at line 1171:
> ```python
>             other: Hom[Diagram | None, C, D] = None,
> ```
> forget about this weird ass signature, let's just have mix as a @rule for the binary case and tensor as a method taking only `tensor(self, *others)`
>
>
> lets go back to making a clear distinction between rules (annotated) and methods (regular python typing, no description of coherence)

- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 12:00 `ax` rule forwarding to a plain `id` classmethod, on `abc.Category` and every implementation
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 12:00 `cut` rule forwarding to a plain variadic `then`; `mix` rule forwarding to a plain variadic `tensor`
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 12:00 `Hom[Nat, ...]` rather than `Hom[Self, ...]` in `python.finset` and the other `ty: ignore[invalid-type-form]`
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 12:00 Plain methods lose their pattern annotations: rules annotated, methods typed
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 12:00 Tests, docs, checks, changelog; report the other rules that could split
