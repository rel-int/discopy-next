> Review round on a07066f3 (cubic-dev-ai and chatgpt-codex-connector, 2026-11-27 20:02–20:05 UTC), verbatim:
>
> > **chatgpt-codex-connector** on `discopy/linear/symmetric.py:64`: **</sub>  Give symmetric linear terms a permutation-aware abstraction** For terms that exchange ordinary variables, this class still inherits the planar `term_factory`, whose `linear.monoidal.Kappa.__init__` requires bound variables to occur in their original order. For example, with this module's `x, y, z = map(Ty, 'xyz')` and `f = Box('f', y @ x, z)`, `(x @ y)(lambda a, b: f(b, a))` raises `ValueError` rather than evaluating to `Swap(x, y) >> f`. Overriding `crossing` fixes direct `Diagram.wiring` calls but cannot help because abstraction fails before evaluation reaches wiring. Supply permutation-aware linear term factories while retaining the restrictions on copying and discarding unbanged variables; `linear.markov` inherits the same limitation.
> >
> > **cubic-dev-ai** on `discopy/compact.py:214`: P2: The new traced functor base is bypassed for delayed compact boxes. Route delayed feedback values through `traced.Functor.__call__` while retaining the ribbon-specific dispatch, so compact functors preserve the delay semantics introduced here.
> >
> > **cubic-dev-ai** on `discopy/compact.py:193`: P2: This removes the ribbon-braid behavior from `compact.Swap`, despite the class still being documented as a ribbon braid. Keep the compact swap in the `ribbon.Braid` hierarchy so ribbon functors and braid operations continue to recognize it.
> >
> > **cubic-dev-ai** on `discopy/feedback/cat.py:234`: P2: `Arrow.generator_factory` still points to `cat.Box`, so generated arrows contain boxes that do not implement delay. Set the generator factory to this module’s `Box` alongside `box_factory`.
> >
> > **cubic-dev-ai** on `discopy/feedback/cat.py:97`: P2: Delayed bubbles render their delayed boundary expressions instead of the reset bubble, making `str` non-transparent unless the boundary names happen to be in scope. Render `self.reset()` and append the delay suffix when `self.args` is present.
> >
> > **cubic-dev-ai** on `discopy/feedback/cat.py:196`: P2: `Box.from_tree` bypasses the constructor's `time_step` validation, allowing malformed serialized boxes with negative or non-integer time steps. Validate the decoded value before assigning it, just as `Box.__init__` does.
> >
> > **cubic-dev-ai** on `discopy/linear/symmetric.py:64`: P2: Give this diagram a permutation-aware `term_factory`; abstraction currently rejects out-of-order variable use before `crossing` can emit the swap.
> >

- [WIP] @T-7f2f8ff8-2026-11-27 20:08 compact.py:193 compact.Swap is a ribbon.Braid again
- [ ] compact.py:214 delayed compact boxes through traced.Functor: check, answer
- [ ] feedback/cat.py:234 Arrow.generator_factory is feedback.cat.Box
- [ ] feedback/cat.py:97 a delayed bubble prints its reset bubble, delayed
- [ ] feedback/cat.py:196 Box.from_tree checks its time step as __init__ does
- [ ] linear/symmetric.py:64 permutation-aware linear terms (codex and cubic): file as an issue
