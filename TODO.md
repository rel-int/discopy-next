> Joseph and Alexis decided the hierarchy is too messy if every module is feedback and traced by default. Split feedback and bang into their own packages.
>
> "it's too messy to have every module be feedback/traced by default, we should have a feedback package with submodules for balanced/symmetric/markov/closed"
>
> "same for bang, it shouldn't be a branch separate from the rest of the package, it should be its own package with modules for everything that makes sense to bang. feedback and bang shouldn't have to interact until we've figured out a meaningful way to do so"

- [WIP] @amp-T-f829d881-2026-11-25 18:56 `discopy.feedback` a package: the monoidal level in `__init__`, then `balanced`, `symmetric`, `markov`, `comarkov`, `closed` and the levels below that carry a delay
- [WIP] @amp-T-f829d881-2026-11-25 18:56 the ordinary modules neither feedback nor traced: no time step, no `Trace` bubble; pivotal and below keep the trace their cups and caps give
- [ ] `discopy.bang` a package: one module per level that bangs
- [ ] no feedback-of-bang, no bang-of-feedback
- [ ] tests, proptest matrix, docs and changelog follow
