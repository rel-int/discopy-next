# TODO

> actually, Category inheriting from Testable was fine, its Serialisable that shouldn't infect all concrete categories (for example python functions arent serialisable). all the sampling logic should be naive and implemented in a single function (no helpers outside of it) that samples diagrams by applying rules at random

- [ ] `abc.Category` inherits `Testable` again; what only serialisation needs moves from `Testable` to `Serialisable`
- [ ] One naive sampling function applying rules at random; the search, its plumbing and its helpers go
- [ ] Tests, docs, checks, changelog
