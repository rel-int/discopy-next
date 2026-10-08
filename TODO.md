# TODO

> actually, Category inheriting from Testable was fine, its Serialisable that shouldn't infect all concrete categories (for example python functions arent serialisable). all the sampling logic should be naive and implemented in a single function (no helpers outside of it) that samples diagrams by applying rules at random

- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-08 09:00 `abc.Category` inherits `Testable` again; what only serialisation needs moves from `Testable` to `Serialisable`
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-08 09:00 One naive sampling function applying rules at random; the search, its plumbing and its helpers go
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-08 09:00 Tests, docs, checks, changelog
