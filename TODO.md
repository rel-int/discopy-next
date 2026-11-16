> > adding pointwise inequality to matrices is a must, we should also extend slightly the property testing from equations to inéquations
>
> > Go!
>
> For the first-order logic of the tool paper's plan, #564: its axioms are inclusions of relations, checked in Boolean matrices.

- [x] `Matrix.__le__`, the pointwise order on parallel matrices, which `Tensor` inherits
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:14 `testable.Inequation`, an `Equation` whose terms are compared with `<=`, so that an axiom can state an inclusion
- [ ] tests, doctests, CHANGELOG.md bullet, every gate of next-checks
