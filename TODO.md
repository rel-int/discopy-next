# TODO

> There is some conceptual duplicate between discopy.group and discopy.hopf, i.e. a representation is an action in the category of vector spaces. How can we refactor this?

> bridge lives in hopf
> you can delete the duplicate no need for aliases

- [x] `Algebra.group_algebra(G)` takes a `group.Group` class: basis `G.elements()`, product `>>`, antipode `dagger`; the algebra keeps `G`
- [x] delete `Algebra.cyclic` and `Algebra.symmetric`, callers use `group_algebra(Cyclic[n])`, `group_algebra(Symmetric[n])`
- [x] `Representation.anyon` reads the group from the algebra instead of `argmax` on `mult`
- [x] `Representation.permutation(action)`: the linearisation of a `group.Action`, with doctests that `Action.tensor` is the tensor of modules and the orbits of `a.tensor(b)` count the intertwiners
- [x] serialisation of a group algebra still round-trips
- [ ] CHANGELOG, lint, pylint, ruff, ty, tests
