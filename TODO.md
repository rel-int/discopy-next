# TODO

> every strategy method on `Testable[T]` should have as an exact signature `def strategy(self, pattern: Pattern[T]) -> st.SearchStrategy[T]`. parametrize the Pattern class to make this possible

Answers to the clarifying questions: patterns are runtime instances generic in
what they stand for, `Hom(dom, cod)` a `Pattern[C1]` and the objects
`Pattern[C0]`, absorbing the sorts; `strategy` is a classmethod.

- [ ] `Pattern[T]` instances: `Hom`, `Obj`, `Atom`, `Unit`, `Tensor` and the formers, the sorts folded in
- [ ] every `strategy` is `strategy(cls, pattern: Pattern[T]) -> st.SearchStrategy[T]`
- [ ] `Testable.sample` and the axioms sample from the instances
- [ ] tests, docs, CHANGELOG and checks
