add everything, fix everything

## Bugs
- [x] `twist = id` / `identity = id` register the `id` rule under another name
- [x] the Python functions lost their `id` rule (and keep stray ones)

## Laws
- [x] `HypergraphCategory`: spider fusion, specialness, commutativity
- [x] yanking: the trace of a swap is the identity
- [x] `BraidedCategory`: braid invertibility, both ways
- [x] `FeedbackCategory`: tightening, sliding, superposing
- [x] objects: tensor associativity and unit; delay zero, addition, tensor; unit exponentials
- [x] `BiclosedCategory`: the η law of currying, currying naturality
- [x] `ClosedCategory`: left and right currying agree up to a swap
- [x] `RigidCategory`: `cups_coherence`
- [x] `BalancedCategory`: twist naturality, twist of the unit
- [x] `MonoidalCategory`: `cut` is `x @ f @ y >> g`

## Rules
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 09:03 `dagger` as a rule
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 09:03 `delay` as a rule
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 09:03 functor application through an `Image` pattern

## Structure the semantic categories have
- [ ] `Matrix`, `Tensor`, `Channel` declare the structure they implement
- [ ] `Hypergraph[C]` is a `HypergraphCategory`

## Broken declarations
- [ ] serialisation of twists, copies and spiders (#742)
- [ ] `Functor.unitality`
- [ ] free currying, modulo `to_compact`
- [ ] rigid `normal_form_soundness`, `foliation_soundness`, `foliation_idempotence`
- [ ] `pivotality` on pivotal and ribbon
- [ ] frobenius `foliation_idempotence`
