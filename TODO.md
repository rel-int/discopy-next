add everything, fix everything

## Bugs
- [x] `twist = id` / `identity = id` register the `id` rule under another name
- [x] the Python functions lost their `id` rule (and keep stray ones)

## Laws
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 08:46 `HypergraphCategory`: spider fusion, specialness, commutativity
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 08:46 yanking: the trace of a swap is the identity
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 08:46 `BraidedCategory`: braid invertibility, both ways
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 08:46 `FeedbackCategory`: tightening, sliding, superposing
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 08:46 objects: tensor associativity and unit; delay zero, addition, tensor; unit exponentials
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 08:46 `BiclosedCategory`: the η law of currying, currying naturality
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 08:46 `ClosedCategory`: left and right currying agree up to a swap
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 08:46 `RigidCategory`: `cups_coherence`
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 08:46 `BalancedCategory`: twist naturality, twist of the unit
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 08:46 `MonoidalCategory`: `cut` is `x @ f @ y >> g`

## Rules
- [ ] `dagger` as a rule
- [ ] `delay` as a rule
- [ ] functor application through an `Image` pattern

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
