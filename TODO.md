add everything, fix everything

## Bugs
- [x] `twist = id` / `identity = id` register the `id` rule under another name
- [x] the Python functions lost their `id` rule (and keep stray ones)

## Laws
- [ ] `HypergraphCategory`: spider fusion, specialness, commutativity
- [ ] yanking: the trace of a swap is the identity
- [ ] `BraidedCategory`: braid invertibility, both ways
- [ ] `FeedbackCategory`: tightening, sliding, superposing
- [ ] objects: tensor associativity and unit; delay zero, addition, tensor; unit exponentials
- [ ] `BiclosedCategory`: the η law of currying, currying naturality
- [ ] `ClosedCategory`: left and right currying agree up to a swap
- [ ] `RigidCategory`: `cups_coherence`
- [ ] `BalancedCategory`: twist naturality, twist of the unit
- [ ] `MonoidalCategory`: `cut` is `x @ f @ y >> g`

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
