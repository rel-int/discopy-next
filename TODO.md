> some time ago, on the original property testing PR (discopy-next#1), i implemented the feature of auto-generating box subclasses so that we no longer had to do the whole class Swap(symmetric.Swap): ... and Diagram.swap_factory = Swap gymnastic, but instead invert control by automatically generating Diagram.Swap and optionally add a shortcut to each module as Swap = Diagram.Swap. We already have @rules, which describe the internal logic of each category as inference rules. In this refactor that never landed (which you can consult at commit ef86d2e9435c0705d9febc51cb8e5a30dc14034a), I added a special kind of rule for axioms (in the inference rule sense, a @rule with no hom premisse, not an @axiom), which I called Generator, which replaced factories altogether by dynamically creating (and caching) subclasses of each Box from one level of the hierarchy to another, using the systematic inheritance pattern. I ended up removing it from the property testing PR to simplify it, but we're considering re-introducing it on the latest discopy-next.
> Analyse the linked commit and the current state of discopy-next and report to me how to best re-integrate that global refactor and make the most out of it. The ultimate goal is to do away with all `*_factory`s of discopy and ease the definition of new diagram classes such a way that code grows linearly instead of quadratically in the number of generators.
>
> [after a grilling session whose settled design is https://ampcode.com/artifacts/re-integrating-generator-from-factory-to-declared-generators-273913b5-b368-46f6-8672-a7ba93fcb3c6]
>
> yes, get to work, machine. let's drop the max 1000 LOC diff constraint on this PR and do everything on the same one. after implementing the necessary infrastructure, the global refactor itself will be mechanical and hopefully delete a bunch of LOCs too.

- [x] Mechanism: split Declaration (classmethod moves to Rule and Axiom), add Slot with per-level building, .extend mixins, lazy module exports, subscripts carrying their slots, the *_factory deprecation shim and the tests of ef86d2e9 (is_generator/generator stay, as atoms() already means the generators a term names)
- [x] cat and monoidal on slots, equation_factory becomes Equation
- [x] The symmetric spine on slots: braided, balanced, traced, symmetric, markov, comarkov, spacial
- [ ] Dualities on slots: rigid, pivotal, ribbon, compact, frobenius, tensor (dtype fix), hypergraph, cmap
- [x] Exponentials and time on slots: biclosed, closed, feedback, stream, bang, interaction
- [x] Terms on slots, on Ty and TermBase
- [x] Applications on slots: lambek, quantum, optyx, neural, peirce, egraph and the rest
- [ ] Generator(Slot, Rule): rule and premises on root classes, one functor dispatch through box.rule, Rule.admissible
- [ ] Plumbing as an admissible rule; structural_boxes read from declarations, the boundary law replacing assert_structural_boxes
- [ ] Regenerate DisCoLean/ABC.lean if needed, remove the shim, docs and changelog
