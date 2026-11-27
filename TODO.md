* rebase your discopy-next branch on https://github.com/rel-int/discopy-next/pull/763 and make the most out of the new architecture
* optimize the shit out of the boltzmann/ library
* simplify diagrams/ as much as possible, don't try to make ad-hoc adjustments for some axioms or categories, make a clear classification that makes the sampling predictable over a few different sorts of things we want to boltzmann-sample (types, diagrams, terms, cmaps/hypergraphs, ...). even better if you can share as much code across them
* streamline it, so that every axiom gets:
   * its own species, also generated from the recoloring signature
   * when the signature changes in discopy, automatically rebuild the weights and update them in the lake, per structure, reusing past weights as much as possible to hopefully accelerate convergence
* the sampling should work not just for diagrams, but any law-carrying class. for those that are not sampled via boltzmann, keep them in your discopy-next branch if you ever need to change them

- [x] Rebase onto #763 (amp/generators): constructors, providers, alphabet and type equations on slots
- [WIP] @session_01C7YfAQS26HQesQ3UXcBUjA-2026-11-27 12:40 The generators of a category (declared slots, rules with no hom premise) as the signature its species is read from: `generator_strategy(dom, cod, name)` on every class with rules
- [ ] Changes needed upstream for law-carrying classes that are not Boltzmann-sampled
