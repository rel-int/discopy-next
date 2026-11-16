#import "template.typ": paper

#show: paper.with(
  title: [DisCoPy: string diagrams tested against their axioms],
  authors: ("The DisCoPy developers",),
  abstract: [
    DisCoPy is a Python toolkit for computing with string diagrams. This
    paper describes what was added to it since the hierarchy of graphical
    languages and the hypergraph diagrams of its last description. The
    method is what changed most. Every law of every structure of the
    hierarchy is now stated once, as a typed method of an abstract base
    class, and checked by property-based testing on every category that
    implements the structure: more than two thousand pairs of a law and a
    category, each one holding on the terms drawn for it, or declared
    broken with its reason. The free categories are the syntax, and we add
    lambda terms, combinatorial maps, coloured regions, double-pushout
    rewriting and e-graphs of diagrams. The codomains of functors are the
    semantics, and we add Kleisli categories, lenses, parametric maps, the
    representations of a Hopf algebra, constraint satisfaction and drawing
    itself, which is now a functor. The applications are neural networks,
    photonic and quantum computing, categorial grammars and first-order
    logic. Every listing of this paper runs in its test suite, but for
    the source that it quotes from the library, and every figure is drawn
    by the library from the listing above it.
  ],
)

#include "sections/introduction.typ"

#include "sections/method.typ"

#include "sections/syntax.typ"

#include "sections/semantics.typ"

#include "sections/applications.typ"

#bibliography("../docs/discopy.bib")
