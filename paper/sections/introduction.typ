#import "../template.typ": drawing, gap

= Introduction <sec:introduction>

A string diagram is a term of a free monoidal category, drawn. DisCoPy
implements string diagrams as a Python data structure and their
interpretations as functors, _i.e._ a `Functor` takes an object for each
wire and an arrow for each box, and evaluates a diagram in any category
that the user implements as a Python class. Its first description
@FeliceEtAl21 covered monoidal, rigid and symmetric categories with their
application to natural language processing, its second @ToumiEtAl22 the
quantum circuits and the ZX calculus, its third @ToumiEtAl23 the whole
hierarchy of graphical languages of Selinger's survey @Selinger10, one
module per level from `monoidal` to `frobenius`, with the `Hypergraph`
data structure for diagrams up to the axioms of symmetric categories and
above.

This paper describes what was added since. It has three parts and one
method.

- *Syntax* (@sec:syntax) is every free category. The hierarchy gained
  three levels, `feedback`, `spacial` and `comarkov`, and diagrams gained
  coloured regions, so that they are the 2-cells of a free premonoidal
  2-category. Beside `Diagram` and `Hypergraph` there are two more data
  structures for the same terms, `Term` for the lambda calculus of closed
  categories and `CMap` for diagrams in compact categories with their
  embedding in a surface. Rewriting is by double pushout, on hypergraphs
  and on planar maps, and by equality saturation on e-graphs of diagrams.
- *Semantics* (@sec:semantics) is every codomain of a functor that is not
  free. We add the constructions `op`, `para`, `lens` and `kleisli`, which
  take a category and give another, the representations of a Hopf
  algebra, constraint satisfaction, and drawing itself.
- *Applications* (@sec:applications) are neural networks, photonic and
  quantum computing, categorial grammars and first-order logic, each one
  closed by an example.

The method (@sec:method) is property-based testing. The previous papers
said which axioms each module implements. This one can say which axioms
each module was tested against, because the axioms are now part of the
library: a law is a method of an abstract base class, stated once, and
every category that inherits it is checked against it on terms it
generates itself. Where tests cannot reach, a universal property for
instance, the core is modelled and proved in Lean.

Two constructions that the library gained in the same period are left
out, since each has its own account: the `Int` construction
@JoyalEtAl96, also called the geometry of interaction, and the monoidal
streams of @DiLavoreEtAl22.

== How this paper is built <sec:build>

The paper lives in the repository of the library, in the directory
`paper`, and it is checked as the code is.

- Every listing runs. The Python listings are run in order by the test
  suite, in one namespace, so that a listing can use the names of the
  ones before it, and a listing that raises an error fails the build.
- Every figure is drawn by the library. A figure is the drawing of a
  value that a listing named, and the build draws it from that value:
  no picture is committed.
- The source is Typst and the LaTeX is written from it. The preprint is
  compiled from the Typst source, with its figures drawn by the Typst
  backend of the library, new with this paper. The journal's version is
  compiled from LaTeX, which a transpiler of a few hundred lines writes
  from the same source, with its figures drawn by the TikZ backend. The
  transpiler reads the small part of Typst that the paper is written in
  and refuses anything else, so that the two versions cannot silently
  differ. Both are built by the same continuous integration on every
  change.

#gap[
  The title and the list of authors are placeholders. The citation of the
  release that the paper describes waits for that release: the last tag
  is 1.2.2.
]
