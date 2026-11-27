#import "../template.typ": drawing, gap

= Syntax: the free categories <sec:syntax>

The library has four data structures for the same string diagrams, lists
of layers, lambda terms, combinatorial maps and hypergraphs, and a fifth
for sets of equal diagrams, the e-graph. They differ in the axioms that
their equality decides, and each one converts to and from diagrams.

== The hierarchy <sec:hierarchy>

A `monoidal.Diagram` is a domain and a list of layers, each a list of
boxes and types with at least one box. Composition concatenates the
lists and the tensor is `f @ g.dom >> f.cod @ g`, one diagram after the
other. With one box per layer, these are the morphisms of the free
strict premonoidal category on the boxes @PowerRobinson97, which is
proved on the Lean model of @sec:lean. Every other level is a subclass
with more boxes, which states its axioms by inheriting a class of
`discopy.abc` (@sec:laws). Three levels are new since @ToumiEtAl23.

- `feedback` is the free feedback category @KatisEtAl02 @DiLavoreEtAl22.
  Its objects have a `delay`, and `feedback` takes a diagram from
  `x @ m.delay()` to `y @ m` to one from `x` to `y`
  (@fig:syntax_feedback). `abc.FeedbackCategory` states seven laws, two
  for the delay then vanishing, joining, strength, tightening and
  sliding. A traced category is the case where the delay is the
  identity: `abc.TracedCategory` is a subclass, and the module `traced`
  is deprecated for this one. The `feedback` package has a module for
  each level that carries a feedback or a trace:
  `feedback.balanced`, `symmetric`, `markov`, `comarkov` and `closed`.
- `spacial` is the free spacial category @Selinger10, where a scalar on
  the left of a diagram is the same scalar on its right. A free scalar
  is a box, which does not slide on a list of layers, and the law is
  declared `failing` there (@sec:declarations). It holds on maps, where
  matching finds the scalar on either side, and modulo hypergraphs from
  symmetric categories on.
- `comarkov` is the dual of `markov` @FritzLiang23, with a commutative
  monoid on every object: the boxes `Merge` and `Noise`, where `markov`
  has `Copy` and `Discard`. `abc.HypergraphCategory` inherits from both.

```python
from discopy import comarkov, feedback, spacial

x, y, m = map(feedback.Ty, "xym")
f = feedback.Box('f', x @ m.delay(), y @ m)
syntax_feedback = f.feedback(mem=m)
assert (syntax_feedback.dom, syntax_feedback.cod) == (x, y)
s = spacial.Box('s', spacial.Ty(), spacial.Ty())
h = spacial.Box('h', spacial.Ty('x'), spacial.Ty('x'))
assert s @ h != h @ s
assert len(list((h @ s).to_map().match((s @ h).to_map()))) == 1
a = comarkov.Ty('a')
unitality = comarkov.Noise(a) @ a >> comarkov.Merge(a)
assert comarkov.Equation(unitality, comarkov.Id(a))
```

#drawing("syntax_feedback", caption: [
  The feedback of a box `f` from `x @ m.d` to `y @ m`, with memory `m`.
]) <fig:syntax_feedback>

== Coloured regions <sec:regions>

The types of a monoidal category were the free monoid on its objects,
and are now a free coloured monoid. A `Colour` is a 0-cell and a `Wire`
a generating 1-cell with a colour on either side, `None` for no colour.
A `Ty` is a composable path of wires: its tensor composes paths, which
raises an error when the colours do not meet, and `Ty.id(c)` is the
empty path on the colour `c`. A box has to be globular, _i.e._ its
domain and its codomain go from the same colour to the same colour.
Diagrams are then 2-cells, which `>>` composes along a type and the
tensor along a colour, with no more interchange than they had with one
colour. The wires of the other levels take colours as well, _e.g._ the
adjoint of a rigid wire swaps them, and a functor takes a `colour_map`.
The listing states an adjunction: two categories as regions, two
functors as wires, the unit and the counit as boxes, and one of the two
snakes.

```python
from discopy.monoidal import Box, Colour, Id, Ty, Wire

C, D = Colour('lightblue', label='C'), Colour('lightyellow', label='D')
F, G = Ty(Wire('F', C, D)), Ty(Wire('G', D, C))
assert ((F @ G).dom, (F @ G).cod) == (C, C) and Ty.id(C) @ F == F
unit = Box('unit', Ty.id(C), F @ G)
counit = Box('counit', G @ F, Ty.id(D))
snake = unit @ F >> F @ counit
assert (snake.dom, snake.cod) == (F, F) and snake != Id(F)
```

The name of a colour is the colour that fills its region. The matplotlib
backend fills the regions of a diagram, and `draw(legend=True)` adds a
legend with the `label` of each colour.

#gap[
  The picture of this snake. The Typst and TikZ backends, which draw the
  figures of this paper, do not fill regions yet. The Lean model of
  @sec:lean has no colours, and the property matrix draws no coloured
  type.
]

Plumbing changed for symmetric diagrams too. A `symmetric.Permutation`
is a box that reorders its wires, `Swap` being the permutation of two
wires, and a `symmetric.Layer` takes permutations as plumbing beside
types: a permutation is one layer where it was a staircase of swaps.

== Terms, maps and hypergraphs <sec:structures>

=== Lambda terms <sec:terms>

A `Term` is a lambda term in the internal language of a closed category.
The terms of `biclosed` are planar and linear, with an exponential on
each side, `x >> y` and `y << x`. Those of `closed`, now the free closed
Markov category, have one exponential, products with tuples and
projections, `let`, and variables that occur any number of times. A type
applied to a string is a constant, a type applied to a Python function
is an abstraction, whose variable takes the name of the parameter, and a
term applied to a term is an application.

A term is also a box, and `eval` compiles it to a diagram, with a
`Curry` bubble for an abstraction and an `Eval` box for an application.
Conversely, `Diagram.to_term` reads a diagram with copies and discards
as a term with one `let` for each box (@fig:syntax_term). Equality of
terms is syntactic, and `alpha_eq` is up to the names of bound
variables. `reduce` contracts beta redexes, leftmost outermost by
default, down to a head normal form, returned as a `closed.BohmTree`
whose arguments are reduced when they are read, and `normal_form` is the
term of that tree. Both take a budget of redexes, since a term need not
have a normal form: `Unitype` is its own exponential, and its terms are
those of the untyped lambda calculus. The property matrix checks that
the normal form is idempotent, keeps the type and is the same for two
strategies.

```python
from discopy.closed import Box, Diagram, Ty, Unitype

X, Y, U = Ty('X'), Ty('Y'), Ty(Unitype())
two = (X >> X)(lambda f: X(lambda x: f(f(x))))
four = (X >> X)(lambda f: two(two(f)))
assert str(four.normal_form())\
    == "(X >> X)(lambda f: X(lambda x: f(f(f(f(x))))))"
assert two.alpha_eq((X >> X)(lambda g: X(lambda y: g(g(y)))))
delta = U(lambda x: x(x))
assert U == U >> U and delta(delta).reduce(budget=100) is None
f, h = Box('f', X, Y @ Y), Box('h', Y @ Y, Y)
syntax_term = Diagram.copy(X) >> f @ Diagram.discard(X) >> h
assert str(syntax_term.to_term())\
    == "let(f(x0), lambda x1, x2: h(Tuple(x1, x2)))"
```

#drawing("syntax_term", caption: [
  A closed Markov diagram, which `to_term` reads as
  `let(f(x0), lambda x1, x2: h(Tuple(x1, x2)))`.
]) <fig:syntax_term>

=== Combinatorial maps <sec:maps>

A `CMap` is an open combinatorial map @DelpeuchVicary22: an involution
with no fixed point on the ports, its `edges`, where the ports of a box
are listed clockwise and the boundary of the diagram is one more node.
It is the wiring of a diagram with the order of the wires around each
box, which a hypergraph forgets. `to_map` sends a diagram of any level
to its map. The edges and the rotation around each node give the faces,
hence the `genus`, and `is_planar` says that it is zero.

A map is always a morphism of a compact category. `to_diagram` is where
it has to be one of its own category: cups and caps are made explicit
if the category has them and cycles become traces if it has those, with
an error otherwise. `to_compact` opens the `Curry` bubbles of a closed
diagram, so that a lambda term has a map. At the planar levels, a map is
the map of a diagram exactly when a line can sweep it downwards:
`CMap.sweep` gives the order in which the line meets the boxes with the
offset of each one, `is_progressive` says whether there is one, and a
connected map is then its diagram up to interchange.

```python
from discopy import monoidal, symmetric
from discopy.drawing import spiral

x, y, z = map(symmetric.Ty, "xyz")
g = symmetric.Box('g', x @ y, z)
assert g.to_map().is_planar
assert (symmetric.Swap(y, x) >> g).to_map().genus == 1
syntax_spiral, w = spiral(2), monoidal.Ty('x')
unit, cap, _, counit, cup, _ = syntax_spiral.boxes
planar = syntax_spiral.to_map()
assert planar.is_progressive and planar.to_diagram() == syntax_spiral
rules = [((cap >> w @ counit).to_map(), unit.to_map()),
         ((unit @ w >> cup).to_map(), counit.to_map()),
         ((unit >> counit).to_map(), monoidal.Id().to_map())]
assert len(list(planar.normalize(*rules))) == 5
```

#drawing("syntax_spiral", caption: [
  The spiral of @DelpeuchVicary22 with two turns: a unit, two caps, a
  counit in the centre and two cups.
]) <fig:syntax_spiral>

=== Hypergraphs <sec:hypergraphs>

A `Hypergraph` is a cospan of hypergraphs @BonchiEtAl22: boxes and a
wiring of their ports, and of the boundary, to spiders. `Hypergraph[C]`
has the boxes of the category `C`, and its equality is isomorphism. The
levels of the hierarchy are properties of the wiring. It is bijective
when every spider has two ports or none, as in a compact category,
monogamous when every spider is written once and read once, as in a
traced one, and causal when every spider is written once, before it is
read, as in a Markov one. `to_diagram` decodes a hypergraph with the
structure that its category has, and raises an error when the wiring
needs more: in the listing, the same loop is a trace over symmetric
diagrams and a cap and a cup over compact ones.

Every module of the hierarchy has an `Equation`, a list of diagrams that
is true when they are equal up to a function `up_to`. It is the identity
for the levels with no symmetry, and the translation to hypergraphs for
`symmetric`, `markov`, `comarkov`, `closed`, `compact` and `frobenius`.
The naturality of the swap is not an equality of lists of layers, and it
is one of hypergraphs (@fig:syntax_swap).

```python
from discopy import compact
from discopy.feedback.symmetric import Box, Equation, Swap, Ty

x, y = Ty('x'), Ty('y')
f, g, k = Box('f', x, x), Box('g', y, y), Box('k', x @ x, x @ x)
syntax_swap = Equation(f @ g, Swap(x, y) >> g @ f >> Swap(y, x))
left, right = syntax_swap.terms
assert left != right and syntax_swap
assert left.to_hypergraph().is_causal
loop, c = k.trace().to_hypergraph(), compact.Ty('x')
assert loop.is_monogamous and not loop.is_causal
assert str(loop.to_diagram()) == "Trace(k)"
bent = compact.Hypergraph(
    c, c, (compact.Box('k', c @ c, c @ c), ), loop.wires)
assert str(bent.to_diagram())\
    == "x @ Cap(x, x.r) >> k @ x.r >> x @ Cup(x, x.r)"
```

#drawing("syntax_swap", caption: [
  Two symmetric diagrams, different as lists of layers and equal as
  hypergraphs.
]) <fig:syntax_swap>

== Rewriting <sec:rewriting>

A rule is a pair of parallel diagrams, and a match of its left side is
seldom a sublist of the layers. Rewriting is therefore on hypergraphs,
on maps and on e-graphs.

On hypergraphs it is by double pushout. `Hypergraph.match` gives the
homomorphisms of a pattern that have a pushout complement. Over a
hypergraph category, this is rewriting modulo the Frobenius axioms
@BonchiEtAl22. Over a symmetric category the hypergraph has to be
monogamous and acyclic, and the matches are the convex ones, where no
path leaves the image and comes back @BonchiEtAl22b: this is sound for
the symmetric diagrams of the library, and misses the rewrites that
need the trace of `discopy.feedback.symmetric`.
`rewrite` gives every rewrite in one step, `normalize` the steps of a
reduction and `normal_form` its last one. `hypergraph.Rewriter` does the
same in place, for rules that are left-linear, _i.e._ the boundary of
the left side goes to distinct spiders and the complement is unique.
Boxes and spiders keep their numbers, so that a step costs the size of
the rule and not that of the host. The simplifier of ZX diagrams is
built on it (@sec:quantum). In the listing, the unit law of a monoid
applies where a swap separates the unit from the product
(@fig:syntax_rewrite).

```python
from discopy.hypergraph import Rewriter
from discopy.symmetric import Box, Hypergraph, Swap, Ty

x = Ty('x')
m, u = Box('m', x @ x, x), Box('u', Ty(), x)
unit_law = (u @ x >> m).to_hypergraph(), Hypergraph.id(x)
assoc = (m @ x >> m).to_hypergraph(), (x @ m >> m).to_hypergraph()
term = x @ u @ x @ x >> Swap(x, x) @ x @ x >> m @ x @ x >> m @ x >> m
host = term.to_hypergraph()
steps = [step.to_diagram() for step in host.normalize(unit_law, assoc)]
assert list(map(str, steps)) == ["m @ x >> m", "x @ m >> m"]
syntax_rewrite = Equation(term, *steps)
rewriter = Rewriter(host)
assert rewriter.normalize(unit_law, assoc) == 2
assert rewriter.to_hypergraph() == host.normal_form(unit_law, assoc)
```

#drawing("syntax_rewrite", caption: [
  A term of the theory of monoids and the two steps of its reduction,
  by the unit law then by associativity.
]) <fig:syntax_rewrite>

Planar diagrams are rewritten on their maps. `CMap.match` looks for the
homomorphisms of maps whose image can be cut out of the plane without
crossing a wire: the image is contracted into one box, the hole, and the
map is swept again. A match is a diagram with a hole, and there is at
most one for each homomorphism. `CMap.rewrite`, `normalize` and
`normal_form` read as those of hypergraphs: the listing of @sec:maps
unwinds the spiral of @fig:syntax_spiral in five steps. Over a spacial
category the host need not be connected, since a component that touches
nothing else can go in any face.

An e-graph holds many equal terms at once @WillseyEtAl21. An `EGraph` is
a diagram with sharing, _i.e._ an acyclic hypergraph over a Markov
category where every spider is written once, with an equivalence on its
spiders: the boxes are the e-nodes and the classes the e-classes. The
e-graphs of egg are the case where every box is a function with one
output. The e-hypergraphs of @TiurinEtAl24 have nested classes of any
arity, where these are flat with one output per class.

`merge` relates spiders, `add` adds a box that reads given spiders, or
finds the equal deterministic box that reads their classes, and `rebuild`
is the congruence closure: it relates the outputs of equal deterministic
boxes on related inputs, so that two samples of one distribution stay
apart. `matches` finds a pattern modulo the classes, `saturate` applies
rules in rounds until none changes the e-graph, and `extract` returns a
diagram of least cost, where a shared box counts once for each use and
`dag_cost` counts it once. `egraph.Builder` is the mutable version,
built as egg is, with a union-find, a hash-cons and a worklist, and its
`matches` is a join over one table per kind of box, as in relational
e-matching @ZhangEtAl22. It freezes into what
the same operations give on the pure e-graph: this is one of the 17
laws that `EGraph` states for the property matrix, and a theorem of the
Lean model (@sec:lean). The listing saturates the example of egg,
`(a * 2) / 2`, which ends in the class of its input `a`.

```python
from discopy.egraph import Markov
from discopy.markov import Box, Diagram, Ty

x = Ty('x')
one, two = Box('1', Ty(), x), Box('2', Ty(), x)
times, divide, shift = (Box(op, x @ x, x) for op in ("*", "/", "<<"))
rules = [(x @ two >> times, x @ one >> shift),
         (times @ x >> divide, x @ divide >> times),
         (Diagram.copy(x) >> divide, Diagram.discard(x) >> one),
         (x @ one >> times, Diagram.id(x))]
graph = Markov.from_diagram(x @ two @ two >> times @ x >> divide)
*_, saturated = graph.saturate(*rules)
assert len(graph.graph.boxes) == 4 and len(saturated.graph.boxes) == 8
assert saturated.extract().to_diagram() == Diagram.id(x)
double = x @ two >> times
square = Markov.from_diagram(Diagram.copy(x) >> double @ double >> times)
assert (square.cost(), square.dag_cost()) == (7, 4)
```

#gap[
  The picture of an e-graph: `EGraph.to_diagram` draws each class as a
  frame around its e-nodes, which the Typst and TikZ backends do not
  draw yet.
]
