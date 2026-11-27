#import "../template.typ": drawing, gap

= Semantics: the codomains of functors <sec:semantics>

Semantics is everything that is not a free category. A `Functor` from a
free category takes as its `cod` any Python class that implements the
abstract base class of its level: it calls the methods of the class,
`id`, `then`, `tensor`, `swap` or `trace`, and never looks at how the
arrows are stored. This section lists the codomains that the library
gained: constructions on categories, the representations of a Hopf
algebra, constraint satisfaction on top of tensors, and drawings.

== Constructions on categories <sec:constructions>

Five modules take a category and give another. The first three have one
class for each doctrine that the construction preserves.

- `op` is the opposite category: composition is reversed and the tensor
  is not, and the opposite of a Markov category is comarkov.
- `para` is the category of parametric maps over a symmetric category
  @FongEtAl19: an arrow from `x` to `y` is a parameter `p` with an arrow
  from `x @ p` to `y`, and composition tensors the parameters. An arrow
  may also carry a coparameter on its codomain @CapucciEtAl21.
- `lens` is the category of monoidal lenses, also called optics
  @Riley18: a lens from `(x, x')` to `(y, y')` is a residual `m` with a
  forward arrow from `x` to `y @ m` and a backward arrow from `m @ y'` to
  `x'`. With the input as residual it is a `get` and a `put`
  @ClarkeEtAl20, and `lens.Comarkov` is a Markov category whose copy
  merges the backward halves, as reverse derivatives do @CruttwellEtAl22.

```python
from discopy import lens
from discopy.symmetric import Ty, Box, Functor

x, y, z, p, q = map(Ty, "xyzpq")
a, b = Box('a', x, y), Box('b', y, z)
X, Y, Z = (lens.Ty(t, t) for t in (x, y, z))
F = Functor({x: X, y: Y, z: Z}, {
    a: lens.Symmetric(X, Y, Box('f', x, y @ p), Box('f*', p @ y, x), p),
    b: lens.Symmetric(Y, Z, Box('g', y, z @ q), Box('g*', q @ z, y), q)},
    cod=lens.Symmetric)
composite = F(a >> b)
assert composite.residual == p @ q
semantics_lens = composite.forward @ z >> z @ composite.backward
```

#drawing("semantics_lens", caption: [
  The composite of two lenses, its forward leg on the left and its
  backward leg on the right, with the two residuals between them.
]) <fig:semantics_lens>

Here `lens.Symmetric`, over the free symmetric category, is the codomain
of a functor: the image of a composite is a lens whose residual is the
tensor of the two (@fig:semantics_lens). The three modules have more than
four hundred cells in the property matrix of @sec:matrix.

`discopy.kleisli` is the Kleisli category of a monad on the Python
functions, _e.g._ `Maybe`, `Powerset` or `Subdistribution`, the finitely
supported probabilities of total mass at most one. It has two tensors.
With tuples it is premonoidal @PowerRobinson97: the effects of `f` run
before those of `g` in `f @ g`. With the disjoint union it is comarkov,
and traced when the monad is an
#link("https://ncatlab.org/nlab/show/Elgot+monad")[Elgot monad], _i.e._
has an iteration operator, as these three do: the trace is a loop that a
token goes around until it leaves. Below, a rejection sampler is a
diagram of the free comarkov category, evaluated by a functor. A coin is
drawn, fair or biased four to one towards heads, and flipped. On heads
the token leaves with the coin, on tails it goes back to the start.

```python
from discopy.feedback import comarkov
from discopy.kleisli.additive import Channel, Tagged
from discopy.kleisli.monad import Subdistribution

start, coin = comarkov.Ty('start'), comarkov.Ty('coin')
draw = comarkov.Box('draw', start, coin)
flip = comarkov.Box('flip', coin, coin @ start)
sampler = (comarkov.Diagram.merge(start) >> draw >> flip).trace()
M, heads = Channel[Subdistribution], {"fair": .5, "biased": .8}
F = comarkov.Functor({start: (str, ), coin: (str, )}, {
    draw: M(lambda _: frozenset((c, .5) for c in heads), (str, ), (str, )),
    flip: M(lambda c: frozenset({(Tagged(c, 0), heads[c]), (
        Tagged("", 1), 1 - heads[c])}), (str, ), (str, str))}, cod=M)
posterior = dict(F(sampler)(""))
assert round(posterior["biased"], 3) == round(.4 / .65, 3) == 0.615
```

The result is the posterior of Bayes' rule: conditioning is a trace.
`kleisli.token` builds on it the token machines of @DalLagoHoshino19
for a lambda calculus with sampling and scoring: one transition is a
channel and the machine is its trace. `discopy.monad` has the algebras of
a monad with their homomorphisms, and the free algebras among them, whose
category is isomorphic to the Kleisli category.

== Representations <sec:representations>

`discopy.hopf` is the category of representations of a finite-dimensional
Hopf algebra `H`. An `hopf.Algebra` holds its unit, counit,
multiplication, comultiplication and antipode as tensor diagrams, with an
optional R-matrix, and each axiom is an equation of diagrams, checked by
contracting both sides. The arrows of `Intertwiner[H]` are the linear
maps between modules, `Representation[H]`, that commute with the actions.
With an R-matrix and a ribbon element, which the module checks, the
category is ribbon: the braiding is the R-matrix followed by a swap, and
the twist is the trace of the braiding.

`discopy.group` supplies the algebras. A finite group is an instance of
`abc.Group`, a groupoid (`abc.Groupoid`) with one object, whose elements
are normal forms: cyclic and symmetric groups, their direct and wreath
products. An `Action` on a finite set is a functor into permutations.
`hopf.GroupAlgebra` is the algebra of a group and `hopf.Double` the
Drinfeld double of an algebra, which has an R-matrix. On the sum `V` of
two irreducible modules of the double of the group with two elements, a
ribbon functor tells the Hopf link from two circles.

```python
import numpy as np
from discopy import hopf, ribbon
from discopy.group import Cyclic

H = hopf.Double(hopf.GroupAlgebra(Cyclic[2]))
assert H.is_valid() and H.dim == 4
x, Rep = ribbon.Ty('x'), hopf.Representation[H]
V = Rep.direct_sum([Rep.anyon(0, -1), Rep.anyon(1, 1)])
F = hopf.Functor({x: V}, {}, cod=hopf.Intertwiner[H])
value = lambda link: complex(F(link).eval(dtype=complex))
circle = ribbon.Cap(x, x.r) >> ribbon.Cup(x, x.r)
link = (ribbon.Braid(x, x) >> ribbon.Braid(x, x)).trace(n=2)
assert np.allclose([value(circle @ circle), value(link)], [4, 0])
semantics_link = link.to_ribbons()
```

#drawing("semantics_link", caption: [
  The Hopf link, the trace of two braids, each strand drawn as two rails.
]) <fig:semantics_link>

#gap[
  `Intertwiner` has no strategy, hence no cell in the property matrix for
  its ribbon laws, and `discopy.hopf` cites no reference.
]

== Matrices, tensors and constraints <sec:tensors>

`Matrix` and its subclass `Tensor` gained three things beside the
pointwise order of @sec:inequations.

- *Contractors.* A tensor functor translates a diagram to a combinatorial
  map (@sec:syntax) and contracts it in one Einstein summation, run by
  its `contractor`: the `einsum` of the array library in use, or a
  `matrix.Contractor` for another engine, `OptEinsum` or `Quimb`.
- *Traces.* The tensor of `Matrix` is the direct sum, and the trace of a
  matrix with blocks `A`, `B`, `C` and `D`, the last one on the traced
  dimensions, is `A + B D* C`. Over the Booleans `D*` is the reflexive
  transitive closure. Over a field it is now the inverse of the identity
  minus `D`, the partial trace of Haghverdi and Scott @MalherbeEtAl11,
  with an error when that matrix is singular.
- *Hypergraphs.* A tensor functor contracts a `frobenius.Hypergraph` as
  it stands: one index for each spider, however many ports it has, and
  one array for each box.

`discopy.csp` is built on the third. A constraint satisfaction problem is
an arrow of the free hypergraph category on a signature: a variable is a
spider and a constraint is a box with no input, plugged on the variables
of its scope. An instance with outputs is a conjunctive query with its
free variables @ChandraMerlin77 @BonchiEtAl18 @FeliceEtAl20. A
`csp.Language` gives a size to each sort and a predicate to each symbol.
It is a functor into `Tensor` over the Booleans, _i.e._ into finite
relations, which sends an instance to the projection of its solutions on
its outputs. Over the integers it counts them.

```python
from discopy import csp, frobenius, matrix

x = frobenius.Ty('x')
edges = [("neq", (0, 1)), ("neq", (1, 2)), ("neq", (0, 2))]
triangle = csp.instance(edges, 3 * [x], outputs=(0, ))
colours = lambda n: csp.Language({x: n}, {"neq": lambda a, b: a != b})
assert colours(3).functor(int)(triangle).array.tolist() == 3 * [2]
assert not colours(2).functor(bool)(triangle).array.any()
assert colours(2).propagate(triangle) == 3 * [{0, 1}]
grid, digits = csp.sudoku(2)
assert (len(grid.spider_types), len(grid.boxes)) == (16, 12)
assert digits.functor(int, contractor=matrix.Quimb())(grid).array == 288
assert len(list(digits.solutions(grid))) == 288
semantics_triangle = triangle.to_diagram()
```

#drawing("semantics_triangle", caption: [
  The colourings of a triangle: a constraint for each edge, a spider for
  each vertex, and the first vertex as output.
]) <fig:semantics_triangle>

With three colours, each colour of the first vertex of a triangle extends
to two colourings, and with two there is none (@fig:semantics_triangle).
The four by four sudoku, 16 spiders and 12 boxes, has 288 grids: the
functor counts them on the hypergraph and `Language.solutions` finds them
by backtracking, while the diagram of the same instance fails in NumPy
with `iterator is too large`. `Language.propagate` is arc consistency:
each round keeps, for each variable, the values that every constraint
supports. It removes no value of a solution, and on a tree its fixed
point is the projection of the solutions @Freuder82. On a cycle it is
not: with two colours, propagation removes nothing from the triangle.

== Drawing is a functor <sec:drawing>

The library used to draw a diagram with a loop over its boxes and their
offsets, which computed the position of each node of a graph. Since
version 1.2 `Diagram.to_drawing` is a `Functor` like any other, whose
codomain is the class `Drawing`: a labelled graph embedded in the plane,
with a node for each box and each port and a point for each node, the
progressive plane graphs of @JoyalStreet88. A `Drawing` has its own
identity, composition and tensor, which keep the wires straight.

```python
from discopy.drawing import Drawing
from discopy.monoidal import Ty, Box

x, y, z, w = map(Ty, "xyzw")
f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, w)
draw = lambda diagram: diagram.to_drawing()
assert draw(f >> g) == draw(f) >> draw(g)
assert draw(x @ g @ w) == Drawing.id(x) @ draw(g) @ Drawing.id(w)

left, right = draw(f @ h), draw(f) @ draw(h)
assert left != right and set(left.nodes) == set(right.nodes)
assert set(left.edges) == set(right.edges)
moved = [n for n in left.nodes if left.positions[n] != right.positions[n]]
assert len(moved) == 5 and len(left.nodes) == 10 and all(
    left.positions[n] == right.positions[n].shift(y=1) for n in moved)
semantics_isotopy = left.add(right, symbol="and")
```

#drawing("semantics_isotopy", caption: [
  The drawing of a tensor, on the left, and the tensor of the drawings,
  on the right: the same graph at other points.
]) <fig:semantics_isotopy>

Two drawings are equal when they have the same nodes, the same edges and
the same positions, and the functor preserves composition and whiskering
on the nose. It does not preserve the tensor of two boxes
(@fig:semantics_isotopy): a diagram is a list of layers, so that its
tensor puts `f` above `h`, while the tensor of two drawings puts them
side by side. The two drawings have the same ten nodes and the same
edges, and five of the nodes, the box `f` with its two ports and the two
inputs, are one unit higher in the first. So drawing is a strict
premonoidal functor, and a monoidal one up to an isotopy of the plane.

A backend turns a drawing into a picture. There are three: Matplotlib,
TikZ and, new with this paper, Typst. `Diagram.to_typst` returns a syntax
tree for the part of Typst that a CeTZ canvas needs, a dozen classes in
`drawing.typst_ast`, whose `render` is the source. Every figure of this
paper is drawn by the Typst backend in the preprint and by the TikZ one
in the journal's version, from the same `Drawing`. Drawings also gained
four features. A box with `shape="triangle"` is drawn as a triangle, its
base on the side with more wires. A box is drawn as high as its `height`.
A ribbon diagram is drawn by `to_ribbons` with two rails for each wire, a
twist being a braid of the two (@fig:semantics_link). The coloured
regions of @sec:syntax are filled, with a legend on request, by the
Matplotlib backend alone, so that no figure of this paper shows them.

#gap[
  `Drawing` is not enrolled in the property matrix: it has no strategy
  and no equality up to isotopy, which would make a law of the
  measurement above. The isotopy between two drawings is not computed,
  nor compiled to an animation. Nothing reads a diagram back from a
  picture, which the theorem of @sec:lean on drawing and reading needs.
]
