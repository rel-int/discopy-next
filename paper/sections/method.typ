#import "../template.typ": drawing, gap

= Method: laws as code <sec:method>

== A law is stated once <sec:laws>

The module `discopy.abc` holds one abstract base class for each structure
of the hierarchy, 28 of them, from `Category` and `MonoidalCategory` to
`CompactCategory` and `HypergraphCategory`, with `Monoid` and `Pregroup`
for the types. A class lists the operations of its structure as abstract
methods, and its laws as methods decorated with `axiom`. The two laws of a
category read as follows in the source.

```py
@axiom
def unitality[A: Obj[C0], B: Obj[C0]](
        cls, f: Hom[C1, A, B]) -> Equation[Hom[C1, A, B]]:
    """ Left and right unitality of composition. """
    return cls.equation_factory(
        cls.id(f.dom).then(f), f, f.then(cls.id(f.cod)))

@axiom
def associativity[A: Obj[C0], B: Obj[C0], C: Obj[C0], D: Obj[C0]](
        cls, f: Hom[C1, A, B], g: Hom[C1, B, C], h: Hom[C1, C, D]
) -> Equation[Hom[C1, A, D]]:
    """ Associativity of composition. """
    return cls.equation_factory(
        f.then(g).then(h), f.then(g.then(h)))
```

The signature of a law is a sequent. Its type parameters are the
variables, here the objects `A` to `D`, its parameters are the premises,
a morphism `f` from `A` to `B`, and its return annotation is the type of
the conclusion, an equation between morphisms from `A` to `D`. The body
builds the two sides with the operations of the class it is called on.
An `Equation` is a list of terms that is true when they are all equal,
and each category chooses its own `equation_factory`, that is what equal
means for its arrows.

Every class below `Category` inherits the two laws, the free category
`cat.Arrow` as well as matrices, Python functions or quantum circuits,
and no module restates them. A law can be asked for a counterexample.
On integer matrices the search for one fails, as it should.

```python
from hypothesis import settings
from hypothesis.errors import NoSuchExample
from discopy.matrix import Matrix

try:
    Matrix[int].associativity.falsify(settings=settings(max_examples=100))
    raise AssertionError
except NoSuchExample:
    pass
```

A law can also be read as a schema, with a generating object for each
variable and a box for each premise. The snake equations of a rigid
category, stated once in `abc.RigidCategory`, give @fig:snake on the free
rigid category: the figure is drawn from the statement of the law, not
from a diagram written for the paper.

```python
from discopy import rigid

snake = rigid.Diagram.snake_equations.canonical()
assert str(snake) == (
    "Equation(X @ Cap(X.r, X) >> Cup(X, X.r) @ X, Id(X), "
    "Cap(X, X.l) @ X >> X @ Cup(X.l, X))")
```

#drawing("snake", caption: [
  The snake equations, drawn from `RigidCategory.snake_equations`.
]) <fig:snake>

== The property matrix <sec:matrix>

A category enrols in the tests by inheriting `testable.Testable` and
giving a `strategy`, the generator of its own objects and arrows for the
Hypothesis library @MacIverEtAl19, in the tradition of QuickCheck
@ClaessenHughes00. A concrete category samples its values, _e.g._ small
matrices. A free category derives its terms from its boxes by inference
rules, which are stated on the abstract base classes like the laws, with
the decorator `rule` and named as in proof theory: the identity is the
rule `ax` of `Category` and composition its `cut`, the tensor is the
`mix` of `MonoidalCategory`, the cap and the cup are the `unit` and the
`counit` of `RigidCategory`. From
symmetric categories on, terms are also sampled as hypergraphs and
combinatorial maps, the representations that their equations are decided
by (@sec:syntax).

The property matrix has one cell for each law of each enrolled class. A
cell draws the arguments of the law from the strategy of the class,
within the types that the sequent asks for, and checks the equation
that the law returns.

```python
from discopy.testable import Testable

assert len(Testable.enrolled()) >= 90
assert len(Testable.matrix()) >= 2215
```

The matrix runs on every change to the library with a small seeded
budget, and nightly with a large one. Its examples are kept in a database
that the continuous integration shares, so that a counterexample found
once is replayed first everywhere.

== What a cell says <sec:declarations>

Not every law holds on every class that inherits it, and the interesting
content of the matrix is where it does not. A class declares what it
knows about a law, in one line, with one of four words.

- `failing`: the law is broken here, for a stated reason. The cell is
  then expected to fail, and the run reports it when it does not.
- `inapplicable`: the structure does not apply, and the cell is skipped.
- `modulo`: the law holds up to a quotient, given as a function to a
  normal form or to another data structure.
- `weaken`: the law holds on a subspace, _e.g._ on the diagrams of a
  smaller level of the hierarchy.

The yanking equation of a traced category shows two of these at once.
The free balanced traced diagrams of `discopy.feedback.balanced` are
lists of layers where a trace is a bubble, and yanking is false on them,
on the smallest term there is. The same law holds on symmetric diagrams
modulo their translation to hypergraphs.

```python
from discopy.feedback import balanced, symmetric
from discopy.testable import AxiomFailure

counterexample = balanced.Diagram.yanking.falsify()
assert list(counterexample) == ["x"]
try:
    balanced.Diagram.yanking(**counterexample)
    raise AssertionError
except AxiomFailure as failure:
    assert "not quotiented by the yanking axiom" in str(failure)

yanking = symmetric.Diagram.yanking(x=symmetric.Ty('a'))
assert yanking
```

#drawing("yanking", caption: [
  Yanking on a symmetric diagram: the left trace of a swap, the identity
  and its right trace, equal as hypergraphs and not as lists of layers.
]) <fig:yanking>

The library carries some 140 such declarations. They are its list of known
differences between the data structures and the mathematics, kept where
the code is, and each one is tested: when a law declared broken starts
to hold, its cell passes unexpectedly and the run says so.

A green cell says that the law held on the terms that were drawn, which
is less than the law. The matrix therefore ends every run with what it
checked. It names the cells that cannot fail, those whose strategy has a
finite support smaller than the budget, _e.g._ a law about the dimensions
zero to three that says nothing at four. Given the report of an earlier
run, it names the cells that this run did not check and that one did,
since a law skipped or declared broken is as green as one that held.

== Inequations <sec:inequations>

Some laws are inclusions. A matrix over the Booleans is a relation, and
the laws of the calculus of relations, which @sec:applications needs for
first-order logic, are stated with an order. `Matrix` has the pointwise
order, which `Tensor` inherits, and `testable.Inequation` is an equation
whose consecutive terms are compared with it, so that a law returning one
is checked by the same matrix.

```python
from discopy.matrix import Matrix
from discopy.testable import Inequation

R = Matrix[bool]([1, 1, 0, 1], 2, 2)
assert Inequation(R, R >> R.dagger() >> R)
assert not Inequation(R >> R.dagger(), Matrix[bool].id(2))
```

== Proofs where tests cannot reach <sec:lean>

A property test compares terms. It cannot say that a category is free,
which quantifies over every category that the boxes could be sent to.
For the core of the library, the directory `lean` holds a model in Lean 4
@MouraUllrich21, with no dependency, and proofs on the model.

- *Diagrams.* `Diagram(S)`, a domain and a list of layers as
  `monoidal.Diagram` encodes it, is a strict premonoidal category
  @PowerRobinson97. It is the free one: an object for each generating
  object and a morphism for each box extend to exactly one strict
  premonoidal functor (theorems `lift` and `lift_unique`). It is not
  monoidal: interchange holds for two boxes only when they are one and
  the same scalar (`not_monoidal`).
- *E-graphs.* The pure e-graph of @sec:syntax is modelled with its three
  operations. `rebuild` gives the least congruent equivalence containing
  the merges (`rebuild_least`), and the mutable builder with a
  union-find, a hash-cons and a worklist freezes into what the same
  operations give on the pure one (`building`).

A proof on a model is not a proof about the Python. The two are linked
by a differential test: Hypothesis draws cases, the Lean model answers
them as an executable oracle, and each answer is compared with the
Python's. An audit fails the build on `sorry` and on any axiom beyond
the three standard ones.

#gap[
  The same theorem for the other free data structures, `Hypergraph`,
  `CMap` and `Term`, and that drawing and reading a diagram are inverse
  to each other (@sec:semantics).
]

== Benchmarks <sec:benchmarks>

The directory `benchmark` measures what the tests do not: how the cost of
an operation grows. It reproduces the scaling experiments of
@Patterson21 for the composition of diagrams and of hypergraphs, with
the same cases on combinatorial maps, the conversions between the three,
the simplification of ZX diagrams against PyZX, and the construction of
e-graphs on recorded traces. On every change to the main branch, the
continuous integration measures the commit and the one before it on the
same machine, and reports the differences.

#gap[
  The scaling plots, which the same build can draw from a recorded run.
]
