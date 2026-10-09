# TODO

> save this branch, then in a new branch and a new claude session, extract only the part that is relevant to property testing and integrate it with a new experiment currently sitting at https://github.com/daydream6728/discopy/tree/proptest-strategies. this new approach to generation makes use of the coherence theorems to devise smarter sampler for diagrams via combinatorial representations that absorb some of the structural rules in the quotient, at the cost of relying more heavily on the cmap/hypergraph in testing. don't care about generators, don't care about typing, don't care about rule-based search.

## Base

The experiment `daydream6728/discopy@proptest-strategies` (`611bc27e`) is three
commits on top of `d9c0d664`, which is `main` of this repository (`4d960251`)
plus five upstream commits. `proptest/doctrine-search` forked from the same
`4d960251` and went its own way for 110 commits. This branch starts from the
experiment's head, keeping its history, and brings over the property-testing
layer of `proptest/doctrine-search` without its sampler, its sequents, its
generators and its typing.

## Mathematics

A law of a category `C` is an equation between two terms built from
arguments: objects, sides, counts and arrows of `C` whose boundaries are
objects built from the other arguments, e.g. `f : m ⊗ x → m ⊗ y` for the
naturality of a trace. A law is checked by sampling its arguments and
comparing its terms in the quotient `C` compares its equations in. Every
arrow is sampled by the strategy of `C` with the boundary it is asked for,
and the sampler of each level is a section of that quotient:

| level | sampled as | quotient compared in |
| --- | --- | --- |
| `monoidal`, `braided`, `balanced`, `traced`, `biclosed`, `rigid`, `pivotal`, `ribbon` | words of moves on the open wires (box, braid, twist, cup, cap, trace), put in `normal_form` where it exists | syntax, up to the normal form where the level declares it |
| `symmetric`, `compact` | a `CMap`, i.e. an involution on ports, monogamous resp. pairing adjoints, read back by `to_diagram` | the hypergraph of the diagram, up to isomorphism |
| `markov`, `closed`, `frobenius` | a `Hypergraph` with left-monogamous resp. arbitrary spiders, read back by `to_diagram` | the hypergraph, up to isomorphism |
| `feedback` | the `feedback` of an acyclic left-monogamous hypergraph from `dom ⊗ mem.delay()` to `cod ⊗ mem` | the hypergraph, feedback being a box |
| `CMap[C]`, `Hypergraph[C]` | their own wiring, as above | strict equality of maps, isomorphism of hypergraphs |

By the coherence theorems, the laws that the quotient identifies hold by
construction once the quotient map is a functor: on symmetric diagrams
(Joyal–Street) the interchange law, the naturality of the swap, the
hexagons and the involutivity of the swap, and the trace laws of a
symmetric traced category; on compact diagrams (Kelly–Laplaza) also the
snake equations and the yanking of a trace; on Markov diagrams (Fox) the
comonoid laws of copy and discard and their naturality for the swap; on
frobenius diagrams (the spider theorem) the fusion, specialness and
commutativity of spiders. The cells stating them stay, but what they check
is that the operations of the diagrams (`then`, `tensor`, `swap`, `trace`,
`cups`, `copy`, `spiders`, `dagger`) commute with the quotient map, i.e.
the implementation rather than the mathematics.

What is left to check is what the quotient does not see:

- the categorical, monoidal and structural laws on the planar levels, whose
  equality is syntactic up to their normal form, where a free braid, twist,
  trace or curry is a box and its laws fail as declared;
- the conversions the sampler now relies on: `to_hypergraph` and `to_map`
  are sections of `to_diagram` up to the quotient, and functors for
  composition and identities;
- the rewriting and drawing laws: `normal_form` and `foliation` are
  idempotent and sound, drawing is deterministic and commutes with the
  dagger;
- the serialisation roundtrips of every enrolled type: `repr`, pickle,
  deep copy and tree;
- the functor laws: functors preserve identities and composition;
- the laws of `CMap[C]` and `Hypergraph[C]` as categories in their own
  right, where a map compares the order of its boxes and a hypergraph does
  not.

Arguments are plain parameters of the law: an object, an arrow, a term, a
side `left: bool` or a count `n: int`, each drawn by the strategy of its
category, and an arrow with a boundary depending on other arguments drawn by
`arrow(dom, cod)`, which asks the strategy of the category for exactly that
boundary. `Axiom.weaken(**params)` restricts every arrow of a law to the
subspace those parameters of the strategy generate, e.g. the diagrams with
no cups and caps, with no filtering.

## Work

- [ ] `discopy/axioms.py`: `Equation` (one symbol between consecutive terms, `checked`), `Axiom` (`failing`, `inapplicable`, `modulo`, `weaken`, `strategy` of equations, `falsify`, `canonical`, `draw`), `Testable` (`strategy`, `axioms`, `enrolled`, `matrix(once)`, `subclasses`) and `Serialisable` (`repr_transparency`, `pickling`, `copying`, `serialisation`); the arguments of a law drawn from plain parameters, replacing `Grid`, `ComposablePair`, `ComposableTriple`, `Subspace`, `resolve`, `substitute` and `assert_axioms`
- [ ] the laws of `discopy/abc.py` restated with plain parameters
- [ ] the functor, conversion, rewriting and drawing laws of `cat`, `monoidal` and `symmetric`
- [ ] the `.failing`, `.inapplicable`, `.modulo` and `.weaken` declarations of each level, with the shared reasons in `discopy/messages.py`
- [ ] `proptest/`: one test over `Testable.matrix()`, the profiles `fast`, `dev`, `pr` and `explore`; `proptest/categories.py` goes
- [ ] `test/axioms.py`
- [ ] the bug fixes the laws found on `proptest/doctrine-search` that apply here: colour trees at the class default, the balanced functor on a daggered twist, `curry(0)` and the exponential of the unit, `Curry.to_drawing` with `n` wires, `Drawing.dagger` of several boxes, the tree and repr of a trace, the delay of `feedback.Trace`
- [ ] CHANGELOG, CONTRIBUTING and AGENTS
- [ ] every check, and the full `dev` profile
