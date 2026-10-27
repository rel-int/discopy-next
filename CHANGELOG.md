# Changelog

All notable changes to DisCoPy are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Changes since [`1.2.2`](https://github.com/discopy/discopy/releases/tag/1.2.2).

### Added

- `axioms.Rule` and `@rule`, inference rules stated like laws on each
  structure of `abc`, and `FreeCategory.search`, the terms of a free
  category derived from its rules, which the property matrix draws:
  `Testable.enrolled` and `Testable.matrix` say what it checks,
  `proptest/test_rules.py` types every rule and the `fast` profile checks
  each law once ([next#1]).
- `axioms.Serialisable`, the roundtrips of a term as laws, and the sorts
  `Atom`, `Perm` and `Objs` ([next#1]).
- The laws of traced and feedback categories: the vanishing, iteration,
  superposing, naturality and dinaturality of the trace, `delay_unit`,
  `delay_composition`, and `sliding` with a memory that changes type
  ([next#1]).
- The laws of monoidal, braided, symmetric, balanced, Markov and hypergraph
  categories: bifunctoriality, the hexagons, the braid, twist and yanking,
  `permutation_*`, copying and the spiders, compared up to hypergraph from
  symmetric categories on ([next#1]).
- The laws of rigid, pivotal, compact and closed categories: the snakes,
  cups and caps, transposes, adjunction, pivotality, the Reidemeister moves
  and currying, a planar level comparing them on the nose; `Axiom.canonical`
  and `Axiom.draw` state a law as a schema ([next#1]).
- `Hypergraph.strategy` and `CMap.strategy`, which sample the symmetric
  levels and above, and the laws of the conversions they rely on:
  `to_hypergraph` is a strict monoidal functor, decoding is its section,
  maps agree with hypergraphs, foliation is idempotent, and maps and
  hypergraphs read back from their diagrams and their strings; a sampled
  Markov diagram reads back from its string only up to interchange
  ([next#1]).
- Lenses, parametric maps and integer diagrams in the property matrix,
  compared up to the equation of their category by `axioms.quotient`
  ([next#1]).
- Kleisli channels, monad homomorphisms and `python.Function` in the
  property matrix, compared extensionally by `axioms.Extensional` on
  `axioms.values` ([next#1]).
- Lambda terms in the property matrix with the laws of beta reduction,
  `biclosed.Normalising`, and proof nets with `sequentialisation`
  ([next#1]).
- `discopy.abc`, the abstract base classes of the hierarchy, from `Category`
  to `PROP`, `Monoid` and `Pregroup` ([#326], [#331], [#727]).
- `discopy.axioms`, laws stated once on `abc` and checked by the Hypothesis
  matrix in `proptest/`, so far on `cat.Arrow` ([#658]).
- `discopy.cmap`, combinatorial maps for diagrams in compact categories, with
  `genus`, `is_planar` and `to_compact` ([#338], [#532], [#690]).
- `discopy.para`, parametric and coparametric maps over any symmetric
  category ([#559], [#575]).
- `discopy.hopf`, ribbon semantics in the representations of a
  finite-dimensional Hopf algebra and its Drinfeld double ([#405], [#484]).
- `grammar.abstract`, abstract categorial grammars ([#400]), and
  `categorial.Ty` with type raising and crossed composition ([#332], [#341]).
- Lambda terms in `biclosed` and `closed`, with products, `Let` and
  capture-avoiding substitution, to and from diagrams ([#322], [#332], [#400],
  [#489]).
- 2-categories, i.e. coloured regions, `None` where there is no colour, drawn
  with an opt-in legend ([#354], [#355], [#357], [#754], [#830]).
- `symmetric.Permutation` and `symmetric.Layer`, with `Swap` the two-wire
  permutation ([#362], [#594]).
- `monoidal.List`, the free monoid on a type ([#747]).
- `cat.Transformation`, natural transformations between functors ([#351]).
- `Equation` at every level, true when its terms are equal up to that level's
  axioms, e.g. `symmetric.Equation` up to hypergraph isomorphism ([#415]).
- `Sum` in `traced`, `pivotal`, `compact` and `frobenius` ([#812]).
- Ribbon diagrams drawn as dual rails ([#358], [#360]).
- Rich display of diagrams and drawings in Jupyter ([#445]).
- The constructor arguments the optyx boxes were not keeping, as public
  attributes, so that they can print themselves: `qubits.Ket.value`,
  `qubits.Bra.value`, `qubits.BitFlipError.prob`, `qubits.DephasingError.prob`,
  `photonic.Create.internal_states`, `photonic.DualRail.internal_states`,
  `photonic.PhaseShiftDR.phase`, `classical.Scalar.value`,
  `classical.PostselectBit.bits`, `classical.PostselectDigit.digits`,
  `classical.And.n`, `classical.Z.phase`, `classical.X.phase`,
  `classical.ClassicalFunction.function` and
  `classical.BitControlledPhaseShift.function` ([next#351]).
- `discopy.op`, the opposite of a category ([#646]), and `discopy.comarkov`,
  the free comarkov category, the dual of `markov` ([next#94]).
- `discopy.spacial` and `abc.SpacialCategory`, whose scalars slide,
  `s @ f == f @ s`.
- `abc.DaggerCategory`, where the dagger laws are stated ([#731]), and
  `abc.Groupoid` and `abc.Group` with `discopy.group`, finite groups and
  their actions ([next#121]).
- Plumbing: `finset.Cospan` and `frobenius.Cospan`, `markov.Function` and
  `comarkov.Cofunction` ([#472]).
- `markov.Box(is_deterministic=, is_affine=)`, flags of each box rather than
  of its class.
- `discopy.kleisli`, the Kleisli category of a monad, traced when the monad
  is Elgot ([#443]), `kleisli.token`, probabilistic token passing on lambda
  terms ([#618], [#639]), and `discopy.monad`, the algebras of a monad
  ([next#134]).
- `discopy.lens`, monoidal lenses, one class per doctrine as in `para`
  ([#701]).
- Double-pushout rewriting of hypergraphs, `Hypergraph.match`, `rewrite` and
  `normal_form`, after Bonchi et al., and of planar diagrams on `CMap`, with
  `CMap.sweep` and `CMap.match`.
- `hypergraph.Rewriter`, double-pushout rewriting in place with left-linear
  rules, each step in the size of the rule rather than of the host.
- `quantum.graphlike`, graph-like ZX diagrams as hypergraphs with one box for
  Hadamard, simplified up to a scalar by the rules of PyZX's `full_reduce` on
  a `hypergraph.Rewriter`, raced against PyZX in `benchmark/test_zx.py`
  ([next#302]).
- `discopy.ehypergraph`, e-graphs of diagrams with sharing, with congruence
  closure, rewriting and e-matching.
- `Hypergraph.pass_token`, evaluation over a disjoint-union tensor ([#363],
  [#366], [#639]), and `Hypergraph[python.Function]` with
  `hypergraph.Functor` ([#363]).
- `CMap.permutation`, a permutation as boundary wiring ([#735]), and
  `CMap.to_tree`, `Hypergraph.to_tree` and `utils.resolve_factory`.
- `biclosed.TermBase.alpha_eq` ([#692], [#767]), and beta reduction on
  `closed.Term` to a `closed.BohmTree` ([#442]).
- `discopy.neural`, neural networks as diagrams ([#702], [#736], [#743]),
  `neural.signature` ([#739]), and `neural.wiring`, `cell`, `recursion`,
  `lattice`, `act` and `train`, the networks and trainers of Tiny Recursive
  Message Passing ([Recursive_GNNs#1]).
- `neural.geometric`, the same networks run along the blocks of their
  symmetry group, reductions in place of gathers ([Recursive_GNNs#1]).
- `grammar.proofnet`, parses as proof nets read back to normal terms, and
  `grammar.neural`, a PyTorch parser into them ([lambek#8]).
- `categorial.parse`, the `k` best terms of a sentence by CKY ([lambek#2]).
- `grammar.neural.Parser.search`, parsing as search over the correct proof
  nets of the `k` best types of each word ([lambek#12]).
- `grammar.proofnet.Sequent.linkings`, the `m` best linkings by Murty's
  ranking of the `assignments` of each block, and `grammar.neural.Parser.search`'s
  `depth` and `links`, its `width` counting only the sequents with a goal, so
  that it searches past those with none and the linkings with no term
  ([lambek#12]).
- `quantum.reservoir`, quantum reservoir computing ([#519]), the `Rydberg`
  ansatz ([#586]), and `quantum.contextuality`, the contextual fraction
  ([next#283]).
- `discopy.optyx`, moved in from rel-int/optyx with its 358 commits and
  their authors: ZX, ZW and linear-optical diagrams, channels, feedback and
  a compiler to photonic hardware, under the Apache License 2.0 in its own
  `LICENSE` and `NOTICE`, behind the new `optyx` extra ([next#225],
  [next#276]).
- `discopy.optyx.sampling`: `channel.Diagram.sample` runs a diagram with
  feedback loops tick by tick, closed or fed by a `source`, its burn-in
  certified when it is passive, and `channel.Observable` is the effect of a
  normal operator or of a function of the outcomes ([optyx#76]).
- `discopy.optyx.sampling.Kernel`, one tick of an `Interferometer` in JAX
  that compiles, maps over trajectories and differentiates in its
  parameters, with `log_prob` and an exact `distribution` ([optyx#82]).
- `optyx.interaction`, combinatorial maps of recurrent channels, each box
  with a private memory and a prediction, unrolled or run to an approximate
  fixed point ([optyx#16]).
- `quantum.contextuality.Model.from_diagram`, the empirical model of an
  optyx diagram with a classical setting and outcome for each party, read
  off `optyx.contextuality.distributions` (Apache-2.0) with
  `EvalResult.prob_dist(normalise=False)`, with `Model.deterministic`,
  `Scenario.bell` taking settings for each party and `Inequality.violation`
  zero when the inequality holds, rather than raising on the trivial one
  ([optyx#87]).
- `hopf.Intertwiner.chart` and `hopf.Algebra.symmetric(n)` ([#408]).
- `copying`, a law of `Testable` beside `pickling` ([next#1]), and the
  generators of each level as typed class variables ([next#1]).
- `str_transparency`, a law of `Testable` beside `transparency`: `str(x)`
  evaluates back to `x`, each generator bound by its name ([next#163]).
- The property matrix reports the cells that cannot fail and those a
  baseline checked and this run did not ([#773], [#775]).
- `drawing.widget.DiagramWidget`, an opt-in anywidget ([#466]).
- Notebooks: *Minimality of the pure qubit ZX calculus* ([#626], [#688]),
  *Categorical deep learning* ([next#124], [next#125]), *Neural Church
  arithmetic* ([#677]), *Parsing arithmetic* and *Boolean circuits*,
  Lafont's maps, linear maps, Boolean functions and reversible circuits as
  double-pushout rewriting ([next#310], [next#316], [next#320]).

### Changed

- `NamedGeneric` takes the parameters of PEP 695, `class Matrix[dtype](...,
  NamedGeneric)`, the abstract base classes bound theirs, and the package
  typechecks with `ty`, pinned in the `dev` group and run by the `lint` job
  ([next#1]).
- `Category.parameters` is renamed `Category.typeparams` ([next#214]).
- DisCoPy requires Python 3.12 and is packaged with uv ([#311]).
- `closed` is the closed *symmetric* monoidal category; the planar one, with
  `Over` and `Under`, is `biclosed` ([#322]).
- An arrow class is its own category, so a functor takes `cod=Tensor` where
  it took `cod=Category(Dim, Tensor)` ([#326]).
- `Functor(ob=, ar=)` is `Functor(ob_map=, ar_map=)`, and `Arrow.ty_factory`
  and `Arrow.factory` are `Arrow.ob` and `Arrow.ar` ([#329], [#330], [#369]).
- `drawing.Equation` is `monoidal.Equation` ([#415]).
- Symmetric categories build their swaps with `swap_factory` ([#440]), and
  `Hypergraph[C]` replaces `hypergraph_factory` ([#379], [#532]).
- A `monoidal.Layer` is a list of boxes and types with at least one box,
  rather than an odd-length list alternating them ([#438], [#590], [#603]).
- `+` is no longer the tensor of types, use `@` ([#747]).
- `python.Function` types its wires with `List[type]` rather than tuples,
  `python.finset` and `Matrix` with `Nat` rather than `int` ([#727], [#747]).
- The quantum `SWAP` is a gate, kept by `to_tk`; the symmetry is still
  `Circuit.swap` ([#594]).
- Tensor networks are contracted in one `einsum` through `CMap`, with
  `opt_einsum` past 52 indices ([#402], [#448], [#812]).
- Drawings fit their labels ([#364], [#365]), are reproducible ([#457],
  [#469]), and as SVG adapt to dark mode ([#725]).
- Faster `Ty` construction ([#420]), `Hypergraph` equality ([#353]),
  `Hypergraph.from_diagram` ([#644]) and `quantum.Channel` ([#812]).
- `discopy.feedback` absorbs `discopy.traced`: a traced category is a
  feedback category whose delay is the identity, every `trace` also takes
  `trace(dom=, cod=, mem=)`, and yanking is a law ([next#95]).
- `markov` is Markov only, its merges are in `comarkov` ([next#94]).
- `Matrix` is a `MarkovCategory` ([#535]).
- Every level has a `Bubble`, which says its Markov flags and winds as a box
  does, and the dagger of a phased `frobenius.Spider` is marked, not negated,
  while the dagger of a ZX spider still negates its phase ([next#31]).
- `dtype` is `int` where it was `numpy.int64`, as for `float`, `complex` and
  `bool`, and `map` and `eval` infer it ([next#31], [next#88], [next#162]).
- `hopf.Intertwiner[H]` holds a tensor network `inside` between
  `Representation` ends, rather than being one, and modules have trees
  ([next#145]).
- A law is a sequent ([next#1]), and `Equation` has one symbol between each
  pair of terms ([next#1]).
- `abc.Nat` hashes and is the sort of a law's number variable `N: Nat`, and
  powers, spiders and copies take any `__index__` ([next#164]).
- Properties are read with `isinstance` rather than `hasattr`, and a
  subscripted class is named by `factory_name` ([next#1]).
- A box is plumbing by what it holds, `Layer.is_plumbing_component`, and
  `Layer.merge` fuses any plumbing.
- `to_diagram` routes the wires of each box with one permutation,
  `finset.Permutation.gather` ([#539], [#546]).
- `Hypergraph.to_diagram` and `CMap.to_diagram` share `hypergraph.Decoder`,
  output unchanged except that a box may now be a diagram ([next#56]).
- Every `then` and `tensor` takes `(self, *others)` ([#760], [#761], [#763],
  [next#60]).
- The `quantum` extra pins pyzx 0.10.7 ([next#225]).
- Faster tensor of `n` morphisms ([#623]), of Python functions, and
  `finset.Permutation.then` ([#763]).
- Faster `to_map` and `to_hypergraph` of permutations, read from their `perm`;
  `CMap.permutation` and so `to_map` take `Nat` ([next#31]).
- A functor maps a layer with one `n`-ary `tensor` rather than a fold of
  `@`, with one `tensor` call per layer ([next#161]).
- Biclosed types hold `biclosed.Wire` ([next#16]), their `**` is `<<`, and a
  term inside a diagram is drawn as one box ([next#31]).
- `Functor.id` is built from `utils.identity`, and `then` drops an identity
  on either side, so that composing functors is unital ([next#142]).
- `Box.rebuild` keeps the fields `==` compares, e.g. `tensor.Bubble.func`, a
  `tensor.Diff` in `grad`, through `dagger`, `subs` and functors ([next#118],
  [next#169]).
- `discopy.optyx` keeps no private members in `channel` and `core.diagram`:
  `decomp`, `dual_rail`, `Ob.classical`, `Ob.quantum` and `Box.array` are
  public, and the imports that make no cycle are at the top ([optyx#29]).
- The notebooks of `discopy.optyx` are marimo, rendered and checked by
  `docs/export_notebooks.py` along with DisCoPy's notebooks, four at a time
  on CI with `--jobs 4`, and the `notebooks` extra takes the pandas, seaborn
  and tqdm they import ([optyx#67]).
- `Layer >> Layer` raises, plumbing has `time_step` `None`, a permutation
  reprs with its class, nan data equals itself ([next#31], [next#168]).
- `dumps` writes tuples, numpy, sympy, `interaction`, `para`, `CQ`, `Equation`
  and dict functors ([next#168]); `para` and `interaction` hash ([next#31]).

### Deprecated

- Every `Ob` but `cat.Ob` is `Wire` ([#566]), and `PRO` is `Nat` ([#727]):
  the old names still work, with a `DeprecationWarning`.

### Removed

- Parameters nothing read: `Tree.to_diagram(contravariant)`,
  `Tree.from_nltk(lexicalised, word_types)`, `Drawing.id(length)`,
  `markov.Diagram.discard(n)` and `Backend(linewidth)` ([#768]).

### Fixed

- What the property matrix of [next#1] found, each with its test:
  `Feedback.dagger` and `rotate`, `Layer.merge` on a snake, the curry of
  `para.Compact` and of `para.Closed` with coparameters, the copies and
  discards of abstract diagrams, a layer holding `Spider(1, 1, x)`, the
  hypergraph of a traced copy of a wire whose dagger is not itself, e.g. a
  `Dims(Dim(2, 3))`, and the strings of delayed boxes, structural boxes and
  adjoint wires, which read back once their `atoms` are the generators they
  name.
- `Category.typeparams` reads `ob` with no fallback: a monoid's objects are
  `NoneType`, so `Nat.typeparams()` no longer gives `(Nat, Nat)` ([next#214]).
- `quantum.zx.Diagram.from_pyzx`, and its copy in `optyx.core.zx`, wires a
  spider with several outputs to the outputs it is joined to: it used to move
  the leg already placed at an earlier output, so the tensor differed from
  PyZX's ([next#313]).
- `quantum.graphlike.size` counts two spiders as joined when an odd number of
  Hadamards lie between them, as `to_pyzx` does: a Hadamard self-loop is a
  phase and two parallel Hadamards cancel ([next#349]).
- `optyx.core.zx.And` and `Or`: the dagger keeps the number of bits instead of
  passing the flag as it, and a daggered gate has the types it is evaluated
  with, `bit -> bit ** n` ([next#292]); `optyx.core.zx.SWAP` evaluates, its
  `truncation` no longer an unbound function set on the instance
  ([next#293]).
- `docs/notebooks/optyx/channel_error.ipynb`, a 0-byte file that is not valid
  JSON and so not a notebook, is removed: it was added empty by `0e6c24bd`
  ([optyx#136]) and never had content to restore ([next#250]).
- Every box of `discopy.optyx` prints the call that built it, so that
  `eval(repr(x)) == x` holds for it as STYLE.md asks and `axioms.transparency`
  states: the boxes of `channel`, `qubits`, `photonic` and `classical` narrow
  `__init__` and used to print `cat.Box`'s `(name, dom, cod, data=...)`, which
  their own constructor cannot take. `core.diagram.Mode` and `Bit` print their
  count. A box holding a Python function — `channel.Observable` of a
  `function`, `classical.ClassicalFunction`, `classical.BitControlledPhaseShift`
  — prints that function as Python does, so it reads back only if the
  function's representation does, as for `python.Function` ([next#351]).
- `cat.Functor` reads the image of a daggered box off `ar_map` when the
  codomain has no dagger to take, instead of daggering the image: a functor
  into `stream.Stream` or `python.Function` used to raise an `AttributeError`,
  so `double` and `unroll` stopped commuting on a loop whose doubling makes a
  daggered box ([next#236]).
- `symmetric.Functor` sends a `Swap` to a native permutation like any other
  permutation, so that it commutes with whiskering on the nose when it widens
  an atomic type ([next#301]).
- `optyx.core.zx.ZXBox` reads its equality and its hash off `cat.Box` like
  every other box, so a spider equals the one-layer diagram holding it and
  can go in a set ([next#359]).
- `closed.TermBase.normal_form` and `reduce` take a constant as the head of a
  spine, e.g. on the terms of `grammar.abstract` ([next#278]).
- `optyx.channel.Swap.dagger` is the swap the other way round ([next#290]).
- `optyx.core.zw.ZBox.conjugate` takes amplitudes given as a list, as
  `dagger` does ([next#254]).
- `optyx.classical.BitControlledPhaseShift` leaves its modes quantum, its
  codomain `qmode ** n_modes` ([next#258]).
- `optyx.core.diagram.Scalar` and `optyx.photonic.Scalar` keep any expression
  with free symbols symbolic, e.g. `x ** 2` or `x + 1` ([next#297]).
- `optyx.channel.Diagram.power_fix` contracts each depth with a fresh
  optimiser and warns at the first depth `max_chi` truncates ([next#289]).
- `python.finset.Function.then` raises `AxiomError` on functions that do not
  compose, instead of an `IndexError` or a wrong function ([next#167]).
- `optyx.channel.Sum` without terms keeps its types in `get_kraus` and
  `double`, and evaluates to the zero tensor, so the gradient by an absent
  variable doubles and evaluates ([next#291]).
- `optyx` and perceval: `PercevalBackend` evaluates a closed diagram to the
  amplitude or probability of its selections ([next#294]);
  `Diagram.from_perceval` reads a negative feed-forward offset `k` as `-k - 1`
  modes in between ([next#295]) and keeps a detector with nothing after it
  ([next#299]).
- `Hypergraph`: its hash, `simplify`, `rotate`, and `to_diagram` on
  non-self-dual, `Nat` and `Dim` types ([#324], [#387], [#393], [#622],
  [#719]).
- A box and the diagram equal to it have the same hash ([#306]).
- `frobenius.Functor` keeps the phase of a spider, e.g. ZX ([#622]).
- `trace(0)` is the identity ([#588]).
- `Bubble.dagger` ([#629]) and the drawing of bubbles ([#431], [#564]).
- `str(Ty(""))` ([#312]), and `Eval` and `Curry` read back from their `repr`
  ([#400]).
- `discopy.quantum`: `CQ.__str__`, `Measure(override_bits=True)`, `Encode`
  types, `CRz`/`CRx`/`CU1` at any distance, `Sqrt.dagger` and
  `Channel[float]` ([#812]).
- `Matrix.trace(left=True)`, `Tensor.to_quimb(dtype)` and `jacobian(params)`
  ([#768]), and spiders on PyTorch ([#584]).
- `python.additive.Function.trace` on wires of different lengths ([#561]),
  and `python.Function.tensor` is variadic ([#489]).
- `grammar.categorial.cat2ty` on brackets, features and associativity
  ([#529]).
- Drawing controlled gates ([#439]) and discards ([#514]).
- `Hypergraph`: equality on spiders with no legs, `depth`, `make_monogamous`,
  `make_bijective`, `make_causal` and the offsets of states ([next#31],
  [next#101], [next#106], [next#108], [next#187]), and its cups, caps and
  traces read its host category.
- `CMap`: `trace` checks its types, `to_diagram` places states and caps, a
  subclass reprs as itself, and maps and hypergraphs refuse sums ([next#31],
  [next#168]); `EHypergraph.rewrite` refuses what it does not have
  ([next#114]).
- The hypergraph of a rotated box is its transpose, as its map is, and both
  put the legs of a cup or cap in scan order ([next#56], [next#139]).
- `Diagram.foliation` is idempotent and an equality up to interchangers
  ([#699], [next#1]) and drops plumbing that composes to the identity;
  `braided.simplify` and `pregroup.normal_form` take foliated diagrams
  ([next#31]).
- `monoidal.normalize` and `rigid.snake_removal` say they are planar
  ([next#56] 8a) and take foliated diagrams, as do `monoidal.substitute`,
  `braided.naturality` and `Circuit.to_tn` ([next#9]).
- Boxes read back from their tree: quantum gates ([#664], [#780]), lambda
  terms ([#692]), `Trace`, `Copy`, `Merge` and `tensor.Dim` ([#736]), heads,
  tails and `FollowedBy` ([next#96]), `Twist`, `Spider` and inverse `Braid`
  ([next#1]), rotated rigid boxes ([next#113]) and `"is_dagger": false`
  ([#744]).
- `utils.dumps` writes complex data ([#775]), lists ([next#168]) and
  fractions ([next#376]), and
  `tensor` and `matrix` round-trip through JSON ([#794], [next#175]).
- `eval(repr(x)) == x` for `monoidal.List` ([#773]), phased spiders
  ([next#97]), delayed qudits ([next#130]), `Measure`, `Encode`, `CQ` and the
  rest ([next#168]), and `cfg` and `pregroup` terms ([next#31]).
- A `Wire` reads back from its `repr`, uncoloured or daggered ([next#140]),
  and a `Matrix` or a `Tensor` does up to NumPy's elision ([next#156]).
- Subscripted classes pickle with their parameter ([#749], [next#121]), and
  an unpickled `Layer` recomputes its name ([next#168]).
- Bubbles keep their `data` and `is_dagger` ([#637]), substitute in their
  args ([next#31]) and say when their args share no boundary ([next#133]).
- `subs` accepts pairs, lists and dicts like `sympy.subs`; `lambdify` keeps a
  box's winding and delay, and functors keep bubbles ([next#31], [next#118]).
- Delays: every gate ([next#130], [next#176]), closed types ([next#129]),
  `Nat` and `Dim` ([next#112]) and `FollowedBy` ([next#105]).
- Traces and feedback of compact, Frobenius, tensor, ZX and circuit diagrams
  ([next#128]), of integer diagrams and lenses ([#732], [next#95]), of
  `feedback.Diagram` ([next#1]), past the boundary ([next#31]), and
  `kleisli.additive.Channel.trace(0)` ([#578]); `interaction.Diagram.caps`
  and `interaction.Ty.unit` ([#701]).
- Functors: the identity on a `Permutation`, whiskered swaps, `ob_map` on
  `Dim` ([next#31]) and to tuples ([next#83]), braided and ribbon functors
  ([next#1]), `Tensor[complex].ar` ([next#77]) and closed constants
  ([next#177]).
- `Tensor`: the spider with no legs is the circle ([next#116]), a box is not
  its dagger ([next#119]), `trace` and `repeat` ([next#89]), n-ary `then`
  ([next#60]), `to_tn` and `cups` ([next#31]).
- Tensor functors bend a compound wire the same alone and in a diagram, and a
  cup on a self-dual object is the spider with two legs ([next#31]).
- `Matrix`: `copy` is the diagonal of the direct sum ([#652], [#795],
  [next#80]), composites keep their dtype ([#538], [#733]), `swap` on JAX
  ([next#49]), `basis` and `repr` ([next#88]).
- `python.Function` is hashable ([#743]), composes without recursion ([#760],
  [#763]), carries tuples ([next#86]) and traces ([next#73]); `copy(x, 1)`,
  and `finset.Function.swap` and `permutation` ([next#147], [next#31]).
- `discopy.quantum`: controlled gates under dagger and rotation, `to_tk`,
  `to_tn`, PennyLane adjoints and `Channel.then` types ([next#31]).
- `quantum.tk`: `to_tk` keeps bits on their wire and post-selections on their
  bit, `from_tk` reads gates on qubits and bits in any order ([next#31]).
- `zx`: `eval` is the standard interpretation ([#587], [#625]),
  `spider_factory` ([next#149]), `grad` ([next#75]), and the pyzx round trip
  keeps scalars and parallel wires ([next#74], [next#82]).
- `optyx.sampling.twomode` is unitary at any number of photons, as the
  exponential of the lift of `sampling.generator`, a logarithm of the
  two-mode unitary, where expanding the products cancelled ([optyx#84]).
- An array-backed `discopy.optyx` box reads its size off the array's shape,
  so the array can be a PyTorch tensor that requires grad ([optyx#56]).
- `hopf.Intertwiner`: `cups`, `caps` and `twist` ([next#31]).
- `hopf.Functor` sends the dagger of a twist to its inverse, `to_braided`
  takes traces and generators, and commutes with `delay` ([next#31]).
- The dagger of a rigid box keeps its winding, `Id(x).width` and five
  one-liners ([next#31]), the rotation of a daggered braid ([#606]), and the
  adjoints of a `frobenius.Wire` ([next#1]).
- `closed` and `biclosed`: exponentials of the unit, `Eval` and `Coeval`
  sides, error messages, `from_callable` ([next#1], [next#31]), equality of
  terms by their parts ([#767]) and `Copy.dagger` ([next#94]).
- `Stream.sequence`, `para.Closed.curry` and the evaluation of categorial
  terms ([next#1]); daggers of grammar rules and words, and `cfg` compares
  and hashes ([next#31]).
- Categorial `FC`, `BC`, `FX` and `BX` are boxes, so they compose and draw,
  and a nested `cfg.Tree` has a `dom`, so it applies ([next#31]).
- A `cfg.Tree` reads back from its tree ([next#168]), and an ill-typed one
  says which types do not compose ([next#31]).
- Drawing: `Drawing.dagger` ([next#1]), regions through crossings ([#614]),
  `draw` leaves a drawing as it was ([#606]), TikZ colours ([next#31]) and a
  test restyling `H` ([next#99]).
- `Drawing.permutation` permutes blocks, step-by-step `then` is a list past an
  identity, TikZ twists take xcolor names, hexcode styles drop `#` ([next#31]).
- A law reads its type parameters when another class shares its name.
- `discopy.grammar.neural` installs with the `grammar` extra ([next#52]).
- `zx.H`, quantum gates that do not send basis states to basis states,
  `gates.Bra`, `gates.Match` and `tensor.Sum` no longer claim to be cartesian
  ([next#115]).
- Docstrings of `para` and two others ([#494], [#683], [#693]), a citation
  in `kleisli.additive` ([#639]), and notebook titles in the docs ([#526]).

### Project

- `AGENTS.md`, `RULES.md` and `STYLE.md` for agents, and an LLM contribution
  needs its own handle and a human approval other than its prompter's
  ([#378], [#720]).
- Notebooks are marimo ([#404]), doc images are SVG drawing tests ([#419],
  [#470]), and `test/` mirrors `discopy/` ([#403]).
- `benchmark/` and `proptest/` suites, run on `main` and on pull requests
  labelled `benchmark` or `proptest` ([#346], [#459], [#658]).
- `pylint` fails on unused names and on a score below `fail-under` ([#768]).
- CodeRabbit reviews style, and a pull request is a draft while it has a
  `TODO.md` ([#485], [#708]).
- The notebook pages of the docs are titled by their first heading ([#825]).
- Every pull request gets a preview of the docs from Read the Docs ([#824]).
- `fail-under` rises from 8.77 to 8.85 ([#443], [#639], [#767], [next#94],
  [next#121]), and pylint reads `@axiom` with no plugin ([#770], [#776],
  [next#1]).
- The `docs` extra installs torch and jax for the notebooks ([next#201]);
  a fork's docs link their own source and canonical URL.
- `no-todo-on-main.yml`'s guard says why it fails ([next#21]).
- `CONTRIBUTING.md`'s benchmark command runs the `zx` suite, which needs PyZX,
  and `test/pivotal.py` skips its sympy test instead of failing on it, so
  `--skip-extra` holds there too ([next#323], [next#350]).
- The tests keep the `--skip-extra` contract, those that could not fail can,
  `CONTRIBUTING.md` names `setuptools`, and `build.yml` drops credentials
  ([next#31]).
- Coverage measures with `sys.monitoring` on every Python, twice as fast
  as the C tracer on 3.12 and 3.13, and the docs job has 25 minutes, the
  API pages of `discopy.optyx` taking its build to about fifteen
  ([next#276]).
- Coverage's 98% gate reads `discopy/` alone, new tests bring `ehypergraph` to
  100%, and pytest imports by path, so `test/abc.py` is collected ([next#31]).
- `ruff check` and `ty check` lint beside `pylint`, each failing on every rule
  `main` passes, the rules it breaks listed in `pyproject.toml` ([next#263]).

## [1.2.2] - 2025-12-19

See the [GitHub release](https://github.com/discopy/discopy/releases/tag/1.2.2).

## Older releases

See the [GitHub releases page](https://github.com/discopy/discopy/releases)
for the changelog of `1.2.1` and earlier.

[#306]: https://github.com/discopy/discopy/pull/306
[#311]: https://github.com/discopy/discopy/pull/311
[#312]: https://github.com/discopy/discopy/pull/312
[#322]: https://github.com/discopy/discopy/pull/322
[#324]: https://github.com/discopy/discopy/pull/324
[#326]: https://github.com/discopy/discopy/pull/326
[#329]: https://github.com/discopy/discopy/pull/329
[#330]: https://github.com/discopy/discopy/pull/330
[#331]: https://github.com/discopy/discopy/pull/331
[#332]: https://github.com/discopy/discopy/pull/332
[#338]: https://github.com/discopy/discopy/pull/338
[#341]: https://github.com/discopy/discopy/pull/341
[#346]: https://github.com/discopy/discopy/pull/346
[#351]: https://github.com/discopy/discopy/pull/351
[#353]: https://github.com/discopy/discopy/pull/353
[#354]: https://github.com/discopy/discopy/pull/354
[#355]: https://github.com/discopy/discopy/pull/355
[#357]: https://github.com/discopy/discopy/pull/357
[#358]: https://github.com/discopy/discopy/pull/358
[#360]: https://github.com/discopy/discopy/pull/360
[#362]: https://github.com/discopy/discopy/pull/362
[#363]: https://github.com/discopy/discopy/pull/363
[#364]: https://github.com/discopy/discopy/pull/364
[#365]: https://github.com/discopy/discopy/pull/365
[#366]: https://github.com/discopy/discopy/pull/366
[#369]: https://github.com/discopy/discopy/pull/369
[#378]: https://github.com/discopy/discopy/pull/378
[#379]: https://github.com/discopy/discopy/pull/379
[#387]: https://github.com/discopy/discopy/pull/387
[#393]: https://github.com/discopy/discopy/pull/393
[#400]: https://github.com/discopy/discopy/pull/400
[#402]: https://github.com/discopy/discopy/pull/402
[#403]: https://github.com/discopy/discopy/pull/403
[#404]: https://github.com/discopy/discopy/pull/404
[#405]: https://github.com/discopy/discopy/pull/405
[#408]: https://github.com/discopy/discopy/pull/408
[#415]: https://github.com/discopy/discopy/pull/415
[#419]: https://github.com/discopy/discopy/pull/419
[#420]: https://github.com/discopy/discopy/pull/420
[#431]: https://github.com/discopy/discopy/pull/431
[#438]: https://github.com/discopy/discopy/pull/438
[#439]: https://github.com/discopy/discopy/pull/439
[#440]: https://github.com/discopy/discopy/pull/440
[#442]: https://github.com/discopy/discopy/pull/442
[#443]: https://github.com/discopy/discopy/pull/443
[#445]: https://github.com/discopy/discopy/pull/445
[#448]: https://github.com/discopy/discopy/pull/448
[#457]: https://github.com/discopy/discopy/pull/457
[#459]: https://github.com/discopy/discopy/pull/459
[#466]: https://github.com/discopy/discopy/pull/466
[#469]: https://github.com/discopy/discopy/pull/469
[#470]: https://github.com/discopy/discopy/pull/470
[#472]: https://github.com/discopy/discopy/pull/472
[#484]: https://github.com/discopy/discopy/pull/484
[#485]: https://github.com/discopy/discopy/pull/485
[#489]: https://github.com/discopy/discopy/pull/489
[#494]: https://github.com/discopy/discopy/pull/494
[#514]: https://github.com/discopy/discopy/pull/514
[#519]: https://github.com/discopy/discopy/pull/519
[#526]: https://github.com/discopy/discopy/pull/526
[#529]: https://github.com/discopy/discopy/pull/529
[#532]: https://github.com/discopy/discopy/pull/532
[#535]: https://github.com/discopy/discopy/pull/535
[#538]: https://github.com/discopy/discopy/pull/538
[#539]: https://github.com/discopy/discopy/pull/539
[#546]: https://github.com/discopy/discopy/pull/546
[#559]: https://github.com/discopy/discopy/pull/559
[#561]: https://github.com/discopy/discopy/pull/561
[#564]: https://github.com/discopy/discopy/pull/564
[#566]: https://github.com/discopy/discopy/pull/566
[#575]: https://github.com/discopy/discopy/pull/575
[#578]: https://github.com/discopy/discopy/pull/578
[#584]: https://github.com/discopy/discopy/pull/584
[#586]: https://github.com/discopy/discopy/pull/586
[#587]: https://github.com/discopy/discopy/pull/587
[#588]: https://github.com/discopy/discopy/pull/588
[#590]: https://github.com/discopy/discopy/pull/590
[#594]: https://github.com/discopy/discopy/pull/594
[#603]: https://github.com/discopy/discopy/pull/603
[#606]: https://github.com/discopy/discopy/pull/606
[#614]: https://github.com/discopy/discopy/pull/614
[#618]: https://github.com/discopy/discopy/pull/618
[#622]: https://github.com/discopy/discopy/pull/622
[#623]: https://github.com/discopy/discopy/pull/623
[#625]: https://github.com/discopy/discopy/pull/625
[#626]: https://github.com/discopy/discopy/pull/626
[#629]: https://github.com/discopy/discopy/pull/629
[#637]: https://github.com/discopy/discopy/pull/637
[#639]: https://github.com/discopy/discopy/pull/639
[#644]: https://github.com/discopy/discopy/pull/644
[#646]: https://github.com/discopy/discopy/pull/646
[#652]: https://github.com/discopy/discopy/pull/652
[#658]: https://github.com/discopy/discopy/pull/658
[#664]: https://github.com/discopy/discopy/pull/664
[#677]: https://github.com/discopy/discopy/pull/677
[#683]: https://github.com/discopy/discopy/pull/683
[#688]: https://github.com/discopy/discopy/pull/688
[#690]: https://github.com/discopy/discopy/pull/690
[#692]: https://github.com/discopy/discopy/pull/692
[#693]: https://github.com/discopy/discopy/pull/693
[#699]: https://github.com/discopy/discopy/pull/699
[#701]: https://github.com/discopy/discopy/pull/701
[#702]: https://github.com/discopy/discopy/pull/702
[#708]: https://github.com/discopy/discopy/pull/708
[#719]: https://github.com/discopy/discopy/pull/719
[#720]: https://github.com/discopy/discopy/pull/720
[#725]: https://github.com/discopy/discopy/pull/725
[#727]: https://github.com/discopy/discopy/pull/727
[#731]: https://github.com/discopy/discopy/pull/731
[#732]: https://github.com/discopy/discopy/pull/732
[#733]: https://github.com/discopy/discopy/pull/733
[#735]: https://github.com/discopy/discopy/pull/735
[#736]: https://github.com/discopy/discopy/pull/736
[#739]: https://github.com/discopy/discopy/pull/739
[#743]: https://github.com/discopy/discopy/pull/743
[#744]: https://github.com/discopy/discopy/pull/744
[#747]: https://github.com/discopy/discopy/pull/747
[#749]: https://github.com/discopy/discopy/pull/749
[#754]: https://github.com/discopy/discopy/pull/754
[#760]: https://github.com/discopy/discopy/pull/760
[#761]: https://github.com/discopy/discopy/pull/761
[#763]: https://github.com/discopy/discopy/pull/763
[#767]: https://github.com/discopy/discopy/pull/767
[#768]: https://github.com/discopy/discopy/pull/768
[#770]: https://github.com/discopy/discopy/pull/770
[#773]: https://github.com/discopy/discopy/pull/773
[#775]: https://github.com/discopy/discopy/pull/775
[#776]: https://github.com/discopy/discopy/pull/776
[#780]: https://github.com/discopy/discopy/pull/780
[#794]: https://github.com/discopy/discopy/pull/794
[#795]: https://github.com/discopy/discopy/pull/795
[#812]: https://github.com/discopy/discopy/pull/812
[#824]: https://github.com/discopy/discopy/pull/824
[#825]: https://github.com/discopy/discopy/pull/825
[#830]: https://github.com/discopy/discopy/pull/830
[next#1]: https://github.com/rel-int/discopy-next/pull/1
[next#9]: https://github.com/rel-int/discopy-next/issues/9
[next#16]: https://github.com/rel-int/discopy-next/issues/16
[next#21]: https://github.com/rel-int/discopy-next/issues/21
[next#31]: https://github.com/rel-int/discopy-next/issues/31
[next#49]: https://github.com/rel-int/discopy-next/issues/49
[next#52]: https://github.com/rel-int/discopy-next/issues/52
[next#56]: https://github.com/rel-int/discopy-next/issues/56
[next#60]: https://github.com/rel-int/discopy-next/issues/60
[next#73]: https://github.com/rel-int/discopy-next/issues/73
[next#74]: https://github.com/rel-int/discopy-next/issues/74
[next#75]: https://github.com/rel-int/discopy-next/issues/75
[next#77]: https://github.com/rel-int/discopy-next/issues/77
[next#80]: https://github.com/rel-int/discopy-next/issues/80
[next#82]: https://github.com/rel-int/discopy-next/issues/82
[next#83]: https://github.com/rel-int/discopy-next/issues/83
[next#86]: https://github.com/rel-int/discopy-next/issues/86
[next#88]: https://github.com/rel-int/discopy-next/issues/88
[next#89]: https://github.com/rel-int/discopy-next/issues/89
[next#94]: https://github.com/rel-int/discopy-next/issues/94
[next#95]: https://github.com/rel-int/discopy-next/issues/95
[next#96]: https://github.com/rel-int/discopy-next/issues/96
[next#97]: https://github.com/rel-int/discopy-next/issues/97
[next#99]: https://github.com/rel-int/discopy-next/issues/99
[next#101]: https://github.com/rel-int/discopy-next/issues/101
[next#105]: https://github.com/rel-int/discopy-next/issues/105
[next#106]: https://github.com/rel-int/discopy-next/issues/106
[next#108]: https://github.com/rel-int/discopy-next/issues/108
[next#112]: https://github.com/rel-int/discopy-next/issues/112
[next#113]: https://github.com/rel-int/discopy-next/issues/113
[next#114]: https://github.com/rel-int/discopy-next/issues/114
[next#115]: https://github.com/rel-int/discopy-next/issues/115
[next#116]: https://github.com/rel-int/discopy-next/issues/116
[next#118]: https://github.com/rel-int/discopy-next/issues/118
[next#119]: https://github.com/rel-int/discopy-next/issues/119
[next#121]: https://github.com/rel-int/discopy-next/issues/121
[next#124]: https://github.com/rel-int/discopy-next/issues/124
[next#125]: https://github.com/rel-int/discopy-next/issues/125
[next#128]: https://github.com/rel-int/discopy-next/issues/128
[next#129]: https://github.com/rel-int/discopy-next/issues/129
[next#130]: https://github.com/rel-int/discopy-next/issues/130
[next#133]: https://github.com/rel-int/discopy-next/issues/133
[next#134]: https://github.com/rel-int/discopy-next/issues/134
[next#139]: https://github.com/rel-int/discopy-next/issues/139
[next#140]: https://github.com/rel-int/discopy-next/issues/140
[next#142]: https://github.com/rel-int/discopy-next/issues/142
[next#145]: https://github.com/rel-int/discopy-next/issues/145
[next#147]: https://github.com/rel-int/discopy-next/issues/147
[next#149]: https://github.com/rel-int/discopy-next/issues/149
[next#156]: https://github.com/rel-int/discopy-next/issues/156
[next#161]: https://github.com/rel-int/discopy-next/issues/161
[next#162]: https://github.com/rel-int/discopy-next/issues/162
[next#163]: https://github.com/rel-int/discopy-next/issues/163
[next#164]: https://github.com/rel-int/discopy-next/issues/164
[next#167]: https://github.com/rel-int/discopy-next/issues/167
[next#168]: https://github.com/rel-int/discopy-next/issues/168
[next#169]: https://github.com/rel-int/discopy-next/issues/169
[next#175]: https://github.com/rel-int/discopy-next/issues/175
[next#176]: https://github.com/rel-int/discopy-next/issues/176
[next#177]: https://github.com/rel-int/discopy-next/issues/177
[next#187]: https://github.com/rel-int/discopy-next/issues/187
[next#201]: https://github.com/rel-int/discopy-next/issues/201
[next#214]: https://github.com/rel-int/discopy-next/pull/214
[next#225]: https://github.com/rel-int/discopy-next/issues/225
[next#236]: https://github.com/rel-int/discopy-next/issues/236
[next#250]: https://github.com/rel-int/discopy-next/issues/250
[next#254]: https://github.com/rel-int/discopy-next/issues/254
[next#258]: https://github.com/rel-int/discopy-next/issues/258
[next#263]: https://github.com/rel-int/discopy-next/pull/263
[next#276]: https://github.com/rel-int/discopy-next/pull/276
[next#278]: https://github.com/rel-int/discopy-next/issues/278
[next#283]: https://github.com/rel-int/discopy-next/pull/283
[next#289]: https://github.com/rel-int/discopy-next/issues/289
[next#290]: https://github.com/rel-int/discopy-next/issues/290
[next#291]: https://github.com/rel-int/discopy-next/issues/291
[next#292]: https://github.com/rel-int/discopy-next/issues/292
[next#293]: https://github.com/rel-int/discopy-next/issues/293
[next#294]: https://github.com/rel-int/discopy-next/issues/294
[next#295]: https://github.com/rel-int/discopy-next/issues/295
[next#297]: https://github.com/rel-int/discopy-next/issues/297
[next#299]: https://github.com/rel-int/discopy-next/issues/299
[next#301]: https://github.com/rel-int/discopy-next/issues/301
[next#302]: https://github.com/rel-int/discopy-next/pull/302
[next#310]: https://github.com/rel-int/discopy-next/pull/310
[next#313]: https://github.com/rel-int/discopy-next/issues/313
[next#316]: https://github.com/rel-int/discopy-next/pull/316
[next#320]: https://github.com/rel-int/discopy-next/pull/320
[next#323]: https://github.com/rel-int/discopy-next/issues/323
[next#349]: https://github.com/rel-int/discopy-next/issues/349
[next#350]: https://github.com/rel-int/discopy-next/issues/350
[next#351]: https://github.com/rel-int/discopy-next/issues/351
[next#359]: https://github.com/rel-int/discopy-next/issues/359
[next#376]: https://github.com/rel-int/discopy-next/issues/376
[optyx#16]: https://github.com/rel-int/optyx/pull/16
[optyx#29]: https://github.com/rel-int/optyx/pull/29
[optyx#56]: https://github.com/rel-int/optyx/pull/56
[optyx#67]: https://github.com/rel-int/optyx/pull/67
[optyx#76]: https://github.com/rel-int/optyx/pull/76
[optyx#82]: https://github.com/rel-int/optyx/pull/82
[optyx#84]: https://github.com/rel-int/optyx/pull/84
[optyx#87]: https://github.com/rel-int/optyx/pull/87
[optyx#136]: https://github.com/rel-int/optyx/pull/136
[lambek#2]: https://github.com/rel-int/lambek/pull/2
[lambek#8]: https://github.com/rel-int/lambek/pull/8
[lambek#12]: https://github.com/rel-int/lambek/pull/12
[Recursive_GNNs#1]: https://github.com/rel-int/Recursive_GNNs/issues/1
