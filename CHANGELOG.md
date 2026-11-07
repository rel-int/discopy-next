# Changelog

All notable changes to DisCoPy are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Changes since [`1.2.2`](https://github.com/discopy/discopy/releases/tag/1.2.2).

### Added

- `discopy.abc`, the abstract base classes of the hierarchy, from `Category`
  to `PROP`, `Monoid` and `Pregroup` ([#326], [#331], [#727]).
- `discopy.testable`, laws stated once on `abc` and checked by the Hypothesis
  matrix in `proptest/` ([#658], [#829]).
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
- The contextual fraction is piecewise linear in the table and
  `quantum.contextuality.Model.gradient` is its gradient, a subgradient at
  its kinks: the solution of the dual program, the shadow prices of the
  constraints that the solver returns with the value. `ascend` is
  gradient ascent on the fraction of any parametrised model, with a step
  that doubles after an improvement and halves otherwise since the fraction
  is piecewise linear in the table. `Model.normalise` conditions a
  postselected model on its success, when that has the same probability in
  every context, and `Model.from_diagram` groups several output wires into
  one party's outcome. `optyx.contextuality.distributions` contracts the
  diagram once with its bit inputs open rather than once per setting, and
  `optyx.contextuality.MessagePassingScenario` is the measurement scenario
  of a `CMap` for any physical system on its wires: the nodes are the
  parties, each with a family of local measurements of its prediction
  chosen by a classical wire, `local_measurement` for a qubit and
  `dual_rail_measurement` for a pair of modes by default, and `learn` finds
  the angles maximising the contextual fraction from the most contextual of
  a few random starts. A source node sending half of a fresh Bell pair down
  its edge at every tick reaches the Tsirelson bound on qubits and on
  dual-rail photons alike ([optyx#87]). The notebook
  `docs/notebooks/optyx/contextuality.md` shows the calls on a diagram, on
  a network and on two photons meeting at a beam splitter, where counting
  every event halves the fraction conditioned on a coincidence: the bunched
  events are non-contextual and the fraction of the mixture is the
  probability of a coincidence times the conditioned fraction, at every
  bias of the source.
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
- `yanking_left`, the yanking axiom on the left trace, i.e. Reidemeister move 1
  for either trace on a compact category, and `SymmetricCategory.left_trace`,
  the left trace of the python functions, the additive channels and the
  traced lenses ([next#458]).

### Changed

- `NamedGeneric` takes the parameters of PEP 695, `class Matrix[dtype](...,
  NamedGeneric)`, and the abstract base classes bound theirs; `ty` errors on
  all but the six rules main still breaks, which stay `"warn"` ([next#1]).
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
- A `Matrix` with no dtype reads it off its array without copying it, so it
  copies once, as a `Matrix[dtype]` does ([next#8]).
- `feedback.generators` is `feedback.structural_boxes`, and
  `feedback.wire_generators` `feedback.structural_wires`: a feedback, trace,
  head or tail is a structural box, not a generator ([next#459]).
- Composing a `DualRail` with any functor but the identity, on either side and
  at every level (balanced, symmetric, ribbon, compact), raises `AxiomError`
  saying to apply them in turn, `E(D(f))`, set by a new `Functor.composes`
  that is false on `DualRail`: `D >> F` used to raise an unrelated `TypeError`,
  and `F >> D` returned a functor that disagreed with `D(F(-))` on cups and
  raised on twists, braids and swaps ([next#445]).
- `monoidal.Wire` equality and hash count `is_dagger`, and a daggered wire
  prints with `[::-1]` as a daggered box does: `Ty('x')[::-1] != Ty('x')`,
  a reversed coloured wire no longer equals a fresh one with its colours
  swapped, and a wire and its dagger, which a functor may send to different
  images, have different heads and tails. Wires and types break ties in
  their order by `repr`, so of two unequal types exactly one is less than
  the other ([next#445]).
- A bubble's `data` is part of its identity, as a box's is: bubbles that
  differ only in `data` are unequal, and `str` prints the `data` so that
  `eval(str(b)) == b` still holds. As for a box, a bubble holding unhashable
  `data` is unhashable, and comparing bubbles whose `data` are arrays of more
  than one entry raises `ValueError`, except at the tensor level, where a
  bubble compares array and list `data` by their entries ([next#445]).
- `cat.Transformation.id` has the shared `utils.identity` as its components,
  and `then` drops it on either side and composes components given by a
  dictionary into a dictionary, so that the identity is a unit and composition
  is associative up to `==`, as for `cat.Functor` ([next#445]).
- `Equation` compares by value: `==` compares type, terms, symbols, space and
  `up_to`, a callable `up_to` by identity, and its hash agrees, so an equation
  is hashable when its terms are; its repr prints the symbols unless they are
  all `=`, and a `monoidal.Equation`'s space unless it is 1, so that
  `eval(repr(e)) == e` unless `e` has an `up_to` of its own ([next#445]).
- A law declared `failing` raises `AxiomFailure` only when its equation is
  false or its terms refuse to build, and returns the equation otherwise: a
  law fixed under its declaration XPASSes in the property matrix, and
  `falsify()` returns a real counterexample ([next#445]).
- A non-element on the left of a monoid's `@` raises `TypeError`, as one on
  the right does: `Monoid.whisker` takes elements only, so `'x' @ Nat(2)` and
  `None @ List[int](2)` no longer give back the element ([next#445]).
- `abc.Nat` prints as `Nat(2)` rather than `2`, as `monoidal.Nat` does, so
  `eval(str(n)) == n` holds on the nose; a `Matrix` prints its boundaries as
  `dom=Nat(2), cod=Nat(2)`, so reading its repr back needs `Nat` in scope, as
  `Matrix.environment()` gives ([next#445]).
- Delaying a `Sum` or a `Bubble` works wherever delaying a box does: delay
  distributes over a sum, so `(f + g).delay() == f.delay() + g.delay()`,
  while a delayed bubble, circuit bubble or `ehypergraph.Join` keeps its args
  and carries a `time_step` of its own on a delayed boundary, so
  `f.bubble().delay() != f.delay().bubble()` ([next#445]).
- A tensor or circuit box, and so a tensor bubble, counts its `time_step` in
  `==` and `hash`, as a feedback box does: for circuit types,
  `Box('f', x, y).delay() != Box('f', x.d, y.d)`, where they used to compare
  equal though their resets differ ([next#445]).
- `CMap.loops` are stored sorted by `repr`, so maps compare their loops as a
  multiset: `Diagram.to_map()` is the fold of `then` and `CMap.dagger` is
  contravariant on maps with loops ([next#445]).

### Deprecated

- Every `Ob` but `cat.Ob` is `Wire` ([#566]), and `PRO` is `Nat` ([#727]):
  the old names still work, with a `DeprecationWarning`.

### Removed

- Parameters nothing read: `Tree.to_diagram(contravariant)`,
  `Tree.from_nltk(lexicalised, word_types)`, `Drawing.id(length)`,
  `markov.Diagram.discard(n)` and `Backend(linewidth)` ([#768]).

### Fixed

- The documentation says what the code does: the README, CONTRIBUTING.md,
  AGENTS.md and the module summaries no longer name modules, classes or
  notebooks that are gone, list the public ones they missed, and state each
  fact in one place; the API reference gains `discopy.utils` and
  `discopy.config`, the toctree five optyx notebooks it left out.
- `docs/notebooks/optyx/photon_distinguishability.md` composes its photonic
  circuits from `optyx.photonic`, where it used to take `Create` from
  `optyx.core.zw` and `DualRail` from `optyx.core.diagram` and compose those
  core boxes with channel-level gates: `Create(1, 1, internal_states=...) >>
  BS` raised `TypeError: Expected optyx.core.diagram.Diagram, got
  optyx.photonic.BBS`, so the notebook never reached the Hong-Ou-Mandel effect
  its prose describes. `photonic`'s own module docstring states the
  composition, and `photonic.DualRail(n, internal_states=...)` is the class
  whose signature the notebook was already calling. The two cells that do
  demonstrate the core boxes keep them, module-qualified ([next#435]).
- `docs/export_notebooks.py` exempts from `--check` only the notebooks an
  **open** issue tracks. Three of its four entries named [next#247], closed as
  completed on 2026-10-26: `optyx/feed_forward_example` executes clean and is
  dropped, `optyx/photon_distinguishability` is fixed above, and
  `optyx/readme_example` fails on [next#251]'s cotengra process pool, which it
  now names. `--check` prints nothing for an exemption that has stopped being
  needed, so the list could only grow ([next#435]).
- `optyx.core.path.Matrix.prob_with_perceval` computes the Born rule of the
  amplitudes, as its docstring and `prob`'s other branch say, where it used to
  answer a probability conditional on the selections: two photons bunching
  into one arm of a balanced beam splitter read `1` instead of `0.5`.
  Perceval's own post-selection renormalises over the outcomes that survive
  it, so the selections now pick out those outcomes instead, and the
  `normalisation` that `dilate` leaves outside the unitary is put back on the
  probabilities. It read right exactly where the selections carry all of the
  mass or none of it, which both of its doctests did. As a tensor it also
  raised `IndexError` for every diagram with a selection, giving
  `amplitudes_2_tensor` one basis state per mode of the circuit where the
  matrix has one column per *selected* outcome; the codomain is now the
  occupations of the modes the selections leave open, as `eval` builds it, and
  a selection of more photons than there are raises the same
  `ValueError` as `eval` where it used to enumerate a negative photon count
  ([next#229]).
- `optyx.core.backends.PercevalBackend` evaluates a term as amplitudes and
  applies the Born rule once, to their sum, so a sum of diagrams interferes:
  `Create(1) + Create(1) @ Scalar(-1)` is zero where the probabilities of the
  two terms added to two. It also carries the two factors that a `path.Matrix`
  keeps outside its array and the simulator's circuit therefore cannot: the
  global `scalar`, so that the amplitude of `Create(1) @ Scalar(1j)` is `1j`
  and not `1`, and the `normalisation` that `dilate` raises to the number of
  photons, so that the number operator on two photons is `2` and not the unit
  a renormalised state vector reports ([next#314]).
- A selection that no photon number can satisfy is an amplitude of zero, where
  `PercevalBackend` used to raise `TypeError` from the SLOS backend for an
  effect outside the Fock space it enumerates: `Create(1) >> Select(2)` and
  `Create(1, 0) >> BS >> Select(2, 0)` are zero, as every other backend says,
  rather than a crash on `amps` and `1.0` and `0.5` on `probs` ([next#314]).
- A feed-forward component converted from perceval applies its default
  circuit **once**, when the measured state is in none of its configurations,
  where each configuration that did not match used to apply it: with a
  `PERM([1, 0])` default and two configurations, one of which matched, the
  swap was applied anyway. An even number of non-matching entries composes a
  swap default into the identity, which is why a single configuration and
  three of them — none and two non-matching entries — both read right
  ([next#314]).
- `optyx.utils.perceval_conversion.ff_configurator` gives each symbol of a
  configurator's controlled circuit the value its own name is given in the
  configuration: it used to zip `U.free_symbols`, a set, with the values of
  the configuration dict, so which symbol got which value depended on
  `PYTHONHASHSEED` and the circuit a feed-forward configuration converted to
  was silently another configuration's. A configuration that gives no value
  for a symbol now raises `ValueError` rather than pairing that symbol with
  whatever came next. The dead `use_provider_dom` flag goes with it: a
  parameter threaded through `_feedforward_common` that nothing read, its
  branch commented out, so the signature promised a provider domain the code
  never computed ([next#314]).
- `optyx`: `d ** n` is the tensor power the hierarchy already states in
  `monoidal.Ty.__pow__`, i.e. the identity for `n <= 0` on an endomorphism and
  an assertion on anything else, where three copies of the same `__pow__` —
  on `optyx.core.diagram.Box`, on `channel.Diagram` and a verbatim duplicate
  of the latter on `channel.CQMap` — recursed without a floor for every
  `n <= 0`, e.g. `zw.W(2) ** 0` and `photonic.BS ** 0`. A composite
  `optyx.core.diagram.Diagram` has a power too, where only its boxes had one
- `optyx.photonic.Select()` and `Create()` default to one photon on one mode,
  as the `zw` boxes they wrap do, so that their `photons` and the path matrix
  they build agree with the `qmode` they are typed on ([next#314]).
- `optyx.classical.PostselectDigit()` and `Digit()` declare the one mode
  their `zw` kraus defaults to rather than none, where they used to fail an
  assertion ([next#314]).
- `optyx.classical.PostselectBit()` on no bit is the empty product, the
  scalar one on `bit ** 0`, where reading the first bit raised an
  `IndexError` ([next#314]).
- `optyx.qubits.Z(0, 0).decomp()` is a scalar, where it dropped the empty
  codomain and gave a state on one qubit, the legs in and the legs out now
  read independently; so is `qubits.X(0, 0).decomp()`, which goes through it
  ([next#314]).
- `optyx.core.control.ControlledPhaseShift` applies the phase its function
  reads off two or more control modes: it indexed its array along the first
  axis only, with `array[i, :]` for a whole control configuration `i`, so the
  per-configuration cell had the wrong shape and the reshape raised
  ([next#314]).
- `optyx.core.control.ControlledPhaseShift.conjugate` negates its phases: it
  returned a box built from the arguments of `self`, so the conjugate *was*
  `self` and its truncation equalled the original's. Complex conjugation sends
  `exp(2*pi*1j*y)` to `exp(-2*pi*1j*y)`, so the conjugate is the same box with
  every value of its function negated, and `conjugate` now commutes with
  `dagger` and is an involution. The negation is `partial(negate_phases,
  function)`, a function of the module rather than a closure, so a conjugated
  box pickles whenever the box it is the conjugate of does ([next#314]).
- `optyx.core.zw.ZBox` compares two amplitude lists of different lengths
  instead of raising on the broadcast, with
  `misc.compare_arrays_of_different_sizes`, which is what equality of two
  truncations of the same infinite list is in this module ([next#314]).
- `optyx.utils.misc.compare_arrays_of_different_sizes` counts a NaN as a
  difference, as `np.allclose` does with its default `equal_nan=False`: it
  read `abs(x - y) > tol`, False whenever the difference is a NaN, so a NaN
  amplitude compared equal to anything with a matching prefix and an
  `optyx.core.zw` axiom stated with the helper could not fail on one. It is
  `np.allclose` with an absolute tolerance only instead, so two matching
  infinities stay equal, which makes the `zw.has_nan` guard on `ZBox.__eq__`
  redundant and both go ([next#422]).
- `optyx.core.zw.W` has a `__hash__` of its own: Python used to set it to
  `None`, as it does on any class overriding `__eq__`, so `hash(W(2))` raised
  and a `Functor` could not look the box up in a dict `ar_map`. `ZBox` stays
  unhashable: comparing amplitudes by prefix is not transitive, so no hash
  keeps equal boxes hashing equal ([next#314]).
- Docs: the architecture diagram shows the new syntax modules, the menu names
  `spacial`, `kleisli` and `monad` without their prefix, `feedback` states each
  law once with its trace under it, and the notebooks link the API docs
  relatively ([next#411]).
- `optyx.compiler`: a `Fusion` compared to anything but a fusion is unequal
  rather than an `AttributeError`; `path_cover.find_min_path_cover` covers a
  graph with one node, an edgeless graph and the empty graph, searching up to
  as many paths as the graph has nodes rather than one fewer; and the SEMM
  decompiler refuses three invalid instruction streams it used to accept — an
  instruction before the first `next_node`, a node measured twice when a delay
  records one of the two in the future, and a delayed fusion no photon comes
  back for ([next#314]).
- `optyx.compiler`'s summary renders `compile_to_semm`, `decompile_from_semm`
  and `pattern_satisfies_order` as the functions they are, with their own
  autosummary block, where `class.rst` used to document all three as classes;
  so do the summaries of `compiler.semm`, which listed `compile_to_semm` and
  `decompile_from_semm` the same way, and `compiler.semm_decompiler`, which
  listed `decompile_to_fusion_network` ([next#314]).
- Every code block of `discopy/optyx/README.md` runs, and `test/optyx/readme.py`
  checks that it does, the way `test/readme.py` checks the root README: the
  snippets import `Channel`, `qubits.H` and `photonic.FusionTypeII`, call
  `classical.Add` where `AddN` does not exist, call `Ket` as they import it,
  define the `state_array` they send through the perceval processor, read
  graphix's statevector off the backend as graphix 0.4 leaves it, and give
  `np.isclose` and `np.allclose` no third positional argument, so that the
  teleportation and distinguishability assertions check what the prose around
  them claims: `np.isclose` was reading a `3` as a 300% relative tolerance, and
  `np.allclose` a doubled identity channel as one ([next#314]).
- What the property matrix of [next#1] found, each with its test:
  `Feedback.dagger` and `rotate` raise `AxiomError`, as either would feed
  the memory back in time; `para.Compact.curry` keeps the coparameters and
  `para.Closed.curry` raises `ValueError` on them; a layer drops a spider
  with one leg on each side, e.g. `Spider(1, 1, x)`, for its type; abstract
  diagrams copy and discard with a `Function` of their own; the hypergraph of
  a traced copy of a wire whose dagger is not itself, e.g. a
  `Dims(Dim(2, 3))`, is built; and the strings of delayed boxes, structural
  boxes and adjoint wires read back, their `atoms` being the generators they
  name.
- `optyx.photonic.TBS` and `MZI` are the matrix they say they are once
  conjugated: the global phase **XORs** `is_gate_dagger` and `is_conj` where it
  ORed them, so a conjugated gate carried the daggered gate's phase and
  `g >> g.dagger()` was not the identity; and `decomp` conjugates the beam
  splitters and phase shifters it decomposes into, where it dropped `is_conj`
  altogether, so the gradient of a conjugate — which is the gradient of the
  decomposition — was the gradient ([next#314]).
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
- `docs/notebooks/optyx/zw_examples.md` runs on the current `photonic` API
  instead of the removed `qpath` and `circuit` modules (part of [next#249]).
- Every box of `discopy.optyx` prints the call that built it, so that
  `eval(repr(x)) == x` holds for it as STYLE.md asks and `testable.transparency`
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
- `optyx.channel.Diagram.eval` evaluates a diagram of classical boxes, which
  `is_pure` called pure though a `CQMap` has no Kraus map ([next#260]).
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
- `optyx.core.path.Matrix`: `dagger` conjugates `scalar`, `dilate` keeps
  `scalar` and multiplies `normalisation` in, and `eval` with no photons to
  route reads the empty permanent, 1, as `npperm` now does;
  `optyx.utils.misc.matrix_to_zw` takes a rectangular matrix, a symbolic one
  with zeros and one with a zero row or column, and
  `basis_vector_from_kets` rejects a negative index (part of [next#314]).
- `optyx.classical.Xor(n)` is a parity for every `n`, its scalar
  `sqrt(2 ** (n - 1))`, and `BitControlledGate` of a diagram with no boxes
  discards its bit (part of [next#314]).
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
- `markov.Box` and boxes inheriting its constructor show `name`, `dom`, `cod`
  and their keywords in their signature, not `*args` ([next#114]).
- optyx: `photonic.Swap(left, right)` swaps any numbers of modes
  ([next#300]), `prob_dist` of a closed diagram is the probability of the
  empty outcome ([next#382]), `classical.And(n)` takes `n` bits
  ([next#387]), and `BBS.dagger` and `BBS.lambdify` keep `is_conj`
  ([next#386]).
- `Hypergraph.is_convex` stops at the last box that writes into the image
  instead of walking every box downstream of the match, wherever the boxes of
  the host are topologically ordered, which `is_topologically_ordered` now
  caches and a `Rewriter` declares false: matching a pattern over a host used
  to cost the size of the host on every homomorphism ([next#286]).
- The SEMM compiler of `discopy.optyx` on a path cover of more than one path,
  which used to recurse forever: `compiler.mbqc.add_fusion_order` closes the
  past of a node without recursion, so a node fused to something in its own
  past comes back; `GFlow.partial_order` leaves two nodes of one layer
  incomparable, as a foliation does, rather than each in the other's past;
  `compile_to_semm` extends the order to the breaks that join the paths,
  which are nodes of the resource state and not of the graph; and
  `get_creation_times` and `get_measurement_times` are keyed by node ID
  rather than by position in the path, counting no measurement photon for an
  output, so that the fusion photons pair up. What the instructions should
  say about a break, so that a decompile reads the graph back, is open
  ([next#298]).
- A doctest image of another size than its baseline raises "Drawing differs"
  like any other mismatch ([next#399]).
- Docstrings of `para` and two others ([#494], [#683], [#693]), a citation
  in `kleisli.additive` ([#639]), and notebook titles in the docs ([#526]).
- optyx's docstrings draw with `doctest=`, so their images are checked against
  the committed ones ([next#388]).
- `optyx.core.zx.Z(0, 0, phase)` is the scalar `1 + exp(2i pi phase)`, the sum
  of its amplitudes, which is what a spider with no index left to carry them
  is and what pyzx gives: it used to be a legless `zw.ZBox` carrying both
  amplitudes, so `qubits.Z(0, 0)`, `qubits.X(0, 0)` and `Z(0, 0, .25)` raised
  `cannot reshape array of size 2 into shape ()` rather than evaluating
  ([next#421]).
- `optyx.core.zx.Spider.grad` is `1j * pi * phase'` times the difference
  between the spider and the spider shifted by half a turn, which is the
  formula `quantum.zx.Spider.grad` next door already had. The optyx copy
  returned the shifted spider alone, so a spider with legs had a spurious
  value on the all-zero entry of its gradient, where the derivative has none,
  and a legless one had the wrong value outright. `optyx.core.zx.ZBox`
  overrides it: its parameter is an amplitude rather than an angle, its array
  is linear in it, and its gradient is the difference between the box at one
  and the box at zero ([next#426]).
- `optyx.core.zw.ZBox(0, 0, amplitudes)` reads the first of its amplitudes,
  as its truncation already did for an infinite amplitude list and as `zw`'s
  own axioms write a scalar: the list branch was missing that index, and a
  legless box carrying more than one amplitude raised on the reshape
  ([next#421]).
- `foliation` and `depth` work again on coloured monoidal and braided diagrams,
  and on non-symmetric diagrams that hold a `Sum`: the hypergraph is now built
  only for symmetric diagrams ([next#445]).
- Composing a zero (an empty `Sum`) with an arrow it does not compose with now
  raises `AxiomError`, on either side, rather than returning an ill-typed zero
  ([next#445]).
- `dumps` refuses box data holding a dict whose keys are not strings, with its
  usual 'not JSON serialisable' `TypeError`, instead of silently writing int,
  float, bool and None keys as strings that `loads` cannot read back
  ([next#445]).
- `F >> G` now agrees with `G(F(t))` when `F`'s maps use the sugar its call
  accepts (a name for an object, a tuple for an arrow), rather than crashing
  when composed or applied, and two `quantum.circuit.Functor`s compose
  ([next#445]).
- `subs` on a box, a bubble or a diagram substitutes a key that is not a
  symbol, e.g. `sin(phi)` or `phi + psi`, as `sympy.subs` does, instead of
  leaving the box unchanged ([next#445]).
- An arrow, a diagram, a type or a `Dim` built from an iterator is checked and
  stored as if built from its tuple, instead of skipping the composability
  checks or coming out empty ([next#445]).
- `Sum` keeps its terms as a tuple, so `Sum([f, g]) == Sum((f, g)) == f + g`
  and is hashable, and a zero gradient `Sum([], dom, cod)` equals
  `Sum((), dom, cod)` ([next#445]).
- `Box.free_symbols` follows an in-place change to the box's data, so `subs`
  and `lambdify` no longer return the box unchanged after a symbol is added to
  mutable data ([next#445]).
- `Box.free_symbols` reads a numeric array's dtype instead of visiting every
  entry, so a no-op `subs` on large tensor boxes takes milliseconds instead of
  seconds ([next#445]).
- `Diagram.normal_form` and `normalize` read the diagram's boxes and offsets
  once per sweep, and `Diagram.boxes` is linear, so normalising an
  already-normal 800-box diagram takes about 1ms instead of 1.4s ([next#445]).
- `Diagram.merge_layers`, and so `foliation`, resumes at the pair above each
  merge instead of starting again from the first pair, with the same result
  ([next#445]).
- `Sum.generator` is a property, as on every other arrow: a singleton sum's
  generator is its only term, and any other sum's is `None` ([next#445]).
- `str` of a bubble prints a non-default `name` and `method`, as its `repr`
  does, so `eval(str(b)) == b` holds for a named bubble ([next#445]).
- `Nat(-1)` raises a `ValueError` instead of building a natural number that
  cancels `Nat(1)`, and `Dim(0)`'s `ValueError` names the dimension
  ([next#445]).
- `monoidal.Ty(Colour('red'))` raises a `TypeError` that names both classes, as
  every other level does, instead of an `AxiomError` printing raw class reprs
  ([next#445]).
- The default `op.Comarkov` and `op.Markov`, over `python.finset.Function`,
  read back from their repr in their own `environment()` ([next#445]).
- `Axiom.falsify` returns the shrunk arguments a law refuses to build its terms
  on, instead of raising the law's error ([next#445]).
- `Equation.modulo` keeps every attribute of the equation but `up_to`, e.g. the
  `space` of a `monoidal.Equation` ([next#445]).
- `str` of an `AxiomFailure` is the reason the broken law declares, no longer
  the repr of the reason and the whole equation together ([next#445]).
- `Hypergraph.make_bijective`, and so `to_diagram` over `compact`, raise
  `AxiomError` (`NOT_FROBENIUS`) on a spider of three or more legs over a
  category with no spiders, where they raised `AttributeError`; the unused
  messages `MATRIX_TWO_DTYPES`, `HAS_NO_ATTRIBUTE` and
  `COMPLEX_TYPE_HAS_NO_ATTR` are gone ([next#445]).
- A delayed box keeps every attribute it was built with: a markov sample stays
  a sample, and so evaluates under a stream functor, and in frobenius
  `f.r.delay() == f.delay().r` ([next#445]).
- Rotating a ribbon twist, or a dual rail braid, twist, cup or cap, gives a box
  on the adjoints, so `.r`, `.l` and `to_ribbons().r` of a ribbon diagram are
  well-typed ([next#445]).
- `Diagram.naturality` raises `AxiomError` unless the layers next to the box
  are the braid it slides through, instead of silently deleting whatever box
  was there ([next#445]).
- A Markov (comarkov) functor into a category with native functions sends a
  `Function` (`Cofunction`) to one, so `Functor.id` is the identity on it
  ([next#445]).
- `to_ribbons` maps a hand-built `Trace`, `Head` or `Tail` to the trace, head
  or tail of its argument's dual rail, and a delayed box to the delay of its
  dual rail, instead of one opaque box ([next#445]).
- `Permutation`, `Function` and `Cofunction` take numpy integer entries, the
  identity included, like copies and merges do ([next#445]).
- `eval(repr(x)) == x` for `ribbon.DualRailCup` and `DualRailCap`, daggered or
  delayed ([next#445]).
- `Diagram.cup(i, i)` in ribbon and compact diagrams raises a `ValueError`
  instead of cupping the wrong wires ([next#445]).
- Docstrings in `symmetric`, `balanced` and `ribbon` match the code: the trace
  axioms of symmetric diagrams hold up to hypergraph, not on the nose, `Twist`
  documents `dom` and `is_dagger`, and `ribbon.Diagram.cup`, `trace` and
  `symmetric.Diagram.permutation` name their real parameters ([next#445]).
- `Swap(x @ y, z)` says a swap takes atomic types instead of reporting a
  permutation of length 2, and `Swap` with a non-swap permutation and
  `finset.Function` (so `markov.Function`) with an out-of-range or wrong-length
  list raise a `ValueError` that says what is wrong ([next#445]).
- A `biclosed` or `closed` functor into rigid, pivotal, pregroup or tensor
  diagrams evaluates the terms it meets instead of raising 'Expected a term',
  so `to_rigid` and categorial `to_pregroup` of a diagram with a term work
  again ([next#445]).
- A closed functor maps the unitype `U` as a generator, so untyped terms can be
  evaluated, drawn and turned into maps instead of recursing forever
  ([next#445]).
- `to_rigid` is the identity on rigid diagrams: it keeps winding numbers, cups
  and caps instead of dropping them ([next#445]).
- `to_closed` and `closed.TermBase.from_biclosed` keep the data of each
  constant ([next#445]).
- `eval(repr(c)) == c` for a biclosed or closed `Constant` with data, and for
  the terms built on one ([next#445]).
- An explicit `Curry(f, 0)` goes through biclosed and closed functors,
  `to_map`, closed `to_compact` and delayed `to_drawing` as its argument `f`,
  where it raised a bare `AssertionError` ([next#445]).
- Products and exponentials print their name only when it is first read, so
  nested `Tuple`s and exponentials build in linear time, and a thousand nested
  `Tuple`s no longer hit the recursion limit ([next#445]).
- `Hypergraph.to_graph` attaches each input and output to the spider it is
  wired to, so `==` sees where an adjoint boundary attaches
  (`f @ g @ Cap(x, x.r)` no longer equals `g @ f @ Cap(x, x.r)` as compact
  hypergraphs), `hash` no longer raises `KeyError` and
  `from_graph(to_graph(h))` gains no spider ([next#445]).
- Equal hypergraphs hash equal when their boxes are equal but print
  differently (`data=1` and `data=1.0`): the fast path of
  `Hypergraph.__hash__` hashes the boxes, not their `repr`, so a box with
  unhashable data makes its hypergraph unhashable, as it does its diagram
  ([next#445]).
- `frobenius.Diagram.unfuse` unfuses the spiders a `Cospan` holds, so
  `(Spider(1, 4, x) @ y).unfuse()` and `Diagram.spiders(1, 4, x @ y).unfuse()`
  have no point with more than three legs, as upstream ([next#445]).
- `Hypergraph.from_callable` restores its category's own `__call__` and no
  longer deletes it, so `Hypergraph[python.Function].from_callable` leaves
  `Function` callable and a `from_callable` nested in another works
  ([next#445]).
- `Hypergraph.__repr__` writes `offsets` when any is set, as `to_tree` does,
  so a planar hypergraph read back from its repr decodes without swaps
  ([next#445]).
- `Hypergraph.tensor` shifts each factor's offsets past the codomains of the
  factors before it, so `x @ s.to_hypergraph()` decodes in a category without
  swaps, like `(x @ s).to_hypergraph()` ([next#445]).
- `Hypergraph.permutation` moves each type of a list of compound types as a
  block, as `symmetric.Diagram.permutation` does, instead of raising or
  wiring it as if each type were one wire ([next#445]).
- `Hypergraph.simplify` tries each pair of boxes once, not every ordered
  pair, and its docstring gives its real cost, cubic per pass, instead of
  quadratic ([next#445]).
- A legacy pickled `frobenius.Spider` loads with its legs in the right order
  in its name and with its `typ` set, so its `repr` no longer raises
  ([next#445]).
- The docstring of `Hypergraph.is_monogamous` counts the wires correctly and
  says what each spider must satisfy ([next#445]).
- `Hypergraph` spiders with no port can take labels of any type: two of them
  with incomparable labels, e.g. `'a'` and `0`, no longer raise `TypeError`
  ([next#445]).
- `CMap.copy`, `merge`, `discard`, `noise` and `spiders` are the maps of the
  host category's own: `copy(x, 1)` is the identity, a composite type gets
  one box per atomic object, and Markov maps no longer gain identity boxes
  over round trips ([next#445]).
- `CMap.match` sends a pattern's straight wires only to the wires
  `CMap.beside` its image, so matching a box with a few wires beside it no
  longer takes time exponential in their number (about 35 s down to 0.2 s at
  five wires), with the same matches ([next#445]).
- A cup on two outputs of a state, e.g.
  `Box('s', Ty(), x @ x.r) >> Cup(x, x.r)`, decodes from its `CMap` with no
  swap ([next#445]).
- `group.Cyclic`, `Symmetric`, `Product` and `Wreath` compute each action
  once and return the same `Action` every time, so a wreath whose top is
  itself a wreath costs about as much as its factors (one multiply at nesting
  4 went from about 1 s to 0.3 ms). `Action` is a frozen dataclass, so the
  action a group shares cannot be changed under its other callers ([next#445]).
- `CMap.eval` on a map with no boxes (an identity, cup, cap or loop) uses its
  category's dtype, e.g. `tensor.Diagram[complex]`, instead of falling back
  to float, so it agrees with `Diagram.eval` ([next#445]).
- In rigid and pivotal maps, a cup wired between two outputs of one box reads
  them left to right, as `Hypergraph` does: rigid accepts the map of a cup on
  a box onto `x @ x.r` and refuses one onto `x.r @ x`, and pivotal decodes
  both ([next#445]).
- `tensor.Box` compares its winding number `z`, so on a self-dual `Dim` a
  rotated box `f.r` no longer equals or hashes like `f` ([next#445]).
- `tensor.Functor` evaluates `dtype=Expr` diagrams exactly and functorially:
  its boundary and loop identities are integers, not floats, so
  `F(f >> f) == F(f) >> F(f)` ([next#445]).
- `hopf.Representation.qdim` is the loop in `Intertwiner[H]`, twisted by the
  pivotal element, rather than `dim V`: 0, not 4, on Sweedler's regular module
  ([next#445]).
- `Matrix.transpose`, `conjugate`, `dagger`, `merge` and `round`, and
  `Tensor.transpose`, `.l`, `.r` and `round`, work under `backend('pytorch')`
  ([next#445]).
- `tensor.Bubble.grad` applies the chain rule on any boundary (states, effects,
  compound `Dim`s), where it used to raise `ValueError` ([next#445]).
- Equal `hopf.Algebra`s hash equal, so `Representation[H]` is one class per
  algebra up to equality and `Intertwiner` accepts a module over an equal
  algebra ([next#445]).
- `tensor.Diagram.to_tn` evaluates the spider with no legs to its dimension,
  the circle, as `eval` does, not to 1 ([next#445]).
- `Matrix.repeat`, and with it Boolean `Matrix.trace` and `Tensor.repeat`,
  takes ceil(log2 n) products instead of n(n+1)/2: milliseconds instead of
  seconds at n = 100 ([next#445]).
- `Matrix.is_close` gives its real `rtol` default (`1.e-8`), and
  `tensor.Diagram.jacobian` says it returns a `Sum` diagram whose `eval` is
  `Tensor.jacobian` ([next#445]).
- `Stream.delay` works on streams whose memory starts non-empty, e.g.
  `Stream.sequence(name, x, y, m)` and the tail of a feedback, instead of
  raising `AxiomError` ([next#445]).
- `feedback.Functor` sends a daggered `FollowedBy` to the dagger of its image,
  so even the identity functor now keeps its boundary ([next#445]).
- The `discopy.stream` docs check the associativity of composition for real: it
  holds as hypergraphs, i.e. up to the axioms of symmetric categories, not on
  the nose as they claimed ([next#445]).
- `feedback.Diagram.feedback` and `trace` raise `AxiomError` on a wrong `dom`
  or `cod` in the keyword form when the memory is not delayed, instead of
  silently using the diagram's own boundary ([next#445]).
- `kleisli.additive.unpack_value` and `kleisli.multiplicative.unpack_value`
  take the packed type, so a channel on a single wire passes a `Tagged` or
  `Row` value through whole instead of stripping, re-tagging or splatting it
  ([next#445]).
- Kleisli channels of different monads raise `TypeError` when composed with
  `>>` or tensored with `@`, instead of composing silently or failing only at
  call time ([next#445]).
- `stream.Ty.sequence('mem')` is one wire `mem0`, `mem1`, ... per step instead
  of splitting the name into its characters ([next#445]).
- `para.Closed.curry` raises `ValueError` ("Cannot split n atoms off dom") for
  `n < 0` or `n > len(dom)`, as `para.Compact.curry` and `para.Traced.trace`
  do, instead of currying into the parameter wires and failing with an
  `AxiomError` about an internal composite ([next#445]).
- optyx's `PercevalBackend` labels each outcome by the output modes alone, so a
  diagram with a `Select` or a non-unitary part no longer puts its amplitudes
  on the wrong outcomes once more than one outcome is non-zero ([next#445]).
- optyx's `QuimbBackend` with a hyperoptimiser keeps the result's axes in wire
  order, where it used to permute them once a result had eleven or more input
  or output wires ([next#445]).
- `optyx.classical.BitControlledPhaseShift` is controlled by its classical
  `mode` wires, where the photon number on the `qmode` it shifts used to drive
  the phase ([next#445]).
- optyx's `PermanentBackend` evaluates a sum of diagrams whose terms carry
  different numbers of photons, padding each term's amplitudes with zeros,
  where it used to add them by broadcasting or raise ([next#445]).
- `optyx.core.zx.ZBox` exports to PyZX as a `Z_BOX` vertex labelled with its
  amplitude, and `from_pyzx` reads that vertex back as a `ZBox`, where
  `to_pyzx` made it an X spider of phase twice the amplitude, so that
  `ZBox(1, 1, .5)` became a NOT gate ([next#445]).
- `optyx.core.path.Matrix` compares equal only when its creations, selections,
  normalisation and scalar agree too: matrices that evaluate differently used
  to compare equal, and matrices with different numbers of creations were
  compared by numpy broadcasting or raised `ValueError` ([next#445]).
- An `optyx.core.zw.ZBox` with no inputs keeps every amplitude of a finite
  list: its outputs have dimension `max(2, len(amplitudes))` instead of 2, so
  `ZBox(0, 1, [1, 2, 3]) >> Select(2)` is 3, as its transpose gives
  ([next#445]).
- `matrix.Matrix.permutation` and `Matrix.copy` hold integers again, so a
  `Matrix[Expr]`, and any symbolic optyx path matrix with a swap, creation or
  selection, composes instead of raising `TypeError`; optyx's
  `Permutation.to_path` and `Swap.to_path` return the dtype they are asked for
  ([next#445]).
- `optyx.core.diagram.Scalar(s).inflate(d)` is `Scalar(s)` rather than
  `s ** d`, and a `Sum` inflates term by term, so a core diagram holding a
  scalar or a sum keeps its amplitude when inflated ([next#445]).
- `optyx.core.zx.ZBox(0, 0, w)` evaluates to `1 + w`, the sum of its
  amplitudes, as pyzx and spider fusion give, where it used to be `1`
  ([next#445]).
- Evaluating a diagram no longer changes the `photon_preservation_behaviour` of
  the `optyx.core.control.BitControlledBox` boxes in it ([next#445]).
- `optyx.qubits.Circuit` converts discopy's `CRx`, `CU1` and controlled gates
  at any distance to the right unitary, converts `SWAP` and `Controlled(U1)`
  instead of raising `KeyError`, and raises `NotImplementedError` on a mixed
  scalar instead of squaring it ([next#445]).
- `optyx.channel.CQMap.inflate`, and with it `Observable`, puts each `qmode`'s
  inflated copies in the order of the inflated type, so an inflated `CQMap` on
  qmodes no longer evaluates to zero or to complex probabilities ([next#445]).
- `optyx.classical.Sub` subtracts the first number from the second, as its
  docstring says, where it computed the first minus the second ([next#445]).
- `grad` on an `optyx.channel` box that has free symbols and no gradient of its
  own (`qubits.Z`, `X`, `Scalar`, `photonic.BBS`, `Gate`, a `CQMap` or a
  feedback loop) raises `NotImplementedError` instead of recursing forever, and
  still gives zero for a variable the box does not depend on; the
  `free_symbols` of a `Channel` include those of its Kraus map and of a `CQMap`
  those of its density matrix, so a box built from a symbolic diagram raises
  too, where it gave a false zero ([next#445]).
- `subs` and `lambdify` substitute the parameter of optyx's `qubits.Scalar`,
  `photonic.Gate`, `PhotonLoss`, `PhaseShiftDR` and `classical.Z` and `X`,
  which report it in `free_symbols`: they used to leave the symbol in place,
  and their gradient was silently zero. A `Gate` with a symbolic matrix and no
  `data` is substituted too ([next#445]).
- `optyx.channel.Diagram.fix` warns that the result is truncated to `max_chi`
  only when its backend compresses: `DiscopyBackend` and a `QuimbBackend`
  without a compressed optimiser return the exact state without the false
  warning ([next#445]).
- The teleportation example of `optyx.qubits` is the standard circuit, a
  normalised Bell pair, a CNOT and a Hadamard on its control, and its doctest
  checks that it is the identity channel, where it used to be skipped with a
  note blaming the library ([next#445]).
- The docstrings of `optyx.photonic.ZMeasurementDR` and `XMeasurementDR` say
  what they measure, the Z or X spider of phase alpha, i.e. the X or Z basis at
  alpha = 0: the old text of `ZMeasurementDR` left out its `HadamardBS`
  ([next#445]).
- The docstring of `optyx.photonic.BBS` gives the matrix it computes, `i cos`
  on the diagonal and `sin` off it, where the old one had its columns swapped,
  and those of `TBS` and `MZI` say their matrices include their `global_phase`
  ([next#445]).
- `optyx.qubits.X(1, 1, .25).dual_rail()` is `BS` times `exp(-i pi / 4)`, and
  `X(1, 1, .75)`, also written `-.25`, is `BS.dagger()` times `exp(i pi / 4)`,
  where the old code returned the wrong beam splitter or raised
  `NotImplementedError` ([next#445]).
- `optyx.channel.Diagram.from_perceval` applies a `PostSelect` to the
  processor's modes, heralded ones included, as perceval does: a condition on a
  mode after a herald used to read the mode to its right ([next#445]).
- `Diagram.from_perceval` accepts a `PostSelect` that uses `!=`, which perceval
  accepts and prints back, instead of raising 'Unexpected operator'
  ([next#445]).
- `Diagram.from_perceval` creates photons only for input heralds
  (`p.in_heralds`) and selects them only for output heralds, so an INPUT-only
  or OUTPUT-only herald gives the same domain as perceval ([next#445]).
- Optyx reads a foliated diagram one box at a time: `Diagram.sample`, `dilate`,
  `get_kraus`, `explode_channel` and `BitControlledGate` no longer raise
  `ValueError` on a layer that holds several boxes ([next#445]).
- Mixed evaluation of a daggered or transposed pure box, e.g. `S.dagger()` or
  `U.r`, is the doubled pure evaluation, by `channel.Functor` and by
  `Circuit.eval(mixed=True, contractor=...)`, so circuits with a measurement
  give the right probabilities ([next#445]).
- `to_tk` emits a transposed `Y` with its sign, since `Y.r == -Y`: `CY.l` and
  `CY.r` go to tket as `CY` followed by `Z` on the control, so tket samples the
  same outcomes as `eval` ([next#445]).
- Rotating or conjugating a `Ket`, `Bra`, `Bits`, `Digits`, `Discard` or
  `MixedState` reverses its wires: `Ket(1, 0).r == Bra(0, 1)`,
  `Ket(1, 0).conjugate() == Ket(0, 1)` and
  `Discard(bit @ qubit).r == MixedState(qubit @ bit)` ([next#445]).
- `to_tk` translates `Measure(override_bits=True)` after classical
  post-processing correctly: it discards the old bits in post-processing and
  measures into fresh ones, where before the post-processing read the new value
  and a `Copy` before it raised `IndexError` ([next#445]).
- `Circuit.eval(backend=..., measure_all=True)` and `Circuit.get_counts`
  measure each open qubit into a fresh bit, so `eval` no longer raises
  `IndexError` and a circuit with a `Bra` no longer post-selects on the wrong
  qubit ([next#445]).
- `circuit2zx` maps `sqrt(x)` to the scalar of its root, conjugated for its
  dagger, so the images of `Circuit.cups`, `caps` and 0-input spiders are no
  longer off by sqrt(2) ([next#445]).
- `circuit2zx` maps `Rz(a)` and `Rx(a)` to their spiders times `exp(-i pi a)`,
  so the image of every rotation, and of a sum or gradient of them, equals the
  circuit exactly ([next#445]).
- `Scalar.grad` is the derivative of the scalar's evaluation in each mode
  (`2 Re(conj(s) s')` under mixed evaluation), and `Sqrt.grad` differentiates
  the root instead of the radicand ([next#445]).
- `eval(repr(x)) == x` now holds for `Ket`, `Bra`, `Copy`, `Match` and
  `MixedScalar` (delayed too), and for a `Scalar` with a custom name, so
  circuits that prepare, post-select, copy or match round-trip their repr
  ([next#445]).
- `Circuit.eval(*others, contractor=...)` evaluates every circuit it is given,
  so `Sum.eval(contractor=...)` returns the sum of its terms instead of one
  term's array summed along its first axis ([next#445]).
- `lambdify` on a parametrised quantum gate or scalar evaluates its phase in
  the current backend, whatever `modules` it is given, and passes its other
  keyword arguments to `sympy.lambdify`, so torch gradients flow through
  nonlinear phases and `jax.grad` works through them ([next#445]).
- `Circuit.measure` measures every open qubit when it evaluates as a channel,
  instead of discarding it, so `(Ket(0) >> H).measure(mixed=True)` is
  `[.5, .5]` and `((Ket(0) >> X) @ Bits(1)).measure()` is the joint
  distribution `[[0, 0], [0, 1]]` ([next#445]).
- A pure circuit `Bubble` evaluates under `mixed=True` as the doubling of its
  pure evaluation (single for a classical one), instead of raising
  `AttributeError` on `is_classical` ([next#445]).
- `to_pennylane` converts `U1` and `CCX` (and so every `Rydberg` ansatz)
  instead of raising `KeyError` ([next#445]).
- `Scalar(data, name=...)` and the other keyword forms of its constructor build
  the scalar instead of raising `TypeError` ([next#445]).
- `Circuit.measure` and `get_counts` without a backend accept a delayed circuit
  and agree with the undelayed one, instead of raising `AxiomError`; `to_tk`
  raises `NotImplementedError` on one, as on anything else it cannot translate
  ([next#445]).
- `Circuit.measure` and `init_and_discard` prepare a digit of any dimension in
  `Digits(0, dim=...)`, so a classical circuit on `Digit(4)` measures instead
  of raising `AxiomError` ([next#445]).
- `quantum.Measure`'s `destructive` and `Encode`'s `constructive` are
  documented the right way round, `Circuit.from_tk` says it adds logical swaps
  rather than SWAP gates, and `Controlled` says that a zero distance raises
  `ValueError` ([next#445]).
- Calling a `grammar.cfg.Tree` on arguments whose types do not tensor to its
  `dom` (all identities, an `Id` root, too many or too few arguments) raises
  `AxiomError`, where it used to return an ill-typed or truncated tree
  ([next#445]).
- `grammar.cfg.Algebra` maps `Id` and any tree with an `Id` branch to the
  identity of its codomain operad, where it used to raise `AttributeError`
  ([next#445]).
- `grammar.cfg.Id(x).to_diagram()` is `monoidal.Id(x)`, so the diagram of a
  tree with an identity branch has no spurious `Id(x)` box ([next#445]).
- `quantum.graphlike.size` counts a phase-free spider with three or more ports
  and no box (e.g. a reduced `Z(1, 2)`), as PyZX does after its own
  `full_reduce`; only a bare wire, cup or cap counts as no spider ([next#445]).
- `grammar.categorial.tree2diagram(tree, dom=...)` gives `dom` to every word of
  the tree, not only to a tree that is a single word ([next#445]).
- `eval(repr(r)) == r` for a `grammar.thue.Rule` or `grammar.cfg.Rule` with an
  empty name, e.g. the root of `cfg.Tree.from_nltk` on an unlabelled tree
  ([next#445]).
- `grammar.categorial.FX` and `BX` name subterms that compose, `C << B` with
  `A >> B` and `B << C` with `B >> A`, where the pairs they documented raised
  `AxiomError` ([next#445]).
- `pregroup.Diagram.normal_form` passes `normalize` and its other params on to
  `rigid.Diagram.normal_form`, where it used to drop them on every input
  ([next#445]).
- The `quantum.contextuality` module docs say the gradient is constant on each
  context on the non-contextual polytope, not zero, and that the dual solution
  is not unique anywhere on it when the scenario has more than one context
  ([next#445]).
- `reservoir.Readout` compares by value: `eval(repr(r)) == r` (with `array` in
  scope), and two identical fits compare equal ([next#445]).
- `python.finset.Permutation` refuses a `Function` whose domain is not its
  codomain, so that `then` and `tensor` with a discard or a projection raise
  rather than return a permutation of the wrong size, and `Permutation.then`
  raises `AxiomError` on a `Function` that does not compose with it, as
  `Function.then` does ([next#445]).
- `Drawing.dagger` reflects a drawing with a bubble, a frame or a slot, e.g. an
  `Equation` of coloured terms, where it used to raise a bare `AssertionError`
  ([next#445]).
- The dagger of a drawing with two or more boxes has a graph of its own, not a
  frozen networkx view, so that `add_nodes`, `add_edges` and `add_box_corners`
  work on it ([next#445]).
- `draw(doctest=..., to_tikz=True)` compares the drawing against its `.tikz`
  and `.tikzstyles` baselines and raises when they differ, where it used to
  overwrite them ([next#445]).
- TikZ output with `use_tikzstyles=True` writes its `.tikzstyles` beside the
  `.tikz` file for any path, a `pathlib.Path`, a dotted folder or a name with
  no extension among them ([next#445]).
- An SVG drawn to a `BytesIO` or to an open file, text or binary, carries the
  dark-mode style, where it used to crash after Matplotlib had written it
  ([next#445]).
- `Drawing.bubble` takes a `dom` and a `cod` of plain types, e.g. symmetric,
  rigid, compact or frobenius ones, casting them with `to_drawing`: it used to
  leave nodes with no position when their length matched the inside's, so that
  drawing raised `KeyError` ([next#445]).
- `Drawing.__eq__` compares the edges of the plane graph, so that drawings with
  the same nodes and positions but different wiring are not equal ([next#445]).
- `python.additive.Function.braid` works on an instance as well as on the
  class, where it used to raise `TypeError` ([next#445]).
- `Function.no_type_checking` on a subclass of `python.function.Function`
  leaves the subclass as it found it, so that the switch on the base class
  still reaches it and nesting the two no longer turns its type checking off
  for good ([next#445]).
- The label of a box or a braket coloured `config.TRANSPARENT` is drawn black
  and turns white on a dark page, where it used to be drawn white, invisible on
  a light one ([next#445]).

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
  `main` passes, the rules it breaks listed in `pyproject.toml` ([next#263]),
  and `E714` and `RUF022` leave that list ([next#438]).
- `stream.Ty.singleton` and `stream.Stream.singleton` pass the callee itself as
  `_later` instead of a lambda wrapping it, which takes two of the three
  `PLW0108` instances off `discopy/` before the rule graduates out of ruff's
  preview, the third being left with [next#443]. Equality still compares
  `_later` by identity, so those two now compare equal to a twin built the same
  way, where every other non-constant stream does not ([next#457]).
- `test_Diagram_permutation` and `test_Kauffman` patch class attributes through
  `monkeypatch`, so a failure no longer leaves them patched for the tests after
  it, and `test/balanced.py` defines `test_functor_twist_dagger` once
- Every push to `main` gets its own `build` and `benchmark` run, since a
  concurrency group per ref let a newly queued push cancel the pending one
  ([next#445]).
- The shared setup action runs setup-uv 10.2.0 and setup-python 7.0.0, the
  versions `benchmark.yml` runs, and Dependabot now scans `.github/actions/*`
  so they keep moving together ([next#445]).
- pylint's `fail-under` is raised from 9.09 to 9.33, main's score rounded
  down, so the score can no longer drop 0.25 without lint failing ([next#445]).
- The `test` extra includes `notebooks`, so `pip install -e '.[test]'` brings
  anywidget and the "Build without uv" recipe's `coverage run -m pytest` no
  longer stops at collection on `discopy/drawing/widget.py` ([next#445]).
- The benchmark commands in `CONTRIBUTING.md` and
  `benchmark/test_composition.py` create `benchmark-results/` first, so they
  run on a fresh clone ([next#445]).
- `test/readme.py` checks every relative link and anchor of the README, so a
  link to a deleted notebook or to a heading no section defines fails the suite
  ([next#445]).
- `pytest --skip-extra <file>` skips a module that needs an extra instead of
  failing at collection, for every module `--skip-extra` covers, not only
  `discopy.optyx` ([next#445]).
- `pytest --help` no longer says `--skip-extra` skips nothing once the extras
  are installed: the flag skips its fixed list whether or not they import
  ([next#445]).
- `benchmark/report.py`'s `NAME-results.md` writes medians with 3 significant
  figures, so microsecond-scale cases no longer read 0.0000 ([next#445]).
- `docs/export_notebooks.py` runs each notebook on the discopy of its own tree,
  so `--check` from a worktree no longer tests the editable install's code
  ([next#445]).
- `docs/export_notebooks.py` fails when any named notebook matches nothing,
  naming it, instead of silently checking only the names that match
  ([next#445]).

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
[#829]: https://github.com/discopy/discopy/pull/829
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
[next#8]: https://github.com/rel-int/discopy-next/issues/8
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
[next#249]: https://github.com/rel-int/discopy-next/issues/249
[next#250]: https://github.com/rel-int/discopy-next/issues/250
[next#254]: https://github.com/rel-int/discopy-next/issues/254
[next#258]: https://github.com/rel-int/discopy-next/issues/258
[next#260]: https://github.com/rel-int/discopy-next/issues/260
[next#263]: https://github.com/rel-int/discopy-next/pull/263
[next#276]: https://github.com/rel-int/discopy-next/pull/276
[next#278]: https://github.com/rel-int/discopy-next/issues/278
[next#283]: https://github.com/rel-int/discopy-next/pull/283
[next#286]: https://github.com/rel-int/discopy-next/issues/286
[next#289]: https://github.com/rel-int/discopy-next/issues/289
[next#290]: https://github.com/rel-int/discopy-next/issues/290
[next#291]: https://github.com/rel-int/discopy-next/issues/291
[next#292]: https://github.com/rel-int/discopy-next/issues/292
[next#293]: https://github.com/rel-int/discopy-next/issues/293
[next#294]: https://github.com/rel-int/discopy-next/issues/294
[next#295]: https://github.com/rel-int/discopy-next/issues/295
[next#297]: https://github.com/rel-int/discopy-next/issues/297
[next#298]: https://github.com/rel-int/discopy-next/issues/298
[next#299]: https://github.com/rel-int/discopy-next/issues/299
[next#300]: https://github.com/rel-int/discopy-next/issues/300
[next#301]: https://github.com/rel-int/discopy-next/issues/301
[next#302]: https://github.com/rel-int/discopy-next/pull/302
[next#310]: https://github.com/rel-int/discopy-next/pull/310
[next#313]: https://github.com/rel-int/discopy-next/issues/313
[next#314]: https://github.com/rel-int/discopy-next/issues/314
[next#316]: https://github.com/rel-int/discopy-next/pull/316
[next#320]: https://github.com/rel-int/discopy-next/pull/320
[next#323]: https://github.com/rel-int/discopy-next/issues/323
[next#349]: https://github.com/rel-int/discopy-next/issues/349
[next#350]: https://github.com/rel-int/discopy-next/issues/350
[next#351]: https://github.com/rel-int/discopy-next/issues/351
[next#359]: https://github.com/rel-int/discopy-next/issues/359
[next#376]: https://github.com/rel-int/discopy-next/issues/376
[next#382]: https://github.com/rel-int/discopy-next/issues/382
[next#386]: https://github.com/rel-int/discopy-next/issues/386
[next#387]: https://github.com/rel-int/discopy-next/issues/387
[next#388]: https://github.com/rel-int/discopy-next/issues/388
[next#399]: https://github.com/rel-int/discopy-next/issues/399
[next#229]: https://github.com/rel-int/discopy-next/issues/229
[next#435]: https://github.com/rel-int/discopy-next/issues/435
[next#247]: https://github.com/rel-int/discopy-next/issues/247
[next#251]: https://github.com/rel-int/discopy-next/issues/251
[next#421]: https://github.com/rel-int/discopy-next/issues/421
[next#422]: https://github.com/rel-int/discopy-next/issues/422
[next#426]: https://github.com/rel-int/discopy-next/issues/426
[next#411]: https://github.com/rel-int/discopy-next/pull/411
[next#438]: https://github.com/rel-int/discopy-next/issues/438
[next#443]: https://github.com/rel-int/discopy-next/issues/443
[next#457]: https://github.com/rel-int/discopy-next/issues/457
[next#458]: https://github.com/rel-int/discopy-next/pull/458
[next#459]: https://github.com/rel-int/discopy-next/pull/459
[next#445]: https://github.com/rel-int/discopy-next/issues/445
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
