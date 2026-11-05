> I got some spare tokens to spend: let’s use them to polish all things discopy-next. Make a deep review of the code and the infra around it, summarising your findings and suggestions in an issue, fixing any bugs that you can fix without requiring a design decisions from my side.
>
> go ahead with the fixes once workflow A lands

The fixes of rel-int/discopy-next#445 in `quantum` and `grammar`: bugs the review confirmed that need no design decision, one commit and one regression test each.

- [x] **S10-1** channel.Functor doubles a pure box's raw array, ignoring is_dagger and z.
- [x] **S10-3** to_tk ignores box.z: a transposed Y (Y.r == -Y) is emitted as Y.
- [x] **S10-4** SelfConjugate.rotate is the dagger and SelfConjugate.conjugate is self, and neither reverses the wires.
- [x] **S10-5** to_tk measures Measure(override_bits=True) into a tket bit that classical post-processing reads afterwards.
- [x] **S10-6** Circuit.eval(backend=..., measure_all=True), which the eval docstring recommends for unitary circuits, raises IndexError on any real backend.
- [x] **S10-7** circuit2zx maps Sqrt(x) to scalar(x): gate2zx reads Sqrt.data (the radicand) instead of its root, so sqrt(2), Sqrt(-1).dagger() and the images of …
- [x] **S10-8** circuit2zx maps Rz(a) and Rx(a) to Z(1,1,a) and X(1,1,a), which are exp(i*pi*a) times the gates, so single rotations come out only proportional and …
- [x] **S10-9** Scalar.grad ignores `mixed`, so under the default mixed=True a parametrised pure scalar's gradient evaluates to |s'|^2 where it should be d|s|^2 = 2 …
- [x] **S10-13** eval(repr(x)) fails for Ket, Bra, Copy, Match and MixedScalar, so no circuit that prepares, post-selects, copies or matches round-trips its repr;
- [x] **S10-14** Circuit.eval(*others, contractor=...) evaluates self alone and drops the others, so Sum.eval(contractor=...) silently returns a wrong bare array or …
- [x] **S10-15** Parametrized.lambdify passes dict(kwargs, modules=np) as sympy's positional `modules`, so the backend never reaches sympy:
- [x] **S10-16** Circuit.measure's channel branch discards open qubits instead of measuring them:
- [x] **S10-17** A pure circuit Bubble has no is_classical, so the mixed evaluation of any circuit containing one raises AttributeError;
- [x] **S10-18** to_pennylane raises KeyError on U1 and CCX, which to_tk emits, so every Rydberg ansatz (built from U1 and CU1) fails to convert
- [x] **S10-19** Scalar(data, name=...) raises TypeError: tensor.Box.__new__ binds the scalar's first positional argument (data) to `name`, so the keyword `name` …
- [x] **S10-20** measure, get_counts and to_tk raise AxiomError on a delayed circuit that eval accepts:
- [x] **S10-21** Five docstrings contradict the code: Controlled's default distance (0 vs 1), Measure's `destructive` and Encode's `constructive` (both inverted), …
- [x] **S11-1** cfg.Tree.__call__ skips the type check Tree.__init__ performs:
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S11-2** cfg.Algebra raises AttributeError on Id and on any tree with an Id branch:
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S11-3** cfg.Id.to_diagram() is a box named 'Id(x)', not monoidal.Id(x), so Tree.to_diagram breaks compositionality:
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S11-4** graphlike.size counts no phase-free spider with three or more ports and no boxes (e.g.
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S11-5** categorial.tree2diagram gives `dom` only to a tree that is a single word:
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S11-6** thue.Rule.__repr__ (reused by cfg.Rule) drops an empty name, so eval(repr(Rule(n, s, name=''))) rebuilds the default name 'Rule(n, s)' and compares …
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S11-7** categorial.FX and BX docstrings name subterm types that do not compose (the documented pairs raise AxiomError)
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S11-8** pregroup.Diagram.normal_form drops its params on every input:
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S11-9** contextuality module Note says the gradient is zero on the non-contextual polytope (with a non-unique dual only on its boundary);
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S11-10** reservoir.Readout compares by identity (dataclass eq=False), so eval(repr(r)) != r even with `array` bound, and two identical fits compare unequal;
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 full suite and lint gates on the branch
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 one `CHANGELOG.md` line per user-facing fix
