> I got some spare tokens to spend: let’s use them to polish all things discopy-next. Make a deep review of the code and the infra around it, summarising your findings and suggestions in an issue, fixing any bugs that you can fix without requiring a design decisions from my side.
>
> go ahead with the fixes once workflow A lands

The fixes of rel-int/discopy-next#445 in the syntax core and the law machinery (`cat`, `monoidal`, `utils`, `abc`, `testable`, `op`): bugs the review confirmed that need no design decision, one commit and one regression test each.

- [x] **S1-1** monoidal.Diagram.foliation (and so depth) computes the hypergraph before checking for a symmetric category.
- [x] **S1-2** Sum.then never checks that the two sides compose, so a zero (empty Sum) composes with any arrow on either side and returns an ill-typed zero.
- [x] **S1-3** dumps silently turns int, float, bool and None dict keys in box data into strings, so loads(dumps(f)) != f, breaking the serialisation law.
- [x] **S1-4** Functor composition (cat and monoidal then) applies G to F's raw map values before F's cast, so F >> G crashes (or, with callable maps, the composite …
- [x] **S1-5** deferred to #445: already fixed by 1b31981a (S1-4); DualRail composites are a separate pick
- [x] **S1-8** Box.subs (and so Bubble, diagram and Layer subs) ignores a key that is not a bare symbol, e.g.
- [x] **S1-9** A one-shot iterator passed as inside skips the composability and type checks in FreeCategory, and is silently consumed into an empty path by monoidal …
- [x] **S1-10** Sum stores a list of terms as given, so Sum([f, g]) != Sum((f, g)) == f + g and is unhashable;
- [x] **S1-12** Box.free_symbols is a cached_property, so after an in-place change to Box.data (a use the Functor docstring documents) subs and lambdify use stale …
- [x] **S1-13** Box.free_symbols visits every scalar of a numeric array in Python:
- [x] **S1-14** Diagram.normalize rebuilds the whole diagram's boxes and offsets lists on every pair it checks, and Diagram.boxes concatenates with a quadratic sum():
- [x] **S1-15** monoidal.Diagram.merge_layers restarts at pair 0 after every merge, re-running Layer.merge on pairs already known not to merge:
- [x] **S1-16** cat.Sum.generator lacks @property, so a sum's generator is always a truthy bound method, never the box or None
- [x] **S1-17** str of a one-argument cat.Bubble drops a non-default name and method, so eval(str(b)) != b and the str_transparency law fails on a named bubble …
- [x] **S1-20** monoidal.Nat (and abc.Nat) accept negative numbers, giving a 'natural number' unequal to Nat(0) that cancels Nat(1) and whose len raises;
- [x] **S1-21** monoidal.Ty (and biclosed.Ty) raise AxiomError with raw '<class ...>' reprs for a non-Wire cat.Ob such as Colour, where every other level raises …
- [x] **S1-22** test/utils.py defines test_MappingOrCallable twice (142 and 184), so the first copy is shadowed and never collected;
- [x] **S2-3** The default `op.Comarkov` and `op.Markov` (over finset.Function, the module's own FinSet example) fail `Testable.transparency` with a NameError, …
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S2-4** Axiom.falsify on a law not declared broken lets a refusal (any exception from the law) escape hypothesis.find, instead of returning the shrunk …
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S2-7** Equation.modulo rebuilds the equation from its terms and symbols only, so monoidal.Equation.space (the only extra attribute any subclass sets) resets …
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S2-8** str(AxiomFailure) is the repr of the tuple (reason, equation), not the reason its docstring calls the message
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S2-9** Axiom.modulo/failing/inapplicable docstrings cite laws not on main (two exist only on PR #1's branch, CompactCategory.dagger_contravariance nowhere);
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S2-10** Three messages (MATRIX_TWO_DTYPES, HAS_NO_ATTRIBUTE, COMPLEX_TYPE_HAS_NO_ATTR) are dead, as they are upstream.
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 full suite and lint gates on the branch
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 one `CHANGELOG.md` line per user-facing fix
