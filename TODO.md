> USER, 11-16, in a live session (brief: #562):
>
> > write the brief for the modality, it can go all the way down to biclosed with closed and categorial as subclasses, it needs to be a free endo functor as for the delay, ie an integer for each Ob and Box together with diagram methods and Box subclasses for the structural primitives of a bang modality
>
> > of course we should also reproduce the experiments!
>
> > Go!
>
> USER, 11-16, same session, after a comparison with Bierman's linear categories:
>
> > ok drop the free monoidal endofunctor idea it wasn't a good idea, just add an Ob subclass for BangOb and a Box subclass for Bang, add the weakening (just don't use it for the example) and call it LinearCategory, it's fine if we're slightly more general without symmetry
>
> > monoidality sounds like an axiom not a box, can we call it maybe Laxator?

- [x] abc: LinearCategory(BiclosedCategory), abstract bang, the six primitives as classmethods, Bierman's laws as axioms (comonad, lax monoidal, comonoid, naturality, coalgebra morphisms, exchange inverse, naturality, hexagon)
- [x] biclosed: BangOb(arg) a wire like Exp, Ty.bang, str/repr/eq/hash/serialisation, drawn !x
- [x] biclosed: Bang(arg) a bubble like Curry, Diagram.bang, factories
- [x] biclosed: Dereliction, Digging, Contraction, Weakening, Exchange, Laxator
- [WIP] @session_01MEJxy2QJCwia83KHppaJ1H-2026-11-16 21:55 biclosed: Functor through cod.bang and the primitives, else ob_map/ar_map
- [WIP] @session_01MEJxy2QJCwia83KHppaJ1H-2026-11-16 21:55 biclosed: the laws declared failing on the free category, testable.Bang, proptest cells, rigid's inherited cells
- [WIP] @session_01MEJxy2QJCwia83KHppaJ1H-2026-11-16 21:55 tests in test/biclosed.py and test/abc.py, the brief's doctests, CHANGELOG
