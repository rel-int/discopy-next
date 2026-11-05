> I got some spare tokens to spend: let’s use them to polish all things discopy-next. Make a deep review of the code and the infra around it, summarising your findings and suggestions in an issue, fixing any bugs that you can fix without requiring a design decisions from my side.
>
> go ahead with the fixes once workflow A lands

The fixes of rel-int/discopy-next#445 in the symmetric tower and the closed levels (`symmetric` to `ribbon`, `rigid`, `closed`, `biclosed`): bugs the review confirmed that need no design decision, one commit and one regression test each.

- [x] **S3-1** feedback.Box.delay rebuilds from _args/_params, which never hold a markov box's flags or a rigid box's winding z.
- [x] **S3-2** ribbon.Twist, DualRailTwist, DualRailBraid, DualRailCup and DualRailCap rotate to boxes on the wrong objects, so .r/.l of any ribbon diagram with a …
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S3-3** braided.Diagram.naturality never checks that the layers next to box i are the braid it slides through, so it silently deletes whatever is there:
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S3-4** The identity markov (comarkov) functor does not return a Function (Cofunction) unchanged:
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S3-7** ribbon.DualRail rebuilds every non-Sum, non-Bubble box as a fresh Box:
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S3-8** symmetric.Permutation, markov.Function and comarkov.Cofunction raise a bare TypeError on numpy integer entries (Function even on the identity …
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S3-9** eval(repr(x)) raises for ribbon.DualRailCup and DualRailCap:
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S3-10** ribbon.Diagram.cup(i, i) (and so compact's) cups wire i-1 with wire i instead of raising;
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S3-11** Slice tests: test_functor_twist_dagger is defined twice (the 10-27 copy shadows the 10-17 one), test_Diagram_permutation and test_Kauffman leave …
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S3-12** Docstrings in the slice contradict the code:
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S3-13** Swap.__init__ reads any length-2 `left` as a (dom, perm) pair, so Swap(x @ y, z) reports a 'permutation of length 2' instead of NOT_ATOMIC;
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S4-1** biclosed.Functor sends terms to map_term whenever cod.ob subclasses biclosed.Ty, so any functor into rigid, pivotal, pregroup or Tensor raises …
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S4-2** Every closed functor recurses forever on the unitype U, the identity functor included, because the Exp branch maps U.base == U first.
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S4-3** to_rigid on a rigid diagram is not the identity:
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S4-5** closed.TermBase.from_biclosed (and so biclosed TermBase.to_closed) drops each constant's data, so biclosed terms that differ only in constant data …
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S4-6** biclosed.Constant.__repr__ (inherited by closed.Constant) omits data, so eval(repr(c)) != c for a constant with data, and for every term built on one
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S4-7** An explicit Curry(f, 0), which the constructor accepts and to_drawing draws as f, crashes biclosed/closed functors (and so to_map), closed to_compact …
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S4-8** Products and exponentials print their whole type for an eager name, so building nested products (e.g.
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 **S4-9** The second example of rigid.Diagram.normal_form compares double_snake with the first example's two_snakes, whose type differs, so its `False` cannot …
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 full suite and lint gates on the branch
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 one `CHANGELOG.md` line per user-facing fix
