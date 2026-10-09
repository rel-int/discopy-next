# TODO

Round 2: review of @toumix, 2026-10-06, quoted verbatim.

> **discopy/abc.py:112** let's move these comments to the docstring
>
> **discopy/pattern.py:107** waiiit how is this even possible? abc imports axioms which imports pattern ... which imports abc?
>
> **discopy/pattern.py:159** I'd avoid having this string `"C0"` hard-coded here, either find a workaround and make it a named constant
>
> **discopy/pattern.py:193** Can we add a type annotation for bound?
>
> **discopy/pattern.py:197** Same here it's sketchy to use this hardcoded `"Ob"`
>
> **discopy/pattern.py:206** Looks very hacky, why can't we make `Count` a subclass of `Sort` rather than using hardcoded strings?
>
> **discopy/pattern.py:218** This is misleading, it didn't expect a sort but a bound
>
> **discopy/pattern.py:246** I feel like at this level of abstraction it doesn't make sense to even talk about categories, the least structure should be a bare Python `object`. This should also help remove the circularity abc < axioms < pattern < abc
>
> **discopy/pattern.py:253** you can avoid this getattr by adding a `__dataclass_fields__ = ()` default to the Pattern class
>
> **discopy/pattern.py:256** this wrapping into tuple seems sketchy, can we get rid of it?
>
> **discopy/pattern.py:297** Can you also show an example when `value is None`?
>
> **discopy/pattern.py:349** grammar seems to be broken here "its sort the bound"
>
> **discopy/pattern.py:852** What's the type of annotation?
>
> **discopy/pattern.py:895** I must say I'm not fan of these `foo_of` methods, seems to go against Python's object oriented philosophy. Can we replace this with a subclass of `Callable`?
>
> **discopy/pattern.py:921** That's the crazeist meta programming I've ever seen, not sure if I love it or hate it 😂
>
> **discopy/pattern.py:926** grammar is broken here "a pattern a premise"
>
> **discopy/pattern.py:974** goes against STYLE.md, this should be a standalone function with its own docstring
>
> **discopy/pattern.py:1054** This docstring doesn't even mention `Declaration` what's going on?
>
> **discopy/pattern.py:1144** can we make the `dict` annotation more precise?
>
> **discopy/pattern.py:1147** grammar is broken here too "a count the number two"?
>
> **discopy/pattern.py:1160** conditioning on a hardcoded string to return a hardcoded int... 😱
>
> **discopy/pattern.py:1177** This is confusing, why would the `generate` method "draw"? Is this draw as in drawing a picture or drawing a sample?
>
> **discopy/pattern.py:1250** hardcoding "x" and "y" feels wrong
>
> **discopy/axioms.py:358** Looks wrong, there should be one less relation symbol than elements in the list

- [x] `axioms.Equation`: one symbol fewer than terms (axioms.py:358)
- [x] `Count` a subclass of `Sort`, no hard-coded `"Count"` (pattern.py:206, :1160)
- [x] the heads `"C0"`, `"C1"`, `"Self"` as named constants (pattern.py:159, :197)
- [x] `sort_of`: annotate `bound`, say it expected a bound (pattern.py:193, :218)
- [x] `Pattern.__dataclass_fields__ = ()`, no tuple wrapping in `parts` (pattern.py:253, :256)
- [x] doctest of `match` with `value is None` (pattern.py:297)
- [x] grammar of the docstrings (pattern.py:349, :926, :1147, :1054)
- [x] annotate `annotation`, the `dict` of `canonical` (pattern.py:852, :1144)
- [ ] the `*_of` functions: proposal posted, waiting on @toumix (pattern.py:895)
- [x] `stated` a standalone function with its own docstring (pattern.py:974)
- [x] "draw" in `generate` reads as sampling (pattern.py:1177)
- [ ] the names `x` and `y` of `cell`: options posted, waiting on @toumix (pattern.py:1250)
- [ ] `#:` comments of `abc.Category` as docstrings: waiting on @toumix after @daydream6728's reply (abc.py:112)
- [ ] decoupling patterns from `abc`, the import cycle: deferred by the user to later in this PR (pattern.py:107, :246)
- [x] carried over from round 1: pylint in the dev group and CI, scoring 9.25 under the locked pylint

Round 3: @daydream6728's directive, [2026-10-06T11:43:35Z](https://github.com/rel-int/discopy-next/pull/1#issuecomment-6015513416), quoted verbatim.

> > I really like the sequent idea but for now it's not clear if this saves us more bureaucracy than it creates.
>
> Agreed, I think its possible to make it work in a nicer way. The branch as such accumulates a lot of past attempts so we'd better clear it first and strip it down to a minimum. For example, Count isn't even used right now. We have two options:
> - for now i recommend to keep trying to make Count and other type level shenanigans work to be able to express all rules.
> - if it doesn't end up being conclusive, don't use rules to sample diagrams and instead annotate generators (which can be thought of atomic logical rules), while structural rules would just become part of the search strategies.
>
> Regardless, the alternative to the sequents would be the current shapes in main, e.g. ComposablePair, TraceSuperposing, etc... which is arguably even more boilerplate and definitely less general. I'm not sure what would be a good third approach.

- [WIP] @session_01TM6ep2BEKAeLmFZomE14Qx-2026-10-09 00:15 measure the directive's example: what reads `Count`, and what is still hardcoded about it
- [WIP] @session_01TM6ep2BEKAeLmFZomE14Qx-2026-10-09 00:15 a criterion for *conclusive* (option 1) and its first measurement, since nothing on the branch measures one
- [WIP] @session_01TM6ep2BEKAeLmFZomE14Qx-2026-10-09 00:15 strip what nothing reads, measured rather than guessed: coverage of the new modules over the whole suite
- [WIP] @session_01TM6ep2BEKAeLmFZomE14Qx-2026-10-09 00:15 answer the directive on the pull request with those numbers
