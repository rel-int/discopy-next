Found by merging all nine #445 pull requests onto `main` (`d110a140`) and running the suite there, 11-06:

> FAILED test/optyx/test_channel.py::test_subs_substitutes_the_parameter_of_every_symbolic_box
> TypeError: Z.__init__() got an unexpected keyword argument 'is_dagger'

#447's S1-8 makes `cat.Box.subs` rebuild the box with its data substituted. `classical.Z` and `X` inherit that `subs`, since they are `CQMap`s not `Channel`s, and their constructor takes `(n_legs_in, n_legs_out, phase)`, which `rebuild` cannot call. On this branch alone they substitute through the `lambdify` S14-7 gives them; once #447 lands they raise.

- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-06 02:35 `classical.Z` and `X` substitute their phase themselves, as `quantum.zx.Spider` does, so `subs` holds whichever of #447 and #452 lands first
