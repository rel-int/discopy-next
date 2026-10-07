# TODO

> once you're done, freeze the feature set of the pattern language and clean it up. then, check EVERY diff chunk from main and think about whether its really useful. be very wary of toplevel functions and cryptic names. integrate the property testing as much as possible into the hierarchy, simplify the property testing to the bare minimum as long as good statistics are ok.

- [ ] Measure the search statistics before, to compare after
- [ ] `discopy.pattern`: freeze the patterns, fold the top-level functions into the classes they belong to, rename the cryptic names
- [ ] `discopy.axioms`: integrate into the hierarchy, cut to the bare minimum
- [ ] `discopy.abc`: the search helpers, every chunk from main
- [ ] Every other chunk of `discopy/` from main
- [ ] `proptest/` and the tests: the bare minimum keeping good statistics
- [ ] Checks, changelog, report
