> CI of `main` on the merge commit of #658 (7822dd06), `proptest` job, `explore` profile
> (https://github.com/rel-int/discopy-next/actions/runs/44822302056/job/136585985879):
>
> > [gw0] [  3%] FAILED proptest/test_axioms.py::test_axiom[balanced.TermBase.str_transparency]
>
> Reproduced locally with `HYPOTHESIS_PROFILE=explore uv run pytest proptest/ -k "TermBase and str_transparency"`:
> the same cell fails for `spacial`, `braided`, `feedback`, `balanced`, `comarkov`, `symmetric` and `markov`, on
>
> > Draw 1 (str_transparency): cls.generate(Ty('a'), [1], [Ty('a')], ['x'])
> > testable.Equation(monoidal.Kappa((monoidal.Variable('x0', ...),), monoidal.Variable('x0', ...), 0), balanced.Kappa((balanced.Variable('x0', ...),), balanced.Variable('x0', ...), 0))
> > assert inhabits(verdict, law.output(scope))
>
> The equation holds; its left side is not a `balanced.TermBase`, which `Equation[Self]` asks for.

- [x] str_transparency of a term with no box: a regression test of the cell's judgement at every level, then the law stated as what a string can say
- [x] run the term cells of the matrix under the explore profile and fix or file whatever else it finds
