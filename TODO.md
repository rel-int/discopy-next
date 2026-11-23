> > generated terms never reuse a variable name
>
> make sure we catch that next time

- [ ] Generators: a naming of the kappa terms may reuse names, so that a let shadows a variable its expression uses up, in cat and in the samplers of monoidal, symmetric and markov; show the matrix catches #704 without the fix
- [ ] Fix #704: Application.contractible only refuses the redex when a variable left bound would capture a free variable of the argument, with a regression test
- [ ] Checks and changelog
