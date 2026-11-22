> Review round on #694 (cubic-dev-ai[bot]), head 846cc544:
>
> 1. CHANGELOG.md:779 (thread 4656032787)
> [P3] This entry overstates the fix. Normalization happens only in `Wire.__init__` and `__setstate__`, so a wire whose `z` is assigned after construction still has nonzero `z` and still recurses in the functor. Reword it to say the wire is normalized when built or unpickled, and drop 'hand-built wire with a nonzero z'.
>
> 2. docs/adr/0008-a-frobenius-winding-is-read-modulo-one.md:34 (thread 4656032795)
> [P3] The Status line omits the pull request carrying this decision. Add the #694 reference as the index requires until the change merges.
>
> ```suggestion
> Accepted, 2026-11-22, carried out by #694.
> ```
>
> 3. docs/adr/0008-a-frobenius-winding-is-read-modulo-one.md:17 (thread 4656032816)
> [P3] The Decision states a general modulo rule, but only `frobenius.Wire` implements it. Scope the sentence to frobenius so the record doesn't imply `rigid` or `pivotal` normalise `z`.
>
> ```suggestion
> A winding number on a frobenius object is read modulo its one winding, so it is always zero:
> `Wire('x', 1) == Wire('x')`. A nonzero `z` on a self-dual wire is a
> ```

- [x] CHANGELOG: say built or unpickled, not hand-built, since z assigned after the fact is not read again
- [WIP] @session_01C7YfAR6V7pj9o6FNkELjMG-2026-11-22 01:51 docs/adr/0008 Status: name #694, which carries it out, as the index asks until it merges
- [WIP] @session_01C7YfAR6V7pj9o6FNkELjMG-2026-11-22 01:51 docs/adr/0008 Decision: scope the modulo rule to frobenius, which is where it is implemented
