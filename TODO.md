> ask, one 🚀: `frobenius.Wire` reads `z` modulo its windings, so `Wire('x', 1) == Wire('x', 0)` and the functor in the report terminates. reading 2 above.
>
> not taken: the constructor refuses a nonzero `z` (reading 1).
>
> default: reading 2. asked 2026-11-20, due 11-22: with no word an agent builds it then, and one word reverses it.

- [x] frobenius.Wire reads z modulo its windings, so a nonzero z normalises to zero
- [x] test/frobenius.py: the report's Functor term and Wire('x', 1).l.l, plus equality
- [ ] one [Unreleased] bullet in CHANGELOG.md citing the PR, with its link definition
- [WIP] @session_01C7YfAR6V7pj9o6FNkELjMG-2026-11-22 01:31 docs/adr/0008: the ruling that a nonzero z on a self-dual wire has a normal form
