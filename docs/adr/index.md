# Architecture decisions

An architecture decision record (ADR) states one design decision: what forced it, what was
decided and what follows. The changelog says what changed in a release; an ADR says why a
design is the way it is, so that nobody reverses it by accident.

Each one is `docs/adr/NNNN-title.md`, numbered in order and under a page, with four sections:

- **Context**: the problem, and the rulings, issues and pull requests it was settled in.
- **Decision**: what is decided, as a rule that holds from now on.
- **Consequences**: what it costs, what it breaks and what it rules out.
- **Status**: `Accepted` with its date, and the pull request that carries it out if it has not
  merged yet.

An ADR is not rewritten once accepted. A decision that reverses it is a new ADR, whose context
links the old one; the old one's status becomes `Superseded by NNNN`, and nothing else in it
changes.

```{toctree}
:glob:
:maxdepth: 1

0*
```
