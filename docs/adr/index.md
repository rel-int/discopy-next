# Architecture decisions

An architecture decision record (ADR) states one design decision: what forced it, what was
decided and what follows. The changelog says what changed in a release; an ADR says why a
design is the way it is, so that nobody reverses it by accident.

Each one is `docs/adr/YYYY-MM-DD-title.md`, dated the day it was accepted and shorter than a
page, so that the files sort in the order they were decided and two pull requests writing one
each never collide. A record has no number here: it gets one only when it is sliced upstream to
[discopy/discopy](https://github.com/discopy/discopy). It has four sections:

- **Context**: the problem, and the rulings, issues and pull requests it was settled in.
- **Decision**: what is decided, as a rule that holds from now on.
- **Consequences**: what it costs, what it breaks and what it rules out.
- **Status**: `Accepted` with its date and, until the change has merged, the issue or pull
  request that carries it out.

An ADR is not rewritten once accepted. A decision that reverses it is a new ADR, whose context
links the old one by its file; the old one's status becomes ``Superseded by `<file>` ``, a link
to the new record, and nothing else in it changes. `test/adr.py` checks every file name and that
every link to a record, here or anywhere in the repo, names one that exists.

```{toctree}
:glob:
:maxdepth: 1

2*
```
