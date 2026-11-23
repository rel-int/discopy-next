# 1. A renamed module keeps no alias

## Context

Two modules of next were renamed in November 2026. The property-testing module `discopy.axioms`
became `discopy.testable` upstream ([discopy#829](https://github.com/discopy/discopy/pull/829)),
merged down with no alias in [#403](https://github.com/rel-int/discopy-next/pull/403).
`discopy.grammar` became `discopy.lambek` in [#569](https://github.com/rel-int/discopy-next/pull/569),
first with a deprecated alias, which toumix had removed four days later in
[#675](https://github.com/rel-int/discopy-next/pull/675): "remove the grammar alias".

## Decision

A module renamed on next is renamed everywhere at once, its tests, pictures and extra included,
and the old name is not kept as an alias. An alias that was never released is removed rather than
deprecated.

## Consequences

- Code importing the old name breaks, and the open branches that use it are fixed when they
  next merge `main`.
- A tree or a pickle saved under the old module path does not load.
- Nothing else needs to keep working under two names, `discopy.__getattr__` and
  `utils.RENAMED` included.
- Upstream releases are unaffected: deprecating a released name is a decision for upstream.

## Status

Accepted, 2026-11-20.
