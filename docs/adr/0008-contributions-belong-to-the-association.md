# 8. Contributions belong to the DisCoPy association

## Context

DisCoPy is under the BSD 3-Clause License with "Copyright (c) 2019-2022, DisCoPy" and, until
now, no terms for contributions: no CLA, no DCO, no copyright headers. Each contributor kept the
copyright in their work, so relicensing the library, e.g. to the Apache License 2.0 of
`discopy.optyx`, needs the consent of every author with surviving code and of their employers.
An audit of the history on 2026-11-21 found about seven such people and up to six
organisations. The same day toumix asked "let's also add a CLA or DCO for the future edits to all
belong to the DisCoPy nonprofit?". DisCoPy is a French nonprofit association.

## Decision

Every contribution, from the first one a contributor submits after accepting the
[CLA](https://github.com/rel-int/discopy-next/blob/main/CLA.md), is assigned to the DisCoPy association. If that assignment fails, the
contribution is licensed to the association with the right to relicense. The association
licenses each contribution back to its author and is bound to distribute it under an
OSI-approved licence. A DCO is not enough: it certifies the right to submit, but the copyright
stays with the author. An employer, university or funder with rights in a contribution signs the
[entity agreement](https://github.com/rel-int/discopy-next/blob/main/CLA-entity.md). An agent's contributions are covered by the agreement of
the human who runs it.

## Consequences

- Relicensing code merged after this needs no one's consent but the association's. Code merged
  before it still needs its authors' consent: the CLA is not retroactive unless an author signs
  it for their past contributions too.
- A pull request cannot merge until its author has signed. That is one more step for a first
  contribution, and some contributors will not sign an assignment.
- The association must exist and act: it signs, keeps the records and decides any relicensing.

## Status

Proposed, 2026-11-21, in the pull request that adds `CLA.md`. It needs a lawyer's reading of
the assignment under French law, the association's details, and the signing process.
