> The deprecated alias for `drawing.drawing` goes against docs/adr/0001-renames-take-no-alias.md (accepted 2026-11-20): a module renamed on next is renamed everywhere at once and keeps no alias. Remove the `drawing.drawing` alias, rename every remaining import and reference, and update the changelog line to match. The `traced` alias is not part of this PR.

- [x] Remove the drawing.drawing alias, per ADR 0001, and update the changelog
