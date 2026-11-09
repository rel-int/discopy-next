> python.Function.uncurry drops abc's n, so uncurry(0) is not the identity
>
> Evening: next-repo bug, no design decision (desire OPERATIONS.md 'Bugs in a next repo')

- [x] Failing-first tests: uncurry(0) is the identity, uncurry(n) inverts curry(n) on both sides, the comment's swapped call
- [x] Give Function.uncurry abc's n: override base_and_exponent only, curry(0) the identity
- [WIP] @session_01MEJxy65KogNCEukoKHtHe1-2026-11-09 00:38 CHANGELOG.md bullet
