> > we can write custom code that transpiles this paper adhoc but i agree both typst and latex should be in the same Ci From the start, check there is an upstream PR for typst backend on discopy
>
> > go with the typst move to next of course
>
> The upstream pull request is discopy/discopy#467, "Add proper Typst rendering backend" (0x0f0f0f, draft, head `38403e40`). The paper is the tool paper of #564.

- [x] merge discopy/discopy#467's head `38403e40` as a merge commit, its commits kept, each conflict resolved once
- [ ] the Typst tests and every gate of next-checks pass on the merged tree
- [ ] CHANGELOG.md bullet under [Unreleased], its links defined
