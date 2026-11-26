> cubic-dev-ai review 6078105959 on c16f3acc:
>
> > .github/workflows/build.yml:184 [P2] A failed `git ls-remote` is treated as “main has moved on” and exits successfully, silently skipping publication. Check the lookup's exit status and fail or retry when it cannot verify `main`.
>
> > .github/workflows/build.yml:192 [P2] The `main` check is not atomic with this force-push: an older run can pass it, pause, then overwrite notebooks published by a newer run. Serialize publishers and recheck `main` inside that serialization.

- [ ] build.yml:184 fail when `git ls-remote` cannot read `main`
- [ ] build.yml:192 serialize the publishers
