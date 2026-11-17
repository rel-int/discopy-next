> Review round on #591 (cubic-dev-ai[bot]), head 95780720:
>
> 1. .github/workflows/proptest.yml:19 (thread 4594196813)
> [P2] A scheduled or manually dispatched run can still cancel a pending main push when both target the same commit, because all non-PR events use `github.sha` in this group. Key only push events by SHA and give other non-PR events a unique key such as `github.run_id`.
>
> ```suggestion
>   group: ${{ github.workflow }}-${{ github.event_name == 'pull_request' && github.ref || github.event_name == 'push' && github.sha || github.run_id }}
> ```

- [ ] .github/workflows/proptest.yml:19 A scheduled or manually dispatched run can still cancel a pending main push when both target the same commit,…
