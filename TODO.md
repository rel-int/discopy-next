> Review round on #692 (cubic-dev-ai[bot]), head 5b311302:
>
> 1. .github/tests/workflows.py:105 (thread 4655955143)
> [P2] This detector misses valid Graphviz requests written as the YAML boolean `graphviz: true`, allowing a future job to use `ubuntu-latest` without this test failing. Accept both the string and boolean forms when identifying Graphviz jobs.
>
> ```suggestion
>         step.get("with", {}).get("graphviz") in ("true", True)
> ```

- [ ] .github/tests/workflows.py:105 This detector misses valid Graphviz requests written as the YAML boolean `graphviz: true`, allowing a future …
