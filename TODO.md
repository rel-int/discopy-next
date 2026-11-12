> Review round on #520 (cubic-dev-ai[bot]), head 027849bd:
>
> 1. test/neural/propagation.py:55 (thread 4550090026)
> [P2] This assertion makes the test fail for its own seeded tensors because the per-round maximum error is not monotonic. Remove the ordering requirement or assert only that insufficient rounds differ from the exact result and the final round is exact.

- [ ] test/neural/propagation.py:55 This assertion makes the test fail for its own seeded tensors because the per-round maximum error is not mono…
- [ ] the law is sharper than stated: on a tree sum-product is exact after half the diameter, rounded up, and not one round before (200 random trees); say so in the docs and test it
