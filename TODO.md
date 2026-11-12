> Review round on #520 (cubic-dev-ai[bot]), head 4e050508:
>
> 1. test/neural/propagation.py:38 (thread 4549988389)
> [P2] This test passes a `numpy.int64` as `states`, but `Dims` only recognizes builtin `int` values, so the generated network cannot construct its message dimensions. Cast the sampled state count to `int` before passing it to `sum_product`.
>
> ```suggestion
>         factors, states = random_tree(rng), int(rng.choice([2, 3]))
> ```
>
> 2. discopy/neural/propagation.py:126 (thread 4549988411)
> [P2] `reply` fails for factors with more than 26 nodes because the einsum label alphabet is exhausted. Expand the supported label scheme (or use integer-label einsum) instead of indexing a shorter string.
>
> 3. discopy/neural/propagation.py:106 (thread 4549988414)
> [P2] `belief` uses an unstable softmax, so sufficiently sharp but valid messages can produce NaN beliefs. Subtract the per-example maximum before exponentiating, or use a framework log-sum-exp/softmax operation.
>
> 4. discopy/neural/propagation.py:213 (thread 4549988428)
> [P2] `marginals` silently ignores extra tensors and contracts too few factors when tensors are missing because `zip` truncates. Validate that `len(tensors) == len(table)` before building `operands`, otherwise malformed input can produce incorrect marginals.
>
> ```suggestion
>     if len(tensors) != len(table):
>         raise ValueError(
>             f"Expected {len(table)} tensors, got {len(tensors)}.")
>     operands = [x for pair in zip(tensors, map(list, table)) for x in pair]
> ```
>
> 5. discopy/neural/__init__.py:11 (thread 4549988438)
> [P3] This summary implies exact contraction on every tree without stating the required number of rounds. Match `propagation`’s documented law by saying the result is exact after as many rounds as the incidence-graph diameter.
>
> ```suggestion
> network with a trace, exact on a tree after as many rounds as its
> incidence-graph diameter, giving the contraction of its tensors.
> ```

- [ ] test/neural/propagation.py:38 This test passes a `numpy.int64` as `states`, but `Dims` only recognizes builtin `int` values, so the generat…
- [ ] discopy/neural/propagation.py:126 `reply` fails for factors with more than 26 nodes because the einsum label alphabet is exhausted.
- [ ] discopy/neural/propagation.py:106 `belief` uses an unstable softmax, so sufficiently sharp but valid messages can produce NaN beliefs.
- [ ] discopy/neural/propagation.py:213 `marginals` silently ignores extra tensors and contracts too few factors when tensors are missing because `zi…
- [ ] discopy/neural/__init__.py:11 This summary implies exact contraction on every tree without stating the required number of rounds.
