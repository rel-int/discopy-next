> Review round on #508 (cubic-dev-ai[bot]), head fdfc0b0a:
>
> 1. discopy/optyx/core/control.py:486 (thread 4545605100)
> [P2] `BinaryMatrixBox` computes over GF(2) with `% 2`, but this setoid records the raw matrix entries. Consequently, matrices such as `[[0]]` and `[[2]]` compute the same map but compare unequal and hash differently; normalize the matrix modulo 2 before adding it to the setoid.
>
> ```suggestion
>             tuple(map(tuple, (np.asarray(self.matrix, dtype=np.uint8) % 2).tolist())), )
> ```
>
> 2. discopy/optyx/channel.py:1677 (thread 4545605110)
> [P2] This setoid change makes channels with an unhashable Kraus diagram unhashable even though their box data is hashable. `hashable()` does not recurse into diagrams; preserve a hashable semantic representation for the Kraus map (and density matrix) before adding it to the setoid.

- [ ] discopy/optyx/core/control.py:486 `BinaryMatrixBox` computes over GF(2) with `% 2`, but this setoid records the raw matrix entries.
- [ ] discopy/optyx/channel.py:1677 This setoid change makes channels with an unhashable Kraus diagram unhashable even though their box data is h…
