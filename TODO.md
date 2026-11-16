> chatgpt-codex-connector on #568, 2026-11-16:
>
> > Accept tuple coordinates when drawing controlled triangles: `Controlled(QuantumGate('U', qubit, qubit, [1, 0, 0, 1], shape='triangle')).draw()` raises `AttributeError: 'tuple' object has no attribute 'x'`, `draw_controlled_gate`'s `fake_positions` are plain tuples.
>
> > Account for rotations when orienting equal-arity triangles: `f.r` points down like `f`, `f.conjugate()` points up though a conjugate is a horizontal reflection.

- [x] read port positions by index, so tuples work: test a controlled triangle
- [x] apex of an equal-arity triangle flips with dagger and transpose, not conjugate: test f, f.r, f.dagger(), f.conjugate()
- [x] merge main
