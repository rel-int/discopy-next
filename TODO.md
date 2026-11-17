> **Check rendered wire curves for box intersections** (chatgpt-codex-connector, P2): for `(Box('f', 'x', 'x') @ Ty('y')).to_drawing()`, moving the bypass wire's input to x = 0.6 and its output to x = 2 is accepted, although the drawn quadratic Bézier passes through `f`: `through()` tests the straight chord.

- [WIP] @session_01MEJxy3Gb9UVB1Y7ExMgpNV-2026-11-17 00:50 shape and crossings read each wire as the Bézier curve the backends draw, sampled; test Codex's example
