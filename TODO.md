> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

- [x] #478: explicit_trace takes n and traces n wires in one pass, trace_backward_wires calls it once with the cut count
- [x] #478: Hypergraph.trace's non-symmetric branch calls explicit_trace(n, left) instead of recursing one wire at a time
- [x] #478: a test pinning the n-wire trace against the n nested ones, both trace_factory branches, both sides
- [ ] #478: the measurement on the issue — reading 2 is the same term as reading 1, and the quadratic cost is scan_types, not the trace count
- [ ] #478: cmap.CMap's identical explicit_trace loop, filed as its own issue
- [WIP] @session_01C7YfAQNzY28EVbk6YirCHJ-2026-11-13 01:44 CHANGELOG.md bullet and the gates of CONTRIBUTING.md
- [WIP] @session_01C7YfAQNzY28EVbk6YirCHJ-2026-11-13 01:50 #478: the real quadratic cause — scan_types on every intermediate hypergraph — filed as its own issue
