# TODO

🌙 Evening's prompt for the night of 2026-10-30, verbatim:

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

[`EVENING.md`](https://github.com/daydream6728/desire/blob/main/EVENING.md) sends it at the next
repo's bugs that need no decision, oldest first and without asking. These five are boxes of
[#314](https://github.com/rel-int/discopy-next/issues/314), all in `discopy/optyx/compiler/`, all
reproduced on `main` `13d68a38`, and each fix is the one its own module already states.

- [ ] `mbqc.Fusion.__eq__` reads `other.fusion_type` before knowing it is a `Fusion`, so
      `Fusion(0, 1, "X") == None` raises `AttributeError`
- [ ] `path_cover.find_min_path_cover` searches 1 to n-1 paths, so a graph with one node, an
      edgeless graph and the empty graph all raise `ValueError("unreachable!")`, where its own
      comment says the worst case is one path per vertex
- [ ] `semm_decompiler`: an instruction before the first `next_node` raises `IndexError` rather
      than the `ValidationError` the machine states for an invalid stream
- [ ] `semm_decompiler`: a node measured twice through a delay goes unnoticed, since
      `__node_has_been_measured` scans only the times up to now and a delayed measurement is
      recorded in the future
- [ ] `semm_decompiler`: a delayed fusion no photon comes back for is dropped silently, since
      `fuse` never consumes the entry it reads and nothing reads what is left
