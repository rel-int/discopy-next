# TODO

> Perfect. Start this, and use a swarm of sonnet subagents to help you step by step. You are the orchestrator. At every step, where the code should run on GPUs, have an opus subagent decide speedups, implementations, etc. Work independently for a very long time, make many tests, make sure that what you do is correct.

- [x] `Kronecker`: gather the weights only when the tie is not the identity, plan the free axes of the terms once, sum them with `unbind`
- [x] `Partitions`: float `mobius`, einsums planned once, partitions sharing an output pattern summed before spreading
- [x] outputs and gradients bitwise equal to the previous slice, tests of the buffers
- [x] H100 acceptance (Opus, ACCEPT_fast_perf_v2): eager and compiled at or below the previous slice beyond noise
