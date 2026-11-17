#import "../template.typ": drawing, gap

== Photonic and quantum computing <sec:quantum>

The circuits and the ZX calculus of @ToumiEtAl22 are still the module
`discopy.quantum`, which gained reservoirs, an ansatz for neutral atoms,
a simplifier of ZX diagrams and a measure of contextuality. Beside it,
the library gained `discopy.optyx`.

=== Optyx <sec:optyx>

The subpackage `discopy.optyx` is optyx, software for networked quantum
computing based on the ZX calculus. It moved in from its own repository
with its history, and stays under the Apache License 2.0 with its own
`LICENSE` and `NOTICE`, the rest being under the BSD 3-Clause License.

An optyx diagram has four kinds of wires: `qubit` and `qmode`, a qubit
and a bosonic mode, and their classical counterparts `bit` and `mode`.
Each box is a `Channel`, given by its Kraus map into its codomain and a
discarded environment, a ZX diagram on qubits and a ZW diagram on modes.
`double` puts the Kraus maps beside their conjugates @CoeckeKissinger17,
which evaluation contracts as a tensor network. The boxes are linear
optics with photon loss and photon counting, the dual-rail encoding of a
qubit in two modes, fusion measurements, gates on bits, arithmetic on
modes and the control of a channel by a bit.

Two photons that meet on a balanced beam splitter leave together, the
Hong-Ou-Mandel effect. If one of them survives with probability 0.8
only, a single photon is counted with probability 0.2.

```python
from discopy.optyx import qmode
from discopy.optyx.photonic import (
    BS, MZI, Create, PhotonLoss, NumberResolvingMeasurement)

dip = (Create(1, 1) >> BS).eval().prob_dist()
assert round(dip[2, 0], 9) == .5 and round(dip[1, 1], 9) == 0
lossy = Create(1, 1) >> PhotonLoss(.8) @ qmode >> BS
counts = (lossy >> NumberResolvingMeasurement(2)).eval().prob_dist()
assert round(counts[1, 0] + counts[0, 1], 9) == .2

loop = (Create(1) @ qmode >> MZI(.06, 0)).feedback(
    mem=qmode, state=Create(0))
assert loop.sample(ticks=4, seed=0) == [(2,), (1,), (1,), (1,)]
```

`eval` takes a backend. The default contracts the doubled diagram with
quimb along a path that cotengra optimises, `DiscopyBackend` with the
tensor functor of @sec:tensors, and for linear optics `PermanentBackend`
computes permanents and `PercevalBackend` calls Perceval. The four agree
on the first diagram of the listing.

The last lines close a loop: `feedback` plugs the last outputs of a
diagram into its last inputs one time step later, as in a monoidal
stream @DiLavoreEtAl22. Here a photon is emitted at every tick and meets
on an interferometer the mode that left it one tick before. `unroll`
composes a number of steps, `eigen_fix` computes the stationary state as
an eigenvector of the transfer matrix of a step @BiriukovDyakonov26, and
`sample` runs the diagram tick after tick: linear optics with loss as a
pure state of the photons in the loop, any other diagram as a quantum
trajectory, one box at a time.

Qubits, modes and bits meet in a fusion. Two Bell pairs each give one
qubit, encoded in two modes, to a type II fusion, which measures the
four modes and returns two bits. Given the bits of a success with no
correction, of probability one quarter, the two qubits left are a Bell
pair (@fig:quantum_fusion).

```python
from discopy.optyx import qubit, Discard
from discopy.optyx.photonic import DualRail, FusionTypeII
from discopy.optyx.qubits import Z, Scalar
from discopy.optyx.classical import PostselectBit

pair, herald = Z(0, 2), PostselectBit(1) @ PostselectBit(0)
quantum_fusion = pair @ pair >> qubit @ (
    DualRail(2) >> FusionTypeII() >> herald) @ qubit
link, bell = Scalar(.5) @ quantum_fusion, Scalar(.5 ** .5) @ pair
success = (link >> Discard(qubit ** 2)).eval().tensor.array
overlap = (link >> bell.dagger()).eval().tensor.array
assert abs(success - .25) < 1e-9 and abs(overlap - .25) < 1e-9
```

#drawing("quantum_fusion", caption: [
  Two pairs of qubits and the fusion of one qubit of each: its optics,
  its detectors, the function of their counts and the postselection of
  its two bits. The scalars are left out.
]) <fig:quantum_fusion>

`optyx.compiler` compiles a measurement pattern, an open graph with a
generalised flow, to a linear resource state with the fusions that give
the edges it lacks, then to the instructions of a machine with one
emitter of photons. Diagrams of qubits convert to and from PyZX, and
diagrams are read from Perceval processors and graphix patterns.

#gap[
  The README of optyx asks for no citation. The documentation of
  `optyx.core` names the papers on the ZXW calculus and on linear optics
  that it implements with no identifier to copy: only the fusion
  protocols of @FeliceEtAl24 have one.
]

=== Circuits, reservoirs and Rydberg atoms <sec:circuits>

A quantum reservoir @FujiiNakajima17 is a fixed unitary on memory qubits
and input qubits. At each step of a time series,
`quantum.reservoir.Reservoir` encodes the value in the input qubits,
applies the unitary and discards them. The features are the
probabilities of the outcomes of the memory after each step, and only a
linear readout is trained, by ridge regression. The unitary can be an
array of neutral atoms @BravoEtAl22: `quantum.ansatze.Rydberg` cuts
their evolution into Trotter steps, each one a rotation for the drive of
each atom, a phase for its detuning and a controlled phase for each
pair. The listing fits the readout of three atoms, two of them the
memory, to recall the previous value of a random series on its first
twenty values. On the next twenty, the mean squared error is under half
the variance of the targets.

```python
from random import Random
import numpy as np
from discopy.quantum.ansatze import Rydberg
from discopy.quantum.reservoir import Reservoir

atoms = Rydberg([(0, 0), (0, 1), (1, 0)], 4, 4, 2, steps=2)
reservoir, rng = Reservoir(memory=2, inputs=1, unitary=atoms), Random(0)
series = [rng.random() for _ in range(40)]
targets = np.array([0.] + series[:-1])
readout = reservoir.fit(series[:20], targets[:20])
errors = reservoir.predict(series, readout)[:, 0] - targets
assert (errors[20:] ** 2).mean() < targets[20:].var() / 2
```

A circuit evaluates with the contractors of @sec:tensors, and the ansatz
converts to PennyLane, where it evaluates to the same state, and to
tket.

`quantum.graphlike` simplifies ZX diagrams by the rewriting of
@sec:rewriting. A ZX diagram is graph-like when its spiders are green
and joined by Hadamard edges only @DuncanEtAl20. As a hypergraph, the
green spiders are the spiders, so that fusing two of them is the pushout
itself, a Hadamard edge is a box and a phase is an effect on its spider.
The rules are those of the `full_reduce` of PyZX @KissingerWetering20,
each one built for the match at hand and applied in place by
`hypergraph.Rewriter`, up to a scalar. In the listing, two rotations by
an eighth of a turn on the parity of two qubits fuse into a Clifford
phase: eight spiders, ten Hadamard edges and two phases that are not
Clifford become three, three and none, with the same tensor up to a
scalar.

```python
from discopy.quantum import CX, Id, T, graphlike
from discopy.quantum.zx import circuit2zx

parity = CX >> Id(1) @ T >> CX
gadgets = circuit2zx(parity >> parity)
graph = graphlike.from_diagram(gadgets)
reduced = graphlike.full_reduce(graph)
assert graphlike.size(graph) == (8, 10, 2)
assert graphlike.size(reduced) == (3, 3, 0)
before, after = gadgets.eval().array, reduced.to_diagram().eval().array
assert np.isclose(abs(np.vdot(before, after)),
                  np.linalg.norm(before) * np.linalg.norm(after))
```

The benchmark of @sec:benchmarks races the simplifier against PyZX on
random circuits of CNOT, Hadamard and T gates. On 10 and 20 qubits with
100 to 1000 gates, the two reach the same count of phases that are not
Clifford.

#gap[
  The timings of the two simplifiers, from a recorded run of the
  benchmark and not from the machine the paper was drafted on.
]

=== Example: contextuality as a resource <sec:contextuality>

A measurement scenario is a set of measurements covered by contexts, the
sets that can be measured together. An empirical model gives a
distribution on the joint outcomes of each context, and is
non-contextual when these are the marginals of one distribution on the
outcomes of all the measurements. The contextual fraction
@AbramskyEtAl17 is one minus the largest weight of a non-contextual part
of the model, a linear program, whose dual is a Bell inequality that the
model violates, once normalised, by exactly its fraction.

`quantum.contextuality.Model.from_diagram` reads the model of an optyx
diagram with a classical input and output for each party, the setting
and the outcome of its measurement (@fig:quantum_bell). In the listing,
a bit chooses between two phases on each half of the `pair` above before
it is measured: none or a quarter turn on one side, an eighth of a turn
either way on the other. At these settings the fraction is the square
root of two minus one, the Tsirelson bound.

```python
from discopy.optyx import bit, qubit, Diagram
from discopy.optyx.qubits import Z, H, Scalar, Measure
from discopy.optyx.classical import BitControlledGate
from discopy.quantum.contextuality import Model

alice = BitControlledGate(Z(1, 1, .25), Z(1, 1)) >> H() >> Measure(1)
bob = Diagram.swap(qubit, bit) >> BitControlledGate(
    Z(1, 1, .125), Z(1, 1, -.125)) >> H() >> Measure(1)
quantum_bell = bit @ pair @ bit >> alice @ bob
model = Model.from_diagram(Scalar(.5 ** .5) @ quantum_bell)
fraction = model.contextual_fraction()
assert abs(fraction - (2 ** .5 - 1)) < 1e-6
assert abs(model.bell_inequality().violation(model) - fraction) < 1e-6
```

#drawing("quantum_bell", caption: [
  A Bell test as one diagram: a bit for the setting of each party, a
  pair of qubits and a bit for each outcome.
]) <fig:quantum_bell>

Two notebooks of the documentation go on from here. The first follows
@AbramskyEtAl17 on tables and circuits: the Bell state measured at
angles zero and a sixth of a turn has fraction one quarter, the GHZ
state of three qubits at the settings of Mermin has fraction one. The
second treats the fraction as what a setup is optimised for: the
solution of the dual program is its gradient in the table, and `ascend`
climbs it through the phases of the diagram, from a fraction of 0.17 to
the bound. In a network of channels where a source sends half of a Bell
pair at every tick, the fraction is zero at the first tick and at the
bound at the second. When two photons of orthogonal polarisations meet
on a beam splitter, it is at the bound given a coincidence, of
probability one half, and half of that with every event counted, since
the events where the photons bunch are non-contextual.
