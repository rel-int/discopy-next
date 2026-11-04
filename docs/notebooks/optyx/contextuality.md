---
title: Contextuality of optical setups
marimo-version: 0.24.0
---

```python {.marimo}
import marimo as mo
```

# The contextual fraction of an optical setup

The [contextual fraction](../contextual-fraction.html) of Abramsky, Barbosa and
Mansfield measures how far an empirical model, a table of probability
distributions with one row per context, is from the marginals of one
distribution on all the measurements at once. The [previous
notebook](../contextual-fraction.html) reproduces their paper from tables
typed in by hand and from qubit circuits. This one calls the same linear
program on what `discopy.optyx` evaluates: a diagram with a classical input
choosing each party's measurement, a network of recurrent channels passing
messages, and a linear optical experiment where the fraction is read with
and without postselection.

```python {.marimo}
import numpy as np
import matplotlib.pyplot as plt

from discopy.optyx import bit, qubit, qmode, Diagram, Discard
from discopy.optyx.channel import Ty
from discopy.optyx.classical import BitControlledGate
from discopy.optyx.contextuality import (
    MessagePassingScenario, dual_rail_measurement)
from discopy.optyx.interaction import Box, CMap
from discopy.optyx.photonic import BBS, Create
from discopy.optyx.qubits import Z, H, Ket, Scalar, Measure
from discopy.quantum.contextuality import Model, ascend

import warnings

warnings.filterwarnings("ignore", message="Trial error")  # cotengra probes for optional tree-builders
```

## 1. From a diagram

`Model.from_diagram` takes a diagram with one classical input and one
classical output for each party. The input chooses the party's measurement,
here a bit controlling a phase before the qubit is measured, and the output
is its outcome. The diagram is contracted once with the setting bits left
open, so the four contexts of the Bell scenario come from one tensor.

```python {.marimo}
def bell_test(a0, a1, b0, b1):
    alice = bit @ Z(1, 1, a0) >> BitControlledGate(Z(1, 1, a1 - a0))\
        >> H() >> Measure(1)
    bob = Diagram.swap(qubit, bit) >> bit @ Z(1, 1, b0)\
        >> BitControlledGate(Z(1, 1, b1 - b0)) >> H() >> Measure(1)
    return bit @ Scalar(.5 ** .5) @ Z(0, 2) @ bit >> alice @ bob


bell_test(0, .25, -.125, .125).draw(figsize=(7, 4))
```

Phases are in turns. Alice's settings a quarter turn apart and Bob's offset
by an eighth are the Tsirelson settings, where the fraction is
$\sqrt 2 - 1$, the most any quantum model of this scenario can reach.

```python {.marimo}
tsirelson = Model.from_diagram(bell_test(0, .25, -.125, .125))
mo.md(f"""
| | |
|---|---|
| contexts | `{tsirelson.scenario.contexts}` |
| first row | `{tuple(round(p, 4) for p in tsirelson.table[0])}` |
| contextual fraction | {tsirelson.contextual_fraction():.6f} |
| $\\sqrt 2 - 1$ | {2 ** .5 - 1:.6f} |
""")
```

The dual of the linear program is a Bell inequality that the model violates
by exactly its contextual fraction, and its gradient with respect to the
table: the direction in which the statistics become contextual fastest.

```python {.marimo}
inequality = tsirelson.bell_inequality()
gradient = tsirelson.gradient()
mo.md(f"""
| | |
|---|---|
| normalised violation | {inequality.violation(tsirelson):.6f} |
| gradient against the model | {gradient.vector @ tsirelson.vector:.6f} |
| the fraction minus one | {tsirelson.contextual_fraction() - 1:.6f} |
""")
```

## 2. Learning the measurements

The gradient with respect to the table is exact, so the chain rule through
finite differences of the diagram gives the gradient with respect to its
phases. `ascend` climbs it, with a step that doubles after an improvement and
halves otherwise, since the fraction is piecewise linear in the table. It
only moves from a contextual starting point: inside the non-contextual
polytope the fraction is zero and so is its gradient.

```python {.marimo}
start = [.3, .05, .17, .42]
phases, learned = ascend(
    lambda params: Model.from_diagram(bell_test(*params)), start, steps=6)
mo.md(f"""
| | phases | fraction |
|---|---|---|
| start | `{np.round(start, 3)}` | {Model.from_diagram(bell_test(*start)).contextual_fraction():.3f} |
| after six steps | `{np.round(phases, 3)}` | {learned.contextual_fraction():.3f} |

Each party's two settings end a quarter turn apart: `{abs(phases[1] - phases[0]):.3f}` and `{abs(phases[3] - phases[2]):.3f}`.
""")
```

## 3. Message passing

A `CMap` is a network of boxes carrying channels, with edges pairing ports
that are fed back one tick later. Its `MessagePassingScenario` makes the
nodes the parties: each measures the system it predicts at the last tick with
a local measurement chosen by a bit, and since the nodes only interact through
the messages on the edges, the model is contextual only through them.

Here a source sends half of a fresh Bell pair down its edge at every tick and
predicts the half it kept a tick later, when the sink predicts the half it
received.

```python {.marimo}
source = Box("source", Ty(), qubit,
             Discard(qubit) @ qubit >> Z(0, 2) @ qubit @ Scalar(.5 ** .5),
             memory=qubit, prediction=qubit)
sink = Box("sink", qubit, Ty(), Ket(0) @ qubit, prediction=qubit)
network = CMap([source, sink], [((0, 0), (1, 0))])
network.draw(figsize=(7, 5), wire_labels=False)
```

At the first tick the predictions are independent; from the second on they
are a Bell pair. The default measurement of a qubit node takes four angles,
and `learn` finds them from the most contextual of a few random starts.

```python {.marimo}
first_tick = MessagePassingScenario(network, Ket(0) ** 3)
second_tick = MessagePassingScenario(network, Ket(0) ** 3, n_steps=1)
angles = [0, 0, .25, .25, .25, .125, 0, .25]
second_tick.diagram(angles).draw(figsize=(8, 9), wire_labels=False)
```

The scenario's diagram is the network unrolled over two ticks, the first
tick's predictions discarded, the second's each passed to its node's
measurement, whose setting bit comes in from the left. Its domain is a bit
per node and its codomain a bit per node, which is what `Model.from_diagram`
reads.

```python {.marimo}
found, best = second_tick.learn(restarts=8, steps=3)
mo.md(f"""
| | fraction |
|---|---|
| first tick, Tsirelson angles | {first_tick.model(angles).contextual_fraction():.3f} |
| second tick, Tsirelson angles | {second_tick.model(angles).contextual_fraction():.3f} |
| second tick, learned in three steps from eight random starts | {best.contextual_fraction():.3f} |
""")
```

## 4. Three nodes

The scenario is the same for any number of nodes. A hub creates a GHZ state
$(|000\rangle + |111\rangle) / \sqrt 2$ at every tick, sends one qubit down
each of its two edges and keeps the third in its memory, to predict it a tick
later, when the two leaves predict what they received.

```python {.marimo}
hub = Box("hub", Ty(), qubit ** 2,
          Discard(qubit ** 2) @ qubit >> Z(0, 3) @ qubit @ Scalar(.5 ** .5),
          memory=qubit, prediction=qubit)
leaf = Box("leaf", qubit, Ty(), Ket(0) @ qubit, prediction=qubit)
star = CMap([hub, leaf, leaf], [((0, 0), (1, 0)), ((0, 1), (2, 0))])
star.draw(figsize=(8, 5), wire_labels=False)
```

Spelling the scenario out:

- the **parties** are the three nodes, since each predicts a qubit, named
  `a`, `b` and `c` in the order of the boxes;
- the **measurements** are `a1, a2, b1, b2, c1, c2`, the two settings of
  each node's `local_measurement`, chosen by its bit;
- the **contexts** are the eight choices of one setting per node, since the
  three predictions of a tick are written at once;
- the **outcomes** are the bits the measurements return.

The memory is the four paired ports and the hub's own qubit, started in
$|0\rangle$.

```python {.marimo}
three_nodes = MessagePassingScenario(star, Ket(0) ** 5, n_steps=1)
mermin = [0, .25, .25, .25] * 3
strong = three_nodes.model(mermin)
mo.md(f"""
| | |
|---|---|
| measurements | `{strong.scenario.measurements}` |
| contexts | `{strong.scenario.contexts}` |
| outcomes per measurement | {strong.scenario.outcomes} |
""")
```

```python {.marimo}
three_nodes.diagram(mermin).draw(figsize=(9, 9), wire_labels=False)
```

The angles put each node's first setting on $Y$ and its second on $X$: a
quarter turn about $X$ turns the computational readout into $Y$, and the
second setting's further quarter turns about $Z$ then $X$ turn it into $X$.
These are Mermin's settings, and the GHZ state answers them with parities
that no global assignment can satisfy at once: the model is strongly
contextual, its fraction is one, the largest there is.

```python {.marimo}
contexts = strong.scenario.contexts
mo.md(f"""
| | |
|---|---|
| fraction at the first tick | {MessagePassingScenario(star, Ket(0) ** 5).model(mermin).contextual_fraction():.3f} |
| fraction at the second tick | {strong.contextual_fraction():.3f} |
| `{' '.join(contexts[0])}` | `{tuple(round(p, 3) for p in strong.table[0])}` |
| `{' '.join(contexts[-1])}` | `{tuple(round(p, 3) for p in strong.table[-1])}` |

Tilting the second setting halfway, a1 = 0.125 rather than 0.25, leaves the
model contextual but not strongly so: {three_nodes.model([0, .25, .125, .25] * 3).contextual_fraction():.3f}.
""")
```

## 5. Experiment: two photons at a beam splitter

Two single photons enter a beam splitter from opposite sides, with orthogonal
polarisations encoded as two modes each: Alice holds modes $a_H, a_V$ and Bob
$b_H, b_V$, the photons start in $a_H$ and $b_V$, and the splitter acts on
$a_H \leftrightarrow b_H$ and on $a_V \leftrightarrow b_V$. With probability
one half the photons leave on opposite sides, in the singlet
$(|H V\rangle - |V H\rangle) / \sqrt 2$; otherwise both bunch on one side.

```python {.marimo}
def splitter(bias=0):
    return Create(1, 0, 0, 1) >> qmode @ Diagram.swap(qmode, qmode) @ qmode\
        >> BBS(bias) @ BBS(bias) >> qmode @ Diagram.swap(qmode, qmode) @ qmode


splitter().draw(figsize=(6, 4))
```

Each party measures its pair of modes with `dual_rail_measurement`: a phase
and a beam splitter, a second pair when the setting bit is set, then two
photon counters. The outcome of a party is the pair of photon numbers it
counts, so a measurement has nine outcomes rather than two, and the table
keeps every event, including the bunched ones where one party counts two
photons and the other none.

```python {.marimo}
def experiment(angles, bias=0):
    measurements = dual_rail_measurement(angles[:4])\
        @ dual_rail_measurement(angles[4:])
    return Model.from_diagram(
        bit ** 2 @ splitter(bias) >> Diagram.permutation(
            [0, 2, 3, 1, 4, 5], bit ** 2 @ qmode ** 4) >> measurements,
        outputs=2)


def coincidences(model):
    """ The rows of a model restricted to one photon per party. """
    def keep(index):
        alice, bob = divmod(index, model.scenario.outcomes)
        return sum(divmod(alice, 3)) == 1 and sum(divmod(bob, 3)) == 1
    return Model(model.scenario, [
        [p if keep(i) else 0 for i, p in enumerate(row)]
        for row in model.table])


photonic = [0, 0, .25, 0, .125, 0, .25, 0]
(bit ** 2 @ splitter() >> Diagram.permutation(
    [0, 2, 3, 1, 4, 5], bit ** 2 @ qmode ** 4)
    >> dual_rail_measurement(photonic[:4]) @ dual_rail_measurement(photonic[4:])
).draw(figsize=(9, 9), wire_labels=False)
```

```python {.marimo}
every_event = experiment(photonic)
kept = coincidences(every_event)
conditioned = kept.normalise()
mo.md(f"""
| | fraction |
|---|---|
| every event counted | {every_event.contextual_fraction():.4f} |
| conditioned on a coincidence | {conditioned.contextual_fraction():.4f} |
| probability of a coincidence | {sum(kept.table[0]):.4f} |
""")
```

The angles put both settings of each party on a balanced beam splitter a
quarter turn of phase apart, with Bob's offset by an eighth: the Tsirelson
settings again, and conditioned on a coincidence the fraction is
$\sqrt 2 - 1$. Counting every event halves it. The bunched events are a
product, since the party that counts no photon sees nothing of the other's
setting, so they are non-contextual, and the model with every event is the
mixture of the conditioned one with weight the probability of a coincidence
and a non-contextual one with the rest. The fraction of such a mixture is the
weight times the fraction of the conditioned model.

That identity survives detuning the source. A biased splitter makes the
photons bunch less, so coincidences become more likely, but their state is no
longer the singlet and the conditioned fraction falls faster than the
probability rises.

```python {.marimo}
biases = np.linspace(0, .1, 6)
scan = [experiment(photonic, bias) for bias in biases]
probabilities = [sum(coincidences(model).table[0]) for model in scan]
conditional = [coincidences(model).normalise().contextual_fraction()
               for model in scan]
unconditional = [model.contextual_fraction() for model in scan]

fig, axis = plt.subplots(figsize=(6, 3.5))
axis.plot(biases, probabilities, color="#6c757d", linewidth=2,
          label="probability of a coincidence")
axis.plot(biases, conditional, color="#1f77b4", linewidth=2,
          label="fraction conditioned on a coincidence")
axis.plot(biases, unconditional, color="#d62728", linewidth=2,
          label="fraction with every event counted")
axis.plot(biases, np.multiply(probabilities, conditional), color="#d62728",
          linewidth=0, marker="o", markersize=6,
          label="probability times conditioned fraction")
axis.set_xlabel("bias of the source beam splitter")
axis.set_ylim(0, 1)
axis.spines[["top", "right"]].set_visible(False)
axis.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig
```

The dots sit on the red line: with every event counted, the fraction is the
product of the probability of a coincidence and the fraction conditioned on
one, at every bias. The gap between the two curves is the entangling step that
passive linear optics cannot perform deterministically, and postselection is
how it is paid for.
