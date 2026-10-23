---
title: Contextual Fraction
marimo-version: 0.23.14
---

```python {.marimo}
import marimo as mo
```

# The contextual fraction

This notebook reproduces the examples of Abramsky, Barbosa and Mansfield, [*The contextual fraction as a measure of contextuality*](https://arxiv.org/abs/1705.07918) (PRL 119, 050504, 2017), with the module
[`quantum.contextuality`](https://docs.discopy.org/en/main/_api/discopy.quantum.contextuality.html).

## Empirical models

A measurement scenario is a set of measurements covered by contexts, the sets of measurements that can be performed together, each measurement with a set of outcomes.
In the (2, 2, 2) Bell scenario, Alice and Bob each choose one of two measurements, `a1` or `a2` and `b1` or `b2`, each with two outcomes.

```python {.marimo}
from fractions import Fraction
from itertools import product
from math import pi, sqrt

import matplotlib.pyplot as plt
import numpy as np

from discopy.quantum import CX, H, Ket, Measure, Rz, qubit
from discopy.quantum.contextuality import Model, Scenario

bell = Scenario.bell(parties=2, settings=2, outcomes=2)
bell.contexts
```

An empirical model is a table with a probability distribution for each context, on the joint outcomes of its measurements.
We print them as in the paper, with fractions where the probabilities are rational.

```python {.marimo}
def show(probability):
    fraction = Fraction(probability).limit_denominator(64)
    return str(fraction) if abs(fraction - probability) < 1e-9 else (
        f"{probability:.4f}")


def table(model):
    contexts, outcomes = model.scenario.contexts, model.scenario.outcomes
    sections = product(range(outcomes), repeat=len(contexts[0]))
    return "\n".join(
        ["| context | " + " | ".join(
            "".join(map(str, s)) for s in sections) + " |",
         "|---" * (1 + outcomes ** len(contexts[0])) + "|"]
        + [f"| {' '.join(context)} | " + " | ".join(map(show, row)) + " |"
           for context, row in zip(contexts, model.table)])
```

The Popescu–Rohrlich box of Table I is typed in row by row.

```python {.marimo}
PR = Model(bell, [[1 / 2, 0, 0, 1 / 2]] * 3 + [[0, 1 / 2, 1 / 2, 0]])
mo.md(table(PR))
```

The Bell–CHSH model of Table I comes from local measurements on the Bell state $|\Phi^+\rangle = (|00\rangle + |11\rangle) / \sqrt 2$.
A measurement at angle $\phi$ on the equator of the Bloch sphere rotates the qubit by $-\phi$ around the Z axis, then measures it in the X basis.
Rotations in DisCoPy take their angle in turns, hence the $2\pi$.
Each context is a circuit, here `a1` at angle $0$ and `b2` at angle $\pi / 3$:

```python {.marimo}
def measure(phi):
    return Rz(-phi / (2 * pi)) >> H >> Measure()


def GHZ(n):
    circuit = Ket(*n * [0]) >> H @ qubit ** (n - 1)
    for i in range(n - 1):
        circuit = circuit >> qubit ** i @ CX @ qubit ** (n - i - 2)
    return circuit


(GHZ(2) >> measure(0) @ measure(pi / 3)).foliation()
```

`Model.from_circuits` evaluates the state and each measurement as a channel, then composes them in each context.

```python {.marimo}
CHSH = Model.from_circuits(GHZ(2), 2 * [[measure(0), measure(pi / 3)]])
mo.md(table(CHSH))
```

## The contextual fraction

A model is non-contextual when its distributions are the marginals of one distribution on the global assignments of outcomes to all the measurements.
The non-contextual fraction is the largest weight of a subdistribution on global assignments whose marginals are bounded by the model, a linear program.
The contextual fraction is one minus that: $0$ for non-contextual models and $1$ for strongly contextual ones, like the PR box.

```python {.marimo}
mo.md(f"""
| model | contextual fraction |
|---|---|
| PR box | {PR.contextual_fraction():.4f} |
| Bell–CHSH | {CHSH.contextual_fraction():.4f} |
""")
```

## Figure 1: equatorial measurements

Each party measures its qubit of a GHZ state at one of the same two angles $\phi_1$ and $\phi_2$, with $|\psi_{GHZ(n)}\rangle = (|0\rangle^{\otimes n} + |1\rangle^{\otimes n}) / \sqrt 2$ and the Bell state for $n = 2$.
Angles $\phi$ and $\phi + \pi$ give the same measurement up to relabelling the outcomes.

```python {.marimo}
def equatorial(n, phi1, phi2):
    return Model.from_circuits(GHZ(n), n * [[measure(phi1), measure(phi2)]])


angles = np.linspace(0, pi, 25)
fractions = {n: np.array([
    [equatorial(n, phi1, phi2).contextual_fraction() for phi2 in angles]
    for phi1 in angles]) for n in (2, 3, 4)}
```

```python {.marimo}
figure = plt.figure(figsize=(12, 4))
_x, _y = np.meshgrid(angles, angles, indexing="ij")
for _i, (_n, _z) in enumerate(fractions.items()):
    _axes = figure.add_subplot(1, 3, _i + 1, projection="3d")
    _axes.plot_surface(_x, _y, _z, cmap="viridis")
    _axes.set(xlabel="$\\phi_1$", ylabel="$\\phi_2$", zlabel="CF",
              title="Bell state" if _n == 2 else f"GHZ({_n})")
figure
```

The Bell–CHSH model at $(0, \pi / 3)$ is not the most contextual.
The maxima of the plot for the Bell state are at $\{\pi / 8, 5 \pi / 8\}$ and $\{7 \pi / 8, 3 \pi / 8\}$, both the model of Table III with $p = (\sqrt 2 + 2) / 8$, which reaches the Tsirelson bound of the CHSH inequality, with contextual fraction $\sqrt 2 - 1$.
At $(0, \pi / 2)$, a shift of the maxima by $\pi / 8$, the model is non-contextual.

```python {.marimo}
tsirelson = equatorial(2, pi / 8, 5 * pi / 8)
mo.md(f"""
{table(tsirelson)}

| angles | contextual fraction |
|---|---|
| (π/8, 5π/8) | {tsirelson.contextual_fraction():.4f} |
| (7π/8, 3π/8) | {equatorial(2, 7 * pi / 8, 3 * pi / 8).contextual_fraction():.4f} |
| (0, π/2) | {abs(equatorial(2, 0, pi / 2).contextual_fraction()):.4f} |
| √2 − 1 | {sqrt(2) - 1:.4f} |
""")
```

For GHZ states, the maxima are strongly contextual: equations (14) and (15) list them for three and four qubits.
Proposition 4 generalises them: measuring at $((n + k) \pi / 2n, k \pi / 2n)$ gives, up to relabelling, the GHZ–Mermin model, strongly contextual for every $0 \leq k < n$.

```python {.marimo}
def _angle(fraction):
    numerator = {0: "0", 1: "π"}.get(
        fraction.numerator, f"{fraction.numerator}π")
    return numerator if fraction.denominator == 1 else (
        f"{numerator}/{fraction.denominator}")


_rows = [
    (n, k, Fraction(n + k, 2 * n), Fraction(k, 2 * n))
    for n in (3, 4, 5) for k in range(n)]
mo.md("| n | k | (φ₁, φ₂) | contextual fraction |\n|---|---|---|---|\n"
      + "\n".join(
          f"| {n} | {k} | ({_angle(phi1)}, {_angle(phi2)}) | "
          f"{equatorial(n, phi1 * pi, phi2 * pi).contextual_fraction():.4f} |"
          for n, k, phi1, phi2 in _rows))
```

## Table II: decompositions are not unique

The optimal subdistribution splits a model $e = \mathsf{NCF}(e) e^{NC} + \mathsf{CF}(e) e^{SC}$ into a non-contextual model and a strongly contextual one, but not uniquely.
Table II gives two such decompositions of the same model in the (3, 2, 2) Bell scenario, with entries in sixteenths:

```python {.marimo}
tripartite = Scenario.bell(parties=3)
e1_SC, e1_NC, e2_SC, e2_NC = (
    Model(tripartite, np.array(rows) / 16) for rows in [
        [[0, 4, 4, 0, 4, 0, 0, 4], [2] * 8, [2] * 8, [4, 0, 0, 4, 0, 4, 4, 0],
         [2] * 8, [4, 0, 0, 4, 0, 4, 4, 0], [4, 0, 0, 4, 0, 4, 4, 0], [2] * 8],
        [[1, 3, 3, 1, 3, 1, 1, 3], [2] * 8, [2] * 8, [3, 1, 1, 3, 1, 3, 3, 1],
         [2] * 8, [3, 1, 1, 3, 1, 3, 3, 1], [3, 1, 1, 3, 1, 3, 3, 1], [2] * 8],
        [[0, 2, 2, 0, 6, 0, 0, 6], [0, 2, 2, 0, 2, 4, 4, 2],
         [0, 2, 2, 0, 2, 4, 4, 2], [2, 0, 0, 2, 0, 6, 6, 0],
         [2, 0, 0, 2, 4, 2, 2, 4], [2, 0, 0, 2, 0, 6, 6, 0],
         [2, 0, 0, 2, 0, 6, 6, 0], [0, 2, 2, 0, 2, 4, 4, 2]],
        [[1, 5, 5, 1, 1, 1, 1, 1], [4, 2, 2, 4, 2, 0, 0, 2],
         [4, 2, 2, 4, 2, 0, 0, 2], [5, 1, 1, 5, 1, 1, 1, 1],
         [2, 4, 4, 2, 0, 2, 2, 0], [5, 1, 1, 5, 1, 1, 1, 1],
         [5, 1, 1, 5, 1, 1, 1, 1], [4, 2, 2, 4, 2, 0, 0, 2]]])
mixture = e1_NC.mix(e1_SC, 1 / 2)
assert mixture == e2_NC.mix(e2_SC, 1 / 2)
mo.md(f"""
| model | contextual fraction |
|---|---|
| $e_1^{{SC}}$ | {e1_SC.contextual_fraction():.4f} |
| $e_1^{{NC}}$ | {abs(e1_NC.contextual_fraction()):.4f} |
| $e_2^{{SC}}$ | {e2_SC.contextual_fraction():.4f} |
| $e_2^{{NC}}$ | {abs(e2_NC.contextual_fraction()):.4f} |
| $\\frac 1 2 e_1^{{NC}} + \\frac 1 2 e_1^{{SC}} = \\frac 1 2 e_2^{{NC}} + \\frac 1 2 e_2^{{SC}}$ | {mixture.contextual_fraction():.4f} |
""")
```

`Model.decompose` returns the weight of the non-contextual part and the two models of the decomposition found by the linear program, here for the Bell–CHSH model:

```python {.marimo}
weight, NC, SC = CHSH.decompose()
assert NC.mix(SC, weight).is_close(CHSH)
mo.md(f"""
$\\mathsf{{NCF}} = {weight:.4f}$, where $e^{{NC}}$ is:

{table(NC)}

and $e^{{SC}}$ is:

{table(SC)}
""")
```

## Theorem 1: Bell inequalities

The dual linear program gives a Bell inequality $a \cdot v^e \leq 0$, one coefficient for each row and outcome, which every non-contextual model satisfies.
Its normalised violation by the model is exactly the contextual fraction.
It is tight at $e^{NC}$ and maximally violated at $e^{SC}$.

```python {.marimo}
inequality = CHSH.bell_inequality()
mo.md(f"""
{table(inequality)}

| | value |
|---|---|
| non-contextual bound | {inequality.non_contextual_bound:.4f} |
| algebraic bound | {inequality.algebraic_bound:.4f} |
| normalised violation by $e$ | {inequality.violation(CHSH):.4f} |
| value at $e^{{NC}}$ | {abs(inequality(NC)):.4f} |
| value at $e^{{SC}}$ | {inequality(SC):.4f} |
""")
```

## Theorem 2: operations on models

The contextual fraction is invariant under relabelling and behaves well under the operations that combine models:
it is convex under mixing, the controlled choice `&` takes the larger of the two, and the product `@` of independent models adds up to $\mathsf{CF}(e) + \mathsf{CF}(e') - \mathsf{CF}(e)\mathsf{CF}(e')$.
The measurements of the two models must be disjoint, so we rename those of the second one.

```python {.marimo}
other = PR.mix(Model(bell, [[1 / 4] * 4] * 4), 3 / 4).relabel(
    {'a1': 'c1', 'a2': 'c2', 'b1': 'd1', 'b2': 'd2'})
_cf, _cf_other = CHSH.contextual_fraction(), other.contextual_fraction()
mo.md(f"""
| model | contextual fraction | expected |
|---|---|---|
| $e$ | {_cf:.4f} | |
| $e'$ | {_cf_other:.4f} | |
| $e \\,\\&\\, e'$ | {(CHSH & other).contextual_fraction():.4f} | {max(_cf, _cf_other):.4f} |
| $e \\otimes e'$ | {(CHSH @ other).contextual_fraction():.4f} | {_cf + _cf_other - _cf * _cf_other:.4f} |
| $\\frac 1 2 e + \\frac 1 2 \\mathrm{{PR}}$ | {CHSH.mix(PR, 1 / 2).contextual_fraction():.4f} | at most {(_cf + 1) / 2:.4f} |
""")
```
