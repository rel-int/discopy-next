---
title: Feed-forward photonics trained by equilibrium propagation
marimo-version: 0.24.0
---

```python {.marimo}
import marimo as mo
```

# A photonic loop with feed-forward, trained by equilibrium propagation

The chip has a unitary $U$ on $M + M$ modes. At every tick:

1. a source puts a state $\psi$ on the first $M$ inputs;
2. the loop feeds the first $M$ outputs of the last tick into the other $M$
   inputs;
3. the other $M$ outputs are counted;
4. a fast switch, a fixed function of the counts, acts on the loop modes
   before they re-enter $U$.

The switch is a measurement and feed-forward on light that is still in the
loop. Shot by shot, it is the nonlinearity of linear optics: the state of
the loop after a detection is conditioned on the counts, and the switch
acts on that conditioned state. Measurement-based schemes are universal
with one binary choice of this kind per measurement, a parity of earlier
counts deciding a rail swap or the sign of an angle (Gimeno-Segovia *et al.*,
[PRL 115, 020502](https://doi.org/10.1103/PhysRevLett.115.020502);
Bartolucci *et al.*, [Nat. Commun. 14, 912](https://doi.org/10.1038/s41467-023-36493-1)).
That needs non-classical resource states, many adaptive layers and low loss,
none of which a notebook of four modes can show.

This notebook builds the smallest such chip and trains it. The switch stays
fixed and fast. Slow electronic units set the preparation of $\psi$ at each
tick. Equilibrium propagation (EP) trains them on the averaged counts, whose
energy is an expectation value of the chip.

## The chip

$M = 2$: the fresh state is a dual-rail qubit, one photon in two modes
prepared by a tunable beam splitter, $\psi(\theta) = \sin(\pi\theta)|10\rangle
+ \cos(\pi\theta)|01\rangle$, and the loop holds two modes. $U$ is a fixed
network of Hadamard beam splitters. The switch reads the parity of the first
detector's count and swaps the two loop modes when it is odd.

```python {.marimo}
import numpy as np

from discopy.optyx import qmode, mode
from discopy.optyx.channel import Diagram, bit
from discopy.optyx.classical import BitControlledGate, ClassicalFunction, CopyN
from discopy.optyx.photonic import (
    TBS, Create, HadamardBS, NumberResolvingMeasurement, Swap)

U = qmode @ HadamardBS() @ qmode >> HadamardBS() @ HadamardBS()
parity = ClassicalFunction(lambda n: [n[0] % 2], mode, bit)
flip = BitControlledGate(Swap())


def tick(theta, switch=True):
    """One tick: a fresh qubit and the loop through U, two outputs counted,
    the switch on the two that stay in the loop."""
    fresh = Create(1) @ Create(0) >> TBS(theta)
    step = (fresh @ qmode ** 2 >> U
            >> NumberResolvingMeasurement(2) @ qmode ** 2)
    if not switch:
        return step
    return (step >> CopyN(2) @ mode @ qmode ** 2
            >> mode @ Diagram.swap(mode, mode) @ qmode ** 2
            >> mode @ mode @ parity @ qmode ** 2
            >> mode @ mode @ flip)


tick(0.2).foliation().draw(figsize=(8, 8))
```

The loop is the feedback of the tick on its two loop modes, starting with one
photon in the first.

```python {.marimo}
loop = tick(0.2).feedback(mem=qmode ** 2, state=Create(1) @ Create(0))
loop.sample(ticks=6, burn_in=0, seed=0)
```

Each tuple is one tick's counts. The sampler unravels the chip one shot at a
time: the counts are drawn, the loop collapses on them and the switch acts on
what is left.

## Exact averages

A run of $T$ ticks, with an angle $\theta_t$ for the fresh qubit of each tick,
ends by counting the loop. Its readout is the dual-rail value of the loop,

$$Q(\theta_1, \dots, \theta_T) = \langle n_0 - n_1 \rangle,$$

computed exactly from `eval` on the unrolled run.

```python {.marimo}
T = 3


def run(thetas, switch=True):
    """The unrolled run: T ticks from the loop's initial photon, then the
    loop counted."""
    result = Create(1) @ Create(0)
    for i, theta in enumerate(thetas):
        result = result >> mode ** (2 * i) @ tick(theta, switch)
    return result >> mode ** (2 * len(thetas)) @ NumberResolvingMeasurement(2)


def average(thetas, switch=True):
    """The readout Q, exactly."""
    probabilities = np.real(run(thetas, switch).eval().tensor.array)
    loop_counts = probabilities.sum(tuple(range(probabilities.ndim - 2)))
    n = np.arange(loop_counts.shape[0])
    return float((loop_counts * (n[:, None] - n[None, :])).sum())


angles = np.array([.1, .3, .2])
{"with the switch": average(angles), "without": average(angles, False)}
```

The switch changes the average: the counts it reads steer the loop.

**Shot by shot against on average.** Per shot the switch is a nonlinear map
of the loop's state, a projection followed by a conditional swap. On average
the run is a quantum channel, linear in the state of each fresh qubit. Each
fresh qubit enters only through $\cos 2\pi\theta_t$ and $\sin 2\pi\theta_t$,
so $Q$ is a trigonometric polynomial of degree one in every angle. It is
fixed by its values on a grid of $3^T$ angles, and so are its gradients.

```python {.marimo}
import itertools

GRID = (0., .25, .125)


def basis(theta):
    theta = np.asarray(theta)
    return np.stack([np.ones_like(theta), np.cos(2 * np.pi * theta),
                     np.sin(2 * np.pi * theta)], -1)


def coefficients(switch=True):
    """The coefficients of Q in the basis (1, cos, sin) of every angle."""
    values = np.zeros((3, ) * T)
    for index in itertools.product(range(3), repeat=T):
        values[index] = average([GRID[i] for i in index], switch)
    inverse = np.linalg.inv(basis(np.array(GRID)))
    for axis in range(T):
        values = np.moveaxis(np.tensordot(
            inverse, np.moveaxis(values, axis, 0), axes=(1, 0)), 0, axis)
    return values


CHIPS = {"with the switch": coefficients(), "without": coefficients(False)}


def readout(c, thetas):
    """Q for a batch of angles, of shape (batch, T)."""
    b = basis(thetas)
    return np.einsum("ijk,ni,nj,nk->n", c, b[:, 0], b[:, 1], b[:, 2])


{name: (readout(c, angles[None])[0], average(angles, name == "with the switch"))
 for name, c in CHIPS.items()}
```

The closed form agrees with `eval`.

**The parameter-shift rule.** A chip measures the gradient in each angle from
two settings of that angle alone:

$$\frac{\partial Q}{\partial \theta_t} = \sqrt 2\, \pi
\left[Q(\theta_t + \tfrac18) - Q(\theta_t - \tfrac18)\right].$$

It is exact because each angle acts on a single photon.

```python {.marimo}
def shifted(c, thetas):
    """The gradient of Q by parameter shifts, for a batch of angles."""
    eye = np.eye(T) / 8
    return np.stack([np.sqrt(2) * np.pi * (readout(c, thetas + e)
                                            - readout(c, thetas - e))
                     for e in eye], -1)


step = 1e-6
c_switch = CHIPS["with the switch"]
finite = np.array([(readout(c_switch, (angles + step * e)[None])
                    - readout(c_switch, (angles - step * e)[None]))[0]
                   / (2 * step) for e in np.eye(T)])
np.abs(shifted(c_switch, angles[None])[0] - finite).max()
```

## Slow units and their energy

Each tick's angle is set by a slow electronic unit with state $r_t \in (-1, 1)$,
$\theta_t = (1 + r_t)/4$. The units integrate their input $x$ and the chip:

$$z \leftarrow (1 - a)\, z + a\left(\lambda\, \partial_r Q + A x + b\right),
\qquad r = \tanh z.$$

Their fixed points are the critical points of the energy

$$E(r) = \sum_t \left(r_t\, \mathrm{artanh}\, r_t + \tfrac12 \ln(1 - r_t^2)\right)
- \lambda\, Q(\theta(r)) - (A x + b)^\top r,$$

a Hopfield energy whose quadratic term is replaced by an expectation value of
the chip. The units' input $\partial_r Q$ comes from the parameter shifts. The
output is $y = w\, Q + d$.

**EP.** The cost is $C = \tfrac12 (y - y^*)^2$. Nudging the units by
$-\beta\, \partial_r C$ gives two equilibria $r_\pm$, and

$$\partial_{(A, b)} J \approx -\frac{1}{2\beta}\,(r_+ - r_-)\,(x, 1), \qquad
\partial_\lambda J \approx -\frac{1}{2\beta}\,(Q_+ - Q_-),$$

local to the units. The readout $(w, d)$ has its exact gradient at the free
equilibrium.

```python {.marimo}
LEAK = .5


class Units:
    """The slow units on one chip, their parameters and their EP estimate."""

    def __init__(self, c, seed=0):
        rng = np.random.default_rng(seed)
        self.c = c
        self.weights = rng.normal(0, 1, T)
        self.offsets = rng.normal(0, .5, T)
        self.coupling, self.scale, self.shift = 1., 1., 0.

    def settle(self, x, target=None, beta=0., steps=3000, tol=1e-12):
        """The equilibria for a batch of inputs, nudged by beta."""
        z = np.zeros((len(x), T))
        for _ in range(steps):
            thetas = (1 + np.tanh(z)) / 4
            gradient = shifted(self.c, thetas) / 4
            force = (self.coupling * gradient + np.outer(x, self.weights)
                     + self.offsets)
            if beta:
                error = self.scale * readout(self.c, thetas) + self.shift - target
                force -= beta * (error * self.scale)[:, None] * gradient
            new = (1 - LEAK) * z + LEAK * force
            done = np.abs(new - z).max() < tol
            z = new
            if done:
                break
        return np.tanh(z)

    def output(self, r):
        return self.scale * readout(self.c, (1 + r) / 4) + self.shift

    def loss(self, x, y):
        return float(np.mean((self.output(self.settle(x)) - y) ** 2) / 2)

    def parameters(self):
        return np.r_[self.weights, self.offsets, self.coupling, self.scale,
                     self.shift]

    def assign(self, p):
        self.weights, self.offsets = p[:T], p[T:2 * T]
        self.coupling, self.scale, self.shift = p[2 * T:]

    def gradient(self, x, y, beta=1e-4):
        """The EP estimate of the gradient of the mean cost."""
        r = self.settle(x)
        q = readout(self.c, (1 + r) / 4)
        error = self.scale * q + self.shift - y
        plus, minus = self.settle(x, y, beta), self.settle(x, y, -beta)
        difference = (plus - minus) / (2 * beta)
        units = -(readout(self.c, (1 + plus) / 4)
                  - readout(self.c, (1 + minus) / 4)) / (2 * beta)
        return np.r_[-(difference * x[:, None]).mean(0),
                     -difference.mean(0), units.mean(),
                     (error * q).mean(), error.mean()]
```

EP against finite differences of the loss, on every parameter:

```python {.marimo}
xs = np.linspace(-1, 1, 9)
check = Units(CHIPS["with the switch"], seed=3)
target = .3 * np.sin(np.pi * xs)
estimate = check.gradient(xs, target)
point, h = check.parameters(), 1e-6
differences = []
for e in np.eye(len(point)):
    check.assign(point + h * e)
    up = check.loss(xs, target)
    check.assign(point - h * e)
    down = check.loss(xs, target)
    differences.append((up - down) / (2 * h))
check.assign(point)
differences = np.array(differences)
estimate @ differences / np.linalg.norm(estimate) / np.linalg.norm(differences)
```

## Training

Three targets on $x \in [-1, 1]$, nine training points, the error on forty
test points, Adam with a decaying rate, the same seed for both chips.

```python {.marimo}
TASKS = {"linear": lambda x: .3 * x,
         "square": lambda x: .6 * x ** 2 - .3,
         "sine": lambda x: .3 * np.sin(np.pi * x)}
test = np.linspace(-.95, .95, 40)


def train(c, task, steps=100, rate=.05, seed=0):
    units = Units(c, seed)
    target = TASKS[task](xs)
    p = units.parameters()
    first, second = np.zeros_like(p), np.zeros_like(p)
    for t in range(1, steps + 1):
        g = units.gradient(xs, target)
        first = .9 * first + .1 * g
        second = .999 * second + .001 * g * g
        p = p - rate * (1 - t / (steps + 1)) * (first / (1 - .9 ** t)) / (
            np.sqrt(second / (1 - .999 ** t)) + 1e-8)
        units.assign(p)
    return units.loss(test, TASKS[task](test))


errors = {(task, name): train(c, task) for task in TASKS
          for name, c in CHIPS.items()}
errors
```

## What this shows

- **The stack runs end to end.** The chip is an optyx diagram with feedback,
  a classical function and a bit-controlled gate. It is sampled shot by shot
  and evaluated exactly, unrolled. Slow units read it by parameter shifts, and
  EP trains them with the gradient of the loss.
- **The fast nonlinearity is fixed, the slow parameters are learnt.** EP never
  touches the switch. The program of the chip, the angle of each tick, is set
  by the units.
- **What it does not show.** Universality, and any classical hardness of $Q$,
  need non-classical resource states, many adaptive layers and loss below
  threshold. With one photon per tick and three ticks, $Q$ is a small
  trigonometric polynomial that this notebook computes exactly. The averaged
  run is linear in each fresh state: the switch makes it a different channel,
  not a nonlinear one, and the nonlinearity of the trained map comes from the
  units.
