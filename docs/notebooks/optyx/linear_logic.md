---
title: Linear Logic
marimo-version: 0.24.0
---

```python {.marimo}
import marimo as mo
```

# From a linear term with bangs to an optical protocol

A diagram of `discopy.bang`, the free linear category, uses each wire once
unless its type is banged: `!x` may be copied by `Contraction`, discarded by
`Weakening` and read once by `Dereliction`. Over quantum states, `!` is the
bosonic Fock space: `!q` is the Fock space of the two modes of a dual-rail
qubit `q`, and a functor into `discopy.optyx` sends:

- the bang `f.bang()` of a single-photon unitary `U` to the linear-optical
  gate acting on every photon at once, its second quantisation,
- `Contraction` to the W node of ZW on each mode, the bosonic comonoid that
  shares the photons of a mode between two copies,
- `Weakening` to post-selecting the vacuum, and `Dereliction` to heralding
  exactly one photon, i.e. reading the copy back as a dual-rail qubit.

The digging `!x -> !!x` and the laxator `!x @ !y -> !(x @ y)` have no such
image: the Fock space of a Fock space is not a set of modes, and
`!x @ !y` is the Fock space of `x ⊕ y`, not of `x @ y`. So the functor is
defined on the fragment of terms that do not use them.

```python {.marimo}
import numpy as np

from discopy import bang
from discopy.bang import (
    Ty, Box, Bang, Contraction, Dereliction, Weakening)
from discopy.optyx import channel, photonic, qmode, qubit
from discopy.optyx.core import diagram, zw
```

## The term

The non-linear term `λx. (H x, H x)` uses its argument twice. Its translation
into linear logic gives `x` the type `!q`, applies the bang of `H` and
contracts the result into two copies, each one derelicted. The function
syntax of diagrams writes it as a term, with one variable per wire.

```python {.marimo}
q = Ty("q")
H = Box("H", q, q, data=np.array([[1, 1], [1, -1]]) / 2 ** .5)


@bang.Diagram.from_callable(q.bang(), q @ q)
def proof(x):
    y, z = Contraction(q)(H.bang()(x))
    return Dereliction(q)(y), Dereliction(q)(z)


proof.draw(figsize=(3, 4))
```

## The functor

`Gate` reads its matrix from its inputs to its outputs, i.e. it takes the
transpose of the unitary on one photon.

```python {.marimo}
def to_optics(box):
    "The image of a box of the fragment without digging and laxator."
    if isinstance(box, Bang):
        return photonic.Gate(box.arg.data.T, 2, 2, f"!{box.arg.name}")
    if isinstance(box, Contraction):
        return channel.Channel(
            "Δ", zw.W(2) @ zw.W(2) >> diagram.mode @ zw.SWAP @ diagram.mode)
    if isinstance(box, Weakening):
        return channel.Channel("e", zw.Select(0, 0))
    if isinstance(box, Dereliction):
        return channel.Channel("ε", diagram.DualRail(is_dagger=True))
    raise NotImplementedError(box)


F = bang.Functor(
    ob_map={q: qubit, q.bang(): qmode @ qmode},
    ar_map=to_optics, cod=channel.Diagram)


def amplitudes(protocol):
    return protocol.get_kraus().to_tensor().eval().array
```

The laws the image has to satisfy hold on a unitary that is not symmetric:
dereliction undoes the bang on one photon, contraction is natural and
weakening is its counit, here on states of up to three photons.

```python {.marimo}
from discopy.optyx.utils.misc import compare_arrays_of_different_sizes

theta, phi = .3, .7
U = np.array([[np.cos(theta), -np.exp(1j * phi) * np.sin(theta)],
              [np.sin(theta), np.exp(1j * phi) * np.cos(theta)]])
u = Box("U", q, q, data=U)

for i, photons in enumerate([(1, 0), (0, 1)]):
    assert np.allclose(amplitudes(
        photonic.Create(*photons) >> F(u.bang() >> Dereliction(q))), U[:, i])

for photons in [(1, 1), (2, 1), (0, 3)]:
    state = photonic.Create(*photons)
    assert compare_arrays_of_different_sizes(
        amplitudes(state >> F(u.bang() >> Contraction(q))),
        amplitudes(state >> F(Contraction(q) >> u.bang() @ u.bang())))
    assert compare_arrays_of_different_sizes(
        amplitudes(state >> F(Contraction(q) >> Weakening(q) @ q.bang())),
        amplitudes(state))
```

## The protocol

Feeding the state of one photon in each mode into the image of the term gives
an optical protocol: the second quantisation of `H` is a Hong-Ou-Mandel beam
splitter, which bunches the two photons, and the contraction shares them
between two dual-rail qubits that are heralded with one photon each. The two
copies agree: the protocol prepares the Bell state `|00> - |11>`.

```python {.marimo}
protocol = photonic.Create(1, 1) >> F(proof)
bell = amplitudes(protocol)
assert np.allclose(bell, [[1, 0], [0, -1]])
protocol.draw(figsize=(4, 5))
```

The W node is the exact comonoid, not a device. A device splits each mode on a
balanced beam splitter, the W node followed by an amplitude of `1/√2` on
each output, which is an isometry. The protocol keeps its two photons, so
the split only scales it by `1/2`: the heralding succeeds with probability
`1/2`, and then gives the same Bell state.

```python {.marimo}
balanced = zw.W(2) >> zw.Endo(2 ** -.5) @ zw.Endo(2 ** -.5)
split = channel.Channel(
    "Δ", balanced @ balanced >> diagram.mode @ zw.SWAP @ diagram.mode)
device = photonic.Create(1, 1) >> photonic.Gate(H.data.T, 2, 2, "!H")\
    >> split >> F(Dereliction(q) @ Dereliction(q))
success = np.sum(np.abs(amplitudes(device)) ** 2)
assert np.allclose(amplitudes(device), bell / 2) and np.isclose(success, .5)
success
```
