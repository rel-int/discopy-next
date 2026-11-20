---
title: Linear Logic
marimo-version: 0.24.0
---

```python {.marimo}
import marimo as mo
```

# From a linear term with bangs to an optical protocol

A term of `discopy.optyx.term` is a diagram of `discopy.bang`, the free
linear category, on the dual-rail `qubit`: it uses each wire once unless its
type is banged, as `!qubit` may be copied by `Contraction`, discarded by
`Weakening` and read once by `Dereliction`. Over quantum states, `!` is the
bosonic Fock space: `!qubit` is the Fock space of the two modes of a dual-rail
qubit, and `Term.to_channel` compiles a term to `discopy.optyx`, sending:

- a `Box` on one qubit, which holds the unitary of one photon in two modes,
  to that gate on the dual-rail qubit, heralded,
- its bang `f.bang()` to the linear-optical gate acting on every photon at
  once, its second quantisation,
- `Contraction` to the W node of ZW on each mode, the bosonic comonoid that
  shares the photons of a mode between two copies,
- `Weakening` to post-selecting the vacuum, and `Dereliction` to heralding
  exactly one photon, i.e. reading the copy back as a dual-rail qubit.

The digging `!x -> !!x` and the laxator `!x @ !y -> !(x @ y)` have no such
image: the Fock space of a Fock space is not a set of modes, and
`!x @ !y` is the Fock space of `x ⊕ y`, not of `x @ y`. So `to_channel`
raises `NotImplementedError` on the terms that use them.

```python {.marimo}
import numpy as np

from discopy.optyx import channel, photonic
from discopy.optyx.core import diagram, zw
from discopy.optyx.term import (
    Term, Box, Contraction, Dereliction, Weakening, qubit)
```

## The term

The non-linear term `λx. (H x, H x)` uses its argument twice. Its translation
into linear logic gives `x` the type `!qubit`, applies the bang of `H` and
contracts the result into two copies, each one derelicted. The function
syntax of terms writes it with one variable per wire.

```python {.marimo}
H = Box("H", qubit, qubit, data=np.array([[1, 1], [1, -1]]) / 2 ** .5)


@Term.from_callable(qubit.bang(), qubit @ qubit)
def proof(x):
    y, z = Contraction(qubit)(H.bang()(x))
    return Dereliction(qubit)(y), Dereliction(qubit)(z)


proof.draw(figsize=(3, 4))
```

## The compilation

The laws the image has to satisfy hold on a unitary that is not symmetric:
dereliction undoes the bang on one photon and is natural, contraction is
natural and weakening is its counit, here on states of up to three photons.

```python {.marimo}
from discopy.optyx.utils.misc import compare_arrays_of_different_sizes


def amplitudes(protocol):
    return protocol.get_kraus().to_tensor().eval().array


theta, phi = .3, .7
U = np.array([[np.cos(theta), -np.exp(1j * phi) * np.sin(theta)],
              [np.sin(theta), np.exp(1j * phi) * np.cos(theta)]])
u = Box("U", qubit, qubit, data=U)

for i, photons in enumerate([(1, 0), (0, 1)]):
    state = photonic.Create(*photons)
    for term in [u.bang() >> Dereliction(qubit), Dereliction(qubit) >> u]:
        assert np.allclose(amplitudes(state >> term.to_channel()), U[:, i])

for photons in [(1, 1), (2, 1), (0, 3)]:
    state = photonic.Create(*photons)
    assert compare_arrays_of_different_sizes(
        amplitudes(state >> (u.bang() >> Contraction(qubit)).to_channel()),
        amplitudes(state >> (
            Contraction(qubit) >> u.bang() @ u.bang()).to_channel()))
    assert compare_arrays_of_different_sizes(
        amplitudes(state >> (
            Contraction(qubit) >> Weakening(qubit) @ qubit.bang()
        ).to_channel()),
        amplitudes(state))
```

## The protocol

Feeding the state of one photon in each mode into the image of the term gives
an optical protocol: the second quantisation of `H` is a Hong-Ou-Mandel beam
splitter, which bunches the two photons, and the contraction shares them
between two dual-rail qubits that are heralded with one photon each. The two
copies agree: the protocol prepares the Bell state `|00> - |11>`.

```python {.marimo}
protocol = photonic.Create(1, 1) >> proof.to_channel()
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
device = photonic.Create(1, 1) >> H.bang().to_channel() >> split\
    >> (Dereliction(qubit) @ Dereliction(qubit)).to_channel()
success = np.sum(np.abs(amplitudes(device)) ** 2)
assert np.allclose(amplitudes(device), bell / 2) and np.isclose(success, .5)
success
```
