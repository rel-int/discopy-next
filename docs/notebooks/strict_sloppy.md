---
title: Strict and sloppy ellipsis
marimo-version: 0.24.0
---

```python {.marimo}
import marimo as mo
```

# Strict and sloppy readings of ellipsis, with photons

*John likes his code. Bill does too.* has two readings:

- **strict**: Bill likes John's code. The anaphor *his* is resolved with
  *John* first, then the ellipsis *does too* with the verb phrase *likes
  John's code*.
- **sloppy**: Bill likes Bill's code. The ellipsis is resolved with *likes
  his code* first, then each *his* with its own subject.

McPheat, Wijnholds, Sadrzadeh, Correia and Toumi, *Anaphora and Ellipsis in
Lambek Calculus with a Relevant Modality: Syntax and Semantics*
([arXiv:2110.10641](https://arxiv.org/abs/2110.10641)), derive the two
readings in `!L*`, the Lambek calculus with a relevant modality `!` that
allows contraction and permutation. They send `!A` to the fermionic Fock
space `F(A) = ⊕ₙ A^∧n` (§4.1), contraction to a comultiplication
`Δ : F(A) → F(A) ⊗ F(A)` and dereliction to `ε : F(A) → A`. The canonical
`Δ`, dual to the wedge product, is a `2ⁿ × 4ⁿ` matrix, so §5 replaces it
with two copies of vectors: the **k-extension** `v ↦ v ⊗ k + k ⊗ v` and the
**basis copy** `eᵢ ↦ eᵢ ⊗ eᵢ`. They then show that the two readings are two
different linear maps (§5.3).

This notebook reproduces §5 with photons instead of fermions. A vector of
`A`, with basis `e₁, …, e_d`, is one photon spread over `d` modes, and `!A`
is the bosonic Fock space of those modes: any number of photons. The two
copies are the two spiders of the ZW calculus, on every mode:

- the **W** node is the coproduct of the bosonic Fock space, dual to the
  symmetric product: it sends one photon to `|10⟩ + |01⟩`, i.e. `v` to
  `v ⊗ vac + vac ⊗ v`, which is the k-extension once the dereliction reads
  the vacuum as `k`;
- the **Z** spider copies the photon number of each mode, `|n⟩ ↦ |n⟩|n⟩`,
  which is the basis copy.

Fermions and bosons differ only on two particles or more: `eᵢ ∧ eᵢ = 0` and
an exchange of two fermions costs a sign, neither of which a single photon
sees. Every word below is one photon, and `ε` keeps exactly one photon, so
no reading reaches two particles on a wire and the photonic and fermionic
maps agree. The basis copy is not a coproduct of either Fock space; the
paper brings it in as a choice, and so does this notebook.

Each reading is first a term of `discopy.lambek.modality`, the Lambek
calculus `!L*` of the paper: its words have the types of §3.3, a type
called on a function, `N.bang()(lambda x: ...)`, binds `x`, and a variable
of banged type `!A` used twice is copied by a `Contraction` and read by a
`Dereliction`. A `modality.Functor` then sends the term to an optyx diagram
of photons, made of the boxes of `discopy.optyx.core.zw`: `Create`, `W` and
its dagger, `Z` spiders, `Endo` for amplitudes and `Select`. That diagram is
evaluated by optyx, and its amplitudes on one photon per output are the
meaning of the reading. Words are vectors drawn at random with a fixed seed,
and every reading is checked against the paper's formulas, written out
independently with `numpy.einsum`.

```python {.marimo}
import itertools

import cotengra
import numpy as np

from discopy.linear.monoidal import BangOb
from discopy.biclosed import Exp
from discopy.lambek.modality import (
    Ty, Box, Eval, Contraction, Dereliction, Exchange, Functor)
from discopy.matrix import OptEinsum
from discopy.optyx.core.diagram import Diagram, Mode
from discopy.optyx.core.zw import W, ZBox, Create, Select, Endo
```

## 1. Words as photons

`Wn(n)` is the W node with `n` outputs, built from binary ones since W is
coassociative; with no outputs it selects the vacuum. A word with array `a` from registers of `d₁, …, d_r`
modes to a register of `d` modes is the diagram `gate(a, [d₁, …, d_r], d)`:
each input mode is split into one copy per index tuple that uses it and
where `a` is nonzero, a mode that no such tuple uses is selected empty, and
a Z spider on each tuple fuses its copies, so that it holds a photon exactly when every
input has its photon on that tuple's modes, and the photon of each tuple
is routed by W nodes, `Endo` amplitudes and W daggers to the output modes
where `a` is nonzero. With no input, the tuple photon is created. With one
input, it is a linear optical circuit.

```python {.marimo}
def Wn(n):
    if n == 0:
        return Select(0)
    return Diagram.id(Mode(1)) if n == 1 else W(2) >> Mode(1) @ Wn(n - 1)


def gate(array, dims, cod):
    """A multilinear map on one-photon registers, as a ZW diagram."""
    array = np.asarray(array).reshape(tuple(dims) + (cod, ))
    tuples = [t for t in itertools.product(*map(range, dims)) if array[t].any()]
    if dims:
        uses = [[sum(t[r] == i for t in tuples) for i in range(d)]
                for r, d in enumerate(dims)]
        first = {(r, i): sum(map(sum, uses[:r])) + sum(uses[r][:i])
                 for r, d in enumerate(dims) for i in range(d)}
        wires, seen = [], {}
        for t in tuples:
            for r, i in enumerate(t):
                seen[r, i] = seen.get((r, i), -1) + 1
                wires.append(first[r, i] + seen[r, i])
        fuse = ZBox(len(dims), 1, lambda _: 1) if len(dims) > 1\
            else Diagram.id(Mode(1))
        tuple_modes = Diagram.tensor(*(
            Wn(n) if n else Select(0) for counts in uses for n in counts))\
            >> Diagram.permutation(wires, Mode(len(wires)))\
            >> Diagram.tensor(*len(tuples) * [fuse])
    else:
        tuple_modes = Create(1) >> Wn(len(tuples))
    routes = [[k for k in range(cod) if array[t + (k, )]] for t in tuples]
    route = Diagram.id(Mode(0)).tensor(*(
        Wn(len(ks)) >> Diagram.tensor(*(
            Diagram.id(Mode(1)) if array[t + (k, )] == 1
            else Endo(array[t + (k, )])
            for k in ks))
        for t, ks in zip(tuples, routes)))
    ends = [(t, k) for t, ks in enumerate(routes) for k in ks]
    fan = [sum(k in ks for ks in routes) for k in range(cod)]
    merge = Diagram.permutation(
        [ends.index((t, k)) for k in range(cod) for t in range(len(tuples))
         if k in routes[t]], Mode(len(ends))) >> Diagram.tensor(*(
            Wn(n).dagger() if n else Create(0) for n in fan))
    return tuple_modes >> route >> merge


def amplitudes(diagram, cods):
    """The amplitudes of one photon on each output register."""
    array = diagram.to_tensor().eval(
        contractor=OptEinsum(), optimize=cotengra.HyperOptimizer(
            methods=["greedy"], max_repeats=16, minimize="size",
            progbar=False)).array
    assert np.allclose(array.imag, 0), "the words are real vectors"
    result = np.zeros(cods)
    for js in itertools.product(*map(range, cods)):
        result[js] = array[sum(
            (tuple(int(m == j) for m in range(c)) for c, j in zip(cods, js)),
            ())].real
    return result
```

```python {.marimo}
_rng = np.random.default_rng(1)
_v, _u = _rng.random(2), _rng.random(3)
_M, _T = _rng.random((2, 3)), _rng.random((2, 3, 2))
assert np.allclose(amplitudes(gate(_v, [], 2), [2]), _v)
assert np.allclose(amplitudes(gate(np.zeros(2), [], 2), [2]), 0)
assert np.allclose(amplitudes(gate(_v, [], 2) >> gate(_M, [2], 3), [3]),
                   _v @ _M)
assert np.allclose(amplitudes(
    gate(_v, [], 2) @ gate(_u, [], 3) >> gate(_T, [2, 3], 2), [2]),
    np.einsum("i,j,ijk->k", _v, _u, _T))
mo.md("A gate on photons computes its array, alone or applied to others.")
```

The gate of a word with one input of two modes, e.g. *He*, splits the photon
of each mode towards the two outputs, weighs each path by its entry and
merges the paths that end on the same mode:

```python {.marimo}
gate(np.array([[.1, .2], [.3, .4]]), [2], 2).draw(figsize=(6, 7))
```

## 2. Copy and dereliction

`Δ` is a layer of spiders, one per mode, followed by the permutation that
collects the first outputs, then the second. `ε : !A → A` reads a register
back as one photon and the vacuum as `k = e₁ + … + e_d` (`k = 1`, as in §6
of the paper): a Z spider on `|0⟩ + |1⟩` cuts each mode to at most one
photon, W adds one photon to each mode with amplitude one, and a copy of the
photon numbers, merged by W dagger and selected on one photon, keeps the
terms with exactly one photon. Then `ε(vac) = k`, `ε(eᵢ) = eᵢ`, and `ε` is
zero on two photons or more.

```python {.marimo}
def cut(d):
    return Diagram.tensor(*d * [
        Mode(1) @ ZBox(0, 1, [1, 1]) >> ZBox(2, 1, lambda _: 1)])


def layer(d, spider):
    return Diagram.tensor(*d * [spider]) >> Diagram.permutation(
        [2 * m for m in range(d)] + [2 * m + 1 for m in range(d)],
        Mode(2 * d))


def epsilon(d):
    add_one = Diagram.tensor(*d * [
        Mode(1) @ ZBox(0, 1, [1, 1]) >> W(2).dagger()])
    one_photon = layer(d, ZBox(1, 2, lambda _: 1)) >> Mode(d) @ (
        Wn(d).dagger() >> Select(1))
    return add_one >> cut(d) >> one_photon


k_extension, basis_copy = W(2), ZBox(1, 2, lambda _: 1)
copies = {"k-extension": k_extension, "basis copy": basis_copy}
layer(2, k_extension).draw(figsize=(4, 3))
```

On the two modes of `N`, the W layer above is the k-extension, the Z layer
below the basis copy, and last comes the dereliction:

```python {.marimo}
layer(2, basis_copy).draw(figsize=(4, 3))
```

```python {.marimo}
epsilon(2).draw(figsize=(8, 10))
```

```python {.marimo}
_v = np.array([.3, .7])
_state = gate(_v, [], 2)
assert np.allclose(amplitudes(Create(0, 0) >> epsilon(2), [2]), [1, 1])
assert np.allclose(amplitudes(_state >> epsilon(2), [2]), _v)
assert np.allclose(amplitudes(Create(1, 1) >> epsilon(2), [2]), 0)
_copied = {name: amplitudes(_state >> layer(2, spider)
                            >> epsilon(2) @ epsilon(2), [2, 2])
           for name, spider in copies.items()}
assert np.allclose(_copied["k-extension"],
                   np.outer(_v, [1, 1]) + np.outer([1, 1], _v))
assert np.allclose(_copied["basis copy"], np.diag(_v))
mo.md("`ε` sends the vacuum to `k`, one photon to itself and two to zero; "
      "W copies `v` to `v ⊗ k + k ⊗ v` and Z to `Σᵢ vᵢ eᵢ ⊗ eᵢ`.")
```

## 3. The lexicon of `!L*`

A word is a box of `discopy.lambek.modality` whose codomain is its type in
§3.3 of the paper, where `x >> y` is `x\y` and `y << x` is `y/x`. A
function word is applied to its argument by `Eval`, the elimination rule of
the Lambek calculus, which `fa` and `ba` below write as forward and
backward application on terms. A variable of banged type used twice is
copied by `Contraction`, and `der` reads a banged term once by
`Dereliction`.

| word | type of §3.3 |
|---|---|
| John, Bill, Mary | `!N` |
| code, guitar | `N` |
| sleeps, snores | `N\S` |
| He | `!N\N` |
| plays, likes | `!((!VP)/N)` |
| his | `(!H)/N` |
| does too | `(!VP)\VP` |

where `VP = !N\S` and `H = !N\N`. The full stop `.` puts two sentences side
by side, so that a reading is one term of type `S ⊗ S`.

```python {.marimo}
N, S = Ty("N"), Ty("S")
VP, H = N.bang() >> S, N.bang() >> N
types = {"John": N.bang(), "Bill": N.bang(), "Mary": N.bang(), "code": N,
         "guitar": N, "sleeps": N >> S, "snores": N >> S, "He": H,
         "plays": (VP.bang() << N).bang(), "likes": (VP.bang() << N).bang(),
         "his": H.bang() << N, "does too": VP.bang() >> VP}
John, Bill, Mary, code, guitar, sleeps, snores, He, plays, likes, his, \
    does_too = (Box(name, Ty(), typ) for name, typ in types.items())
stop = Box(".", S @ S, S @ S)


def fa(function, argument):
    """Forward application, of a function of type y << x on its left."""
    return Eval(function.cod)(function, argument)


def ba(argument, function):
    """Backward application, of a function of type x >> y on its right."""
    return Eval(function.cod)(argument, function)


def der(term):
    """The dereliction of a term of banged type, which reads it once."""
    return Dereliction(term.cod.inside[0].arg)(term)
```

The arrays `a` of the words are drawn at random with their arguments first,
and `order` transposes those of `plays`, `likes` and `his` to the order of
their types, with the argument of the `/` last.

```python {.marimo}
rng = np.random.default_rng(0)
shapes = {"N": (2, ), "VP": (2, 2), "He": (2, 2), "V": (2, 2, 2),
          "his": (2, 2, 2), "does too": (2, 2, 2, 2)}
kinds = {"sleeps": "VP", "snores": "VP", "He": "He", "plays": "V",
         "likes": "V", "his": "his", "does too": "does too"}
a = {name: rng.random(shapes[kinds.get(name, "N")]) for name in types}
order = {"V": (1, 2, 0), "his": (1, 2, 0)}
```

## 4. From `!L*` to photons

A `modality.Functor` into optyx diagrams sends each type to modes: an atom
to two, `!x` to the modes of `x`, whose Fock space they carry, and an
exponential `x >> y` or `y << x` to one photon over the product of the modes
of `x` and `y`, a matrix. optyx diagrams are not a closed category, so the
functor sends exponentials and evaluations through its `ob_map` and
`ar_map`, like any other generator.

`Eval` goes to the gate of a Kronecker delta, which contracts the argument
with the matching index of the function. An argument of banged type `!x` is
read by `ε` first: the paper's words of type `!x\y` are linear maps from
`x`, as in its `sleeps(ε(John))` and `He(k)`, and this is where the vacuum
left by a k-extension becomes `k`. A contraction goes to `Δ` with the chosen
spider, a dereliction to `ε`, an exchange to a permutation of modes, the
full stop to the identity and a word to the gate of its array, cut to at
most one photon per mode.

```python {.marimo}
def modes(wire):
    if isinstance(wire, BangOb):
        return len(ob(wire.arg))
    if isinstance(wire, Exp):
        return len(ob(wire.base)) * len(ob(wire.exponent))
    return 2


def ob(x):
    return Mode(sum(map(modes, x.inside)))


def banged(x):
    return len(x) == 1 and isinstance(x.inside[0], BangOb)


def evaluation(box):
    exp = box.x.inside[0]
    m, n = len(ob(exp.exponent)), len(ob(exp.base))
    delta = np.einsum("ac,bd->abcd", np.eye(m), np.eye(n))
    read = epsilon(m) if banged(exp.exponent) else Diagram.id(Mode(m))
    if box.left:
        array = delta.transpose(1, 0, 2, 3).reshape(n * m, m, n)
        return Diagram.id(Mode(n * m)) @ read >> gate(array, [n * m, m], n)
    array = delta.transpose(2, 0, 1, 3).reshape(m, m * n, n)
    return read @ Mode(m * n) >> gate(array, [m, m * n], n)


def ar(spider):
    def image(box):
        if isinstance(box, Dereliction):
            return epsilon(len(ob(box.x)))
        if isinstance(box, Contraction):
            return layer(len(ob(box.x)), spider)
        if isinstance(box, Exchange):
            split_at = -1 if box.is_dagger else 1
            x, y = len(ob(box.dom[:split_at])), len(ob(box.dom[split_at:]))
            return Diagram.permutation(
                list(range(x, x + y)) + list(range(x)), Mode(x + y))
        if isinstance(box, Eval):
            return evaluation(box)
        if box == stop:
            return Diagram.id(Mode(4))
        kind, cod = kinds.get(box.name, "N"), len(ob(box.cod))
        array = a[box.name].transpose(order[kind]) if kind in order\
            else a[box.name]
        return gate(array, [], cod) >> cut(cod)
    return image


computed = {}


def meaning(term, spider):
    """
    The amplitudes of a reading, with contraction given by a spider,
    computed once: contracting the photons of a reading takes a minute.
    """
    key = (str(term), spider.name)
    if key not in computed:
        functor = Functor(ob_map=ob, ar_map=ar(spider), cod=Diagram)
        computed[key] = amplitudes(term.eval(functor), (2, 2))
    return computed[key]
```

## 5. Anaphora: *John sleeps. He snores.*

`John` is used twice, so it is copied: one copy is the subject of *sleeps*
and the other goes through *He* to *snores* (figure 11 of the paper). With
k-extension the paper finds
`sleeps(ε(John)) ⊗ snores(He(k)) + sleeps(ε(k)) ⊗ snores(He(John))`.

```python {.marimo}
anaphora = N.bang()(lambda john: stop(
    ba(der(john), sleeps()), ba(ba(john, He()), snores())))(John())
anaphora.eval().draw(figsize=(6, 6))
```

```python {.marimo}
ones = np.ones(2)
_j, _sl, _sn, _he = a["John"], a["sleeps"], a["snores"], a["He"]
assert np.allclose(meaning(anaphora, k_extension),
                   np.outer(_j @ _sl, (ones @ _he) @ _sn)
                   + np.outer(ones @ _sl, (_j @ _he) @ _sn))
assert np.allclose(meaning(anaphora, basis_copy),
                   np.einsum("i,is,ij,jt->st", _j, _sl, _he, _sn))
mo.md("Both copies give the paper's map of §5.1.")
```

## 6. Ellipsis: *John plays guitar. Mary does too.*

The verb phrase *plays guitar* is bound and used twice: one copy is
applied to *John*, the other goes through *does too* to *Mary* (figure 12).
With k-extension the paper finds
`pg(John) ⊗ does_too(k)(Mary) + k(John) ⊗ does_too(pg)(Mary)`, where
`pg = plays(guitar)`.

```python {.marimo}
ellipsis = VP.bang()(lambda vp: stop(
    ba(John(), der(vp)), ba(Mary(), ba(vp, does_too()))))(
    fa(der(plays()), guitar()))
ellipsis.eval().draw(figsize=(6, 6))
```

```python {.marimo}
_j, _m, _dt = a["John"], a["Mary"], a["does too"]
_pg = np.einsum("o,ons->ns", a["guitar"], a["plays"])
K = np.ones((2, 2))


def says(vp, subject):
    return np.einsum("n,ns->s", subject, vp)


def too(vp, subject):
    return np.einsum("m,ns,nsmt->t", subject, vp, _dt)


assert np.allclose(meaning(ellipsis, k_extension),
                   np.outer(says(_pg, _j), too(K, _m))
                   + np.outer(says(K, _j), too(_pg, _m)))
assert np.allclose(meaning(ellipsis, basis_copy), np.einsum(
    "n,ns,nsmt,m->st", _j, _pg, _dt, _m))
mo.md("Both copies give the paper's map of §5.2.")
```

## 7. Anaphora with ellipsis: strict and sloppy

**Strict** (figures 3 and 13): two contractions. *John* is used twice, as
the subject and as the owner in *his code*, and the verb phrase *likes
John's code* is used twice, once for *John* and once for *does too* and
*Bill*.

```python {.marimo}
strict = N.bang()(lambda john: VP.bang()(lambda vp: stop(
    ba(john, der(vp)), ba(Bill(), ba(vp, does_too()))))(
    fa(der(likes()), ba(john, der(fa(his(), code()))))))(John())
mo.md(f"`{strict}`")
```

```python {.marimo}
strict.eval().draw(figsize=(9, 11))
```

**Sloppy** (figures 4 and 14): four contractions. *his code* is bound and
used twice, and *John*, *Bill* and *likes* are each used twice, so each
subject owns its own code. The variables are bound in the order the body
first uses them; it is planar, and the copies are brought where they are
read by exchanges.

```python {.marimo}
sloppy = (N.bang() @ likes.cod @ H.bang() @ N.bang())(
    lambda john, verb, h, bill: stop(
        ba(john, der(fa(der(verb), ba(john, der(h))))),
        ba(bill, ba(fa(der(verb), ba(bill, der(h))), does_too()))))(
    John(), likes(), fa(his(), code()), Bill())
mo.md(f"`{sloppy}`")
```

```python {.marimo}
sloppy.eval().draw(figsize=(16, 16))
```

The same readings written with `einsum`, with `copy` the paper's `Δ` on the
vectors of `A`, i.e. `ε ⊗ ε ∘ Δ` on one photon:

```python {.marimo}
def k_copy(v):
    flat = v.reshape(-1)
    return (np.outer(flat, np.ones_like(flat))
            + np.outer(np.ones_like(flat), flat)).reshape(v.shape * 2)


def b_copy(v):
    return np.diag(v.reshape(-1)).reshape(v.shape * 2)


def strict_formula(copy, dt=None):
    dt = a["does too"] if dt is None else dt
    j1j2 = copy(a["John"])
    vp = np.einsum("c,cjr,rns->jns", a["code"], a["his"],
                   a["likes"])                            # owner j
    vp = np.einsum("aj,jns->ans", j1j2, vp)                # John's copy a
    pairs = np.stack([copy(vp[a]) for a in range(2)])     # a, n, s, m, t
    return np.einsum("ansmt,an,mtbu,b->su", pairs, np.eye(2), dt, a["Bill"])


def sloppy_formula(copy, dt=None):
    dt = a["does too"] if dt is None else dt
    J, B = copy(a["John"]), copy(a["Bill"])
    H = copy(np.einsum("c,cpr->pr", a["code"], a["his"]))  # p, r, q, x
    L = copy(a["likes"])                                    # r,n,s, x,m,t
    return np.einsum("ap,bq,prqx,rnsxmt,an,mtcu,bc->su",
                     J, B, H, L, np.eye(2), dt, np.eye(2))


for _spider, _copy in [(k_extension, k_copy), (basis_copy, b_copy)]:
    assert np.allclose(meaning(strict, _spider), strict_formula(_copy))
    assert np.allclose(meaning(sloppy, _spider), sloppy_formula(_copy))
mo.md("The terms and the formulas agree, for both copies.")
```

The paper writes the strict reading under k-extension as four terms
(§5.3), one for each choice of which copy of *John* and which copy of the
verb phrase gets `k`. With `vp(x) = likes(his(x, code))` and `K` the
all-ones verb phrase:

```text
vp(k)(John) ⊗ too(K)(Bill) + K(John) ⊗ too(vp(k))(Bill)
  + vp(John)(k) ⊗ too(K)(Bill) + K(k) ⊗ too(vp(John))(Bill)
```

The paper's typeset version has *Bob* for *Bill*, and its fourth term
misplaces the arguments of *likes* and *k*. The four terms above are what
its Sweedler formula expands to.

```python {.marimo}
def vp_of(owner):
    return np.einsum("c,cpr,p,rns->ns", a["code"], a["his"], owner,
                     a["likes"])


_jo = a["John"]
assert np.allclose(meaning(strict, k_extension), sum(
    np.outer(says(vp, subject), too(elided, a["Bill"]))
    for vp, subject, elided in [
        (vp_of(ones), _jo, K), (K, _jo, vp_of(ones)),
        (vp_of(_jo), ones, K), (K, ones, vp_of(_jo))]))
mo.md("The strict reading is the paper's four terms.")
```

### The two readings are two maps

This is the claim of §5.3: the strict and the sloppy readings have
different meanings, for either copy.

```python {.marimo}
readings = {name: (meaning(strict, spider), meaning(sloppy, spider))
            for name, spider in copies.items()}
for _name, (_strict, _sloppy) in readings.items():
    assert not np.allclose(_strict, _sloppy), _name
mo.md("\n".join(
    f"- **{name}**: strict `{np.round(s, 3).tolist()}`, "
    f"sloppy `{np.round(t, 3).tolist()}`"
    for name, (s, t) in readings.items()))
```

Under basis copying they differ only because *does too* has a subject slot
of its own. If *does too* is the identity, the copy of the verb phrase
forces John and Bill onto the same basis vector, which then owns the code
in both readings. The strict and sloppy readings then coincide. With
k-extension they still differ.

```python {.marimo}
_id = np.einsum("nm,st->nsmt", np.eye(2), np.eye(2))
assert np.allclose(strict_formula(b_copy, _id), sloppy_formula(b_copy, _id))
assert not np.allclose(strict_formula(k_copy, _id),
                       sloppy_formula(k_copy, _id))
mo.md("With *does too* the identity: basis copy collapses the readings, "
      "k-extension keeps them apart.")
```

Table 2 of the paper, the copying models on the ellipsis disambiguation
dataset, is reproduced on the released word spaces in the wiki page
*Ellipsis with a relevant modality, reproduced* (rel-int/wiki#207).
