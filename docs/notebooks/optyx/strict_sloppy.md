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

Each reading is a term of `discopy.bang`, the planar kappa calculus with a
bang: a type called on a function, `N.bang()(lambda x: ...)`, binds `x`, and
a variable of banged type `!A` used twice is copied by a `Contraction` and
read by a `Dereliction`. A `bang.Functor` sends the term to an optyx diagram
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

from discopy.bang import (
    Ty, Box, BangOb, Contraction, Dereliction, Exchange)
from discopy import bang
from discopy.matrix import OptEinsum
from discopy.optyx.core.diagram import Diagram, Mode
from discopy.optyx.core.zw import W, ZBox, Create, Select, Endo
```

## 1. Words as photons

`Wn(n)` is the W node with `n` outputs, built from binary ones since W is
coassociative, and `split(d, n)` splits each of `d` modes into `n` copies,
grouped copy by copy. A word with array `a` from registers of `d₁, …, d_r`
modes to a register of `d` modes is the diagram `gate(a, [d₁, …, d_r], d)`:
each input register is split into one copy per index tuple, a Z spider on
each tuple fuses its copies, so that it holds a photon exactly when every
input has its photon on that tuple's modes, and the photon of each tuple
is routed by W nodes, `Endo` amplitudes and W daggers to the output modes
where `a` is nonzero. With no input, the tuple photon is created. With one
input, it is a linear optical circuit.

```python {.marimo}
def Wn(n):
    return Diagram.id(Mode(1)) if n == 1 else W(2) >> Mode(1) @ Wn(n - 1)


def split(d, n):
    return Diagram.tensor(*d * [Wn(n)]) >> Diagram.permutation(
        [i * n + c for c in range(n) for i in range(d)], Mode(d * n))


def gate(array, dims, cod):
    """A multilinear map on one-photon registers, as a ZW diagram."""
    array = np.asarray(array).reshape(tuple(dims) + (cod, ))
    tuples = list(itertools.product(*map(range, dims)))
    if dims:
        wires, seen = [], [dict() for _ in dims]
        offsets = np.cumsum([0] + [d * (len(tuples) // d) for d in dims])
        for t in tuples:
            for r, i in enumerate(t):
                copy = seen[r][i] = seen[r].get(i, -1) + 1
                wires.append(int(offsets[r]) + copy * dims[r] + i)
        fuse = ZBox(len(dims), 1, lambda _: 1) if len(dims) > 1\
            else Diagram.id(Mode(1))
        tuple_modes = Diagram.tensor(*(
            split(d, len(tuples) // d) for d in dims)) >> Diagram.permutation(
            wires, Mode(len(wires))) >> Diagram.tensor(*len(tuples) * [fuse])
    else:
        tuple_modes = Create(1) >> Wn(len(tuples))
    routes = [[k for k in range(cod) if array[t + (k, )]] for t in tuples]
    route = Diagram.tensor(*(
        Wn(len(ks)) >> Diagram.tensor(*(
            Diagram.id(Mode(1)) if array[t + (k, )] == 1
            else Endo(array[t + (k, )])
            for k in ks)) if ks else Select(0)
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

## 3. The lexicon

A word is a box of `discopy.bang`. A function type of the paper is an atom,
and the elimination rule of `!L*`, from `A ⊗ A\B` to `B`, is a box that
contracts the argument with the first index of the function. A word whose
type has a `!` outside outputs a banged wire, which a term may copy; a `!` on
an argument is read by a dereliction before the argument is passed, and
`ε` is the identity on one photon, so it is left out of the boxes.

| word | type of §3.3 | box here |
|---|---|---|
| John, Bill, Mary | `!N` | `John : 1 → !N` |
| code, guitar | `N` | `code : 1 → N` |
| sleeps, snores | `N\S` | `sleeps : 1 → VP` |
| He | `!N\N` | `He : N → N` |
| plays, likes | `!((!(!N\S))/N)` | `likes : 1 → !V` |
| his | `(!(!N\N))/N` | `his : N → !H`, applied to the possessed |
| does too | `(!(!N\S))\(!N\S)` | `does too : VP ⊗ N → S` |

where `VP = !N\S`, `H = !N\N` and `V = (!VP)/N`, evaluated by
`ev : N ⊗ VP → S`, `of : N ⊗ H → N` and `obj : N ⊗ V → !VP`. `and` puts two
sentences side by side.

```python {.marimo}
N, S, VP, H, V = map(Ty, ("N", "S", "VP", "H", "V"))
John, Bill, Mary = (Box(name, Ty(), N.bang()) for name in (
    "John", "Bill", "Mary"))
code, guitar = Box("code", Ty(), N), Box("guitar", Ty(), N)
sleeps, snores = Box("sleeps", Ty(), VP), Box("snores", Ty(), VP)
plays, likes = Box("plays", Ty(), V.bang()), Box("likes", Ty(), V.bang())
He, his, does_too = Box("He", N, N), Box("his", N, H.bang()), Box(
    "does too", VP @ N, S)
ev, of, obj = Box("ev", N @ VP, S), Box("of", N @ H, N), Box(
    "obj", N @ V, VP.bang())
and_ = Box("and", S @ S, S @ S)


def der(term):
    """The dereliction of a term of banged type, which reads it once."""
    return Dereliction(term.cod.inside[0].arg)(term)
```

The arrays `a` of the words are drawn at random with their arguments first,
and those of the elimination boxes are Kronecker deltas.

```python {.marimo}
rng = np.random.default_rng(0)
shapes = {"N": (2, ), "VP": (2, 2), "He": (2, 2), "V": (2, 2, 2),
          "his": (2, 2, 2), "does too": (2, 2, 2, 2)}
kinds = {"sleeps": "VP", "snores": "VP", "He": "He", "plays": "V",
         "likes": "V", "his": "his", "does too": "does too"}
words = ("John", "Bill", "Mary", "code", "guitar", "sleeps", "snores", "He",
         "plays", "likes", "his", "does too")
a = {name: rng.random(shapes[kinds.get(name, "N")]) for name in words}
_I = np.eye(2)
a |= {"ev": np.einsum("nm,st->nmst", _I, _I),
      "of": np.einsum("pq,rt->pqrt", _I, _I),
      "obj": np.einsum("or,nm,st->ornsmt", _I, _I, _I)}
```

A `bang.Functor` into optyx diagrams sends an atom of dimension `d` and its
bang to `d` modes, a dereliction to `ε`, a contraction to `Δ` with the
chosen spider, an exchange to a permutation of modes, `and` to the identity
and every other box to its gate, cut to at most one photon per mode.

```python {.marimo}
size = {N: 2, S: 2, VP: 4, H: 4, V: 8}


def modes(wire):
    return size[wire.arg] if isinstance(wire, BangOb) else size[Ty(wire)]


def ob(x):
    return Mode(sum(map(modes, x.inside)))


def ar(spider):
    def image(box):
        if isinstance(box, Dereliction):
            return epsilon(size[box.x])
        if isinstance(box, Contraction):
            return layer(size[box.x], spider)
        if isinstance(box, Exchange):
            split_at = -1 if box.is_dagger else 1
            x, y = len(ob(box.dom[:split_at])), len(ob(box.dom[split_at:]))
            return Diagram.permutation(
                list(range(x, x + y)) + list(range(x)), Mode(x + y))
        if box == and_:
            return Diagram.id(Mode(4))
        cod = modes(box.cod.inside[0])
        return gate(a[box.name], list(map(modes, box.dom.inside)), cod)\
            >> cut(cod)
    return image


computed = {}


def meaning(term, spider):
    """
    The amplitudes of a reading, with contraction given by a spider,
    computed once: contracting the photons of a reading takes a minute.
    """
    key = (str(term), spider.name)
    if key not in computed:
        functor = bang.Functor(ob_map=ob, ar_map=ar(spider), cod=Diagram)
        computed[key] = amplitudes(term.eval(functor), (2, 2))
    return computed[key]
```

## 4. Anaphora: *John sleeps. He snores.*

`John` is used twice, so it is copied: one copy is the subject of *sleeps*
and the other goes through *He* to *snores* (figure 11 of the paper). With
k-extension the paper finds
`sleeps(ε(John)) ⊗ snores(He(k)) + sleeps(ε(k)) ⊗ snores(He(John))`.

```python {.marimo}
anaphora = N.bang()(lambda john: and_(
    ev(der(john), sleeps()), ev(He(der(john)), snores())))(John())
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

## 5. Ellipsis: *John plays guitar. Mary does too.*

The verb phrase *plays guitar* is bound and used twice: one copy is
applied to *John*, the other goes through *does too* to *Mary* (figure 12).
With k-extension the paper finds
`pg(John) ⊗ does_too(k)(Mary) + k(John) ⊗ does_too(pg)(Mary)`, where
`pg = plays(guitar)`.

```python {.marimo}
ellipsis = VP.bang()(lambda vp: and_(
    ev(der(John()), der(vp)), does_too(der(vp), der(Mary()))))(
    obj(guitar(), der(plays())))
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

## 6. Anaphora with ellipsis: strict and sloppy

**Strict** (figures 3 and 13): two contractions. *John* is used twice, as
the subject and as the owner in *his code*, and the verb phrase *likes
John's code* is used twice, once for *John* and once for *does too* and
*Bill*.

```python {.marimo}
strict = N.bang()(lambda john: VP.bang()(lambda vp: and_(
    ev(der(john), der(vp)), does_too(der(vp), der(Bill()))))(
    obj(of(der(john), der(his(code()))), der(likes()))))(John())
mo.md(f"`{strict}`")
```

```python {.marimo}
strict.eval().draw(figsize=(8, 10))
```

**Sloppy** (figures 4 and 14): four contractions. *his code* is bound and
used twice, and *John*, *Bill* and *likes* are each used twice, so each
subject owns its own code. The variables are bound in the order the body
first uses them; it is planar, and the copies are brought where they are
read by exchanges.

```python {.marimo}
sloppy = (N.bang() @ H.bang() @ V.bang() @ N.bang())(
    lambda john, h, verb, bill: and_(
        ev(der(john), der(obj(of(der(john), der(h)), der(verb)))),
        does_too(der(obj(of(der(bill), der(h)), der(verb))), der(bill))))(
    John(), his(code()), likes(), Bill())
mo.md(f"`{sloppy}`")
```

```python {.marimo}
sloppy.eval().draw(figsize=(9, 12))
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
