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
allows contraction and permutation. They send `!A` to a Fock space `F(A)`,
contraction to a comultiplication `Δ : F(A) → F(A) ⊗ F(A)`, and dereliction
to `ε : F(A) → A`. They then show that the two readings are two different
linear maps (§5.3).

This notebook reproduces §5 with `discopy.optyx`. A vector of `A` with basis
`e₁, …, e_d` is a single photon spread over `d` modes, and the Fock space is
those modes with at most one photon each, of dimension `2^d`. That is the
dimension of the paper's fermionic Fock space. The paper's two copying maps
are then the two spiders of the ZW calculus, applied on every mode:

- the **W** node splits a photon over two outputs, `|1⟩ ↦ |10⟩ + |01⟩`. On a
  vector that is `v ↦ v ⊗ vac + vac ⊗ v`, the paper's **k-extension**
  `v ↦ v ⊗ k + k ⊗ v` once `ε` reads the vacuum as `k`;
- the **Z** spider copies each mode's photon number, `|n⟩ ↦ |n⟩|n⟩`, the
  paper's **basis copy** `eᵢ ↦ eᵢ ⊗ eᵢ`.

Words are vectors drawn at random with a fixed seed. Each reading is a
`tensor.Diagram` built with `from_callable`, and its copies are the optyx
diagrams evaluated as tensors. Every reading is checked against the paper's
formulas, written out independently with `numpy.einsum`.

```python {.marimo}
import numpy as np

from discopy.tensor import Dim, Box, Diagram
from discopy.optyx.core import diagram as optyx
from discopy.optyx.core.zw import W, ZBox
```

## 1. Fock spaces and the two copies

`ι : A → F(A)` puts the photon of `eᵢ` on mode `i`, the first layer of the
Fock space. `ε : F(A) → A` reads it back and sends the vacuum to
`k = (1, …, 1)`, taking `k = 1` as the paper does in its §6. `Δ` is a layer
of spiders, one per mode, followed by the permutation that collects the
first outputs of all modes, then the second outputs.

```python {.marimo}
N = S = Dim(2)
k = 1


def size(A):
    return int(np.prod(A.inside))


def fock(A):
    return Dim(2 ** size(A))


def photon(A):
    """The index in F(A) of one photon on each mode, mode 0 most significant."""
    return [2 ** (size(A) - 1 - m) for m in range(size(A))]


def iota(A):
    array = np.zeros((size(A), 2 ** size(A)))
    array[range(size(A)), photon(A)] = 1
    return Box("ι", A, fock(A), array.reshape(A.inside + (-1, )))


def epsilon(A):
    array = np.zeros((2 ** size(A), size(A)))
    array[photon(A), range(size(A))] = 1
    array[0] = k
    return Box("ε", fock(A), A, array.reshape((-1, ) + A.inside))


def layer(A, spider):
    d = size(A)
    return optyx.Diagram.tensor(*d * [spider]) >> optyx.Diagram.permutation(
        [2 * m for m in range(d)] + [2 * m + 1 for m in range(d)],
        optyx.Mode(2 * d))


def delta(A, spider):
    array = layer(A, spider).to_tensor(input_dims=size(A) * [2]).eval().array
    return Box(f"Δ({spider.name})", fock(A), fock(A) @ fock(A),
               array.real.reshape(3 * (2 ** size(A), )))


def contraction(A, spider):
    """The copies of a vector of A, each derelicted: ι >> Δ >> ε @ ε."""
    return iota(A) >> delta(A, spider) >> epsilon(A) @ epsilon(A)


k_extension, basis_copy = W(2), ZBox(1, 2, lambda _: 1)
layer(N, k_extension).draw(figsize=(4, 4))
```

On the two modes of `N`, the W layer above is the k-extension and the Z
layer below is the basis copy:

```python {.marimo}
layer(N, basis_copy).draw(figsize=(4, 4))
```

```python {.marimo}
ones = np.ones(2)
_v = np.array([.3, .7])
_state = Box("v", Dim(1), N, _v)
_copied = {name: (_state >> contraction(N, spider)).eval().array
           for name, spider in [("k", k_extension), ("basis", basis_copy)]}
assert np.allclose(_copied["k"], np.outer(_v, ones) + np.outer(ones, _v))
assert np.allclose(_copied["basis"], np.diag(_v))
assert np.allclose(
    (_state >> iota(N) >> epsilon(N)).eval().array, _v), "ε ∘ ι = id"
mo.md("W copies `v` to `v ⊗ k + k ⊗ v` and Z copies it to `Σᵢ vᵢ eᵢ ⊗ eᵢ`.")
```

## 2. The lexicon

A word is a state, and a function type `A ⇒ B` is the space `A ⊗ B`. A
function is applied by `ev`, a cup on `A` that leaves `B`. Every `!` of the
paper's types is a Fock space that a contraction copies and a dereliction
`ε` reads. A `!` that is only derelicted contributes `ε ∘ ι = id`, so it is
left out.

| word | type of §3.3 | here |
|---|---|---|
| John, Bill | `!N` | `N` |
| likes | `!((!(!N\S))/N)` | `N ⊗ N ⊗ S`: object, subject, sentence |
| his | `(!(!N\N))/N` | `N ⊗ N ⊗ N`: possessed, owner, result |
| code | `N` | `N` |
| does too | `(!(!N\S))\(!N\S)` | a map `N ⊗ S → N ⊗ S` |

```python {.marimo}
rng = np.random.default_rng(0)


def word(name, A):
    return Box(name, Dim(1), A, rng.random(A.inside))


def ev(A, B):
    return Diagram.cups(A, A) @ B


John, Bill, Mary = word("John", N), word("Bill", N), word("Mary", N)
code, guitar = word("code", N), word("guitar", N)
sleeps, snores = word("sleeps", N @ S), word("snores", N @ S)
plays, likes = word("plays", N @ N @ S), word("likes", N @ N @ S)
his = word("his", N @ N @ N)
He = Box("He", N, N, rng.random((2, 2)))
does_too = Box("does too", N @ S, N @ S, rng.random((2, 2, 2, 2)))
copies = {"k-extension": k_extension, "basis copy": basis_copy}
```

## 3. Anaphora: *John sleeps. He snores.*

`John` is copied, one copy is the subject of *sleeps* and the other goes
through *He* to *snores* (figure 11 of the paper). With k-extension the
paper finds
`sleeps(ε(John)) ⊗ snores(He(k)) + sleeps(ε(k)) ⊗ snores(He(John))`.

```python {.marimo}
def anaphora(spider):
    @Diagram.from_callable(Dim(1), S @ S)
    def sentence():
        john, he = contraction(N, spider)(John())
        return (ev(N, S)(john, *sleeps()),
                ev(N, S)(He(he), *snores()))
    return sentence


anaphora(k_extension).draw(figsize=(6, 6))
```

```python {.marimo}
def array(box):
    return box.array


_j, _sl, _sn, _he = map(array, (John, sleeps, snores, He))
_out = anaphora(k_extension).eval().array
assert np.allclose(_out, np.outer(_j @ _sl, (ones @ _he) @ _sn)
                   + np.outer(ones @ _sl, (_j @ _he) @ _sn))
_out = anaphora(basis_copy).eval().array
assert np.allclose(_out, np.einsum("i,is,ij,jt->st", _j, _sl, _he, _sn))
mo.md("Both copies give the paper's map of §5.1.")
```

## 4. Ellipsis: *John plays guitar. Mary does too.*

The verb phrase *plays guitar* is copied: one copy is applied to *John*, the
other goes through *does too* to *Mary* (figure 12). With k-extension the
paper finds `pg(John) ⊗ does_too(k)(Mary) + k(John) ⊗ does_too(pg)(Mary)`,
where `pg = plays(guitar)`.

```python {.marimo}
def ellipsis(spider):
    @Diagram.from_callable(Dim(1), S @ S)
    def sentence():
        vp = ev(N, N @ S)(guitar(), *plays())
        vp1n, vp1s, vp2n, vp2s = contraction(N @ S, spider)(*vp)
        return (ev(N, S)(John(), vp1n, vp1s),
                ev(N, S)(Mary(), *does_too(vp2n, vp2s)))
    return sentence


ellipsis(k_extension).draw(figsize=(6, 6))
```

```python {.marimo}
_j, _m, _dt = John.array, Mary.array, does_too.array
_pg = np.einsum("o,ons->ns", guitar.array, plays.array)
K = np.ones((2, 2))


def says(vp, subject):
    return np.einsum("n,ns->s", subject, vp)


def too(vp, subject):
    return np.einsum("m,ns,nsmt->t", subject, vp, _dt)


assert np.allclose(ellipsis(k_extension).eval().array,
                   np.outer(says(_pg, _j), too(K, _m))
                   + np.outer(says(K, _j), too(_pg, _m)))
assert np.allclose(ellipsis(basis_copy).eval().array, np.einsum(
    "n,ns,nsmt,m->st", _j, _pg, _dt, _m))
mo.md("Both copies give the paper's map of §5.2.")
```

## 5. Anaphora with ellipsis: strict and sloppy

**Strict** (figures 3 and 13): two contractions. *John* is copied and one
copy is the owner in *his code*. Then the verb phrase *likes John's code* is
copied, one copy for *John* and one for *does too* and *Bill*.

```python {.marimo}
def strict(spider):
    @Diagram.from_callable(Dim(1), S @ S)
    def sentence():
        john, owner = contraction(N, spider)(John())
        johns_code = ev(N, N)(owner, *ev(N, N @ N)(code(), *his()))
        vp = ev(N, N @ S)(johns_code, *likes())
        vp1n, vp1s, vp2n, vp2s = contraction(N @ S, spider)(*vp)
        return (ev(N, S)(john, vp1n, vp1s),
                ev(N, S)(Bill(), *does_too(vp2n, vp2s)))
    return sentence


strict(k_extension).draw(figsize=(8, 9))
```

**Sloppy** (figures 4 and 14): four contractions, one each for *John*,
*Bill*, *his code* and *likes*. Each subject gets its own copy of *likes*
and of *his code*, and owns its own code.

```python {.marimo}
def sloppy(spider):
    @Diagram.from_callable(Dim(1), S @ S)
    def sentence():
        john, owner1 = contraction(N, spider)(John())
        bill, owner2 = contraction(N, spider)(Bill())
        h1p, h1r, h2p, h2r = contraction(N @ N, spider)(
            *ev(N, N @ N)(code(), *his()))
        l1o, l1n, l1s, l2o, l2n, l2s = contraction(N @ N @ S, spider)(
            *likes())
        s1 = ev(N, S)(john, *ev(N, N @ S)(
            ev(N, N)(owner1, h1p, h1r), l1o, l1n, l1s))
        s2 = ev(N, S)(bill, *does_too(*ev(N, N @ S)(
            ev(N, N)(owner2, h2p, h2r), l2o, l2n, l2s)))
        return s1, s2
    return sentence


sloppy(k_extension).draw(figsize=(9, 11))
```

The same readings written with `einsum`, with `copy` the paper's `Δ` on the
vectors of `A` (`ε ⊗ ε ∘ Δ ∘ ι`) rather than on photons:

```python {.marimo}
def k_copy(v):
    flat = v.reshape(-1)
    return (np.outer(flat, np.ones_like(flat))
            + np.outer(np.ones_like(flat), flat)).reshape(v.shape * 2)


def b_copy(v):
    return np.diag(v.reshape(-1)).reshape(v.shape * 2)


def strict_formula(copy, dt=None):
    dt = does_too.array if dt is None else dt
    j1j2 = copy(John.array)
    vp = np.einsum("c,cjr,rns->jns", code.array, his.array,
                   likes.array)                            # owner j
    vp = np.einsum("aj,jns->ans", j1j2, vp)                # John's copy a
    pairs = np.stack([copy(vp[a]) for a in range(2)])     # a, n, s, m, t
    return np.einsum("ansmt,an,mtbu,b->su", pairs, np.eye(2), dt, Bill.array)


def sloppy_formula(copy, dt=None):
    dt = does_too.array if dt is None else dt
    J, B = copy(John.array), copy(Bill.array)
    H = copy(np.einsum("c,cpr->pr", code.array, his.array))  # p, r, q, x
    L = copy(likes.array)                                    # r,n,s, x,m,t
    return np.einsum("ap,bq,prqx,rnsxmt,an,mtcu,bc->su",
                     J, B, H, L, np.eye(2), dt, np.eye(2))


for _name, _spider, _copy in [("k-extension", k_extension, k_copy),
                              ("basis copy", basis_copy, b_copy)]:
    assert np.allclose(strict(_spider).eval().array, strict_formula(_copy))
    assert np.allclose(sloppy(_spider).eval().array, sloppy_formula(_copy))
mo.md("The optyx diagrams and the formulas agree, for both copies.")
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
    return np.einsum("c,cpr,p,rns->ns", code.array, his.array, owner,
                     likes.array)


_jo = John.array
assert np.allclose(strict(k_extension).eval().array, sum(
    np.outer(says(vp, subject), too(elided, Bill.array))
    for vp, subject, elided in [
        (vp_of(ones), _jo, K), (K, _jo, vp_of(ones)),
        (vp_of(_jo), ones, K), (K, ones, vp_of(_jo))]))
mo.md("The strict reading is the paper's four terms.")
```

### The two readings are two maps

This is the claim of §5.3: the strict and the sloppy readings have
different meanings, for either copy.

```python {.marimo}
readings = {name: (strict(spider).eval().array.real,
                   sloppy(spider).eval().array.real)
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
