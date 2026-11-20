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

Each reading is a `tensor.Term`, a lambda term whose types are dimensions:
`N(lambda x: ...)` is an abstraction, and a variable used twice is copied.
The copy is chosen by the diagram class the term is evaluated in, here the
optyx diagrams evaluated as tensors. Words are vectors drawn at random with a
fixed seed. Every reading is checked against the paper's formulas, written
out independently with `numpy.einsum`.

```python {.marimo}
import numpy as np

from discopy.tensor import Dim, Box, Diagram, Variable, Tuple
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
_v = np.array([.3, .7])
_state = Box("v", Dim(1), N, _v)
_copied = {name: (_state >> contraction(N, spider)).eval().array
           for name, spider in [("k", k_extension), ("basis", basis_copy)]}
assert np.allclose(_copied["k"], np.outer(_v, [1, 1]) + np.outer([1, 1], _v))
assert np.allclose(_copied["basis"], np.diag(_v))
assert np.allclose(
    (_state >> iota(N) >> epsilon(N)).eval().array, _v), "ε ∘ ι = id"
mo.md("W copies `v` to `v ⊗ k + k ⊗ v` and Z copies it to `Σᵢ vᵢ eᵢ ⊗ eᵢ`.")
```

## 2. The lexicon

A word is a variable of a type built with `<<`, where `y << x` is the
dimension `y @ x.l`. A function word is applied to its argument by a cup.
A variable used twice is copied with the `copy` of the diagram class the
term is evaluated in, so the two copies of the paper are two subclasses of
`tensor.Diagram`. The diagram of a reading is then the paper's map, from its
words to `S ⊗ S`, and its meaning is that map applied to the word states.
Every `!` of the paper's types is a Fock space that a contraction copies and
a dereliction `ε` reads; a `!` that is only derelicted contributes
`ε ∘ ι = id`, so it is left out.

| word | type of §3.3 | here |
|---|---|---|
| John, Bill | `!N` | `N` |
| likes | `!((!(!N\S))/N)` | `(S << N) << N`: object, then subject |
| his | `(!(!N\N))/N` | `(N << N) << N`: possessed, then owner |
| code | `N` | `N` |
| does too | `(!(!N\S))\(!N\S)` | `(S << N) << (S << N)` |

```python {.marimo}
class KExtension(Diagram):
    @classmethod
    def copy(cls, x, n=2):
        assert n == 2
        return contraction(x, k_extension)


class BasisCopy(Diagram):
    @classmethod
    def copy(cls, x, n=2):
        assert n == 2
        return contraction(x, basis_copy)


copies = {"k-extension": KExtension, "basis copy": BasisCopy}
VP = S << N
types = {"John": N, "Bill": N, "Mary": N, "code": N, "guitar": N,
         "sleeps": VP, "snores": VP, "He": N << N, "plays": VP << N,
         "likes": VP << N, "his": (N << N) << N, "does too": VP << VP}
John, Bill, Mary, code, guitar, sleeps, snores, He, plays, likes, his, \
    does_too = (Variable(name, typ) for name, typ in types.items())
```

The arrays `a` of the words are drawn with their arguments first, the order
the formulas below use, and transposed into the order of their types.

```python {.marimo}
rng = np.random.default_rng(0)
shapes = {"N": (2, ), "VP": (2, 2), "He": (2, 2), "V": (2, 2, 2),
          "his": (2, 2, 2), "does too": (2, 2, 2, 2)}
kinds = {"sleeps": "VP", "snores": "VP", "He": "He", "plays": "V",
         "likes": "V", "his": "his", "does too": "does too"}
order = {"VP": (1, 0), "He": (1, 0), "V": (2, 1, 0), "his": (2, 1, 0),
         "does too": (3, 2, 0, 1)}
a = {name: rng.random(shapes[kinds.get(name, "N")]) for name in types}
lexicon = {name: Box(name, Dim(1), typ, a[name].transpose(
    order.get(kinds.get(name), None))) for name, typ in types.items()}


def meaning(term, copy):
    """The map of a reading, applied to the states of its words."""
    words = Diagram.id(Dim(1)).tensor(
        *(lexicon[var.name] for var in term.freevars))
    return words >> term.eval(copy)
```

## 3. Anaphora: *John sleeps. He snores.*

`John` is used twice, so it is copied: one copy is the subject of *sleeps*
and the other goes through *He* to *snores* (figure 11 of the paper). With
k-extension the paper finds
`sleeps(ε(John)) ⊗ snores(He(k)) + sleeps(ε(k)) ⊗ snores(He(John))`.

```python {.marimo}
anaphora = Tuple(sleeps(John), snores(He(John)))
meaning(anaphora, KExtension).draw(figsize=(6, 6))
```

```python {.marimo}
ones = np.ones(2)
_j, _sl, _sn, _he = a["John"], a["sleeps"], a["snores"], a["He"]
_out = meaning(anaphora, KExtension).eval().array
assert np.allclose(_out, np.outer(_j @ _sl, (ones @ _he) @ _sn)
                   + np.outer(ones @ _sl, (_j @ _he) @ _sn))
_out = meaning(anaphora, BasisCopy).eval().array
assert np.allclose(_out, np.einsum("i,is,ij,jt->st", _j, _sl, _he, _sn))
mo.md("Both copies give the paper's map of §5.1.")
```

## 4. Ellipsis: *John plays guitar. Mary does too.*

The verb phrase *plays guitar* is abstracted and used twice: one copy is
applied to *John*, the other goes through *does too* to *Mary* (figure 12).
With k-extension the paper finds
`pg(John) ⊗ does_too(k)(Mary) + k(John) ⊗ does_too(pg)(Mary)`, where
`pg = plays(guitar)`.

```python {.marimo}
ellipsis = VP(lambda vp: (vp(John), does_too(vp)(Mary)))(plays(guitar))
meaning(ellipsis, KExtension).draw(figsize=(6, 6))
```

```python {.marimo}
_j, _m, _dt = a["John"], a["Mary"], a["does too"]
_pg = np.einsum("o,ons->ns", a["guitar"], a["plays"])
K = np.ones((2, 2))


def says(vp, subject):
    return np.einsum("n,ns->s", subject, vp)


def too(vp, subject):
    return np.einsum("m,ns,nsmt->t", subject, vp, _dt)


assert np.allclose(meaning(ellipsis, KExtension).eval().array,
                   np.outer(says(_pg, _j), too(K, _m))
                   + np.outer(says(K, _j), too(_pg, _m)))
assert np.allclose(meaning(ellipsis, BasisCopy).eval().array, np.einsum(
    "n,ns,nsmt,m->st", _j, _pg, _dt, _m))
mo.md("Both copies give the paper's map of §5.2.")
```

## 5. Anaphora with ellipsis: strict and sloppy

**Strict** (figures 3 and 13): two contractions. *John* is used twice, as
the subject and as the owner in *his code*, and the verb phrase *likes
John's code* is used twice, once for *John* and once for *does too* and
*Bill*.

```python {.marimo}
strict = VP(lambda vp: (vp(John), does_too(vp)(Bill)))(
    likes(his(code)(John)))
mo.md(f"`{strict}`")
```

```python {.marimo}
meaning(strict, KExtension).draw(figsize=(8, 9))
```

**Sloppy** (figures 4 and 14): four contractions. *his code* is
abstracted and used twice, and *John*, *Bill* and *likes* are each used
twice, so each subject owns its own code.

```python {.marimo}
sloppy = (N << N)(lambda h: (
    likes(h(John))(John), does_too(likes(h(Bill)))(Bill)))(his(code))
mo.md(f"`{sloppy}`")
```

```python {.marimo}
meaning(sloppy, KExtension).draw(figsize=(9, 11))
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


for _cls, _copy in [(KExtension, k_copy), (BasisCopy, b_copy)]:
    assert np.allclose(meaning(strict, _cls).eval().array,
                       strict_formula(_copy))
    assert np.allclose(meaning(sloppy, _cls).eval().array,
                       sloppy_formula(_copy))
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
assert np.allclose(meaning(strict, KExtension).eval().array, sum(
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
readings = {name: (meaning(strict, cls).eval().array.real,
                   meaning(sloppy, cls).eval().array.real)
            for name, cls in copies.items()}
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
