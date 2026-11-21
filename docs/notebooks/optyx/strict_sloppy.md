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

Each reading is a term of `discopy.bang`, the planar kappa calculus with a
bang: a type called on a function, `N.bang()(lambda x: ...)`, binds `x`, and
a variable of banged type `!A` used twice is copied by a `Contraction` and
read by a `Dereliction`, the two rules of `!L*` that the readings use. A
`bang.Functor` sends the term to tensors, with `!A` as the Fock space of `A`,
the dereliction as `ε` and the contraction as an optyx diagram evaluated as
a tensor. Words are vectors drawn at random with a fixed seed. Every reading
is checked against the paper's formulas, written out independently with
`numpy.einsum`.

```python {.marimo}
import numpy as np

from discopy import bang, tensor
from discopy.bang import (
    Ty, Box, BangOb, Contraction, Dereliction, Exchange)
from discopy.tensor import Dim
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
    return tensor.Box("ι", A, fock(A), array.reshape(A.inside + (-1, )))


def epsilon(A):
    array = np.zeros((2 ** size(A), size(A)))
    array[photon(A), range(size(A))] = 1
    array[0] = k
    return tensor.Box("ε", fock(A), A, array.reshape((-1, ) + A.inside))


def layer(A, spider):
    d = size(A)
    return optyx.Diagram.tensor(*d * [spider]) >> optyx.Diagram.permutation(
        [2 * m for m in range(d)] + [2 * m + 1 for m in range(d)],
        optyx.Mode(2 * d))


def delta(A, spider):
    array = layer(A, spider).to_tensor(input_dims=size(A) * [2]).eval().array
    return tensor.Box(f"Δ({spider.name})", fock(A), fock(A) @ fock(A),
                      array.real.reshape(3 * (2 ** size(A), )))


k_extension, basis_copy = W(2), ZBox(1, 2, lambda _: 1)
copies = {"k-extension": k_extension, "basis copy": basis_copy}
layer(Dim(2), k_extension).draw(figsize=(4, 4))
```

On two modes, the W layer above is the k-extension and the Z layer below is
the basis copy:

```python {.marimo}
layer(Dim(2), basis_copy).draw(figsize=(4, 4))
```

```python {.marimo}
_v = np.array([.3, .7])
_state = tensor.Box("v", Dim(1), Dim(2), _v)
_copied = {name: (_state >> iota(Dim(2)) >> delta(Dim(2), spider)
                  >> epsilon(Dim(2)) @ epsilon(Dim(2))).eval().array
           for name, spider in copies.items()}
assert np.allclose(_copied["k-extension"],
                   np.outer(_v, [1, 1]) + np.outer([1, 1], _v))
assert np.allclose(_copied["basis copy"], np.diag(_v))
assert np.allclose(
    (_state >> iota(Dim(2)) >> epsilon(Dim(2))).eval().array, _v), "ε ∘ ι = id"
mo.md("W copies `v` to `v ⊗ k + k ⊗ v` and Z copies it to `Σᵢ vᵢ eᵢ ⊗ eᵢ`.")
```

## 2. The lexicon

A word is a box of `discopy.bang`. A function type of the paper is an atom,
and the elimination rule of `!L*`, from `A ⊗ A\B` to `B`, is a box that
contracts the argument with the first index of the function. A word whose
type has a `!` outside outputs a banged wire, which a term may copy; a `!` on
an argument is read by a dereliction before the argument is passed, and
`ε ∘ ι = id`, so it is left out of the boxes.

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
John, Bill, Mary = (Box(name, Ty(), N.bang()) for name in ("John", "Bill", "Mary"))
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
and those of the elimination boxes are identities.

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
      "obj": np.einsum("or,nm,st->ornsmt", _I, _I, _I),
      "and": np.einsum("su,tv->stuv", _I, _I)}
```

A `bang.Functor` into `tensor.Diagram` sends `!A` to `F(A)`, a dereliction
to `ε`, a contraction to `Δ` with the chosen spider, an exchange to a swap
and a word to its array, followed by `ι` when its output is banged.

```python {.marimo}
dim = {N: Dim(2), S: Dim(2), VP: Dim(2, 2), H: Dim(2, 2), V: Dim(2, 2, 2)}


def ob(x):
    wire, = x.inside
    return fock(dim[wire.arg]) if isinstance(wire, BangOb) else dim[x]


def dims(typ):
    return Dim(1).tensor(*(ob(typ[i:i + 1]) for i in range(len(typ))))


def ar(spider):
    def image(box):
        if isinstance(box, Dereliction):
            return epsilon(dim[box.x])
        if isinstance(box, Contraction):
            return delta(dim[box.x], spider)
        if isinstance(box, Exchange):
            split = -1 if box.is_dagger else 1
            return tensor.Diagram.swap(
                dims(box.dom[:split]), dims(box.dom[split:]))
        banged = isinstance(box.cod.inside[0], BangOb)
        cod = dim[box.cod.inside[0].arg] if banged else dims(box.cod)
        value = tensor.Box(box.name, dims(box.dom), cod, a[box.name])
        return value >> iota(cod) if banged else value
    return image


def meaning(term, spider):
    """The tensor of a reading, with contraction given by a spider."""
    functor = bang.Functor(ob_map=ob, ar_map=ar(spider), cod=tensor.Diagram)
    return term.eval(functor).eval().array
```

## 3. Anaphora: *John sleeps. He snores.*

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

## 4. Ellipsis: *John plays guitar. Mary does too.*

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

## 5. Anaphora with ellipsis: strict and sloppy

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
