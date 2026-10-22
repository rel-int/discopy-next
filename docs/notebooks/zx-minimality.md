---
title: ZX Minimality
marimo-version: 0.23.14
---

```python {.marimo}
import marimo as mo
```

# Minimality of the pure qubit ZX calculus

[Vilmart (LICS 2019)](https://arxiv.org/abs/1812.09114) gave a
*near-optimal* complete axiomatisation $\mathrm{ZX_V}$ of the pure qubit ZX
calculus: nine rules, each shown necessary except three, the bialgebra
$(B)$ and the two identity rules $(I_g)$ and $(I_r)$, of which he showed
only that one at least is needed. [Stoltz & Vilmart, *Minimality of the
Pure Qubit ZX Calculus* (arXiv:2608.14872)](https://arxiv.org/abs/2608.14872)
close the question: the red identity $(I_r)$ is derivable from the other
rules, and once it is dropped every remaining rule is necessary, in
$\mathrm{ZX_V}$ and in a second presentation $\mathrm{ZX'_V}$ alike. The two
hard cases are the bialgebra, broken by an interpretation over the dual
numbers $\mathbb{C}[\varepsilon]/(\varepsilon^2)$ where the Hadamard is
perturbed by a nilpotent skew rotation, and the green identity, broken by
an interpretation into relations over the Booleans.

This notebook merges two independent computational accounts of that result:

* [discopy#626](https://github.com/discopy/discopy/pull/626) reproduces the
  paper — rulesets, interpretations, soundness, the derivability of $(I_r)$
  lemma by lemma, both necessity theorems down to the exact witness — and
  generalises its bialgebra countermodel: modulo perturbations that do
  nothing, the paper's is the only one on three qubits, the least possible
  size.
* [discopy#688](https://github.com/discopy/discopy/pull/688) proves the same
  theorem for $\mathrm{ZX_V}$ from Vilmart's 2019 paper and sources
  available before 2020 only, and so comes with its own proofs: a separating
  model for *every* rule, a finite countermodel for the bialgebra over the
  complex numbers, checked exactly, and a shorter, basis-free derivation of
  $(I_r)$.

Where the two proofs differ both are kept, side by side, and run against the
same rules. Running every model against both rulesets gives one thing
neither had alone: a separating model for each of the seven rules of
$\mathrm{ZX'_{opt}}$ as well.

```python {.marimo}
import math
from cmath import phase
from itertools import combinations

import numpy as np
from discopy import frobenius, symmetric, tensor
from discopy.tensor import Dim
from discopy.quantum import zx
from discopy.quantum.zx import Z, X, H, SWAP, Id, Diagram
from discopy.monoidal import Equation
```

## The rules as pairs of diagrams

A ZX diagram is built from green spiders `Z(n, m, phase)`, red spiders
`X(n, m, phase)`, the Hadamard `H` and swaps. Phases are in **turns**, as
`zx.Diagram.eval` reads them: `Z(1, 1, .5)` is the Pauli $Z$ and $\pi/2$ is
`0.25` (discopy#688 wrote them in half-turns, translated here). A rule is a
pair of diagrams with the same domain and codomain. Scalars are diagrams
with no wires; the rules use three scalar gadgets: the *dumbbell* $\lambda$
worth $\sqrt{2}$, the triple-edged dumbbell $\theta$ worth
$\frac{1}{\sqrt{2}}$ and the *global phase* pair worth
$\sqrt{2}e^{i\gamma}$.

```python {.marimo}
lam = X(0, 1) >> Z(1, 0)
theta = X(0, 3) >> Z(3, 0)
lam_pi = X(0, 1, .5) >> Z(1, 0)
global_phase = lambda gamma: X(0, 1, .5) >> Z(1, 0, gamma)

mo.vstack([Equation(lam, theta, global_phase(.1), symbol="  ,  ")])
```

The only rules with a side condition are the Euler decompositions $(EU)$
and $(EU')$, which decompose a green-red-green rotation (respectively a
green-Hadamard-green one) into a red-green-red one. The angles
$\beta_1, \beta_2, \beta_3$ and the global phase $\gamma$ are given by the
formulas of Figures 2 and 3 of the paper, transcribed in radians and checked
against the matrix identities they encode. Taking determinants gives a
necessary condition, noted in discopy#688:
$\alpha_1 + \alpha_2 + \alpha_3 = \beta_1 + \beta_2 + \beta_3 + 2\gamma$
for $(EU)$, and $\alpha_1 + \alpha_2 + \pi = \beta_1 + \beta_2 + \beta_3 +
2\gamma$ for $(EU')$, since $\det H = e^{i\pi}$.

```python {.marimo}
def euler(a1, a2, a3):
    """The side condition of (EU), angles in radians."""
    xp, xm = (a1 + a3) / 2, (a1 - a3) / 2
    z = np.cos(a2 / 2) * np.cos(xp) + 1j * np.sin(a2 / 2) * np.cos(xm)
    zp = np.cos(a2 / 2) * np.sin(xp) - 1j * np.sin(a2 / 2) * np.sin(xm)
    b2 = 0 if abs(zp) < 1e-12 else 2 * phase(1j + abs(z / zp))
    return phase(z) + phase(zp), b2, phase(z) - phase(zp),\
        xp - phase(z) + (a2 - b2) / 2


def euler_prime(a1, a2):
    """The side condition of (EU'), angles in radians."""
    xp, xm = (a1 + a2) / 2, (a1 - a2) / 2
    z = -np.sin(xp) + 1j * np.cos(xm)
    zp = np.cos(xp) - 1j * np.sin(xm)
    b2 = 0 if abs(zp) < 1e-12 else 2 * phase(1j + abs(z / zp))
    return phase(z) + phase(zp), b2, phase(z) - phase(zp),\
        xp - phase(z) + (math.pi - b2) / 2
```

```python {.marimo}
hadamard_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
diagonal_phase = lambda angle: np.diag([1, np.exp(1j * angle)])
red_green_red = lambda b1, b2, b3: (
    hadamard_matrix @ diagonal_phase(b3) @ hadamard_matrix
    @ diagonal_phase(b2) @ hadamard_matrix @ diagonal_phase(b1)
    @ hadamard_matrix)
on_circle = lambda angle: np.isclose(np.exp(1j * angle), 1)

for _i in range(100):
    _a1, _a2, _a3 = np.random.default_rng(_i).uniform(-math.pi, math.pi, 3)
    _b1, _b2, _b3, _g = euler(_a1, _a2, _a3)
    assert np.allclose(
        diagonal_phase(_a3) @ hadamard_matrix @ diagonal_phase(_a2)
        @ hadamard_matrix @ diagonal_phase(_a1),
        np.exp(1j * _g) * red_green_red(_b1, _b2, _b3))
    assert on_circle(_a1 + _a2 + _a3 - _b1 - _b2 - _b3 - 2 * _g)
    _b1, _b2, _b3, _g = euler_prime(_a1, _a2)
    assert np.allclose(
        diagonal_phase(_a2) @ hadamard_matrix @ diagonal_phase(_a1),
        np.exp(1j * _g) * red_green_red(_b1, _b2, _b3))
    assert on_circle(_a1 + _a2 + math.pi - _b1 - _b2 - _b3 - 2 * _g)
mo.md("The transcribed side conditions hold on the nose, determinants "
      "included. ✓")
```

Now the rules themselves, transcribed from Figures 2 and 3. The rules
$(S)$, $(EU)$, $(EU')$, $(IV)$ and $(H)$ are schemas — one rule per choice
of arities and phases — so each is a function and we check a family of
instances.

```python {.marimo}
def fusion(n1, m1, n2, m2, k, a, b):
    """(S): two green spiders fuse along k >= 1 wires."""
    return (Z(n1, m1 + k, a) @ Id(n2) >> Id(m1) @ Z(k + n2, m2, b),
            Z(n1 + n2, m1 + m2, a + b))


def colour_change(n, m, a):
    """(H): a red spider with Hadamards on every leg is a green spider."""
    return (Id().tensor(*(n * [H])) >> X(n, m, a) >> Id().tensor(*(m * [H])),
            Z(n, m, a))


def turns(angles):
    """Radians to turns."""
    return (angle / (2 * math.pi) for angle in angles)


def eu(a1, a2, a3):
    """(EU): the Euler decomposition of a green-red-green rotation."""
    b1, b2, b3, g = turns(euler(*(2 * math.pi * a for a in (a1, a2, a3))))
    return (Z(1, 1, a1) >> X(1, 1, a2) >> Z(1, 1, a3),
            theta @ global_phase(g)
            @ (X(1, 1, b1) >> Z(1, 1, b2) >> X(1, 1, b3)))


def eu_prime(a1, a2):
    """(EU'): the Euler decomposition of a green-Hadamard-green rotation."""
    b1, b2, b3, g = turns(euler_prime(*(2 * math.pi * a for a in (a1, a2))))
    return (Z(1, 1, a1) >> H >> Z(1, 1, a2),
            theta @ global_phase(g)
            @ (X(1, 1, b1) >> Z(1, 1, b2) >> X(1, 1, b3)))


iv = lambda a: ((X(0, 1) >> Z(1, 0, a)) @ theta, Id(0))

Ig_rule = Z(1, 1), Id(1)
Ir_rule = X(1, 1), Id(1)
E_rule = Z(0, 1, 1 / 8) >> X(1, 0, -1 / 8), Id(0)
CP_rule = lam @ (X(0, 1) >> Z(1, 2)), X(0, 1) @ X(0, 1)
B_rule = (lam @ (Z(1, 2) @ Z(1, 2) >> Id(1) @ SWAP @ Id(1)
                 >> X(2, 1) @ X(2, 1)), X(2, 1) >> Z(1, 2))
HD_rule = (Z(1, 1, 1 / 4) >> Id(1) @ Z(0, 1, -1 / 4) >> X(2, 1)
           >> Z(1, 1, 1 / 4), H)
```

```python {.marimo}
class Rewrite(frobenius.Functor):
    """An endofunctor on Diagram that rewrites some boxes and leaves every
    other box, in particular a swap, unchanged."""
    def __init__(self):
        super().__init__(
            ob_map=lambda x: x, ar_map=None, dom=Diagram, cod=Diagram)

    def rewrite(self, box):
        """The image of a box this functor rewrites, else None."""
        raise NotImplementedError

    def __call__(self, other):
        rewritten = self.rewrite(other)
        if rewritten is not None:
            return rewritten
        if isinstance(other, zx.Box) and not isinstance(other, zx.Swap):
            return other
        return super().__call__(other)


class Round(Rewrite):
    """Round the spider phases of a diagram, for display only."""
    def rewrite(self, box):
        if isinstance(box, zx.Spider):
            return type(box)(
                len(box.dom), len(box.cod), round(float(box.phase), 3))
        return None


rounded = Round()

mo.vstack([
    Equation(*map(rounded, B_rule), symbol="=(B)="),
    Equation(*map(rounded, HD_rule), symbol="=(HD)="),
    Equation(*map(rounded, eu(.1, .2, .3)), symbol="=(EU)=")])
```

The two rulesets of the paper are $\mathrm{ZX_{opt}}$, Vilmart's
$\mathrm{ZX_V}$ without $(I_r)$, and $\mathrm{ZX'_{opt}}$, his
$\mathrm{ZX'_V}$ without it, where $(IV)$ and $(EU')$ replace $(E)$, $(HD)$
and $(EU)$. The instance $(\frac18, \frac12, \frac18)$ of $(EU)$ is the one
discopy#688 uses below.

```python {.marimo}
fusion_cases = [(1, 1, 1, 1, 1, .1, .2), (2, 0, 1, 2, 2, .3, -.1),
                (0, 1, 2, 1, 1, .7, .4), (1, 2, 0, 1, 2, .25, .5),
                (0, 0, 0, 0, 1, .1, .9), (1, 1, 0, 0, 2, 0., 0.)]
colour_cases = [(0, 1, .3), (1, 1, .1), (2, 1, .25), (1, 2, -.4),
                (2, 2, .5), (1, 0, 0.), (1, 1, 0.), (2, 0, 0.), (0, 2, 0.)]
eu_cases = [(.1, .2, .3), (.4, -.3, .2), (0., .5, 0.), (.25, .25, .25),
            (-.4, .45, .15), (.5, .5, .5), (.125, .5, .125)]
eu_prime_cases = [(.1, .2), (.4, -.3), (0., 0.), (.25, -.25), (.5, .5)]
iv_cases = [(0., ), (.3, ), (.5, ), (-.25, )]

instances = lambda rule, cases: [rule(*case) for case in cases]

rules = {
    "S": instances(fusion, fusion_cases), "Ig": [Ig_rule], "Ir": [Ir_rule],
    "CP": [CP_rule], "B": [B_rule], "HD": [HD_rule],
    "H": instances(colour_change, colour_cases), "E": [E_rule],
    "EU": instances(eu, eu_cases), "IV": instances(iv, iv_cases),
    "EU'": instances(eu_prime, eu_prime_cases)}
ZX_opt = ["S", "Ig", "CP", "B", "HD", "H", "E", "EU"]
ZX_opt_prime = ["S", "Ig", "CP", "B", "H", "IV", "EU'"]


def verdicts(close, names=tuple(rules)):
    """Which rules hold, all their instances, for a predicate ``close``."""
    return {name: all(close(*rule) for rule in rules[name])
            for name in names}
```

## Interpretations are functors

An interpretation sends each wire to a space and each generator to a
tensor. In DisCoPy it is a `symmetric.Functor` into `tensor.Diagram`, given
the array of each generator: the image of a diagram is a tensor network,
evaluated in one `einsum`, and the swaps come for free. A symmetric functor
reads every spider as a box like any other, so its array is ours to choose,
where a `frobenius.Functor` would send a phaseless spider to the spider of
the codomain.

Most of the models below are *phase models*: a dimension, a family of
diagonal phases for the green spiders and a matrix for the Hadamard, with
red spiders the Hadamard-conjugated green ones, as in Section 3.4 of the
paper.

```python {.marimo}
class Interpretation(symmetric.Functor):
    """A ZX diagram as a tensor network, from the array of each generator."""
    def __init__(self, dim, arrow, dtype=complex):
        self.dim, self.arrow, self.dtype = dim, arrow, dtype
        super().__init__(ob_map=lambda _: Dim(dim), ar_map=self.box,
                         dom=Diagram, cod=tensor.Diagram[dtype])

    def box(self, box):
        """The tensor box of a generator."""
        return tensor.Box[self.dtype](
            box.name, Dim(self.dim) ** len(box.dom),
            Dim(self.dim) ** len(box.cod),
            np.asarray(self.arrow(box), dtype=self.dtype))

    def evaluate(self, diagram):
        """The array a diagram evaluates to."""
        return np.asarray(self(diagram).eval().array)

    def close(self, lhs, rhs):
        """Whether two diagrams evaluate to the same array."""
        return np.allclose(self.evaluate(lhs), self.evaluate(rhs), atol=1e-9)


def on_legs(matrix, array, n_inputs):
    """Apply ``matrix`` on every leg of an array, inputs then outputs."""
    for axis in range(array.ndim):
        array = np.moveaxis(np.tensordot(
            matrix, array, axes=([1 if axis < n_inputs else 0], [axis])),
            0, axis)
    return array


def green(dim, n, m, phases):
    """The green spider with ``n`` inputs, ``m`` outputs and diagonal
    ``phases``."""
    if not n + m:
        return np.sum(phases)
    array = np.zeros((dim, ) * (n + m), dtype=complex)
    for i, scalar in enumerate(phases):
        array[(i, ) * (n + m)] = scalar
    return array


def phase_arrow(dim, phases, hadamard):
    """The arrays of a phase model."""
    def arrow(box):
        n, m = len(box.dom), len(box.cod)
        if isinstance(box, (Z, X)):
            array = green(dim, n, m, phases(float(box.phase)))
            return on_legs(hadamard, array, n) if isinstance(box, X) \
                else array
        if box.name == 'H':
            return hadamard
        raise TypeError(box)
    return arrow


def phase_model(dim, phases, hadamard):
    """The phase model given by a family of phases and a Hadamard."""
    return Interpretation(dim, phase_arrow(dim, phases, hadamard))
```

The standard interpretation sends a wire to $\mathbb{C}^2$, the green
spider to $|0^m\rangle\langle 0^n| + e^{i\alpha}|1^m\rangle\langle 1^n|$
and the Hadamard to the Hadamard: it is what `zx.Diagram.eval` computes,
and the phase model with the qubit phases agrees with it on both sides of
every rule. Both rulesets are sound — every rule holds, and so does
$(I_r)$:

```python {.marimo}
standard = phase_model(
    2, lambda a: [1, np.exp(2j * math.pi * a)], hadamard_matrix)

for _name, _cases in rules.items():
    for _rule in _cases:
        for _side in _rule:
            assert np.allclose(standard.evaluate(_side), _side.eval().array)
assert all(verdicts(
    lambda lhs, rhs: np.allclose(lhs.eval().array, rhs.eval().array))
    .values())
mo.md(r"Soundness of $\mathrm{ZX_{opt}}$, $\mathrm{ZX'_{opt}}$ and "
      r"$(I_r)$ under `zx.Diagram.eval`. ✓")
```

## Completeness: the red identity is derivable

Vilmart proved $\mathrm{ZX_V}$ and $\mathrm{ZX'_V}$ complete, so what
remains for completeness is to derive $(I_r)$ from the other rules. The two
upstream accounts give two derivations.

**The paper's** (Theorems 2.4 and 5.1) is a chain of lemmas, Appendices A
and C, each a rewrite between two diagrams. discopy#626 transcribes the
statement of every lemma and checks it in the standard interpretation, so
a mistranscribed or unsound lemma would fail here; the derivations
themselves are sequences of rule applications, in the paper.

```python {.marimo}
hang = lambda state: Id(1) @ state >> X(2, 1)

lemmas_A = [
    ("H0", H >> X(1, 1) >> H, Id(1)),
    ("HC", X(1, 1) >> H, H >> X(1, 1)),
    ("LIr", theta @ lam_pi @ X(1, 1), Id(1)),
    ("H-1", H >> H, theta @ lam_pi @ Id(1)),
    ("LSr", theta @ lam_pi @ (X(1, 2, .1) >> Id(1) @ X(1, 1, .2)),
     X(1, 2, .3)),
    ("LSr", theta @ lam_pi @ (X(2, 2, .3) >> X(2, 1, .4)), X(2, 1, .7)),
    ("HF", lam @ lam @ (X(1, 2) >> Z(2, 1)), X(1, 0) >> Z(0, 1)),
    ("IV'", lam @ theta, Id(0)),
    ("IV''", Id(0), theta @ lam_pi @ theta @ lam_pi),
    ("D-", X(0, 1) >> Z(1, 1, -.25), X(0, 1)),
    ("D'pi", X(0, 1) >> Z(1, 0, .5), X(0, 1) >> Z(1, 0)),
    ("Ir", *Ir_rule)]

lemmas_C = [
    ("HX", X(1, 1, .3), H >> Z(1, 1, .3) >> H),
    ("HX", X(2, 1, .3), H @ H >> Z(2, 1, .3) >> H),
    ("RT", hang(X(0, 1)) >> hang(X(0, 1)), X(1, 1) >> X(1, 1)),
    ("ZC", Z(1, 1, .2) >> H >> H, H >> H >> Z(1, 1, .2)),
    ("X+-", X(1, 1, -.125) >> X(1, 1, .125), Id(1)),
    ("HH", H >> H, Id(1)),
    ("Ir", *Ir_rule)]

for _name, _lhs, _rhs in lemmas_A + lemmas_C:
    assert standard.close(_lhs, _rhs), _name
mo.md("Every lemma of Appendices A and C is sound. ✓")
```

**discopy#688's** works in $\mathrm{ZX_{opt}}$ and packages the same
obstruction algebraically. Write $R = X_{1,1}$, $q = \lambda$ the dumbbell,
$q_a = X^a_{0,1};Z_{1,0}$, $c = \theta$ and $\omega = q_\pi c$.

1. The degree-two $(H)$ instance and $(I_g)$ give $HRH = 1$: $H$ has a left
   and a right inverse, so it is invertible and $R = H^{-2}$.
2. The zero-angle $(EU)$ instance, whose side condition gives
   $\beta_1 = \beta_2 = \beta_3 = \gamma = 0$, reads $R = \omega R^2$;
   cancelling the invertible $R$ gives $\omega R = 1$. With $(E)$, which
   exhibits the unit as a retract of a wire, $\omega$ is an invertible
   scalar, and so are $q_\pi$ and $c$.
3. Write $\rho = \omega^{-1}$, so $R = \rho$: colour change turns green
   fusion into a *projective* red fusion, joining two red spiders along $k$
   legs costing $\rho^k$.
4. Postcomposing $(CP)$ with an explicit effect shows $q$ invertible.
   Precomposing $(B)$ with a red state and using projective fusion on both
   sides gives $\rho^4 = \rho^2$, hence $\rho^2 = 1$.
5. Vilmart's own appendix derives the Hopf law $q^2(\delta;\mu) =
   \epsilon;x$ from $(S)$, $(CP)$ and $(B)$, using $(I_r)$ only in two
   red-unit reductions, each worth $\rho^2 = 1$ by projective fusion. Its
   trace gives $q^2 c = q$, hence $qc = 1$.
6. A phase character $\chi_a$, with $q_a = q\chi_a$, is trivial at
   $a = \pi/2$ by $(HD)$, so $q_\pi = q$. Hence $\omega = q_\pi c = qc = 1$
   and, by step 2, $R = 1$.

The two proofs track the same scalar anomaly through the same lemmas,
which is what makes them comparable: step 2 is the paper's $(LI_r)$, step 5
its $(IV')$, step 6 its $(D'_\pi)$ up to where the phase sits, and $\omega^2 = 1$ its $(IV'')$. The
paper's chain makes every scalar movement a visible rewrite; discopy#688's
is shorter and names the mechanism — invertibility, projective fusion, a
Hopf trace and a phase character — at the cost of citing Vilmart's printed
Hopf derivation for step 5. The diagrams it uses, among them the effect
inverting $q$ and the traced Hopf law:

```python {.marimo}
split_effect = H >> Z(1, 1, .125) >> X(1, 0, -.125)
copy_scalar_inverse = X(0, 1) >> Z(1, 2) >> split_effect @ split_effect
hopf_left = lam @ lam @ (Z(1, 2) >> X(2, 1))
hopf_right = Z(1, 0) >> X(0, 1)

_steps = [
    ("step 2 = LIr", theta @ lam_pi @ X(1, 1), Id(1)),
    ("step 2, zero angle", *eu(0, 0, 0)),
    ("step 4", X(0, 1) >> split_effect, Id(0)),
    ("step 4", lam @ copy_scalar_inverse, Id(0)),
    ("step 5", hopf_left, hopf_right),
    ("step 5, traced", hopf_left.trace(), hopf_right.trace()),
    ("step 5 = IV'", lam @ theta, Id(0)),
    ("step 6 = D'pi", lam_pi, lam),
    ("omega = 1", lam_pi @ theta, Id(0))]
for _name, _lhs, _rhs in _steps:
    assert standard.close(_lhs, _rhs), _name
mo.md("Every diagram of the algebraic derivation is sound. ✓")
```

Why no simpler argument works shows in a *near*-countermodel: scale the
Hadamard to $iH$ and every red spider of degree $d$ by $(-i)^d$. This
satisfies $(S)$, $(I_g)$, $(CP)$, $(B)$, $(HD)$ and $(H)$ and has
$R = -1$, but it breaks $(E)$ and $(EU)$ as well as $(I_r)$: the scalar
rules are exactly what forces $\rho = 1$. More generally, discopy#688 shows
that no central grading — every generator multiplied by a scalar affine in
its degree and its phase — can separate $(I_r)$: one wire fusion forces the
form $w(Z^a_d) = u(d - 2) + f(a)$, $w(X^a_d) = (u - h)d - 2u + f(a)$,
$w(H) = h$, multiwire fusion forces $2u = 0$, and then the residual of $(E)$
is $-h$, so $(E)$ alone kills the scale $h$ of the Hadamard that
$(I_r)$ and $(B)$ would need, their residuals being $-2h$ and $-4h$.

```python {.marimo}
def projective_arrow(box):
    """The near-countermodel isolating the projective obstruction."""
    degree = len(box.dom) + len(box.cod)
    array = phase_arrow(
        2, lambda a: [1, np.exp(2j * math.pi * a)], hadamard_matrix)(box)
    if isinstance(box, X):
        return (-1j) ** degree * array
    return 1j * array if box.name == 'H' else array


projective = Interpretation(2, projective_arrow)
projective_verdicts = verdicts(projective.close)
assert [name for name, ok in projective_verdicts.items() if not ok]\
    == ["Ir", "E", "EU", "EU'"]
mo.md(r"The projective model breaks $(I_r)$, $(E)$, $(EU)$ and $(EU')$ "
      "and nothing else. ✓")
```

## Minimality: one countermodel per rule

A rule is necessary when some interpretation satisfies every other rule
and violates it. The paper builds the two that were open, for $(B)$ and
$(I_g)$, and cites Vilmart for the other six; discopy#688 builds one for
every rule of $\mathrm{ZX_{opt}}$. Each model below is a predicate on pairs
of diagrams, and `verdicts` runs it on every instance of every rule.

### Gradings and connectivity: $(S)$, $(H)$, $(CP)$, $(E)$, $(EU)$

Five of the rules are separated by models with almost nothing in them
(discopy#688), each a compact functor into a one-object category, so that
wires, swaps, cups and caps cost nothing:

* $(S)$: the parity of the number of spiders of degree at least four. No
  other rule has such a spider on one side and not the other — $(H)$ changes
  the colour, not the degree — while fusing two trivalent spiders along one
  wire makes one.
* $(H)$: the parity of the number of *red* spiders of degree at least four,
  which works because the fusion rule is green only.
* $(EU)$: the sum of the phases in $\mathbb{R}/\frac14\mathbb{Z}$, i.e.
  modulo $\pi/2$, the Hadamard counting zero. $(HD)$ has phase sum
  $\frac14 = 0$, every other rule preserves it, while the $(EU)$ instance
  $(\frac18, \frac12, \frac18)$ has $(\beta_1, \beta_2, \beta_3, \gamma) =
  (\frac14, 0, \frac14, \frac18)$ and its two sides differ by
  $\frac18 \ne 0$. It is the universal additive phase grading with the
  Hadamard at zero: $(HD)$ kills $\frac14$ and the $(EU)$ instances kill
  the rest.

```python {.marimo}
def high_degree_parity(spider_type):
    """The parity of the spiders of degree at least four."""
    def parity(diagram):
        return sum(
            isinstance(box, spider_type) and len(box.dom) + len(box.cod) >= 4
            for box in diagram.boxes) % 2
    return lambda lhs, rhs: parity(lhs) == parity(rhs)


def phase_sum(diagram):
    """The sum of the phases of the spiders of a diagram, in turns."""
    return sum(float(box.phase) for box in diagram.boxes
               if isinstance(box, zx.Spider))


def same_phase_class(lhs, rhs):
    """Whether two diagrams have the same phase sum modulo a quarter."""
    quarters = (phase_sum(lhs) - phase_sum(rhs)) * 4
    return bool(np.isclose(quarters, round(quarters)))


assert np.allclose(list(turns(euler(math.pi / 4, math.pi, math.pi / 4))),
                   [.25, 0, .25, .125])
```

* $(CP)$: finite corelations, i.e. boundary connectivity. Every spider of
  positive degree is the partition of its legs into one block, every scalar
  the unique one, the Hadamard the identity. Every rule but $(CP)$ connects
  its boundary the same way on both sides, while the left side of $(CP)$
  connects its two outputs and the right side does not.
* $(E)$: the support quotient of cospans, which collapses every morphism
  with a boundary to one element and keeps two scalars, empty or not.
  Every rule has the same support on both sides but $(E)$, which says that
  something non-empty is equal to nothing.

DisCoPy computes both with `frobenius.Diagram.to_hypergraph`, the cospan
of a diagram of spiders.

```python {.marimo}
point = frobenius.Ty("x")


def connectivity_arrow(box):
    """A ZX generator as the spider joining all its legs."""
    if isinstance(box, zx.Spider):
        return frobenius.Diagram.spiders(len(box.dom), len(box.cod), point)
    if box.name == 'H':
        return frobenius.Diagram.id(point)
    raise TypeError(box)


connectivity = symmetric.Functor(
    ob_map=lambda _: point, ar_map=connectivity_arrow,
    dom=Diagram, cod=frobenius.Diagram)


def boundary_partition(diagram):
    """The partition of the boundary of a diagram into connected blocks."""
    hypergraph = connectivity(diagram).to_hypergraph()
    blocks = {}
    label = lambda wires: tuple(
        blocks.setdefault(wire, len(blocks)) for wire in wires)
    return label(hypergraph.wires[0]), label(hypergraph.wires[2])


def support(diagram):
    """The support quotient: one morphism with a boundary, two scalars."""
    if len(diagram.dom) + len(diagram.cod):
        return "*"
    return int(connectivity(diagram).to_hypergraph().n_spiders > 0)


def same_partition(lhs, rhs):
    return boundary_partition(lhs) == boundary_partition(rhs)


def same_support(lhs, rhs):
    return support(lhs) == support(rhs)


assert boundary_partition(CP_rule[0]) == ((), (0, 0))
assert boundary_partition(CP_rule[1]) == ((), (0, 1))
assert (support(E_rule[0]), support(E_rule[1])) == (1, 0)
```

These are minimal in a modest sense: the three Boolean models have the
smallest possible non-trivial hom-set, and the corelation witness the
smallest boundary with two distinct partitions.

### The green identity: $(I_g)$

**The paper's** model (Section 4, after Backens, Perdrix & Wang) is
relational, i.e. tensors over the Boolean semiring. Every green spider,
whatever its phase, and the Hadamard become the one-point relation
$P_{n,m} = \{(0^n, 0^m)\}$; so do the red spiders, except at phase zero and
degree two where they are the two-point relation
$C_{n,m} = \{(0^n, 0^m), (1^n, 1^m)\}$, so that $(I_r)$ holds and $(I_g)$
fails.

**discopy#688's** is the same support pattern over $\mathbb{C}^2$: with
$p = |0\rangle\langle 0|$, every green spider of positive degree is the
all-zero tensor $|0 \dots 0\rangle$, every scalar spider is one, the
Hadamard is $p$, and the red spiders likewise except that a phase-zero red
spider of degree two is the full identity, cup or cap. One subtlety: a
`zx` cup is the green two-legged spider, which identifies the two compact
structures through $(I_g)$, the very rule under test, so the rules are
checked as written, with no cups; every tensor here is symmetric, so the
compact structure given by the full delta is consistent with it.

```python {.marimo}
def support_arrow(dim, zero, unit, dtype):
    """Every spider at the basis state zero, the identity spiders aside."""
    def arrow(box):
        degree = len(box.dom) + len(box.cod)
        if box.name == 'H' and not isinstance(box, zx.Spider):
            return np.diag([unit, zero]).astype(dtype)
        if isinstance(box, X) and degree == 2\
                and np.isclose(float(box.phase) % 1, 0):
            return np.eye(dim, dtype=dtype).reshape((dim, ) * 2)
        array = np.full((dim, ) * degree, zero, dtype=dtype)
        array[(0, ) * degree] = unit
        return array
    return arrow


relational = Interpretation(2, support_arrow(2, False, True, bool), bool)
linear_support = Interpretation(2, support_arrow(2, 0, 1, complex))
assert np.array_equal(relational.evaluate(Z(1, 1)), [[True, False],
                                                     [False, False]])
```

Two is the least dimension for a linear countermodel over a field: in
dimension one write $p$ for $Z_{1,1}$ and $g$ for the green $\frac18$
state; fusion gives $pg = g$ and $(E)$ makes $g$ invertible, so $p = 1$.

### The Hadamard decomposition: $(HD)$

discopy#688 doubles the colours: a wire is $\mathbb{C}^2 \otimes
\mathbb{C}^2$, a green spider is the standard green spider tensored with
the standard red one, a red spider the other way round, and the Hadamard
is the swap of the two factors. Every rule without a Hadamard is the
tensor of two sound rules, and $(H)$ is the naturality of the swap, but in
$(HD)$ the Hadamard is the swap while its decomposition is $H \otimes H$,
and the two already differ on $|00\rangle$.

```python {.marimo}
standard_arrow = standard.arrow
factor_swap = np.eye(4)[[0, 2, 1, 3]]


def interleave(left, right):
    """Tensor two arrays leg by leg."""
    if not left.ndim:
        return left * right
    order = sum(([i, left.ndim + i] for i in range(left.ndim)), [])
    return np.multiply.outer(left, right).transpose(order)\
        .reshape((4, ) * left.ndim)


def doubled_arrow(box):
    """The doubled-colour model separating (HD)."""
    if not isinstance(box, zx.Spider):
        return factor_swap
    other = (X if isinstance(box, Z) else Z)(
        len(box.dom), len(box.cod), box.phase)
    return interleave(np.asarray(standard_arrow(box)),
                      np.asarray(standard_arrow(other)))


doubled = Interpretation(4, doubled_arrow)
assert np.allclose(doubled.evaluate(HD_rule[1])[:, 0], [1, 0, 0, 0])
assert np.allclose(doubled.evaluate(HD_rule[0])[:, 0], [.5, .5, .5, .5])
```

### The bialgebra: an infinitesimal countermodel

The paper's model (Section 3) interprets a wire as
$V = \mathbb{D}^{\{0,1\}^3}$ for $\mathbb{D} = \mathbb{C}[\varepsilon] /
(\varepsilon^2)$ the dual numbers: three qubits whose scalars carry an
infinitesimal. The green spider gets the phases
$t_x(\alpha) = e^{i|x|\alpha}$ for $|x|$ the Hamming weight, and the
Hadamard is perturbed by conjugation

$$H' := O^{-1} H_0 O
     = H_0 + \varepsilon(H_0 N - N H_0), \qquad O := 1 + \varepsilon N$$

where $H_0 = h^{\otimes 3}$ is the Sylvester–Walsh matrix and $N$ skew
rotates the Hamming weight one subspace:
$N|100\rangle = |010\rangle - |001\rangle$ and cyclically.

Rather than computing with dual numbers entry by entry, discopy#626 reads
the model as the *first-order jet* of the family of interpretations
$t \mapsto (1 + tN)^{-1} H_0 (1 + tN)$ at $t = 0$: a diagram evaluates to
$A + \varepsilon B$ where $A$ is its unperturbed evaluation and $B$ its
derivative along the family. Composition and tensor are bilinear, so by the
product rule $B$ is a sum over the occurrences of $H$ in the diagram, each
replaced by a box $K$ evaluating to the commutator $[H_0, N]$ —
forward-mode differentiation, done by diagram surgery. First every red
spider is expanded into Hadamards around a green spider, which is how the
model defines them anyway:

```python {.marimo}
class Expand(Rewrite):
    """Expand every X spider into Hadamards around a Z spider."""
    def rewrite(self, box):
        if isinstance(box, X):
            n, m = len(box.dom), len(box.cod)
            return Id().tensor(*(n * [H])) >> Z(n, m, box.phase)\
                >> Id().tensor(*(m * [H]))
        return None


expand = Expand()

K = zx.Box('K', zx.Nat(1), zx.Nat(1))


def replace(diagram, index, box):
    """Replace the box at ``index`` by ``box``."""
    boxes = list(diagram.boxes)
    boxes[index] = box
    return Diagram.decode(
        dom=diagram.dom, boxes_and_offsets=zip(boxes, diagram.offsets))


def is_hadamard(box):
    """Whether a box is the Hadamard."""
    return not isinstance(box, zx.Spider) and box.name == 'H'


class Jet:
    """Evaluate diagrams to pairs (value, epsilon) by the product rule."""
    def __init__(self, model, derivative):
        self.value = model
        self.tangent = Interpretation(model.dim, lambda box: derivative
                                      if box.name == 'K' else model.arrow(box))

    def __call__(self, diagram):
        diagram = expand(diagram)
        eps = sum((self.tangent.evaluate(replace(diagram, index, K))
                   for index, box in enumerate(diagram.boxes)
                   if is_hadamard(box)), start=0j)
        return self.value.evaluate(diagram), eps

    def close(self, lhs, rhs):
        (lv, le), (rv, re) = self(lhs), self(rhs)
        return np.allclose(lv, rv, atol=1e-9)\
            and np.allclose(le, re, atol=1e-9)


def qubit_hadamard(k):
    """The Sylvester-Walsh matrix on k qubits."""
    result = np.ones((1, 1))
    for _ in range(k):
        result = np.kron(result, hadamard_matrix)
    return result


def hamming_weights(k):
    """The Hamming weight of each basis state of k qubits."""
    return np.array([bin(i).count('1') for i in range(2 ** k)])


def qubit_power(k, hadamard=None):
    """The phase model of k qubits, with any Hadamard."""
    weights = hamming_weights(k)
    return phase_model(
        2 ** k, lambda a: np.exp(2j * math.pi * a * weights),
        qubit_hadamard(k) if hadamard is None else hadamard)


def jet_model(N, k):
    """The jet of the k-fold qubit model perturbed by ``N``."""
    return Jet(qubit_power(k),
               qubit_hadamard(k) @ N - N @ qubit_hadamard(k))
```

```python {.marimo}
paper_N = np.zeros((8, 8), dtype=complex)
for _source, _plus, _minus in [(4, 2, 1), (2, 1, 4), (1, 4, 2)]:
    paper_N[_plus, _source], paper_N[_minus, _source] = 1, -1

paper_jet = jet_model(paper_N, 3)
```

The witness is the one the paper computes: evaluating both sides of $(B)$
on $|010\rangle \otimes |010\rangle$ and reading the coefficient of
$|000\rangle \otimes |001\rangle$ gives $-\frac{\sqrt{2}}{4}\varepsilon$
on the left and $0$ on the right.

```python {.marimo}
witness = (2, 2, 0, 1)
lhs_value, lhs_eps = (a.reshape(4 * (8, )) for a in paper_jet(B_rule[0]))
rhs_value, rhs_eps = (a.reshape(4 * (8, )) for a in paper_jet(B_rule[1]))

assert np.isclose(lhs_eps[witness], -np.sqrt(2) / 4)
assert np.isclose(rhs_eps[witness], 0)
mo.md(f"""The coefficient is
    ${np.round(lhs_value[witness], 12)}
    {np.round(lhs_eps[witness].real, 8)}\\,\\varepsilon
    = -\\frac{{\\sqrt 2}}{{4}}\\varepsilon$
    on the left-hand side and
    ${np.round(rhs_value[witness], 12)}$
    on the right-hand side. ✓""")
```

### The bialgebra: a finite countermodel

discopy#688 finds a countermodel with no nilpotent scalars, over the
complex numbers. On $V = \mathbb{C}^{2^N}$ with the same weight phases,
interpret the Hadamard as $K = Q H_N Q^T$ for $H_N = h^{\otimes N}$ and $Q$
a real orthogonal matrix that commutes with every phase $P_a =
\mathrm{diag}(e^{i|x|a})$ and fixes every phase state $p_a = \sum_x
e^{i|x|a}|x\rangle$. Such a $Q$ is block diagonal by Hamming weight and
fixes the uniform vector of each block, so these deformations form the
group

$$G_N = \prod_{w=0}^N O\left(\tbinom{N}{w} - 1\right).$$

Every rule but two is $Q$-conjugate to its $N$-fold qubit instance, so
holds. $(B)$ need not. Nor need $(EU)$, but only through its scalar
$\theta$, whose value is $C_3(K) = \sum_{x,y} K_{xy}^3$: $(EU)$ holds for
every angle exactly when $C_3(K) = 2^{-N/2}$, one cubic equation, where
$(B)$ is the much stronger family $2^{N/2}X_{abc}X_{abd} = \delta_{cd}
X_{abc}$ for $X_{abc} = \sum_j K_{aj}K_{bj}K_{cj}$.

For $N = 4$, rotate the plane of

$$f_1 = (5,-1,-1,-1,-1,-1)/\sqrt{30},\qquad
  f_2 = (0,4,-1,-1,-1,-1)/(2\sqrt5)$$

inside the weight-two block, ordered $(0011, 0101, 0110, 1001, 1010,
1100)$, by an angle of cosine $c$ and positive sine $s$. Expanding modulo
$c^2 + s^2 = 1$ gives $C_3(K) = \frac14 + \frac{(1 - c)P(c)}{768}$ for

$$P(c) = 25c^5 + 25c^4 + 25c^3 - 335c^2 - 398c - 110,$$

which has a root in $(-\frac12, -\frac25)$ since $P(-\frac12) =
\frac{93}{32} > 0 > -\frac{702}{125} = P(-\frac25)$. At that root every
rule but $(B)$ holds, and the component of $(B)$ with inputs
$(1010, 1010)$ and outputs $(0000, 1111)$ is
$\frac{7 - 6c - c^2 + 4\sqrt6 s}{96} > \frac{61}{640}$ on the left and zero
on the right.

```python {.marimo}
def plane_rotation(size, f1, f2, cosine, sine, identity):
    """The rotation of the plane of two orthonormal vectors."""
    return identity(size) + (cosine - 1) * (f1 * f1.T + f2 * f2.T)\
        + sine * (f2 * f1.T - f1 * f2.T)


def block(matrix, indices, size, identity):
    """The identity with ``matrix`` on the rows and columns ``indices``."""
    result = identity(size)
    for row, x in enumerate(indices):
        for column, y in enumerate(indices):
            result[x, y] = matrix[row, column]
    return result


weight_two = [int(x) for x in np.flatnonzero(hamming_weights(4) == 2)]
f1_vector = np.array([[5, -1, -1, -1, -1, -1]]).T / math.sqrt(30)
f2_vector = np.array([[0, 4, -1, -1, -1, -1]]).T / (2 * math.sqrt(5))
quintic = np.array([25, 25, 25, -335, -398, -110])
finite_cosine = next(
    root.real for root in np.roots(quintic)
    if abs(root.imag) < 1e-10 and -.5 < root.real < -.4)
finite_sine = math.sqrt(1 - finite_cosine ** 2)

finite_Q = block(plane_rotation(
    6, f1_vector, f2_vector, finite_cosine, finite_sine, np.eye),
    weight_two, 16, np.eye)
finite_K = finite_Q @ qubit_hadamard(4) @ finite_Q.T
finite = qubit_power(4, finite_K)

assert np.allclose(finite_Q.T @ finite_Q, np.eye(16))
assert np.allclose(finite_K, finite_K.T)
assert np.allclose(finite_K @ finite_K, np.eye(16))
assert np.allclose(np.sum(finite_K ** 3), .25)
assert np.isclose(finite.evaluate(theta), .25)

_left = finite.evaluate(B_rule[0]).reshape(4 * (16, ))
_right = finite.evaluate(B_rule[1]).reshape(4 * (16, ))
assert np.isclose(_left[10, 10, 0, 15], (
    7 - 6 * finite_cosine - finite_cosine ** 2
    + 4 * math.sqrt(6) * finite_sine) / 96)
assert _left[10, 10, 0, 15] > 61 / 640 and _right[10, 10, 0, 15] == 0
mo.md(f"$c = {finite_cosine:.6f}$, $C_3(K) = \\theta = 1/4$, and the "
      f"witness is ${_left[10, 10, 0, 15]:.6f}$ against $0$. ✓")
```

The floating-point root above is only a witness. The exact claims —
the reduction of $C_3$, the root, the sign of $s$ and the lower bound on the
witness — are checked in SymPy, on the same matrices built symbolically.

```python {.marimo}
import sympy as sp

c, s = sp.symbols("c s", real=True)


def on_circle_remainder(expression):
    """Reduce a polynomial in c and s modulo c ** 2 + s ** 2 = 1."""
    return sp.factor(sp.rem(sp.Poly(sp.together(expression), s),
                            sp.Poly(s ** 2 + c ** 2 - 1, s)).as_expr())


def exact_walsh(k):
    """The Sylvester-Walsh matrix on k qubits, exactly."""
    return sp.Matrix(2 ** k, 2 ** k, lambda x, y: (-1) ** bin(x & y).count(
        '1') / sp.sqrt(2) ** k)


def exact_deformation(k, indices, f1, f2):
    """The Hadamard of k qubits deformed by a plane rotation in a block."""
    Q = block(plane_rotation(len(indices), f1, f2, c, s, sp.eye),
              indices, 2 ** k, sp.eye)
    return (Q * exact_walsh(k) * Q.T).applyfunc(sp.expand)


def cubic_scalar(matrix):
    """The value of theta when the Hadamard is ``matrix``."""
    return on_circle_remainder(sum(entry ** 3 for entry in matrix))


def red_comultiplication(matrix, a, b, d):
    """An entry of the red spider with two inputs and one output."""
    return on_circle_remainder(sum(
        matrix[a, j] * matrix[b, j] * matrix[d, j]
        for j in range(matrix.rows)))


P = sp.Poly(25 * c ** 5 + 25 * c ** 4 + 25 * c ** 3 - 335 * c ** 2
            - 398 * c - 110, c)
exact_K = exact_deformation(
    4, weight_two, sp.Matrix([5, -1, -1, -1, -1, -1]) / sp.sqrt(30),
    sp.Matrix([0, 4, -1, -1, -1, -1]) / (2 * sp.sqrt(5)))
exact_witness = (7 - 6 * c - c ** 2 + 4 * sp.sqrt(6) * s) / 96

assert sp.expand(cubic_scalar(exact_K) - sp.Rational(1, 4)
                 - (1 - c) * P.as_expr() / 768) == 0
assert red_comultiplication(exact_K, 10, 10, 0) == sp.Rational(1, 4)
assert sp.expand(red_comultiplication(exact_K, 10, 10, 15)
                 - exact_witness) == 0
assert P.eval(-sp.Rational(1, 2)) == sp.Rational(93, 32)
assert P.eval(-sp.Rational(2, 5)) == -sp.Rational(702, 125)
assert P.count_roots(-sp.Rational(1, 2), -sp.Rational(2, 5)) == 1
assert sp.gcd(P, P.diff(c)) == 1
exact_cosine, = [root for root in P.real_roots()
                 if -sp.Rational(1, 2) < root < -sp.Rational(2, 5)]
exact_sine = sp.sqrt(1 - exact_cosine ** 2)
assert exact_sine.is_positive
assert (exact_witness.subs({c: exact_cosine, s: exact_sine})
        - sp.Rational(61, 640)).is_positive
mo.md(r"$C_3(K) - \frac14 = \frac{(1 - c)P(c)}{768}$, the witness is "
      r"$\frac{7 - 6c - c^2 + 4\sqrt6 s}{96}$, and $P$ has one simple root "
      r"in $(-\frac12, -\frac25)$, where the witness exceeds "
      r"$\frac{61}{640}$, exactly. ✓")
```

### All the countermodels at once

Here is every model against every rule of both rulesets, $(I_r)$ included.
Each model of the table breaks the rule it was built for and holds every
other rule of $\mathrm{ZX_{opt}}$, which makes $\mathrm{ZX_{opt}}$ minimal.
The table also reads off $\mathrm{ZX'_{opt}}$, which discopy#688 did not
analyse: the support quotient separates $(IV)$, which says that a
non-empty scalar is equal to nothing just as $(E)$ does, and the
doubled-colour model separates $(EU')$, whose Hadamard it swaps; with the
other five rules separated by the same models as before, the seven rules
of $\mathrm{ZX'_{opt}}$ each have a separating model too. Only the phase
grading is no use there, as it breaks $(IV)$ and $(EU')$ at once.

```python {.marimo}
models = {
    "parity of degree ≥ 4": high_degree_parity(zx.Spider),
    "parity of red degree ≥ 4": high_degree_parity(X),
    "phase sum mod 1/4": same_phase_class,
    "corelations": same_partition,
    "support": same_support,
    "relations (paper)": relational.close,
    "linear support": linear_support.close,
    "doubled colours": doubled.close,
    "jet over ε (paper)": paper_jet.close,
    "finite deformation": finite.close,
    "projective": projective.close}
table = {model: verdicts(close) for model, close in models.items()}
broken = {model: [name for name, ok in row.items() if not ok]
          for model, row in table.items()}

assert broken == {
    "parity of degree ≥ 4": ["S"], "parity of red degree ≥ 4": ["H"],
    "phase sum mod 1/4": ["EU", "IV", "EU'"], "corelations": ["CP"],
    "support": ["E", "IV"], "relations (paper)": ["Ig"],
    "linear support": ["Ig"], "doubled colours": ["HD", "EU'"],
    "jet over ε (paper)": ["B"], "finite deformation": ["B"],
    "projective": ["Ir", "E", "EU", "EU'"]}
for _ruleset in (ZX_opt, ZX_opt_prime):
    for _rule in _ruleset:
        assert any(set(_broken) & set(_ruleset) == {_rule}
                   for _broken in broken.values()), _rule

mark = lambda ok: "✓" if ok else "**✗**"
mo.md("\n".join(
    [f"| model | {' | '.join(rules)} |", "|---" * (len(rules) + 1) + "|"]
    + [f"| {model} | {' | '.join(map(mark, row.values()))} |"
       for model, row in table.items()]))
```

## Which perturbations break the bialgebra?

The two bialgebra countermodels are one idea, seen at a point and along a
tangent. discopy#626 asks which tangents work; discopy#688 asks which finite
points do.

**Tangents.** The soundness checks only ever use three properties of the
perturbation $N$ of the jet model:

* $N$ is **antisymmetric**, so that $O^T = O^{-1}$ and $H'$ stays symmetric
  and self-inverse — Hadamards can still bend through the compact
  structure;
* $N$ **preserves each Hamming weight subspace**, so that $O$ commutes with
  every diagonal phase — the Euler and Hadamard decompositions survive
  conjugation;
* $N$ **annihilates the constant vector on each weight subspace** — every
  spider state $\sum_x e^{i|x|\alpha}|x\rangle$ is fixed by $O$, so the
  scalar gadgets and the copy rule keep their values.

These are exactly the conditions defining $G_N$, linearised: the
*admissible* perturbations are its Lie algebra
$\bigoplus_w \mathfrak{so}\left(\binom{N}{w} - 1\right)$, spanned by
*three-cycles*, skew rotations of three basis states of equal weight like
the paper's $N$.

```python {.marimo}
def three_cycle(dim, cycle):
    """The antisymmetric matrix skew-rotating a three-cycle of basis
    states, zero elsewhere."""
    N = np.zeros((dim, dim))
    for a, b in zip(cycle, cycle[1:] + cycle[:1]):
        N[b, a], N[a, b] = 1, -1
    return N


def admissible_basis(k):
    """Three-cycles spanning the antisymmetric weight-preserving matrices
    that annihilate the constant vector on each weight subspace."""
    weights, basis = hamming_weights(k), []
    for w in range(k + 1):
        orbit = list(np.flatnonzero(weights == w))
        basis += [three_cycle(2 ** k, (orbit[0], orbit[i], orbit[j]))
                  for i, j in combinations(range(1, len(orbit)), 2)]
    return basis


lie_dimension = lambda k: sum(
    math.comb(math.comb(k, w) - 1, 2) for w in range(k + 1))
assert [len(admissible_basis(k)) for k in (1, 2, 3, 4)]\
    == [lie_dimension(k) for k in (1, 2, 3, 4)] == [0, 0, 2, 16]
assert np.allclose(sum(
    c * b for c, b in zip(np.linalg.lstsq(
        np.stack([b.ravel() for b in admissible_basis(3)], 1),
        paper_N.ravel().real)[0], admissible_basis(3))), paper_N.real)
mo.md("""The admissible space has the dimension of the Lie algebra of
    $G_N$: none on one or two qubits, two on three qubits — with the
    paper's $N$ the first basis vector — and sixteen on four. ✓""")
```

For any rule, the $\varepsilon$-part of $[\![\mathrm{lhs}]\!] -
[\![\mathrm{rhs}]\!]$ in the jet model is **linear in $N$**, so checking a
rule on every basis three-cycle proves it for every admissible $N$ at once.
Concretely:

1. **Every admissible $N$ gives a model of everything but $(B)$, and $(B)$
   fails exactly when the perturbation does something.** $(B)$ holds if
   and only if $N$ commutes with $H_0$, in which case $H' = H_0$ on the nose
   and the model is the unperturbed one. A rule schema is a family indexed
   by phases and arities, so the cells check this on the instances of
   `rules` on three qubits and on smaller ones on four, to keep the
   notebook fast: the three conditions above are why it holds beyond them,
   and the checks sample that argument rather than prove it.
2. **The kernel of the bialgebra defect is exactly the space of trivial
   perturbations**: stacking the defects of the basis cycles gives the
   defect as a matrix on the admissible space, whose rank is that of
   $N \mapsto [H_0, N]$ and which vanishes on that map's kernel.

```python {.marimo}
def bialgebra_defect(N, k):
    """The epsilon-part of the difference between the two sides of (B)."""
    jet = jet_model(N, k)
    (_, lhs_e), (_, rhs_e) = jet(B_rule[0]), jet(B_rule[1])
    return lhs_e - rhs_e


small_rules = {
    "S": [fusion(*fusion_cases[i]) for i in (0, 2, 4)],
    "Ig": rules["Ig"], "Ir": rules["Ir"], "CP": rules["CP"],
    "B": rules["B"], "HD": rules["HD"],
    "H": instances(colour_change, [
        (1, 1, .1), (1, 2, -.4), (1, 1, 0.), (2, 0, 0.), (0, 2, 0.)]),
    "E": rules["E"], "EU": [eu(*eu_cases[i]) for i in (0, 2)],
    "IV": [iv(.3)], "EU'": [eu_prime(*eu_prime_cases[i]) for i in (0, 4)]}

generalisation_report = []
for _k in (3, 4):
    _basis = admissible_basis(_k)
    _suite = small_rules if _k == 4 else rules
    for _N in _basis:
        _jet = jet_model(_N, _k)
        _trivial_N = np.allclose(
            qubit_hadamard(_k) @ _N - _N @ qubit_hadamard(_k), 0)
        for _name, _cases in _suite.items():
            assert all(_jet.close(*_rule) for _rule in _cases)\
                == (_name != "B" or _trivial_N), (_k, _name)
    _defects = np.stack([bialgebra_defect(b, _k).ravel() for b in _basis])
    _commutators = np.stack([
        (qubit_hadamard(_k) @ b - b @ qubit_hadamard(_k)).ravel()
        for b in _basis])
    _rank = np.linalg.matrix_rank(_defects, tol=1e-9)
    _trivial = len(_basis) - np.linalg.matrix_rank(_commutators, tol=1e-9)
    assert _rank == len(_basis) - _trivial
    _U = np.linalg.svd(_commutators)[0]
    assert np.allclose(_U[:, len(_basis) - _trivial:].T @ _defects, 0)
    generalisation_report.append(
        f"$k = {_k}$: admissible space of dimension {len(_basis)}, "
        f"of which {_trivial} trivial, and the defect has rank {_rank} "
        f"$= {len(_basis)} - {_trivial}$. ✓")
mo.md("  \n".join(generalisation_report))
```

So the admissible perturbations modulo the trivial ones form a space of
dimension $0, 0, 1, 12$ on $1, 2, 3, 4$ qubits. The rank count is exact,
$(B)$ having no parameter; that every other rule holds rests on the three
conditions above and is checked on sampled instances only. Granting it,
these are exactly the countermodels of this shape: **three qubits is the
least possible size**, and **at that size the countermodel is unique** up
to scale and trivial perturbations — the second basis cycle, the mirror of the paper's $N$ on the
weight-two subspace, differs from $-N$ by a perturbation commuting with
$H_0$, so it defines the same interpretation. On four qubits the sample is
smaller still, so the twelve dimensions there are candidate countermodels
rather than a classification.

```python {.marimo}
mirror_N, minus_paper_N = admissible_basis(3)[1], -paper_N.real
assert np.allclose(
    qubit_hadamard(3) @ (mirror_N - minus_paper_N)
    - (mirror_N - minus_paper_N) @ qubit_hadamard(3), 0)
mo.md("The weight-two mirror cycle equals $-N$ up to a trivial "
      "perturbation. ✓")
```

**Finite points.** discopy#688 classifies the connected part of $G_3$,
$SO(2) \times SO(2)$, exactly. The Walsh transform exchanges the weight-one
and weight-two planes, so $K$ depends only on the sum $\vartheta$ of the two
block angles — the finite form of the one effective tangent direction above
— and

$$C_3(K) - 2^{-3/2} = \frac{\sqrt2}{3}(\cos\vartheta - 1)
(2\cos\vartheta + 1)^2 = -\frac{2\sqrt2}{3}\sin^2\frac{3\vartheta}{2}.$$

So $(EU)$ allows only $\vartheta \in \frac{2\pi}{3}\mathbb{Z}$, where $K$ is
a Walsh matrix with permuted characters, $\pm\frac{\sqrt2}{4}$ entrywise,
and satisfies $(B)$: there is no finite countermodel in this family below
$N = 4$, dimension sixteen — which is no lower bound on unrelated models of
dimension three to fifteen.

```python {.marimo}
weight_one = [int(x) for x in np.flatnonzero(hamming_weights(3) == 1)]
exact_K3 = exact_deformation(
    3, weight_one, sp.Matrix([1, -1, 0]) / sp.sqrt(2),
    sp.Matrix([1, 1, -2]) / sp.sqrt(6))
assert sp.simplify(cubic_scalar(exact_K3) - sp.sqrt(2) / 4
                   - sp.sqrt(2) * (c - 1) * (2 * c + 1) ** 2 / 3) == 0


def walsh_signs(matrix):
    """The signs of a matrix whose entries are all plus or minus root two
    over four."""
    signs = matrix * 4 / sp.sqrt(2)
    assert set(signs.applyfunc(sp.simplify)) == {-1, 1}
    return np.array(signs.applyfunc(sp.simplify).tolist(), dtype=int)


for _sign in (-1, 1):
    _S = walsh_signs(exact_K3.subs({c: -sp.Rational(1, 2),
                                    s: _sign * sp.sqrt(3) / 2}))
    _T = np.einsum("aj,bj,dj->abd", _S, _S, _S)
    _delta = np.eye(8, dtype=int)
    assert np.array_equal(np.einsum("abc,abd->abcd", _T, _T),
                          8 * np.einsum("abc,cd->abcd", _T, _delta))
mo.md(r"On three qubits $C_3(K) - 2^{-3/2} = \frac{\sqrt2}{3}(c - 1)"
      r"(2c + 1)^2$, and both non-trivial roots satisfy $(B)$, exactly. ✓")
```

The last check is $(B)$ in integers: with $K = \frac{\sqrt2}{4}S$ for a
sign matrix $S$ and $T_{abd} = \sum_j S_{aj}S_{bj}S_{dj}$, the bialgebra
$2^{3/2}X_{abc}X_{abd} = \delta_{cd}X_{abc}$ reads $T_{abc}T_{abd} =
8\delta_{cd}T_{abc}$.

This is why the paper's model is simpler than discopy#688's. Along a
one-parameter subgroup of $G_3$ the Euler obstruction is
$O(\vartheta^2)$ while the bialgebra defect has a linear term: over the
dual numbers $\vartheta = \varepsilon$ kills the first and not the second,
which is the jet model; over the complex numbers the quadratic obstruction
has to vanish at an actual point, which on three qubits it only does at
Walsh points, and the first finite countermodel in the family lives on four
qubits. The sizes are not directly comparable either: $\mathbb{D}^8$ has
complex dimension sixteen, but its tensor product and unit are those of
$\mathbb{D}$-modules.

## Conclusion

Everything in arXiv:2608.14872 that can be checked by computation has
been, twice where the two upstream accounts differ: the two rulesets are
sound; $(I_r)$ is derivable, by the paper's chain of lemmas and by
discopy#688's algebraic argument, which pass through the same scalar
identities; and every rule of $\mathrm{ZX_{opt}}$ and of
$\mathrm{ZX'_{opt}}$ has a model of the others breaking it, the bialgebra
twice, infinitesimally over the dual numbers and finitely over
$\mathbb{C}^{16}$. The countermodels for the bialgebra are a tangent and a
point of one group $G_N$ of deformations: its admissible tangents exist
from three qubits up, and on three qubits the paper's is the only one that
does anything, while its finite points need four.

The jet construction — evaluate a diagram in a one-parameter family of
models and differentiate — is not specific to the ZX calculus: it is
forward-mode automatic differentiation of a functor, done by diagram
surgery, and `Jet` works for any `Interpretation`.
