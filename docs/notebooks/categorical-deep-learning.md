---
title: Categorical Deep Learning
marimo-version: 0.23.14
---

```python {.marimo}
import marimo as mo
```

# Categorical deep learning, implemented

[Gavranović et al. (2024)](https://arxiv.org/abs/2402.15332) argue that one categorical framework, algebras of monads valued in a 2-category of parametric maps, covers both how deep learning architectures are *constrained* and how they are *implemented*.
This notebook reproduces their examples with what DisCoPy has today, to see which of their diagrams already run and which do not.

The paper draws two kinds of diagram, and they turn into code differently:

* a **commuting square** is a constraint, an equation a layer must satisfy: it is solved, enforced by construction or checked;
* a **parametric string diagram** is an architecture, a diagram with parameters as wires: it is compiled by a functor.

## Constraints are solved: equivariance

A group $G$ acting on a set $X$ is an algebra of the monad $G \times -$, and a map between two such algebras that makes the square commute is an equivariant map: $f(g \cdot x) = g \cdot f(x)$.
For a linear map between permutation representations, the square says that the matrix is constant on the orbits of $G$ acting on pairs of points, i.e. on the orbits of the tensor of the two actions, which [`discopy.group`](https://docs.discopy.org/en/main/_api/discopy.group.html) computes from the generators alone.
Here is the weight tying of Examples H.1 and H.3, the cyclic group of order two swapping two pixels, with the swap or the trivial action on the output:

```python {.marimo}
import numpy as np

from discopy.group import Action, Cyclic, Symmetric


def tying(cod, dom):
    """ The index of the weight each entry of a matrix from dom to cod reads. """
    index = np.zeros(cod.size * dom.size, dtype=int)
    for _k, _orbit in enumerate(cod.tensor(dom).orbits()):
        index[list(_orbit)] = _k
    return index.reshape(cod.size, dom.size)


def latex(index):
    rows = (" & ".join(f"w_{k + 1}" for k in row) for row in index)
    return r"\begin{pmatrix}" + r" \\ ".join(rows) + r"\end{pmatrix}"


pixels = Cyclic[2].action()
constant = Action(Cyclic[2], 2, ((0, 1), ))
mo.md(rf"""
equivariant: $W = {latex(tying(pixels, pixels))}$,
invariant: $W = {latex(tying(constant, pixels))}$
""")
```

The equivariant matrix is symmetric, and each output of the invariant one is its own multiple of $x_1 + x_2$.
The same computation gives the graph networks of Appendix C.1: the symmetric group $\Sigma_n$ acting on the $n$ nodes leaves two weights for a map from node features to node features, the identity and the all-ones matrix, i.e. $a x + b \sum_i x_i$.
A map from an adjacency matrix to node features has five: in general the equivariant maps from the $k$-th to the $l$-th tensor power are counted by the Bell number $B_{k + l}$, once $n \geq k + l$.
Each matrix commutes with the action of every generator, hence of the whole group:

```python {.marimo}
nodes = Symmetric[5].action()


def permutation_matrix(permutation):
    matrix = np.zeros((len(permutation), len(permutation)))
    matrix[list(permutation), range(len(permutation))] = 1
    return matrix


node_index = tying(nodes, nodes)
weights = np.random.default_rng(0).normal(size=node_index.max() + 1)
W = weights[node_index]
assert all(
    np.allclose(W @ permutation_matrix(g), permutation_matrix(g) @ W)
    for g in nodes.images)
edges = nodes.tensor(nodes)
assert (node_index.max() + 1, tying(nodes, edges).max() + 1) == (2, 5)
mo.md(f"""
node to node: {node_index.max() + 1} weights,
edge to node: {tying(nodes, edges).max() + 1} weights
""")
```

## Architectures are parametric maps

In the 2-category $\mathbf{Para}$, a map $A \to B$ is a pair of a parameter object $P$ and a map $P \times A \to B$, drawn with $P$ as one more wire.
[`discopy.para`](https://docs.discopy.org/en/main/_api/discopy.para.html) implements it over any symmetric category, here the free Markov category of [`discopy.markov`](https://docs.discopy.org/en/main/_api/discopy.markov.html).
Composition puts the parameters side by side (Figure 4):

```python {.marimo}
from discopy import markov, para
from discopy.markov import Box, Diagram, Ty
from discopy.monoidal import Equation

Para = para.Markov[Diagram]
A, B, C, P, Q = map(Ty, "ABCPQ")
f = Para(A, B, Box("f", A @ P, B), param=P)
g = Para(B, C, Box("g", B @ Q, C), param=Q)
assert (f >> g).param == P @ Q
(f >> g).inside
```

A 2-cell is a reparametrisation, precomposing the parameters with a map (Figure 3), and Theorem G.10 says which reparametrisations tie weights: the 2-cells of a lax algebra are exactly a comonoid on the parameter, i.e. copying it.
In a Markov category every object has one, so weight tying is a reparametrisation by `copy`:

```python {.marimo}
cell = Para(A, A, Box("cell", A @ P, A), param=P)
untied, tied = cell >> cell, (cell >> cell).reparam(Diagram.copy(P))
assert untied.param == P @ P and tied.param == P
Equation(untied.inside, tied.inside, symbol="$\\Rightarrow$")
```

## Folds are functors

A list of elements of $A$ is built from the two constructors of the endofunctor $1 + A \times -$: `Nil` and `Cons`.
A map out of a sum is a pair of maps, so in a monoidal category an algebra of this endofunctor is one box per constructor, and a list of $n$ elements is the diagram of its constructors:

```python {.marimo}
S = Ty("S")
nil, cons = Box("Nil", Ty(), S), Box("Cons", A @ S, S)


def as_list(n):
    """ The list of n elements of A as a diagram of its constructors. """
    diagram = A ** n @ nil
    for _i in range(n, 0, -1):
        diagram = diagram >> A ** (_i - 1) @ cons
    return diagram


as_list(3)
```

The list is the initial algebra, the free one, so an algebra is a functor out of it: a folding recurrent cell sends `Nil` to an initial state and `Cons` to a cell, each with its own parameter.
Applying the functor is the fold, and the unique homomorphism of the paper is functoriality.
The image of a list of three elements has four copies of the parameter, tied by one `copy` (Figure 10):

```python {.marimo}
initial = Para(Ty(), S, Box("initial", P, S), param=P)
recurrent_cell = Para(A @ S, S, Box("cell", A @ S @ P, S), param=P)
folding = markov.Functor(
    ob_map={A: A, S: S}, ar_map={nil: initial, cons: recurrent_cell},
    cod=Para)
assert folding(as_list(3)).param == P ** 4
folding(as_list(3)).reparam(Diagram.copy(P, 4)).inside
```

The same functor into parametric Python functions computes: equations (1) and (2) of the paper, $f(\mathtt{Nil}) = r_0$ and $f(\mathtt{Cons}(h, t)) = r_1(h, f(t))$, hold of every list by functoriality, and the network agrees with the recursion they define.

```python {.marimo}
from discopy.python import Function, Ty as PythonTy

Real = PythonTy(float)
numeric = markov.Functor(
    ob_map={A: Real, S: Real},
    ar_map={
        nil: para.Markov[Function](
            PythonTy(), Real, Function(lambda p: p, Real, Real), param=Real),
        cons: para.Markov[Function](
            Real @ Real, Real,
            Function(lambda a, s, p: a + p * s, Real ** 3, Real),
            param=Real)},
    cod=para.Markov[Function])


def recursion(xs, p):
    """ The fold of equations (1) and (2), by recursion. """
    return p if not xs else xs[0] + p * recursion(xs[1:], p)


fold = numeric(as_list(3)).reparam(Function.copy(Real, 4)).inside
assert fold(1., 2., 3., .5) == recursion([1., 2., 3.], .5)
mo.md(f"fold: `{fold(1., 2., 3., .5)}`")
```

Binary trees are the initial algebra of $A + (-)^2$, built from `Leaf` and `Node`, and a recursive neural network is a functor out of them (Figure 12): the architecture depends on the shape of the data, the parameter does not.
Drawings with many parameters read better after `simplify`, the same diagram up to the axioms of symmetric categories with fewer crossings.

```python {.marimo}
leaf, node = Box("Leaf", A, S), Box("Node", S @ S, S)
tree = (leaf @ leaf >> node) @ leaf >> node
recursive = markov.Functor(
    ob_map={A: A, S: S},
    ar_map={leaf: Para(A, S, Box("leaf", A @ P, S), param=P),
            node: Para(S @ S, S, Box("node", S @ S @ P, S), param=P)},
    cod=Para)
Equation(tree, recursive(tree).reparam(Diagram.copy(P, 5)).inside.simplify(),
         symbol="$\\mapsto$")
```

## Unfolds are feedback

Coalgebras go the other way, from a state to what it produces.
A Mealy machine, a coalgebra of $I \to O \times -$, is a cell from an input and the previous state to an output and the next state, and the recurrent network it defines is its feedback in the category of [`discopy.feedback`](https://docs.discopy.org/en/main/_api/discopy.feedback.html), where the state comes back delayed:

```python {.marimo}
I, O = Ty("I"), Ty("O")
mealy = Box("mealy", I @ S.delay(), O @ S)
recurrent = mealy.feedback(dom=I, cod=O, mem=S)
recurrent
```

Its semantics are [monoidal streams](https://docs.discopy.org/en/main/_api/discopy.stream.html), the final coalgebra: a functor sends it to a stream of boxes, one per time step, and unrolling the stream gives the first steps as a finite diagram (Figure 13):

```python {.marimo}
from discopy import stream, symmetric

sequences = {x: stream.Ty.sequence(str(x)) for x in (I, O, S)}
to_stream = markov.Functor(
    ob_map=sequences,
    ar_map={mealy: stream.Stream.sequence(
        "mealy", sequences[I] @ sequences[S].delay(),
        sequences[O] @ sequences[S])},
    cod=stream.Stream)
unrolled = to_stream(recurrent).unroll(2).now
unrolled
```

The time steps are boxes `mealy0`, `mealy1`, `mealy2`, so tying their weights is the same move as for folds: a functor sends every step to one parametric cell, the first one starting from an initial state, and one `copy` ties them all.

```python {.marimo}
step = Para(I @ S, O @ S, Box("step", I @ S @ P, O @ S), param=P)
start = Para(Ty(), S, Box("start", P, S), param=P)
in_time = symmetric.Functor(
    ob_map=lambda x: Ty(str(x)[:-1]),
    ar_map=lambda box: I @ start >> step if box.name == "mealy0" else step,
    cod=Para)
assert in_time(unrolled).param == P ** 4
in_time(unrolled).reparam(Diagram.copy(P, 4)).inside.simplify()
```

That ties the weights of a finite prefix, chosen after unrolling.
To tie them across every time step, the parameter becomes a constant stream: read once at time zero, then followed by its own copy, i.e. [`FollowedBy`](https://docs.discopy.org/en/main/_api/discopy.feedback.FollowedBy.html) from `P.head @ P.delay()` to `P` and a `copy` fed back.
The cell reads it at every step like any other input:

```python {.marimo}
from discopy.markov import FollowedBy

always = (FollowedBy(P) >> Diagram.copy(P)).feedback(
    dom=P.head, cod=P, mem=P)
weighted = Box("mealy", I @ P @ S.delay(), O @ S)
tied_in_time = I @ always >> weighted.feedback(dom=I @ P, cod=O, mem=S)
tied_in_time
```

Over streams, `P` is a constant stream of types, and unrolling gives a network with one parameter, copied to every step:

```python {.marimo}
P_always = stream.Ty(symmetric.Ty("P"))
to_streams = markov.Functor(
    ob_map=sequences | {P: P_always},
    ar_map={weighted: stream.Stream[Diagram].sequence(
        "mealy", sequences[I] @ P_always @ sequences[S].delay(),
        sequences[O] @ sequences[S])},
    cod=stream.Stream[Diagram])
everywhere = to_streams(tied_in_time).unroll(2).now
assert [str(x) for x in everywhere.dom] == ["I0", "P", "I1", "I2"]
everywhere.foliation()
```

## Figure 1, five cells

Each architecture of Figure 1 is a (co)algebra of a polynomial endofunctor, and each is a handful of boxes: a map out of a sum is one box per summand, and a map into an exponential $I \to -$ takes $I$ as one more input.

| network | (co)algebra | boxes |
|---|---|---|
| folding recurrent | $1 + A \times S \to S$ | `Nil`, `Cons` |
| unfolding recurrent | $S \to O \times S$ | `S.d -> O @ S`, fed back |
| recursive | $A + S^2 \to S$ | `Leaf`, `Node` |
| full recurrent (Mealy) | $S \to (I \to O \times S)$ | `I @ S.d -> O @ S`, fed back |
| Moore | $S \to O \times (I \to S)$ | `S.d -> O` and `I @ S.d -> S`, fed back |

The two not drawn above differ from the Mealy machine in their wiring only: an unfolding network has no input, and the output of a Moore machine reads the state before the input arrives.

```python {.marimo}
unfold = Box("unfold", S.delay(), O @ S).feedback(dom=Ty(), cod=O, mem=S)
output, update = Box("output", S.delay(), O), Box("update", I @ S.delay(), S)
moore = (I @ Diagram.copy(S.delay()) >> I @ output @ S.delay()
         >> Diagram.swap(I, O) @ S.delay() >> O @ update
         ).feedback(dom=I, cod=O, mem=S)
Equation(unfold, moore, symbol=",")
```

## What is missing

Everything above runs on today's DisCoPy.
Writing it hit two gaps:

* **Equivariant layers as boxes.** The weight tying above is computed by hand: a box whose module is the matrix tied on orbitals, or better factorised along the construction of the group, is slice 3 of [#121](https://github.com/rel-int/discopy-next/issues/121).
* **A module holding the parameters.** A network in `para.Markov[Network]` compiles to a pure function of its inputs and parameters, as JAX wants it, but [`neural.torch.Module`](https://docs.discopy.org/en/main/_api/discopy.neural.torch.html) takes every wire as an input: nothing yet turns a copied parameter wire into one `torch.nn.Parameter` ([#124](https://github.com/rel-int/discopy-next/issues/124)).
