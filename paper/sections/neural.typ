#import "../template.typ": drawing, gap

== Neural networks <sec:neural>

The package `discopy.neural` interprets diagrams as neural networks: a
wire carries a tensor, a box is a layer, and a functor compiles the
diagram to the function that a tensor framework runs. Importing it
imports no framework, so that a network is built and rewired without one.

=== Networks as diagrams <sec:neural-networks>

The objects are lists of shapes. A `Dims` has one shape per leg, _i.e._
it types a tuple of tensors: `Dims(4)` is a vector and `Dims(Dim(4, 4))`
a matrix. A `Box` is a layer from one list of shapes to another, with an
optional `module`, any callable from one tensor per input leg to one per
output leg. A `Network` is a diagram in the free traced Markov and
comarkov category on such boxes: an activation can be copied, discarded,
merged and swapped, and an output can be fed back to an input.

`Network.to_function` is the functor into `python.Function` that sends
every leg to the type of tensors and every box to its module. A box
checks each tensor against its leg on the trailing dimensions, so that
the same network runs on a batch. The result is a plain Python function:
`jax.jit` compiles it as it is, and `neural.torch.Module` wraps it as a
PyTorch module. The residual block of @fig:neural_residual, with its
weights on a wire, runs in both.

```python
import torch
from jax import jit, numpy as jnp
from discopy.monoidal import Dim
from discopy.neural.network import Box, Dims, Network, namespace
from discopy.neural.torch import Module

x, w = Dims(4), Dims(Dim(4, 4))
layer = Box('layer', x @ w, x,
            module=lambda v, W: namespace(v).tanh(v @ W))
add = Box('add', x @ x, x, module=lambda v, u: v + u)
neural_residual = Network.copy(x) @ w >> x @ layer >> add
function = jit(neural_residual.to_function())
assert function(jnp.ones(4), jnp.zeros((4, 4))).tolist() == [1, 1, 1, 1]
module = Module(neural_residual)
assert module(torch.ones(5, 4), torch.zeros(4, 4)).shape == (5, 4)
```

#drawing("neural_residual", caption: [
  A residual block as a `Network`, from an input and a matrix of weights.
]) <fig:neural_residual>

=== Learning as lenses <sec:neural-lenses>

The derivative is a second functor on the same diagram. `Network.to_lens`
sends a feedforward network to its reverse derivative @CruttwellEtAl22, a
lens of functions on tensors (@sec:constructions). A leg goes to the pair
of the types of its tensor and of its gradient, a box to the lens that
runs the layer forward, keeps its pullback as residual and calls it
backward. The `get` of the lens is `to_function` and its `put` takes the
inputs and the gradients of the outputs to the gradients of the inputs:
it is backpropagation, where composition is the chain rule and a copy
sums the gradients of its branches. The pullback of a box is its `vjp`
when it is given one, else the reverse mode of PyTorch or JAX on that box
alone.

A network with weights is a parametric map (@sec:constructions) over
`Network`, and its reverse derivative `Learner.reverse` is a parametric
lens, a learner in the sense of @FongEtAl19 and @CruttwellEtAl22. A
reparametrisation by a copy ties the weights of two layers, so that `put`
sums their gradients. One by `Lens.descent`, the lens whose `put`
subtracts the gradient times a learning rate, gives the new weights:
`Learner.descent` is one step of an optimiser. The listing checks the
gradients and the step of two tied blocks against the automatic
differentiation of PyTorch on their composite, and that the tied weight
is one parameter of the PyTorch module.

```python
from discopy import para
from discopy.neural.network import Learner

block = para.Markov[Network](x, x, param=w, inside=neural_residual)
tied = (block >> block).reparam(Network.copy(w))
learner = Learner.reverse(tied, torch.Tensor)
torch.manual_seed(0)
v, W, dy = torch.randn(4), torch.randn(4, 4), torch.randn(4)
a, A = v.clone().requires_grad_(), W.clone().requires_grad_()
y = module(module(a, A), A)
da, dA = torch.autograd.grad(y, (a, A), dy)
assert torch.equal(learner.inside.get(v, W), y)
dv, dW = learner.inside.put(v, W, dy)
assert torch.allclose(dv, da) and torch.allclose(dW, dA)
_, stepped = learner.descent(rate=.1).inside.put(v, W, dy)
assert torch.allclose(stepped, W - .1 * dA)
assert len(list(Module(tied, W).parameters())) == 1
```

=== Recursion and message passing <sec:neural-recursion>

A network with a trace is recurrent and does not compile as it is.
`Network.unroll` is the functor to feedforward networks that runs every
trace for a number of rounds: the traced legs start at zero, and every
round reads a copy of the input and the state that the round before left
(@fig:neural_unroll). This is feedback with a delay and not a trace:
one round of the trace of a swap gives zero, where yanking asks for the
identity. The lens of the unrolled network is backpropagation through
time.

Belief propagation is an instance. A constraint satisfaction problem is
a tensor network, a tensor for each factor and a wire for each node, and
the marginal of each node is a contraction, which `propagation.marginals`
computes with one `einsum`. The sum-product algorithm @Pearl88 computes
it by passing messages, and `sum_product` builds it as a network with a
trace, from the tensor of each factor to the belief of each node, whose
traced legs are the messages. When the factors form a tree, the rounds
give the marginals exactly as soon as there are as many of them as the
radius of the tree, and not one round before.

```python
import numpy as np
from discopy.neural.propagation import marginals, radius, sum_product

step = Box('step', x @ x, x @ x, module=lambda v, m: (v + m, v + m))
neural_unroll = step.trace().unroll(2)
assert neural_unroll.to_function()(np.ones(4)).tolist() == [2, 2, 2, 2]
path = ((0, 1), (1, 2))
tensors = np.array([[1., 2.], [3., 4.]]), np.array([[5., 6.], [7., 8.]])
network, exact = sum_product(path, states=2), marginals(path, *tensors)
assert radius(path) == 2 and len(network.arg.boxes) == 13
assert np.allclose(network.to_function(rounds=2)(*tensors), exact)
assert not np.allclose(network.to_function(rounds=1)(*tensors), exact)
```

#drawing("neural_unroll", caption: [
  Two rounds of the trace of a layer: the state starts at zero, and that
  of the last round is discarded.
]) <fig:neural_unroll>

The rest of the package is message passing with learned boxes, the
recursive networks of
#link("https://arxiv.org/abs/2610.18649")[Tiny Recursive Message Passing],
as PyTorch modules.

- `wiring` reads off a factor table, the nodes that each factor holds,
  the tables of integers that messages are routed along.
- `cell` holds the boxes: a `Site` that every node shares, which reads a
  message per port, updates a recurrent state and emits a belief, and a
  `DeepSet` that every factor shares.
- `recursion` fires them in synchronous rounds, grouped in cycles of
  which the first ones run without a gradient.
- `lattice` reads the answer out by deduction on a lattice of candidate
  sets, and `act` by adaptive computation time: a halting head decides
  after each segment of the recursion whether the answer is final.
- `geometric` runs the same recursion along the blocks of the symmetry
  group of the wiring, by reductions in place of gathers.

#gap[
  These recursive networks are PyTorch modules with their rounds as a
  Python loop, not yet a `Network` with a trace. The reference of their
  paper is missing: the repository gives its title and arXiv number only.
]

=== Example: a transformer block as a term <sec:neural-example>

The lambda terms of a closed category (@sec:syntax) write a network as a
program. The term below is a self-attention block: an embedded sequence
`x` is sent to a query, a key and a value, which `let` binds to three
variables, then to their attention and to a feed-forward layer `mlp`.
The five layers are constants of function types. The same block is also
a first-order diagram, a copy followed by five boxes (@fig:neural_block).

```python
from discopy import closed
from discopy.closed.markov import Ty, Tuple, let

E = Ty("E")
names = ("query", "key", "value", "mlp")
query, key, value, mlp = [(E >> E)(name) for name in names]
attention = (E.product(E, E) >> E)("attention")
term = E(lambda x: let(Tuple(query(x), key(x), value(x)),
    lambda q, k, v: let(attention(Tuple(q, k, v)), lambda a: mlp(a))))
assert term.cod == E >> E

q, k, v, m = [closed.markov.Box(name, E, E) for name in names]
neural_block = closed.markov.Diagram.copy(E, 3) >> q @ k @ v\
    >> closed.markov.Box("attention", E ** 3, E) >> m
```

#drawing("neural_block", caption: [
  A self-attention block as a first-order diagram.
]) <fig:neural_block>

A closed functor evaluates a term in any closed category, and Python
functions form one. The functor `F` sends `E` to the type of tensors and
each constant to its layer, hence the term to a function on tensors. The
functor `G` sends the diagram to a `Network` with the same layers. The
term written by hand, the term that `to_term` reads off the diagram,
with a `let` for each box, and the network agree on three tokens.

```python
from discopy import python

layers = {name: torch.nn.Linear(4, 4) for name in names}
layers["attention"] = lambda q, k, v: torch.softmax(q @ k.T / 2, -1) @ v
F = closed.markov.Functor(
    {E: python.Ty(torch.Tensor)}, lambda c: python.Function(
        lambda: layers[c.name], python.Ty(), F(c.cod)),
    cod=python.Function)
G = closed.markov.Functor({E: x}, lambda f: Box(
    f.name, G(f.dom), G(f.cod), module=layers[f.name]), cod=Network)
tokens = torch.randn(3, 4)
output = Module(G(neural_block))(tokens)
assert torch.equal(F(term)()(tokens), output)
assert torch.equal(F(neural_block.to_term())(tokens), output)
```

#gap[
  The term has no image in `Network`, which has no exponentials: `G`
  fails on it, and the diagram is its first-order form written by hand.
  Missing are the translation of a term with first-order constants to a
  `Network`, and a transformer built with it: several heads and blocks
  with residual connections and normalisation, an embedding, a training
  run to report.
]
