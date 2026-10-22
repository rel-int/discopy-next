---
title: Equivalence Hypergraphs
marimo-version: 0.23.14
---

```python {.marimo}
import marimo as mo
```

# Equivalence hypergraphs, reproduced

[Tiurin, Barrett, Ghica and Hu (2024)](https://arxiv.org/abs/2406.15882) give e-graphs a categorical semantics: an acyclic e-graph is a string diagram of the free semilattice-enriched Cartesian category, where an e-class is the join of its e-nodes, drawn as an *e-box*.
Their e-hypergraphs are the combinatorial representation of such diagrams, with double-pushout rewriting, and since only the monoidal structure is needed they extend e-graphs from algebraic theories to monoidal ones.

[`discopy.ehypergraph`](https://docs.discopy.org/en/main/_api/discopy.ehypergraph.html) arrived at the e-graph case independently: an `EHypergraph` over a copy-discard category is a diagram with sharing and an equivalence on its wires, whose classes are the e-classes.
This notebook checks it against the paper and reproduces its examples.

## Joins

The paper adds one generator to string diagrams, the join $f_1 + \dots + f_n$ of parallel morphisms (Fig. 2), drawn as an e-box with a slot for each term, here a `Join`.
Composition and tensor distribute over joins, which are associative, commutative and idempotent (Fig. 4), so that a diagram with joins denotes a non-empty set of diagrams without: `distribute` expands it into a formal sum, whose terms read as a set are its semantics.
Here is the first equation of Fig. 4:

```python {.marimo}
from discopy.markov import Box, Diagram, Ty
from discopy.monoidal import Equation
from discopy.ehypergraph import Join, Markov, distribute

x = Ty("x")
f, g_1, g_2 = Box("f", x, x), Box("g_1", x, x), Box("g_2", x, x)
left, right = f >> Join(g_1, g_2), Join(f >> g_1, f >> g_2)
assert set(distribute(left).terms) == set(distribute(right).terms)
assert set(distribute(Join(f, f)).terms) == {f}
Equation(left, right)
```

## Figure 1, rule by rule

The running example is egg's: the term $(a \times 2) / 2$, with $a$ and $2$ constants, under $x \times 2 \to x \ll 1$, $(x \times y) / z \to x \times (y / z)$ and $x / x \to 1$, applied one at a time.
`from_diagram` hash-conses the term as egg does, `step` applies every match of a rule and rebuilds, and `nodes` gives the e-nodes of each e-class, two equal boxes on related inputs being one:

```python {.marimo}
a, one, two = (Box(name, Ty(), x) for name in "a12")
times, divide, shift = (Box(op, x @ x, x) for op in ("*", "/", "<<"))
rules = [(x @ two >> times, x @ one >> shift),
         (times @ x >> divide, x @ divide >> times),
         (Diagram.copy(x) >> divide, Diagram.discard(x) >> one)]
steps = [Markov.from_diagram(a @ two @ two >> times @ x >> divide)]
for _lhs, _rhs in rules:
    steps.append(steps[-1].step(
        [(_lhs.to_hypergraph(), _rhs.to_hypergraph())]))


def eclasses(graph):
    """ The e-nodes of each class by name, as the dashed boxes of Fig. 1. """
    def name(port):
        return graph.graph.boxes[graph.box_of(port)].name

    return sorted("{" + ", ".join(sorted(map(name, ports))) + "}"
                  for ports in graph.nodes.values())


def code(eclasses):
    return ", ".join(f"`{eclass}`" for eclass in eclasses)


mo.md("\n".join(
    ["| | e-classes | sections |", "|---|---|---|"] + [
        f"| ({step}) | {code(eclasses(graph))} "
        f"| {len(graph.sections())} |"
        for step, graph in zip("abcd", steps)]))
```

The classes are the dashed boxes of the first row of Fig. 1: the class of $a \times 2$ gains $a \ll 1$, the root gains $a \times (2 / 2)$, then $2 / 2$ joins $1$.
`to_diagram` is the translation of Fig. 3 and Algorithm 1, which gives the second row: a class of several e-nodes is their join, each reading the inputs of the class it needs and discarding the others, and a class read twice is copied.
Steps (a) and (b), the latter being Fig. 25 too:

```python {.marimo}
Equation(*(_step.to_diagram().foliation() for _step in steps[:2]),
         symbol="$\\mapsto$")
```

Step (c):

```python {.marimo}
steps[2].to_diagram().foliation()
```

Step (d):

```python {.marimo}
steps[3].to_diagram().foliation()
```

## One choice per class

The copy after each e-box says that a term of the e-graph chooses an e-node once per class, however many times the class is read: by Fig. 4, $(p + q) ; \Delta = p ; \Delta + q ; \Delta$.
A *section* of an `EHypergraph` makes the same choice, so that the law `EHypergraph.translation`, checked by the property matrix, states that distributing the joins of the translation gives the sections, up to the naturality of copy and discard.
The terms an e-graph represents choose per occurrence instead, here $r(p, q)$ and $r(q, p)$ too, which only agrees up to the equations of the theory, while egg's extraction of a best term chooses per class as well:

```python {.marimo}
p, q, r = Box("p", Ty(), x), Box("q", Ty(), x), Box("r", x @ x, x)
shared = Markov.from_diagram(p >> Diagram.copy(x) >> r)
shared = shared.rewrite((0, ), (), (0, ), q.to_hypergraph())
assert len(shared.sections()) == len(distribute(shared.to_diagram())) == 2
Equation(shared.to_diagram(), distribute(shared.to_diagram()),
         symbol="$\\mapsto$")
```

## The cycle the paper leaves out

Applying $x \times 1 \to x$ to (d) relates the root to $a$: the class of $a \times (2 / 2)$ now contains $a$, which one of its e-nodes reads.
The paper limits itself to acyclic e-graphs, as such a cycle needs a trace to be drawn, so `to_diagram` refuses it.
A section is an acyclic choice all the same, so that extraction is unaffected and gives $a$:

```python {.marimo}
from discopy.utils import AxiomError

cyclic = steps[3].step([((x @ one >> times).to_hypergraph(),
                         Diagram.id(x).to_hypergraph())])
try:
    cyclic.to_diagram()
    translated = True
except AxiomError:
    translated = False
assert not translated
mo.md(f"""
e-classes: {code(eclasses(cyclic))}, extracted: ${cyclic.extract().to_diagram()}$
""")
```

## Congruence: Figures 26 and 27

The proof of Lemma A.44 rewrites $f(a, b) + f(a, c)$, with $+$ a binary symbol of the theory, after merging $b$ and $c$: $f(a, b)$ and $f(a, c)$ become congruent, and rebuilding merges them upward.
`merge` relates the two constants and `rebuild` is the congruence closure, after which the two occurrences of $f$ are one e-node, shared by a copy as in Fig. 27 (e):

```python {.marimo}
b, c = Box("b", Ty(), x), Box("c", Ty(), x)
k, plus = Box("f", x @ x, x), Box("+", x @ x, x)
term = Markov.from_diagram(a @ b @ a @ c >> k @ k >> plus)
_b, _c = (term.graph.box_wires[term.graph.boxes.index(_box)][1][0]
          for _box in (b, c))
rebuilt = term.merge(_b, _c).rebuild()
assert eclasses(rebuilt) == ["{+}", "{a}", "{b, c}", "{f}"]
rebuilt.to_diagram()
```

## Deletion: Figure 17 and Appendix G

Discarding a constant, $a ; ! = \mathrm{id}_I$, is where double-pushout rewriting gets stuck in the paper: nothing in an e-box links $a$ to the component it sits in, and Appendix G introduces auxiliary edges to make room for it.
Here the equation is no rewrite at all: a section keeps a box only if it is read or is not affine, so the rule matches $a$ and changes nothing.

```python {.marimo}
g = Box("g", x, x)
stuck = Markov.from_diagram(f @ a >> x @ Diagram.discard(x))
stuck = stuck.rewrite((0, ), (0, ), (1, ), g.to_hypergraph())
deletion = (a >> Diagram.discard(x), Diagram.id(Ty()))
assert stuck.matches(deletion[0].to_hypergraph())
assert not list(stuck.saturate(deletion))
assert stuck.sections() == [f.to_hypergraph(), g.to_hypergraph()]
stuck.to_diagram()
```

## What is left

The e-hypergraphs of the paper are hierarchical: an e-box has any number of inputs and outputs and its components may contain e-boxes, so that the join of any two morphisms is one (Definition VI.1), and they represent every morphism of the free semilattice-enriched symmetric monoidal category of any theory, soundly and completely (Theorems VI.7 and VI.8).
The ones of `discopy.ehypergraph` are flat, one output per e-box, which is the image of e-graphs under the translation: a rewrite with several outputs is entered as its projections, which is only sound in a Cartesian theory, so that `rewrite` refuses a box that is not a total function.
Nested e-boxes, the join of Definition VI.1 and double-pushout rewriting over monoidal theories are left to [rel-int/discopy-next#103](https://github.com/rel-int/discopy-next/pull/103).

Two things go beyond the paper: cyclic e-graphs, as above, and effects. A `sample` that is not deterministic and a `condition` that is not affine are generators without the naturality of copy or discard, so two samples stay apart and every section keeps a condition, and `to_diagram` places them outside the e-boxes, as context, see the examples of `discopy.ehypergraph`.
