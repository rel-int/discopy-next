#import "../template.typ": drawing, gap

== Grammar, logic and rewriting <sec:language>

The last applications are syntax again: the derivations of a grammar,
the formulae of a logic and the nets of a rewriting system are arrows of
free categories. The package of formal grammars is `discopy.lambek`.

=== Categorial grammars <sec:lambek>

The first description of the library @FeliceEtAl21 covers pregroup
grammars. What is new is categorial: a categorial grammar is a free
biclosed category with words as boxes. A `categorial.Ty` is built from
atoms with the two exponentials `<<` and `>>`, and a derivation is a
term of the internal language (@sec:structures) with one class for each
rule: `FA` and `BA` for application, `FC` and `BC` for composition,
`FTR` and `BTR` for type raising, `FX` and `BX` for crossed composition.

The function `categorial.parse` takes the candidate terms of each word
with their scores and returns the `k` best terms of the sentence by the
CKY algorithm, combining adjacent terms by application and composition.
Type raising is not a rule of its chart: a raised term is a candidate
like any other, here for the subject of a relative clause, which has no
derivation without it.

```python
from discopy.lambek.categorial import FTR, Ty, parse

n, s = Ty("n"), Ty("s")
Alice, Bob, sleeps = n("Alice"), n("Bob"), (n >> s)("sleeps")
loves = ((n >> s) << n)("loves")
who = ((n >> n) << (s << n))("who")
words = [Alice, who, Bob, loves, sleeps]
sentence = [[(word, 0.)] for word in words]
assert not parse(sentence)
sentence[2].append((FTR(s, Bob), -1.))
(term, score), = parse(sentence, k=3, goal=s)
assert score == -1 and "FC(FTR(s, n('Bob'))," in str(term)
```

Abstract categorial grammars @deGroote01 are the module
`lambek.abstract`: a vocabulary of typed constants generates a free
closed category, and a `Lexicon` is a functor between two of them.
`to_abstract` sends a categorial term to an abstract one, forgetting the
direction of the exponentials. A parse is then a proof, which
`lambek.abstract.proofnet` holds as a proof net @Girard87. Every
occurrence of an atom in the types of the words and of the goal is a
port, given or wanted, and a `ProofNet` pairs each wanted port with a
given port of the same atom. `from_term` links a closed linear term and
`to_term` reads a term back from the goal, in beta-normal eta-long form:
a linking is correct when the reading succeeds. That linking the term
read off a net gives the net back is a law of the property matrix
(@sec:matrix).

```python
from discopy.lambek.abstract.proofnet import ProofNet

abstract = term.to_abstract()
net = ProofNet.from_term(abstract, [w.to_abstract() for w in words])
normal = "sleeps(who(n(lambda x1: loves(x1)(Bob)))(Alice))"
assert str(net.to_term()) == normal
assert net.to_term().alpha_eq(abstract.normal_form())
scores = [12 * [0] for port in net.sequent.ports]
nets = list(net.sequent.linkings(scores, 100))
assert len(nets) == 48 and sum(other.is_correct for other in nets) == 8
```

The net is that of the normal form: it does not remember that the
subject was raised. These five words have 48 linkings, of which 8 are
correct, since an abstract type does not say on which side a word takes
its arguments. `Sequent.linkings` ranks them by a matrix of scores, and
a parser is what gives the scores. `lambek.abstract.neural` is one in PyTorch, in
the two steps of @KogkalidisEtAl20: a `Tagger`
writes the type of each word one symbol at a time, so that it can write
a type that it has never seen, and a `Linker` scores each pair of ports
from the states of any encoder of the words. `Parser.search` returns the
correct net of highest joint score among the most likely sequents.

#gap[
  Two published neural parsers reproduced from their trained weights,
  each in a short listing: SPINDLE, a parser of Dutch into proof nets,
  and Bobcat, a parser for combinatory categorial grammar. The library
  has their two targets, `lambek.abstract.neural.Parser` and `categorial.parse`,
  with `cat2ty` to read the categories of a treebank. It has no encoder
  of words and no function that reads the published weights of either
  parser, and the chart of `parse` has application and composition only.
  The reference of each parser is to be added.
]

=== First-order logic <sec:peirce>

A relation is a Boolean tensor, and a formula of first-order logic is a
diagram, as in the existential graphs of @Peirce06: a box is a
predicate, a spider is a variable and a model is a functor into
`Tensor[bool]`, as for the constraint satisfaction of @sec:semantics.
Composition quantifies existentially over the wire in the middle. The
laws of this fragment, those of a cartesian bicategory @BonchiEtAl18,
are inclusions, which the pointwise order of `Tensor` and
`testable.Inequation` state (@sec:inequations). One of them says that
copying is lax natural: copying the output of `R` is included in
applying `R` to each copy of its input. The listing states it on
diagrams and checks it in every model on a set of two elements.

```python
from itertools import product
from discopy import frobenius
from discopy.monoidal import Equation
from discopy.tensor import Dim, Tensor
from discopy.testable import Inequation

x = frobenius.Ty("x")
R, copy = frobenius.Box("R", x, x), frobenius.Spider(1, 2, x)
language_lax = Equation(R >> copy, copy >> R @ R, symbol="$<=$")
models = [frobenius.Functor(
    {x: Dim(2)}, {R: Tensor[bool](bits, Dim(2), Dim(2))},
    cod=Tensor[bool]) for bits in product([0, 1], repeat=4)]
assert all(Inequation(*map(F, language_lax.terms)) for F in models)
assert sum(F(R >> copy) == F(copy >> R @ R) for F in models) == 9
```

#drawing("language_lax", caption: [
  Copying is lax natural: an inclusion for each of the 16 relations on
  two elements, and an equality for the 9 that are partial functions.
]) <fig:language_lax>

Universal quantification and disjunction need a second composition,
the relative sum: it holds of `a` and `c` when, for every `b`, `R` holds
of `a` and `b` or `S` of `b` and `c`, and it is the complement of the
composition of the complements. The two compositions are related by
linear distributivity, an inclusion again, which Hypothesis checks on
random relations on three elements.

```python
from hypothesis import given, settings

def relative_sum(R, S, no=lambda bit: not bit):
    return (R.map(no) >> S.map(no)).map(no)

@settings(max_examples=200, deadline=None)
@given(*3 * [Tensor[bool].strategy(dom=Dim(3), cod=Dim(3))])
def linear_distributivity(R, S, T):
    assert Inequation(
        R >> relative_sum(S, T), relative_sum(R >> S, T))

linear_distributivity()
```

#gap[
  The module `discopy.peirce`: a class `Formula` for the formulae of
  first-order logic, with a method that sends a formula to a diagram of
  the calculus of @BonchiEtAl24. Its diagrams have two colours, one
  for each composition, to be drawn as the coloured regions of
  @sec:regions, and its axioms are to be laws of the property matrix,
  tested as inclusions of Boolean tensors like the two above.
]

=== Example: interaction nets and Boolean circuits <sec:lafont>

The paper closes on rewriting (@sec:rewriting), with the interaction
combinators of @Lafont97. A cell is a box with its principal port as
domain and its two auxiliary ports as codomain, and a net is a
hypergraph where every wire has two ends. In a diagram, the rotation `r`
turns a cell upside down, and it is the same box of the hypergraph. A
rule rewrites two cells that meet on their principal ports: two cells of
one symbol are replaced by wires, and two of different symbols go
through each other (@fig:language_rule).

```python
from discopy.frobenius import Box, Cap, Cup, Diagram, Ty

x = Ty("x")
gamma, delta = (
    Box(name, x, x @ x, shape="triangle") for name in ("gamma", "delta"))
swap = Diagram.swap(x, x)
rules = [
    (gamma.r >> gamma, Diagram.id(x @ x)), (delta.r >> delta, swap),
    (gamma.r >> delta,
     delta @ delta >> x @ swap @ x >> gamma.r @ gamma.r)]
language_rule = Equation(*rules[2], symbol="$->$")
```

#drawing("language_rule", caption: [
  The rule for two interaction combinators of different symbols, drawn
  from the pair of diagrams that states it.
]) <fig:language_rule>

There is no module for interaction nets: reduction runs on the
double-pushout rewriter of `Hypergraph`, sound and complete for
rewriting modulo the axioms of hypergraph categories @BonchiEtAl22. The
left side of each rule is linear, so that a match has one pushout
complement. The net of @fig:language_net, from the documentation of
`Hypergraph.normalize`, reduces to two wires in five steps.

```python
net = Cap(x, x) >> delta @ delta >> x @ Cup(x, x) @ x\
    >> gamma.r >> delta >> gamma @ gamma
graphs = [[side.to_hypergraph() for side in rule] for rule in rules]
steps = list(net.to_hypergraph().normalize(*graphs))
wires = Cap(x, x) @ Cap(x, x)
assert len(steps) == 5 and all(step.is_bijective for step in steps)
assert steps[-1] == wires.to_hypergraph()
language_net = Equation(net, wires, symbol="$->$")
```

#drawing("language_net", caption: [
  A net of six cells and its normal form, two wires, reached in five
  steps of double-pushout rewriting.
]) <fig:language_net>

A circuit has inputs and outputs, and it is rewritten modulo the axioms
of symmetric categories only: `symmetric.Hypergraph` runs the same
rewriter on the matches that are convex @BonchiEtAl22b. The notebook
_Boolean circuits_ of the documentation replays in this way the
algebraic theory of Boolean circuits @Lafont03. The maps of finite sets,
the linear maps over a finite field and the Boolean functions are each
presented by generators and relations, and checked as they are proved:
every random circuit must reduce to the canonical form of its map.

#gap[
  A reducer specialised to interaction nets and checked against the
  double-pushout rewriter above: an open issue of the repository plans
  it as a module `discopy.inet`. A listing for Boolean circuits also
  waits: the rules of the notebook are schemas, written as functions of
  a match.
]
