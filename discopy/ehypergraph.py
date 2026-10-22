# -*- coding: utf-8 -*-

"""
E-hypergraphs, i.e. e-graphs of diagrams with sharing.

A cartesian diagram, whose boxes are total functions, is a term with sharing,
and an e-graph gathers equal terms in e-classes :cite:`WillseyEtAl21`. An
e-hypergraph over a copy-discard category is a diagram with sharing, i.e. a
left-monogamous acyclic :class:`discopy.hypergraph.Hypergraph` whose boxes are
the e-nodes, with an equivalence on its spiders whose classes are the
e-classes, so that over cartesian diagrams it is an e-graph. A **section**
chooses a producer for each class it needs. Two samples are independent and
every section keeps a condition: equations are between cartesian diagrams,
effects being context.

Tiurin, Barrett, Ghica and Hu :cite:`TiurinEtAl24` give e-graphs their
semantics in the free semilattice-enriched Cartesian category, where an
e-class is the :class:`Join` of its e-nodes, drawn as an e-box, and
composition and tensor distribute over joins. :meth:`EHypergraph.to_diagram`
is their translation, and distributing its joins gives the sections, one
choice per class as the copy after an e-box says, see the law
:meth:`EHypergraph.translation`. Their e-hypergraphs are hierarchical, with
nested e-boxes of any arity, so as to represent any symmetric monoidal
theory, where these are flat with one output per e-box, the image of
e-graphs: an equation with several outputs is entered as its projections,
which is sound for cartesian diagrams only. Unlike the paper's, a class may
depend on itself, as in egg, which :meth:`EHypergraph.to_diagram` refuses, as
drawing it would need a trace.

Example
-------
Egg's example, ``(a * 2) / 2`` with ``a`` an input and each constant a box
with no input, the two ``2`` one class from the start as egg hash-conses
them, saturates in three rounds with egg's classes and extracts ``a``:

>>> from discopy.markov import Ty, Box, Diagram
>>> x = Ty('x')
>>> one, two = Box('1', Ty(), x), Box('2', Ty(), x)
>>> times, divide, shift = (Box(op, x @ x, x) for op in ("*", "/", "<<"))
>>> rules = [(x @ two >> times, x @ one >> shift),
...          (times @ x >> divide, x @ divide >> times),
...          (Diagram.copy(x) >> divide, Diagram.discard(x) >> one),
...          (x @ one >> times, Diagram.id(x))]
>>> graph = Markov.from_diagram(x @ two @ two >> times @ x >> divide)
>>> rounds = list(graph.saturate(*rules))
>>> [len(set(rewritten.classes)) for rewritten in [graph] + rounds]
[4, 6, 5, 4]
>>> print(rounds[-1].extract().to_diagram())
Id(x)

A ``sample`` is not deterministic and a ``condition`` not affine: under
``x - x = 0``, two independent samples of ``p`` stay apart, while one sample
shared cancels, then is discarded with ``p`` unless a condition reads it:

>>> zero, minus = Box('0', Ty(), x), Box('-', x @ x, x)
>>> sample = Box('sample', x, x, is_deterministic=False)
>>> condition = Box('condition', x, Ty(), is_affine=False)
>>> cancel = (Diagram.copy(x) >> minus, Diagram.discard(x) >> zero)
>>> for program in (Diagram.copy(x) >> sample @ sample >> minus,
...                 sample >> Diagram.copy(x) >> minus,
...                 sample >> Diagram.copy(x, 3) >> condition @ minus):
...     graph = Markov.from_diagram(program)
...     *_, last = graph, *graph.saturate(cancel)
...     print(last.extract().to_diagram())
Copy(x) >> sample @ sample >> -
Discard(x) @ 0
sample >> 0 @ condition

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    EHypergraph
    Join

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        partition
        is_pattern
        ast_size
        distribute
"""

from collections import Counter
from collections.abc import Callable, Iterator
from functools import cached_property, reduce
from itertools import accumulate, pairwise, repeat, takewhile
from math import inf
from operator import add
from typing import TYPE_CHECKING, Self

from networkx import DiGraph
from networkx.algorithms.isomorphism import is_isomorphic

from discopy import markov, messages, monoidal
from discopy.abc import HypergraphCategory, MonoidalCategory
from discopy.axioms import (
    GENERATORS, Count, Equation, Hom, Obj, Testable, Var, axiom)
from discopy.drawing import Node
from discopy.hypergraph import Hypergraph
from discopy.monoidal import Ty
from discopy.utils import (
    AxiomError, NamedGeneric, assert_isparallel, classproperty, factory_name,
    from_tree, pushout, resolve_factory, unbiased)

if TYPE_CHECKING:
    from hypothesis import strategies as st


def partition(n: int, pairs) -> tuple[int, ...]:
    """
    The finest partition of ``range(n)`` relating each pair, as the least
    element of the class of each element.

    >>> partition(4, [(3, 1), (1, 2)])
    (0, 1, 1, 1)
    """
    label = list(range(n))

    def find(i):
        while label[i] != i:
            label[i] = label[label[i]]
            i = label[i]
        return i

    for i, j in pairs:
        i, j = find(i), find(j)
        label[max(i, j)] = min(i, j)
    return tuple(map(find, range(n)))


def relabel(wires, label: Callable):
    """ Relabel the spiders of a :class:`hypergraph.Wiring`. """
    dom_wires, box_wires, cod_wires = wires
    return tuple(map(label, dom_wires)), tuple(
        (tuple(map(label, src)), tuple(map(label, tgt)))
        for src, tgt in box_wires), tuple(map(label, cod_wires))


def is_pattern(graph: Hypergraph) -> bool:
    """ Whether a hypergraph is a pattern, see :meth:`EHypergraph.matches`. """
    return all(box.is_cartesian for box in graph.boxes)\
        and graph.is_left_monogamous and graph.is_acyclic\
        and not set(graph.dom_wires) & set(graph.cod_wires)


def pattern(graph: Hypergraph) -> Hypergraph:
    """ A hypergraph without its wires from the domain to the codomain. """
    dom_wires, box_wires, cod_wires = graph.wires
    kept = [k for k, s in enumerate(cod_wires) if s not in dom_wires]
    return type(graph)(
        graph.dom, type(graph.cod)().tensor(*(graph.cod[k] for k in kept)),
        graph.boxes, (dom_wires, box_wires, tuple(cod_wires[k] for k in kept)))


def rewire(graph: Hypergraph, i: int, j: int) -> Hypergraph:
    """
    A hypergraph in a topological order with the ``i``-th input of its boxes
    read from the ``j``-th other spider of its type before it, which a matcher
    missing a constraint finds where the original is, for the laws.
    """
    graph = graph.topological_order()
    wires, types = list(graph.flat_wires), graph.spider_types
    inputs = [p for p, node in enumerate(graph.ports) if node.kind == "dom"]
    port = inputs[i % len(inputs)] if inputs else 0
    others = [s for s in dict.fromkeys(wires[:port])
              if s != wires[port] and types[s] == types[wires[port]]]
    if not others:
        return graph
    wires[port] = others[j % len(others)]
    return type(graph)(
        graph.dom, graph.cod, graph.boxes, graph.rebracket(wires))


def ast_size(_box) -> int:
    """ The cost of a box in egg's ``AstSize``: one, whatever the box. """
    return 1


class Join(monoidal.Bubble, markov.Box):
    """
    The join of parallel diagrams, an e-box of the string diagrams of a
    semilattice-enriched category :cite:`TiurinEtAl24`, drawn as a frame with
    a slot for each term. It is deterministic when it has one term, which is,
    and affine when every term is, discarding being idempotent.

    Parameters:
        terms : The diagrams joined, with the same domain and codomain.
        params : Passed to :class:`monoidal.Bubble`, e.g. by :meth:`dagger`.

    Example
    -------
    >>> from discopy.markov import Ty, Box
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> join = Join(f, g)
    >>> assert join.is_affine and not join.is_deterministic
    >>> print(join)
    Join(f, g)
    >>> print(distribute(x @ join >> Box('h', x @ x, x)))
    (x @ f >> h) + (x @ g >> h)
    """
    is_id_on_objects = True

    def __init__(self, *terms, **params):
        params.pop("method", None)
        super().__init__(*terms, method="join", **params)
        for term in terms:
            assert_isparallel(self, term)
        self.deterministic = len(terms) == 1 and terms[0].is_deterministic
        self.affine = all(term.is_affine for term in terms)

    def __str__(self):
        return f"Join({', '.join(map(str, self.args))})"


def distribute(diagram: monoidal.Diagram) -> markov.Sum:
    """
    The formal sum of the terms of a diagram with joins, composition and
    tensor distributing over each of them :cite:`TiurinEtAl24` (Fig. 4), see
    :class:`Join`.
    """
    functor = markov.Functor(lambda ob: ob, lambda box: reduce(
        add, map(functor, box.args)) if isinstance(box, Join) else box)
    result = functor(diagram)
    return result if isinstance(result, markov.Sum)\
        else markov.Sum((result, ))


class EHypergraph(
        MonoidalCategory, NamedGeneric['category'], Testable):
    """
    An e-hypergraph over a copy-discard ``category``, equal to another up to
    renaming its spiders, see :mod:`discopy.ehypergraph`.

    Parameters:
        graph : A left-monogamous acyclic hypergraph, its boxes the e-nodes.
        classes : The least spider related to each spider, the e-classes,
            the identity relation by default.

    Example
    -------
    Rewriting ``f``, the box 0 from the spider 0 to 1, into ``g`` gives two
    sections, ``f`` extracted first unless it costs more:

    >>> from discopy.markov import Ty, Box
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> graph = Markov.from_diagram(f).rewrite(
    ...     (0, ), (0, ), (1, ), g.to_hypergraph())
    >>> assert graph.sections() == [f.to_hypergraph(), g.to_hypergraph()]
    >>> print(graph.extract(), graph.extract(lambda box: 1 + (box == f)))
    f g
    """
    ob = classproperty(lambda cls: cls.category.ob)

    def __init__(
            self, graph: Hypergraph, classes: tuple[int, ...] | None = None):
        if not issubclass(self.category or type, markov.Diagram):
            raise AxiomError(messages.NOT_MARKOV.format(factory_name(
                type(self))))
        if not graph.is_left_monogamous or not graph.is_acyclic:
            raise AxiomError(messages.NOT_SHARING.format(repr(graph)))
        n = graph.n_spiders
        classes = tuple(range(n) if classes is None else classes)
        if len(classes) != n or any(
                graph.spider_types[s] != graph.spider_types[c]
                for s, c in enumerate(classes)):
            raise AxiomError(messages.TYPE_ERROR.format(
                f"a class of one type for each of {n} spiders", classes))
        self.graph, self.classes = graph, partition(n, enumerate(classes))

    dom = property(lambda self: self.graph.dom)
    cod = property(lambda self: self.graph.cod)

    @classmethod
    def from_diagram(cls, diagram) -> Self:
        """ The e-hypergraph of a diagram, rebuilt as egg hash-conses. """
        return cls(Hypergraph[cls.category].from_diagram(diagram)).rebuild()

    @classmethod
    def id(cls, dom=None) -> Self:
        return cls(Hypergraph[cls.category].id(dom))

    def carry(self, other: Self, graph: Hypergraph, wires, labels) -> Self:
        """ The relations of two e-hypergraphs, their spiders labelled by a
        pair of functions, on a hypergraph built from the wires labelled. """
        dom_wires, box_wires, cod_wires = wires
        label = dict(zip(dom_wires + sum((
            src + tgt for src, tgt in box_wires), ()) + cod_wires,
            graph.flat_wires))
        return type(self)(graph, partition(graph.n_spiders, (
            (label[f(s)], label[f(c)]) for f, part in zip(labels, (
                self, other)) for s, c in enumerate(part.classes))))

    @unbiased
    def then(self, other: Self) -> Self:
        """ Composition, the relations carried along the pushout. """
        left, right = pushout(
            self.graph.n_spiders, other.graph.n_spiders,
            self.graph.cod_wires, other.graph.dom_wires)
        (dom_wires, box_wires, _), (_, others, cod_wires) = relabel(
            self.graph.wires, left.get), relabel(other.graph.wires, right.get)
        return self.carry(other, self.graph >> other.graph, (
            dom_wires, box_wires + others, cod_wires), (left.get, right.get))

    @unbiased
    def tensor(self, other: Self) -> Self:
        """ Tensor, the disjoint union of the hypergraphs and relations. """
        n = self.graph.n_spiders
        return self.carry(other, self.graph @ other.graph, tuple(map(
            tuple.__add__, self.graph.wires, relabel(
                other.graph.wires, n.__add__))), ((0).__add__, n.__add__))

    def merge(self, *spiders: int) -> Self:
        """ Relate spiders of one type, i.e. state that they are equal. """
        pairs = list(enumerate(self.classes)) + list(zip(spiders, spiders[1:]))
        return type(self)(self.graph, partition(self.graph.n_spiders, pairs))

    def pair(self, i: int, j: int) -> tuple[int, ...]:
        """ The ``i``-th spider and the ``j``-th of its type, for the laws. """
        types = self.graph.spider_types
        if not types:
            return ()
        same = [s for s, t in enumerate(types) if t == types[i % len(types)]]
        return i % len(types), same[j % len(same)]

    def twice(self) -> Self:
        """ Two copies side by side on related inputs, rebuilt. """
        result, n = self @ self, len(self.dom)
        for k in range(n):
            result = result.merge(*result.graph.dom_wires[k::n])
        return result.rebuild()

    def inputs(self, box: int) -> tuple[int, ...]:
        """ The classes a box reads, given by its index. """
        return tuple(self.classes[s] for s in self.graph.box_wires[box][0])

    def box_of(self, port: int) -> int | None:
        """ The box of an output port, :obj:`None` for a domain position. """
        node = self.graph.ports[port]
        return node.depth if node.kind == "cod" else None

    def reads(self, port: int) -> tuple[int, ...]:
        """ The classes the producer at a port reads, none for an input. """
        return () if (box := self.box_of(port)) is None else self.inputs(box)

    def rebuild(self) -> Self:
        """ The congruence closure, egg's rebuild, relating the outputs of
        equal deterministic boxes on related inputs, not of two samples. """
        graph, classes = self.graph, self.classes
        while True:
            first, pairs = {}, list(enumerate(classes))
            for box, (src, tgt) in zip(graph.boxes, graph.box_wires):
                if box.is_deterministic:
                    key = box, tuple(classes[s] for s in src)
                    pairs += zip(first.setdefault(key, tgt), tgt)
            if partition(graph.n_spiders, pairs) == classes:
                return type(self)(graph, classes)
            classes = partition(graph.n_spiders, pairs)

    def add(self, box, inputs: tuple[int, ...]) -> tuple[Self, tuple]:
        """ Add a box reading some spiders, egg's add, its outputs the next
        spiders, or those of an equal deterministic box on their classes. """
        key = tuple(self.classes[s] for s in inputs)
        for j, other in enumerate(self.graph.boxes):
            if box.is_deterministic and other == box and self.inputs(j) == key:
                return self, self.graph.box_wires[j][1]
        graph, n = self.graph, self.graph.n_spiders
        outputs = tuple(range(n, n + len(box.cod)))
        wires = graph.dom_wires, graph.box_wires + (
            (tuple(inputs), outputs), ), graph.cod_wires
        return type(self)(type(graph)(
            graph.dom, graph.cod, graph.boxes + (box, ), wires),
            self.classes + outputs), outputs

    def assert_equation(self, region: tuple[int, ...], dom: tuple, cod: tuple,
                        rhs: Hypergraph):
        """ Raise unless :meth:`rewrite` can enter an equation, see there. """
        graph, classes = self.graph, self.classes
        for box in tuple(graph.boxes[j] for j in region) + rhs.boxes:
            if not box.is_cartesian:
                raise AxiomError(messages.NOT_CARTESIAN.format(box))
        if not rhs.is_left_monogamous or not rhs.is_acyclic:
            raise AxiomError(messages.NOT_SHARING.format(repr(rhs)))
        if (len(dom), len(cod)) != (len(rhs.dom), len(rhs.cod)) or not all(
                not rhs.spider_wires[t][1] if s is None
                else graph.spider_types[s] == rhs.spider_types[t]
                for s, t in zip(dom + cod, rhs.dom_wires + rhs.cod_wires)):
            raise AxiomError(messages.NOT_A_BOUNDARY.format(rhs, dom, cod))
        reached, todo = {classes[s] for s in dom if s is not None}, set(region)
        while fired := {j for j in todo if set(self.inputs(j)) <= reached}:
            todo -= fired
            reached |= {
                classes[s] for j in fired for s in graph.box_wires[j][1]}
        if todo or any(classes[s] not in reached for s in cod):
            raise AxiomError(messages.NOT_A_REGION.format(region, dom, cod))

    def rewrite(self, region: tuple[int, ...], dom: tuple, cod: tuple,
                rhs: Hypergraph) -> Self:
        """
        State that the boxes of a region from ``dom`` to ``cod``, :obj:`None`
        where ``rhs`` does not read, equal ``rhs``: glue it onto ``dom`` with
        :meth:`add` and relate ``cod`` to its outputs, keeping the region, or
        raise :class:`AxiomError` on a box that is not cartesian, on other
        types, on ``rhs`` without sharing or on a region that cannot fire.
        """
        dom, cod = tuple(dom), tuple(cod)
        self.assert_equation(region, dom, cod, rhs)
        rhs, result = rhs.topological_order(), self
        image = dict(zip(rhs.dom_wires, dom))
        for box, (src, tgt) in zip(rhs.boxes, rhs.box_wires):
            result, outputs = result.add(box, tuple(image[s] for s in src))
            image.update(zip(tgt, outputs))
        for spider, target in zip(cod, rhs.cod_wires):
            result = result.merge(spider, image[target])
        return result

    def matches(self, lhs: Hypergraph) -> list[tuple[tuple, tuple, tuple]]:
        """
        The matches modulo the classes of a pattern, a cartesian diagram with
        sharing whose boxes produce its codomain: egg's e-matching, binding
        each box from the codomain up to an equal box, flags included, with
        :meth:`bind`. A match is ``(region, dom, cod)``, :obj:`None` where the
        pattern does not read, so that ``rewrite(*match, rhs)`` applies a rule.

        >>> from discopy.markov import Ty, Box, Diagram
        >>> x = Ty('x')
        >>> f, g = Box('f', x, x), Box('g', x @ x, x)
        >>> graph = Markov.from_diagram(Diagram.copy(x) >> f @ f >> g)
        >>> graph.matches((Diagram.copy(x) >> g).to_hypergraph())
        [((2,), (1,), (3,))]
        """
        if not is_pattern(lhs):
            raise AxiomError(messages.NOT_A_PATTERN.format(lhs))
        lhs, partial = lhs.topological_order(), [({}, ())]
        for j in reversed(range(len(lhs.boxes))):
            partial = [(new, region + (h, )) for env, region in partial
                       for h, new in self.bind(lhs, j, env)]
        return list(dict.fromkeys((
            tuple(sorted(set(region))), tuple(map(env.get, lhs.dom_wires)),
            tuple(map(env.get, lhs.cod_wires))) for env, region in partial))

    def bind(self, lhs: Hypergraph, j: int, env: dict) -> Iterator[tuple]:
        """ The boxes equal to the ``j``-th box of a pattern on the classes a
        binding gives its spiders, among the producers of the class of one of
        its outputs if any, each with the binding it extends to. """
        (src, tgt), classes = lhs.box_wires[j], self.classes
        k = next((k for k, s in enumerate(tgt) if s in env), None)
        hosts = range(len(self.graph.boxes)) if k is None else [
            node.depth for node in map(
                self.graph.ports.__getitem__, self.producers[env[tgt[k]]])
            if node.kind == "cod" and node.i == k]
        for h in hosts:
            new = dict(env)
            if self.graph.boxes[h] == lhs.boxes[j] and all(
                    new.setdefault(p, classes[s]) == classes[s] for p, s in
                    zip(src + tgt, sum(self.graph.box_wires[h], ()))):
                yield h, new

    def step(self, rules: list[tuple[Hypergraph, Hypergraph]]) -> Self:
        """ A round of egg's schedule: the matches as it begins, rewritten. """
        result = self
        for match, rhs in [(match, rhs) for lhs, rhs in rules
                           for match in self.matches(lhs)]:
            result = result.rewrite(*match, rhs)
        return result.rebuild()

    def saturate(self, *rules) -> Iterator[Self]:
        """
        Each :meth:`step` that adds a box or relates two classes, until one
        does neither, if ever. A rule is a pair ``(lhs, rhs)`` of cartesian
        diagrams of one type, ``lhs`` a pattern and ``rhs`` reading only what
        ``lhs`` reads, as in egg, else it raises :class:`AxiomError`.
        """
        def reads(graph):
            return {k for k, s in enumerate(graph.dom_wires)
                    if graph.spider_wires[s][1]}

        def size(graph):
            return len(graph.graph.boxes), len(set(graph.classes))

        pairs = []
        for lhs, rhs in rules:
            left, right = lhs.to_hypergraph(), rhs.to_hypergraph()
            if (lhs.dom, lhs.cod) != (rhs.dom, rhs.cod) or not is_pattern(
                    left) or not all(box.is_cartesian for box in right.boxes)\
                    or not reads(right) <= reads(left):
                raise AxiomError(messages.NOT_A_RULE.format(lhs, rhs))
            pairs.append((left, right))
        rounds = pairwise(accumulate(
            repeat(pairs), type(self).step, initial=self.rebuild()))
        return (new for _, new in takewhile(
            lambda pair: size(pair[0]) != size(pair[1]), rounds))

    @cached_property
    def producers(self) -> dict[int, tuple[int, ...]]:
        """ The producer ports of each class, domain positions first. """
        result: dict[int, list[int]] = {}
        for spider, (ports, _) in enumerate(self.graph.spider_wires):
            result.setdefault(self.classes[spider], []).extend(ports)
        return {q: tuple(sorted(ports)) for q, ports in result.items()}

    def occurrence(self, box: int):
        """ A box up to the classes it reads if it is deterministic, else its
        index, i.e. its e-node up to the output, as :meth:`key` and
        :meth:`roots` read it. """
        return (self.graph.boxes[box], self.inputs(box))\
            if self.graph.boxes[box].is_deterministic else box

    def key(self, port: int):
        """ The e-node of a producer port: its :meth:`occurrence` and output
        for a deterministic box, else the port itself. """
        box = self.box_of(port)
        if box is None or not self.graph.boxes[box].is_deterministic:
            return port
        return self.occurrence(box) + (self.graph.ports[port].i, )

    @cached_property
    def nodes(self) -> dict[int, tuple[int, ...]]:
        """ The e-nodes of each class, as their least producer port: equal
        deterministic boxes on related inputs are one, egg's hashcons
        invariant :cite:`WillseyEtAl21`. """
        return {q: tuple(sorted({self.key(port): port for port in reversed(
            ports)}.values())) for q, ports in self.producers.items()}

    def roots(self) -> tuple[int, ...]:
        """ The boxes every section keeps, not affine, once for equal
        deterministic ones on related inputs: a constraint holds once. """
        found = {}
        for j, box in enumerate(self.graph.boxes):
            if not box.is_affine:
                found.setdefault(self.occurrence(j), j)
        return tuple(found.values())

    def needs(self) -> tuple[int, ...]:
        """ The classes every section needs, of the codomain and roots. """
        return tuple(self.classes[s] for s in self.graph.cod_wires) + tuple(
            q for j in self.roots() for q in self.inputs(j))

    def section(self, choice: dict[int, int]) -> Hypergraph:
        """ The section of a choice of a producer for the classes it needs, the
        roots and chosen boxes in a causal order on a spider per class and
        fresh ones for the others, or :class:`AxiomError` if it is cyclic. """
        graph, classes, needed, todo = self.graph, self.classes, set(), list(
            self.needs())
        while todo:
            if (q := todo.pop()) not in needed:
                needed.add(q)
                todo += self.reads(choice[q])
        kept = sorted(set(self.roots()).union(
            j for q in needed if (j := self.box_of(choice[q])) is not None))
        ports = [min(producers) for producers, _ in graph.spider_wires]

        def produce(wires):
            return tuple(classes[s] if classes[s] in needed and choice[classes[
                s]] == ports[s] else graph.n_spiders + s for s in wires)

        result = type(graph)(graph.dom, graph.cod, tuple(
            graph.boxes[j] for j in kept), (produce(graph.dom_wires), tuple(
                (self.inputs(j), produce(graph.box_wires[j][1])) for j in kept
            ), tuple(classes[s] for s in graph.cod_wires)))
        if not result.is_acyclic:
            raise AxiomError(messages.NOT_ACYCLIC.format(result))
        return result.topological_order()

    def sections(self) -> list[Hypergraph]:
        """ The section of every acyclic choice of an e-node per class, where
        the terms of an e-graph choose per occurrence, so that relating
        classes removes some. """
        producers, result = self.nodes, []

        def extend(choice, todo):
            if not todo:
                try:
                    result.append(self.section(choice))
                except AxiomError:
                    pass
                return
            q, rest = todo[0], todo[1:]
            for port in (choice[q], ) if q in choice else producers[q]:
                reads = () if q in choice else self.reads(port)
                extend({**choice, q: port}, rest + reads)

        extend({}, self.needs())
        return result

    def tree_cost(self, box: int, costs: dict[int, float],
                  box_cost: Callable = ast_size) -> float:
        """ The tree cost of a box, given those of the classes it reads. """
        return box_cost(self.graph.boxes[box]) + sum(
            costs[q] for q in self.inputs(box))

    def relax(self, box_cost: Callable = ast_size, input_cost: float = 1
              ) -> tuple[dict[int, float], dict[int, int]]:
        """ The least tree cost of each class and its least producer of that
        cost, a box costing ``box_cost(box)`` and an input ``input_cost``, by
        Bellman-Ford relaxation, exact when every box costs at least one. """
        costs = dict.fromkeys(self.producers, inf)

        def weight(port):
            box = self.box_of(port)
            return input_cost if box is None\
                else self.tree_cost(box, costs, box_cost)

        while lower := {q: min(map(weight, ports)) for q, ports in
                        self.producers.items()
                        if min(map(weight, ports)) < costs[q]}:
            costs.update(lower)
        return costs, {q: next(p for p in ports if weight(p) == costs[q])
                       for q, ports in self.producers.items()}

    def cost(self, box_cost: Callable = ast_size, input_cost: float = 1
             ) -> float:
        """ The least tree cost of a section, of its codomain and roots. """
        costs, _ = self.relax(box_cost, input_cost)
        return sum(costs[self.classes[s]] for s in self.graph.cod_wires) + sum(
            self.tree_cost(j, costs, box_cost) for j in self.roots())

    def extract(self, box_cost: Callable = ast_size, input_cost: float = 1
                ) -> Hypergraph:
        """ A section of least tree cost, egg's extraction. """
        return self.section(self.relax(box_cost, input_cost)[1])

    def is_inside(self, box: int) -> bool:
        """ Whether a box is an e-node inside an e-box, being cartesian with
        one output, rather than context, see :meth:`to_diagram`. """
        return self.graph.boxes[box].is_cartesian and len(
            self.graph.boxes[box].cod) == 1

    def to_diagram(self) -> monoidal.Diagram:
        """
        The string diagram of the e-hypergraph, the translation of e-graphs
        :cite:`TiurinEtAl24` (Fig. 3 and Algorithm 1): a class of several
        e-nodes is their :class:`Join`, each reading the inputs of the class
        it needs and discarding the others, a class of one e-node is that
        e-node, and a class read twice is copied. A box that is not cartesian
        with one output stays outside the joins, once per e-node, as context,
        an output of it being an input of the joins that may choose it.
        :class:`AxiomError` if a class depends on itself, as the paper only
        translates acyclic e-graphs.
        """
        graph, category = self.graph, Hypergraph[self.category]
        types, placed, done, context = dict(enumerate(graph.dom)), [], {}, {}
        for j in range(len(graph.boxes)):
            if not self.is_inside(j):
                context.setdefault(self.occurrence(j), j)

        def place(box, inputs):
            outputs = tuple(range(len(types), len(types) + len(box.cod)))
            types.update(zip(outputs, box.cod))
            placed.append((box, (inputs, outputs)))
            return outputs

        def enode(port):
            node = graph.ports[port]
            if node.kind == "input":
                return (node.i, ), None
            if not self.is_inside(node.depth):
                box = self.occurrence(node.depth)
                return (emit("box", box)[node.i], ), None
            return tuple(emit("class", q) for q in self.inputs(
                node.depth)), graph.boxes[node.depth]

        def term(inputs, reads, box):
            dom, n = type(graph.dom)().tensor(*map(
                types.__getitem__, inputs)), len(inputs)
            wires = tuple(map(inputs.index, reads))
            if box is None:
                return category(dom, types[reads[0]], (), (
                    tuple(range(n)), (), wires)).to_diagram()
            return category(dom, box.cod, (box, ), (
                tuple(range(n)), ((wires, (n, )), ), (n, ))).to_diagram()

        def join(q):
            enodes = list(map(enode, self.nodes[q]))
            if len(enodes) == 1:
                (reads, box), = enodes
                return reads[0] if box is None else place(box, reads)[0]
            inputs = tuple(dict.fromkeys(
                spider for reads, _ in enodes for spider in reads))
            return place(Join(*(
                term(inputs, *enode) for enode in enodes)), inputs)[0]

        def emit(kind, key):
            if (kind, key) in done and done[kind, key] is None:
                raise AxiomError(messages.NOT_ACYCLIC.format(self))
            if (kind, key) not in done:
                done[kind, key] = None
                done[kind, key] = join(key) if kind == "class" else place(
                    graph.boxes[context[key]], tuple(
                        emit("class", q) for q in self.inputs(context[key])))
            return done[kind, key]

        cod_wires = tuple(
            emit("class", self.classes[s]) for s in graph.cod_wires)
        for item in [("box", key) for key in context] + [
                ("class", q) for q in sorted(set(self.classes))]:
            emit(*item)
        boxes, box_wires = zip(*placed) if placed else ((), ())
        return category(graph.dom, graph.cod, boxes, (
            tuple(range(len(graph.dom))), box_wires, cod_wires)).to_diagram()

    def to_graph(self) -> DiGraph:
        """ The labelled graph of the hypergraph, with a node per class. """
        graph = self.graph.to_graph()
        for spider, (label, obj) in enumerate(zip(
                self.classes, self.graph.spider_types)):
            graph.add_node(("class", label), box="class")
            graph.add_edge(("class", label), Node("spider", i=spider, obj=obj))
        return graph

    def __eq__(self, other) -> bool:
        return isinstance(other, EHypergraph) and self.graph == other.graph\
            and is_isomorphic(self.to_graph(), other.to_graph(), dict.__eq__)

    def __hash__(self) -> int:
        return hash((self.graph, *sorted(Counter(self.classes).values())))

    def __repr__(self) -> str:
        return factory_name(type(self))\
            + f"(graph={self.graph!r}, classes={self.classes!r})"

    def to_tree(self) -> dict:
        """ Serialise the e-hypergraph, with the name of its category. """
        return {'factory': factory_name(EHypergraph),
                'category': factory_name(self.category),
                'graph': self.graph.to_tree(), 'classes': list(self.classes)}

    @classmethod
    def from_tree(cls, tree: dict) -> Self:
        """ Decode a serialised e-hypergraph, see :meth:`to_tree`. """
        factory = cls[resolve_factory(tree['category'])]
        return factory(from_tree(tree['graph']), tuple(tree['classes']))

    @classmethod
    def strategy(cls, effects: bool = True, max_merges: int = 3, **params
                 ) -> "st.SearchStrategy[EHypergraph]":
        """ An acyclic hypergraph, its types and boxes named after two
        generators, some not deterministic or not affine if ``effects``, then
        up to ``max_merges`` merges, ``params`` going to the hypergraph. """
        from hypothesis import strategies as st

        category = cls.category
        if not issubclass(category or type, markov.Diagram)\
                or issubclass(category, HypergraphCategory):
            raise NotImplementedError(
                f"No strategy implemented for {factory_name(cls)}.")
        words = st.lists(st.sampled_from(GENERATORS[:2]), max_size=2)
        graphs = Hypergraph[category].strategy(**{
            "types": words.map(lambda word: category.ob(*word)),
            "acyclic": True, "max_scalars": 0, **params})
        effect = st.booleans() if effects else st.just(False)

        @st.composite
        def ehypergraphs(draw):
            graph = draw(graphs)
            boxes = tuple(category.box_factory(
                draw(st.sampled_from(GENERATORS[:2])), box.dom, box.cod,
                is_deterministic=not draw(effect), is_affine=not draw(effect))
                for box in graph.boxes)
            result = cls(type(graph)(graph.dom, graph.cod, boxes, graph.wires))
            for _ in range(draw(st.integers(0, max_merges))):
                result = result.merge(*result.pair(
                    draw(st.integers(0, 9)), draw(st.integers(0, 9))))
            return result

        return ehypergraphs()

    @axiom
    def graph_functoriality[
            A: Obj[Ty], B: Obj[Ty], C: Obj[Ty], D: Obj[Ty], E: Obj[Ty]](
            cls, f: Hom[Self, A, B], g: Hom[Self, B, C], h: Hom[Self, D, E]
    ) -> Equation[Hypergraph]:
        """ The hypergraph of an e-hypergraph is a strict monoidal functor. """
        return Equation(((f >> g) @ h).graph, (f.graph >> g.graph) @ h.graph)

    @axiom
    def merge_equivalence[I: Count, J: Count, K: Count](
            cls, graph: Self, i: Var[int, I], j: Var[int, J],
            k: Var[int, K]) -> Equation[tuple[Self, ...]]:
        """ Merging is reflexive, symmetric and transitive. """
        if not graph.graph.n_spiders:
            return Equation((), ())
        (s, t), (_, u) = graph.pair(i, j), graph.pair(i, k)
        chained = graph.merge(s, t).merge(t, u)
        return Equation(
            (graph.merge(s, s), graph.merge(s, t), chained.merge(s, u)),
            (graph, graph.merge(t, s), chained))

    @axiom
    def rebuild_closure[I: Count, J: Count](
            cls, graph: Self, i: Var[int, I], j: Var[int, J]
    ) -> Equation[Self]:
        """ Rebuilding is idempotent, extensive and confluent. """
        pair = graph.pair(i, j)
        closed = graph.merge(*pair).rebuild()
        return Equation(closed, closed.rebuild(), closed.merge(*pair),
                        graph.rebuild().merge(*pair).rebuild())

    @axiom
    def rebuild_minimality(cls, graph: Self) -> Equation[tuple[int, ...]]:
        """ A closure is generated by the relation and congruent outputs. """
        closed, first, pairs = graph.rebuild(), {}, list(enumerate(
            graph.classes))
        for spider, (ports, _) in enumerate(closed.graph.spider_wires):
            node, j = closed.graph.ports[min(ports)], closed.box_of(min(ports))
            if j is not None and closed.graph.boxes[j].is_deterministic:
                key = closed.classes[spider], closed.graph.boxes[j], node.i
                pairs.append((first.setdefault(key, spider), spider))
        return Equation(partition(len(closed.classes), pairs), closed.classes)

    @axiom
    def congruence[A: Obj[Ty], B: Obj[Ty], C: Obj[Ty]](
            cls, f: Hom[Self, A, B], g: Hom[Self, B, C]
    ) -> Equation[tuple[bool, ...]]:
        """ Relating the inputs of two copies relates two outputs exactly when
        every box they depend on is deterministic. """
        composite, m = f >> g, len(g.cod)
        graph, twice = composite.graph, type(f)(composite.graph).twice()
        cod, classes = twice.graph.cod_wires, twice.classes

        def determined(spider):
            box = composite.box_of(min(graph.spider_wires[spider][0]))
            return box is None or graph.boxes[box].is_deterministic and all(
                map(determined, graph.box_wires[box][0]))

        return Equation(
            tuple(classes[cod[k]] == classes[cod[m + k]] for k in range(m)),
            tuple(map(determined, graph.cod_wires)))

    @axiom
    def extraction_soundness(cls, graph: Self) -> Equation[tuple]:
        """ What is extracted is a section, decodes to a diagram encoding back
        to it, and is what its own e-hypergraph extracts. """
        extracted = graph.extract()
        return Equation((extracted in graph.sections(), extracted), (
            True, extracted.to_diagram().to_hypergraph()), (
            True, type(graph)(extracted).extract()))

    @axiom
    def extraction_optimality(cls, graph: Self) -> Equation[int]:
        """ What is extracted costs :meth:`cost`, no section costs less. """
        return Equation(type(graph)(graph.extract()).cost(), graph.cost(), min(
            type(graph)(section).cost() for section in graph.sections()))

    @axiom
    def translation(cls, graph: Self) -> Equation[frozenset]:
        """ Distributing the joins of the translation of an e-hypergraph
        whose classes do not depend on themselves gives its sections, modulo
        the naturality of copy and discard, i.e. one choice per class. """
        try:
            terms = distribute(graph.to_diagram()).terms
        except AxiomError:
            return Equation(frozenset(), frozenset())

        def normal(section: Hypergraph) -> Hypergraph:
            return type(graph)(section).rebuild().extract()

        return Equation(
            frozenset(normal(term.to_hypergraph()) for term in terms),
            frozenset(map(normal, graph.sections())))

    @axiom
    def effects_kept(cls, graph: Self) -> Equation[bool]:
        """ Every section of two copies keeps every box that is not affine,
        once for equal deterministic ones on related inputs. """
        def effects(boxes, inputs):
            pairs = [(box, src) for box, src in zip(boxes, inputs)
                     if not box.is_affine]
            hard = {pair for pair in pairs if pair[0].is_deterministic}
            return len(hard), Counter(
                box for box, _ in pairs if not box.is_deterministic)

        twice = graph @ graph
        expected = effects(twice.graph.boxes, map(
            twice.inputs, range(len(twice.graph.boxes))))
        return Equation(all(expected == effects(
            section.boxes, (src for src, _ in section.box_wires))
            for section in twice.sections()), True)

    @axiom
    def rewrite_identity(cls, graph: Self) -> Equation[Self]:
        """ Rewriting a rebuilt e-hypergraph into itself changes nothing. """
        rebuilt, (dom_wires, _, cod_wires) = graph.rebuild(), graph.graph.wires
        return Equation(rebuilt.rewrite(
            tuple(range(len(graph.graph.boxes))),
            tuple(rebuilt.classes[s] for s in dom_wires), cod_wires,
            graph.graph), rebuilt)

    @axiom
    def extraction_rewrite[A: Obj[Ty], B: Obj[Ty]](
            cls, f: Hom[Self, A, B], g: Hom[Self, A, B]) -> Equation[bool]:
        """ Rewriting ``f`` into what ``g`` extracts costs no more. """
        rewritten = f.rewrite(
            tuple(range(len(f.graph.boxes))), f.graph.dom_wires,
            f.graph.cod_wires, g.extract())
        return Equation(type(f)(rewritten.extract()).cost() <= min(
            f.cost(), g.cost()), True)

    @axiom
    def matching_completeness[A: Obj[Ty], B: Obj[Ty], C: Obj[Ty]](
            cls, f: Hom[Self, A, B], g: Hom[Self, B, C]) -> Equation[bool]:
        """ Binding the boxes of ``g`` to two copies of ``f >> g`` in turn is
        a match of its pattern: matching is complete modulo the classes. """
        composite, lhs = f >> g, pattern(g.graph)
        n, m, twice = len(composite.graph.boxes), len(f.graph.boxes), \
            composite.twice()
        region = [m + j + n * (j % 2) for j in range(len(lhs.boxes))]
        image = {s: twice.classes[t] for j, (src, tgt) in zip(
            region, lhs.box_wires) for s, t in zip(
            src + tgt, sum(twice.graph.box_wires[j], ()))}
        match = tuple(sorted(region)), tuple(map(
            image.get, lhs.dom_wires)), tuple(map(image.get, lhs.cod_wires))
        return Equation(match in twice.matches(lhs), True)

    @axiom
    def matching_soundness[A: Obj[Ty], B: Obj[Ty], C: Obj[Ty], I: Count,
                           J: Count](
            cls, f: Hom[Self, A, B], g: Hom[Self, B, C], i: Var[int, I],
            j: Var[int, J]) -> Equation[bool]:
        """ Matching is sound, flags included: in ``(f >> c) @ g``, ``c`` the
        boxes of ``g`` made cartesian, rewriting a match into its pattern
        changes nothing, for the pattern of ``c`` and for it rewired. """
        graph, factory = g.graph, cls.category.box_factory
        graph = type(graph)(graph.dom, graph.cod, tuple(factory(
            box.name, box.dom, box.cod) for box in graph.boxes), graph.wires)
        composite = ((f >> type(g)(graph)) @ g).rebuild()
        lhs = pattern(graph)
        return Equation(all(
            composite.rewrite(*match, lhs) == composite
            for lhs in (lhs, rewire(lhs, i, j))
            for match in composite.matches(lhs)), True)

    @axiom
    def saturation(cls, graph: Self) -> Equation[bool]:
        """ A round after saturation changes nothing, under the rules equating
        the boxes of a type, then also each box with a fresh one. """
        boxes, fresh = dict.fromkeys(graph.graph.boxes), GENERATORS[2]
        same = [(b, c) for b in boxes for c in boxes
                if (b.dom, b.cod) == (c.dom, c.cod)]
        renamed = [(b, cls.category.box_factory(fresh, b.dom, b.cod))
                   for b in boxes]

        def saturated(rules):
            *_, last = graph.rebuild(), *graph.saturate(*rules)
            return last == last.rebuild() and all(
                last.rewrite(*match, rhs.to_hypergraph()) == last
                for lhs, rhs in rules
                for match in last.matches(lhs.to_hypergraph()))

        return Equation(saturated(same), saturated(same + renamed), True)

    rewrite_identity = rewrite_identity.weaken(effects=False)
    extraction_rewrite = extraction_rewrite.weaken(effects=False)
    matching_completeness = matching_completeness.weaken(effects=False)
    saturation = saturation.weaken(effects=False)


Markov = EHypergraph[markov.Diagram]
