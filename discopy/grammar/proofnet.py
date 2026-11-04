# -*- coding: utf-8 -*-

"""
Proof nets for abstract categorial grammars, i.e. parses as axiom linkings.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Sequent
    ProofNet
    Lazy

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    ports
    count
    hungarian
    assignments
    products

Ports and links
---------------

A sentence to parse is a :class:`Sequent` ``w1 : A1, ..., wn : An |- G``,
words with their types and a goal, and a parse is a proof of it. A proof is
a :class:`ProofNet` (Girard, `Linear logic (1987)
<https://doi.org/10.1016/0304-3975(87)90045-4>`_): every atom occurring in a
type is a port, with a polarity saying whether it is given or wanted, and an
axiom link pairs every wanted port with a given port of the same atom. The
type ``x >> y`` wants an ``x`` to give a ``y``, so its ports are those of
``x`` with the opposite polarity followed by those of ``y``, i.e. the
pregroup type ``x.r @ y`` of the compact translation, and the goal is
wanted. A linking is a matching, one square block of positive and negative
ports per atom, so the words can only be linked when every atom is given as
many times as it is wanted.

The axiom linking of a term, :meth:`ProofNet.from_term`, is the geometry of
interaction: constants are ports, application links the exponent of the
function to its argument, and a variable is a wire from its abstraction to
its occurrence, which links are read through.

Sequentialisation
-----------------

Not every linking is the net of a term. :meth:`ProofNet.to_term` reads the
term back, following the links from the goal: a wanted type ``x >> y`` is an
abstraction over a variable of type ``x``, a given type is the word or the
variable whose type it is, applied to what its exponents want. The reading
is the correctness criterion: it fails exactly when a variable is used out of
the scope of its abstraction or not at all, or when a word cannot be reached
from the goal, i.e. when some switching of the net is cyclic or disconnected
(Danos and Regnier, `The structure of multiplicatives (1989)
<https://doi.org/10.1007/BF01622878>`_).
What it reads is the beta-normal eta-long form, so that two linear terms
have the same net if and only if they are equal up to beta and eta.

>>> n, np, s = Ty("n"), Ty("np"), Ty("s")
>>> every, unicorn = (n >> ((np >> s) >> s))("every"), n("unicorn")
>>> sleeps = (np >> s)("sleeps")
>>> net = ProofNet.from_term(every(unicorn)(sleeps))
>>> print(net.to_term())
every(unicorn)(np(lambda x1: sleeps(x1)))

Decoding
--------

A parser scores how likely each wanted port is to link to each given port,
and :meth:`Sequent.decode` takes the best linking, the maximum assignment of
each block, which the Hungarian algorithm finds (Kuhn, `The Hungarian method
for the assignment problem (1955) <https://doi.org/10.1002/nav.3800020109>`_).

>>> Alice, loves, Bob = n("Alice"), (n >> (n >> s))("loves"), n("Bob")
>>> sequent = Sequent((Alice, loves, Bob), s)
>>> for i, port in enumerate(sequent.ports):
...     print(i, *port)
0 n True
1 n False
2 n False
3 s True
4 n True
5 s False
>>> scores = [[0, 0, 0, 0, 0, 0],
...           [1, 0, 0, 0, 2, 0],
...           [2, 0, 0, 0, 1, 0],
...           [0, 0, 0, 0, 0, 0],
...           [0, 0, 0, 0, 0, 0],
...           [0, 0, 0, 1, 0, 0]]
>>> net = sequent.decode(scores)
>>> net.links
((1, 4), (2, 0), (5, 3))
>>> print(net.to_term())
loves(Bob)(Alice)

Ranking
-------

The best linking need not be correct. :meth:`Sequent.linkings` ranks the
linkings best first, the :func:`products` of the rankings of each block by
:func:`assignments`, Murty's ranking of the assignments of a matrix, each
block ranked only as far as the product reads it.

>>> for net in sequent.linkings(scores, 2):
...     print(net.to_term())
loves(Bob)(Alice)
loves(Alice)(Bob)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property, reduce
import heapq
import itertools
import operator

import numpy as np

from discopy.grammar.abstract import (
    Ty, Constant, Variable, Application, Abstraction, Term)
from discopy.utils import AxiomError

Port = tuple[Ty, bool]


def ports(ty: Ty, polarity: bool = True) -> list[Port]:
    """
    The atoms of a type with their polarity, given when ``True``.

    Parameters:
        ty : An implicative type.
        polarity : Whether the type is given rather than wanted.

    Example
    -------
    >>> x, y, z = map(Ty, "xyz")
    >>> for atom, polarity in ports((x >> y) >> z):
    ...     print(atom, polarity)
    x True
    y False
    z True
    """
    if ty.is_exp:
        return ports(ty.exponent, not polarity) + ports(ty.base, polarity)
    return [(ty, polarity)]


def count(ty: Ty, polarity: bool = True) -> Counter:
    """
    How many more times each atom of a type is given than wanted, the count
    invariant of van Benthem, `The semantics of variety in categorial
    grammar (1988) <https://doi.org/10.1075/lis.25.04ben>`_: the words of a
    sequent can only be linked into a proof of its goal when they have its
    count.

    Parameters:
        ty : An implicative type.
        polarity : Whether the type is given rather than wanted.

    Example
    -------
    >>> np, s = Ty("np"), Ty("s")
    >>> assert count(np >> (np >> s)) == {s: 1, np: -2}
    """
    result = Counter()
    for atom, given in ports(ty, polarity):
        result[atom] += 1 if given else -1
    return result


def hungarian(scores) -> list[int]:
    """
    The assignment of maximum total score, i.e. the column of each row, in
    cubic time by shortest augmenting paths over dual potentials.

    Parameters:
        scores : A square matrix.

    Example
    -------
    >>> hungarian([[1, 2], [3, 5]])
    [0, 1]
    """
    n = len(scores)
    cost = np.pad(-np.asarray(scores, dtype=float).reshape(n, n), (1, 0))
    row, way = np.zeros(n + 1, dtype=int), np.zeros(n + 1, dtype=int)
    potential, dual = np.zeros(n + 1), np.zeros(n + 1)
    for i in range(1, n + 1):
        row[0], column = i, 0
        slack, used = np.full(n + 1, np.inf), np.zeros(n + 1, dtype=bool)
        while row[column]:
            used[column] = True
            reduced = cost[row[column]] - potential[row[column]] - dual
            better = ~used & (reduced < slack)
            better[0] = False
            slack[better], way[better] = reduced[better], column
            free = np.where(used, np.inf, slack)
            column = int(np.argmin(free))
            potential[row[used]] += free[column]
            dual[used] -= free[column]
            slack[~used] -= free[column]
        while column:
            row[column], column = row[way[column]], way[column]
    return [int(j) for j in np.argsort(row[1:])]


def assignments(scores, m: int) -> Iterator[tuple[list[int], float]]:
    """
    The ``m`` best assignments of a square matrix with their total score,
    best first: Murty's ranking, which splits the assignments left after
    each best one by the first row where they differ from it, and solves
    each part with :func:`hungarian` on the scores masked to it (Murty, `An
    algorithm for ranking all the assignments in order of increasing cost
    (1968) <https://doi.org/10.1287/opre.16.3.682>`_). Each assignment is
    computed when it is read, and none after the ``m``-th.

    A score can be ``-inf``, e.g. the log-probability of an impossible
    link: :func:`hungarian` is given a finite score instead, low enough that
    the assignments through it come after all the others, and the masked
    scores lower still.

    Parameters:
        scores : A square matrix of reals or ``-inf``.
        m : The number of assignments.

    Example
    -------
    >>> for columns, total in assignments([[3, 1], [2, 1]], 3):
    ...     print(columns, total)
    [0, 1] 4.0
    [1, 0] 3.0
    >>> for columns, total in assignments([[0, -np.inf], [-np.inf, 0]], 2):
    ...     print(columns, total)
    [0, 1] 0.0
    [1, 0] -inf
    """
    n = len(scores)
    scores = np.asarray(scores, dtype=float).reshape(n, n)

    def below(matrix):
        real = np.isfinite(matrix)
        least = matrix.min(initial=0, where=real)
        return least - n * (matrix.max(initial=0, where=real) - least) - 1

    finite = np.where(np.isneginf(scores), below(scores), scores)
    low, heap, ties = below(finite), [], itertools.count()

    def solve(forced, forbidden):
        allowed = np.ones((n, n), dtype=bool)
        for i, j in forced:
            allowed[i], allowed[:, j] = False, False
            allowed[i, j] = True
        for i, j in forbidden:
            allowed[i, j] = False
        columns = hungarian(np.where(allowed, finite, low))
        if allowed[range(n), columns].all():
            total = float(scores[range(n), columns].sum())
            heapq.heappush(
                heap, (-total, next(ties), forced, forbidden, columns))

    if m:
        solve((), ())
    while heap:
        total, _, forced, forbidden, columns = heapq.heappop(heap)
        yield columns, -total
        m -= 1
        if not m:
            return
        for i in sorted(set(range(n)).difference(dict(forced))):
            solve(forced, forbidden + ((i, columns[i]), ))
            forced += ((i, columns[i]), )


@dataclass
class Lazy:
    """
    A list read from an iterator only as far as it is indexed, e.g. a ranking
    computed on demand for :func:`products`. It is indexed from its start
    only, since the iterator may have no end to index from.

    Parameters:
        iterator : The items, in order.
        items : The items read so far, none by default.

    Raises:
        IndexError : When the index is negative or past the last item.

    Example
    -------
    >>> squares = Lazy(i * i for i in itertools.count())
    >>> squares[3], squares.items
    (9, [0, 1, 4, 9])
    """
    iterator: Iterator
    items: list = field(default_factory=list)

    def __post_init__(self):
        self.iterator = iter(self.iterator)

    def __getitem__(self, index: int):
        if operator.index(index) < 0:
            raise IndexError(f"{index}: a lazy list is read from its start.")
        self.items += itertools.islice(
            self.iterator, max(index + 1 - len(self.items), 0))
        return self.items[index]


def products(beams, width: int) -> Iterator[tuple[tuple[int, ...], float]]:
    """
    The ``width`` best choices of one item in each of some lists, best first
    by the sum of their scores: the lazy enumeration of the product of lists
    sorted by score, each choice followed by the choices one step down in
    one list (Huang and Chiang, `Better k-best parsing (2005)
    <https://aclanthology.org/W05-1506/>`_). The lists are read only as far
    as the choices so far need, and no further after the last one, so that
    each can be a :class:`Lazy` ranking.

    Parameters:
        beams : Lists of pairs of an item and its score, by decreasing score.
        width : The number of choices.

    Example
    -------
    >>> beams = [[("a", 0.), ("b", -1.)], [("x", 0.), ("y", -.5)]]
    >>> for choice, score in products(beams, 3):
    ...     print(choice, score)
    (0, 0) 0.0
    (0, 1) -0.5
    (1, 0) -1.0
    """
    if not width:
        return
    start = (0, ) * len(beams)
    try:
        heap, seen = [(-sum(beam[0][1] for beam in beams), start)], {start}
    except IndexError:
        return
    while heap:
        score, choice = heapq.heappop(heap)
        yield choice, -score
        width -= 1
        if not width:
            return
        for i, j in enumerate(choice):
            successor = choice[:i] + (j + 1, ) + choice[i + 1:]
            try:
                following = beams[i][j + 1][1]
            except IndexError:
                continue
            if successor not in seen:
                seen.add(successor)
                heapq.heappush(heap, (
                    score + beams[i][j][1] - following, successor))


@dataclass(frozen=True)
class Sequent:
    """
    A sequent ``w1 : A1, ..., wn : An |- G``, words with their types to be
    linked into a proof of the goal ``G``.

    Parameters:
        words : The constants of the sentence, in order.
        goal : The type of the sentence.
    """
    words: tuple[Constant, ...]
    goal: Ty

    def __post_init__(self):
        object.__setattr__(self, "words", tuple(self.words))

    @cached_property
    def ports(self) -> list[Port]:
        "The ports of the words, given, then those of the goal, wanted."
        given = [
            ports(word.cod)  # ty: ignore[invalid-argument-type]
            for word in self.words]
        return sum(given, []) + ports(self.goal, False)

    @cached_property
    def offsets(self) -> list[int]:
        "The index of the first port of each word, then of the goal."
        return list(itertools.accumulate(
            [len(ports(word.cod))  # ty: ignore[invalid-argument-type]
             for word in self.words], initial=0))

    @cached_property
    def blocks(self) -> dict[Ty, tuple[list[int], list[int]]]:
        "The negative and positive ports of each atom, in order."
        result = defaultdict(lambda: ([], []))
        for i, (atom, polarity) in enumerate(self.ports):
            result[atom][polarity].append(i)
        return dict(result)

    @property
    def is_balanced(self) -> bool:
        """
        Whether every atom is given as often as it is wanted, i.e. the words
        have the :func:`count` of the goal, without which there is no linking.

        Example
        -------
        >>> n, s = Ty("n"), Ty("s")
        >>> Alice, loves = n("Alice"), (n >> (n >> s))("loves")
        >>> assert Sequent((Alice, loves), n >> s).is_balanced
        >>> assert not Sequent((Alice, loves), s).is_balanced
        """
        return all(len(negatives) == len(positives)
                   for negatives, positives in self.blocks.values())

    @classmethod
    def infer(cls, words) -> Sequent:
        """
        The sequent of some words with the smallest goal of their
        :func:`count`: the atom they give once more than they want, or
        ``x >> y`` when they also want an ``x`` once more than they give it.

        Parameters:
            words : The constants of the sentence, in order.

        Raises:
            AxiomError : When the count of the words is that of no such goal.

        Example
        -------
        >>> n, s = Ty("n"), Ty("s")
        >>> Alice, loves = n("Alice"), (n >> (n >> s))("loves")
        >>> print(Sequent.infer((Alice, loves)).goal)
        (n >> s)
        """
        words, total = tuple(words), Counter()
        for word in words:
            total.update(count(word.cod))
        given, wanted = ([atom for atom, n in total.items()
                          for _ in range(max(sign * n, 0))]
                         for sign in (1, -1))
        if len(given) != 1 or len(wanted) > 1:
            raise AxiomError(
                f"No smallest goal gives {given} and wants {wanted}.")
        return cls(words, given[0] if not wanted else wanted[0] >> given[0])

    def decode(self, scores) -> ProofNet:
        """
        The linking of maximum total score, the :func:`hungarian` assignment
        of each block, i.e. the first of the :meth:`linkings`.

        Parameters:
            scores : A square matrix over the ports, with a row for each
                negative port and a column for each positive port.

        Raises:
            AxiomError : When an atom is not given as often as it is wanted.
        """
        return next(self.linkings(scores, 1))

    def linkings(self, scores, m: int) -> Iterator[ProofNet]:
        """
        The ``m`` best linkings, best first by their total score: the
        :func:`products` of the :func:`assignments` of each block, each block
        ranked only as far as the product reads it, so that the best linking
        comes before any block is ranked further than its best assignment.

        Parameters:
            scores : A square matrix over the ports, with a row for each
                negative port and a column for each positive port.
            m : The number of linkings.

        Raises:
            AxiomError : When an atom is not given as often as it is wanted.

        Example
        -------
        >>> a = Ty("a")
        >>> f, x = (a >> a)("f"), a("x")
        >>> scores = [[0, 2, 1, 0], [0] * 4, [0] * 4, [0, 1, 2, 0]]
        >>> for net in Sequent((f, x), a).linkings(scores, 3):
        ...     print(net.links, net.is_correct)
        ((0, 1), (3, 2)) False
        ((0, 2), (3, 1)) True
        """
        scores, blocks = np.asarray(scores, dtype=float), []
        for atom, (negatives, positives) in self.blocks.items():
            if len(negatives) != len(positives):
                raise AxiomError(
                    f"{atom} is wanted {len(negatives)} times "
                    f"and given {len(positives)} times.")
            block = scores[np.ix_(negatives, positives)]
            blocks.append((negatives, positives, Lazy(assignments(block, m))))
        for choice, _ in products([ranking for *_, ranking in blocks], m):
            yield ProofNet(self, [
                (negatives[i], positives[j])
                for (negatives, positives, ranking), c in zip(blocks, choice)
                for i, j in enumerate(ranking[c][0])])


@dataclass(frozen=True)
class ProofNet:
    """
    A proof net is a sequent with an axiom linking, a matching of each
    negative port with a positive port of the same atom.

    Parameters:
        sequent : The words and the goal.
        links : The pairs of a negative and a positive port.

    Raises:
        AxiomError : When the links are not such a matching.
    """
    sequent: Sequent
    links: tuple[tuple[int, int], ...]

    def __post_init__(self):
        object.__setattr__(self, "links", tuple(sorted(
            (int(negative), int(positive))
            for negative, positive in self.links)))
        ports_ = self.sequent.ports
        if sorted(sum(self.links, ())) != list(range(len(ports_))):
            raise AxiomError("Every port must be linked exactly once.")
        for negative, positive in self.links:
            if ports_[negative][1] or not ports_[positive][1]\
                    or ports_[negative][0] != ports_[positive][0]:
                raise AxiomError(
                    f"Cannot link port {negative} to port {positive}.")

    @classmethod
    def from_term(cls, term: Term, words=None) -> "ProofNet":
        """
        The axiom linking of a closed linear term, read through the wires of
        its variables.

        Parameters:
            term : The term of the sentence.
            words : Its constants in the order of the sentence, the order in
                which the term uses them by default. An occurrence takes the
                first equal word that no earlier occurrence took.

        Raises:
            AxiomError : When the term is not closed and linear, or when its
                constants are not the words.

        Example
        -------
        >>> n, s = Ty("n"), Ty("s")
        >>> Alice, loves, Bob = n("Alice"), (n >> (n >> s))("loves"), n("Bob")
        >>> net = ProofNet.from_term(loves(Bob)(Alice), (Alice, loves, Bob))
        >>> net.links
        ((1, 4), (2, 0), (5, 3))
        """
        if term.freevars or not term.is_linear:
            raise AxiomError(f"{term} is not a closed linear term.")
        constants, edges, wires = [], [], itertools.count()

        def visit(term, scope):
            if isinstance(term, Abstraction):
                bound = [("wire", next(wires)) for _ in ports(term.var.cod)]
                return bound + visit(term.body, scope | {term.var: bound})
            if isinstance(term, Application):
                func, args = visit(term.func, scope), visit(term.args, scope)
                edges.extend(zip(func, args))
                return func[len(args):]
            if isinstance(term, Variable):
                return scope[term]
            constants.append(term)
            return [("port", len(constants) - 1, i)
                    for i in range(len(ports(term.cod)))]

        edges.extend(zip(
            visit(term, {}), (("goal", i) for i in itertools.count())))
        given = constants if words is None else words
        sequent = Sequent(
            given,  # ty: ignore[invalid-argument-type]
            term.cod)  # ty: ignore[invalid-argument-type]
        unused, position = list(range(len(sequent.words))), []
        for constant in constants:
            match = [i for i in unused if sequent.words[i] == constant]
            if not match:
                raise AxiomError(f"{constant} is not one of the words.")
            unused.remove(match[0])
            position.append(match[0])
        if unused:
            raise AxiomError(f"{sequent.words[unused[0]]} is not used.")

        neighbours = defaultdict(list)
        for left, right in edges:
            neighbours[left].append(right)
            neighbours[right].append(left)

        def index(end):
            if end[0] == "goal":
                return sequent.offsets[-1] + end[1]
            return sequent.offsets[position[end[1]]] + end[2]

        def follow(previous, end):
            while end[0] == "wire":
                previous, end = end, next(
                    other for other in neighbours[end] if other != previous)
            return index(end)

        links = [(index(end), follow(end, neighbours[end][0]))
                 for end in neighbours if end[0] != "wire"]
        kept = [(i, j) for i, j in links if sequent.ports[j][1]]
        return cls(sequent, kept)  # ty: ignore[invalid-argument-type]

    @property
    def is_correct(self) -> bool:
        """
        Whether the net is that of a proof, i.e. :meth:`to_term` reads it.

        Example
        -------
        >>> a = Ty("a")
        >>> f, x = (a >> a)("f"), a("x")
        >>> sequent = Sequent((f, x), a)
        >>> assert ProofNet(sequent, [(0, 2), (3, 1)]).is_correct
        >>> assert not ProofNet(sequent, [(0, 1), (3, 2)]).is_correct
        """
        try:
            self.to_term()
        except AxiomError:
            return False
        return True

    def to_term(self) -> Term:
        """
        The beta-normal eta-long term of the net, read from the goal.

        Raises:
            AxiomError : When a variable is used out of scope or a word
                is not connected to the goal, i.e. the net is not correct.

        Example
        -------
        >>> a, b, c = Ty("a"), Ty("b"), Ty("c")
        >>> g, u = (((a >> a) >> b) >> c)("g"), b("u")
        >>> sequent = Sequent((g, u), c)
        >>> print(ProofNet(sequent, [(0, 1), (2, 4), (5, 3)]).to_term())
        Traceback (most recent call last):
        ...
        discopy.utils.AxiomError: x1 is not used in its scope.
        >>> f, x = (a >> a)("f"), a("x")
        >>> ProofNet(Sequent((f, x), a), [(0, 1), (3, 2)]).to_term()
        Traceback (most recent call last):
        ...
        discopy.utils.AxiomError: f is not connected to the goal.
        """
        partner, heads, names = dict(self.links), {}, itertools.count(1)
        read = set()

        def spine(ty, offset):
            exponents = []
            while ty.is_exp:
                exponents.append((ty.exponent, offset))
                offset += len(ports(ty.exponent))
                ty = ty.base
            return exponents, offset

        def given(port):
            if port not in heads:
                raise AxiomError(f"Port {port} is used out of scope.")
            read.add(port)
            head, exponents = heads[port]
            return reduce(
                lambda func, exponent: func(wanted(*exponent)),
                exponents, head)

        def wanted(ty, offset):
            exponents, port = spine(ty, offset)
            variables = [Variable(f"x{next(names)}", exponent)
                         for exponent, _ in exponents]
            bound = [spine(*exponent) for exponent in exponents]
            for variable, (arguments, head) in zip(variables, bound):
                heads[head] = variable, arguments
            body = given(partner[port])
            for variable, (_, head) in zip(variables, bound):
                if head not in read:
                    raise AxiomError(f"{variable} is not used in its scope.")
                del heads[head]
            return reduce(
                lambda body, variable: Abstraction(variable, body),
                reversed(variables), body)

        for word, offset in zip(self.sequent.words, self.sequent.offsets):
            exponents, head = spine(word.cod, offset)
            heads[head] = word, exponents
        term = wanted(self.sequent.goal, self.sequent.offsets[-1])
        for word, offset in zip(self.sequent.words, self.sequent.offsets):
            if spine(word.cod, offset)[1] not in read:
                raise AxiomError(f"{word} is not connected to the goal.")
        return term
