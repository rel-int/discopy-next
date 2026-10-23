"""
egglog as a reference for e-graphs of terms, over nested tuples.

A term is one of::

    ("var", name)            a variable, i.e. an input of the diagram, cost 0
    ("const", n)             an integer constant, a box with no input, cost 1
    (op, t0, ..., tk)        an operation, a box with k + 1 inputs, cost 1,
                             but 2 for ``*``

with ``op`` one of ``+ - * / <<`` or any other identifier, of arity one to
three. A rule is a pair ``(lhs, rhs)`` of terms whose variables are its
pattern variables, e.g. ``(("*", X, ("const", 2)), ("<<", X, ("const", 1)))``,
its left-hand side not a variable.

The cost of a term is the design's tree cost: ``t(var) = 0``,
``t(const) = 1`` and ``t(op(ts)) = c(op) + sum(t(ts))``, a shared subterm
counted once per occurrence, computed natively by egglog: a variable is a
nullary function of cost 0, a constant a constructor of cost 0 whose ``i64``
literal costs 1, an operation a method or a function of cost ``c(op)``, one
but for ``*``, which costs two, so that the costs are not egg's ``AstSize``.

The schedule is egg's: one round searches every rule on the e-graph as it
was when the round began, applies every match, then rebuilds; there is no
match limit nor backoff, and a round that changes nothing is a fixpoint.

Egg's example, from the egg and egglog documentation:

>>> term = ("/", ("*", ("var", "a"), C2), C2)
>>> egglog_extract(term, EGG)
(('var', 'a'), 0)
>>> for state in egglog_trace(term, EGG):
...     print(state.round, state.updated, state.matches, state.cost,
...           state.n_classes, state.n_nodes)
0 True () 5 4 4
1 True (1, 1, 0, 0) 4 6 8
2 True (1, 1, 1, 0) 3 5 8
3 True (1, 1, 1, 1) 0 4 8
4 False (1, 1, 1, 1) 0 4 8
>>> for members in egglog_classes(term, EGG):
...     print(members)
(('*', ('var', 'a'), ('const', 2)),)
(('/', ('*', ('var', 'a'), ('const', 2)), ('const', 2)), ('var', 'a'))
(('const', 2),)
>>> egglog_equal(term, ("var", "a"), EGG), egglog_equal(term, C2, EGG)
(True, False)

Sampling: a naive ``sample`` is a function, hash-consed, so that two
samples are one and ``x - x = 0`` wrongly fires; giving each sample a
variable of its own as an id, i.e. an input that nothing rewrites and that
costs 0, keeps them apart, while one sample used twice still cancels.

>>> P, ZERO = ("var", "p"), [(("-", X, X), C0)]
>>> egglog_extract(("-", ("sample", P), ("sample", P)), ZERO)
(('const', 0), 1)
>>> two_samples = ("-", ("sample", P, ("var", "id0")),
...                     ("sample", P, ("var", "id1")))
>>> egglog_extract(two_samples, ZERO)[1]
3
>>> one_sample = ("sample", P, ("var", "id0"))
>>> egglog_extract(("-", one_sample, one_sample), ZERO)
(('const', 0), 1)

Conditions: a program is its value, or ``("observe", c, program)``, so that
extraction keeps every condition, the cost of a condition being the
design's cost of a root, ``1 + t(c)``.

>>> s = ("sample", P, ("var", "id0"))
>>> program = ("observe", s, ("-", ("/", ("*", s, C2), C2), s))
>>> egglog_extract(program, EGG + ZERO)
(('observe', ('sample', ('var', 'p'), ('var', 'id0')), ('const', 0)), 3)

A condition registered beside the value of a program, rather than in it,
is dropped by extraction, which starts from the value:

>>> s, absorb = ("sample", P), [(("*", C0, X), C0)]
>>> egglog_extract(("*", C0, s), absorb, extra=[("condition", s)])
(('const', 0), 1)
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from functools import cache

import pytest

pytest.importorskip("egglog")

from egglog import (  # noqa: E402
    EGraph, Expr, i64Like, method, function, rewrite, ruleset, var)
from egglog.declarations import (  # noqa: E402
    CallDecl, FunctionRef, InitRef, LitDecl, MethodRef)
from egglog.egraph import expr_parts  # noqa: E402

__all__ = [
    "X", "Y", "Z", "C0", "C1", "C2", "COSTS",
    "EGG", "COMMUTATIVITY", "IDENTITIES", "RULESETS",
    "cost", "subterms", "random_term",
    "Round", "egglog_trace", "egglog_classes", "egglog_extract",
    "egglog_equal",
]


class Term(Expr):
    """The sort of terms, its binary operators the methods below."""

    @method(cost=0)
    def __init__(self, value: i64Like) -> None: ...
    def __add__(self, other: Term) -> Term: ...
    def __sub__(self, other: Term) -> Term: ...
    @method(cost=2)
    def __mul__(self, other: Term) -> Term: ...
    def __truediv__(self, other: Term) -> Term: ...
    def __lshift__(self, other: Term) -> Term: ...


BINARY = {
    "+": "__add__", "-": "__sub__", "*": "__mul__",
    "/": "__truediv__", "<<": "__lshift__"}
METHOD = {value: key for key, value in BINARY.items()}


def named(name, head_cost, stub):
    """
    An egglog function named ``name``: egglog identifies a function by its
    Python name and module, so ``stub`` is renamed before it is declared.
    """
    stub.__name__ = stub.__qualname__ = name
    return function(cost=head_cost, egg_fn=name)(stub)


@cache
def variable(name):
    """The nullary function ``var_<name>``, of cost 0."""
    if not name.isidentifier():
        raise ValueError(f"variable name {name!r} is not an identifier")

    def stub() -> Term: ...
    return named(f"var_{name}", 0, stub)


@cache
def operation(name, arity):
    """The function ``op_<name>_<arity>``, of cost 1."""
    if not name.isidentifier():
        raise ValueError(f"operation name {name!r} is not an identifier")
    if arity == 1:
        def stub(a: Term) -> Term: ...
    elif arity == 2:
        def stub(a: Term, b: Term) -> Term: ...
    elif arity == 3:
        def stub(a: Term, b: Term, c: Term) -> Term: ...
    else:
        raise ValueError(f"arity {arity} of {name!r} is not one to three")
    return named(f"op_{name}_{arity}", 1, stub)


def to_egglog(term, pattern=False):
    """
    The egglog expression of a term, its variables egglog's pattern variables
    when ``pattern``.
    """
    kind, *args = term
    if kind == "var":
        name, = args
        return var(name, Term) if pattern else variable(name)()
    if kind == "const":
        value, = args
        return Term(value)
    children = [to_egglog(arg, pattern) for arg in args]
    if kind in BINARY:
        if len(children) != 2:
            raise ValueError(f"{kind!r} is binary, got {term!r}")
        left, right = children
        return getattr(left, BINARY[kind])(right)
    return operation(kind, len(children))(*children)


def from_egglog(expr):
    """The term of an extracted egglog expression."""
    return from_decl(expr_parts(expr).expr)


def from_decl(decl):
    """The term of the declaration of an egglog expression."""
    if not isinstance(decl, CallDecl):
        raise TypeError(f"not a call: {decl!r}")
    ref, args = decl.callable, [arg.expr for arg in decl.args]
    if isinstance(ref, InitRef):
        literal, = args
        assert isinstance(literal, LitDecl)
        return ("const", literal.value)
    if isinstance(ref, MethodRef):
        return (METHOD[ref.method_name], *map(from_decl, args))
    if isinstance(ref, FunctionRef) and ref.ident.name.startswith("var_"):
        return ("var", ref.ident.name[len("var_"):])
    if isinstance(ref, FunctionRef) and ref.ident.name.startswith("op_"):
        op, _ = ref.ident.name[len("op_"):].rsplit("_", 1)
        return (op, *map(from_decl, args))
    raise TypeError(f"unknown callable {ref!r}")


COSTS = {"*": 2}


def cost(term):
    """The design's tree cost of a term, as egglog computes it."""
    kind, *args = term
    if kind == "var":
        return 0
    if kind == "const":
        return 1
    return COSTS.get(kind, 1) + sum(map(cost, args))


def subterms(term):
    """The distinct subterms of a term, children before parents."""
    seen = {}

    def visit(term):
        if term[0] not in ("var", "const"):
            for arg in term[1:]:
                visit(arg)
        seen.setdefault(term, None)
    visit(term)
    return list(seen)


def to_ruleset(rules):
    """
    An egglog ruleset of ``(lhs, rhs)`` pairs of terms, and the declaration
    of each rule, which keys the match counts of a run report.
    """
    rewrites = []
    for lhs, rhs in rules:
        if lhs[0] == "var":
            raise ValueError(f"the left-hand side {lhs!r} is a variable")
        rewrites.append(
            rewrite(to_egglog(lhs, True)).to(to_egglog(rhs, True)))
    return ruleset(*rewrites), [r.decl for r in rewrites]


@dataclass(frozen=True)
class Round:
    """
    The state after a round: ``round`` 0 is the term before any rule,
    ``updated`` whether the round changed the e-graph, ``True`` for round 0,
    and ``seconds`` how long its run took.

    ``matches`` counts the matches of each rule that round on the e-graph as
    the round began, a match being a class and a substitution of the pattern
    variables of the left-hand side, old matches included: the e-graph is
    evaluated naively, ``seminaive=False``, for that. ``classes`` partitions
    the subterms of the input, each class sorted by ``repr`` and the classes
    too, ``extracted`` is a term of least ``cost`` in the class of the root,
    and ``n_classes`` and ``n_nodes`` count every e-class and every e-node,
    the e-nodes hash-consed, i.e. modulo congruence.
    """
    round: int
    updated: bool
    matches: tuple[int, ...]
    classes: tuple[tuple, ...]
    extracted: tuple
    cost: int
    n_classes: int
    n_nodes: int
    seconds: float


def state(egraph, root, term, index, updated, matches, seconds):
    """The ``Round`` of an e-graph ``egraph`` of ``term`` at ``root``."""
    by_class = {}
    for sub in subterms(term):
        found = egraph.lookup_function_value(to_egglog(sub))
        by_class.setdefault(expr_parts(found).expr, []).append(sub)
    classes = tuple(sorted(
        (tuple(sorted(members, key=repr)) for members in by_class.values()),
        key=repr))
    best, best_cost = egraph.extract(root, include_cost=True)
    extracted = from_egglog(best)
    assert cost(extracted) == best_cost, (extracted, best_cost)
    e_classes = egraph.freeze().decl.e_classes
    return Round(
        round=index, updated=updated, matches=matches, classes=classes,
        extracted=extracted, cost=best_cost, n_classes=len(e_classes),
        n_nodes=sum(len(nodes) for _, nodes in e_classes.values()),
        seconds=seconds)


def egglog_trace(term, rules, rounds=10, extra=()):
    """
    The state of egglog's e-graph of ``term`` before any rule, then after
    each round of ``rules``, up to the first round that changes nothing or to
    ``rounds`` rounds, so that the last state's ``updated`` is ``False``
    exactly when the rules saturated. The ``extra`` terms are registered in
    the e-graph beside ``term``, which extraction starts from.
    """
    egraph = EGraph(seminaive=False)
    root = egraph.let("root", to_egglog(term))
    for other in extra:
        egraph.register(to_egglog(other))
    schedule, decls = to_ruleset(rules)
    states = [state(egraph, root, term, 0, True, (), 0.)]
    for index in range(1, rounds + 1):
        start = time.perf_counter()
        report = egraph.run(schedule)
        seconds = time.perf_counter() - start
        matches = tuple(report.num_matches_per_rule.get(d, 0) for d in decls)
        states.append(state(
            egraph, root, term, index, report.updated, matches, seconds))
        if not report.updated:
            break
    return states


def egglog_classes(term, rules, rounds=10):
    """The partition of the subterms of ``term`` after the last round."""
    return egglog_trace(term, rules, rounds)[-1].classes


def egglog_extract(term, rules, rounds=10, extra=()):
    """A term of least cost equal to ``term`` after the rounds, its cost."""
    last = egglog_trace(term, rules, rounds, extra)[-1]
    return last.extracted, last.cost


def egglog_equal(left, right, rules, rounds=10):
    """Whether egglog proves ``left == right`` within ``rounds`` rounds."""
    for members in egglog_classes(("pair", left, right), rules, rounds):
        if left in members:
            return right in members
    raise AssertionError("unreachable")


X, Y, Z = ("var", "x"), ("var", "y"), ("var", "z")
C0, C1, C2 = ("const", 0), ("const", 1), ("const", 2)

EGG = [
    (("*", X, C2), ("<<", X, C1)),
    (("/", ("*", X, Y), Z), ("*", X, ("/", Y, Z))),
    (("/", X, X), C1),
    (("*", X, C1), X),
]
COMMUTATIVITY = [
    (("+", X, Y), ("+", Y, X)),
    (("*", X, Y), ("*", Y, X)),
]
IDENTITIES = [
    (("+", X, C0), X),
    (("*", X, C0), C0),
]
RULESETS = {
    "egg": EGG,
    "commutativity": COMMUTATIVITY,
    "identities": IDENTITIES,
    "egg+identities": EGG + IDENTITIES,
    "all": EGG + COMMUTATIVITY + IDENTITIES,
}


def random_term(rng, depth, ops=("+", "*", "/", "<<"), constants=(0, 1, 2),
                variables=("a", "b", "c"), leaf=0.3):
    """
    A random term of depth at most ``depth`` with an operation at its root,
    each subterm below it a leaf with probability ``leaf``, so that
    ``leaf=0`` gives the full binary tree of depth ``depth``.
    """
    def leaf_term():
        if rng.random() < .5:
            return ("var", rng.choice(variables))
        return ("const", rng.choice(constants))

    def grow(depth, root=False):
        if depth == 0 or (not root and rng.random() < leaf):
            return leaf_term()
        return (rng.choice(ops), grow(depth - 1), grow(depth - 1))
    return grow(depth, root=True)


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
    print(random_term(random.Random(0), 3))
