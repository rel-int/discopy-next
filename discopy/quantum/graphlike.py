# -*- coding: utf-8 -*-

"""
Graph-like ZX diagrams as hypergraphs, simplified by double-pushout rewriting.

A ZX diagram is graph-like when its spiders are all green, a red spider being
a green one with a Hadamard on each leg, and when green spiders are joined by
Hadamard edges only, see :cite:t:`DuncanEtAl20`. As a
:class:`discopy.hypergraph.Hypergraph` over :class:`discopy.quantum.zx.Diagram`
the green spiders are the spiders of the hypergraph, so that fusing two of them
is the pushout itself, each Hadamard edge is the box :data:`zx.H` and a phase
is the effect ``Z(1, 0, phase)`` on a spider, in turns as in
:class:`discopy.quantum.zx.Z`.

The rules are those of the ``full_reduce`` of :cite:t:`KissingerWetering20`:
each step is a pair of hypergraphs built for its match, the left side the
neighbourhood it rewrites and the right side what replaces it, applied in
place by a :class:`discopy.hypergraph.Rewriter`. Their left sides are
left-linear, so each match has one pushout complement, and each step is a
sound equation of the ZX calculus up to a scalar, which is dropped.

Summary
-------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    from_diagram
    from_pyzx
    to_pyzx
    size
    full_reduce

Example
-------
Two Hadamards between two spiders with a phase of an eighth of a turn
cancel, and the spiders fuse into one with a quarter-turn:

>>> from discopy.quantum.zx import Z, H
>>> diagram = Z(1, 1, .125) >> H >> H >> Z(1, 1, .125)
>>> graph = full_reduce(from_diagram(diagram))
>>> print(graph)
Z(1, 2) >> Z(1, 0, 1/4) @ Nat(1)
>>> assert size(graph) == (1, 0, 0)
"""

from __future__ import annotations

from fractions import Fraction
from functools import cache
from heapq import heappop, heappush
from itertools import combinations
from typing import Callable, NamedTuple

from discopy import hypergraph, messages
from discopy.hypergraph import Rewriter
from discopy.quantum import zx
from discopy.rigid import Nat

Hypergraph = hypergraph.Hypergraph[zx.Diagram]
qubit = Nat(1)
ZERO = Fraction(0)

Step = tuple[Hypergraph, Hypergraph, hypergraph.Match]
""" A rule for one match: its left and right sides, and the match. """


@cache
def phase_box(phase: Fraction) -> zx.Z:
    """ The phase of a spider as an effect, in turns, one box per phase. """
    return zx.Z(1, 0, phase)


def is_pauli(phase: Fraction) -> bool:
    """ Whether a phase in turns, in ``[0, 1)``, is a half-turn or none. """
    return phase.denominator <= 2


def is_clifford(phase: Fraction) -> bool:
    """ Whether a phase in turns, in ``[0, 1)``, is a quarter-turn. """
    return phase.denominator in (1, 2, 4)


class Local(NamedTuple):
    """
    The neighbourhood of a spider: its phase boxes and their sum, its
    Hadamard boxes to each other spider and to itself, its boundary ports
    and whether it has any other box, in which case no rule applies there.
    """
    phases: list[int]
    phase: Fraction
    edges: dict[int, list[int]]
    loops: list[int]
    ports: int
    other: bool

    @property
    def degree(self) -> int:
        """ The number of neighbours, a boundary port being one. """
        return self.ports + len(self.edges)


def kind(box) -> str:
    """ Whether a box is a Hadamard edge, a phase, or some other box. """
    if box is zx.H:
        return "edge"
    if isinstance(box, zx.Z) and len(box.dom) == 1 and not box.cod:
        return "phase"
    return "edge" if box == zx.H else "other"


class Simplifier(Rewriter):
    """
    A :class:`discopy.hypergraph.Rewriter` that keeps an index of its boxes
    by spider: the Hadamard edges to each other spider, the loops, the
    phases and the number of other boxes, updated with each box added,
    removed or moved by a merge, and the spiders merged so far.

    Parameters:
        graph : The hypergraph to simplify.
    """
    def __init__(self, graph: Hypergraph):
        super().__init__(graph)
        self.edges = [{} for _ in self.spider_types]
        self.loops = [[] for _ in self.spider_types]
        self.phases = [[] for _ in self.spider_types]
        self.others = [0 for _ in self.spider_types]
        self.merged = set()
        for j, box in enumerate(self.boxes):
            if kind(box) == "phase":
                self.boxes[j] = phase_box(Fraction(box.phase) % 1)
            self.index(j)

    def index(self, j: int, remove: bool = False) -> None:
        """ Add a box to the index, or remove it from there. """
        found, (dom, cod) = kind(self.boxes[j]), self.box_wires[j]
        if found == "edge" and dom[0] == cod[0]:
            groups = [self.loops[dom[0]]]
        elif found == "edge":
            groups = [self.edges[dom[0]].setdefault(cod[0], []),
                      self.edges[cod[0]].setdefault(dom[0], [])]
        elif found == "phase":
            groups = [self.phases[dom[0]]]
        else:
            for spider in set(dom + cod):
                self.others[spider] += -1 if remove else 1
            return
        for group in groups:
            if remove:
                group.remove(j)
            else:
                group.append(j)
        if found == "edge" and remove and dom[0] != cod[0]\
                and not self.edges[dom[0]][cod[0]]:
            del self.edges[dom[0]][cod[0]], self.edges[cod[0]][dom[0]]

    def add_spider(self, typ) -> int:
        for index, empty in ((self.edges, {}), (self.loops, []),
                             (self.phases, []), (self.others, 0)):
            index.append(empty)  # ty: ignore[invalid-argument-type]
        return super().add_spider(typ)

    def add_box(self, box, dom, cod, offset=None) -> int:
        j = super().add_box(box, dom, cod, offset)
        self.index(j)
        return j

    def remove_box(self, j: int) -> None:
        self.index(j, remove=True)
        super().remove_box(j)

    def merge(self, spider: int, other: int) -> int:
        if spider == other:
            return spider
        moved = list(dict.fromkeys(j for j, *_ in self.incidence[spider]))
        moved += [j for j in dict.fromkeys(
            j for j, *_ in self.incidence[other]) if j not in moved]
        for j in moved:
            self.index(j, remove=True)
        kept = super().merge(spider, other)
        for j in moved:
            self.index(j)
        self.merged.add(kept)
        return kept

    def local(self, spider: int) -> Local:
        """
        The :class:`Local` neighbourhood of a spider, read off the index: it
        shares its lists, so it holds until the next step.
        """
        phases = self.phases[spider]
        phase = self.boxes[phases[0]].phase if len(phases) == 1\
            else sum(self.boxes[j].phase for j in phases) % 1 if phases\
            else ZERO
        return Local(phases, phase, self.edges[spider], self.loops[spider],
                     self.ports[spider], bool(self.others[spider]))


def is_boundary(rewriter: Simplifier, spider: int) -> bool:
    """
    Whether a spider is a boundary behind a Hadamard edge, i.e. a phase-free
    spider with one port and one Hadamard box to another spider.
    """
    if rewriter.ports[spider] != 1 or len(rewriter.incidence[spider]) != 1:
        return False
    (j, side, _), = rewriter.incidence[spider]
    return kind(rewriter.boxes[j]) == "edge"\
        and rewriter.box_wires[j][1 - side][0] != spider


def step(rewriter: Simplifier, interface: list[int], inner: list[int],
         boxes: list[int], new: list[tuple], glue: dict | None = None) -> Step:
    """
    The rule that deletes ``boxes`` and the ``inner`` spiders, keeps the
    ``interface`` spiders and adds the ``new`` boxes, each given with its
    domain and codomain spiders: host spiders of the interface, or labels of
    new spiders. Gluing an interface spider to another, as ``glue`` says,
    merges them. Both sides are built with ``_scan=False``.
    """
    glue = glue or {}
    dom = Nat(len(interface))
    types = dict.fromkeys(interface + inner, qubit)
    left = Hypergraph(
        dom, Nat(0), tuple(rewriter.boxes[j] for j in boxes),
        (tuple(interface), tuple(rewriter.box_wires[j] for j in boxes), ()),
        types, _scan=False)
    flat = interface + [s for j in boxes for wires in rewriter.box_wires[j]
                        for s in wires]
    labels = dict(zip(left.flat_wires, flat))
    labels.update(enumerate(
        sorted(set(types) - set(flat)), len(set(flat))))
    spiders = tuple(labels[i] for i in range(left.n_spiders))
    boundary = tuple(glue.get(s, s) for s in interface)
    right_types = dict.fromkeys(boundary, qubit)
    right_types.update((s, qubit) for _, *wires in new
                       for side in wires for s in side)
    right = Hypergraph(
        dom, Nat(0), tuple(box for box, *_ in new),
        (boundary, tuple(tuple(map(tuple, wires)) for _, *wires in new), ()),
        right_types, _scan=False)
    return left, right, (spiders, tuple(boxes))


def normalize(rewriter: Simplifier, spider: int) -> Step | None:
    """
    Keep the graph simple: fuse the phases of a spider, turn a Hadamard loop
    into a half-turn, cancel parallel Hadamard edges in pairs by the Hopf
    law, and drop a spider, or a pair of them, cut off from the boundary.
    """
    here = rewriter.local(spider)
    if here.other:
        return None
    own = here.phases + here.loops
    if not here.ports and not here.edges:
        return step(rewriter, [], [spider], own, [])
    if not here.ports and len(here.edges) == 1:
        (target, hadamards), = here.edges.items()
        there = rewriter.local(target)
        if len(hadamards) == 1 and not there.ports and not there.other\
                and list(there.edges) == [spider]:
            return step(rewriter, [], [spider, target],
                        own + hadamards + there.phases + there.loops, [])
    phase = (here.phase + Fraction(len(here.loops), 2)) % 1
    parallel = {t: js for t, js in here.edges.items() if len(js) > 1}
    if not here.loops and not parallel and (
            len(here.phases) == 1 and phase or not here.phases):
        return None
    new = [(phase_box(phase), [spider], [])] if phase else []
    new += [(zx.H, [spider], [t]) for t, js in parallel.items() if len(js) % 2]
    return step(rewriter, [spider] + list(parallel), [],
                own + [j for js in parallel.values() for j in js], new)


def remove_identity(rewriter: Simplifier, spider: int) -> Step | None:
    """
    Remove a phase-free spider between two Hadamard edges, fusing its two
    neighbours, as ``H >> H`` is the identity.
    """
    here = rewriter.local(spider)
    if here.ports or here.phases or here.other or here.loops\
            or len(here.edges) != 2:
        return None
    (left, (first, )), (right, (second, )) = here.edges.items()
    return step(rewriter, [left, right], [spider], [first, second], [],
                glue={right: left})


def is_interior(rewriter: Simplifier, spider: int, here: Local) -> bool:
    """ Whether a spider is a green spider away from the boundary. """
    return not here.ports and not here.other and not here.loops\
        and not is_boundary(rewriter, spider)


def toggle(locals_: dict[int, Local],
           pairs: list[tuple[int, int]]) -> tuple[list[int], list[tuple]]:
    """
    The Hadamard edges between each pair of spiders to delete, and those to
    add where there was none: the edges are toggled.
    """
    old, new = [], []
    for a, b in pairs:
        edges = locals_[a].edges.get(b, [])
        old += edges
        if not edges:
            new.append((zx.H, [a], [b]))
    return old, new


def add_phases(locals_: dict[int, Local], shift: dict[int, Fraction]
               ) -> tuple[list[int], list[tuple]]:
    """ The phase boxes to delete and to add, shifting some phases. """
    old, new = [], []
    for spider, phase in shift.items():
        if phase % 1 == 0:
            continue
        old += locals_[spider].phases
        total = (locals_[spider].phase + phase) % 1
        if total:
            new.append((phase_box(total), [spider], []))
    return old, new


def classes(locals_: dict[int, Local], u: int, v: int,
            skip: set[int] | frozenset[int] = frozenset()
            ) -> tuple[list[int], ...]:
    """
    The neighbours of ``u`` alone, of ``v`` alone and of both, other than
    ``u``, ``v`` and the spiders to skip, in a reproducible order.
    """
    near_u = [t for t in locals_[u].edges if t not in skip | {v}]
    near_v = [t for t in locals_[v].edges if t not in skip | {u}]
    both = set(near_u) & set(near_v)
    return ([t for t in near_u if t not in both],
            [t for t in near_v if t not in both],
            [t for t in near_u if t in both])


def across(*groups: list[int]) -> list[tuple[int, int]]:
    """ The pairs of spiders in different groups. """
    return [(a, b) for i, group in enumerate(groups)
            for other in groups[i + 1:] for a in group for b in other]


def neighbourhood(rewriter: Simplifier, *spiders: int,
                  known: dict[int, Local] | None = None
                  ) -> dict[int, Local]:
    """
    The :class:`Local` neighbourhoods of spiders and their neighbours, adding
    to those already ``known``.
    """
    result = dict(known or {})
    result.update((spider, rewriter.local(spider)) for spider in spiders
                  if spider not in result)
    for spider in spiders:
        for target in result[spider].edges:
            if target not in result:
                result[target] = rewriter.local(target)
    return result


def complement(rewriter: Simplifier, spider: int) -> Step | None:
    """
    Local complementation: remove an interior spider with a quarter-turn
    phase, toggling the edges between its neighbours and subtracting its
    phase from theirs.
    """
    here = rewriter.local(spider)
    if not is_interior(rewriter, spider, here)\
            or not is_clifford(here.phase) or is_pauli(here.phase):
        return None
    locals_ = neighbourhood(rewriter, spider)
    near = list(here.edges)
    if any(locals_[t].other or is_boundary(rewriter, t) for t in near):
        return None
    old_edges, new_edges = toggle(
        locals_, list(combinations(near, 2)))
    old_phases, new_phases = add_phases(
        locals_, dict.fromkeys(near, -here.phase))
    return step(
        rewriter, near, [spider],
        here.phases + [js[0] for js in here.edges.values()] + old_phases
        + old_edges, new_phases + new_edges)


def boundaries(rewriter: Simplifier, here: Local) -> list:
    """
    The boundaries of a spider: its ports, and the boundary spiders it has a
    Hadamard edge to.
    """
    return here.ports * [None] + [
        t for t in here.edges if is_boundary(rewriter, t)]


def pivot(rewriter: Simplifier, spider: int) -> Step | None:
    """
    Pivot along a Hadamard edge between two spiders with Pauli phases and at
    most one boundary between them, which goes to the spider that stays.
    """
    here = rewriter.local(spider)
    if here.other or here.loops or not is_pauli(here.phase)\
            or is_boundary(rewriter, spider):
        return None
    bound = boundaries(rewriter, here)
    if len(bound) > 1:
        return None
    for other in here.edges:
        if other in bound:
            continue
        there = rewriter.local(other)
        if there.other or there.loops or not is_pauli(there.phase)\
                or len(bound) + len(boundaries(rewriter, there)) > 1:
            continue
        result = pivot_along(rewriter, spider, other)
        if result is not None:
            return result
    return None


def pivot_along(rewriter: Simplifier, u: int, v: int) -> Step | None:
    """ The pivot along the edge from ``u`` to ``v``, see :func:`pivot`. """
    locals_ = neighbourhood(rewriter, u, v)
    there = locals_[v]
    if there.other or there.loops or not is_pauli(there.phase)\
            or is_boundary(rewriter, v):
        return None
    bound_u = boundaries(rewriter, locals_[u])
    bound_v = boundaries(rewriter, there)
    if len(bound_u) + len(bound_v) > 1:
        return None
    if bound_v:
        u, v, bound_u, bound_v = v, u, bound_v, bound_u
    skip = {t for t in bound_u if t is not None}
    near_u, near_v, both = classes(locals_, u, v, skip)
    if any(locals_[t].other for t in near_u + near_v + both):
        return None
    a_u, a_v = locals_[u].phase, locals_[v].phase
    old_edges, new_edges = toggle(
        locals_, across(near_u, near_v, both))
    shift = dict.fromkeys(near_u, a_v)
    shift.update(dict.fromkeys(near_v, a_u))
    shift.update(dict.fromkeys(both, a_u + a_v + Fraction(1, 2)))
    old_phases, new_phases = add_phases(locals_, shift)
    interface = near_u + near_v + both
    deleted = locals_[u].phases + locals_[v].phases + [
        j for t in (u, v) for target, js in locals_[t].edges.items()
        if t == u or target != u for j in js] + old_phases + old_edges
    if not bound_u:
        return step(rewriter, interface, [u, v], deleted,
                    new_phases + new_edges)
    new = new_phases + new_edges + spider_like(a_v, near_v + both)
    if bound_u[0] is None:
        return step(rewriter, interface + [u], [v], deleted,
                    new + [(zx.H, [u], ["kept"])])
    boundary, = bound_u
    return step(rewriter, interface + [boundary], [u, v], deleted, new,
                glue={boundary: "kept"})


def spider_like(phase: Fraction, near: list[int]) -> list[tuple]:
    """
    A new spider with a phase and Hadamard edges to ``near``, standing for a
    spider that PyZX keeps: the rule deletes the old one so that it is an
    equation on its own, the old one being inside the neighbourhood the
    equation rewrites.
    """
    return [(phase_box(phase), ["kept"], [])] * bool(phase) + [
        (zx.H, ["kept"], [t]) for t in near]


def degree(rewriter: Simplifier, spider: int) -> int:
    """ The number of neighbours of a spider, see :attr:`Local.degree`. """
    return rewriter.local(spider).degree


def pivot_gadget(rewriter: Simplifier, spider: int) -> Step | None:
    """
    Pivot along an edge from a Pauli spider ``u`` to a spider ``w`` with any
    other phase, which goes to a new leaf on ``u``: ``u`` becomes the hub
    of a phase gadget.
    """
    here = rewriter.local(spider)
    if not is_interior(rewriter, spider, here) or not is_pauli(here.phase):
        return None
    locals_ = neighbourhood(rewriter, spider, known={spider: here})
    if any(locals_[t].other or locals_[t].loops or locals_[t].degree == 1
           or is_boundary(rewriter, t) for t in here.edges):
        return None
    for w in here.edges:
        if not is_pauli(locals_[w].phase) and not locals_[w].ports:
            result = pivot_gadget_along(rewriter, spider, w, locals_)
            if result is not None:
                return result
    return None


def pivot_gadget_along(rewriter: Simplifier, u: int, w: int,
                       locals_: dict[int, Local]) -> Step | None:
    """ The pivot of :func:`pivot_gadget` from ``u`` to ``w``. """
    locals_ = neighbourhood(rewriter, w, known=locals_)
    if any(locals_[t].other or is_boundary(rewriter, t)
           for t in locals_[w].edges):
        return None
    near_u, near_w, both = classes(locals_, u, w)
    a_u, a_w = locals_[u].phase, locals_[w].phase
    old_edges, new_edges = toggle(
        locals_, across(near_u, near_w, both))
    shift = dict.fromkeys(near_w, a_u)
    shift.update(dict.fromkeys(both, a_u + Fraction(1, 2)))
    old_phases, new_phases = add_phases(locals_, shift)
    leaf = [(phase_box(a_w), ["leaf"], []), (zx.H, ["kept"], ["leaf"])]
    return step(
        rewriter, near_u + near_w + both, [u, w],
        locals_[u].phases + locals_[w].phases + [
            j for t in (u, w) for target, js in locals_[t].edges.items()
            if t == w or target != w for j in js] + old_phases + old_edges,
        new_phases + new_edges + leaf + spider_like(a_u, near_u + both))


def pivot_boundary(rewriter: Simplifier, spider: int) -> Step | None:
    """
    Pivot along an edge from an interior Pauli spider ``v`` to a spider
    ``w`` with one boundary, after unfusing the phase of ``w`` into a new
    phase gadget: the boundary goes to ``v``.
    """
    here = rewriter.local(spider)
    if here.other or len(boundaries(rewriter, here)) != 1:
        return None
    for v in here.edges:
        there = rewriter.local(v)
        if is_interior(rewriter, v, there) and is_pauli(there.phase):
            result = pivot_boundary_from(rewriter, v, there)
            if result is not None:
                return result
    return None


def near_boundary(rewriter: Simplifier) -> list[int]:
    """
    The spiders with a boundary, i.e. those with a port that are not a
    boundary themselves, and those with a Hadamard edge to a boundary.
    """
    result = []
    for spider in rewriter.dom_wires + rewriter.cod_wires:
        if is_boundary(rewriter, spider):
            (j, side, _), = rewriter.incidence[spider]
            result.append(rewriter.box_wires[j][1 - side][0])
        else:
            result.append(spider)
    return list(dict.fromkeys(result))


def pivot_boundary_from(rewriter: Simplifier, v: int, here: Local
                        ) -> Step | None:
    """ The :func:`pivot_boundary` from an interior spider ``v``. """
    locals_ = neighbourhood(rewriter, v, known={v: here})
    if not is_interior(rewriter, v, here) or not is_pauli(here.phase) or any(
            locals_[t].other or is_boundary(rewriter, t)
            or locals_[t].degree == 1 for t in here.edges):
        return None
    eligible = [t for t in here.edges if len(boundaries(
        rewriter, locals_[t])) == 1 and not locals_[t].loops]
    if not eligible:
        return None
    quarter = [t for t in eligible if not is_pauli(locals_[t].phase)
               and is_clifford(locals_[t].phase)]
    w = quarter[-1] if quarter else eligible[0]
    locals_.update(neighbourhood(rewriter, w))
    bound, = boundaries(rewriter, locals_[w])
    near_v, near_w, both = classes(
        locals_, v, w, set() if bound is None else {bound})
    if any(locals_[t].other for t in near_w):
        return None
    a_v = here.phase
    old_edges, new_edges = toggle(
        locals_, across(near_v, near_w, both))
    shift = dict.fromkeys(near_w, a_v)
    shift.update(dict.fromkeys(both, a_v + Fraction(1, 2)))
    old_phases, new_phases = add_phases(locals_, shift)
    new_gadget = [(zx.H, ["hub"], ["leaf"])] + [
        (zx.H, ["hub"], [t]) for t in near_v + both]
    new_gadget += [(phase_box(locals_[w].phase), ["leaf"], [])]\
        if locals_[w].phase else []
    new_gadget += [(phase_box(a_v), ["hub"], [])] if a_v else []
    deleted = locals_[v].phases + locals_[w].phases + [
        j for t in (v, w) for target, js in locals_[t].edges.items()
        if t == w or target != w for j in js] + old_phases + old_edges
    interface = near_v + near_w + both
    new = new_phases + new_edges + new_gadget + spider_like(
        a_v, near_v + both)
    if bound is None:
        return step(rewriter, interface + [w], [v], deleted,
                    new + [(zx.H, [w], ["kept"])])
    return step(rewriter, interface + [bound], [v, w], deleted, new,
                glue={bound: "kept"})


def gadget(rewriter: Simplifier, hub: int) -> tuple[int, frozenset] | None:
    """
    The leaf of a phase gadget, i.e. the first neighbour of a Pauli spider
    with no other neighbour and a phase that is not Clifford, and the other
    neighbours of the hub.
    """
    here = rewriter.local(hub)
    if here.other or here.loops or is_boundary(rewriter, hub)\
            or not is_pauli(here.phase):
        return None
    for leaf in here.edges:
        if rewriter.ports[leaf] or len(rewriter.incidence[leaf]) > 2:
            continue
        there = rewriter.local(leaf)
        if there.degree == 1 and not there.other and not there.loops\
                and not is_clifford(there.phase):
            return leaf, frozenset(here.edges) - {leaf}
    return None


def fuse_gadgets(rewriter: Simplifier, spider: int) -> Step | None:
    """
    Fuse the phase gadgets with the same neighbours into one, adding their
    phases, negated for a hub with a half-turn, and move the half-turn of a
    lone hub into its leaf.
    """
    here = rewriter.local(spider)
    hubs = [spider] if gadget(rewriter, spider) else []
    hubs += [t for t in here.edges
             if len(here.edges) == 1 and not here.ports]
    for hub in hubs:
        found = gadget(rewriter, hub)
        if found is None:
            continue
        leaf, key = found
        if not key or rewriter.ports[hub]:
            group, leaves = [hub], [leaf]
        else:
            nearest = min(key, key=lambda t: len(rewriter.incidence[t]))
            others = {t: gadget(rewriter, t) for t in rewriter.edges[nearest]
                      if t != hub and not rewriter.ports[t]
                      and len(rewriter.incidence[t]) - len(key) in (1, 2)}
            group = [hub] + [t for t, found in others.items()
                             if found is not None and found[1] == key]
            leaves = [leaf] + [
                others[t][0]  # ty: ignore[not-subscriptable]
                for t in group[1:]]
        phases = {t: rewriter.local(t).phase for t in group + leaves}
        if len(group) == 1 and not phases[hub]:
            continue
        total = sum(phases[l] if not phases[h] else -phases[l]
                    for h, l in zip(group, leaves)) % 1
        deleted = list(dict.fromkeys(
            j for t in group + leaves for j, *_ in rewriter.incidence[t]))
        if rewriter.ports[hub]:
            new = [(zx.H, [hub], ["leaf"])]
            new += [(zx.H, [hub], [t]) for t in sorted(key)]
            new += [(phase_box(total), ["leaf"], [])] if total else []
            return step(rewriter, sorted(key) + [hub], [leaf], deleted, new)
        new = [(zx.H, ["hub"], ["leaf"])] + [
            (zx.H, ["hub"], [t]) for t in sorted(key)]
        new += [(phase_box(total), ["leaf"], [])] if total else []
        return step(rewriter, sorted(key), group + leaves, deleted, new)
    return None


def copy(rewriter: Simplifier, spider: int) -> Step | None:
    """
    Copy a Pauli state through its neighbour: a spider with no phase or a
    half-turn and one Hadamard edge, to an interior spider, goes with that
    neighbour, whose other neighbours each gain its phase.
    """
    here = rewriter.local(spider)
    if here.ports or here.other or here.loops or len(here.edges) != 1\
            or not is_pauli(here.phase):
        return None
    (target, hadamards), = here.edges.items()
    there = rewriter.local(target)
    if len(hadamards) != 1 or not is_interior(rewriter, target, there):
        return None
    near = [t for t in there.edges if t != spider]
    locals_ = neighbourhood(rewriter, target, known={spider: here})
    old_phases, new_phases = add_phases(
        locals_, dict.fromkeys(near, here.phase))
    deleted = here.phases + there.phases + [
        j for js in there.edges.values() for j in js]
    return step(rewriter, near, [spider, target], deleted + old_phases,
                new_phases)


def supplement(rewriter: Simplifier, spider: int) -> Step | None:
    """
    Supplementarity: remove two interior spiders whose phases are not
    Clifford and add up to a half-turn or differ by one, with the same
    neighbours, the one aside, adding a half-turn to the neighbours when the
    phases are negated.
    """
    here = rewriter.local(spider)
    if not is_interior(rewriter, spider, here) or is_clifford(here.phase)\
            or not here.edges:
        return None
    near = set(here.edges)
    nearest = min(near, key=lambda t: len(rewriter.incidence[t]))
    for other in dict.fromkeys(list(rewriter.edges[nearest]) + list(near)):
        there = rewriter.local(other)
        if other == spider or is_clifford(there.phase)\
                or not is_interior(rewriter, other, there):
            continue
        plus = (here.phase + there.phase) % 1
        minus = (here.phase - there.phase) % 1
        if other not in near and set(there.edges) == near\
                and Fraction(1, 2) in (plus, minus):
            negate = plus == Fraction(1, 2)
        elif other in near and set(there.edges) - {spider} == near - {other}\
                and (not plus or minus == Fraction(1, 2)):
            negate = not plus
        else:
            continue
        interface = sorted(near - {other})
        if negate and any(is_boundary(rewriter, t) for t in interface):
            continue
        locals_ = neighbourhood(rewriter, *interface, known={
            spider: here, other: there})
        old_phases, new_phases = add_phases(locals_, dict.fromkeys(
            interface, Fraction(1, 2) if negate else ZERO))
        deleted = list(dict.fromkeys(
            j for t in (spider, other) for j, *_ in rewriter.incidence[t]))
        return step(rewriter, interface, [spider, other],
                    deleted + old_phases, new_phases)
    return None


Rule = Callable[[Simplifier, int], Step | None]
""" A rule tried at a spider, giving the step to take or ``None``. """

RULES: list[tuple[list[Rule], Callable | None]] = [
    ([normalize], None), ([remove_identity, pivot, complement], None),
    ([pivot_boundary], near_boundary), ([fuse_gadgets], None),
    ([copy, supplement], None), ([pivot_gadget], None)]
"""
The rules of :func:`full_reduce` by priority, each level with the spiders
to try them at: those a step touched, or those that a function of the
rewriter gives, e.g. :func:`near_boundary`. A rule is tried only when no
rule before it applies anywhere.
"""


class Agenda:
    """
    The spiders to try the rules of a level at, the one with fewest ports
    first: pivots and local complementations eliminate spiders the way
    Gaussian elimination does variables, and taking the sparsest first is
    what keeps the edges they add, the fill-in, low. A spider queued again
    moves to its current degree.

    Parameters:
        rewriter : The rewriter whose spiders these are.
        spiders : The spiders to try first.
    """
    def __init__(self, rewriter: Rewriter, spiders=()):
        self.rewriter, self.spiders, self.heap = rewriter, {}, []
        self.update(spiders)

    def update(self, spiders) -> None:
        """ Add spiders to try, or try them again, at their current degree. """
        for spider in spiders:
            degree = len(self.rewriter.incidence[spider])
            if self.spiders.get(spider) != degree:
                self.spiders[spider] = degree
                heappush(self.heap, (degree, spider))

    def pop(self) -> int | None:
        """ The next spider to try, ``None`` if there is none. """
        while self.heap:
            degree, spider = heappop(self.heap)
            if self.spiders.get(spider) == degree:
                del self.spiders[spider]
                return spider
        return None

    def __bool__(self):
        return bool(self.spiders)


def reduce(rewriter: Simplifier, rules=None) -> int:
    """
    Rewrite until no rule applies, returning the number of steps: each spider
    is tried once for each level of rules, the sparsest first, see
    :class:`Agenda`, and again whenever a step touches it or a neighbour.

    Parameters:
        rewriter : The hypergraph to rewrite in place.
        rules : The rules by priority, :data:`RULES` by default.
    """
    rules = RULES if rules is None else rules
    alive = [s for s, typ in enumerate(rewriter.spider_types) if typ]
    todo = [Agenda(rewriter, alive) for _ in rules]
    stale = [anchors is not None for _, anchors in rules]
    steps = 0
    while True:
        level = next((i for i, agenda in enumerate(todo)
                      if agenda or stale[i]), None)
        if level is None:
            return steps
        if stale[level]:
            todo[level].update(
                rules[level][1](rewriter))  # ty: ignore[call-non-callable]
            stale[level] = False
            continue
        spider = todo[level].pop()
        if spider is None or rewriter.spider_types[spider] is None:
            continue
        for rule in rules[level][0]:
            result = rule(rewriter, spider)
            if result is None:
                continue
            left, right, match = result
            touched = match[0][:len(left.dom)]
            new = rewriter.substitute(left, right, match)
            region = dict.fromkeys(rewriter.find(s) for s in touched)
            region.update(dict.fromkeys(
                s for j in new for wires in rewriter.box_wires[j]
                for s in wires))
            region.update(dict.fromkeys(
                rewriter.find(s) for s in rewriter.merged))
            rewriter.merged.clear()
            todo[0].update(region)
            region.update(dict.fromkeys(
                t for s in list(region) for t in rewriter.edges[s]))
            for i, (_, anchors) in enumerate(rules[1:], 1):
                if anchors is None:
                    todo[i].update(region)
                else:
                    stale[i] = True
            steps += 1
            break


def full_reduce(graph: Hypergraph) -> Hypergraph:
    """
    Simplify a graph-like ZX diagram with the rules of :data:`RULES`.

    Parameters:
        graph : A hypergraph with :data:`zx.H` and phase boxes only.
    """
    rewriter = Simplifier(graph)
    reduce(rewriter)
    return rewriter.to_hypergraph()


def from_diagram(diagram: zx.Diagram) -> Hypergraph:
    """
    A graph-like hypergraph from a ZX diagram: the legs of a green spider
    are one spider of the hypergraph, with its phase, a red spider is a green
    one with a Hadamard edge on each leg, and scalars are dropped.

    Parameters:
        diagram : A ZX diagram with green and red spiders and Hadamards.

    Example
    -------
    >>> from discopy.quantum.zx import Z, H
    >>> print(from_diagram(Z(1, 1, .25) >> H))
    Z(1, 2) >> Z(1, 0, 1/4) @ H
    """
    graph = diagram.to_hypergraph()
    parent = list(range(graph.n_spiders))

    def find(spider):
        while parent[spider] != spider:
            parent[spider] = parent[parent[spider]]
            spider = parent[spider]
        return spider

    for box, (dom, cod) in zip(graph.boxes, graph.box_wires):
        if isinstance(box, zx.Z):
            for spider in dom + cod:
                parent[find(spider)] = find((dom + cod)[0])
    spiders = list(dict.fromkeys(find(s) for s in range(graph.n_spiders)))
    index = {spider: i for i, spider in enumerate(spiders)}
    boxes, wires = [], []
    for box, (dom, cod) in zip(graph.boxes, graph.box_wires):
        legs = [index[find(s)] for s in dom + cod]
        if isinstance(box, (zx.Z, zx.X)):
            if isinstance(box, zx.X) or not legs:
                centre = len(spiders)
                spiders.append(None)
            else:
                centre = legs[0]
            if isinstance(box, zx.X):
                boxes += len(legs) * [zx.H]
                wires += [((centre, ), (leg, )) for leg in legs]
            if box.phase:
                boxes.append(phase_box(Fraction(box.phase) % 1))
                wires.append(((centre, ), ()))
        elif box is zx.H or box == zx.H:
            boxes.append(zx.H)
            wires.append(((legs[0], ), (legs[1], )))
        elif not isinstance(box, zx.Scalar):
            raise ValueError(messages.NOT_GRAPHLIKE.format(box))
    return Hypergraph(
        diagram.dom, diagram.cod, tuple(boxes), (
            tuple(index[find(s)] for s in graph.dom_wires), tuple(wires),
            tuple(index[find(s)] for s in graph.cod_wires)),
        len(spiders) * [qubit])


def from_pyzx(graph) -> Hypergraph:
    """
    A graph-like hypergraph from a PyZX graph: its green spiders joined by
    plain edges are one spider, a red spider is green with its edges toggled
    between plain and Hadamard, and a boundary behind a Hadamard edge is a
    spider of its own.

    Parameters:
        graph : A PyZX graph with boundaries, green and red spiders.
    """
    from pyzx import VertexType, EdgeType
    boundary = set(graph.inputs()) | set(graph.outputs())
    red = {v for v in graph.vertices()
           if graph.type(v) == VertexType.X}  # noqa: E721
    parent = {v: v for v in graph.vertices()}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    hadamards = []
    for edge in graph.edges():
        a, b = graph.edge_st(edge)
        plain = (graph.edge_type(edge) == EdgeType.SIMPLE) == (
            (a in red) == (b in red))
        if plain:
            parent[find(a)] = find(b)
        else:
            hadamards.append((a, b))
    spiders = list(dict.fromkeys(find(v) for v in graph.vertices()))
    index = {s: i for i, s in enumerate(spiders)}
    boxes, wires = [], []
    for a, b in hadamards:
        boxes.append(zx.H)
        wires.append(((index[find(a)], ), (index[find(b)], )))
    for v in graph.vertices():
        if v not in boundary and graph.phase(v):
            boxes.append(phase_box(Fraction(graph.phase(v)) / 2 % 1))
            wires.append(((index[find(v)], ), ()))
    dom_wires = tuple(index[find(v)] for v in graph.inputs())
    cod_wires = tuple(index[find(v)] for v in graph.outputs())
    return Hypergraph(
        Nat(len(dom_wires)), Nat(len(cod_wires)), tuple(boxes),
        (dom_wires, tuple(wires), cod_wires), len(spiders) * [qubit])


def to_pyzx(graph: Hypergraph):
    """
    The PyZX graph of a graph-like hypergraph, with a green spider for each
    spider and a boundary on a plain edge for each port, see
    :func:`from_pyzx`.

    Parameters:
        graph : A hypergraph with :data:`zx.H` and phase boxes only.
    """
    from pyzx import Graph, VertexType, EdgeType
    result = Graph()
    spiders = [result.add_vertex(VertexType.Z, qubit=i, row=1)
               for i, _ in enumerate(graph.spider_types)]
    table = {}
    for box, (dom, cod) in zip(graph.boxes, graph.box_wires):
        if box == zx.H and dom[0] == cod[0]:
            result.add_to_phase(spiders[dom[0]], 1)
        elif box == zx.H:
            key = result.edge(spiders[dom[0]], spiders[cod[0]])
            table.setdefault(key, [0, 0])[1] += 1
        else:
            result.add_to_phase(spiders[dom[0]], Fraction(box.phase) * 2)
    result.add_edge_table(table)
    inputs, outputs = [], []
    for row, wires, ends in (
            (0, graph.dom_wires, inputs), (2, graph.cod_wires, outputs)):
        for qubit_, spider in enumerate(wires):
            ends.append(result.add_vertex(
                VertexType.BOUNDARY, qubit=qubit_, row=row))
            result.add_edge(
                result.edge(ends[-1], spiders[spider]), EdgeType.SIMPLE)
    result.set_inputs(tuple(inputs))
    result.set_outputs(tuple(outputs))
    return result


def size(graph) -> tuple[int, int, int]:
    """
    The number of spiders, of Hadamard edges between them and of spiders
    whose phase is not Clifford, the T-count, counting a boundary behind a
    Hadamard edge as no spider, as PyZX does.

    Parameters:
        graph : A graph-like hypergraph, or a PyZX graph.
    """
    if not isinstance(graph, Hypergraph):
        from pyzx import tcount, EdgeType
        boundary = set(graph.inputs()) | set(graph.outputs())
        return (graph.num_vertices() - len(boundary), sum(
            1 for edge in graph.edges()
            if graph.edge_type(edge) == EdgeType.HADAMARD
            and not boundary & set(graph.edge_st(edge))), tcount(graph))
    rewriter = Simplifier(graph)
    spiders = [s for s in range(graph.n_spiders)
               if not is_boundary(rewriter, s) and (
                   rewriter.ports[s] < 2 or rewriter.incidence[s])]
    inside = set(spiders)
    edges = sum(1 for box, (dom, cod) in zip(graph.boxes, graph.box_wires)
                if box == zx.H and {dom[0], cod[0]} <= inside)
    tcount = sum(1 for s in spiders if not is_clifford(
        rewriter.local(s).phase))
    return len(spiders), edges, tcount
