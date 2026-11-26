# -*- coding: utf-8 -*-

"""
Interaction nets :cite:p:`Lafont97`, checked against double-pushout rewriting
in :mod:`discopy.hypergraph`, and their Geometry of Interaction token machine
:cite:p:`JoyalEtAl96,Abramsky96`, checked against
:func:`discopy.interaction.Int` over :class:`discopy.python.additive.Function`.

A :class:`Signature` gives each :class:`Symbol` an auxiliary arity and maps
active pairs of symbols to right-hand-side nets. A :class:`Net` stores its
cells as flat lists of symbols and port tuples, together with a fixpoint-free
involution ``link`` on all ports and a stack of active pairs. Rewriting an
active pair with :class:`Rewriter` takes time linear in the rule's right-hand
side.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Symbol
    Signature
    Net
    Rewriter

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        combinators
        int_cell
        int_path

Example
-------
Lafont's :cite:p:`Lafont97` Fig. 11 net reduces to two parallel wires in five
steps, matching the double-pushout normal form of
:meth:`discopy.hypergraph.Hypergraph.normalize`:

>>> from discopy.frobenius import Ty, Box, Hypergraph as H
>>> from discopy.inet import Net, combinators
>>> x = Ty('x')
>>> gamma, delta = (Box(name, x, x @ x).to_hypergraph() for name in "γδ")
>>> fig11 = (H.caps(x, x) >> delta @ delta >> x @ H.cups(x, x) @ x
...          >> gamma.transpose() >> delta >> gamma @ gamma)
>>> net = Net.from_hypergraph(fig11, combinators())
>>> steps = list(net.normalize())
>>> assert len(steps) == 5 and steps[-1] == net.reduce()
>>> assert net.reduce().to_hypergraph() == H.caps(x, x) @ H.caps(x, x)
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self

from discopy import frobenius, hypergraph, interaction
from discopy.python import additive


@dataclass(frozen=True)
class Symbol:
    """
    A cell symbol with a name and an optional hashable label.

    Parameters:
        name : The name of the symbol, e.g. ``"γ"``, ``"δ"`` or ``"ε"``.
        label : An optional label, e.g. an integer index on ``δ``.

    Example
    -------
    >>> d0 = Symbol('δ', 0)
    >>> assert str(d0) == 'δ_0' and eval(repr(d0)) == d0
    """
    name: str
    label: Any = None

    @classmethod
    def cast(cls, value: Symbol | str) -> Symbol:
        """Coerce a string or :class:`Symbol` into a :class:`Symbol`."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        raise TypeError(f"Expected Symbol or str, got {value!r}.")

    def with_label(self, label: Any = None) -> Symbol:
        """Return a copy of this symbol carrying ``label``."""
        return Symbol(self.name, label)

    def __repr__(self) -> str:
        if self.label is None:
            return f"Symbol({self.name!r})"
        return f"Symbol({self.name!r}, {self.label!r})"

    def __str__(self) -> str:
        return self.name if self.label is None else f"{self.name}_{self.label}"


Arities = (
    Mapping[Symbol, int]
    | Mapping[str, int]
    | Mapping[Symbol | str, int]
    | Callable[[Symbol], int]
)
Rules = (
    Mapping[tuple[Symbol, Symbol], "Net"]
    | Mapping[tuple[str, str], "Net"]
    | Mapping[tuple[Symbol | str, Symbol | str], "Net"]
    | Callable[[Symbol, Symbol], "Net | None"]
)
Endpoint = int | tuple[int, int]
Context = tuple[tuple[int, ...], tuple[int, ...]]
PathResult = tuple[int, Context] | Literal["erased", "undefined"]

UNLABELLED_COMBINATORS = frozenset({Symbol("γ"), Symbol("δ"), Symbol("ε")})


class Signature:
    """
    An interaction net signature: symbol arities and binary rewrite rules.

    Parameters:
        arities : Mapping or callable giving the auxiliary arity of each
            :class:`Symbol`.
        rules : Mapping or callable sending ``(s1, s2)`` to a right-hand-side
            :class:`Net` whose ``arity(s1) + arity(s2)`` free ports correspond
            to the auxiliary ports of ``s1`` followed by those of ``s2``.
        symbols : Optional explicit sequence of generator symbols.
        expr : Optional expression string used by :meth:`__repr__`.

    Example
    -------
    >>> sig = combinators()
    >>> assert [sig.arity(s) for s in ("γ", "δ", "ε")] == [2, 2, 0]
    """

    def __init__(
        self, arities: Arities, rules: Rules | None = None,
        symbols: Iterable[Symbol | str] = (), expr: str | None = None,
    ):
        if isinstance(arities, Mapping):
            self.arities: Mapping[Symbol, int] | Callable[[Symbol], int] = {
                Symbol.cast(k): int(v) for k, v in arities.items()}
            default_symbols = tuple(self.arities.keys())
        else:
            self.arities, default_symbols = arities, ()
        if rules is None:
            self.rules: (
                Mapping[tuple[Symbol, Symbol], Net]
                | Callable[[Symbol, Symbol], Net | None]
            ) = {}
        elif isinstance(rules, Mapping):
            self.rules = {
                (Symbol.cast(a), Symbol.cast(b)): rhs
                for (a, b), rhs in rules.items()}
        else:
            self.rules = rules
        chosen = tuple(Symbol.cast(s) for s in symbols)
        self.symbols: tuple[Symbol, ...] = chosen or default_symbols
        self.expr = expr

    def arity(self, symbol: Symbol | str) -> int:
        """Return the number of auxiliary ports of ``symbol``."""
        sym = Symbol.cast(symbol)
        if isinstance(self.arities, Mapping):
            return self.arities[sym]
        return self.arities(sym)

    def rule(
        self, first: Symbol | str, second: Symbol | str
    ) -> Net | None:
        """
        Return the right-hand-side :class:`Net` for active pair
        ``(first, second)``, or ``None`` if undefined.
        """
        s1, s2 = Symbol.cast(first), Symbol.cast(second)
        if isinstance(self.rules, Mapping):
            if (s1, s2) in self.rules:
                return self.rules[s1, s2]
            if (s2, s1) in self.rules:
                rhs = self.rules[s2, s1]
                return rhs.swap_halves(self.arity(s2), self.arity(s1))
            return None
        return self.rules(s1, s2)

    def box(self, symbol: Symbol | str) -> frobenius.Box:
        """Encode ``symbol`` as a :class:`discopy.frobenius.Box`."""
        sym = Symbol.cast(symbol)
        x = frobenius.Ty("x")
        return frobenius.Box(
            sym.name, x, x ** self.arity(sym), data=sym.label)

    def hypergraph_rule(
        self, first: Symbol | str, second: Symbol | str
    ) -> hypergraph.Rule:
        """
        Encode the interaction rule for ``(first, second)`` as a DPO
        :data:`discopy.hypergraph.Rule`.
        """
        s1, s2 = Symbol.cast(first), Symbol.cast(second)
        rhs = self.rule(s1, s2)
        if rhs is None:
            raise KeyError((s1, s2))
        m, n = self.arity(s1), self.arity(s2)
        x = frobenius.Ty("x")
        dom_wires = tuple(range(m))
        cod_wires = tuple(range(m + 1, m + 1 + n))
        box_wires = (((m,), dom_wires), ((m,), cod_wires))
        left = frobenius.Hypergraph(
            x ** m, x ** n, (self.box(s1), self.box(s2)),
            (dom_wires, box_wires, cod_wires),
        )
        return (left, rhs.to_hypergraph(n_dom=m))

    def hypergraph_rules(
        self, symbols: Iterable[Symbol | str] | None = None
    ) -> list[hypergraph.Rule]:
        """
        Return the finite list of DPO rules for all unordered symbol pairs
        reachable from ``symbols`` (or :attr:`symbols` when ``None``).
        """
        source = self.symbols if symbols is None else symbols
        ordered = list(dict.fromkeys(Symbol.cast(s) for s in source))
        seen = set(ordered)
        idx = 0
        while idx < len(ordered):
            s1 = ordered[idx]
            for j in range(idx + 1):
                rhs = self.rule(ordered[j], s1)
                if rhs is not None:
                    for sym in rhs.cells:
                        if sym not in seen:
                            seen.add(sym)
                            ordered.append(sym)
            idx += 1
        return [
            self.hypergraph_rule(s1, s2)
            for i, s1 in enumerate(ordered)
            for s2 in ordered[i:]
            if self.rule(s1, s2) is not None
        ]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Signature):
            return False
        if self.expr is not None or other.expr is not None:
            return self.expr == other.expr and self.symbols == other.symbols
        return self.arities == other.arities and self.rules == other.rules

    def __repr__(self) -> str:
        if self.expr is not None:
            return self.expr
        if not self.rules:
            return f"Signature({self.arities!r})"
        return f"Signature({self.arities!r}, {self.rules!r})"


class Net:
    """
    An interaction net over a :class:`Signature`.

    Free boundary ports are numbered ``0 .. n_free - 1``, followed by the
    contiguous ports ``(principal, aux_1, ..., aux_k)`` of each cell. Wires
    are given by ``link``, a fixpoint-free involution on all ports, plus
    ``loops`` closed wires with no ports.

    Parameters:
        signature : The :class:`Signature` of the net.
        cells : The symbol of each cell.
        link : Fixpoint-free involution on ``range(n_ports)``.
        loops : Number of closed wire loops with no ports.

    Example
    -------
    >>> sig = combinators()
    >>> pair = Net.from_wires(sig, ["γ", "γ"], [
    ...     ((0, 0), (1, 0)),
    ...     (0, (0, 1)), (1, (0, 2)),
    ...     (2, (1, 1)), (3, (1, 2)),
    ... ])
    >>> assert pair.reduce() == Net.from_wires(sig, [], [(0, 2), (1, 3)])
    """

    def __init__(
        self,
        signature: Signature,
        cells: Sequence[Symbol | str] = (),
        link: Sequence[int] = (),
        loops: int = 0,
    ):
        self.signature = signature
        self.cells = [Symbol.cast(s) for s in cells]
        arities = [signature.arity(s) for s in self.cells]
        cell_ports = sum(1 + arity for arity in arities)
        self.link = list(link)
        n_free = len(self.link) - cell_ports
        if n_free < 0 or loops < 0:
            raise ValueError(
                f"Invalid port count {len(self.link)} or loops {loops}.")
        for port, other in enumerate(self.link):
            if not 0 <= other < len(self.link) or other == port:
                raise ValueError("link is not a fixpoint-free involution.")
            if self.link[other] != port:
                raise ValueError("link is not an involution.")
        self.free = list(range(n_free))
        self.ports: list[tuple[int, ...]] = []
        cursor = n_free
        for arity in arities:
            self.ports.append(tuple(range(cursor, cursor + 1 + arity)))
            cursor += 1 + arity
        principals = {
            cell_ports[0]: idx for idx, cell_ports in enumerate(self.ports)
        }
        self.active: list[tuple[int, int]] = [
            (idx, other)
            for idx, cell_ports in enumerate(self.ports)
            if (other := principals.get(self.link[cell_ports[0]])) is not None
            and idx < other
        ]
        self.loops = loops

    @staticmethod
    def resolve_endpoint(
        end: Endpoint, offsets: Sequence[int], arities: Sequence[int]
    ) -> int:
        """Resolve a free-port or ``(cell, slot)`` endpoint to a port id."""
        if isinstance(end, int):
            if end < 0:
                raise ValueError(f"Negative free port {end}.")
            return end
        cell_idx, slot = end
        if not (0 <= cell_idx < len(arities)
                and 0 <= slot <= arities[cell_idx]):
            raise ValueError(f"Invalid cell port {end}.")
        return offsets[cell_idx] + slot

    @classmethod
    def from_wires(
        cls,
        signature: Signature,
        cells: Sequence[Symbol | str],
        wires: Iterable[tuple[Endpoint, Endpoint]],
        loops: int = 0,
    ) -> Self:
        """
        Build a :class:`Net` from pairs of endpoints, each either a free port
        index ``i`` or a ``(cell, slot)`` pair with ``slot = 0`` principal.
        """
        arities = [signature.arity(s) for s in cells]
        wire_list = list(wires)
        cursor = max(
            (end + 1 for pair in wire_list for end in pair
             if isinstance(end, int)),
            default=0,
        )
        offsets: list[int] = []
        for arity in arities:
            offsets.append(cursor)
            cursor += 1 + arity
        if 2 * len(wire_list) != cursor:
            raise ValueError("Wires do not partition all ports.")
        link = [-1] * cursor
        for first, second in wire_list:
            u = cls.resolve_endpoint(first, offsets, arities)
            v = cls.resolve_endpoint(second, offsets, arities)
            if u == v or link[u] != -1 or link[v] != -1:
                raise ValueError("Duplicate or self-wired port in wires.")
            link[u], link[v] = v, u
        return cls(signature, cells, link, loops=loops)

    def swap_halves(self, first_len: int, second_len: int) -> Self:
        """Swap the first ``first_len`` and next ``second_len`` free ports."""
        n_free = len(self.free)
        if first_len + second_len != n_free:
            raise ValueError("Half lengths do not sum to free port count.")
        perm = [
            (p - first_len if p >= first_len else p + second_len)
            if p < n_free else p
            for p in range(len(self.link))
        ]
        new_link = [0] * len(self.link)
        for old_p, new_p in enumerate(perm):
            new_link[new_p] = perm[self.link[old_p]]
        return type(self)(
            self.signature, self.cells, new_link, loops=self.loops)

    def step(self) -> Net | None:
        """Perform one active-pair rewrite step, or ``None`` if normal."""
        rewriter = Rewriter(self)
        return rewriter.to_net() if rewriter.step() else None

    def normalize(self) -> Iterator[Net]:
        """Yield each single-step rewritten :class:`Net` until normal."""
        current = self
        while (nxt := current.step()) is not None:
            yield nxt
            current = nxt

    def reduce(self, max_steps: int | None = None) -> Net:
        """
        Reduce active pairs in place on flat lists until none remain.

        Raises:
            RuntimeError : If ``max_steps`` is exceeded while active pairs
                remain.
        """
        if not self.active:
            return self
        rewriter, steps = Rewriter(self), 0
        while rewriter.step():
            steps += 1
            if max_steps is not None and steps > max_steps:
                raise RuntimeError(f"Exceeded {max_steps} reduction steps.")
        return rewriter.to_net()

    def to_hypergraph(self, n_dom: int = 0) -> frobenius.Hypergraph:
        """
        Convert this net to a bijective :class:`discopy.frobenius.Hypergraph`
        with ``n_dom`` free ports in the domain and the rest in the codomain.
        """
        if not 0 <= n_dom <= len(self.free):
            raise ValueError(f"Invalid n_dom={n_dom} for {len(self.free)}.")
        x = frobenius.Ty("x")
        spider_of = [0] * len(self.link)
        n_spiders = 0
        for u, v in enumerate(self.link):
            if u < v:
                spider_of[u] = spider_of[v] = n_spiders
                n_spiders += 1
        dom_wires = tuple(spider_of[p] for p in self.free[:n_dom])
        cod_wires = tuple(spider_of[p] for p in self.free[n_dom:])
        box_wires = tuple(
            ((spider_of[cports[0]],), tuple(spider_of[p] for p in cports[1:]))
            for cports in self.ports
        )
        boxes = tuple(self.signature.box(sym) for sym in self.cells)
        spider_types = (x,) * (n_spiders + self.loops)
        return frobenius.Hypergraph(
            x ** n_dom,
            x ** (len(self.free) - n_dom),
            boxes,
            (dom_wires, box_wires, cod_wires),
            spider_types,
        )

    @staticmethod
    def boxes_to_symbols(
        boxes: Sequence[Any],
    ) -> tuple[list[Symbol], dict[Symbol, int]]:
        """Extract cell symbols and check arity consistency from boxes."""
        cells: list[Symbol] = []
        arities: dict[Symbol, int] = {}
        for box in boxes:
            if len(box.dom) != 1:
                raise ValueError(f"Box {box!r} must have unary domain.")
            sym = Symbol(box.name, box.data)
            if sym in arities and arities[sym] != len(box.cod):
                raise ValueError(f"Inconsistent arities for {sym!r}.")
            arities[sym] = len(box.cod)
            cells.append(sym)
        return cells, arities

    @classmethod
    def from_hypergraph(
        cls,
        graph: frobenius.Hypergraph,
        signature: Signature | None = None,
    ) -> Self:
        """
        Decode a bijective :class:`discopy.frobenius.Hypergraph` whose boxes
        have unary domains (principal ports) into a :class:`Net`.
        """
        if not graph.is_bijective:
            raise ValueError("Hypergraph is not bijective.")
        cells, arities = cls.boxes_to_symbols(graph.boxes)
        if signature is None:
            signature = cls.infer_signature(arities)
        elif any(signature.arity(s) != a for s, a in arities.items()):
            raise ValueError("Box arity does not match signature.")
        spiders = [
            *graph.dom_wires,
            *graph.cod_wires,
            *(s for dom_w, cod_w in graph.box_wires for s in dom_w + cod_w),
        ]
        seen: dict[int, int] = {}
        link = [0] * len(spiders)
        for port_idx, spider in enumerate(spiders):
            if spider in seen:
                other = seen.pop(spider)
                link[port_idx], link[other] = other, port_idx
            else:
                seen[spider] = port_idx
        return cls(signature, cells, link, loops=len(graph.scalar_spiders))

    @staticmethod
    def infer_signature(arities: Mapping[Symbol, int]) -> Signature:
        """Infer a default :class:`Signature` from observed box arities."""
        is_comb = all(
            (sym == Symbol("γ") and arity == 2)
            or (sym.name == "δ" and arity == 2)
            or (sym == Symbol("ε") and arity == 0)
            for sym, arity in arities.items()
        )
        if not is_comb:
            return Signature(arities)
        labels = tuple(dict.fromkeys(
            s.label for s in arities if s.name == "δ" and s.label is not None
        ))
        has_unlabelled = Symbol("δ") in arities
        return combinators(labels if labels and not has_unlabelled else None)

    def port_table(self) -> tuple[list[int], list[int]]:
        """Return ``(cell_of, slot_of)`` arrays over all ports of the net."""
        cell_of = [-1] * len(self.link)
        slot_of = [-1] * len(self.link)
        for cell_idx, cports in enumerate(self.ports):
            for slot, p in enumerate(cports):
                cell_of[p] = cell_idx
                slot_of[p] = slot
        return cell_of, slot_of

    def has_vicious_circle(self) -> bool:
        """
        Whether the directed ``principal -> auxiliary`` graph has a cycle
        (Lafont's vicious circle :cite:p:`Lafont97`).
        """
        cell_of, slot_of = self.port_table()
        state = [0] * len(self.cells)
        for start in range(len(self.cells)):
            curr = start
            while curr >= 0 and state[curr] == 0:
                state[curr] = 1
                target = self.link[self.ports[curr][0]]
                curr = cell_of[target] if slot_of[target] > 0 else -1
            if curr >= 0 and state[curr] == 1:
                return True
            walk = start
            while walk >= 0 and state[walk] == 1:
                state[walk] = 2
                target = self.link[self.ports[walk][0]]
                walk = cell_of[target] if slot_of[target] > 0 else -1
        return False

    def assert_unlabelled_combinators(self) -> None:
        """Raise ``ValueError`` unless every cell is ``γ``, ``δ`` or ``ε``."""
        for sym in self.cells:
            if sym not in UNLABELLED_COMBINATORS:
                raise ValueError(
                    f"Token machine requires unlabelled γ/δ/ε, got {sym!r}.")

    def path(
        self, port: int, context: Context = ((), ())
    ) -> PathResult:
        r"""
        Trace a Geometry of Interaction token entering free ``port`` with
        ``context`` ``(gamma_stack, delta_stack)`` over ``{1, 2}``.

        Entering ``γ`` or ``δ`` at auxiliary ``i`` pushes ``i`` onto that
        symbol's stack and leaves by the principal port; entering at the
        principal port pops ``a`` and leaves by auxiliary ``a`` (or returns
        ``"undefined"`` if empty); entering ``ε`` returns ``"erased"``.

        Example
        -------
        An identity wire forwards any token unchanged, and a ``γ``-``γ``
        active pair pushes then pops to yield the same path as its normal form:

        >>> sig = combinators()
        >>> wire = Net.from_wires(sig, [], [(0, 1)])
        >>> wire.path(0, ((1,), ()))
        (1, ((1,), ()))
        >>> pair = Net.from_wires(sig, ['γ', 'γ'], [
        ...     ((0, 0), (1, 0)),
        ...     (0, (0, 1)), (1, (0, 2)),
        ...     (2, (1, 1)), (3, (1, 2)),
        ... ])
        >>> assert pair.path(0) == pair.reduce().path(0) == (2, ((), ()))
        """
        self.assert_unlabelled_combinators()
        if not 0 <= port < len(self.free):
            raise ValueError(f"Invalid free port {port}.")
        g_stack, d_stack = list(context[0]), list(context[1])
        cell_of, slot_of = self.port_table()
        current = self.link[port]
        while current >= len(self.free):
            cell_idx, slot = cell_of[current], slot_of[current]
            sym = self.cells[cell_idx]
            if sym == Symbol("ε"):
                return "erased"
            stack = d_stack if sym.name == "δ" else g_stack
            if slot == 0:
                if not stack:
                    return "undefined"
                branch = stack.pop()
                current = self.link[self.ports[cell_idx][branch]]
            else:
                stack.append(slot)
                current = self.link[self.ports[cell_idx][0]]
        return (current, (tuple(g_stack), tuple(d_stack)))

    def to_int(self) -> interaction.Diagram:
        """
        Compile an unlabelled combinator net into a morphism of
        ``Int(python.additive.Function)``, composing :func:`int_cell` with
        ``Int`` cups, caps and permutations.

        Each free port has type ``T(X, X)`` with
        ``X = additive.Function.ob(tuple)``, followed by one sink wire
        ``T(X, O)`` carrying ``"erased"`` and ``"undefined"`` outcomes.

        Example
        -------
        >>> sig = combinators()
        >>> wire = Net.from_wires(sig, [], [(0, 1)])
        >>> assert int_path(wire.to_int(), 0, ((2,), ())) == (1, ((2,), ()))
        >>> pair = Net.from_wires(sig, ['γ', 'γ'], [
        ...     ((0, 0), (1, 0)),
        ...     (0, (0, 1)), (1, (0, 2)),
        ...     (2, (1, 1)), (3, (1, 2)),
        ... ])
        >>> assert int_path(pair.to_int(), 1) == pair.path(1) == (3, ((), ()))
        """
        self.assert_unlabelled_combinators()
        func = additive.Function
        int_diag: Any = interaction.Diagram[func]
        port_ty, sink_ty = int_port_and_sink()
        n_free = len(self.free)
        factors, items = self.int_factors(port_ty, sink_ty)
        cell_cups = [
            (u, self.link[u]) for u in range(n_free, len(self.link))
            if u < self.link[u]]
        n_sinks = max(1, len(self.cells))
        desired = [
            *(("port", u) for u in range(n_free)),
            *(("port", end) for pair in cell_cups for end in pair),
            *(("sink", idx) for idx in range(n_sinks))]
        doms = [port_ty if kind == "port" else sink_ty for kind, _ in items]
        perm = int_diag.permutation(
            [items.index(target) for target in desired], doms)
        closers: list[Any] = [
            int_diag.id(port_ty ** n_free),
            *(int_diag.cups(port_ty, port_ty) for _ in cell_cups),
            int_diag.id(sink_ty) if n_sinks == 1 else int_diag(
                func.merge(func.ob(tuple), n_sinks),
                sink_ty ** n_sinks, sink_ty)]
        return (
            functools.reduce(operator.matmul, factors)
            >> perm
            >> functools.reduce(operator.matmul, closers))

    def int_factors(
        self, port_ty: interaction.Ty, sink_ty: interaction.Ty
    ) -> tuple[list[Any], list[tuple[str, int]]]:
        """Generator tensor factors and wire tags for :meth:`to_int`."""
        int_diag: Any = interaction.Diagram[additive.Function]
        n_free = len(self.free)
        factors: list[Any] = []
        items: list[tuple[str, int]] = []
        for u, v in enumerate(self.link[:n_free]):
            if u < v < n_free:
                factors.append(int_diag.caps(port_ty, port_ty))
                items.extend([("port", u), ("port", v)])
        for cell_idx, sym in enumerate(self.cells):
            factors.append(int_cell(sym))
            for p in self.ports[cell_idx]:
                target = self.link[p] if self.link[p] < n_free else p
                items.append(("port", target))
            items.append(("sink", cell_idx))
        if not self.cells:
            empty_fn = additive.Function(
                lambda x, tag=0: x,
                additive.Function.ob(), additive.Function.ob(tuple))
            empty_dom = interaction.Ty[additive.Function.ob]()
            factors.append(int_diag(empty_fn, empty_dom, sink_ty))
            items.append(("sink", 0))
        return factors, items

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Net):
            return False
        if (len(self.free) != len(other.free)
                or self.loops != other.loops
                or len(self.cells) != len(other.cells)):
            return False
        if (self.cells == other.cells
                and self.ports == other.ports
                and self.link == other.link):
            return True
        return self.to_hypergraph() == other.to_hypergraph()

    def __hash__(self) -> int:
        return hash(self.to_hypergraph())

    def __repr__(self) -> str:
        loops = f", loops={self.loops}" if self.loops else ""
        return (
            f"Net({self.signature!r}, {self.cells!r}, "
            f"{self.link!r}{loops})"
        )


class Rewriter:
    r"""
    In-place interaction net rewriter operating on flat lists in
    :math:`O(|\text{rhs}|)` time per step.
    """

    def __init__(self, net: Net):
        self.net = net
        self.cells: list[Symbol | None] = list(net.cells)
        self.ports: list[tuple[int, ...] | None] = list(net.ports)
        self.link = list(net.link)
        self.cell_of = [-1] * len(net.link)
        for idx, cports in enumerate(net.ports):
            for p in cports:
                self.cell_of[p] = idx
        self.active = list(net.active)
        self.loops = net.loops

    def is_principal(self, port: int) -> bool:
        """Whether ``port`` is the principal port of a live cell."""
        cell = self.cell_of[port]
        if cell < 0:
            return False
        cports = self.ports[cell]
        return cports is not None and cports[0] == port

    def connect(self, first_port: int, second_port: int) -> None:
        """Wire two ports and enqueue an active pair if both are principal."""
        self.link[first_port] = second_port
        self.link[second_port] = first_port
        if self.is_principal(first_port) and self.is_principal(second_port):
            c1, c2 = self.cell_of[first_port], self.cell_of[second_port]
            self.active.append((min(c1, c2), max(c1, c2)))

    def step(self) -> bool:
        """Rewrite the next live active pair; return ``False`` if none."""
        while self.active:
            c1, c2 = self.active.pop()
            s1, s2 = self.cells[c1], self.cells[c2]
            p1, p2 = self.ports[c1], self.ports[c2]
            if s1 is None or s2 is None or p1 is None or p2 is None:
                continue
            if self.link[p1[0]] != p2[0]:
                continue
            rhs = self.net.signature.rule(s1, s2)
            if rhs is None:
                raise KeyError((s1, s2))
            self.delete_cell(c1)
            self.delete_cell(c2)
            self.substitute(p1[1:] + p2[1:], rhs)
            return True
        return False

    def delete_cell(self, cell_idx: int) -> None:
        """Remove ``cell_idx`` from the live cell tables."""
        cports = self.ports[cell_idx]
        if cports is not None:
            for p in cports:
                self.cell_of[p] = -1
        self.cells[cell_idx] = None
        self.ports[cell_idx] = None

    def substitute(self, boundary: tuple[int, ...], rhs: Net) -> None:
        """Plug ``rhs`` onto ``boundary`` ports in place."""
        n_bound = len(boundary)
        if len(rhs.free) != n_bound:
            raise ValueError("RHS free port count does not match rule arity.")
        self.loops += rhs.loops
        base_port = len(self.link)
        n_new_ports = len(rhs.link) - n_bound
        self.link.extend([-1] * n_new_ports)
        self.cell_of.extend([-1] * n_new_ports)
        for rsym, rcports in zip(rhs.cells, rhs.ports):
            new_cell = len(self.cells)
            mapped = tuple(base_port + (rp - n_bound) for rp in rcports)
            self.cells.append(rsym)
            self.ports.append(mapped)
            for hp in mapped:
                self.cell_of[hp] = new_cell
        for rq, rr in enumerate(rhs.link):
            if rq < rr:
                self.splice_wire(boundary, base_port, rq, rr)

    def splice_wire(
        self, boundary: tuple[int, ...], base_port: int, rq: int, rr: int
    ) -> None:
        """Splice a single wire ``(rq, rr)`` of ``rhs`` into the host net."""
        n_bound = len(boundary)
        if rq >= n_bound:
            self.connect(
                base_port + (rq - n_bound), base_port + (rr - n_bound))
        elif rr >= n_bound:
            u = self.link[boundary[rq]]
            self.connect(u, base_port + (rr - n_bound))
        else:
            bp1, bp2 = boundary[rq], boundary[rr]
            u, v = self.link[bp1], self.link[bp2]
            if u == bp2:
                self.loops += 1
            else:
                self.connect(u, v)

    def to_net(self) -> Net:
        """Compact live cells and ports into a clean :class:`Net`."""
        n_free = len(self.net.free)
        port_map = {p: p for p in range(n_free)}
        cursor = n_free
        new_cells: list[Symbol] = []
        for sym, cports in zip(self.cells, self.ports):
            if sym is not None and cports is not None:
                new_cells.append(sym)
                for old_p in cports:
                    port_map[old_p] = cursor
                    cursor += 1
        new_link = [0] * cursor
        for old_p, new_p in port_map.items():
            new_link[new_p] = port_map[self.link[old_p]]
        return Net(
            self.net.signature, new_cells, new_link, loops=self.loops)


def int_port_and_sink() -> tuple[interaction.Ty, interaction.Ty]:
    """Return ``(port_ty, sink_ty)`` in ``Int(python.additive.Function)``."""
    func = additive.Function
    state_ty, empty_ty = func.ob(tuple), func.ob()
    int_ty = interaction.Ty[func.ob]
    return int_ty(state_ty, state_ty), int_ty(state_ty, empty_ty)


def int_cell(symbol: Symbol | str) -> interaction.Diagram:
    """
    Return the ``Int(python.additive.Function)`` morphism of a single
    combinator cell ``γ``, ``δ`` or ``ε``.

    Each port has type ``T(X, X)`` with ``X = additive.Function.ob(tuple)``,
    followed by a sink ``T(X, O)`` for ``"erased"`` and ``"undefined"``.
    """
    sym = Symbol.cast(symbol)
    if sym not in UNLABELLED_COMBINATORS:
        raise ValueError(f"Unsupported combinator cell {sym!r}.")
    func = additive.Function
    state_ty = func.ob(tuple)
    int_ty = interaction.Ty[func.ob]
    int_diag: Any = interaction.Diagram[func]
    port_ty, sink_ty = int_port_and_sink()
    if sym == Symbol("ε"):
        inside_eps = func(
            lambda ctx, tag=0: (("erased",), 1), state_ty, state_ty ** 2)
        return int_diag(inside_eps, int_ty(), port_ty @ sink_ty)
    is_delta = sym.name == "δ"

    def step(ctx: Context, tag: int = 0) -> tuple[Any, int]:
        g_stack, d_stack = ctx
        if tag == 2:
            stack = d_stack if is_delta else g_stack
            if not stack:
                return (("undefined",), 3)
            new_ctx = (
                (g_stack, d_stack[:-1]) if is_delta
                else (g_stack[:-1], d_stack))
            return (new_ctx, stack[-1])
        aux = 2 - tag
        new_ctx = (
            (g_stack, (*d_stack, aux)) if is_delta
            else ((*g_stack, aux), d_stack))
        return (new_ctx, 0)

    inside = func(step, state_ty ** 3, state_ty ** 4)
    return int_diag(inside, int_ty(), port_ty @ port_ty @ port_ty @ sink_ty)


def int_path(
    diagram: interaction.Diagram, port: int, context: Context = ((), ()),
) -> PathResult:
    """
    Evaluate an ``Int(python.additive.Function)`` net morphism produced by
    :meth:`Net.to_int` on a token entering free ``port`` with ``context``.
    """
    inside: Any = diagram.inside
    n_free = len(inside.dom)
    if not 0 <= port < n_free:
        raise ValueError(f"Invalid free port {port} for {n_free} ports.")
    out, out_tag = inside(context, n_free - 1 - port)
    return out[0] if out_tag == n_free else (out_tag, out)


def combinators(labels: Iterable[Any] | int | None = None) -> Signature:
    r"""
    The symmetric interaction combinators :math:`\gamma, \delta, \varepsilon`
    :cite:p:`Lafont97`, optionally with a family of labelled duplicators
    :math:`\delta_i` for ``i`` in ``labels``.

    Two cells with the same symbol and the same label annihilate; any other
    pair commutes.

    Parameters:
        labels : Optional iterable of allowed ``δ`` labels, or an integer ``n``
            for ``range(n)``. When ``None``, ``δ`` is unlabelled by default.

    Example
    -------
    Collapsing distinct ``δ`` labels changes the normal form of nested
    duplicators:

    >>> d0, d1 = Symbol('δ', 0), Symbol('δ', 1)
    >>> wires = [
    ...     ((0, 0), (1, 0)),
    ...     ((0, 1), (2, 0)), ((0, 2), (3, 0)),
    ...     ((1, 1), (4, 0)), ((1, 2), (5, 0)),
    ...     (0, (2, 1)), (1, (2, 2)), (2, (3, 1)), (3, (3, 2)),
    ...     (4, (4, 1)), (5, (4, 2)), (6, (5, 1)), (7, (5, 2)),
    ... ]
    >>> nf_lab = Net.from_wires(
    ...     combinators(2), [d0, d1, d1, d1, d0, d0], wires).reduce()
    >>> nf_unlab = Net.from_wires(combinators(), ['δ'] * 6, wires).reduce()
    >>> assert nf_lab.link != nf_unlab.link
    """
    allowed: tuple[Any, ...] | None = (
        None if labels is None
        else tuple(range(labels)) if isinstance(labels, int)
        else tuple(labels)
    )
    allowed_set = None if allowed is None else set(allowed)

    def arity(sym: Symbol) -> int:
        if sym == Symbol("γ"):
            return 2
        if sym == Symbol("ε"):
            return 0
        if sym.name == "δ" and (
            allowed_set is None or sym.label in allowed_set
        ):
            return 2
        raise KeyError(sym)

    symbols = (
        (Symbol("γ"), Symbol("δ"), Symbol("ε"))
        if allowed is None
        else (Symbol("γ"), *(Symbol("δ", lab) for lab in allowed), Symbol("ε"))
    )
    expr = "combinators()" if allowed is None else f"combinators({allowed!r})"
    cache: dict[tuple[Symbol, Symbol], Net] = {}

    def rule(s1: Symbol, s2: Symbol) -> Net:
        m, n = arity(s1), arity(s2)
        if (s1, s2) not in cache:
            if s1 == s2:
                ann_wires: list[tuple[Endpoint, Endpoint]] = [
                    (i, m + i) for i in range(m)
                ]
                cache[s1, s2] = Net.from_wires(sig, (), ann_wires)
            else:
                cells = [s2] * m + [s1] * n
                com_wires: list[tuple[Endpoint, Endpoint]] = [
                    *[(i, (i, 0)) for i in range(m)],
                    *[(m + j, (m + j, 0)) for j in range(n)],
                    *[
                        ((i, 1 + j), (m + j, 1 + i))
                        for i in range(m)
                        for j in range(n)
                    ],
                ]
                cache[s1, s2] = Net.from_wires(sig, cells, com_wires)
        return cache[s1, s2]

    sig = Signature(arity, rule, symbols=symbols, expr=expr)
    return sig
