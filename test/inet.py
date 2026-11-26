from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from pytest import mark, raises

from discopy.frobenius import Box, Hypergraph as H, Spider, Ty
from discopy.inet import (
    Net,
    Rewriter,
    Signature,
    Symbol,
    combinators,
    int_cell,
    int_path,
)


def random_net_strategy(signature: Signature, max_cells: int = 12):
    """Sample a random valid Net over a given Signature."""
    symbols = signature.symbols

    @st.composite
    def strategy(draw):
        n_cells = draw(st.integers(0, max_cells))
        cells = [draw(st.sampled_from(symbols)) for _ in range(n_cells)]
        cell_ports = sum(1 + signature.arity(s) for s in cells)
        extra = draw(st.integers(0, 3))
        n_free = 2 * extra + (cell_ports % 2)
        total = n_free + cell_ports
        perm = draw(st.permutations(range(total)))
        link = [0] * total
        for i in range(0, total, 2):
            u, v = perm[i], perm[i + 1]
            link[u], link[v] = v, u
        loops = draw(st.integers(0, 2))
        return Net(signature, cells, link, loops=loops)

    return strategy()


def context_strategy(max_height: int = 6):
    """Sample a pair of stacks over {1, 2} of height at most max_height."""
    stack = st.lists(st.sampled_from((1, 2)), max_size=max_height).map(tuple)
    return st.tuples(stack, stack)


def test_symbol_and_validation():
    gamma = Symbol("γ")
    d0 = Symbol("δ", 0)
    assert Symbol.cast("γ") == gamma and Symbol.cast(d0) == d0
    assert str(gamma) == "γ" and str(d0) == "δ_0"
    assert d0.with_label(1) == Symbol("δ", 1)
    assert eval(repr(gamma)) == gamma and eval(repr(d0)) == d0
    with raises(TypeError):
        Symbol.cast(42)  # type: ignore[arg-type]

    sig = combinators()
    assert sig == combinators() and sig != "not-a-sig"
    with raises(KeyError):
        sig.arity("unknown")
    with raises(ValueError):
        Net(sig, ["γ"], [1, 0])
    with raises(ValueError):
        Net(sig, [], [0, 1])
    with raises(ValueError):
        Net(sig, [], [1, 2, 0, 3])
    with raises(ValueError):
        Net(sig, [], [1, 0], loops=-1)
    with raises(ValueError):
        Net.from_wires(sig, ["γ"], [(0, (0, 0))])
    with raises(ValueError):
        Net.from_wires(sig, [], [(-1, 0)])
    with raises(ValueError):
        Net.from_wires(sig, ["ε"], [((0, 2), 0)])
    with raises(ValueError):
        Net.from_wires(sig, [], [(0, 0)])

    wire = Net.from_wires(sig, [], [(0, 1)])
    assert wire != "not-a-net" and len({wire, wire}) == 1
    with raises(ValueError):
        wire.swap_halves(1, 2)
    with raises(ValueError):
        wire.to_hypergraph(n_dom=3)


def test_custom_signature_and_closure():
    base = Signature({"a": 1, "b": 1, "c": 2})
    assert eval(repr(base)) == base and base != combinators()
    rhs_wire = Net.from_wires(base, [], [(0, 1)])
    sig = Signature({"a": 1, "b": 1}, {("a", "b"): rhs_wire})
    assert eval(repr(sig)) == sig
    assert sig.rule("a", "b") == rhs_wire
    net = Net.from_wires(
        sig, ["b", "a"], [((0, 0), (1, 0)), (0, (0, 1)), (1, (1, 1))])
    reduced = net.reduce()
    assert reduced == Net.from_wires(sig, [], [(0, 1)])
    assert eval(repr(reduced)) == reduced

    gen_rhs = Net.from_wires(
        base, ["c", "c"],
        [((0, 0), (1, 0)), ((0, 1), (1, 1)), (0, (0, 2)), (1, (1, 2))])
    sig_gen = Signature(
        {"a": 1, "b": 1, "c": 2},
        {("a", "b"): gen_rhs, ("c", "c"): Net.from_wires(
            base, [], [(0, 2), (1, 3)])},
        symbols=("a", "b"),
    )
    assert len(sig_gen.hypergraph_rules(("a", "b"))) == 2

    partial = Signature({"a": 1, "b": 1})
    with raises(KeyError):
        partial.hypergraph_rule("a", "b")
    stuck = Net.from_wires(
        partial, ["a", "b"], [((0, 0), (1, 0)), (0, (0, 1)), (1, (1, 1))])
    with raises(KeyError):
        stuck.reduce()

    bad_sig = Signature({"a": 1}, {("a", "a"): Net(base)})
    bad_net = Net.from_wires(
        bad_sig, ["a", "a"], [((0, 0), (1, 0)), (0, (0, 1)), (1, (1, 1))])
    with raises(ValueError):
        bad_net.reduce()

    rw = Rewriter(stuck)
    rw.active.append((0, 1))
    rw.delete_cell(0)
    assert not rw.step()


def test_from_hypergraph_and_custom_roundtrip():
    x = Ty("x")
    with raises(ValueError):
        Net.from_hypergraph(Spider(1, 2, x).to_hypergraph())
    with raises(ValueError):
        Net.from_hypergraph(Box("f", x @ x, x).to_hypergraph())
    with raises(ValueError):
        Net.from_hypergraph(
            Box("γ", x, x).to_hypergraph(), signature=combinators())
    inconsistent = (
        Box("f", x, x).to_hypergraph() @ Box("f", x, x @ x).to_hypergraph())
    with raises(ValueError):
        Net.from_hypergraph(inconsistent)

    custom_hg = Box("f", x, x @ x @ x).to_hypergraph()
    decoded = Net.from_hypergraph(custom_hg)
    assert decoded.to_hypergraph() == custom_hg


def test_lafont_fig_11():
    """Lafont (1997), Fig. 11 reduces to two parallel wires in 5 steps."""
    x = Ty("x")
    gamma, delta = (Box(name, x, x @ x).to_hypergraph() for name in "γδ")
    fig11_hg = (
        H.caps(x, x) >> delta @ delta >> x @ H.cups(x, x) @ x
        >> gamma.transpose() >> delta >> gamma @ gamma)
    net = Net.from_hypergraph(fig11_hg, combinators())
    steps = list(net.normalize())
    assert len(steps) == 5
    assert steps[-1] == net.reduce()
    assert net.reduce().to_hypergraph() == H.caps(x, x) @ H.caps(x, x)


def test_non_terminating_step_bound():
    """γ facing δ with auxiliary ports looped back never terminates."""
    sig = combinators()
    omega = Net.from_wires(sig, ["γ", "δ"], [
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((0, 2), (1, 2)),
    ])
    with raises(RuntimeError):
        omega.reduce(max_steps=10)


@settings(
    max_examples=60, deadline=None,
    suppress_health_check=[HealthCheck.too_slow])
@given(random_net_strategy(combinators(), max_cells=12))
def test_hypergraph_roundtrip(net: Net):
    rebuilt = Net.from_hypergraph(net.to_hypergraph(), net.signature)
    assert rebuilt == net
    assert rebuilt.to_hypergraph() == net.to_hypergraph()
    assert Net.from_hypergraph(net.to_hypergraph()).to_hypergraph() == (
        net.to_hypergraph())


@settings(
    max_examples=60, deadline=None,
    suppress_health_check=[HealthCheck.too_slow])
@given(random_net_strategy(combinators(), max_cells=10))
def test_dpo_law_combinators(net: Net):
    try:
        reduced = net.reduce(max_steps=40)
    except RuntimeError:
        assume(False)
        return
    rules = net.signature.hypergraph_rules(net.cells)
    dpo_nf = net.to_hypergraph().normal_form(*rules)
    assert Net.from_hypergraph(dpo_nf) == reduced
    assert Net.from_hypergraph(dpo_nf).to_hypergraph() == (
        reduced.to_hypergraph())


def test_dropped_active_pair_breaks_dpo_law():
    """Slice 1 mutation check: dropping one active pair fails the DPO law."""
    sig = combinators()
    net = Net.from_wires(sig, ["γ", "γ", "δ", "δ"], [
        ((0, 0), (1, 0)),
        ((0, 1), (2, 0)),
        ((1, 1), (3, 0)),
        (0, (0, 2)),
        (1, (1, 2)),
        (2, (2, 1)),
        (3, (2, 2)),
        (4, (3, 1)),
        (5, (3, 2)),
    ])
    rules = sig.hypergraph_rules(net.cells)
    dpo_nf = net.to_hypergraph().normal_form(*rules)
    assert net.reduce().to_hypergraph() == dpo_nf

    one_step = net.step()
    assert one_step is not None and len(one_step.active) == 1
    dropped = Net(sig, one_step.cells, one_step.link, loops=one_step.loops)
    assert dropped.to_hypergraph() != dpo_nf


@mark.parametrize("n_labels", [2, 3])
@settings(
    max_examples=50, deadline=None,
    suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_dpo_law_labelled_duplicators(n_labels: int, data):
    sig = combinators(n_labels)
    net = data.draw(random_net_strategy(sig, max_cells=9))
    try:
        reduced = net.reduce(max_steps=40)
    except RuntimeError:
        assume(False)
        return
    rules = sig.hypergraph_rules(net.cells)
    dpo_nf = net.to_hypergraph().normal_form(*rules)
    assert Net.from_hypergraph(dpo_nf, sig) == reduced
    assert Net.from_hypergraph(dpo_nf).to_hypergraph() == (
        reduced.to_hypergraph())


def test_collapsing_labels_changes_normal_form():
    """Collapsing δ_0 and δ_1 to equal labels alters the normal form."""
    sig_labelled = combinators(2)
    d0, d1 = Symbol("δ", 0), Symbol("δ", 1)
    wires = [
        ((0, 0), (1, 0)),
        ((0, 1), (2, 0)),
        ((0, 2), (3, 0)),
        ((1, 1), (4, 0)),
        ((1, 2), (5, 0)),
        (0, (2, 1)), (1, (2, 2)),
        (2, (3, 1)), (3, (3, 2)),
        (4, (4, 1)), (5, (4, 2)),
        (6, (5, 1)), (7, (5, 2)),
    ]
    labelled = Net.from_wires(sig_labelled, [d0, d1, d1, d1, d0, d0], wires)
    collapsed = Net.from_wires(combinators(), ["δ"] * 6, wires)

    nf_labelled = labelled.reduce()
    nf_collapsed = collapsed.reduce()

    assert nf_labelled.cells == [] and nf_collapsed.cells == []
    assert nf_labelled != nf_collapsed
    assert nf_labelled.link == [4, 6, 5, 7, 0, 2, 1, 3]
    assert nf_collapsed.link == [4, 5, 6, 7, 0, 1, 2, 3]

    rules_l = sig_labelled.hypergraph_rules(labelled.cells)
    assert nf_labelled.to_hypergraph() == (
        labelled.to_hypergraph().normal_form(*rules_l))


def test_token_machine_basics_and_errors():
    sig = combinators()
    wire = Net.from_wires(sig, [], [(0, 1)])
    assert not wire.has_vicious_circle()
    ctx = ((1, 2), (2,))
    assert wire.path(0, ctx) == (1, ctx) == int_path(wire.to_int(), 0, ctx)
    assert wire.path(1, ctx) == (0, ctx) == int_path(wire.to_int(), 1, ctx)

    erased = Net.from_wires(sig, ["γ", "ε"], [
        ((0, 0), (1, 0)), (0, (0, 1)), (1, (0, 2)),
    ])
    assert erased.path(0) == "erased" == int_path(erased.to_int(), 0)
    assert erased.reduce().path(0) == "erased"

    single_g = Net.from_wires(sig, ["γ"], [
        (0, (0, 0)), (1, (0, 1)), (2, (0, 2)),
    ])
    assert single_g.path(0) == "undefined" == int_path(single_g.to_int(), 0)
    assert single_g.path(0, ((2,), ())) == (2, ((), ()))
    assert int_path(single_g.to_int(), 0, ((2,), ())) == (2, ((), ()))

    vicious = Net.from_wires(sig, ["γ"], [((0, 0), (0, 1)), (0, (0, 2))])
    assert vicious.has_vicious_circle()

    with raises(ValueError):
        wire.path(2)
    with raises(ValueError):
        int_path(wire.to_int(), 2)
    with raises(ValueError):
        int_cell(Symbol("δ", 0))
    labelled_net = Net.from_wires(
        combinators(1), [Symbol("δ", 0)],
        [(0, (0, 0)), (1, (0, 1)), (2, (0, 2))])
    with raises(ValueError):
        labelled_net.path(0)
    with raises(ValueError):
        labelled_net.to_int()


@settings(
    max_examples=60, deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(net=random_net_strategy(combinators(), max_cells=8), data=st.data())
def test_int_law_and_reduction_invariance(net: Net, data):
    """Fast path == Int(additive.Function) composite == path after reduce."""
    assume(bool(net.free))
    try:
        reduced = net.reduce(max_steps=30)
    except RuntimeError:
        assume(False)
        return
    assume(not net.has_vicious_circle() and not reduced.has_vicious_circle())

    port = data.draw(st.sampled_from(net.free))
    ctx = data.draw(context_strategy(max_height=6))
    diagram = net.to_int()

    expected = net.path(port, ctx)
    assert int_path(diagram, port, ctx) == expected
    assert reduced.path(port, ctx) == expected


def test_swapping_pop_branches_breaks_int_law():
    """Slice 2 mutation check: swapping 1 and 2 on pop fails the Int law."""
    sig = combinators()
    net = Net.from_wires(sig, ["γ", "γ"], [
        ((0, 0), (1, 0)),
        (0, (0, 1)), (1, (0, 2)),
        (2, (1, 1)), (3, (1, 2)),
    ])

    def buggy_path(host: Net, port: int, context=((), ())):
        g_stack, d_stack = list(context[0]), list(context[1])
        cell_of, slot_of = host.port_table()
        current = host.link[port]
        while current >= len(host.free):
            cell_idx, slot = cell_of[current], slot_of[current]
            sym = host.cells[cell_idx]
            if sym == Symbol("ε"):
                return "erased"
            stack = d_stack if sym.name == "δ" else g_stack
            if slot == 0:
                if not stack:
                    return "undefined"
                branch = 3 - stack.pop()
                current = host.link[host.ports[cell_idx][branch]]
            else:
                stack.append(slot)
                current = host.link[host.ports[cell_idx][0]]
        return (current, (tuple(g_stack), tuple(d_stack)))

    from_int = int_path(net.to_int(), 0, ((), ()))
    assert net.path(0, ((), ())) == from_int == (2, ((), ()))
    assert buggy_path(net, 0, ((), ())) != from_int
