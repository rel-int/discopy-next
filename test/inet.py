from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from pytest import mark, raises

from discopy.frobenius import Box, Hypergraph as H, Spider, Ty
from discopy.inet import Net, Signature, Symbol, combinators


def random_net_strategy(signature: Signature, max_cells: int = 12):
    """Sample a random valid Net over a given Signature."""
    symbols = signature.symbols

    @st.composite
    def strategy(draw):
        n_cells = draw(st.integers(0, max_cells))
        cells = [draw(st.sampled_from(symbols)) for _ in range(n_cells)]
        cell_ports = sum(1 + signature.arity(s) for s in cells)
        # Total port count must be even so a fixpoint-free involution exists.
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
    with raises(KeyError):
        sig.arity("unknown")
    with raises(ValueError):
        Net(sig, ["γ"], [1, 0])
    with raises(ValueError):
        Net(sig, [], [0, 1])
    with raises(ValueError):
        Net(sig, [], [1, 0], loops=-1)
    with raises(ValueError):
        Net.from_wires(sig, ["γ"], [(0, (0, 0))])


def test_custom_signature_symmetric_lookup():
    arities = {"a": 1, "b": 1}
    rhs = Net.from_wires(Signature(arities), [], [(0, 1)])
    sig = Signature(arities, {("a", "b"): rhs})
    net = Net.from_wires(
        sig, ["b", "a"], [((0, 0), (1, 0)), (0, (0, 1)), (1, (1, 1))])
    reduced = net.reduce()
    assert reduced == Net.from_wires(sig, [], [(0, 1)])
    assert eval(repr(reduced)) == reduced

    partial = Signature(arities, {})
    stuck = Net.from_wires(
        partial, ["a", "b"], [((0, 0), (1, 0)), (0, (0, 1)), (1, (1, 1))])
    with raises(KeyError):
        stuck.reduce()


def test_from_hypergraph_errors():
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
    # Two independent γ-γ active pairs, plus one active pair created by
    # splicing auxiliary ports of the first pair.
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

    # A broken reducer that reduces only the initial active pair and drops the
    # newly created (δ, δ) active pair leaves two δ cells behind.
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
    """Slice 3 check: collapsing δ_0 and δ_1 to equal labels alters normal form."""
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
