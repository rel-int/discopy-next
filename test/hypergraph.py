import random

from pytest import raises

from discopy.hypergraph import *
from discopy.frobenius import (
    Ty, Box, Cap, Cup, Diagram, Nat, Spider, Hypergraph as H)

def test_pushout():
    with raises(ValueError):
        pushout(1, 1, [0], [0, 1])


def test_Hypergraph_init():
    x, y = map(Ty, "xy")
    with raises(ValueError):
        H(x, x, (), ())
    with raises(AxiomError):
        H(x, y, (), ((0,), (), (0,)))


def test_Hypergraph_str():
    x, y = map(Ty, "xy")
    assert str(H.swap(x, y)) == "Swap(x, y)"
    assert str(H.spiders(1, 0, x @ y)) == "Cospan(x @ y, [0, 1], [])"


def test_Hypergraph_repr():
    x, y = map(Ty, "xy")
    assert repr(H.spiders(1, 0, x @ y))\
        == "hypergraph.Hypergraph[frobenius.Diagram]("\
           "dom=frobenius.Ty(frobenius.Wire('x'), frobenius.Wire('y')), "\
           "cod=frobenius.Ty(), boxes=(), wires=((0, 1), (), ()))"


def test_Hypergraph_hash():
    x, y = map(Ty, "xy")
    assert hash(H.id(x @ y)) == hash(H.id(x) @ H.id(y))


def test_Hypergraph_eq_sees_the_type_of_a_scalar_spider():
    from discopy.frobenius import Equation
    x, y = map(Ty, "xy")
    assert H.spiders(0, 0, x) != H.spiders(0, 0, y)
    assert hash(H.spiders(0, 0, x)) != hash(H.spiders(0, 0, y))
    assert H.spiders(0, 0, x @ y) == H.spiders(0, 0, y @ x)
    assert hash(H.spiders(0, 0, x @ y)) == hash(H.spiders(0, 0, y @ x))
    f = Box('f', x, x)
    assert not Equation(Spider(0, 0, x) @ f, Spider(0, 0, y) @ f)


def test_Hypergraph_then():
    x, y = map(Ty, "xy")
    with raises(AxiomError, match="Expected y, got x"):
        H.id(x) >> H.id(y)


def test_Hypergraph_tensor():
    Id = H.id
    assert Id().tensor(Id(), Id()) == Id().tensor() == Id()


def test_Hypergraph_getitem():
    with raises(NotImplementedError):
        H.spiders(1, 2, Ty('x'))[0]


def test_Hypergraph_bijection():
    with raises(ValueError, match="not bijective"):
        H.spiders(1, 2, Ty('x')).bijection


def test_Hypergraph_make_causal_does_not_assume_topological_order():
    x, y, z = map(Ty, "xyz")
    f, g = Box('f', x, y), Box('g', y, z)
    h = H(
        x, z, (g, f),
        ((0,), (((1,), (2,)), ((0,), (1,))), (2,)))

    assert h.is_left_monogamous
    assert h.is_acyclic
    assert not h.is_topologically_ordered
    assert not h.is_causal
    assert h.make_causal().is_causal
    assert h.topological_order().is_causal


def test_Hypergraph_depth_counts_states_and_effects():
    from discopy import symmetric
    x = Ty('x')
    state, effect = Box('f', Ty(), x), Box('g', x, Ty())
    scalar, f = Box('s', Ty(), Ty()), Box('f', x, x)
    for diagram, depth in [
            (state, 1), (effect, 1), (scalar, 1), (state >> effect, 2),
            (f @ (state >> effect), 2), (f >> f, 2), (f.trace(), 3)]:
        assert diagram.depth() == len(diagram.foliation()) == depth
    sx = symmetric.Ty('x')
    assert symmetric.Box('f', symmetric.Ty(), sx).depth() == 1
    assert symmetric.Box('f', sx, sx).trace().depth() == 1


def test_Hypergraph_draw_is_reproducible(tmp_path):
    x = Ty('x')
    diagram = Box('f', x, x).to_hypergraph()
    first, second = tmp_path / "first.svg", tmp_path / "second.svg"
    state = random.getstate()

    diagram.draw(seed=42, path=first)
    diagram.draw(seed=42, path=second)

    assert first.read_bytes() == second.read_bytes()
    assert random.getstate() == state


def test_Hypergraph_simplify_bubble_size():
    x = Ty('x')
    f = Box('f', Ty(), x)
    g, h = Box('g', x, x), Box('h', x, Ty())
    diagram = f >> (g >> h).bubble()
    hypergraph = diagram.to_hypergraph()
    interchanged = hypergraph.interchange(0, 1)

    assert len(diagram) == 2 and diagram.size == 4
    assert interchanged.to_diagram().size > diagram.size
    assert interchanged.simplify() == hypergraph


def test_Hypergraph_rotate():
    from discopy import compact

    assert H.id() == \
           H.id().rotate(left=False).rotate(left=True)

    x, y = map(compact.Ty, "xy")
    for box in (compact.Box('f', compact.Ty(), x),
                compact.Box('g', x, y),
                compact.Box('h', x @ y, x)):
        graph = box.to_hypergraph()
        assert (graph.r.dom, graph.r.cod) == (graph.cod.r, graph.dom.r)
        assert (graph.l.dom, graph.l.cod) == (graph.cod.l, graph.dom.l)
        assert graph.l.r == graph == graph.r.l
        assert graph.rotate() == box.rotate().to_hypergraph()


def test_Box():
    box = Box('box', Ty('x'), Ty('y')).to_hypergraph()
    assert box == box and box == box @ H.id() and box != 1


def test_AxiomError():
    x, y = map(Ty, "xy")
    with raises(AxiomError):
        H.cups(x @ y, x @ y)
    with raises(AxiomError):
        H.caps(x @ y, x @ y)


def test_non_self_dual():
    from discopy import compact
    x = compact.Ty('x')
    # A non-self-dual cap round-trips through its hypergraph, see issue #390.
    assert compact.Equation(
        compact.Cap(x.r, x).to_hypergraph().to_diagram(), compact.Cap(x.r, x))


def test_non_adjoint_wire():
    from discopy import compact
    x = compact.Ty('x')
    K = compact.Hypergraph
    # A cap with non-adjoint legs (x, x) is rejected at construction.
    with raises(AxiomError):
        K(compact.Ty(), x @ x, (), ((), (), (0, 0)))
    # Adjoint cups and caps, and self-dual frobenius caps, are fine.
    assert K.cups(x, x.r).wires == ((0, 0), (), ())
    assert K.caps(x.r, x).wires == ((), (), (0, 0))
    assert Cap(Ty('x'), Ty('x')).to_hypergraph() == H.spiders(0, 2, Ty('x'))


def test_cups_and_caps_are_checked_in_port_order():
    """ The legs of a cup-shaped or cap-shaped spider are checked in port
    order, the order :meth:`Hypergraph.make_monogamous` reads them in, so a
    rigid cap on ``x.l @ x`` is refused when it is built. """
    from discopy import rigid
    from discopy.hypergraph import Hypergraph
    x, K = rigid.Ty('x'), Hypergraph[rigid.Diagram]
    assert K(rigid.Ty(), x @ x.l, (), ((), (), (0, 0))).to_diagram()\
        == rigid.Cap(x, x.l)
    assert K(x.l @ x, rigid.Ty(), (), ((0, 0), (), ())).to_diagram()\
        == rigid.Cup(x.l, x)
    for dom, cod in [(rigid.Ty(), x.l @ x), (x @ x.l, rigid.Ty())]:
        with raises(AxiomError, match="x is not the left adjoint of x.l"):
            K(dom, cod, (), ((0, 0) if dom else (), (), (0, 0) if cod else ()))


def test_rotation_is_the_transpose():
    """ Decision 7 of #56: a rotated box has the hypergraph of its
    transpose, as it has the map of its transpose, at every level from
    rigid up. """
    from discopy import rigid, pivotal, ribbon, compact, frobenius
    from discopy.hypergraph import Hypergraph
    for module in (rigid, pivotal, ribbon, compact, frobenius):
        x, y = module.Ty('x'), module.Ty('y')
        g = module.Box('g', x, y @ x)
        assert g.r.to_hypergraph() == g.transpose().to_hypergraph()\
            == Hypergraph[module.Diagram].from_box(g.r)\
            == g.to_hypergraph().r
        for rotated in (g.r, g.l, g.r.r, g.l.l, g.r.l):
            assert rotated.to_map().to_hypergraph() == rotated.to_hypergraph()
    for module in (rigid, compact, frobenius):
        x, y = module.Ty('x'), module.Ty('y')
        g = module.Box('g', x, y @ x)
        assert g.l.to_hypergraph()\
            == g.transpose(left=True).to_hypergraph()
        assert g.r.r.to_hypergraph()\
            == g.transpose().transpose().to_hypergraph()


def test_spider_producers_and_consumers():
    from discopy import compact
    K = compact.Hypergraph
    x, e = compact.Ty('x'), compact.Ty()
    f, g, h = (
        compact.Box('f', e, x), compact.Box('g', e, x), compact.Box('h', x, e))
    # A spider with two producers (f, g) and a consumer (h) is a genuine
    # merge, not a cup, so the adjointness check should not apply even
    # though x is not self-dual, see issue #390.
    merge = K(e, e, (f, g, h),
              ((), (((), (0,)), ((), (0,)), ((0,), ())), ()))
    assert merge.spider_wires == [({0, 1}, {2})]

    # Symmetrically, a spider with a producer and two consumers is a
    # genuine copy, not a cap.
    g, h = compact.Box('g', x, e), compact.Box('h', x, e)
    copy = K(e, e, (f, g, h),
             ((), (((), (0,)), ((0,), ()), ((0,), ())), ()))
    assert copy.spider_wires == [({0}, {1, 2})]


def test_cups():
    x = Ty('x')
    assert issubclass(H, DaggerCategory)
    assert H.cups(x, x).make_monogamous().dagger()\
        == H.caps(x, x).make_monogamous()
    assert H.caps(x, x).make_monogamous().dagger()\
        == H.cups(x, x).make_monogamous()
    assert H.caps(x, x).to_diagram() == Cap(x, x)


def test_Hypergraph_is_boundary_connected():
    x, y, z = map(Ty, "xyz")
    f = Box('f', x, x).to_hypergraph()
    assert f.is_boundary_connected

    g = Box('g', x @ z, y @ z).to_hypergraph()
    assert g.trace().is_boundary_connected

    assert not f.trace().is_boundary_connected

    scalar = Box('s', Ty(), Ty()).to_hypergraph()
    assert not (scalar @ scalar).is_boundary_connected
    assert not H.spiders(0, 0, x).is_boundary_connected

    assert H.id(Ty()).is_boundary_connected
    assert H.id(x).is_boundary_connected


def test_Hypergraph_eq_fast_path_trace():
    """ Traces are cyclic but boundary-connected, so they should use the
    fast canonical-form path rather than falling back to VF2. """
    x, y, z = map(Ty, "xyz")
    g = Box('g', x @ z, y @ z).to_hypergraph()
    trace = g.trace()

    assert trace.is_acyclic is False
    assert trace.is_fast_eligible

    shuffled = trace.interchange(0, min(1, len(trace.boxes) - 1))
    assert trace == shuffled
    assert hash(trace) == hash(shuffled)


def test_Hypergraph_eq_fallback_scalars_and_empty_boundary():
    """ Hypergraphs that are not boundary-connected (scalars, closed
    diagrams) must still compare correctly via the VF2 fallback. """
    scalar_a = Box('s', Ty(), Ty()).to_hypergraph()
    scalar_b = Box('s', Ty(), Ty()).to_hypergraph()
    assert not (scalar_a @ scalar_b).is_fast_eligible
    assert scalar_a @ scalar_b == scalar_b @ scalar_a

    x = Ty('x')
    f = Box('f', x, x).to_hypergraph()
    assert f.trace() == f.trace()


def test_Hypergraph_simplify_two_wires():
    x, y = Ty('x'), Ty('y')
    f = Box('f', Ty(), x @ y).to_hypergraph()
    g = Box('g', x @ y, Ty()).to_hypergraph()
    interchanged = (f >> g).interchange(0, 1)

    assert interchanged.to_diagram().size > (f >> g).to_diagram().size
    assert interchanged.simplify().to_diagram() == (f >> g).to_diagram()


def test_simplify():
    from discopy.markov import Box, Ty, Copy, Swap, Trace, Equation
    C, T, P = map(Ty, "CTP")
    linear, param_linear, add, placeholder = (
        Box('linear', T @ P, T),
        Box('param_linear', C, P),
        Box('add', T @ T, T),
        Box('placeholder', C, T),
    )
    residual_block = Trace(Trace(Copy(C) @ T @ P >> C @ C @ linear >> param_linear @ C @ T >> P @ placeholder @ T >> P @ Copy(T) @ T >> P @ T @ Swap(T, T) >> P @ add @ T >> Swap(P, T) @ T >> T @ Swap(P, T)))
    ref = Copy(C) >> param_linear @ C >> P @ placeholder >> P @ Copy(T) >> Swap(P, T) @ T >> linear @ T >> Swap(T, T) >> add
    simpl = residual_block.to_hypergraph().simplify().to_diagram()

    assert Equation(residual_block, ref) and Equation(ref, simpl)

    # to_diagram foliates, so simpl is the (tighter) foliation of ref rather
    # than its point-free, one-box-per-layer presentation.
    assert simpl == ref.foliation()


def test_parameterisation():
    from discopy import frobenius
    assert H == Hypergraph[frobenius.Diagram]
    assert H.category == frobenius.Diagram
    assert H.functor == frobenius.Functor == H.category.functor_factory
    assert H.ob == frobenius.Ty


def _naive_from_diagram(old):
    """ The pre-#623 implementation of :meth:`Hypergraph.from_diagram`,
    folding the image of each box with :meth:`Hypergraph.then` one at a
    time instead of gluing them in a single pass, kept here as an oracle
    against which the linear-time :meth:`Hypergraph.from_glued` is
    checked for regressions. """
    factory = H[type(old).ar]
    return factory.functor(
        ob_map=lambda typ: typ, ar_map=factory.from_box,
        dom=type(old), cod=factory)(old)


def test_Hypergraph_Nat():
    diagram = Box('f', Nat(1), Nat(2)) >> Spider(2, 1, Nat(1))
    assert diagram.to_hypergraph().to_diagram() == diagram
    assert H.spiders(1, 2, Nat(1)).to_diagram() == Spider(1, 2, Nat(1))
    assert H.cups(Nat(1), Nat(1)).to_diagram() == Cup(Nat(1), Nat(1))


def test_Hypergraph_to_diagram_offset():
    x = Ty('x')
    g = Box('g', Ty(), x)
    graph = H(Ty(), x, (g, ), ((), (((), (0, )), ), (0, )), (x, ), (3, ))
    assert graph.to_diagram() == g


def test_Hypergraph_cap_after_states():
    """ A cap goes beside the wire its first reader reads next to it, see
    #108 and #187. """
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', Ty(), x), Box('g', y @ x, x)
    for diagram in [
            x @ f >> Cap(x, x) @ x @ x,
            x @ f >> x @ Cap(x, x) @ x, y @ Cap(x, x) >> g @ x]:
        decoded = diagram.to_hypergraph().to_diagram()
        assert decoded in (diagram, diagram.foliation())
    # f @ Cap(x, x) whiskers the cap into a cospan after f, as h @ Swap(x, x)
    # gives a permutation after h, and decodes to them side by side.
    side_by_side = Diagram((Diagram.layer_factory(Ty(), f, Cap(x, x)), ),
                           Ty(), x @ x @ x)
    assert (f @ Cap(x, x)).to_hypergraph().to_diagram() == side_by_side


def test_Hypergraph_make_causal_without_spiders():
    """ A category with cups and caps but no merges bends its wires instead,
    e.g. to compute the depth of a transpose. """
    from discopy import compact
    x = compact.Ty('x')
    cup = compact.Cup(x, x.r)
    assert cup.to_hypergraph().make_causal().boxes == (cup, )
    transpose = compact.Box('f', x, x).transpose().to_hypergraph()
    assert transpose.make_causal().is_causal


def test_Hypergraph_from_glued():
    x = Ty('x')
    f, g = Box('f', x, x).to_hypergraph(), Box('g', x, x).to_hypergraph()
    assert H.from_glued(x, x, [(f, 0), (g, 0)]) == f >> g


def test_Hypergraph_from_diagram_closed_loop():
    """ A cap glued directly onto a cup leaves a closed loop: it must
    survive gluing as a scalar spider rather than vanish because it is
    never referenced by :attr:`Hypergraph.dom_wires`, ``box_wires`` or
    ``cod_wires``, see issue #623. """
    x = Ty('x')
    diagram = Cap(x, x) >> Cup(x, x)
    hypergraph = diagram.to_hypergraph()
    assert hypergraph.n_spiders == 1 and hypergraph.scalar_spiders == [0]
    assert not hypergraph.is_acyclic
    assert hypergraph == _naive_from_diagram(diagram)


def test_Hypergraph_from_diagram_matches_naive_composition():
    """ Gluing every box in one pass agrees with folding their images with
    :meth:`Hypergraph.then` one layer at a time, on diagrams that exercise
    a chain of swaps, states and effects, and boxes of different arity and
    coarity sharing a layer, see issue #623. """
    x = Ty('x')
    f = Box('f', x, x)
    state, effect = Box('s', Ty(), x), Box('e', x, Ty())
    split, merge = Box('p', x, x @ x), Box('m', x @ x, x)

    chain = Diagram.id(x @ x)
    for _ in range(6):
        chain = chain >> (f @ x) >> Diagram.swap(x, x)

    diagrams = [
        chain,
        state >> split >> merge >> effect,
        Diagram.id(x) @ split >> f @ merge,
        Cap(x, x) @ x >> x @ Cup(x, x),
    ]
    for diagram in diagrams:
        assert diagram.to_hypergraph() == _naive_from_diagram(diagram)


def test_Hypergraph_from_diagram_pinned_category():
    """ Calling :meth:`from_diagram` on a :class:`Hypergraph` pinned to a
    category other than the diagram's own must still pick the functor of
    the diagram's own category, i.e. ``factory.functor`` rather than
    ``cls.functor``: a :class:`symmetric.Hypergraph` has no ``Functor``
    that expands a :class:`Spider`, so using its functor would flatten the
    spider into an opaque box instead of pure wiring, per the cubic-dev-ai
    review of #644. """
    from discopy import symmetric
    x = Ty('x')
    spider = Spider(1, 2, x)
    result = symmetric.Hypergraph.from_diagram(spider)
    assert result.boxes == () and result == _naive_from_diagram(spider)
    assert result == spider.to_hypergraph()


def test_subclass_to_hypergraph():
    from discopy import frobenius
    from discopy.cat import factory

    @factory
    class Circuit(frobenius.Diagram):
        """ A frobenius diagram with hypergraphs of its own category. """

    class Gate(frobenius.Box, Circuit):
        """ A gate is a box in a circuit. """

    x = Ty('x')
    f, g = Gate('f', x, x), Gate('g', x, x)
    assert (f >> g).to_hypergraph().category == Circuit
    assert isinstance((f >> g).to_hypergraph().to_diagram(), Circuit)


def test_Hypergraph_cups_and_caps_adjoints():
    from discopy import rigid
    x, y = rigid.Ty('x'), rigid.Ty('y')
    R = Hypergraph[rigid.Diagram]
    for left, right in [(x.r, x), (x, x.l)]:
        assert rigid.Cap(left, right).to_hypergraph() == R.caps(left, right)
    for left, right in [(x, x.r), (x.l, x)]:
        assert rigid.Cup(left, right).to_hypergraph() == R.cups(left, right)
    for cups_or_caps, left, right in [
            (R.cups, x.r, x), (R.caps, x, x.r), (R.cups, x, x)]:
        with raises(AxiomError):
            cups_or_caps(left, right)
    with raises(AxiomError, match="x and y are not adjoints"):
        H.cups(Ty('x'), Ty('y'))


def test_cups_and_caps_decode_with_their_legs_in_scan_order():
    """ A cup's or cap's legs go in the order their readers sit in the scan,
    so decoding adds no swap the diagram did not have, see #139. """
    from fixtures.diagrams import bent_wires
    for diagram in bent_wires():
        assert diagram.to_hypergraph().to_diagram() == diagram


def test_to_diagram_routes_with_one_permutation():
    """ Each routing step of the decoder is one :class:`Permutation` layer
    rather than a staircase of swaps, so a permutation round-trips to itself
    and the plumbing of a decoded diagram never spans two consecutive layers,
    see discopy/discopy#539. """
    from discopy import symmetric, compact
    from fixtures.diagrams import random_diagram

    x, y, z, w = map(symmetric.Ty, "xyzw")
    perm = symmetric.Permutation(x @ y @ z, [2, 0, 1])
    g = symmetric.Box('g', z @ x @ y, w)
    assert perm.to_swaps().simplify() == perm
    assert (perm.to_swaps() >> g).simplify() == perm >> g

    for module in (symmetric, compact):
        for seed in range(10):
            diagram, functor = random_diagram(module, seed)
            decoded = diagram.to_hypergraph().to_diagram()
            assert functor(decoded) == functor(diagram)
            plumbing = [layer.is_plumbing for layer in decoded.inside]
            assert not any(a and b for a, b in zip(plumbing, plumbing[1:]))


def test_make_causal_traces_the_type_of_the_producer():
    from discopy import compact
    x = compact.Ty('x')
    graph = compact.Box('f', x @ x.r, x @ x.r).to_hypergraph().trace()
    assert graph.to_diagram().to_hypergraph() == graph


def test_trace_needs_traced_pivotal():
    from discopy import rigid, pivotal, compact
    x = rigid.Ty('x')
    graph = rigid.Box('f', x @ x, x @ x).to_hypergraph()
    for left in (True, False):
        with raises(AxiomError):
            graph.trace(left=left)
    for module in (pivotal, compact):
        x = module.Ty('x')
        graph = module.Box('f', x @ x, x @ x).to_hypergraph()
        assert graph.trace().cod == graph.trace(left=True).cod == x


def test_cups_and_caps_read_the_host_category():
    from discopy import symmetric, rigid, pivotal, ribbon, compact
    x = symmetric.Ty('x')
    for method in (symmetric.Hypergraph.cups, symmetric.Hypergraph.caps):
        with raises(AxiomError, match="not rigid"):
            method(x, x)
    for module in (rigid, pivotal, ribbon):  # planar, so the cups are boxes
        x, y = module.Ty('x'), module.Ty('y')
        cup, cap = module.Cup(x, x.r), module.Cap(x.r, x)
        assert cup.to_hypergraph().boxes == (cup, )
        assert cap.to_hypergraph().boxes == (cap, )
        cups = module.Diagram.cups(x @ y, (x @ y).r)
        assert cups.to_hypergraph().boxes == tuple(cups.boxes)
        assert cups.to_hypergraph().to_diagram() == cups
    x = compact.Ty('x')
    assert compact.Cup(x, x.r).to_hypergraph().boxes == ()
    assert compact.Cap(x.r, x).to_hypergraph().boxes == ()


def test_coloured_diagram_has_no_hypergraph():
    from discopy.monoidal import Box, Colour, Diagram, Ty, Wire
    red = Colour("red")
    x, a, b = Ty(Wire('x', red, red)), Wire('a', cod=red), Wire('b', dom=red)
    scalar = Box('g', Ty(), Ty(a, b)) >> Box('h', Ty(a, b), Ty())
    for convert in (Diagram.to_hypergraph, Diagram.to_map):
        for diagram in (Box('f', x, x), scalar):
            with raises(AxiomError, match="coloured"):
                convert(diagram)


def test_sum_has_no_hypergraph():
    from discopy.symmetric import Box, Diagram, Ty
    x = Ty('x')
    f, g = Box('f', x, x), Box('g', x, x)
    for convert in (Diagram.to_hypergraph, Diagram.to_map):
        for diagram in (f + g, (f + g) @ x, Diagram.sum_factory((), x, x)):
            with raises(AxiomError, match="has no sums"):
                convert(diagram)


def test_Hypergraph_to_tree():
    from discopy.utils import dumps, loads
    from discopy import markov, symmetric
    x, y = map(Ty, "xy")
    f, g = Box('f', x, y @ x), Box('g', Ty(), y)
    graphs = [
        H.spiders(1, 2, x) >> H.cups(x, x), H.spiders(0, 0, x) @ H.id(x),
        (f @ g >> y @ Diagram.swap(x, y)).to_hypergraph(),
        markov.Box('f', markov.Ty('x'), markov.Ty('y')).to_hypergraph(),
        symmetric.Diagram.id(symmetric.Ty('x')).to_hypergraph()]
    for graph in graphs:
        loaded = loads(dumps(graph))
        assert loaded == graph and type(loaded) is type(graph)
        assert loaded.category is graph.category
        assert loaded.offsets == graph.offsets
        assert loaded.spider_types == graph.spider_types


def test_trace_keeps_the_offsets():
    from discopy import symmetric
    x, y = symmetric.Ty('x'), symmetric.Ty('y')
    f = symmetric.Box('f', x, x)
    s, t = symmetric.Box('s', symmetric.Ty(), x), symmetric.Box('t', symmetric.Ty(), y)
    diagram = f @ t @ s
    graph = diagram.to_hypergraph()
    assert graph.trace().offsets == graph.offsets == (None, 1, 2)
    assert graph.trace().to_diagram() == symmetric.Trace(diagram.foliation())


def test_trace_takes_the_keyword_form():
    from discopy import symmetric
    from discopy.utils import AxiomError
    x, y = symmetric.Ty('x'), symmetric.Ty('y')
    f = symmetric.Box('f', x @ y, x @ y).to_hypergraph()
    assert f.trace() == f.trace(mem=y) == f.trace(dom=x, cod=x, mem=y)
    assert f.trace(left=True) == f.trace(mem=x, left=True) != f.trace()
    with raises(TypeError):
        f.trace(1, mem=y)
    with raises(AxiomError):
        f.trace(mem=x)


def interaction_combinators():
    x = Ty('x')
    gamma, delta = (Box(name, x, x @ x).to_hypergraph() for name in "γδ")
    epsilon = Box("ε", x, Ty()).to_hypergraph()
    rules = [
        (gamma.transpose() >> gamma, H.id(x @ x)),
        (delta.transpose() >> delta, H.swap(x, x)),
        (gamma.transpose() >> delta, delta @ delta >> x @ H.swap(x, x) @ x
         >> gamma.transpose() @ gamma.transpose()),
        (gamma.transpose() >> epsilon, epsilon @ epsilon),
        (delta.transpose() >> epsilon, epsilon @ epsilon),
        (epsilon.transpose() >> epsilon, H.id())]
    return x, gamma, delta, epsilon, rules


def test_interaction_combinators_rules():
    *_, rules = interaction_combinators()
    for left, right in rules:
        rewrites = list(left.rewrite(*rules))
        assert rewrites and all(rewrite == right for rewrite in rewrites)
        assert list(right.rewrite(*rules)) == []
    automorphic = [len(list(left.rewrite(*rules))) for left, _ in rules]
    assert automorphic == [2, 2, 1, 1, 1, 2]


def test_interaction_combinators_net_reducing_to_a_wiring():
    x, gamma, delta, _, rules = interaction_combinators()
    net = H.caps(x, x) >> delta @ delta >> x @ H.cups(x, x) @ x\
        >> gamma.transpose() >> delta >> gamma @ gamma
    for order in (rules, rules[::-1]):
        steps = list(net.normalize(*order))
        assert len(steps) == 5 and all(step.is_bijective for step in steps)
        assert steps[-1] == H.caps(x, x) @ H.caps(x, x)
    assert [len(step.boxes) for step in net.normalize(*rules)]\
        == [4, 6, 4, 2, 0]


def test_interaction_combinators_vicious_circle():
    x, gamma, _, _, rules = interaction_combinators()
    circle = (gamma.transpose() >> gamma).trace(2)
    loops = circle.normal_form(*rules)
    assert loops == H.spiders(0, 0, x @ x)


def test_rewrite_needs_an_interface():
    x = Ty('x')
    f, g = (Box(name, x, x).to_hypergraph() for name in "fg")
    rule = (f >> g, g >> f)
    assert list((f >> g >> f).rewrite(rule)) == [g >> f >> f]
    graph = f >> H.copy(x) >> (g >> f) @ x
    assert list(graph.rewrite(rule)) == []


def test_rewrite_modulo_frobenius_has_many_contexts():
    x = Ty('x')
    f, g = (Box(name, x, x).to_hypergraph() for name in "fg")
    graph = H.copy(x) >> (f >> g) @ (f >> g)
    rule = (H.id(x), f >> g)
    match = ((3, ), ())
    contexts = list(graph.complements(H.id(x), match))
    assert len(contexts) == 5
    assert all(context.plug(H.id(x)) == graph for context in contexts)
    rewrites = [context.plug(f >> g) for context in contexts]
    assert H.copy(x) >> (f >> g) @ (f >> f >> g >> g) in rewrites
    assert len(list(graph.rewrite(rule))) == 9 + 4 * 5


def test_rewrite_symmetric_boundary_complement():
    from discopy import symmetric
    x = symmetric.Ty('x')
    a1, a2, a3 = (symmetric.Box(name, dom, cod) for name, dom, cod in [
        ('a1', symmetric.Ty(), x), ('a2', x, symmetric.Ty()), ('a3', x, x)])
    graph, rule = (a1 >> a2).to_hypergraph(), (
        symmetric.Hypergraph.id(x), a3.to_hypergraph())
    assert list(graph.rewrite(rule)) == [(a1 >> a3 >> a2).to_hypergraph()]
    x = Ty('x')
    a1, a2, a3 = (Box(name, dom, cod).to_hypergraph() for name, dom, cod in [
        ('a1', Ty(), x), ('a2', x, Ty()), ('a3', x, x)])
    assert len(list((a1 >> a2).rewrite((H.id(x), a3)))) == 5


def test_rewrite_symmetric_is_convex():
    # Bonchi et al. II, Example 4.2: the rewrite needs a trace, so convex
    # rewriting misses it, even though DisCoPy's symmetric diagrams are traced.
    from discopy import frobenius, symmetric
    results = []
    for module in (frobenius, symmetric):
        x = module.Ty('x')
        e1, e2 = module.Box('e1', x, x @ x), module.Box('e2', x @ x, x)
        e3, e4 = module.Box('e3', x, x), module.Box('e4', x, x)
        left = x @ e1 >> module.Diagram.swap(x, x) @ x >> x @ e2
        right = module.Diagram.swap(x, x) >> e4 @ e4
        graph = e1 >> e3 @ x >> e2
        rule = (left.to_hypergraph(), right.to_hypergraph())
        results.append(list(graph.to_hypergraph().rewrite(rule)))
    frobenius_rewrites, symmetric_rewrites = results
    x = frobenius.Ty('x')
    e3, e4 = (frobenius.Box(name, x, x) for name in ("e3", "e4"))
    assert frobenius_rewrites == [(e4 >> e3 >> e4).to_hypergraph()]
    assert symmetric_rewrites == []


def test_rewrite_is_not_rewritable():
    from discopy import monoidal, markov
    x = monoidal.Ty('x')
    f = monoidal.Box('f', x, x).to_hypergraph()
    with raises(AxiomError):
        list(f.rewrite((f, f)))
    copy = markov.Hypergraph.copy(markov.Ty('x'))
    with raises(AxiomError):
        list(copy.match(copy))


def test_homomorphisms_check_the_partial_maps():
    x, y = map(Ty, "xy")
    f, g = (Box(name, x, x).to_hypergraph() for name in "fg")
    assert list((f >> g).homomorphisms(f, boxes={0: 0})) == [((0, 1), (0, ))]
    assert list((f >> g).homomorphisms(f, boxes={0: 1})) == []
    assert list(H.id(x @ y).homomorphisms(H.id(x), spiders={0: 1})) == []
    assert list((f >> f).homomorphisms(f >> f, injective=True)) == [
        ((0, 1, 2), (0, 1))]


def test_is_convex_walks_every_consumer():
    x = Ty('x')
    f, g, k = (Box(name, x, x).to_hypergraph() for name in "fgk")
    host = f >> H.spiders(1, 2, x) >> g @ x >> H.spiders(2, 1, x) >> k
    reordered = type(host)(
        host.dom, host.cod, (host.boxes[0], host.boxes[2], host.boxes[1]), (
            host.dom_wires, (host.box_wires[0], host.box_wires[2],
                             host.box_wires[1]), host.cod_wires))
    for graph in (host, reordered):
        assert [graph.is_convex(match)
                for match in graph.homomorphisms(f >> k)] == [False]


def test_rewrite_symmetric_reads_back_without_trace():
    from discopy import symmetric
    x = symmetric.Ty('x')
    f, g, h = (symmetric.Box(name, x, x) for name in "fgh")
    rewrite, = (f >> g).to_hypergraph().rewrite(
        (f.to_hypergraph(), h.to_hypergraph()))
    assert str(rewrite) == "h >> g"


def test_boundary_complement_is_linear_in_identity_wires():
    from discopy import symmetric
    x = symmetric.Ty('x')
    f = symmetric.Box('f', x, x).to_hypergraph()
    host = symmetric.Hypergraph.id(x).tensor(*8 * [f])
    pattern = symmetric.Hypergraph.id(x ** 8)
    match = next(host.match(pattern))
    assert len(list(host.complements(pattern, match))) == 1


def test_rewrite_errors():
    x, y = map(Ty, "xy")
    f = Box('f', x, y).to_hypergraph()
    with raises(AxiomError):
        list(f.rewrite((f, H.id(x))))
    with raises(AxiomError, match="not the hole"):
        f.plug(f)


def test_Rewriter_interaction_combinators():
    x, gamma, delta, _, rules = interaction_combinators()
    net = H.caps(x, x) >> delta @ delta >> x @ H.cups(x, x) @ x\
        >> gamma.transpose() >> delta >> gamma @ gamma
    rewriter = Rewriter(net)
    assert rewriter.normalize(*rules) == 5
    assert rewriter.to_hypergraph() == net.normal_form(*rules)


def test_Rewriter_substitute_is_the_pushout():
    from discopy import symmetric
    for module in (symmetric, None):
        x = Ty('x') if module is None else module.Ty('x')
        box = Box if module is None else module.Box
        f, g, m = box('f', x, x), box('g', x, x), box('m', x @ x, x)
        host = (f @ g >> m >> f >> g).to_hypergraph()
        left, right = (f >> g).to_hypergraph(), (g >> f).to_hypergraph()
        for match in host.match(left):
            context, = host.complements(left, match)
            rewriter = Rewriter(host)
            rewriter.substitute(left, right, match)
            assert rewriter.to_hypergraph() == context.plug(right)


def test_Rewriter_keeps_offsets():
    from discopy import symmetric
    x = symmetric.Ty('x')
    f, g = symmetric.Box('f', x, x), symmetric.Box('g', x, x)
    state = symmetric.Box('s', symmetric.Ty(), x)
    host = (x @ state >> f @ x >> x @ g >> f @ x).to_hypergraph()
    left, right = g.to_hypergraph(), (g >> g).to_hypergraph()
    match, = host.match(left)
    context, = host.complements(left, match)
    rewriter = Rewriter(host)
    rewriter.substitute(left, right, match)
    result = rewriter.to_hypergraph()
    assert result.offsets == context.plug(right).offsets
    assert not any(isinstance(box, symmetric.Permutation)
                   for box in result.to_diagram().boxes)


def test_Rewriter_merges_spiders():
    x = Ty('x')
    f, g = (Box(name, x, x).to_hypergraph() for name in "fg")
    rewriter = Rewriter(H.copy(x) >> (f >> g) @ (f >> g))
    assert rewriter.normalize((f >> g, H.id(x))) == 2
    assert rewriter.to_hypergraph() == H.copy(x)


def test_Rewriter_errors():
    x = Ty('x')
    f = Box('f', x, x).to_hypergraph()
    with raises(AxiomError):
        Rewriter(f).normalize((H.id(x), f))
    with raises(AxiomError):
        Rewriter(f).normalize((f, H.id(Ty())))
    left = H.copy(x) >> f @ x
    with raises(AxiomError):
        Rewriter(left).substitute(left, left, ((0, 1), (0, )))


def test_Rewriter_rules_are_rewritable():
    from discopy import symmetric
    x, f = symmetric.Ty('x'), symmetric.Box('f', symmetric.Ty('x'),
                                            symmetric.Ty('x'))
    host = (f >> f).to_hypergraph()
    for cod, wires in [(x @ x, (1, 1)), (symmetric.Ty(), ())]:
        rule = symmetric.Hypergraph(
            x, cod, (f, ), ((0, ), (((0, ), (1, )), ), wires))
        with raises(AxiomError, match="not rewritable"):
            Rewriter(host).normalize((rule, rule))


def test_rewriting_edge_cases():
    x = Ty('x')
    f, g, h, k = (Box(name, x, x).to_hypergraph() for name in "fghk")
    m = Box('m', x @ x, x).to_hypergraph()
    assert list((f >> f).homomorphisms(f @ f, injective=True)) == []
    assert list((f >> f).homomorphisms(f @ f))
    assert list(m.homomorphisms(H.copy(x) >> m)) == []
    assert not (f >> f).is_convex(((0, 0), (0, )))
    diamond = f >> H.copy(x) >> g @ h >> H.merge(x) >> k
    match, = diamond.homomorphisms(f)
    assert diamond.is_convex(match)
    glued = f >> H.copy(x) >> g @ x
    match, = glued.homomorphisms(f >> g)
    assert list(glued.complements(f >> g, match)) == []


def test_rewrite_keeps_the_offsets():
    from discopy import symmetric
    x = symmetric.Ty('x')
    f, g = symmetric.Box('f', x, x), symmetric.Box('g', x, x)
    a, k = symmetric.Box('a', symmetric.Ty(), x), symmetric.Box('k', x @ x, x)
    rewrite, = (f @ a >> k).to_hypergraph().rewrite(
        (f.to_hypergraph(), g.to_hypergraph()))
    assert str(rewrite) == "g @ a >> k"


def test_complements_are_lazy():
    x = Ty('x')
    f, g = (Box(name, x, x).to_hypergraph() for name in "fg")
    host = H.spiders(1, 40, x) >> f.tensor(*39 * [f])
    assert next(host.rewrite((H.id(x), f >> g))) is not None
