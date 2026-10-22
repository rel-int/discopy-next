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
    assert str(H.spiders(1, 0, x @ y))\
        == "Spider(1, 0, x) @ Spider(1, 0, y)"


def test_Hypergraph_repr():
    x, y = map(Ty, "xy")
    assert repr(H.spiders(1, 0, x @ y))\
        == "hypergraph.Hypergraph[frobenius.Diagram]("\
           "dom=frobenius.Ty(frobenius.Wire('x'), frobenius.Wire('y')), "\
           "cod=frobenius.Ty(), boxes=(), wires=((0, 1), (), ()))"


def test_Hypergraph_hash():
    x, y = map(Ty, "xy")
    assert hash(H.id(x @ y)) == hash(H.id(x) @ H.id(y))


def test_Hypergraph_then():
    x, y = map(Ty, "xy")
    with raises(AxiomError):
        H.id(x) >> H.id(y)


def test_Hypergraph_tensor():
    Id = H.id
    assert Id().tensor(Id(), Id()) == Id().tensor() == Id()


def test_Hypergraph_getitem():
    with raises(NotImplementedError):
        H.spiders(1, 2, Ty('x'))[0]


def test_Hypergraph_bijection():
    with raises(ValueError):
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
    assert K.cups(x, x.r) and K.caps(x.r, x)
    assert Cap(Ty('x'), Ty('x')).to_hypergraph()


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
    x = Ty('x')
    f = Box('f', Ty(), x)
    diagram = x @ f >> Cap(x, x) @ x @ x
    assert diagram.to_hypergraph().to_diagram() == diagram
    for diagram in [f @ Cap(x, x), x @ f >> x @ Cap(x, x) @ x]:
        decoded = diagram.to_hypergraph().to_diagram()
        assert decoded.to_hypergraph() == diagram.to_hypergraph()


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
