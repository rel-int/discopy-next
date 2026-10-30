from itertools import permutations

import numpy as np
from pytest import raises

from discopy import rigid, ribbon, tensor
from discopy.utils import dumps, loads
from discopy.tensor import Dim, Box
from discopy.group import Cyclic, Symmetric, Product, Wreath
from discopy.hopf import (
    Algebra, GroupAlgebra, Double, Representation, Intertwiner, Functor)


circle = lambda x: ribbon.Cap(x, x.r) >> ribbon.Cup(x, x.r)
unlink = lambda x: circle(x) @ circle(x)
hopf_link = lambda x: (ribbon.Braid(x, x) >> ribbon.Braid(x, x)).trace(n=2)


def double_and_module():
    D = Double(GroupAlgebra(Cyclic[2]))
    e, m = Representation[D].anyon(0, -1), Representation[D].anyon(1, 1)
    return D, Representation[D].direct_sum([e, m])


taft = Algebra.taft(3)
taft_double = Double(taft)
sweedler_double = Double(Algebra.sweedler())


def test_group_algebra_is_valid():
    for n in [1, 2, 3, 5]:
        assert GroupAlgebra(Cyclic[n]).is_valid()
    klein = Product[Cyclic[2], Cyclic[2]]
    cube = Wreath[Cyclic[2], Symmetric[2]]
    assert GroupAlgebra(klein).is_valid() and GroupAlgebra(cube).is_valid()
    assert not GroupAlgebra(cube).is_commutative()


def test_double_is_quasitriangular_hopf_algebra():
    """is_valid checks quasitriangularity whenever there is an R-matrix."""
    D = Double(GroupAlgebra(Cyclic[2]))
    assert D.dim == 4
    assert D.is_valid()


def test_commutativity_properties():
    """D(Z/2) is commutative and cocommutative, Sweedler's H4 is neither."""
    D = Double(GroupAlgebra(Cyclic[2]))
    assert D.is_commutative() and D.is_cocommutative()
    H4 = Algebra.sweedler()
    assert not H4.is_commutative() and not H4.is_cocommutative()


def test_double_of_sweedler():
    """The double of H4 exercises S^-1 in the coadjoint multiplication.

    The axioms are checked on the materialised structure constants: the
    equations are the same but each generator is a single box, where
    ``is_valid`` on the fine-grained double — covered by ``Double(cyclic(2))``
    — contracts networks that blow up under the greedy einsum path."""
    H4 = Algebra.sweedler()
    assert H4.is_valid() and H4.dim == 4
    Sarr = H4.antipode.eval(dtype=complex).array
    assert not np.allclose(Sarr @ Sarr, np.eye(4))
    D, n = Double(H4), 16
    assert D.dim == n
    unit, counit, mult, comult, antipode, R = (
        gen.eval(dtype=complex).array for gen in D.generators)
    materialised = Algebra.from_arrays(
        unit.reshape(n), counit.reshape(n), mult.reshape(n, n, n),
        comult.reshape(n, n, n), antipode.reshape(n, n), R.reshape(n, n))
    assert materialised.is_valid()


def test_double_antipode_inverse():
    """S^-1 = u^-1 S(x) u for the Drinfeld element u = S(R'')R', with
    u^-1 = R''S^2(R'), also when S^2 != id, assembled from the generator
    matrices."""
    for base in [GroupAlgebra(Cyclic[2]), Algebra.sweedler()]:
        D, n = Double(base), base.dim ** 2
        S = D.antipode.eval(dtype=complex).array.reshape(n, n)
        R = D.R.eval(dtype=complex).array.reshape(n, n)
        mult = D.mult.eval(dtype=complex).array.reshape(n, n, n)
        u = np.einsum('ij,ja,aik->k', R, S, mult)
        u_inv = np.einsum('ij,ib,ba,jak->k', R, S, S, mult)
        Si = np.einsum('xy,i,iyk,j,kjl->xl', S, u_inv, mult, u, mult)
        assert np.allclose(Si @ S, np.eye(n))
        assert np.allclose(S @ Si, np.eye(n))


def test_double_is_fine_grained():
    """Every box in the double's generators acts on the base object, so no
    structure tensor of the doubled dimension is ever materialised."""
    D = Double(GroupAlgebra(Cyclic[2]))
    for gen in D.generators:
        for box in gen.boxes:
            assert set(box.dom.inside) <= {2} and set(box.cod.inside) <= {2}


def test_no_r_matrix_paths():
    Z2 = GroupAlgebra(Cyclic[2])
    noR = Algebra(
        Z2.unit, Z2.counit, Z2.mult, Z2.comult, Z2.antipode)
    assert noR.R is None
    assert noR.is_quasitriangular() is False
    assert noR.is_ribbon() is False
    assert noR.is_valid()
    W = Representation[noR].regular()
    try:
        Intertwiner[noR].braid(W, W)
        assert False
    except ValueError:
        pass
    try:
        Intertwiner.braid(W, W)
        assert False
    except ValueError:
        pass
    try:
        Intertwiner.twist(W)
        assert False
    except ValueError:
        pass
    assert W.l.is_module()


def test_non_invertible_antipode_is_flagged():
    Z2 = GroupAlgebra(Cyclic[2])
    zero = Box('S', Z2.ty, Z2.ty, np.zeros((2, 2)))
    H = Algebra(Z2.unit, Z2.counit, Z2.mult, Z2.comult, zero)
    try:
        H.antipode_inv
        assert False
    except ValueError:
        pass


def test_class_generic_over_the_algebra():
    """The algebra lives on the class, parametrisation caches by equality."""
    D = Double(GroupAlgebra(Cyclic[2]))
    assert Representation[D] is Representation[Double(GroupAlgebra(Cyclic[2]))]
    assert Representation[D].algebra == D
    V = Representation[D].regular()
    assert V.algebra == D
    braid = Intertwiner[D].braid(V, V)
    assert Intertwiner[D].algebra == D and braid.algebra == D


def test_representation_is_a_dim():
    D, V = double_and_module()
    assert isinstance(V, Dim) and V == Dim(2)
    e, m = Representation[D].anyon(0, -1), Representation[D].anyon(1, 1)
    assert e.action != m.action
    assert e != Dim(2)
    trivial = Representation[D](Dim(2))
    assert trivial == Dim(2) and trivial.is_module()
    assert hash(trivial) == hash(Dim(2))
    from discopy import (  # noqa: F401  (used by eval)
        group, hopf, compact, frobenius, tensor)
    assert eval(repr(trivial)) == trivial


def test_representation_init_checks():
    D, V = double_and_module()
    try:
        Representation(Dim(2))
        assert False
    except ValueError:
        pass
    try:
        Representation[D](Dim(2), action=D.counit)
        assert False
    except ValueError:
        pass
    try:
        Representation[D](2)
        assert False
    except TypeError:
        pass


def test_anyon_needs_a_double():
    Z2 = GroupAlgebra(Cyclic[2])
    try:
        Representation[Z2].anyon(0, 1)
        assert False
    except ValueError:
        pass


def test_repr_is_transparent():
    from discopy import group, hopf, tensor  # noqa: F401  (used by eval)
    H = GroupAlgebra(Cyclic[2])
    assert eval(repr(H)) == H
    D = Double(H)
    assert eval(repr(D)) == D
    V = Representation[H].regular()
    assert eval(repr(V)) == V and eval(repr(V)).action == V.action


def test_intertwiner_is_a_network_between_modules():
    """#145: an intertwiner is its own factory, holding the tensor network
    between the spaces of its modules, and its trees carry the generators of
    its algebra and the action of each module."""
    from discopy import (  # noqa: F401  (used by eval)
        group, hopf, tensor, frobenius)
    D, V = double_and_module()
    braid = Intertwiner[D].braid(V, V)
    for composite in [braid >> braid, braid @ braid, braid.dagger(),
                      Intertwiner[D].id(V), braid.r]:
        assert type(composite) is Intertwiner[D]
        assert isinstance(composite.inside, tensor.Diagram)
    assert (braid @ braid).eval(dtype=complex).is_close(
        (braid.inside @ braid.inside).eval(dtype=complex))
    anyon = Representation[D].anyon(1, 1)
    for x in [anyon, braid, D, taft, Intertwiner.caps(V, V.r)]:
        assert loads(dumps(x)) == x
    assert loads(dumps(anyon)).action == anyon.action
    assert eval(repr(braid)) == braid
    V = Representation[taft].regular()
    cup = Intertwiner[taft].cups(V, V.r)
    assert cup.inside.to_hypergraph().to_diagram().eval(
        dtype=complex).is_close(cup.eval(dtype=complex))


def test_intertwiner_refuses_a_module_over_another_algebra():
    D, W = double_and_module()
    V = Representation[taft].regular()
    f, g = Intertwiner[D].id(W), Intertwiner[taft].id(V)
    with raises(ValueError):
        Intertwiner[D](tensor.Id(Dim(*V.inside)), V, V)
    for composite in [lambda: f @ g, lambda: g @ f, lambda: f >> g,
                      lambda: Intertwiner.id(W) >> f]:
        with raises(ValueError):
            composite()
    plain = Intertwiner.id(Dim(*W.inside))
    assert (f >> plain).algebra == (f @ plain).algebra == D


def test_intertwiner_trace_zero_is_identity():
    D, V = double_and_module()
    braid = Intertwiner[D].braid(V, V)
    assert braid.trace(0) == braid
    assert braid.trace(0).dom.action == braid.dom.action


def test_representation_is_module():
    """The direct-sum module axiom is checked in the direct_sum doctest."""
    D, V = double_and_module()
    assert V == Dim(2)
    assert Representation[D].regular().is_module()
    for anyon in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        assert Representation[D].anyon(*anyon).is_module()


def test_tensor_of_representations():
    """Products act through the comultiplication, plain Dims trivially."""
    D, V = double_and_module()
    e, m = Representation[D].anyon(0, -1), Representation[D].anyon(1, 1)
    assert (e @ m).is_module() and (e @ m).action is not None
    assert V @ V == Dim(2, 2)
    assert (V @ Dim(2)).is_module() and (V @ Dim(2)) == Dim(2, 2)
    unit = Representation[D]()
    assert unit @ unit == Dim(1) and (unit @ V).action == V.action


def test_braiding_yang_baxter_and_inverse():
    D, V = double_and_module()
    d = 2
    c = Intertwiner[D].braid(V, V).eval(dtype=complex).array
    c = c.reshape(d * d, d * d)
    assert not np.isclose(np.linalg.det(c), 0)
    swap = np.zeros((d * d, d * d))
    for a in range(d):
        for b in range(d):
            swap[a * d + b, b * d + a] = 1
    assert not np.allclose(c, swap)
    ci = Intertwiner[D].braid(V, V, is_dagger=True)\
        .eval(dtype=complex).array
    ci = ci.reshape(d * d, d * d)
    assert np.allclose(c.T @ ci.T, np.eye(d * d))
    R, eye = c.T, np.eye(d)
    R12, R23 = np.kron(R, eye), np.kron(eye, R)
    assert np.allclose(R12 @ R23 @ R12, R23 @ R12 @ R23)


def test_quantum_dimension():
    _, V = double_and_module()
    assert np.isclose(V.qdim(), 2)


def test_dual_representation():
    """The right dual carries the twisted action rho(S h)^T, also when
    S^2 != id and on structured types."""
    D, V = double_and_module()
    assert V.r.is_module()
    assert Representation[Algebra.sweedler()].regular().r.is_module()
    n = D.dim
    rho = V.action.eval(dtype=complex).array.reshape(n, 2, 2)
    S = D.antipode.eval(dtype=complex).array.reshape(n, n)
    dual = V.r.action.eval(dtype=complex).array.reshape(n, 2, 2)
    assert np.allclose(dual, np.einsum('hk,kio->hoi', S, rho))


def test_left_and_right_duals_differ():
    """The left dual twists by S^-1, so it differs from the right dual
    whenever S^2 != id, as for Sweedler's H4 but not a group algebra."""
    H4 = Algebra.sweedler()
    W = Representation[H4].regular()
    assert W.l.is_module() and W.r.is_module()
    left = W.l.action.eval(dtype=complex).array
    right = W.r.action.eval(dtype=complex).array
    assert not np.allclose(left, right)
    Z2 = GroupAlgebra(Cyclic[2])
    U = Representation[Z2].regular()
    assert np.allclose(U.l.action.eval(dtype=complex).array,
                       U.r.action.eval(dtype=complex).array)


def test_functor_maps_winding_to_dual():
    D, V = double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[D])
    assert F(x).action == V.action
    assert F(x.r).action == V.r.action != V.action


def test_twist_is_the_trace_of_the_braid():
    """The twist of the D(Z/3) anyon of flux f and charge omega is
    omega ** f, the trace of the braid on either side: neither is its own
    inverse, as every twist of D(Z/2) is, e.g. minus one on the fermion,
    and on their direct sum it is diagonal."""
    omega = np.exp(2j * np.pi / 3)
    D2, D3 = Double(GroupAlgebra(Cyclic[2])), Double(GroupAlgebra(Cyclic[3]))
    x = ribbon.Ty('x')
    for D, flux, charge, theta in [
            (D3, 1, omega, omega), (D3, 2, omega, omega ** 2), (D2, 1, -1, -1)]:
        V = Representation[D].anyon(flux, charge)
        F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[D])
        for diagram in [ribbon.Twist(x), ribbon.Braid(x, x).trace(),
                        ribbon.Braid(x, x).trace(left=True)]:
            assert np.isclose(complex(F(diagram).eval(dtype=complex)), theta)
    V = Representation[D3].direct_sum([
        Representation[D3].anyon(flux, omega) for flux in (1, 2)])
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[D3])
    for diagram in [ribbon.Twist(x), ribbon.Braid(x, x).trace(),
                    ribbon.Braid(x, x).trace(left=True)]:
        assert np.allclose(F(diagram).eval(dtype=complex).array,
                           np.diag([omega, omega ** 2]))


def test_twist_dagger_is_its_inverse():
    """B51 of #31: over the double of the Taft algebra, which is not
    unitary, the dagger of the twist is its inverse, the action of the
    ribbon element, not the conjugate transpose which is no intertwiner."""
    D = taft_double
    V = Representation[D].regular()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[D])
    twist = ribbon.Twist(x)
    identity = F(ribbon.Id(x)).eval(dtype=complex)
    for diagram in [twist >> twist.dagger(), twist.dagger() >> twist]:
        assert F(diagram).eval(dtype=complex).is_close(identity)
    untwist = F(twist.dagger())
    lhs = V.action >> untwist.inside
    rhs = D.ty @ untwist.inside >> V.action
    assert lhs.eval(dtype=complex).is_close(rhs.eval(dtype=complex))


def test_is_module_rejects_non_module():
    Z2 = GroupAlgebra(Cyclic[2])
    zero = Box('bad', Z2.ty @ Dim(2), Dim(2), np.zeros((2, 2, 2)))
    assert not Representation[Z2](Dim(2), zero).is_module()


def test_snake_equations():
    D, V = double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[D])
    left = ribbon.Id(x.l).transpose(left=True)
    right = ribbon.Id(x.r).transpose(left=False)
    identity = F(ribbon.Id(x)).eval(dtype=complex).array
    assert np.allclose(F(left).eval(dtype=complex).array, identity)
    assert np.allclose(F(right).eval(dtype=complex).array, identity)


def test_nontrivial_link_invariant():
    """The functor returns a tensor network for each link, whose contraction
    separates the circle, the unlink, the Hopf link and the unknot as a
    braid closure."""
    D, V = double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[D])
    unknot = ribbon.Braid(x, x).trace(n=2)
    networks = [F(d) for d in [circle(x), unlink(x), hopf_link(x), unknot]]
    assert all(isinstance(network, Intertwiner[D])
               and isinstance(network.inside, tensor.Diagram)
               for network in networks)
    values = [complex(network.eval(dtype=complex)) for network in networks]
    assert np.allclose(values, [2, 4, 0, 2])


def test_two_colour_mutual_braiding():
    """The e and m anyons of the toric code have mutual statistics -1."""
    D = Double(GroupAlgebra(Cyclic[2]))
    e = Representation[D].anyon(0, -1)
    m = Representation[D].anyon(1, 1)
    xe, xm = ribbon.Ty('e'), ribbon.Ty('m')
    F = Functor(ob_map={xe: e, xm: m}, ar_map={}, cod=Intertwiner[D])

    def value(a, b):
        link = (ribbon.Braid(a, b) >> ribbon.Braid(b, a)).trace(n=2)
        return complex(F(link).eval(dtype=complex))

    assert np.isclose(value(xe, xm), -1)
    assert np.isclose(value(xe, xe), 1)
    assert np.isclose(value(xm, xm), 1)


def test_functor_on_generic_box():
    D, V = double_and_module()
    x = ribbon.Ty('x')
    box = ribbon.Box('f', x, x)
    F = Functor(ob_map={x: V}, ar_map={box: np.array([[1, 2], [3, 4]])},
                cod=Intertwiner[D])
    assert np.allclose(
        F(box).eval(dtype=complex).array, [[1, 2], [3, 4]])


def test_functor_on_type():
    D, V = double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[D])
    assert F(x) == Dim(2)
    assert F(x @ x) == Dim(2, 2)


def test_twist_in_diagram():
    """The functor reads the twist of a product of D(Z/3) anyons off the
    twist of each and their double braiding, which is not an involution:
    it is the twist of the product module."""
    D = Double(GroupAlgebra(Cyclic[3]))
    omega = np.exp(2j * np.pi / 3)
    a, b = Representation[D].anyon(1, omega), Representation[D].anyon(0, omega)
    x, y = ribbon.Ty('x'), ribbon.Ty('y')
    F = Functor(ob_map={x: a, y: b}, ar_map={}, cod=Intertwiner[D])
    twist = F(ribbon.Diagram.twist(x @ y)).eval(dtype=complex)
    assert twist.is_close(Intertwiner[D].twist(a @ b).eval(dtype=complex))
    assert np.isclose(complex(twist), omega ** 2)


def test_rotated_braid_is_transposed_braid():
    """The half-turn rotation of either crossing is its transpose, which
    keeps the same strand over."""
    D = Double(GroupAlgebra(Cyclic[3]))
    omega = np.exp(2j * np.pi / 3)
    a, b = Representation[D].anyon(1, omega), Representation[D].anyon(2, omega)
    x, y = ribbon.Ty('x'), ribbon.Ty('y')
    F = Functor(ob_map={x: a, y: b}, ar_map={}, cod=Intertwiner[D])
    for braid in [ribbon.Braid(x, y), ribbon.Braid(x, y).dagger()]:
        assert F(braid.r).eval(dtype=complex).is_close(
            F(braid.transpose()).eval(dtype=complex))


def test_taft_algebra():
    for n, T in [(2, Algebra.taft(2)), (3, taft)]:
        assert T.is_valid() and T.dim == n * n
    S = taft.antipode.eval(dtype=complex).array
    assert not np.allclose(S @ S, np.eye(9))


def test_drinfeld_element():
    """u = m(S (x) 1)R_21 conjugates the square of the antipode."""
    D = Double(GroupAlgebra(Cyclic[2]))
    n = D.dim
    u = D.drinfeld_element.eval(dtype=complex).array.reshape(n)
    S = D.antipode.eval(dtype=complex).array.reshape(n, n)
    mult = D.mult.eval(dtype=complex).array.reshape(n, n, n)
    assert np.allclose(np.einsum('xw,g,wgz->xz', S @ S, u, mult),
                       np.einsum('g,gxz->xz', u, mult), atol=1e-6)
    Z2 = GroupAlgebra(Cyclic[2])
    noR = Algebra(Z2.unit, Z2.counit, Z2.mult, Z2.comult, Z2.antipode)
    try:
        noR.drinfeld_element
        assert False
    except ValueError:
        pass


def double_and_four_anyons():
    D = Double(GroupAlgebra(Cyclic[2]))
    anyons = [Representation[D].anyon(f, c) for f in (0, 1) for c in (1, -1)]
    return D, anyons, Representation[D].direct_sum(anyons)


def test_chart_fusion_dimensions():
    D, anyons, V = double_and_four_anyons()
    assert Intertwiner[D].chart(V @ V, V).dom == Dim(16, 4, 4)
    assert Intertwiner[D].chart(V, V).dom == Dim(4, 4)
    assert Intertwiner[D].chart(anyons[1] @ anyons[2], anyons[3]).dom == Dim(1)
    for anyon in anyons:
        assert Intertwiner[D].chart(anyon, anyon).dom == Dim(1)


def test_chart_slices_intertwine_and_span():
    D, _, V = double_and_four_anyons()
    Q = Intertwiner[D].chart(V @ V, V)
    lhs = tensor.Id(D.ty) @ Q.inside >> V.action
    rhs = tensor.Diagram.swap(D.ty, Dim(16)) @ tensor.Id(Dim(4, 4)) \
        >> tensor.Id(Dim(16)) @ (V @ V).action >> Q.inside
    assert lhs.eval(dtype=complex).is_close(rhs.eval(dtype=complex))
    basis = Q.eval(dtype=complex).array.reshape(16, -1)
    assert np.allclose(basis @ basis.conj().T, np.eye(16))
    end = Intertwiner[D].chart(V, V).eval(dtype=complex).array.reshape(-1, 16)
    identity = np.eye(4).reshape(-1)
    assert np.allclose(end.T.conj() @ (end.conj() @ identity), identity)


def test_chart_zero_and_no_algebra():
    D, anyons, _ = double_and_four_anyons()
    try:
        Intertwiner[D].chart(anyons[1], anyons[2])
        assert False
    except ValueError as error:
        assert str(error) == "the space of intertwiners is zero"
    try:
        Intertwiner.chart(anyons[1], anyons[2])
        assert False
    except ValueError:
        pass


def test_pivotal_element():
    """The pivot is grouplike and conjugates the square of the antipode;
    it is the unit exactly when S^2 = 1."""
    Z2 = GroupAlgebra(Cyclic[2])
    assert Z2.pivotal_element == Z2.unit
    assert Double(Z2).pivotal_element == Double(Z2).unit
    for H in [Algebra.sweedler(), taft, sweedler_double]:
        assert H.pivotal_element != H.unit
        n = H.dim
        g = H.pivotal_element.eval(dtype=complex).array.reshape(n)
        S = H.antipode.eval(dtype=complex).array.reshape(n, n)
        mult = H.mult.eval(dtype=complex).array.reshape(n, n, n)
        counit = H.counit.eval(dtype=complex).array.reshape(n)
        comult = H.comult.eval(dtype=complex).array.reshape(n, n, n)
        assert np.allclose(np.einsum('xw,g,wgz->xz', S @ S, g, mult),
                           np.einsum('g,gxz->xz', g, mult), atol=1e-6)
        assert np.allclose(np.einsum('i,ipq->pq', g, comult),
                           np.outer(g, g), atol=1e-6)
        assert np.isclose(counit @ g, 1)
    Z2 = GroupAlgebra(Cyclic[2])
    scale = Box('S', Z2.ty, Z2.ty, np.diag([1, 2]))
    doctored = Algebra(Z2.unit, Z2.counit, Z2.mult, Z2.comult, scale)
    try:
        doctored.pivotal_element
        assert False
    except ValueError:
        pass


def test_ribbon_element_criterion():
    """The double of an odd Taft algebra is ribbon, of an even one is not,
    by Kauffman-Radford: accessing the ribbon element validates centrality,
    S(v) = v and v^2 = uS(u), raising when they fail."""
    assert taft_double.ribbon_element is not None
    try:
        sweedler_double.ribbon_element
        assert False
    except ValueError:
        pass


def test_pivotal_pairings_are_intertwiners():
    """The functor images of all four cup and cap orientations commute with
    the actions over the Taft algebra, whose pivot is non-trivial."""
    T = taft
    V = Representation[T].regular()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[T])
    n, d = T.dim, 9
    counit = T.counit.eval(dtype=complex).array.reshape(n)
    comult = T.comult.eval(dtype=complex).array.reshape(n, n, n)

    def rho(rep):
        return rep.action.eval(dtype=complex).array.reshape(n, d, d)

    for box, first, second in [
            (ribbon.Cup(x, x.r), V, V.r), (ribbon.Cup(x.l, x), V.r, V),
            (ribbon.Cap(x, x.r), V, V.r), (ribbon.Cap(x.l, x), V.r, V)]:
        image = F(box).eval(dtype=complex).array.reshape(d, d)
        if isinstance(box, ribbon.Cup):
            lhs = np.einsum('hpq,pio,qjw,ow->hij',
                            comult, rho(first), rho(second), image)
            rhs = np.einsum('h,ij->hij', counit, image)
        else:
            lhs = np.einsum('hpq,pio,qjw,ij->how',
                            comult, rho(first), rho(second), image)
            rhs = np.einsum('h,ow->how', counit, image)
        assert np.allclose(lhs, rhs, atol=1e-6)


def test_left_dual_pairings_are_intertwiners():
    """A rigid functor sends the cups and caps of both adjoints to pivotal
    pairings, which commute with the actions over the Taft algebra."""
    V = Representation[taft].regular()
    x = rigid.Ty('x')
    F = rigid.Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[taft])
    for cup in [rigid.Cup(x.l, x), rigid.Cup(x, x.r)]:
        lhs = F(cup.dom).action >> F(cup).inside
        rhs = taft.counit @ F(cup).inside
        assert lhs.eval(dtype=complex).is_close(rhs.eval(dtype=complex))
    for cap in [rigid.Cap(x, x.l), rigid.Cap(x.r, x)]:
        lhs = taft.ty @ F(cap).inside >> F(cap.cod).action
        rhs = taft.counit @ F(cap).inside
        assert lhs.eval(dtype=complex).is_close(rhs.eval(dtype=complex))
    snake = rigid.Cap(x, x.l) @ x >> x @ rigid.Cup(x.l, x)
    assert F(snake).eval(dtype=complex).is_close(
        F(rigid.Id(x)).eval(dtype=complex))


def test_snake_equations_with_pivot():
    """The pivotal pairings satisfy the snake equations over the Taft
    algebra, where the plain ones would not be intertwiners."""
    T = taft
    V = Representation[T].regular()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[T])
    left = ribbon.Id(x.l).transpose(left=True)
    right = ribbon.Id(x.r).transpose(left=False)
    identity = F(ribbon.Id(x)).eval(dtype=complex).array
    assert np.allclose(F(left).eval(dtype=complex).array, identity,
                       atol=1e-6)
    assert np.allclose(F(right).eval(dtype=complex).array, identity,
                       atol=1e-6)


def test_chart_needs_no_semisimplicity():
    H = Algebra.sweedler()
    W = Representation[H].regular()
    Q = Intertwiner[H].chart(W, W)
    assert Q.dom == Dim(4, 4)
    lhs = tensor.Id(H.ty) @ Q.inside >> W.action
    rhs = tensor.Diagram.swap(H.ty, Dim(4)) @ tensor.Id(Dim(4)) \
        >> tensor.Id(Dim(4)) @ W.action >> Q.inside
    assert lhs.eval(dtype=complex).is_close(rhs.eval(dtype=complex))


def test_chart_braid_action():
    D, _, V = double_and_four_anyons()
    Q3 = Intertwiner[D].chart(V @ V @ V, V)
    r = Q3.dom.inside[0]
    basis = Q3.eval(dtype=complex).array.reshape(r, 64, 4)
    braid = Intertwiner[D].braid(V, V).eval(dtype=complex).array
    braid = braid.reshape(16, 16).T
    action = [np.kron(braid, np.eye(4)), np.kron(np.eye(4), braid)]
    b1, b2 = [
        np.einsum('kva,vw,lwa->kl', basis.conj(), c, basis) for c in action]
    assert np.allclose(b1 @ b1.conj().T, np.eye(r))
    assert np.allclose(b1 @ b2 @ b1, b2 @ b1 @ b2)
    assert not np.allclose(b1 @ b1, np.eye(r))


def double_of_s3_and_anyons():
    D = Double(GroupAlgebra(Symmetric[3]))
    perms = sorted(permutations(range(3)))
    matrices = {i: np.eye(3)[list(p)].T for i, p in enumerate(perms)}
    omega = np.exp(2j * np.pi / 3)
    dft = np.array([
        [1, omega, omega ** 2], [1, omega ** 2, omega]]) / np.sqrt(3)
    standard = {i: dft @ m @ dft.conj().T for i, m in matrices.items()}
    sign = {i: float(np.linalg.det(m)) for i, m in matrices.items()}
    R = Representation[D]
    anyons = [
        R.anyon(0, 1), R.anyon(0, sign), R.anyon(0, standard),
        R.anyon(1, {0: 1, 1: 1}), R.anyon(1, {0: 1, 1: -1}),
        R.anyon(3, {0: 1, 3: 1, 4: 1}),
        R.anyon(3, {0: 1, 3: omega, 4: omega ** 2}),
        R.anyon(3, {0: 1, 3: omega ** 2, 4: omega})]
    return D, anyons


def test_anyon_needs_the_double_of_a_group_algebra():
    with raises(ValueError):
        Representation[sweedler_double].anyon(0, 1)


def test_permutation_representation():
    H, G = GroupAlgebra(Symmetric[3]), Wreath[Cyclic[2], Symmetric[2]]
    with raises(TypeError):
        Representation[H].permutation(G.action())
    with raises(TypeError):
        Representation[Algebra.sweedler()].permutation(G.action())
    a = Symmetric[3].action()
    V = Representation[H].permutation(a)
    assert V.is_module() and (V @ V @ V).is_module()


def test_burnside_lemma():
    """The number of orbits of an action is the average trace of the
    permutation matrices, i.e. the trace of the projection on invariants."""
    G = Wreath[Cyclic[2], Symmetric[2]]
    H, a = GroupAlgebra(G), G.action()
    for action in [a, a.tensor(a), G.action(power=True)]:
        V = Representation[H].permutation(action)
        rho = V.action.eval().array.reshape(H.dim, action.size, action.size)
        traces = np.einsum('gxx->g', rho)
        assert np.isclose(traces.sum() / G.order(), len(action.orbits()))


def test_symmetric_group_algebra():
    S3 = GroupAlgebra(Symmetric[3])
    assert S3.is_valid() and S3.dim == 6 and not S3.is_commutative()


def test_anyons_of_double_s3():
    D, anyons = double_of_s3_and_anyons()
    dims = [int(np.prod(a.inside)) for a in anyons]
    assert dims == [1, 1, 2, 3, 3, 2, 2, 2]
    assert sum(d * d for d in dims) == D.dim
    assert all(a.is_module() for a in anyons)
    for a, d in zip(anyons, dims):
        Q = Intertwiner[D].chart(a, a)
        assert Q.eval(dtype=complex).array.size == d * d


def test_fusion_rules_from_braiding():
    D, anyons = double_of_s3_and_anyons()
    dims = [int(np.prod(a.inside)) for a in anyons]
    n = len(anyons)
    S = np.zeros((n, n), dtype=complex)
    for a in range(n):
        for b in range(n):
            forth = Intertwiner[D].braid(anyons[a], anyons[b])
            back = Intertwiner[D].braid(anyons[b], anyons[a])
            monodromy = forth.eval(dtype=complex).array.reshape(
                dims[a] * dims[b], -1) @ back.eval(
                dtype=complex).array.reshape(dims[b] * dims[a], -1)
            S[a, b] = np.trace(monodromy) / D.base.dim
    assert np.allclose(S @ S.conj().T, np.eye(n))
    N = np.einsum('ax,bx,cx,x->abc', S, S, S.conj(), 1 / S[0])
    assert np.allclose(N.imag, 0) and np.allclose(N, np.round(N.real))
    N = np.round(N.real).astype(int)
    assert (N >= 0).all()
    assert list(N[2, 2]) == [1, 1, 1, 0, 0, 0, 0, 0]
    for a, b in [(2, 2), (3, 4), (3, 3), (2, 6)]:
        for c in range(n):
            try:
                Q = Intertwiner[D].chart(anyons[a] @ anyons[b], anyons[c])
                r = Q.eval(dtype=complex).array.size \
                    // (dims[a] * dims[b] * dims[c])
            except ValueError as error:
                assert str(error) == "the space of intertwiners is zero"
                r = 0
            assert r == N[a, b, c]
