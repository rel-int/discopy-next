from pytest import raises

from discopy import braided, monoidal, spacial, symmetric
from discopy.abc import SpacialCategory
from discopy.utils import AxiomError


def test_braided_categories_are_spacial():
    assert not issubclass(monoidal.Diagram, SpacialCategory)
    for module in (spacial, braided, symmetric):
        assert issubclass(module.Diagram, SpacialCategory)


def test_free_braided_diagrams_declare_the_law():
    from discopy.symmetric import Ty, Box
    x = Ty('x')
    f, s = Box('f', x, x), Box('s', Ty(), Ty())
    assert symmetric.Diagram.scalar_sliding(scalar=s, f=f)
    assert braided.Diagram.scalar_sliding.broken


def test_scalars_slide_on_maps():
    from discopy.spacial import Ty, Box
    x = Ty('x')
    f, g, s = Box('f', x, x), Box('g', x, x), Box('s', Ty(), Ty())
    for host in (f @ s, s @ f, f >> s @ x, x @ s >> f):
        assert len(list(host.to_map().match((s @ f).to_map()))) == 1
        rewrite, = host.to_map().rewrite((f.to_map(), g.to_map()))
        assert sorted(map(str, rewrite.boxes)) == ['g', 's']


def test_a_closed_pattern_matches_floating_components():
    from discopy.spacial import Ty, Box, Id
    x = Ty('x')
    f = Box('f', x, x)
    cap, cup = Box('cap', Ty(), x @ x), Box('cup', x @ x, Ty())
    circle = cap >> cup
    host = (f @ circle @ circle).to_map()
    assert len(list(host.match(circle.to_map()))) == 2
    steps = list(host.normalize((circle.to_map(), Id().to_map())))
    assert len(steps) == 2 and steps[-1] == f.to_map()


def test_planar_maps_still_need_to_be_connected():
    from discopy.monoidal import Ty, Box
    x = Ty('x')
    f, s = Box('f', x, x), Box('s', Ty(), Ty())
    with raises(AxiomError):
        next((f @ s).to_map().match(f.to_map()))


def test_a_pattern_with_a_loop_does_not_match():
    from discopy.compact import Ty, CMap
    x = Ty('x')
    loop = CMap.caps(x.r, x) >> CMap.cups(x.r, x)
    with raises(AxiomError):
        next(CMap.id(x).match(loop))
