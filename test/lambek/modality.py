from pytest import raises

from discopy import monoidal
from discopy.lambek.modality import (
    Ty, Box, Diagram, Eval, Dereliction, Contraction, Exchange, Functor)

N, S = Ty('N'), Ty('S')
VP, H = N.bang() >> S, N.bang() >> N
John, Bill = Box('John', Ty(), N.bang()), Box('Bill', Ty(), N.bang())
code = Box('code', Ty(), N)
likes = Box('likes', Ty(), (VP.bang() << N).bang())
his = Box('his', Ty(), H.bang() << N)
does_too = Box('does too', Ty(), VP.bang() >> VP)
stop = Box('.', S @ S, S @ S)


def fa(function, argument):
    return Eval(function.cod)(function, argument)


def ba(argument, function):
    return Eval(function.cod)(argument, function)


def der(term):
    return Dereliction(term.cod.inside[0].arg)(term)


strict = N.bang()(lambda john: VP.bang()(lambda vp: stop(
    ba(john, der(vp)), ba(Bill(), ba(vp, does_too()))))(
    fa(der(likes()), ba(john, der(fa(his(), code()))))))(John())
sloppy = (N.bang() @ likes.cod @ H.bang() @ N.bang())(
    lambda john, verb, h, bill: stop(
        ba(john, der(fa(der(verb), ba(john, der(h))))),
        ba(bill, ba(fa(der(verb), ba(bill, der(h))), does_too()))))(
    John(), likes(), fa(his(), code()), Bill())


def test_strict_and_sloppy_contractions():
    readings = {name: term.eval() for name, term in [
        ("strict", strict), ("sloppy", sloppy)]}
    for diagram in readings.values():
        assert isinstance(diagram, Diagram)
        assert diagram.dom == Ty() and diagram.cod == S @ S
    copied = {name: sorted(str(box.x) for box in diagram.boxes
                           if isinstance(box, Contraction))
              for name, diagram in readings.items()}
    assert copied == {
        "strict": ["(N.bang() >> S)", "N"],
        "sloppy": ["((N.bang() >> S).bang() << N)", "(N.bang() >> N)",
                   "N", "N"]}


def test_functor_into_a_category_that_is_not_closed():
    with raises(TypeError):
        Functor(ob_map={}, ar_map={}, cod=monoidal.Diagram)
    seen = []

    def ar_map(box):
        seen.append(box)
        return monoidal.Box(box.name, F(box.dom), F(box.cod))
    F = monoidal.Functor(
        ob_map=lambda x: monoidal.Ty(str(x)), ar_map=ar_map)
    assert F(Eval(VP)) == monoidal.Box(
        Eval(VP).name, monoidal.Ty("N.bang()", "(N.bang() >> S)"),
        monoidal.Ty("S"))
    for term in (strict, sloppy):
        assert F(term.eval()).cod == monoidal.Ty("S", "S")
    assert {type(box) for box in seen if box.name.startswith(
        ("Contraction", "Dereliction", "Exchange"))} == {
            Contraction, Dereliction, Exchange}


def test_drawing_names_bang_under_exponentials():
    assert likes.cod.to_drawing().inside[0].name == "!(!(!N >> S) << N)"
    assert his.cod.to_drawing().inside[0].name == "(!(!N >> N) << N)"
    assert str(his.cod) == "((N.bang() >> N).bang() << N)"
