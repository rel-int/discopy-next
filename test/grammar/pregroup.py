from pytest import raises

from discopy import grammar, rigid
from discopy.grammar.pregroup import *
from discopy.utils import from_tree


def test_Word():
    with raises(TypeError):
        Word(0, Ty('n'))
    with raises(TypeError):
        Word('Alice', 1)
    with raises(TypeError):
        Word('Alice', Ty('n'), dom=1)
    Alice = Word('Alice', Ty('n'))
    assert Alice.r[::-1].z == 1 and Alice.r[::-1][::-1] == Alice.r
    namespace = dict(rigid=rigid, grammar=grammar)
    for word in (Alice, Alice.r, Alice[::-1], Word('x', Ty('n'), data=[1])):
        assert eval(repr(word), dict(namespace)) == word


def test_eager_parse():
    s, n = Ty('s'), Ty('n')
    Alice = Word('Alice', n)
    loves = Word('loves', n.r @ s @ n.l)
    Bob = Word('Bob', n)
    grammar = Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    assert eager_parse(Alice, loves, Bob) == grammar << Alice @ loves @ Bob
    who = Word('who', n.r @ n @ s.l @ n)
    assert eager_parse(Bob, who, loves, Alice, target=n).offsets ==\
        [0, 1, 5, 8, 0, 2, 1, 1]
    with raises(NotImplementedError):
        eager_parse(Alice, Bob, loves)
    with raises(NotImplementedError):
        eager_parse(Alice, loves, Bob, who, loves, Alice)


def test_brute_force():
    s, n = Ty('s'), Ty('n')
    Alice = Word('Alice', n)
    loves = Word('loves', n.r @ s @ n.l)
    Bob = Word('Bob', n)
    grammar = Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    gen = brute_force(Alice, loves, Bob)
    assert next(gen) == Alice @ loves @ Alice >> grammar
    assert next(gen) == Alice @ loves @ Bob >> grammar
    assert next(gen) == Bob @ loves @ Alice >> grammar
    assert next(gen) == Bob @ loves @ Bob >> grammar
    gen = brute_force(Alice, loves, Bob, target=n)
    assert next(gen) == Word('Alice', Ty('n'))
    assert next(gen) == Word('Bob', Ty('n'))


def test_normal_form():
    n = Ty('n')
    w1, w2 = Word('a', n), Word('b', n)
    diagram = w1 @ w2 >>\
        Id(n) @ Cap(n, n.l) @ Id(n) >> Id(n @ n) @ Cup(n.l, n)
    expected_result = w1 @ w2
    assert expected_result == diagram.normal_form()\
        == (w2 >> w1 @ Id(n)).normal_form()
    assert diagram.foliation().normal_form() == diagram.normal_form()
    assert (n @ w1).normal_form() == n @ w1


def test_from_tree():
    s, n = Ty('s'), Ty('n')
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)
    sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    assert sentence == from_tree(sentence.to_tree())


def test_pregroup_swap_rotation():
    s, n = Ty('s'), Ty('n')
    assert Swap(n, s).r.dom == Swap(n, s).cod.r
    assert Swap(n, s).l.dom == Swap(n, s).cod.l
    assert Swap(n, s).l.cod == Swap(n, s).dom.l
    assert Swap(n, s).r.cod == Swap(n, s).dom.r


def test_to_hypergraph():
    s, n = Ty('s'), Ty('n')
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)
    sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    hypergraph = sentence.to_hypergraph()
    assert hypergraph.category == Diagram and hypergraph.ob == Ty
    round_trip = hypergraph.to_diagram()
    assert isinstance(round_trip, Diagram)
    assert round_trip.to_hypergraph() == hypergraph
    assert hash(round_trip.to_hypergraph()) == hash(hypergraph)


def test_foliation_with_caps():
    from discopy.tensor import Tensor
    s, n = Ty('s'), Ty('n')
    loves = Box('loves', n @ n, s)
    diagram = Cap(n.r, n) @ Cap(n, n.l) >> n.r @ loves @ n.l
    foliated = diagram.foliation()
    assert foliated.boxes == diagram.boxes and len(foliated) == 2
    assert foliated.to_staircases() == diagram
    F = Functor(ob_map={s: 1, n: 2}, ar_map={loves: [0, 1, 1, 0]}, cod=Tensor)
    assert F(foliated) == F(diagram)
