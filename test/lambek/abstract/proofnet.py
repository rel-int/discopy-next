import subprocess
from collections import Counter
import sys
from itertools import count as naturals, permutations, product
from math import factorial, inf, prod
from random import Random

from pytest import raises

from discopy.lambek.abstract import proofnet
from discopy.lambek.abstract import Ty, Variable, Abstraction
from discopy.lambek.abstract.proofnet import *
from discopy.utils import AxiomError

n, np, s = Ty("n"), Ty("np"), Ty("s")
a, b, c, e = map(Ty, "abce")
Alice, Bob = n("Alice"), n("Bob")
loves = (n >> (n >> s))("loves")


def test_lazy_torch_import():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import discopy.lambek; "
        "assert 'torch' not in sys.modules"], check=True)


def test_hungarian():
    random = Random(0)
    for size in range(6):
        for _ in range(20):
            scores = [[random.randint(-5, 5) for _ in range(size)]
                      for _ in range(size)]

            def total(assignment):
                return sum(scores[i][j] for i, j in enumerate(assignment))
            assignment = hungarian(scores)
            assert sorted(assignment) == list(range(size))
            assert total(assignment) == max(
                map(total, permutations(range(size))), default=0)


def test_round_trip():
    x = Variable("x1", np)
    every, unicorn = (n >> ((np >> s) >> s))("every"), n("unicorn")
    sleeps = (np >> s)("sleeps")
    for term, normal in [
            (loves(Bob)(Alice), loves(Bob)(Alice)),
            (every(unicorn)(sleeps),
             every(unicorn)(Abstraction(x, sleeps(x)))),
            (Abstraction(Variable("y", np), sleeps(Variable("y", np)))(
                np("John")), sleeps(np("John"))),
            (a(lambda y: y),
             Abstraction(Variable("x1", a), Variable("x1", a)))]:
        net = ProofNet.from_term(term)
        assert net.to_term() == normal and ProofNet.from_term(normal) == net


def test_repeated_words():
    the, cat, dog = (n >> np)("the"), n("cat"), n("dog")
    saw = (np >> (np >> s))("saw")
    term = saw(the(dog))(the(cat))
    net = ProofNet.from_term(term, (the, cat, saw, the, dog))
    assert net.to_term() == term
    assert net.sequent.words == (the, cat, saw, the, dog)


def test_from_term_errors():
    f = (a >> (a >> b))("f")
    with raises(AxiomError, match="closed linear"):
        ProofNet.from_term(a(lambda x: f(x)(x)))
    with raises(AxiomError, match="closed linear"):
        ProofNet.from_term(f(Variable("x", a)))
    with raises(AxiomError, match="not one of the words"):
        ProofNet.from_term(loves(Bob)(Alice), (Alice, loves, Alice))
    with raises(AxiomError, match="not used"):
        ProofNet.from_term(loves(Bob)(Alice), (Alice, loves, Bob, Bob))


def test_links():
    sequent = Sequent((Alice, loves, Bob), s)
    for links in [[(1, 4), (2, 0), (5, 3), (0, 4)],
                  [(1, 4), (2, 0), (3, 5)],
                  [(1, 4), (2, 3), (5, 0)],
                  [(1, 4), (2, 0)],
                  [(1, 4), (2, 0), (5, 6)],
                  [(1, 4), (2, 0), (-1, 3)]]:
        with raises(AxiomError):
            ProofNet(sequent, links)


def test_decode():
    sequent = Sequent((Alice, loves), s)
    with raises(AxiomError, match="wanted 2 times and given 1 times"):
        sequent.decode([[0] * 5] * 5)
    sequent = Sequent((Alice, loves, Bob), s)
    scores = [[0, 0, 0, 0, 0, 0],
              [2, 0, 0, 0, 1, 0],
              [1, 0, 0, 0, 2, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0]]
    assert sequent.decode(scores).to_term() == loves(Alice)(Bob)


def test_out_of_scope():
    g, u = ((a >> b) >> c)("g"), b("u")
    v = (a >> (c >> e))("v")
    net = ProofNet(Sequent((g, u, v), e), [(1, 3), (4, 0), (5, 2), (7, 6)])
    with raises(AxiomError, match="out of scope"):
        net.to_term()


def test_switchings():
    f = (a >> a)("f")
    identity = Sequent((f, ), a >> a)
    with raises(AxiomError, match="not connected to the goal"):
        ProofNet(identity, [(0, 1), (3, 2)]).to_term()
    with raises(AxiomError, match="out of scope"):
        ProofNet(Sequent((((a >> a) >> a)("g"), ), a),
                 [(1, 2), (3, 0)]).to_term()
    x = Variable("x1", a)
    assert ProofNet(identity, [(0, 2), (3, 1)]).to_term()\
        == Abstraction(x, f(x))


def test_count():
    assert count(a) == {a: 1} and count(a, polarity=False) == {a: -1}
    assert count((a >> b) >> c) == {a: 1, b: -1, c: 1}
    assert count(a >> a) == Counter()


def test_infer():
    assert Sequent.infer((Alice, loves, Bob)).goal == s
    assert Sequent.infer((loves, Bob)).goal == n >> s
    assert Sequent.infer(iter((Alice, loves))).words == (Alice, loves)
    for words in [(Alice, Bob), (loves, ), ()]:
        with raises(AxiomError, match="No smallest goal"):
            Sequent.infer(words)


def test_products():
    random = Random(0)
    for _ in range(20):
        beams = [sorted([(i, float(random.randint(-9, 0)))
                         for i in range(random.randint(1, 3))],
                        key=lambda item: -item[1])
                 for _ in range(random.randint(1, 4))]
        width = random.randint(1, 30)
        expected = sorted(
            (sum(beam[j][1] for beam, j in zip(beams, choice))
             for choice in product(*(range(len(beam)) for beam in beams))),
            reverse=True)[:width]
        choices = list(products(beams, width))
        assert [score for _, score in choices] == expected
        assert all(score == sum(beam[j][1] for beam, j in zip(beams, choice))
                   for choice, score in choices)
        assert len({choice for choice, _ in choices}) == len(choices)
    assert not list(products([[("a", 0.)], []], 1))
    assert not list(products([[("a", 0.)]], 0))
    beams = [[("a", 0.), ("b", -1.)], [("x", 0.), ("y", -.5)]]
    assert len(list(products(beams, -1))) == 4
    assert list(products(beams, 2.)) == list(products(beams, 2))


def test_products_lazy():
    beams = [Lazy((i, -i) for i in naturals()) for _ in range(2)]
    assert [choice for choice, _ in products(beams, 3)]\
        == [(0, 0), (0, 1), (1, 0)]
    assert [len(beam.items) for beam in beams] == [2, 3]
    unread = Lazy((i, -i) for i in naturals())
    assert not list(products([unread], 0)) and not unread.items


def test_Lazy():
    squares = Lazy(i * i for i in range(4))
    assert squares[1] == 1 and squares.items == [0, 1]
    assert squares[0] == 0 and squares.items == [0, 1]
    with raises(IndexError):
        squares[4]
    assert squares.items == [0, 1, 4, 9]
    assert Lazy([1, 2])[1] == 2
    with raises(IndexError, match="from its start"):
        Lazy([1, 2])[-1]
    with raises(IndexError, match="from its start"):
        squares[-1]
    with raises(TypeError):
        squares[0:2]


def test_assignments():
    random = Random(0)
    for size in range(5):
        for _ in range(20):
            scores = [[random.randint(-2, 2) for _ in range(size)]
                      for _ in range(size)]

            def total(columns):
                return sum(scores[i][j] for i, j in enumerate(columns))
            everything = list(permutations(range(size)))
            m = random.randint(0, len(everything) + 2)
            ranked = list(assignments(scores, m))
            assert [score for _, score in ranked] == sorted(
                map(total, everything), reverse=True)[:m]
            assert all(score == total(columns) for columns, score in ranked)
            assert {tuple(columns) for columns, _ in ranked}\
                <= set(everything)
            assert len({tuple(columns) for columns, _ in ranked})\
                == len(ranked)
            if ranked:
                assert ranked[0][0] == hungarian(scores)
    assert list(assignments([], 2)) == [([], 0.)]
    assert list(assignments([[1, 2], [3, 4]], 0)) == []
    assert len(list(assignments([[1, 2], [3, 4]], -1))) == 2


def test_assignments_impossible():
    random = Random(0)
    for size in range(5):
        for _ in range(40):
            scores = [[random.choice([-2, -1, 0, 1, 2, -inf])
                       for _ in range(size)] for _ in range(size)]

            def total(columns):
                return sum(scores[i][j] for i, j in enumerate(columns))
            everything = list(permutations(range(size)))
            m = random.randint(0, len(everything) + 2)
            ranked = list(assignments(scores, m))
            assert [score for _, score in ranked] == sorted(
                map(total, everything), reverse=True)[:m]
            assert all(score == total(columns) for columns, score in ranked)
            assert len({tuple(columns) for columns, _ in ranked})\
                == len(ranked)
    assert list(assignments([[0, -inf], [-inf, 0]], 2))\
        == [([0, 1], 0.), ([1, 0], -inf)]


def test_assignments_lazy(monkeypatch):
    calls = []

    def counted(scores):
        calls.append(scores)
        return hungarian(scores)
    monkeypatch.setattr(proofnet, "hungarian", counted)
    scores = [[3, 1, 0], [2, 1, 0], [0, 0, 1]]
    assert len(list(assignments(scores, 1))) == 1 and len(calls) == 1
    ranking = assignments(scores, 6)
    next(ranking)
    assert len(calls) == 2
    assert len(list(ranking)) == 5
    sequent = Sequent((Alice, loves, Bob), s)
    calls.clear()
    next(sequent.linkings([[0] * 6] * 6, 4))
    assert len(calls) == len(sequent.blocks) == 2


def test_linkings():
    random, tried = Random(0), 0
    pool = [a, b, a >> b, b >> a, a >> (a >> b), (a >> b) >> b,
            b >> (b >> a), (b >> a) >> a]
    for _ in range(200):
        words = [random.choice(pool)(f"w{i}")
                 for i in range(random.randint(1, 4))]
        try:
            sequent = Sequent.infer(words)
        except AxiomError:
            continue
        blocks = list(sequent.blocks.values())
        if prod(factorial(len(negatives)) for negatives, _ in blocks) > 500:
            continue
        size, tried = len(sequent.ports), tried + 1
        scores = [[random.randint(-2, 2) for _ in range(size)]
                  for _ in range(size)]

        def total(net):
            return sum(scores[i][j] for i, j in net.links)
        everything = [
            ProofNet(sequent, [
                (i, j) for (negatives, _), chosen in zip(blocks, choice)
                for i, j in zip(negatives, chosen)])
            for choice in product(*(
                permutations(positives) for _, positives in blocks))]
        m = random.randint(1, len(everything) + 2)
        ranked = list(sequent.linkings(scores, m))
        assert [total(net) for net in ranked] == sorted(
            map(total, everything), reverse=True)[:m]
        assert len({net.links for net in ranked}) == len(ranked)
        best = ProofNet(sequent, [
            (negatives[i], positives[j]) for negatives, positives in blocks
            for i, j in enumerate(hungarian(
                [[scores[x][y] for y in positives] for x in negatives]))])
        assert ranked[0] == best == sequent.decode(scores)
    assert tried > 50
    with raises(AxiomError, match="wanted 2 times and given 1 times"):
        next(Sequent((Alice, loves), s).linkings([[0] * 5] * 5, 2))
    f, x = (a >> a)("f"), a("x")
    scores = [[0, 2, -inf, 0], [0] * 4, [0] * 4, [0, -inf, 2, 0]]
    assert [net.links for net in Sequent((f, x), a).linkings(scores, 2)]\
        == [((0, 1), (3, 2)), ((0, 2), (3, 1))]


def test_is_balanced_and_is_correct():
    assert Sequent((Alice, loves, Bob), s).is_balanced
    assert not Sequent((Alice, Bob), s).is_balanced
    f, x = (a >> a)("f"), a("x")
    nets = list(Sequent((f, x), a).linkings([[0] * 4] * 4, 2))
    assert sorted(net.is_correct for net in nets) == [False, True]
    assert [net.to_term() for net in nets if net.is_correct] == [f(x)]
