import subprocess
from collections import Counter
import sys
from itertools import permutations
from random import Random

from pytest import raises

from discopy.grammar.abstract import Ty, Variable, Abstraction
from discopy.grammar.proofnet import *
from discopy.utils import AxiomError

n, np, s = Ty("n"), Ty("np"), Ty("s")
a, b, c, e = map(Ty, "abce")
Alice, Bob = n("Alice"), n("Bob")
loves = (n >> (n >> s))("loves")


def test_lazy_torch_import():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import discopy.grammar; "
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
