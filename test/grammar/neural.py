import pytest

torch = pytest.importorskip("torch")

from discopy.grammar.abstract import Ty
from discopy.grammar.neural import *
from discopy.grammar.proofnet import ProofNet, Sequent
from discopy.utils import AxiomError

n, s = Ty("n"), Ty("s")
Alice, Bob = n("Alice"), n("Bob")
loves = (n >> (n >> s))("loves")
signature = Signature.from_types([n >> (n >> s)])


def test_Signature():
    x, y, z = map(Ty, "xyz")
    assert Signature.from_types([x >> (y >> x), z]).atoms == (x, y, z)
    for ty in [x, x >> y, (x >> y) >> (z >> x)]:
        assert Signature((x, y, z)).decode(
            Signature((x, y, z)).encode(ty)) == ty
    assert signature.arities == [2, 0, 0] and len(signature) == 3


def test_Tagger_beam():
    torch.manual_seed(0)
    tagger = Tagger(signature, hidden=8, dim=8)
    states = torch.randn(2, 3, 8)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]]) > 0
    beams = tagger.beam(states, mask, k=3, max_length=5)
    assert len(beams) == 5
    for beam in beams:
        scores = [score for _, score in beam]
        assert len(beam) == 3 and scores == sorted(scores, reverse=True)
        assert all(len(signature.encode(ty)) <= 5 for ty, _ in beam)
    assert all(ty in signature.atoms
               for beam in tagger.beam(states, mask, k=2, max_length=1)
               for ty, _ in beam)


def test_Tagger_beam_exhaustive():
    torch.manual_seed(0)
    tagger = Tagger(Signature.from_types([n, s]), hidden=8, dim=8)
    states, mask = torch.randn(1, 1, 8), torch.ones(1, 1) > 0
    (beam, ) = tagger.beam(states, mask, k=32, max_length=5)
    assert len(beam) == 2 + 4 + 16
    assert all(score == score for _, score in beam)


def test_sinkhorn():
    blocks = [torch.randn(1, 1, requires_grad=True),
              torch.randn(3, 3, requires_grad=True)]
    log_p = sinkhorn(blocks, iterations=50)
    assert log_p.shape == (2, 3, 3) and log_p[0, 0, 0] == 0
    assert torch.equal(log_p[0].exp(), torch.eye(3))
    log_p[1].diagonal().sum().backward()
    assert all(torch.isfinite(block.grad).all() for block in blocks)


def test_Linker():
    torch.manual_seed(0)
    linker = Linker(signature, hidden=8, dim=8, max_index=2)
    net = ProofNet.from_term(loves(Bob)(Alice), (Alice, loves, Bob))
    scores = linker(torch.randn(3, 8), net.sequent)
    assert scores.shape == (6, 6)
    states, mask = torch.randn(2, 4, 8), torch.tensor([[1, 1, 1, 0]] * 2) > 0
    loss = linker.loss(states, mask, [net, net])
    loss.backward()
    assert loss > 0 and torch.isfinite(linker.bilinear.grad).all()


def test_Signature_mask():
    x = Ty("x")
    assert signature.mask() == [True] * 3
    assert Signature((n, s, x)).mask([x, n]) == [True, True, False, True]


def test_Tagger_loss():
    torch.manual_seed(0)
    x = Ty("x")
    tagger = Tagger(Signature((n, s, x)), hidden=8, dim=8)
    states = torch.randn(1, 3, 8)
    mask, types = torch.ones(1, 3) > 0, [n, n >> (n >> s), n]
    outputs = []
    tagger.output.register_forward_hook(
        lambda module, inputs, output: outputs.append(output))
    tagger.smoothing = .1
    loss = tagger.loss(states, mask, types)
    targets = torch.tensor([[1] + [-1] * 4, [0, 1, 0, 1, 2], [1] + [-1] * 4])
    assert torch.allclose(loss, torch.nn.functional.cross_entropy(
        outputs[0].flatten(0, 1), targets.flatten(),
        ignore_index=-1, label_smoothing=.1))
    tagger.smoothing = 0.
    within = tagger.loss(states, mask, types, atoms=[n, s])
    within.backward()
    assert within < tagger.loss(states, mask, types)
    assert tagger.loss(
        states[:, :1], mask[:, :1], [x], atoms=[n, s]) == torch.inf


def test_Tagger_beam_atoms():
    torch.manual_seed(0)
    x = Ty("x")
    tagger = Tagger(Signature((n, s, x)), hidden=8, dim=8)
    states, mask = torch.randn(1, 4, 8), torch.ones(1, 4) > 0
    beams = tagger.beam(states, mask, k=8, max_length=5, atoms=[n, s])
    assert all(x not in Signature.from_types([ty]).atoms
               for beam in beams for ty, _ in beam)
    assert all(len(beam) == 8 for beam in beams)


def test_Parser_infers_the_goal():
    torch.manual_seed(0)
    parser = Parser(signature, hidden=8, dim=8)
    parser.tagger.beam = lambda *args, **kwargs: [
        [(n, 0.)], [(n >> (n >> s), 0.)], [(n, 0.)]]
    states = torch.randn(3, 8)
    assert parser(["Alice", "loves", "Bob"], states).sequent\
        == Sequent((Alice, loves, Bob), s)
    assert parser(["Alice", "loves"], states[:2]).sequent.goal == n >> s


def test_products():
    from itertools import product
    from random import Random
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


def test_Linker_log_likelihood():
    torch.manual_seed(0)
    linker = Linker(signature, hidden=8, dim=8)
    net = ProofNet.from_term(loves(Bob)(Alice), (Alice, loves, Bob))
    states = torch.randn(1, 3, 8)
    scores = linker(states[0], net.sequent)
    loss = linker.loss(states, torch.ones(1, 3) > 0, [net])
    assert abs(linker.log_likelihood(scores, net)
               + len(net.links) * float(loss)) < 1e-5


def test_Parser_search():
    torch.manual_seed(0)
    parser, states = Parser(signature, hidden=8, dim=8), torch.zeros(4, 8)
    comma = Ty(",")
    beams = [[(n, -.1)], [(n >> s, -.1), (n >> (n >> s), -1.)],
             [(n, -.1)], [(comma, -.1)]]
    parser.tagger.beam = lambda *args, k=1, **kwargs: [
        beam[:k] for beam in beams]
    sentence = ["Alice", "loves", "Bob", "."]
    with pytest.raises(AxiomError):
        parser(sentence[:3], states[:3])
    net = parser.search(sentence, states, k=2, width=4, skip=comma)
    assert net.sequent.words == (n("Alice"), (n >> (n >> s))("loves"),
                                 n("Bob"))
    assert net.to_term() in (loves(Bob)(Alice), loves(Alice)(Bob))
    assert parser.search(sentence, states, k=1, width=4, skip=comma) is None
    assert parser.search(sentence, states, k=2, width=1, skip=comma) is None
    assert parser.search(sentence, states, k=2, width=4) is None
