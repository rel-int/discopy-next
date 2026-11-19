from random import Random

import pytest

torch = pytest.importorskip("torch")

from discopy.lambek import proofnet
from discopy.lambek.abstract import Ty
from discopy.lambek.neural import *
from discopy.lambek.proofnet import ProofNet, Sequent, hungarian
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
    assert products is proofnet.products


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


def test_Parser_search_depth():
    parser, states = Parser(signature, hidden=8, dim=8), torch.zeros(3, 8)
    beams = [[(n, -.1)], [(n >> s, -.1), (n >> (n >> s), -1.)], [(n, -.1)]]
    parser.tagger.beam = lambda *args, k=1, **kwargs: [
        beam[:k] for beam in beams]
    sentence = ["Alice", "loves", "Bob"]
    assert parser.search(sentence, states, k=2) is None
    assert parser.search(sentence, states, k=2, depth=1) is None
    net = parser.search(sentence, states, k=2, depth=2)
    assert net.sequent == Sequent((Alice, loves, Bob), s)
    assert parser.search(sentence, states, k=2, width=2) == net
    assert parser.search(sentence, states, k=2, goal=s) is None
    assert parser.search(sentence, states, k=2, goal=s, depth=2) == net
    assert parser.search(sentence, states, k=2, width=-1) == net
    assert parser.search(sentence, states, k=2, width=2.) == net
    assert parser.search(sentence, states, k=2, width=1., depth=2) == net
    assert parser.search(sentence, states, k=2, width=0, depth=2) is None


def test_Parser_search_links():
    a = Ty("a")
    parser = Parser(Signature((a, )), hidden=8, dim=8)
    parser.tagger.beam = lambda *args, **kwargs: [[(a >> a, 0.)], [(a, 0.)]]
    parser.linker.forward = lambda states, sequent: torch.tensor(
        [[0., 2, 1, 0], [0] * 4, [0] * 4, [0, 1, 2, 0]])
    sentence, states = ["f", "x"], torch.zeros(2, 8)
    assert parser.search(sentence, states) is None
    f, x = (a >> a)("f"), a("x")
    assert parser.search(sentence, states, links=2).to_term() == f(x)


@torch.no_grad()
def search_before(parser, sentence, states, goal, k, width, weight, skip):
    """
    :meth:`Parser.search` before ``depth`` and ``links``, with
    :meth:`Sequent.decode` as it was, the hungarian assignment of each block.
    """
    beams = parser.tagger.beam(states[None], None, k=k)
    best, result = -float("inf"), None
    for choice, tagging in products(beams, width):
        types = [beams[i][j][0] for i, j in enumerate(choice)]
        kept = [ty != skip for ty in types]
        constants = tuple(
            ty(word) for word, ty, keep in zip(sentence, types, kept)
            if keep)
        try:
            sequent = Sequent.infer(constants) if goal is None\
                else Sequent(constants, goal)
            scores = parser.linker(
                states[torch.tensor(kept)], sequent).cpu().float()
            if any(len(negatives) != len(positives)
                   for negatives, positives in sequent.blocks.values()):
                raise AxiomError
            net = ProofNet(sequent, [
                (negatives[i], positives[j])
                for negatives, positives in sequent.blocks.values()
                for i, j in enumerate(hungarian(
                    scores[negatives][:, positives]))])
            net.to_term()
        except AxiomError:
            continue
        score = tagging + weight * parser.linker.log_likelihood(scores, net)
        if score > best:
            best, result = score, net
    return result


def test_Parser_search_defaults():
    torch.manual_seed(0)
    random, comma = Random(0), Ty(",")
    parser = Parser(Signature.from_types([n >> (n >> s), comma]), 8, 8)
    pool = [n, s, n >> s, n >> (n >> s), (n >> s) >> s, s >> s, comma]
    found = 0
    for _ in range(200):
        beams = [sorted([(ty, round(random.uniform(-2, 0), 1))
                         for ty in random.sample(pool, random.randint(1, 3))],
                        key=lambda item: -item[1])
                 for _ in range(random.randint(1, 4))]
        parser.tagger.beam = lambda *args, k=1, beams=beams, **kwargs: [
            beam[:k] for beam in beams]
        sentence = [f"w{i}" for i in range(len(beams))]
        states = torch.randn(len(beams), 8)
        goal = random.choice([None, None, s, n >> s])
        k, width = random.randint(1, 3), random.randint(-1, 6)
        weight = random.choice([0., 1., 2.])
        skip = random.choice([None, comma])
        net = parser.search(sentence, states, goal=goal, k=k, width=width,
                            weight=weight, skip=skip)
        assert net == search_before(
            parser, sentence, states, goal, k, width, weight, skip)
        found += net is not None
    assert 20 < found < 180


def test_ChartClassifier():
    from discopy.lambek.neural import ChartClassifier
    torch.manual_seed(0)
    classifier = ChartClassifier(
        ["NP", r"S\NP"], [":", "S", "NP"], hidden=4).eval()
    states = torch.randn(1, 4, 4)
    mask = torch.tensor([[False, True, True, False]])
    tags, spans = classifier(states, mask)
    assert tags.shape == (1, 2, 2) and spans.shape == (1, 3, 3)
    (words, labels), = classifier.decode(states, mask, tag_k=1)
    assert [len(word) for word in words] == [1, 1]
    assert list(labels) == ChartClassifier.spans(2)
    assert all(label != ":" for value in labels.values()
               for label, _ in value)


def test_Parser_from_pretrained(tmp_path):
    import json
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from discopy.lambek.categorial import Parser, Ty
    from discopy.lambek.neural import ChartClassifier
    torch.manual_seed(0)
    vocab = {"[PAD]": 0, "[UNK]": 1, "Alice": 2, "runs": 3}
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]"
    ).save_pretrained(tmp_path)
    config = transformers.DebertaV2Config(
        vocab_size=4, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=8,
        max_position_embeddings=16)
    config.save_pretrained(tmp_path)
    tags, labels = ["NP", r"S\NP"], [":", "S", "NP"]
    encoder = transformers.AutoModel.from_config(config)
    state = {"bert." + key: value
             for key, value in encoder.state_dict().items()}
    state.update(ChartClassifier(tags, labels, 8).state_dict())
    torch.save(state, tmp_path / "pytorch_model.bin")
    files = {
        "native_grammar.json": {"binary": [], "unary": [],
                                "root_categories": ["S"]},
        "label_maps.json": {
            "tag2id": {tag: i for i, tag in enumerate(tags)},
            "cat2id": {label: i for i, label in enumerate(labels)}},
        "rebank_pipeline_config.json": {
            "decoder": {"tag_top_k": 64, "tag_prob_threshold": 1e-9,
                        "span_top_k": 128, "span_prob_threshold": 1e-9},
            "parser": {"beam_size": 256, "use_generic_rules": True,
                       "generic_rule_penalty": .05,
                       "missing_span_penalty": 4.}}}
    for name, data in files.items():
        (tmp_path / name).write_text(json.dumps(data))
    parser = Parser.from_pretrained(tmp_path)
    assert parser("Alice runs").cod == Ty("S")
    assert parser.derive(["Alice", "runs"]).cat == "S"
    assert parser("") is None
    with pytest.raises(ValueError):
        Parser({}, {}).decode([["Alice"]])
    assert Parser.from_pretrained(tmp_path, root=frozenset({"N"}))(
        "Alice runs") is None
