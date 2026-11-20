import json

import pytest

torch = pytest.importorskip("torch")

from discopy.lambek.categorial import Ty
from discopy.lambek.categorial.bobcat import *


def test_structure_render():
    assert structure("((NP))") == "NP"
    assert structure(r"(S\NP)/(S\NP)") == ("/", ("\\", "S", "NP"), ("\\", "S", "NP"))
    assert structure("(NP)/(N)") == ("/", "NP", "N")
    assert structure("/NP") is None and structure("") is None
    assert render(structure(r"((S\NP)/NP)")) == r"(S\NP)/NP"
    assert normalise("PR/NP") == "PP/NP" and normalise("SPR") == "SPR"


def test_schemata():
    def parents(left, right):
        return sorted(schemata(left, right))
    assert parents("NP/N", "N") == ["NP"]
    assert parents("NP", r"S\NP") == ["S"]
    assert parents("N/N", "N/N") == ["N", "N/N"]
    assert parents(r"S\S", r"S\S") == ["S", r"S\S"]
    assert parents("S/S", "S/NP") == ["S/NP"]
    assert parents(r"S\NP", r"(S\NP)\(S\NP)") == [r"S\NP"]
    assert parents(r"NP\NP", r"S\NP") == [r"S\NP"]
    assert parents(r"NP\N", r"S\NP") == [r"S\N"]
    assert parents("conj", "NP") == ["NP[conj]"]
    assert parents("NP", "conj") == ["NP[conj]"]
    assert parents("NP", "NP[conj]") == ["NP"]
    assert parents(",", "NP") == ["NP"] and parents("NP", ".") == ["NP"]
    assert parents("NP", "/N") == []


def test_Derivation():
    a, b = Derivation("NP", word="a"), Derivation(r"S\NP", word="b")
    f, g = Derivation("S/NP", word="f"), Derivation("NP/N", word="g")
    assert a.rule == "lex" and Derivation("S", children=(a, )).rule == "u"
    assert Derivation("S", children=(a, b)).rule == "ba"
    assert Derivation("S", children=(f, a)).rule == "fa"
    assert Derivation("S/N", children=(f, g)).rule == "fc"
    assert Derivation("NP", children=(a, a)).rule == "rule"
    assert Derivation("S", children=(a, b)).to_tree() == {
        "type": "ba", "cat": "S", "children": [
            {"word": "a", "cat": "NP"}, {"word": "b", "cat": r"S\NP"}]}
    diagram = Derivation("NP", children=(a, Derivation(
        "NP", children=(Derivation("N", word="c"), )))).to_diagram()
    assert diagram.cod == Ty("NP") and diagram.dom == Ty()


def test_Parser():
    data = {"binary": [["NP", r"S\NP", "S"]], "unary": [["N", "NP"]],
            "root_categories": ["S"]}
    parser = Parser.from_json(data)
    words, tags = ["a", "b"], [[("N", -1.)], [(r"S\NP", -1.)]]
    assert parser.parse(words, tags).score == -2
    with pytest.raises(ValueError):
        parser.parse(words, tags[:1])
    assert parser.parse([], []) is None
    spans = {(0, 2): [("S::S", -.5), (":", -.1)], (0, 1): [("NP", -.25)]}
    assert parser.parse(words, tags, spans).score == -2.75
    assert parser.parse(words, tags, spans, missing=False).score == -2.75
    assert parser.parse(words, tags, {}, missing=False) is None
    assert parser.parse(["a", "b"], [[("NP/N", 0.)], [("N", 0.)]], {
        (0, 1): [], (1, 2): []}, missing=False) is None
    assert Parser.from_json(data, missing=None).parse(
        words, tags, {}) is None
    assert Parser.from_json(data, root=None).parse(
        ["a"], [[("N", -1.), ("N", -2.)]]).cat == "N"
    assert Parser.from_json(data, beam=0).parse(words, tags).score == -2
    assert Parser.from_json({}).root is None
    lone = Parser.from_json(data, generic=None)
    assert lone.parents("NP/N", "N") == [] and lone.parents(
        "NP", r"S\NP") == ["S"]
    assert parser.parse(["a", "b"], [[("NP/N", 0.)], [("N", 0.)]]) is None
    wide = Parser.from_json(data, beam=1, root=None)
    cell = wide.chart(["a"], [[("N", 0.), ("NP", -1.)]], None)[0, 1]
    assert list(cell) == ["N"]
    assert Parser.labels([("NP::N", -1.), ("N", -2.)]) == {
        "NP": -1., "N": -1.}


def test_ChartClassifier():
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
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
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
