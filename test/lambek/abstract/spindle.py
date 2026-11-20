import json
from collections import Counter
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from discopy.lambek.abstract import Ty
from discopy.lambek.abstract.spindle import *
from discopy.lambek.categorial.bobcat import structure

SYMBOLS = ["<pad>", "/", "\\", ",", "N", "NP", "PR", "S[dcl]"]
CONFIG = dict(hidden_size=8, decoder_dim=4, num_levels=3, self_heads=2,
              max_dist=2, span_dim=4, num_ghost_rules=2)
N, NP, PP, S = map(Ty, ("N", "NP", "PP", "S"))


def parser(**params):
    tagger = Tagger(SimpleNamespace(**CONFIG), len(SYMBOLS), 2)
    return Parser(tagger, None, SYMBOLS, [("N", "NP")], **params)


def test_category():
    p = parser()
    heap = p.heap(structure(r"(S[dcl]\NP)/NP"))
    assert heap == [0, 1, 2, 5, 7, 5, 0, 0]
    assert p.category(heap) == (NP >> (NP >> S), [3, 5, 4])
    assert p.category(p.heap(structure("NP/,"))) == (NP, [2])
    assert p.category(p.heap(structure(",/S[dcl]"))) == (None, [])
    assert p.category(p.heap(structure("PR"))) == (PP, [1])
    assert p.heap(structure("S[b]")) == [0] * 8
    assert p.heap(structure(r"((N/N)/N)/N")) == [0] * 8
    name, heap = p.ghost(1)
    assert name == "N => NP" and p.category(heap)[0] == N >> NP


def test_kept():
    p, ghost = parser(), Counter({N: -1, NP: 1})
    assert p.kept(Counter({N: 1, NP: -1}), [ghost]) == [0]
    assert p.kept(Counter({N: 1, NP: -1}), [ghost, ghost]) == [0]
    assert p.kept(Counter(), [ghost, ghost]) == []
    assert p.kept(Counter({N: 1}), [ghost]) is None
    assert parser(combinations=1).kept(Counter(), [ghost]) is None


def test_Parser_from_pretrained(tmp_path):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    torch.manual_seed(0)
    vocab = {"[PAD]": 0, "[UNK]": 1, "Alice": 2, "Bob": 3}
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]"
    ).save_pretrained(tmp_path)
    config = transformers.DebertaV2Config(
        vocab_size=4, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=8, max_position_embeddings=16, **CONFIG)
    config.save_pretrained(tmp_path)
    tagger = Tagger(config, len(SYMBOLS), 2)
    with torch.no_grad():
        # every node writes NP, and no span has a ghost
        tagger.root_to_fringe.weight.zero_()
        tagger.embedder.weight.zero_()
        tagger.embedder.weight[SYMBOLS.index("NP")] =\
            tagger.root_to_fringe.bias * tagger.paths(0)[0]
        tagger.span_to_rule.weight.zero_()
        tagger.span_to_rule.bias.copy_(torch.tensor([10., 0.]))
    encoder = transformers.AutoModel.from_config(config)
    state = {"encoder." + key: value
             for key, value in encoder.state_dict().items()}
    torch.save(state | tagger.state_dict(), tmp_path / "pytorch_model.bin")
    files = {"atom_map.json": {"id_to_sym": dict(enumerate(SYMBOLS))},
             "unary_rules.json": {"rules": [["N", "NP"]]},
             "ghost_rules.json": {"rules": [["N", "NP"]]}}
    for name, data in files.items():
        (tmp_path / name).write_text(json.dumps(data))
    parser = Parser.from_pretrained(tmp_path)
    net = parser("Alice")
    assert net.sequent.goal == NP and net.links == ((1, 0), )
    assert str(net.to_term()) == "Alice"
    assert parser("Alice Bob") is None and parser("") is None
