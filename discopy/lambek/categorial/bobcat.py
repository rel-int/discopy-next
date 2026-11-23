# -*- coding: utf-8 -*-

"""
A reproduction of Bobcat (Clark, `Something old, something new:
grammar-based CCG parsing with transformer models (2021)
<https://arxiv.org/abs/2109.10044>`_) trained on CCGrebank: a transformer
reads the words, a :class:`ChartClassifier` gives a supertag to each word and
a label to each span, and the CKY chart of a :class:`Parser` combines them
into a :class:`Derivation`, i.e. a diagram of :mod:`discopy.lambek.categorial`.

It needs PyTorch, and :meth:`Parser.from_pretrained` needs ``transformers``,
both of which the ``lambek`` extra installs.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Derivation
    Parser
    Encoder
    ChartClassifier

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        normalise
        structure
        render
        schemata
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch  # pylint: disable=import-error  # torch is an extra
from torch import nn  # pylint: disable=import-error  # torch is an extra

from discopy.lambek.categorial import Diagram, cat2ty, tree2diagram


PUNCTUATION = frozenset({
    ",", ".", ":", ";", "LRB", "RRB", "-LRB-", "-RRB-", "``", "''"})
EMPTY_SPAN = ":"
PR = re.compile(r"(?<![A-Za-z0-9_\]])PR(?![A-Za-z0-9_])")


def normalise(label: str) -> str:
    """
    The category of a label as Rebank writes it: ``NP/N`` is ``NP[nb]/N``
    and the atom ``PR`` is ``PP``.

    >>> print(normalise("NP/N"), normalise(r"(S\\NP)/PR"))
    NP[nb]/N (S\\NP)/PP
    """
    return PR.sub("PP", {"NP/N": "NP[nb]/N"}.get(label, label))


@cache
def structure(category: str) -> str | tuple | None:
    r"""
    A category as an atom or a triple ``(slash, result, argument)``, its
    slashes associating to the left, ``None`` if it is malformed. Unlike
    :func:`discopy.lambek.categorial.cat2ty`, it keeps the features.

    >>> structure(r"(S[dcl]\NP)/NP")
    ('/', ('\\', 'S[dcl]', 'NP'), 'NP')
    """
    category = category.strip()
    while category.startswith("(") and category.endswith(")"):
        depth = 0
        for i, char in enumerate(category):
            depth += {"(": 1, ")": -1}.get(char, 0)
            if depth == 0 and i != len(category) - 1:
                break
        else:
            category = category[1:-1].strip()
            continue
        break
    depth = 0
    for i in reversed(range(len(category))):
        depth += {")": 1, "(": -1}.get(category[i], 0)
        if depth == 0 and category[i] in "/\\":
            result = structure(category[:i])
            argument = structure(category[i + 1:])
            if result is None or argument is None:
                return None
            return category[i], result, argument
    return category or None


def render(category: str | tuple) -> str:
    """ The string of a category split by :func:`structure`. """
    if isinstance(category, str):
        return category
    slash, *sides = category
    result, argument = (
        render(x) if isinstance(x, str) else f"({render(x)})" for x in sides)
    return result + slash + argument


@cache
def schemata(  # pylint: disable=too-many-locals
        left: str, right: str) -> frozenset[str]:
    r"""
    The categories that the schemata of CCG derive from two adjacent ones:
    application, harmonic composition, a modifier ``X|X`` applied to
    anything, punctuation, conjunction and coordination.

    >>> sorted(schemata(r"NP", r"S[dcl]\NP"))
    ['S[dcl]']
    >>> sorted(schemata(r"S/S", r"S/NP"))
    ['S/NP']
    """
    parents = set()
    left_base, left_conj = (left[:-6], True) if left.endswith("[conj]")\
        else (left, False)
    right_base, right_conj = (right[:-6], True) if right.endswith("[conj]")\
        else (right, False)
    x, y = structure(left_base), structure(right_base)
    if x is None or y is None:
        return frozenset()
    for this, other in ((left, right_base), (right, left_base)):
        if this == "conj":
            parents.add(f"{other}[conj]")
        if this in PUNCTUATION:
            parents.add(other)
    if (left_conj or right_conj) and left_base == right_base:
        parents.add(left_base)
    (x_slash, *x_sides), (y_slash, *y_sides) = (
        c if isinstance(c, tuple) else ("", c, c) for c in (x, y))
    if x_slash == "/" and x_sides[1] == y:
        parents.add(render(x_sides[0]))
    if y_slash == "\\" and y_sides[1] == x:
        parents.add(render(y_sides[0]))
    x_modifies, y_modifies = (
        bool(slash) and sides[0] == sides[1]
        for slash, sides in ((x_slash, x_sides), (y_slash, y_sides)))
    if x_modifies:
        parents.add(right_base)
    if y_modifies:
        parents.add(left_base)
    if x_modifies and y_modifies and x == y:
        parents.add(render(x_sides[0]))
    if x_slash == y_slash == "/" and x_sides[1] == y_sides[0]:
        parents.add(render(("/", x_sides[0], y_sides[1])))
    if x_slash == y_slash == "\\" and y_sides[1] == x_sides[0]:
        parents.add(render(("\\", y_sides[0], x_sides[1])))
    return frozenset(parents)


@dataclass(frozen=True)
class Derivation:
    """
    A derivation of a span of words in the chart of a :class:`Parser`.

    Parameters:
        cat : The category it derives, as a string.
        score : Its score, a sum of log-probabilities and penalties.
        word : The word of a lexical derivation.
        children : The one or two derivations it combines.
    """
    cat: str
    score: float = 0.
    word: str | None = None
    children: tuple[Derivation, ...] = ()

    @property
    def rule(self) -> str:
        """
        The rule of the derivation, read from the types of
        :func:`discopy.lambek.categorial.cat2ty`: ``"fa"``, ``"ba"`` or
        ``"fc"`` when it is one, ``"lex"`` for a word, ``"u"`` for any other
        unary rule and ``"rule"`` for any other.
        """
        if self.word is not None:
            return "lex"
        if len(self.children) == 1:
            return "u"
        x, y, z = (cat2ty(c.cat) for c in self.children + (self, ))
        if x.is_over and x.exponent == y and x.base == z:
            return "fa"
        if y.is_under and y.exponent == x and y.base == z:
            return "ba"
        if x.is_over and y.is_over and x.exponent == y.base\
                and z == x.base << y.exponent:
            return "fc"
        return "rule"

    def to_tree(self) -> dict:
        """ The derivation as a tree in the JSON format of depccg. """
        if self.word is not None:
            return {"word": self.word, "cat": self.cat}
        return {"type": self.rule, "cat": self.cat,
                "children": [child.to_tree() for child in self.children]}

    def to_diagram(self) -> Diagram:
        """
        The derivation as a categorial diagram, see
        :func:`discopy.lambek.categorial.tree2diagram`.

        >>> alice, runs = Derivation("NP", word="Alice"), Derivation(
        ...     r"S[dcl]\\NP", word="runs")
        >>> print(Derivation("S[dcl]", children=(alice, runs)).to_diagram())
        Alice >> NP @ runs >> Eval(NP >> S)
        """
        return tree2diagram(self.to_tree())


class Parser:  # pylint: disable=too-many-instance-attributes
    r"""
    The chart parser of the reproduction of Bobcat (Clark, `Something old,
    something new: grammar-based CCG parsing with transformer models (2021)
    <https://arxiv.org/abs/2109.10044>`_) trained on CCGrebank, whose model
    is `rel-int/bobcat-rebank-deberta-v3-large
    <https://huggingface.co/rel-int/bobcat-rebank-deberta-v3-large>`_ on
    Hugging Face: a CKY chart over the binary and unary rules observed in
    the treebank, scored by a supertagger for the words and a span classifier
    for the constituents.

    A derivation scores the sum of the log-probabilities of its supertags
    and of the best label of each of its spans that has its category, less a
    penalty ``generic`` for a binary rule licensed by the schemata of
    :func:`schemata` only, and ``missing`` for a span with no such label. A
    cell keeps the best derivation of each category, then the ``beam`` best,
    and a sentence is parsed when its last cell has a ``root`` category.
    Ties go to the first derivation found, the parents of a rule taken in
    the order of their strings.

    Parameters:
        binary : The parents of each pair of categories in the treebank.
        unary : The parents of each category in the treebank.
        root : The categories of a sentence.
        beam : The size of each cell.
        generic : The penalty of a generic rule, ``None`` for none.
        missing : The penalty of a missing span, ``None`` for none.

    Example
    -------
    >>> parser = Parser.from_json({
    ...     "binary": [[r"(S[dcl]\NP)/NP", "NP", r"S[dcl]\NP"]],
    ...     "unary": [["N", "NP"]], "root_categories": ["S[dcl]"]})
    >>> tags = [[("N", -.1)], [(r"(S[dcl]\NP)/NP", -.2)], [("NP", -.3)]]
    >>> spans = {(1, 3): [(r"S[dcl]\NP", -.4)], (0, 3): [("S[dcl]", -.5)]}
    >>> tree = parser.parse("Alice loves Bob".split(), tags, spans)
    >>> print(tree.cat, tree.rule, round(tree.score, 2))
    S[dcl] ba -5.55

    The score is the sum of the five log-probabilities, less the penalty 4
    of the span of ``Alice`` as ``NP``, which has no label, and the penalty
    .05 of backward application, which the treebank has not seen here.
    >>> print(tree.to_diagram().cod)
    S
    """
    def __init__(  # pylint: disable=too-many-arguments
            self, binary: dict[tuple[str, str], frozenset[str]],
            unary: dict[str, frozenset[str]], *,
            root: frozenset[str] | None = None,
            beam: int = 256, generic: float | None = .05,
            missing: float | None = 4.):
        self.binary, self.unary, self.root = binary, unary, root
        self.beam, self.generic, self.missing = beam, generic, missing
        self.encoder = self.classifier = None
        self.decoding: dict[str, float] = {}

    @classmethod
    def from_json(cls, data: dict, **params) -> Parser:
        """
        A parser from the ``native_grammar.json`` of a model.

        Parameters:
            data : Its ``binary`` triples, ``unary`` pairs and
                ``root_categories``.
            params : The other parameters of :class:`Parser`.
        """
        binary, unary = {}, {}
        for left, right, parent in data.get("binary", []):
            binary.setdefault((left, right), set()).add(parent)
        for child, parent in data.get("unary", []):
            unary.setdefault(child, set()).add(parent)
        params.setdefault(
            "root", frozenset(data.get("root_categories", [])) or None)
        return cls({key: frozenset(value) for key, value in binary.items()},
                   {key: frozenset(value) for key, value in unary.items()},
                   **params)

    @classmethod
    def from_pretrained(  # pylint: disable=too-many-locals
            cls, directory, device: str = "cpu", **params) -> Parser:
        """
        A parser from the directory of a model, e.g. a download of
        `rel-int/bobcat-rebank-deberta-v3-large
        <https://huggingface.co/rel-int/bobcat-rebank-deberta-v3-large>`_:
        its ``native_grammar.json``, ``label_maps.json``,
        ``rebank_pipeline_config.json``, the configuration and the tokenizer
        of its encoder and its weights ``pytorch_model.bin``. It needs
        ``transformers``.

        Parameters:
            directory : The directory, a string or a path.
            device : Where the model runs.
            params : The parameters of :class:`Parser`, by default those of
                the model's pipeline.
        """
        # pylint: disable=import-outside-toplevel  # transformers is heavy
        from transformers import (  # pylint: disable=import-error
            AutoConfig, AutoModel, AutoTokenizer)

        path = Path(directory)
        pipeline = json.loads(
            (path / "rebank_pipeline_config.json").read_text())
        maps = json.loads((path / "label_maps.json").read_text())
        params.setdefault("beam", pipeline["parser"]["beam_size"])
        params.setdefault(
            "generic", pipeline["parser"]["generic_rule_penalty"]
            if pipeline["parser"]["use_generic_rules"] else None)
        params.setdefault(
            "missing", pipeline["parser"]["missing_span_penalty"])
        parser = cls.from_json(json.loads(
            (path / "native_grammar.json").read_text()), **params)
        config = AutoConfig.from_pretrained(path)
        parser.encoder = Encoder(
            AutoModel.from_config(config), AutoTokenizer.from_pretrained(path),
            config.max_position_embeddings)
        tags, labels = (sorted(maps[key], key=maps[key].get)
                        for key in ("tag2id", "cat2id"))
        parser.classifier = ChartClassifier(tags, labels, config.hidden_size)
        state = torch.load(path / "pytorch_model.bin", map_location="cpu")
        parser.encoder.model.load_state_dict({
            key[len("bert."):]: value for key, value in state.items()
            if key.startswith("bert.")})
        parser.classifier.load_state_dict({
            key: value for key, value in state.items()
            if not key.startswith("bert.")})
        parser.encoder.to(device).eval()
        parser.classifier.to(device).eval()
        parser.decoding = {
            "tag_k": pipeline["decoder"]["tag_top_k"],
            "tag_threshold": pipeline["decoder"]["tag_prob_threshold"],
            "span_k": pipeline["decoder"]["span_top_k"],
            "span_threshold": pipeline["decoder"]["span_prob_threshold"]}
        return parser

    def decode(self, sentences: list[list[str]]) -> list[tuple[list, dict]]:
        """
        The supertags and the labels of the spans of each sentence, by the
        model of :meth:`from_pretrained`, see
        :meth:`ChartClassifier.decode`.
        """
        if self.encoder is None or self.classifier is None:
            raise ValueError("no model: see Parser.from_pretrained")
        with torch.inference_mode():
            states, mask = self.encoder(sentences)
            return self.classifier.decode(states, mask, **self.decoding)

    def __call__(self, sentence: str | list[str]) -> Diagram | None:
        """
        The diagram of a sentence, its words split on spaces if it is a
        string, ``None`` if it has no parse: the best parse with a label for
        every span, else the best parse, as Bobcat's command line does. A
        sentence longer than the encoder reads is cut to the words it reads,
        as the reproduction does. Raw text is not tokenised here: split it
        with :func:`discopy.lambek.tokenise.tokenise` and call the parser on
        each of its sentences.
        """
        tree = self.derive(sentence)
        return None if tree is None else tree.to_diagram()

    def derive(self, sentence: str | list[str]) -> Derivation | None:
        """ The derivation of :meth:`__call__`. """
        words = sentence.split() if isinstance(sentence, str) else sentence
        if not words:
            return None
        tags, spans = self.decode([words])[0]
        words = words[:len(tags)]
        return self.parse(words, tags, spans, missing=False)\
            or self.parse(words, tags, spans)

    def parents(self, left: str, right: str) -> list[str]:
        """ The parents of two categories, observed or generic. """
        parents = set(self.binary.get((left, right), ()))
        if self.generic is not None:
            parents |= schemata(left, right)
        return sorted(parents)

    def span_score(self, cat: str, labels: dict[str, float] | None,
                   missing: bool) -> float | None:
        """
        The log-probability of a category in a span, see :meth:`labels`, the
        penalty of a missing span if it has none and ``missing``, else
        ``None``.
        """
        if labels is None:
            return 0.
        if cat in labels:
            return labels[cat]
        return -self.missing if missing and self.missing is not None\
            else None

    @staticmethod
    def labels(labels: list[tuple[str, float]]) -> dict[str, float]:
        """
        The best log-probability of a label of a span with each category,
        that of a chain of unary rules ``"N::S[b]\\NP"`` counting for each.
        """
        best = {}
        for label, score in labels:
            for part in label.split("::"):
                if part and part != EMPTY_SPAN:
                    part = normalise(part)
                    best[part] = max(best.get(part, score), score)
        return best

    @staticmethod
    def add(cell: dict[str, Derivation], tree: Derivation) -> bool:
        """ Keep a derivation in a cell if it beats that of its category. """
        if tree.cat in cell and cell[tree.cat].score >= tree.score:
            return False
        cell[tree.cat] = tree
        return True

    def close(self, cell: dict[str, Derivation], labels, missing: bool):
        """ Add to a cell every derivation its unary rules reach. """
        queue = list(cell.values())
        while queue:
            child = queue.pop(0)
            for parent in sorted(self.unary.get(child.cat, ())):
                score = self.span_score(parent, labels, missing)
                if score is not None and self.add(cell, Derivation(
                        parent, child.score + score, children=(child, ))):
                    queue.append(cell[parent])

    def prune(self, cell: dict[str, Derivation]) -> dict[str, Derivation]:
        """ The ``beam`` best derivations of a cell. """
        if self.beam <= 0 or len(cell) <= self.beam:
            return cell
        return dict(sorted(
            cell.items(), key=lambda item: -item[1].score)[:self.beam])

    def chart(self, words: list[str], tags: list[list[tuple[str, float]]],
              spans: dict[tuple[int, int], list[tuple[str, float]]] | None,
              missing: bool = True) -> dict[tuple[int, int], dict]:
        """
        The chart of a sentence, a cell for each span of words from its first
        to the one after its last, each cell the best derivation of each
        category.

        Parameters:
            words : The words of the sentence.
            tags : The supertags of each word with their log-probability.
            spans : The labels of each span with their log-probability,
                ``None`` for no span classifier.
            missing : Whether a span may have a category it has no label of.
        """
        if len(words) != len(tags):
            raise ValueError(
                f"{len(words)} words but {len(tags)} lists of tags")
        chart = {}
        best = {} if spans is None else {
            span: self.labels(value) for span, value in spans.items()}

        def labels(span):
            return None if spans is None else best.get(span, {})
        for i, (word, candidates) in enumerate(zip(words, tags)):
            chart[i, i + 1] = self.lexical(
                word, candidates, labels((i, i + 1)), missing)
        for width in range(2, len(words) + 1):
            for start in range(len(words) - width + 1):
                self.fill(chart, start, start + width,
                          labels=labels((start, start + width)),
                          missing=missing)
        return chart

    def fill(  # pylint: disable=too-many-arguments
            self, chart, start: int, end: int, *, labels, missing: bool):
        """ Fill the cell of a span from every split of it in two. """
        cell = {}
        for middle in range(start + 1, end):
            for left in chart.get((start, middle), {}).values():
                for right in chart.get((middle, end), {}).values():
                    self.combine(cell, left, right, labels=labels,
                                 missing=missing)
        if cell:
            self.close(cell, labels, missing)
            chart[start, end] = self.prune(cell)

    def lexical(self, word: str, candidates: list[tuple[str, float]],
                labels, missing: bool) -> dict[str, Derivation]:
        """ The cell of a word, its supertags closed under unary rules. """
        cell = {}
        for tag, score in candidates:
            self.add(cell, Derivation(normalise(tag), score, word))
        self.close(cell, labels, missing)
        return self.prune(cell)

    def combine(  # pylint: disable=too-many-arguments
            self, cell, left: Derivation, right: Derivation, *, labels,
            missing: bool):
        """ Add to a cell the derivations of two adjacent ones. """
        seen = self.binary.get((left.cat, right.cat), ())
        for parent in self.parents(left.cat, right.cat):
            score = self.span_score(parent, labels, missing)
            if score is None:
                continue
            penalty = 0. if parent in seen or self.generic is None\
                else self.generic
            self.add(cell, Derivation(
                parent, left.score + right.score + score - penalty,
                children=(left, right)))

    def parse(self, words, tags, spans=None, missing: bool = True
              ) -> Derivation | None:
        """
        The best derivation of a sentence with a root category, ``None`` if
        there is none, see :meth:`chart`.
        """
        if not words:
            return None
        cell = self.chart(words, tags, spans, missing).get((0, len(words)), {})
        candidates = [tree for tree in cell.values()
                      if self.root is None or tree.cat in self.root]
        return max(candidates, key=lambda tree: tree.score, default=None)


class Encoder(nn.Module):
    """
    An encoder of words by a transformer of Hugging Face: the state of a word
    is that of its first subword, as in Clark, `Something old, something new:
    grammar-based CCG parsing with transformer models (2021)
    <https://arxiv.org/abs/2109.10044>`_.

    Parameters:
        model : The transformer, e.g. a ``transformers.AutoModel``.
        tokenizer : Its fast tokenizer.
        length : The number of subwords it reads, the rest cut off.
    """
    def __init__(self, model: nn.Module, tokenizer, length: int = 512):
        super().__init__()
        self.model, self.tokenizer, self.length = model, tokenizer, length

    def forward(self, sentences: list[list[str]]):
        """
        The states of a batch of sentences, ``(batch, length, hidden)``, with
        whether each is the first subword of a word, ``(batch, length)``.
        """
        encodings = [self.tokenizer(
            sentence, is_split_into_words=True, truncation=True,
            max_length=self.length, return_tensors="pt")
            for sentence in sentences]
        firsts = []
        for encoding in encodings:
            ids = encoding.word_ids(batch_index=0)
            firsts.append(torch.tensor([
                i is not None and i not in ids[:j]
                for j, i in enumerate(ids)], dtype=torch.bool))
        pad = nn.utils.rnn.pad_sequence
        device = next(self.model.parameters()).device
        ids = pad([e.input_ids[0] for e in encodings], batch_first=True,
                  padding_value=self.tokenizer.pad_token_id or 0)
        attention = pad([e.attention_mask[0] for e in encodings],
                        batch_first=True)
        mask = pad(firsts, batch_first=True)
        states = self.model(input_ids=ids.to(device),
                            attention_mask=attention.to(device))
        return states.last_hidden_state, mask.to(device)


class ChartClassifier(nn.Module):
    """
    The two classifiers of Bobcat (Clark, 2021) over the states of the
    words: a supertag for each word, and a label for each span from the
    states of its first and its last word, a category or a chain of unary
    rules, ``":"`` for no constituent.

    Parameters:
        tags : The supertags.
        labels : The labels of the spans.
        hidden : The size of the states.
        dropout : The dropout before each classifier.
    """
    def __init__(self, tags: list[str], labels: list[str], hidden: int,
                 dropout: float = .1):
        super().__init__()
        self.tags, self.labels = tags, labels
        self.dropout = nn.Dropout(dropout)
        self.tag_classifier = nn.Linear(hidden, len(tags))
        self.span_classifier = nn.Linear(2 * hidden, len(labels))

    @staticmethod
    def spans(length: int) -> list[tuple[int, int]]:
        """
        The spans of a sentence, from their first word to the one after
        their last, in the order of the classifier.

        >>> ChartClassifier.spans(2)
        [(0, 1), (1, 2), (0, 2)]
        """
        return [(start, end + 1) for end in range(length)
                for start in reversed(range(end + 1))]

    def forward(self, states, mask):
        """
        The logits of the supertags of each word, ``(batch, words, tags)``,
        and of the labels of each span, ``(batch, spans, labels)``.

        Parameters:
            states : The states of a batch, ``(batch, length, hidden)``.
            mask : Whether each state is a word, ``(batch, length)``.
        """
        index = nn.utils.rnn.pad_sequence(
            [row.nonzero().squeeze(-1) for row in mask], batch_first=True)
        heads = states.gather(1, index[..., None].expand(
            *index.shape, states.shape[-1]))
        spans = [(start, end - 1) for start, end in self.spans(
            heads.shape[1])]
        pairs = heads[:, spans].flatten(start_dim=-2)
        return (self.tag_classifier(self.dropout(heads)),
                self.span_classifier(self.dropout(pairs)))

    @staticmethod
    def best(row, labels: list[str], k: int, threshold: float
             ) -> list[tuple[str, float]]:
        """
        The ``k`` most likely labels of a row of log-probabilities, within a
        ratio ``threshold`` of the most likely.
        """
        values, indices = row.topk(min(k, row.numel()) if k else row.numel())
        bound = values[0].item() + math.log(threshold)
        return [(labels[i], value) for value, i in zip(
            values.tolist(), indices.tolist()) if value >= bound]

    @torch.no_grad()
    def decode(  # pylint: disable=too-many-arguments
            self, states, mask, *, tag_k: int = 64,
            tag_threshold: float = .002, span_k: int = 128,
            span_threshold: float = .0003) -> list[tuple[list, dict]]:
        """
        The supertags of each word and the labels of each span of each
        sentence of a batch, with their log-probability, in the format of
        :meth:`Parser.chart`, no constituent left
        out.
        """
        tags, spans = (logits.log_softmax(-1).cpu()
                       for logits in self(states, mask))
        result = []
        for i, length in enumerate(mask.sum(-1).tolist()):
            result.append((
                [self.best(row, self.tags, tag_k, tag_threshold)
                 for row in tags[i, :length]],
                {span: [(label, score) for label, score in self.best(
                    spans[i, j], self.labels, span_k, span_threshold)
                    if label != ":"]
                 for j, span in enumerate(self.spans(length))}))
        return result
