# -*- coding: utf-8 -*-

"""
SPINDLE (Kogkalidis, Moortgat and Moot, `SPINDLE: spinning raw text into
lambda terms with graph attention (2023)
<https://aclanthology.org/2023.eacl-demo.15/>`_) as an abstract categorial
grammar of CCGrebank: Tommaso Salvatori's reproduction, whose weights are
`rel-int/spindle-ccg-deberta-v3-large
<https://huggingface.co/rel-int/spindle-ccg-deberta-v3-large>`_, parses a
sentence into a :class:`discopy.lambek.abstract.proofnet.ProofNet`.

It needs PyTorch, and :meth:`Parser.from_pretrained` needs ``transformers``,
both of which the ``lambek`` extra installs.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Tagger
    Parser

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    atom

An ACG port of CCGrebank
------------------------

A category of CCGrebank with its slashes forgotten is an abstract type: the
category ``(S[dcl]\\NP)/NP`` of a transitive verb is ``NP >> (NP >> S)``, its
features dropped by :func:`atom`. The :class:`Tagger` writes the type of
each word, and of the goal of the sentence, as a binary tree of symbols
breadth first, a heap where node ``p`` has children ``2p`` and ``2p + 1``,
each node read off the state of its word refined by attention over the
sentence and by what the level above wrote. Punctuation and the ``conj`` of
coordination are left out of the linking: a word of no other atom is left
out of the sequent.

A unary rule of CCG, which changes the category of a constituent, has no
proof net: SPINDLE adds a *ghost*, a constant of type ``x >> y`` for a rule
from ``x`` to ``y``, to every span its span classifier gives such a rule with
probability ``threshold`` or more, then drops the fewest ghosts that leave
every atom given as often as it is wanted, the first such choice in the
order of :func:`itertools.product`. Every atom occurring in a type is a port
with a state, the node of the heap that wrote it, a bilinear form scores
each pair of ports and :meth:`discopy.lambek.abstract.proofnet.Sequent.decode`
takes the best linking. The Sinkhorn normalisation SPINDLE trains with only
shifts the rows and the columns of each block, so that it is not needed to
decode.

This is the deterministic parse of Tommaso's evaluation,
``SpindleParser.parse_with_ghosts`` with no resampling of the words, but for
two cases with no abstract type: a slash whose result has no linked atom,
e.g. ``,/S[dcl]``, has none, where the reproduction still wants its
argument, and a goal with no linked atom has no sequent, where the
reproduction gives a net with no links.

Example
-------

.. code-block:: python

    from huggingface_hub import snapshot_download
    from discopy.lambek.abstract.spindle import Parser

    parser = Parser.from_pretrained(
        snapshot_download("rel-int/spindle-ccg-deberta-v3-large"))
    net = parser("Alice loves Bob")
    print(net.sequent.goal, net.is_correct and net.to_term())
"""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import cast

import torch  # pylint: disable=import-error  # torch is an extra
from torch import nn  # pylint: disable=import-error  # torch is an extra
from torch.nn.utils.parametrizations import (  # pylint: disable=import-error
    orthogonal)

from discopy.lambek.abstract import Ty
from discopy.lambek.abstract.proofnet import ProofNet, Sequent, count
from discopy.lambek.categorial.bobcat import Encoder, structure

SLASHES = ("/", "\\")
UNLINKED = frozenset({
    ",", ".", ":", ";", "LRB", "RRB", "-LRB-", "-RRB-", "``", "''", "...",
    "?", "!", "LQU", "RQU", "conj"})
""" The atoms that are never linked: punctuation and coordination. """


def atom(symbol: str) -> str:
    """
    The atom of a symbol of CCGrebank, its feature dropped and ``PR`` read
    as ``PP``.

    >>> print(atom("S[dcl]"), atom("NP"), atom("PR"))
    S NP PP
    """
    index = symbol.find("[")
    base = symbol[:index] if index > 0 else symbol
    return {"PR": "PP"}.get(base, base)


class Paths(nn.Module):  # pylint: disable=abstract-method  # parameters
    """ The orthogonal maps of a left and a right child, and the root. """
    def __init__(self, dim: int):
        super().__init__()
        self.primitives = nn.Parameter(torch.randn(2, dim, dim))
        self.init = nn.Parameter(torch.randn(1, dim))


class Attention(nn.Module):
    """
    Self-attention over the words of a sentence, biased by their distance
    clipped at ``distance``.
    """
    def __init__(self, dim: int, heads: int, distance: int):
        super().__init__()
        self.heads, self.distance = heads, distance
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.dist_bias = nn.Embedding(2 * distance + 2, heads)

    def forward(self, states):
        """ The attention of each word, ``(1, words, dim)``. """
        _, n, dim = states.shape
        q, k, v = (x.view(n, self.heads, -1).transpose(0, 1)
                   for x in self.qkv(states[0]).chunk(3, dim=-1))
        index = torch.arange(n, device=states.device)
        relative = (index[:, None] - index[None]).clamp(
            -self.distance, self.distance) + self.distance + 1
        scores = q @ k.transpose(-1, -2) / math.sqrt(dim // self.heads)\
            + self.dist_bias(relative).permute(2, 0, 1)
        return self.out((scores.softmax(-1) @ v).transpose(0, 1).reshape(
            1, n, dim))


class Norm(nn.Module):
    """ The root mean square normalisation of a state. """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(dim)), eps

    def forward(self, states):
        """ The normalised states. """
        return states * states.pow(2).mean(-1, keepdim=True).add(
            self.eps).rsqrt() * self.weight


class Tagger(  # pylint: disable=too-many-instance-attributes,abstract-method
        nn.Module):
    """
    The tagger of SPINDLE over the states of the goal and of the words,
    with the span classifier of its ghosts. Its parameters have the names of
    the reproduction, so that its weights load as they are.

    Parameters:
        config : The configuration of the model, with its ``hidden_size``,
            ``decoder_dim``, ``num_levels``, ``self_heads``, ``max_dist``,
            ``span_dim`` and ``num_ghost_rules``.
        symbols : The number of symbols, the padding, the slashes and the
            atoms.
        unary : The number of unary rules of the words, which the parse does
            not use.
    """
    def __init__(self, config, symbols: int, unary: int):
        super().__init__()
        hidden, dim = config.hidden_size, config.decoder_dim
        self.levels = config.num_levels
        self.self_attn = Attention(hidden, config.self_heads, config.max_dist)
        self.attn_norm, self.ffn_norm = Norm(hidden), Norm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(),
            nn.Linear(4 * hidden, hidden))
        self.feedback_proj = nn.Linear(dim, hidden)
        self.feedback_norm = Norm(hidden)
        self.root_to_fringe = nn.Linear(hidden, dim)
        self.path_encoder = orthogonal(
            Paths(dim), name="primitives", orthogonal_map="matrix_exp")
        self.embedder = nn.Embedding(symbols, dim)
        self.linker = nn.ParameterDict({"weight": nn.Parameter(
            -torch.ones(dim) / math.sqrt(dim))})
        self.unary_head = nn.Linear(hidden, unary)
        span = config.span_dim
        self.span_start = nn.Linear(hidden, span)
        self.span_end = nn.Linear(hidden, span)
        self.span_to_rule = nn.Linear(span, max(config.num_ghost_rules, 1))
        self.span_to_root = nn.Linear(span, hidden)

    def paths(self, depth: int):
        """
        The positional encoding of each node at a depth, the product of the
        maps of its path from the root, ``(2 ** depth, dim)``.
        """
        primitives = cast(torch.Tensor, self.path_encoder.primitives)
        result = []
        for position in range(2 ** depth, 2 ** (depth + 1)):
            state = cast(torch.Tensor, self.path_encoder.init)
            while position > 1:
                state = state @ primitives[position % 2].T
                position //= 2
            result.append(state[0])
        return torch.stack(result)

    def fringe(self, roots, depth: int):
        """ The states of the nodes at a depth, ``(1, slots, nodes, dim)``. """
        return self.root_to_fringe(roots)[:, :, None] * self.paths(depth)

    @torch.no_grad()
    def decode(self, roots) -> tuple[list[list[int]], list]:
        """
        The heap of symbols of each slot, by the most likely symbol of each
        node, with the states of the nodes at each depth.

        Parameters:
            roots : The states of the goal then of the words,
                ``(1, slots, hidden)``.
        """
        heaps = torch.zeros(roots.shape[1], 2 ** self.levels, dtype=torch.long)
        active = torch.ones(1, roots.shape[1], 1, dtype=torch.bool,
                            device=roots.device)
        states, feedback = [], None
        for depth in range(self.levels):
            if feedback is not None:
                roots = self.feedback_norm(
                    roots + self.feedback_proj(feedback))
            roots = self.attn_norm(roots + self.self_attn(roots))
            roots = self.ffn_norm(roots + self.ffn(roots))
            fringe = self.fringe(roots, depth)
            states.append(fringe[0])
            logits = fringe @ self.embedder.weight.T
            logits[..., 0] = -torch.inf
            ids = torch.where(active, logits.argmax(-1), 0)
            heaps[:, 2 ** depth:2 ** (depth + 1)] = ids[0].cpu()
            nodes = self.embedder(ids) * self.paths(depth)
            weight = active[..., None].float()
            feedback = (nodes * weight).sum(2) / weight.sum(2).clamp(min=1.)
            active = ((ids == 1) | (ids == 2)).repeat_interleave(2, dim=2)
        return heaps.tolist(), states

    def spans(self, roots, spans: list[tuple[int, int]]):
        """
        The states of spans from their first to their last word,
        ``(spans, span_dim)``.
        """
        words = roots[0, 1:]
        first, last = (words[[span[i] for span in spans]] for i in (0, 1))
        return nn.functional.gelu(  # pylint: disable=not-callable  # stubs
            self.span_start(first) + self.span_end(last))

    @torch.no_grad()
    def ghosts(self, roots, threshold: float) -> list[tuple[int, int, int]]:
        """
        The spans ``(first, last, rule)`` whose most likely unary rule other
        than none has probability ``threshold`` or more, by their last word,
        then their first.
        """
        n = roots.shape[1] - 1
        spans = [(i, j) for j in range(n) for i in range(j + 1)]
        if not spans:
            return []
        probs = self.span_to_rule(self.spans(roots, spans)).softmax(-1)
        probs[:, 0] = 0.
        best, rules = probs.max(-1)
        return [(*span, rule) for span, p, rule in zip(
            spans, best.tolist(), rules.tolist()) if rule and p >= threshold]

    @torch.no_grad()
    def ghost_states(self, roots, spans: list[tuple[int, int]]) -> list:
        """ The states of the nodes of ghosts at each depth. """
        ghosts = self.span_to_root(self.spans(roots, spans))[None]
        return [self.fringe(ghosts, depth)[0] for depth in range(self.levels)]


class Parser:
    """
    The parser of SPINDLE, from words to a proof net.

    Parameters:
        tagger : The tagger.
        encoder : The encoder of the words.
        symbols : The symbol of each id, the padding first.
        rules : The ``(source, target)`` category of each unary rule of a
            ghost, from id 1.
        threshold : The probability of a ghost.
        combinations : How many choices of ghosts to drop are tried.
    """
    def __init__(  # pylint: disable=too-many-arguments
            self, tagger: Tagger, encoder: Encoder, symbols: list[str],
            rules: list[tuple[str, str]], *, threshold: float = .5,
            combinations: int = 50000):
        self.tagger, self.encoder = tagger, encoder
        self.symbols, self.rules = symbols, rules
        self.threshold, self.combinations = threshold, combinations

    @classmethod
    def from_pretrained(cls, directory, device: str = "cpu", **params
                        ) -> Parser:
        """
        A parser from the directory of a model, e.g. a download of
        `rel-int/spindle-ccg-deberta-v3-large
        <https://huggingface.co/rel-int/spindle-ccg-deberta-v3-large>`_: its
        ``atom_map.json``, ``unary_rules.json``, ``ghost_rules.json``, the
        configuration and the tokenizer of its encoder and its weights
        ``pytorch_model.bin``. It needs ``transformers``.

        Parameters:
            directory : The directory, a string or a path.
            device : Where the model runs.
            params : The other parameters of :class:`Parser`.
        """
        # pylint: disable=import-outside-toplevel  # transformers is heavy
        from transformers import (  # pylint: disable=import-error
            AutoConfig, AutoModel, AutoTokenizer)

        path = Path(directory)
        ids = json.loads((path / "atom_map.json").read_text())["id_to_sym"]
        symbols = [ids[key] for key in sorted(ids, key=int)]
        rules = {name: [tuple(rule) for rule in json.loads(
            (path / f"{name}_rules.json").read_text())["rules"]]
            for name in ("unary", "ghost")}
        config = AutoConfig.from_pretrained(path)
        tagger = Tagger(config, len(symbols), len(rules["unary"]) + 1)
        encoder = Encoder(
            AutoModel.from_config(config), AutoTokenizer.from_pretrained(path),
            config.max_position_embeddings)
        state = torch.load(path / "pytorch_model.bin", map_location="cpu")
        encoder.model.load_state_dict({
            key[len("encoder."):]: value for key, value in state.items()
            if key.startswith("encoder.")})
        tagger.load_state_dict({
            key: value for key, value in state.items()
            if not key.startswith("encoder.")})
        encoder.to(device).eval()
        tagger.to(device).eval()
        return cls(tagger, encoder, symbols, rules["ghost"], **params)

    def category(self, heap: list[int], position: int = 1
                 ) -> tuple[Ty | None, list[int]]:
        """
        The abstract type of a heap of symbols from a node, ``None`` if it
        has no linked atom, with the node of each of its ports, in order.

        A slash with no linked result has none, and one with no linked
        argument is its result.
        """
        if position >= len(heap) or not heap[position]:
            return None, []
        symbol = self.symbols[heap[position]]
        if symbol not in SLASHES:
            base = atom(symbol)
            return (None, []) if base in UNLINKED else (Ty(base), [position])
        result, nodes = self.category(heap, 2 * position)
        argument, argument_nodes = self.category(heap, 2 * position + 1)
        if result is None or argument is None:
            return result, nodes
        return argument >> result, argument_nodes + nodes

    def heap(self, category: str | tuple) -> list[int]:
        """
        The heap of a category split by
        :func:`discopy.lambek.categorial.bobcat.structure`, empty if a symbol
        is unknown or it is too deep.
        """
        heap, ids = [0] * 2 ** self.tagger.levels, {
            symbol: i for i, symbol in enumerate(self.symbols)}

        def place(node, position):
            symbol = node if isinstance(node, str) else node[0]
            if position >= len(heap) or symbol not in ids:
                return False
            heap[position] = ids[symbol]
            return isinstance(node, str) or place(node[1], 2 * position)\
                and place(node[2], 2 * position + 1)
        return heap if place(category, 1) else [0] * len(heap)

    def ghost(self, rule: int) -> tuple[str, list[int]]:
        """ The name and the heap of the ghost ``target/source`` of a rule. """
        source, target = self.rules[rule - 1]
        return f"{source} => {target}", self.heap(
            ("/", structure(target), structure(source)))

    def kept(self, given: Counter, ghosts: list[Counter]) -> list[int] | None:
        """
        The indices of the ghosts kept, dropping as few as balances the count
        ``given`` of the goal and the words, ``None`` if none does among the
        first ``combinations`` choices.
        """
        total, best = Counter(given), None
        for net in ghosts:
            total.update(net)
        choices = itertools.product((0, 1), repeat=len(ghosts))
        for i, drops in enumerate(choices):
            if i >= self.combinations:
                break
            left = total.copy()
            for net, drop in zip(ghosts, drops):
                if drop:
                    left.subtract(net)
            if any(left.values()) or best is not None\
                    and sum(drops) >= sum(best):
                continue
            best = drops
            if not any(drops):
                break
        return None if best is None else [
            k for k, drop in enumerate(best) if not drop]

    def sequent(self, words: list[str], heaps: list[list[int]],
                ghosts: list[tuple[str, list[int]]]
                ) -> tuple[Sequent, list[tuple[int, int]]] | None:
        """
        The sequent of the words and the ghosts, their heaps after the heap of
        the goal, with the slot and the node of each of its ports, ``None``
        when the goal has no linked atom.
        """
        constants, sites = [], []
        names = words + [name for name, _ in ghosts]
        for slot, heap in enumerate(heaps[1:] + [h for _, h in ghosts], 1):
            ty, nodes = self.category(heap)
            if ty is not None:
                constants.append(ty(names[slot - 1]))
                sites += [(slot, node) for node in nodes]
        goal, nodes = self.category(heaps[0])
        if goal is None:
            return None
        return Sequent(tuple(constants), goal), sites + [
            (0, node) for node in nodes]

    def __call__(self, sentence: str | list[str]) -> ProofNet | None:
        """
        The proof net of a sentence, its words split on spaces if it is a
        string, ``None`` if no choice of ghosts balances it or its goal has
        no linked atom. A sentence longer than the encoder reads is cut to
        the words it reads. The net need not be correct, see
        :attr:`discopy.lambek.abstract.proofnet.ProofNet.is_correct`.
        """
        words = sentence.split() if isinstance(sentence, str) else sentence
        if not words:
            return None
        with torch.inference_mode():
            states, mask = self.encoder([words])
            roots = torch.cat([states[:, :1], states[mask][None]], dim=1)
            words = words[:roots.shape[1] - 1]
            heaps, nodes = self.tagger.decode(roots)
            spans = self.tagger.ghosts(roots, self.threshold)
            ghosts = [self.ghost(rule) for *_, rule in spans]
            kept = self.kept(self.given(heaps), [
                self.given([heap], polarity=True) for _, heap in ghosts])
            if kept is None:
                return None
            if kept:
                nodes = [torch.cat(pair) for pair in zip(
                    nodes, self.tagger.ghost_states(
                        roots, [spans[k][:2] for k in kept]))]
            result = self.sequent(words, heaps, [ghosts[k] for k in kept])
            if result is None or not result[0].is_balanced:
                return None
            return self.link(*result, nodes)

    def given(self, heaps: list[list[int]], polarity: bool = False
              ) -> Counter:
        """
        How many more times each atom is given than wanted by heaps, the
        first the goal, wanted, unless ``polarity``.
        """
        result = Counter()
        for i, heap in enumerate(heaps):
            ty, _ = self.category(heap)
            result.update(
                {} if ty is None else count(ty, polarity or i > 0))
        return result

    def link(self, sequent: Sequent, sites: list[tuple[int, int]],
             nodes: list) -> ProofNet:
        """
        The best linking of a sequent, by the bilinear scores of the states
        ``nodes[depth][slot, i]`` of the sites of its ports.
        """
        ports = torch.stack([
            nodes[depth][slot, node - 2 ** depth] for slot, node, depth in (
                (slot, node, node.bit_length() - 1) for slot, node in sites)])
        scores = ports * self.tagger.linker["weight"] @ ports.T
        return sequent.decode(scores.cpu().double().numpy())
