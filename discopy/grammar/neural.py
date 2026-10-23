# -*- coding: utf-8 -*-

"""
Neural parsing of abstract categorial grammars into proof nets, with PyTorch.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Signature
    Tagger
    Linker
    Parser

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    sinkhorn
    products

Parsing in two steps
--------------------

A parser goes from words to a :class:`discopy.grammar.proofnet.ProofNet` in
two steps, the two halves of a proof net: the sequent, then the linking
(Kogkalidis, Moortgat and Moot, `Neural proof nets (2020)
<https://aclanthology.org/2020.conll-1.3/>`_). Both read the states of the
words given by any encoder, a tensor ``(batch, length, hidden)`` with a
boolean mask of the same batch and length saying which states are words, so
that a parser can be trained jointly with the encoder by summing its losses.

The :class:`Tagger` gives each word a type. It is constructive: rather than
picking a type out of the ones seen in training, it writes one symbol at a
time in Polish notation, the :class:`Signature` of the grammar, and it only
writes symbols that can be completed into a type, so that every output is a
type and it can output types it has never seen (Kogkalidis, Moortgat and
Deoskar, `Constructive type-logical supertagging with self-attention networks
(2019) <https://aclanthology.org/W19-4314/>`_).

The :class:`Linker` scores each pair of a negative and a positive port, from
the states of their words and what they are, and
:meth:`discopy.grammar.proofnet.Sequent.decode` takes the best linking. A
linking is a permutation of each block of ports, and a permutation matrix
is a vertex of the polytope of doubly stochastic matrices: :func:`sinkhorn`
normalises scores to the nearest such matrix in entropy, and the linker is
trained on the likelihood of the gold permutation there (Mena et al.,
`Learning latent permutations with Gumbel-Sinkhorn networks (2018)
<https://arxiv.org/abs/1802.08665>`_). The best linking does not depend on
the normalisation, so that decoding takes the raw scores. The goal of the
sentence need not be given: by default it is the smallest type that the count
invariant of the predicted types allows, see
:meth:`discopy.grammar.proofnet.Sequent.infer`.

A tagger trained on several languages at once over one signature can keep each
language within its own atoms, a sub-signature, both in its loss and in its
beam search.

Parsing as search
-----------------

Taking the most likely type of each word, then the best linking of the
sequent they make, fails whenever that sequent has no goal or that linking
no term. :meth:`Parser.search` searches the space of proof nets instead: it
enumerates the sequents of the ``k`` most likely types of each word best
first, with :func:`products`, keeps those the count invariant gives a goal,
links each with the Hungarian algorithm, keeps the nets that sequentialise,
and returns the one of highest joint score, the log-probability of its
types plus a weight times that of its linking, :meth:`Linker.log_likelihood`.

Example
-------
>>> import torch
>>> from discopy.grammar.abstract import Ty
>>> from discopy.grammar.proofnet import ProofNet
>>> n, s = Ty("n"), Ty("s")
>>> Alice, loves, Bob = n("Alice"), (n >> (n >> s))("loves"), n("Bob")
>>> nets = [ProofNet.from_term(loves(y)(x), (x, loves, y))
...         for x, y in [(Alice, Bob), (Bob, Alice)]]
>>> _ = torch.manual_seed(0)
>>> parser = Parser(Signature.from_types([n, s]), hidden=8, dim=8)
>>> encoder = torch.nn.Embedding(3, 8)
>>> sentences = torch.tensor([[0, 1, 2], [2, 1, 0]])
>>> mask = sentences >= 0
>>> optimiser = torch.optim.Adam(
...     [*parser.parameters(), *encoder.parameters()], lr=.05)
>>> for _ in range(100):
...     optimiser.zero_grad()
...     parser.loss(encoder(sentences), mask, nets).backward()
...     optimiser.step()
>>> states = encoder(sentences)[1]
>>> net = parser.eval()(["Bob", "loves", "Alice"], states)
>>> print(net.to_term())
loves(Alice)(Bob)
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import torch  # pylint: disable=import-error  # torch is an extra
from torch import nn  # pylint: disable=import-error  # torch is an extra

from discopy.grammar.abstract import Ty
from discopy.grammar.proofnet import Sequent, ProofNet, ports
from discopy.utils import AxiomError


@dataclass(frozen=True)
class Signature:
    """
    The symbols of the types of a grammar in Polish notation: the
    exponential, of arity two, then the atoms, of arity zero.

    Parameters:
        atoms : The atomic types.

    Example
    -------
    >>> x, y, z = map(Ty, "xyz")
    >>> signature = Signature((x, y, z))
    >>> signature.encode((x >> y) >> z)
    [0, 0, 1, 2, 3]
    >>> print(signature.decode([0, 0, 1, 2, 3]))
    ((x >> y) >> z)
    """
    atoms: tuple[Ty, ...]

    def __post_init__(self):
        object.__setattr__(self, "atoms", tuple(self.atoms))

    def __len__(self):
        return 1 + len(self.atoms)

    @property
    def arities(self) -> list[int]:
        "The number of arguments of each symbol."
        return [2] + [0] * len(self.atoms)

    def mask(self, atoms=None) -> list[bool]:
        """
        Whether each symbol writes types over some of the atoms, i.e. the
        exponential and these atoms, all of them by default.

        Parameters:
            atoms : The atoms of a sub-signature, e.g. those of a language.

        Example
        -------
        >>> x, y, z = map(Ty, "xyz")
        >>> Signature((x, y, z)).mask([z, x])
        [True, True, False, True]
        """
        if atoms is None:
            return [True] * len(self)
        atoms = set(atoms)
        return [True] + [atom in atoms for atom in self.atoms]

    @classmethod
    def from_types(cls, types) -> Signature:
        "The signature of the atoms of some types, in order of occurrence."
        return cls(tuple(dict.fromkeys(
            atom for ty in types for atom, _ in ports(ty))))

    def encode(self, ty: Ty) -> list[int]:
        "The symbols of a type, the exponent before the base."
        if ty.is_exp:
            return [0] + self.encode(ty.exponent) + self.encode(ty.base)
        return [1 + self.atoms.index(ty)]

    def decode(self, symbols) -> Ty:
        "The type of some symbols, the inverse of :meth:`encode`."
        symbols = iter(symbols)

        def tree():
            symbol = next(symbols)
            if symbol:
                return self.atoms[symbol - 1]
            exponent = tree()
            return exponent >> tree()
        return tree()


def words(states, mask):
    """
    The state of each word, with the states and the mask of its sentence.

    Parameters:
        states : The states of a batch, ``(batch, length, hidden)``.
        mask : Whether each state is a word, ``(batch, length)``.
    """
    sentence = mask.nonzero()[:, 0]
    return states[mask], states[sentence], mask[sentence]


class Tagger(nn.Module):
    """
    A constructive supertagger: a recurrent decoder writing the type of each
    word in Polish notation, starting from the state of the word and
    attending to the states of its sentence.

    Parameters:
        signature : The symbols of the types.
        hidden : The size of the states.
        dim : The size of the embedding of each symbol.
        dropout : The probability of dropping a feature in training.

    The :meth:`loss` is smoothed when ``smoothing``, zero by default, is set
    to the share of each target to spread over the symbols (Szegedy et al.,
    `Rethinking the inception architecture for computer vision (2016)
    <https://arxiv.org/abs/1512.00567>`_).
    """
    smoothing: float = 0.

    def __init__(self, signature: Signature, hidden: int, dim: int = 256,
                 dropout: float = 0.):
        super().__init__()
        self.signature = signature
        self.embedding = nn.Embedding(len(signature) + 1, dim)
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.query = nn.Linear(hidden, hidden)
        self.output = nn.Linear(2 * hidden, len(signature))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "arities", torch.tensor(signature.arities), persistent=False)

    def forward(self, symbols, hidden, keys, mask):
        """
        The scores of the next symbol after each of some symbols, and the
        hidden state after the last one.

        Parameters:
            symbols : The symbols written so far, ``(words, steps)``,
                ``len(signature)`` standing for the start.
            hidden : The hidden state before them, ``(1, words, hidden)``.
            keys : The states of the sentence of each word.
            mask : Whether each of these states is a word.
        """
        outputs, hidden = self.gru(
            self.dropout(self.embedding(symbols)), hidden)
        attention = torch.einsum("wsh,wkh->wsk", self.query(outputs), keys)\
            / keys.shape[-1] ** .5
        attention = attention.masked_fill(~mask[:, None], -torch.inf)
        context = torch.einsum("wsk,wkh->wsh", attention.softmax(-1), keys)
        return self.output(self.dropout(
            torch.cat([outputs, context], -1))), hidden

    def loss(self, states, mask, types, atoms=None) -> torch.Tensor:
        """
        The mean negative log-likelihood of each symbol of the types, among
        the symbols over some atoms, with a ``smoothing`` share of each
        target spread evenly over these symbols.

        Parameters:
            states : The states of a batch, ``(batch, length, hidden)``.
            mask : Whether each state is a word, ``(batch, length)``.
            types : The type of each word, in order.
            atoms : The atoms of the types, all of them by default.
        """
        hidden, keys, key_mask = words(states, mask)
        targets = nn.utils.rnn.pad_sequence([
            torch.tensor(self.signature.encode(ty)) for ty in types],
            batch_first=True, padding_value=-1).to(states.device)
        symbols = torch.cat([
            torch.full_like(targets[:, :1], -1), targets[:, :-1]], 1)
        symbols = symbols.masked_fill(symbols < 0, len(self.signature))
        allowed = torch.tensor(self.signature.mask(atoms), device=mask.device)
        log_p = self(symbols, hidden[None], keys, key_mask)[0].masked_fill(
            ~allowed, -torch.inf).log_softmax(-1)
        likelihood = log_p.gather(-1, targets.clamp(min=0)[..., None])[..., 0]
        uniform = log_p.masked_fill(~allowed, 0.).sum(-1) / allowed.sum()
        return -((1 - self.smoothing) * likelihood
                 + self.smoothing * uniform)[targets >= 0].mean()

    @torch.no_grad()
    def beam(  # pylint: disable=too-many-arguments
            self, states, mask, k: int = 1, max_length: int = 32, *,
            atoms=None) -> list[list[tuple[Ty, float]]]:
        """
        The ``k`` most likely types of each word with their log-probability,
        by beam search over the symbols that can still be completed into a
        type of at most ``max_length`` symbols over some atoms.

        Parameters:
            states : The states of a batch, ``(batch, length, hidden)``.
            mask : Whether each state is a word, ``(batch, length)``.
            k : The number of types of each word.
            max_length : The largest number of symbols of a type.
            atoms : The atoms of the types, all of them by default.
        """
        hidden, keys, key_mask = (
            x.repeat_interleave(k, 0) for x in words(states, mask))
        n_words, start = len(hidden) // k, len(self.signature)
        arities = torch.cat([self.arities, self.arities.new_ones(1)])
        over = torch.tensor(self.signature.mask(atoms), device=states.device)
        scores = torch.full((n_words, k), -torch.inf, device=states.device)
        scores[:, 0] = 0
        slots = torch.ones_like(scores, dtype=torch.long).flatten()
        history = slots.new_empty((n_words * k, 0))
        symbols, hidden = slots.new_full((n_words * k, 1), start), hidden[None]
        for step in range(max_length):
            if not slots[scores.flatten() > -torch.inf].any():
                break
            logits, hidden = self(symbols, hidden, keys, key_mask)
            allowed = over & (
                slots[:, None] + arities[None, :-1] <= max_length - step)
            log_p = logits[:, -1].masked_fill(~allowed, -torch.inf)
            log_p = log_p.where(allowed.any(-1, keepdim=True), 0.)\
                .log_softmax(-1).masked_fill(
                    ~allowed | (slots[:, None] == 0), -torch.inf)
            log_p = torch.cat([log_p, torch.where(
                slots[:, None] == 0, 0., -torch.inf)], 1)
            scores, best = (scores.flatten()[:, None] + log_p)\
                .view(n_words, -1).topk(k)
            beams = (best // (start + 1)
                     + torch.arange(n_words, device=best.device)[:, None] * k
                     ).flatten()
            symbols = (best % (start + 1)).flatten()[:, None]
            hidden, history = hidden[:, beams], torch.cat(
                [history[beams], symbols], 1)
            slots = slots[beams] + arities[symbols[:, 0]] - 1
        return [[(self.signature.decode(row[row < start].tolist()), score)
                 for row, score in zip(rows, beam.tolist())
                 if score > -torch.inf]
                for rows, beam in zip(history.view(n_words, k, -1), scores)]


def sinkhorn(blocks, iterations: int = 10) -> torch.Tensor:
    """
    The log-domain Sinkhorn normalisation of square blocks of scores,
    alternately normalising rows and columns towards a doubly stochastic
    matrix (Sinkhorn and Knopp, `Concerning nonnegative matrices and doubly
    stochastic matrices (1967) <https://doi.org/10.2140/pjm.1967.21.343>`_).

    Parameters:
        blocks : Square matrices of scores.
        iterations : The number of normalisations of rows and columns.

    Returns:
        The log-probabilities, padded into one tensor ``(blocks, n, n)``
        where each padded row only reaches its padded column.

    Example
    -------
    >>> log_p = sinkhorn([torch.randn(2, 2), torch.randn(3, 3)], 100)
    >>> assert torch.allclose(log_p.exp().sum(1), torch.ones(2, 3))
    >>> assert torch.allclose(log_p.exp().sum(2), torch.ones(2, 3))
    """
    n = max(len(block) for block in blocks)
    sizes = torch.tensor([len(block) for block in blocks])
    real = torch.arange(n)[None] < sizes[:, None]
    real = (real[:, :, None] & real[:, None]).to(blocks[0].device)
    padding = torch.eye(n, dtype=torch.bool, device=real.device) & ~real
    scores = torch.stack([
        nn.functional.pad(block, (0, n - len(block), 0, n - len(block)))
        for block in blocks]).masked_fill(~real, -torch.inf)
    scores = scores.masked_fill(padding, 0.)
    for _ in range(iterations):
        scores = scores - scores.logsumexp(2, keepdim=True)
        scores = scores - scores.logsumexp(1, keepdim=True)
    return scores


def products(beams, width: int):
    """
    The ``width`` best choices of one item in each of some lists, best first
    by the sum of their scores: the lazy enumeration of the product of lists
    sorted by score, each choice followed by the choices one step down in
    one list (Huang and Chiang, `Better k-best parsing (2005)
    <https://aclanthology.org/W05-1506/>`_).

    Parameters:
        beams : Lists of pairs of an item and its score, by decreasing score.
        width : The number of choices.

    Example
    -------
    >>> beams = [[("a", 0.), ("b", -1.)], [("x", 0.), ("y", -.5)]]
    >>> for choice, score in products(beams, 3):
    ...     print(choice, score)
    (0, 0) 0.0
    (0, 1) -0.5
    (1, 0) -1.0
    """
    if not all(beams):
        return
    start = (0, ) * len(beams)
    heap, seen = [(-sum(beam[0][1] for beam in beams), start)], {start}
    while heap and width:
        score, choice = heapq.heappop(heap)
        yield choice, -score
        width -= 1
        for i, j in enumerate(choice):
            successor = choice[:i] + (j + 1, ) + choice[i + 1:]
            if j + 1 < len(beams[i]) and successor not in seen:
                seen.add(successor)
                heapq.heappush(heap, (
                    score + beams[i][j][1] - beams[i][j + 1][1], successor))


class Linker(nn.Module):
    """
    A bilinear scorer of the pairs of ports of a sequent: each port is
    embedded from the state of its word, the mean state for the goal, with
    its atom, its polarity and its index within the type of its word, which
    tells apart the subject and the object of a verb.

    Parameters:
        signature : The atoms of the types.
        hidden : The size of the states.
        dim : The size of the embedding of each port.
        max_index : The number of indices within a type told apart.
        iterations : The number of :func:`sinkhorn` normalisations.
    """
    def __init__(self, signature: Signature, hidden: int, dim: int = 256,
                 max_index: int = 32, iterations: int = 10):
        super().__init__()
        self.signature, self.iterations = signature, iterations
        self.atom = nn.Embedding(len(signature.atoms), dim)
        self.polarity = nn.Embedding(2, dim)
        self.index = nn.Embedding(max_index, dim)
        self.port = nn.Linear(hidden + dim, dim)
        self.bilinear = nn.Parameter(nn.init.xavier_uniform_(
            torch.empty(dim, dim)))

    def forward(self, states, sequent: Sequent) -> torch.Tensor:
        """
        The score of each pair of ports, a square matrix over the ports.

        Parameters:
            states : The states of the words of the sequent.
            sequent : The words with their types and the goal.
        """
        owners = [(i, word.cod, True) for i, word in enumerate(sequent.words)]
        features = torch.tensor([
            (i, self.signature.atoms.index(atom), polarity, j)
            for i, ty, given in owners + [(len(owners), sequent.goal, False)]
            for j, (atom, polarity) in enumerate(ports(ty, given))],
            device=states.device)
        states = torch.cat([states, states.mean(0, keepdim=True)])
        rows, atoms, polarities, indices = features.T
        embedding = self.atom(atoms) + self.polarity(polarities)\
            + self.index(indices.clamp(max=self.index.num_embeddings - 1))
        ports_ = torch.tanh(self.port(torch.cat([states[rows], embedding], 1)))
        return ports_ @ self.bilinear @ ports_.T / len(self.bilinear) ** .5

    def loss(self, states, mask, nets) -> torch.Tensor:
        """
        The mean negative log-likelihood of each link of the nets, after
        :func:`sinkhorn` normalisation of each block.

        Parameters:
            states : The states of a batch, ``(batch, length, hidden)``.
            mask : Whether each state is a word, ``(batch, length)``.
            nets : The proof net of each sentence.
        """
        blocks, gold = [], []
        for sentence, real, net in zip(states, mask, nets):
            scores = self(sentence[real], net.sequent)
            partner = dict(net.links)
            for negatives, positives in net.sequent.blocks.values():
                blocks.append(scores[negatives][:, positives])
                gold.append(torch.tensor(
                    [positives.index(partner[i]) for i in negatives]))
        log_p = sinkhorn(blocks, self.iterations)
        return -torch.cat([
            block[torch.arange(len(target)), target]
            for block, target in zip(log_p, gold)]).mean()

    def log_likelihood(self, scores, net: ProofNet) -> float:
        """
        The log-probability of the links of a net after :func:`sinkhorn`
        normalisation of each block of some scores.

        Parameters:
            scores : The score of each pair of ports of its sequent.
            net : The proof net.
        """
        partner, blocks, gold = dict(net.links), [], []
        for negatives, positives in net.sequent.blocks.values():
            blocks.append(scores.detach()[negatives][:, positives])
            gold.append([positives.index(partner[i]) for i in negatives])
        log_p = sinkhorn(blocks, self.iterations)
        return float(sum(block[range(len(target)), target].sum()
                         for block, target in zip(log_p, gold)))


class Parser(nn.Module):
    """
    A parser into proof nets, a :class:`Tagger` and a :class:`Linker` over
    the states of the same encoder, called on a sentence with its states and
    its goal.

    Parameters:
        signature : The symbols of the types.
        hidden : The size of the states.
        dim : The size of the embeddings of the tagger and of the linker.
    """
    def __init__(self, signature: Signature, hidden: int, dim: int = 256):
        super().__init__()
        self.tagger = Tagger(signature, hidden, dim)
        self.linker = Linker(signature, hidden, dim)

    def loss(self, states, mask, nets, atoms=None) -> torch.Tensor:
        """
        The sum of the losses of the tagger and of the linker.

        Parameters:
            states : The states of a batch, ``(batch, length, hidden)``.
            mask : Whether each state is a word, ``(batch, length)``.
            nets : The proof net of each sentence, its words in order.
            atoms : The atoms of the types, all of them by default.
        """
        types = [word.cod for net in nets for word in net.sequent.words]
        return self.tagger.loss(states, mask, types, atoms)\
            + self.linker.loss(states, mask, nets)

    @torch.no_grad()
    def forward(self, sentence: list[str], states, goal: Ty | None = None,
                atoms=None) -> ProofNet:
        """
        The proof net of the most likely types and the best linking.

        Parameters:
            sentence : The words.
            states : Their states, ``(length, hidden)``.
            goal : The type of the sentence, by default the smallest one
                their types allow, see
                :meth:`discopy.grammar.proofnet.Sequent.infer`.
            atoms : The atoms of the types, all of them by default.
        """
        mask = states.new_ones(1, len(states), dtype=torch.bool)
        types = [beam[0][0] for beam in self.tagger.beam(
            states[None], mask, atoms=atoms)]
        constants = tuple(ty(word) for word, ty in zip(sentence, types))
        sequent = Sequent.infer(constants) if goal is None\
            else Sequent(constants, goal)
        return sequent.decode(self.linker(states, sequent).cpu())

    @torch.no_grad()
    def search(  # pylint: disable=too-many-arguments,too-many-locals
            self, sentence: list[str], states, *, goal: Ty | None = None,
            atoms=None, k: int = 1, width: int = 1, weight: float = 1.,
            skip: Ty | None = None) -> ProofNet | None:
        """
        The correct proof net of highest joint score among the ``width``
        best sequents of the ``k`` most likely types of each word, or
        ``None`` if none of them has one.

        Parameters:
            sentence : The words.
            states : Their states, ``(length, hidden)``.
            goal : The type of the sentence, by default the smallest one
                each sequent allows, see
                :meth:`discopy.grammar.proofnet.Sequent.infer`.
            atoms : The atoms of the types, all of them by default.
            k : The number of types of each word.
            width : The number of sequents tried.
            weight : The weight of the linking in the joint score.
            skip : A type whose words are left out of the sequent, e.g.
                the punctuation a treebank does not use.

        Example
        -------
        >>> n, s = Ty("n"), Ty("s")
        >>> parser = Parser(Signature.from_types([n >> (n >> s)]), 8, 8)
        >>> parser.tagger.beam = lambda *args, **kwargs: [
        ...     [(n, -.1)], [(n >> s, -.1), (n >> (n >> s), -1.)],
        ...     [(n, -.1)]]
        >>> net = parser.search(
        ...     ["Alice", "loves", "Bob"], torch.zeros(3, 8), k=2, width=2)
        >>> net.sequent.goal
        grammar.abstract.Ty(closed.Wire('s'))
        """
        mask = states.new_ones(1, len(states), dtype=torch.bool)
        beams = self.tagger.beam(states[None], mask, k=k, atoms=atoms)
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
                scores = self.linker(
                    states[torch.tensor(kept, device=states.device)],
                    sequent).cpu().float()
                net = sequent.decode(scores)
                net.to_term()
            except AxiomError:
                continue
            score = tagging + weight * self.linker.log_likelihood(scores, net)
            if score > best:
                best, result = score, net
        return result
