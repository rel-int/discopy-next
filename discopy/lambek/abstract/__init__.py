# -*- coding: utf-8 -*-

"""
Abstract categorial grammars after de Groote's `Towards abstract categorial
grammars (2001) <https://aclanthology.org/P01-1033/>`_: a vocabulary
generates a free closed category and a :class:`Lexicon` is a functor between
two of them, which is all there is to a grammar.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Exp
    Diagram
    Box
    Eval
    Coeval
    Curry
    Copy
    Discard
    Function
    Permutation
    Swap
    Feedback
    Feedforward
    Trace
    FollowedBy
    Head
    Tail
    Sum
    Functor
    Lexicon
    TermBase
    Constant
    Variable
    Application
    Abstraction

.. autodata:: Position
.. autodata:: String

.. admonition:: Submodules

    .. autosummary::
        :template: module.rst
        :nosignatures:
        :toctree:

        proofnet
        neural

:mod:`discopy.lambek.abstract.proofnet` reads a term as a proof net and back,
and is imported here. :mod:`discopy.lambek.abstract.neural` needs PyTorch,
which the ``lambek`` extra installs, so it is not: import it on its own.

Vocabularies and lexicons
-------------------------

A *vocabulary* is a higher-order signature: atomic types and constants, each
with an implicative type ``x >> y`` built from the atoms, e.g. the words of a
language with their grammatical types. It generates a free closed category,
whose morphisms are the lambda terms of its internal language, here
:class:`Term`. Nothing forces a term to be linear: the closed categories of
DisCoPy are markov by design, so a variable may be copied and discarded, as
the variables of ground type are in
the semantics of the example below; the paper's linear terms are the special
case where every variable occurs once, see ``is_linear``. A :class:`Lexicon`
from one vocabulary to another sends atomic types to types and constants to
terms of the image of their types: it is a :class:`Functor` between two free
closed categories, and lexicons compose. A lexicon is all there is to a
grammar: its abstract vocabulary is the atoms and constants it is defined on,
its abstract language the closed terms over them and its object language their
image. The paper's distinguished type ``s`` picks the sentences out of the
abstract language, the input of a parser to come.

Strings
-------

Strings are the paper's, section 4: one atomic type :data:`Position` and a
string is a map from positions to positions, a term of type
``String = Position >> Position``. A word is a constant of that type, the
empty string is the identity ``Position(lambda x: x)`` and concatenation is
composition, ``John.compose(seeks)`` for *John seeks*, see
:meth:`discopy.closed.TermBase.compose`: the paper writes it
``lambda z. x (y z)`` with the first word applied last, here the first word
is applied first, so a string reads its positions in order.

John seeks a unicorn
--------------------

The two readings of Montague's *John seeks a unicorn* share one syntax, with
a constant for each reading of *seeks* in the abstract vocabulary:

>>> n, np, s = map(Ty, ("n", "np", "s"))
>>> J, U, A = np("J"), n("U"), (n >> np)("A")
>>> S_re, S_dicto = ((np >> (np >> s))(S) for S in ("S_re", "S_dicto"))
>>> John, seeks, a, unicorn = map(
...     String, ("John", "seeks", "a", "unicorn"))
>>> seek = String(lambda x: String(lambda y: x.compose(seeks, y)))
>>> syntax = Lexicon(
...     ob_map={n: String, np: String, s: String},
...     ar_map={J: John, U: unicorn, A: String(lambda x: a.compose(x)),
...             S_re: seek, S_dicto: seek})
>>> from discopy.python import Function
>>> words = Functor(
...     {Position: list}, lambda w: lambda: lambda xs: xs + [w.name],
...     cod=Function)
>>> for reading in (S_re(J)(A(U)), S_dicto(J)(A(U))):
...     print(*words(syntax(reading))()([]))
John seeks a unicorn
John seeks a unicorn

The paper's semantic lexicon reads noun phrases as quantifiers. The image of
*a* copies its variable of ground type, so the semantic terms are not linear,
and the two readings are two closed terms of type ``t``:

>>> e, t = Ty("e"), Ty("t")
>>> Predicate, Quantifier = e >> t, (e >> t) >> t
>>> JOHN, UNICORN = e("JOHN"), Predicate("UNICORN")
>>> TRY_TO, FIND = (e >> Quantifier)("TRY_TO"), (e >> Predicate)("FIND")
>>> exists, and_ = Quantifier("exists"), (t >> (t >> t))("and")
>>> some = Predicate(lambda P: Predicate(lambda Q: exists(
...     e(lambda x: and_(P(x))(Q(x))))))
>>> seek_re = Quantifier(lambda P: Quantifier(lambda Q: Q(e(lambda x: P(
...     e(lambda y: TRY_TO(y)(e(lambda z: FIND(z)(x)))))))))
>>> seek_dicto = Quantifier(lambda P: Quantifier(lambda Q: P(e(lambda x:
...     TRY_TO(x)(e(lambda y: Q(e(lambda z: FIND(y)(z)))))))))
>>> semantics = Lexicon(
...     ob_map={n: Predicate, np: Quantifier, s: t},
...     ar_map={J: Predicate(lambda P: P(JOHN)), U: UNICORN,
...             A: some, S_re: seek_re, S_dicto: seek_dicto})
>>> assert not some.is_linear
>>> de_re, de_dicto = semantics(S_re(J)(A(U))), semantics(S_dicto(J)(A(U)))
>>> assert de_re != de_dicto and de_re.cod == de_dicto.cod == t
>>> assert not de_re.freevars and not de_dicto.freevars

Categorial grammars
-------------------

A categorial grammar is a vocabulary of its own, a planar one: a lexicon out
of it is a :class:`discopy.lambek.categorial.Functor` into abstract terms,
sending each atom to a type and each word to a term of the image of its type,
its strings written per word like every lexicon of the paper.
:meth:`Diagram.from_categorial` forgets planarity instead, collapsing left and
right exponentials into the one exponential of a closed category, so that a
derivation becomes a closed term: application evaluates, composition and type
raising curry, and crossed composition composes.

>>> from discopy.lambek import categorial
>>> N, S = categorial.Ty("n"), categorial.Ty("s")
>>> Alice, loves, Bob = N("Alice"), ((N >> S) << N)("loves"), N("Bob")
>>> LOVES = String("loves")
>>> strings = categorial.Functor(
...     ob_map={N: String, S: String},
...     ar_map={Alice: String("Alice"), Bob: String("Bob"),
...             loves: String(lambda o: String(
...                 lambda x: x.compose(LOVES, o)))},
...     cod=Diagram)
>>> print(*words(strings(Alice(loves(Bob), left=True)))()([]))
Alice loves Bob
>>> print(Alice(loves(Bob), left=True).to_abstract())
loves(Bob)(Alice)

From English to Lean
--------------------

Two lexicons on one abstract vocabulary make a translation: parse with one,
print with the other. The abstract types are a kind of structure ``a``, nouns,
noun phrases and sentences, and *free* is a construction on a kind, the left
adjoint of a forgetful functor, rather than an adjective on a noun:

>>> a = Ty("a")
>>> MONOIDAL, PREMONOIDAL, FREE = a("MONOIDAL"), a("PREMONOIDAL"), (
...     a >> n)("FREE")
>>> DIAGRAMS, DISCOPY, ARROWS = n("DIAGRAMS"), (n >> n)("DISCOPY"), (
...     np >> n)("ARROWS")
>>> THE, BARE, ARE = (n >> np)("THE"), (n >> np)("BARE"), (
...     np >> (np >> s))("ARE")
>>> def sentence(kind):
...     return ARE(THE(ARROWS(THE(FREE(kind)))))(BARE(DISCOPY(DIAGRAMS)))
>>> W = {word: String(word) for word in (
...     "DisCoPy diagrams are the arrows of free category".split()
...     + ["monoidal", "premonoidal"])}
>>> english = Lexicon(
...     ob_map={a: String, n: String, np: String, s: String},
...     ar_map={MONOIDAL: W["monoidal"], PREMONOIDAL: W["premonoidal"],
...             FREE: String(lambda x: W["free"].compose(x, W["category"])),
...             DIAGRAMS: W["diagrams"],
...             DISCOPY: String(lambda x: W["DisCoPy"].compose(x)),
...             ARROWS: String(lambda x: W["arrows"].compose(W["of"], x)),
...             THE: String(lambda x: W["the"].compose(x)),
...             BARE: String(lambda x: x),
...             ARE: String(lambda o: String(
...                 lambda x: x.compose(W["are"], o)))})
>>> print(*words(english(sentence(MONOIDAL)))()([]))
DisCoPy diagrams are the arrows of the free monoidal category

The Lean side is Montague's, with Lean as its logic: ``e`` a Lean term, ``t``
a Lean proposition and ``k`` a Lean structure, the model of ``discolean/``.
A noun phrase holds of two predicates, a weak and a strong one, and a
determiner picks one of them, the weak one for a bare noun and the strong one
for *the*. *Are* is the embedding of its subject into its object
for a bare plural object, *diagrams are arrows of* ``C``, and an equation for
a definite one, *diagrams are the arrows of* ``C``: *the* takes all of them.
*The free* ``k`` *category* is ``Diagram.k S``, *free* contributes that a
functor out of it extends any map of generators and *the* that the extension
is unique, which is what makes the free category unique:

>>> k = Ty("k")
>>> StrictMonoidal, StrictPremonoidal = k("StrictMonoidal"), k(
...     "StrictPremonoidal")
>>> diagram, hom, free = e("diagram"), (e >> e)("hom"), (k >> e)("free")
>>> lift, unique = (k >> (e >> t))("lift"), (k >> (e >> t))("unique")
>>> eq, embeds = (e >> (e >> t))("eq"), (e >> (e >> t))("embeds")
>>> and_ = (t >> (t >> t))("and")
>>> Determiner = Predicate >> (Predicate >> Predicate)
>>> Phrase = Predicate >> (Predicate >> t)
>>> Noun = Determiner >> Phrase
>>> def noun(term):
...     return Determiner(lambda D: Predicate(lambda P: Predicate(
...         lambda R: term(D, P, R))))
>>> def free_category(x, D, P, R):
...     extends = lift(x)(free(x))
...     return D(e(lambda c: and_(P(c))(extends)))(e(lambda c: and_(
...         and_(R(c))(extends))(unique(x)(c))))(free(x))
>>> def arrows(X, D, P, R):
...     arrow = e(lambda c: D(P)(R)(hom(c)))
...     return X(arrow)(arrow)
>>> def are(O, X):
...     predicate = e(lambda x: O(e(lambda y: embeds(x)(y)))(
...         e(lambda y: eq(y)(x))))
...     return X(predicate)(predicate)
>>> to_lean = Lexicon(
...     ob_map={a: k, n: Noun, np: Phrase, s: t},
...     ar_map={MONOIDAL: StrictMonoidal, PREMONOIDAL: StrictPremonoidal,
...             FREE: k(lambda x: noun(
...                 lambda D, P, R: free_category(x, D, P, R))),
...             DIAGRAMS: noun(lambda D, P, R: D(P)(R)(diagram)),
...             DISCOPY: Noun(lambda N: N),
...             ARROWS: Phrase(lambda X: noun(
...                 lambda D, P, R: arrows(X, D, P, R))),
...             THE: Noun(lambda N: N(Predicate(
...                 lambda P: Predicate(lambda R: R)))),
...             BARE: Noun(lambda N: N(Predicate(
...                 lambda P: Predicate(lambda R: P)))),
...             ARE: Phrase(lambda O: Phrase(lambda X: are(O, X)))})

``Predicate`` was ``e >> t``, so ``Predicate(lambda P: ...)`` binds a
predicate. A functor into :class:`discopy.python.Function` prints the Lean,
evaluating the lambdas as it goes:

>>> LIFT = '''(∀ (C : {k}) (ob : S.Ob → C.Ob) (ar : S.Box → C.Hom),
...     (∀ b, C.dom (ar b) = C.prod ((S.dom b).map ob)) →
...     (∀ b, C.cod (ar b) = C.prod ((S.cod b).map ob)) →
...     ∃ F : Functor ({c}) C,
...       (∀ x, F.ob [x] = ob x) ∧ ∀ b, F.map (Diagram.box b) = ar b)'''
>>> UNIQUE = '''∀ (C : {k}) (F G : Functor ({c}) C),
...     (∀ x, F.ob [x] = G.ob [x]) →
...     (∀ b, F.map (Diagram.box b) = G.map (Diagram.box b)) → F = G'''
>>> lean = Functor({e: str, t: str, k: str}, {
...     StrictMonoidal: lambda: "StrictMonoidal",
...     StrictPremonoidal: lambda: "StrictPremonoidal",
...     diagram: lambda: "Diagram S",
...     hom: lambda: lambda c: f"({c}).Hom",
...     free: lambda: lambda x: f"Diagram.{x[6:].lower()} S",
...     lift: lambda: lambda x: lambda c: LIFT.format(k=x, c=c),
...     unique: lambda: lambda x: lambda c: UNIQUE.format(k=x, c=c),
...     eq: lambda: lambda x: lambda y: f"{x} = {y}",
...     embeds: lambda: lambda x: lambda y: f"Nonempty ({x} ↪ {y})",
...     and_: lambda: lambda p: lambda q: f"{p}\\n  ∧ {q}"}, cod=Function)
>>> print(lean(to_lean(sentence(MONOIDAL)))())
(Diagram.monoidal S).Hom = Diagram S
  ∧ (∀ (C : StrictMonoidal) (ob : S.Ob → C.Ob) (ar : S.Box → C.Hom),
    (∀ b, C.dom (ar b) = C.prod ((S.dom b).map ob)) →
    (∀ b, C.cod (ar b) = C.prod ((S.cod b).map ob)) →
    ∃ F : Functor (Diagram.monoidal S) C,
      (∀ x, F.ob [x] = ob x) ∧ ∀ b, F.map (Diagram.box b) = ar b)
  ∧ ∀ (C : StrictMonoidal) (F G : Functor (Diagram.monoidal S) C),
    (∀ x, F.ob [x] = G.ob [x]) →
    (∀ b, F.map (Diagram.box b) = G.map (Diagram.box b)) → F = G

That statement does not elaborate: ``discolean/`` has no ``StrictMonoidal``,
and its ``not_monoidal`` says why, two boxes side by side are two diagrams, as
the tensor of diagrams does the first one first. DisCoPy diagrams are the
arrows of the free *premonoidal* category, which
``discolean/DisCoLean/Statements/English.lean`` proves in the words the
grammar prints:

>>> print(*words(english(sentence(PREMONOIDAL)))()([]))
DisCoPy diagrams are the arrows of the free premonoidal category
>>> print(lean(to_lean(sentence(PREMONOIDAL)))())
(Diagram.premonoidal S).Hom = Diagram S
  ∧ (∀ (C : StrictPremonoidal) (ob : S.Ob → C.Ob) (ar : S.Box → C.Hom),
    (∀ b, C.dom (ar b) = C.prod ((S.dom b).map ob)) →
    (∀ b, C.cod (ar b) = C.prod ((S.cod b).map ob)) →
    ∃ F : Functor (Diagram.premonoidal S) C,
      (∀ x, F.ob [x] = ob x) ∧ ∀ b, F.map (Diagram.box b) = ar b)
  ∧ ∀ (C : StrictPremonoidal) (F G : Functor (Diagram.premonoidal S) C),
    (∀ x, F.ob [x] = G.ob [x]) →
    (∀ b, F.map (Diagram.box b) = G.map (Diagram.box b)) → F = G

From Lean to English
--------------------

The same two lexicons read the other way explain a Lean statement in English:
parse it with the Lean side, :meth:`Lexicon.parse`, and print its parses with
the English side. Lean picks the articles of the object, *the arrows of the
free* category being an equation and a uniqueness, but not that of the
subject, *diagrams* being ``Diagram S`` either way, and it has nothing for
*DisCoPy*, which goes to the identity, so that a statement has as many
readings as the depth lets that word repeat:

>>> theorem = lean(to_lean(sentence(PREMONOIDAL)))()
>>> readings = to_lean.parse(theorem, s, 6, lean)
>>> explanations = [" ".join(words(english(r))()([])) for r in readings]
>>> len(explanations)
8
>>> print(explanations[0])
the diagrams are the arrows of the free premonoidal category

:meth:`Lexicon.likelihood` picks one, the most likely under a model of
derivations counted in a treebank, here the one sentence about monoidal
categories: it has seen *DisCoPy* once after a bare noun and never after
itself, nor *the* in the subject, and never *premonoidal*, which every
reading shares.

>>> treebank = [sentence(MONOIDAL)]
>>> best = max(readings, key=lambda r: to_lean.likelihood(r, treebank))
>>> print(*words(english(best))()([]))
DisCoPy diagrams are the arrows of the free premonoidal category

Only the kind tells the statements apart, so the one Lean refutes reads back
as the same paraphrases with *monoidal*:

>>> refuted = lean(to_lean(sentence(MONOIDAL)))()
>>> assert sorted(" ".join(words(english(r))()([]))
...               for r in to_lean.parse(refuted, s, 6, lean)) == sorted(
...     x.replace("premonoidal", "monoidal") for x in explanations)
"""

from __future__ import annotations

from collections import Counter
from math import log
from typing import ClassVar

from discopy import closed, feedback, cmap, hypergraph
from discopy.cat import factory
from discopy.lambek import categorial


@factory
class Ty(closed.Ty):
    "An implicative type, the type of an abstract categorial grammar."
    variable_factory: ClassVar[type["Variable"]]
    constant_factory: ClassVar[type["Constant"]]
    application_factory: ClassVar[type["Application"]]
    abstraction_factory: ClassVar[type["Abstraction"]]

    @classmethod
    def from_categorial(cls, old: categorial.Ty) -> Ty:
        """
        Translate a categorial type into an abstract type, collapsing left
        and right exponentials into a single exponential.

        Parameters:
            old : The categorial type to translate.
        """
        return cls.from_biclosed(old)


class Exp(closed.Exp):
    "An exponential object in an abstract categorial grammar."

    ob = Ty


@factory
class Diagram(closed.Diagram):
    """
    An abstract diagram is a closed diagram with words as boxes, i.e. a
    derivation in an abstract categorial grammar.

    The rules of a categorial grammar are its closed structure: application
    is evaluation, composition is the currying of two evaluations and crossed
    composition coincides with composition, since a closed category has one
    exponential.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    box_factory: ClassVar[type["Box"]]
    functor_factory: ClassVar[type["Functor"]]
    copy_factory: ClassVar[type["Copy"]]
    permutation_factory: ClassVar[type["Permutation"]]
    swap_factory: ClassVar[type["Swap"]]
    curry_factory: ClassVar[type["Curry"]]
    eval_factory: ClassVar[type["Eval"]]
    coeval_factory: ClassVar[type["Coeval"]]
    trace_factory: ClassVar[type["Trace"]]  # ty: ignore[invalid-type-form]
    discard_factory: ClassVar[type["Discard"]]
    sum_factory: ClassVar[type["Sum"]]
    ob = Ty

    @classmethod
    def fa(cls, left: Ty, right: Ty) -> Diagram:
        "Forward application ``(left << right) @ right -> left``."
        return cls.ev(left, right, left=True)

    @classmethod
    def ba(cls, left: Ty, right: Ty) -> Diagram:
        "Backward application ``left @ (left >> right) -> right``."
        return cls.ev(right, left, left=False)

    @classmethod
    def fc(cls, left: Ty, middle: Ty, right: Ty) -> Diagram:
        """
        Forward composition
        ``(left << middle) @ (middle << right) -> left << right``.
        """
        return (cls.id(left << middle) @ cls.fa(middle, right)
                >> cls.fa(left, middle)).curry(len(right), left=True)

    @classmethod
    def bc(cls, left: Ty, middle: Ty, right: Ty) -> Diagram:
        """
        Backward composition
        ``(left >> middle) @ (middle >> right) -> left >> right``.
        """
        return (cls.ba(left, middle) @ cls.id(middle >> right)
                >> cls.ba(middle, right)).curry(len(left), left=False)

    @classmethod
    def fx(cls, left: Ty, middle: Ty, right: Ty) -> Diagram:
        "Forward crossed composition, which coincides with :meth:`fc`."
        return cls.fc(left, middle, right)

    @classmethod
    def bx(cls, left: Ty, middle: Ty, right: Ty) -> Diagram:
        "Backward crossed composition, which coincides with :meth:`bc`."
        return cls.bc(left, middle, right)

    @classmethod
    def from_categorial(cls, diagram: categorial.Diagram) -> Diagram:
        """
        The abstract diagram of a categorial diagram, or the abstract term of
        a categorial term, dropping planarity: left and right exponentials
        collapse, constants stay constants and words become boxes, crossed
        compositions become compositions and the composition and type-raising
        terms become lambda terms.

        Parameters:
            diagram : The categorial diagram or term to translate.

        Example
        -------
        >>> from discopy.lambek import categorial
        >>> n, s = categorial.Ty("n"), categorial.Ty("s")
        >>> Alice, loves, Bob = n("Alice"), ((n >> s) << n)("loves"), n("Bob")
        >>> print(Diagram.from_categorial(Alice(loves(Bob), left=True)))
        loves(Bob)(Alice)
        """
        functor = categorial.Functor(
            ob_map=lambda x: cls.ob(x.inside[0].name),
            ar_map=lambda box: cls.ob.constant_factory(
                box.name, functor(box.cod))
            if isinstance(box, categorial.Constant)
            else Box(box.name, functor(box.dom), functor(box.cod)),
            cod=cls)
        return functor(diagram)


class Box(closed.Box, Diagram):
    "An abstract box is a closed box in an abstract diagram."


class Eval(closed.Eval, Box):
    "The evaluation of an exponential type."


class Coeval(closed.Coeval, Box):
    "The coevaluation of an exponential type, i.e. the dagger of an Eval."


class Curry(closed.Curry, Box):
    "The currying of an abstract diagram."


class Copy(closed.Copy, Box):
    "The copy of an abstract type, or its discard when ``n=0``."


class Discard(closed.Discard, Copy):
    "The discard of an abstract type, i.e. a copy with zero legs."


class Function(closed.Function, Box):
    """
    A markov function in an abstract diagram, which copies or discards, so
    that plumbing tensors into a function of this level rather than of
    :mod:`discopy.closed`.

    >>> x = Ty('x')
    >>> assert (Diagram.discard(x) @ Id(x)).dom == x @ x
    """


class Permutation(closed.Permutation, Box):
    "A permutation of abstract types."


Function.register(Permutation)


class Swap(Permutation, closed.Swap, Box):
    "The symmetric swap of two abstract types."


class Sum(closed.Sum, Box):
    "A formal sum of abstract diagrams."


class Functor(closed.Functor):
    """
    An abstract functor is a closed functor with abstract diagrams as domain.

    Parameters:
        ob_map (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram


class Lexicon(Functor):
    """
    A lexicon is a functor from one vocabulary to another, i.e. between two
    free closed categories: it sends atomic types to types and constants to
    terms of the image of their types, section 2.2 of de Groote's `Towards
    abstract categorial grammars (2001) <https://aclanthology.org/P01-1033/>`_.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from atomic types to types.
        ar_map (Mapping[Constant, Term]) : Map from constants to terms.

    Lexicons compose: ``first >> second`` sends a constant ``c`` to
    ``second(first(c))``, with the target vocabulary of the first lexicon
    serving as the source vocabulary of the second.

    Example
    -------
    >>> x, y, z = map(Ty, "xyz")
    >>> first = Lexicon(ob_map={x: y}, ar_map={x("a"): y("b")})
    >>> second = Lexicon(ob_map={y: z}, ar_map={y("b"): z("c")})
    >>> assert first(x("a")) == y("b") and (first >> second)(x("a")) == z("c")
    >>> Lexicon(ob_map={x: y}, ar_map={x("a"): (y >> y)("b")})(x("a"))
    Traceback (most recent call last):
        ...
    discopy.utils.AxiomError: Expected a term of type y for a, got ...
    """
    dom = cod = Diagram

    def parse(self, target, start: Ty, depth: int,
              functor: closed.Functor) -> list[Term]:
        """
        The abstract terms of type ``start``, with constants nested at most
        ``depth`` deep, that the lexicon then ``functor`` send to ``target``:
        parsing inverts a lexicon, and gives a list since a lexicon need not
        be injective, de Groote's section 5.

        The terms are generated from the constants of the lexicon, its
        ``ar_map`` being a mapping, and evaluated bottom up: ``functor`` sends
        the image of each constant to a :class:`discopy.python.Function` once,
        and the value of an application is the application of their values,
        since a functor preserves it, unpacked when the argument is several
        wires. The terms are constants applied to such terms, with no
        abstraction: a higher-order lexicon, e.g. a tree-adjoining grammar
        with an argument ``lambda x: B(x)(C)``, has parses it does not find.

        Parameters:
            target : The value to parse, e.g. a string.
            start : The abstract type of the terms, e.g. sentences.
            depth : The largest number of nested constants.
            functor : A functor into :class:`discopy.python.Function`.

        Example
        -------
        >>> from discopy.python import Function
        >>> n, s, e, t = map(Ty, "nset")
        >>> bank, shore, visit = n("bank"), n("shore"), (n >> s)("visit")
        >>> RIVER, VISIT = e("RIVER"), (e >> t)("VISIT")
        >>> semantics = Lexicon(ob_map={n: e, s: t}, ar_map={
        ...     bank: RIVER, shore: RIVER, visit: VISIT})
        >>> show = Functor({e: str, t: str}, {
        ...     RIVER: lambda: "river", VISIT: lambda: lambda x: f"visit {x}"},
        ...     cod=Function)
        >>> print(*semantics.parse("visit river", s, 2, show), sep="\\n")
        visit(bank)
        visit(shore)
        """
        values = {constant: functor(self(constant))()
                  for constant in self.ar_map.mapping}
        language: dict[tuple[Ty, int], list] = {}
        wires: dict[Ty, int] = {}

        def terms(cod, n):
            if (cod, n) not in language:
                language[cod, n] = [] if n <= 0 else [
                    pair for constant, value in values.items()
                    for pair in applied(constant, value, constant.cod, cod, n)]
            return language[cod, n]

        def applied(term, value, dom, cod, n):
            if dom == cod:
                yield term, value
            if len(dom) == 1 and isinstance(exp := dom.inside[0], closed.Exp):
                if exp.exponent not in wires:
                    wires[exp.exponent] = len(functor(self(exp.exponent)))
                for arg, arg_value in terms(exp.exponent, n - 1):
                    yield from applied(term(arg), value(
                        *arg_value) if wires[exp.exponent] != 1
                        else value(arg_value), exp.base, cod, n)
        return [term for term, value in terms(start, depth)
                if value == target]

    @staticmethod
    def productions(term: closed.Term, parent: closed.Term | None = None,
                    position: int = 0):
        """
        The productions of a derivation, i.e. of a term of constants applied
        to such terms: for each constant, its parent, its position among the
        arguments of the parent and itself, the root having no parent.

        Parameters:
            term : The derivation.
            parent : The constant the term is an argument of.
            position : Its position among the arguments of the parent.

        Example
        -------
        >>> n, s = Ty("n"), Ty("s")
        >>> Alice, sleeps = n("Alice"), (n >> s)("sleeps")
        >>> assert list(Lexicon.productions(sleeps(Alice))) == [
        ...     (None, 0, sleeps), (sleeps, 0, Alice)]
        """
        args = []
        while isinstance(term, Application):
            term, args = term.function, [term.argument] + args
        yield parent, position, term
        for i, arg in enumerate(args):
            yield from Lexicon.productions(arg, term, i)

    def likelihood(self, term: Term, treebank: list[Term],
                   smoothing: float = 0.1) -> float:
        """
        The log-likelihood of a derivation, where each argument of a constant
        is headed by a constant with a probability given by the parent and
        the position, i.e. a probabilistic grammar of the derivations of the
        lexicon, estimated by counting in a treebank with Lidstone smoothing
        over the constants of the lexicon. Conditioning on the parent is what
        makes a word likely once and unlikely after itself, which a weight
        per constant cannot.

        Parameters:
            term : The derivation, see :meth:`productions`.
            treebank : The derivations to count.
            smoothing : The count added to every production.

        Example
        -------
        >>> n, s = Ty("n"), Ty("s")
        >>> Alice, Bob, sleeps = n("Alice"), n("Bob"), (n >> s)("sleeps")
        >>> lexicon = Lexicon({n: n, s: s}, {Alice: Alice, Bob: Bob,
        ...                                 sleeps: sleeps})
        >>> treebank = [sleeps(Alice), sleeps(Alice), sleeps(Bob)]
        >>> assert lexicon.likelihood(sleeps(Alice), treebank)\\
        ...     > lexicon.likelihood(sleeps(Bob), treebank)
        """
        counts = Counter(production for derivation in treebank
                         for production in self.productions(derivation))
        totals = Counter((parent, position)
                         for parent, position, _ in counts.elements())
        size = len(self.ar_map.mapping)
        return sum(
            log((counts[parent, position, head] + smoothing)
                / (totals[parent, position] + smoothing * size))
            for parent, position, head in self.productions(term))


CMap = cmap.CMap[Diagram]


class TermBase(Box, closed.TermBase):
    "A term in the internal language of an abstract categorial grammar."
    functor = Functor.id(Diagram)
    strategy = classmethod(closed.TermBase.strategy.__func__)


class Constant(TermBase, closed.Constant):
    "A constant of the vocabulary of an abstract categorial grammar."


Word = Constant


class Variable(TermBase, closed.Variable):
    "A variable term in an abstract categorial grammar."


class Application(TermBase, closed.Application):
    "The application of an abstract term to another."


class Abstraction(TermBase, closed.Abstraction):
    "The abstraction of a variable in an abstract term."


type Term = Constant | Variable | Application | Abstraction

Id = Diagram.id
Feedback, Feedforward, Trace, FollowedBy, Head, Tail = \
    feedback.structural_boxes(Diagram, Box)
Diagram.box_factory, Diagram.functor_factory = Box, Functor
Diagram.copy_factory = Copy
Diagram.function_factory = Function
Diagram.permutation_factory = Permutation
Diagram.swap_factory = Swap
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
Diagram.coeval_factory = Coeval
Diagram.discard_factory = Discard
Diagram.sum_factory = Sum
Ty.under_factory = Ty.over_factory = staticmethod(Exp)
Ty.variable_factory = Variable
Ty.constant_factory = Constant
Ty.application_factory = Application
Ty.abstraction_factory = Abstraction
Hypergraph = hypergraph.Hypergraph[Diagram]
for law in (
        "dagger_involution", "dagger_contravariance", "dagger_monoidality"):
    setattr(Hypergraph, law, vars(closed.Hypergraph)[law])

Position = Ty("o")
""" The atomic type of positions in a string, an arbitrary atom. """

String = Position >> Position
"""
The type of strings, maps from positions to positions: a word is a constant
of type ``String``, ``Position(lambda x: x)`` is the empty string and
``John.compose(seeks)`` their concatenation, see
:meth:`discopy.closed.TermBase.compose`.
"""

# pylint: disable-next=wrong-import-position  # proofnet imports this module
from discopy.lambek.abstract import proofnet  # noqa: E402


Bubble = Box.from_args
