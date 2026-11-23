> Check what input discopy.lambek.categorial.bobcat (our Bobcat reproduction, #603/#604, model rel-int/bobcat-rebank-deberta-v3-large) accepts: raw strings or only pre-tokenised words.
>
> If it only takes tokens, propose a PR adding a thin optional tokeniser helper (e.g. discopy.lambek.tokenise), like lambeq's lambeq.tokeniser.SpacyTokeniser (spaCy's rule-based English tokenizer + sentencizer, no model): spaCy behind an optional extra, imported lazily, with tests that skip without spaCy. Its tokens should match the CCGrebank/PTB conventions the model was trained on (contractions, hyphens, punctuation); check where spaCy's defaults differ and handle or document those differences, and say in the PR which you chose. Keep it small and readable, and state the review cost (lines changing existing code, lines in new files, core modules touched).
>
> If it already accepts raw text, make no PR and just report how it tokenises.

- [ ] discopy.lambek.tokenise: spaCy's rule-based English tokenizer and sentencizer, lazily imported, with the CCGrebank/PTB differences handled
- [ ] spacy in the lambek extra, uv.lock
- [ ] tests that skip without spaCy, docs and changelog
