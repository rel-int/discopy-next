`lambek.tokenise.tokenise` splits raw English text into sentences of words
as the Penn Treebank and CCGrebank write them, for the Bobcat parser: the
rule-based tokenizer and sentencizer of spaCy, with no model, keeping
hyphenated words whole and writing quotes and brackets as the treebank does.
`spacy` joins the `lambek` extra ([next#699](https://github.com/rel-int/discopy-next/pull/699)).
