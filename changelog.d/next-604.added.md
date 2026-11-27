`lambek.categorial.Parser.from_pretrained` loads the model of the Bobcat
reproduction, `rel-int/bobcat-rebank-deberta-v3-large` on Hugging Face, and
a parser called on a sentence gives its categorial diagram, with a label for
every span if it can. `lambek.neural.Encoder` gives each word the state of
its first subword, and `lambek.neural.ChartClassifier` gives a supertag for
each word and a label for each span. `transformers` joins the `lambek` extra
([next#604](https://github.com/rel-int/discopy-next/pull/604)).
