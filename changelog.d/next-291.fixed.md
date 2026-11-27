`optyx.channel.Sum` without terms keeps its types in `get_kraus` and
`double`, and evaluates to the zero tensor, so the gradient by an absent
variable doubles and evaluates ([next#291](https://github.com/rel-int/discopy-next/issues/291)).
