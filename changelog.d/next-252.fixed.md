`optyx.channel.Sum.double` doubles a sum coherently, cross terms included,
as the doubling of the channel whose Kraus map is `get_kraus()`: the doubled
Bell sum is the Bell state and no longer the mixture, `(BS + BS).double()` is
four times `BS.double()`, and a sum with an impure term refuses; the
conjugate of an empty `optyx.core.diagram.Sum` keeps its types ([next#252](https://github.com/rel-int/discopy-next/issues/252)).
