`optyx.core.diagram.Feedback` no longer says its delay endofunctor is the
identity: it is a delay line, so feedback is not a trace and yanking fails
— the delay is real, it is only left unwritten on the memory type
([next#681](https://github.com/rel-int/discopy-next/pull/681)).
