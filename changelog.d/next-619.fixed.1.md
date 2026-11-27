`optyx.channel.Channel.lambdify` and `subs` substitute in its Kraus map,
so `subs` on e.g. `Phase(x).inflate(2)` no longer keeps `x`; the
channel keeps its name ([next#619](https://github.com/rel-int/discopy-next/pull/619)).
