`optyx.core.zx.Spider.grad` is `1j * pi * phase'` times the difference
between the spider and the spider shifted by half a turn, which is the
formula `quantum.zx.Spider.grad` next door already had. The optyx copy
returned the shifted spider alone, so a spider with legs had a spurious
value on the all-zero entry of its gradient, where the derivative has none,
and a legless one had the wrong value outright. `optyx.core.zx.ZBox`
overrides it: its parameter is an amplitude rather than an angle, its array
is linear in it, and its gradient is the difference between the box at one
and the box at zero ([next#426](https://github.com/rel-int/discopy-next/issues/426)).
