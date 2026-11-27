The cup of an optyx type is its own spider with two legs in, and the cap
its dagger: `optyx.core.diagram.Diagram.cup_factory` and
`optyx.channel.Diagram.cup_factory` build them where both inherited
`frobenius.Cup`, which is not an optyx box, so `trace()` and every other
way of bending a wire raised `TypeError` ([next#681](https://github.com/rel-int/discopy-next/pull/681)).
