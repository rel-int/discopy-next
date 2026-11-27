`optyx.channel.Diagram.fix` and `power_fix` raise `ValueError` naming
`max_chi >= 2` when compressed contraction is asked for `max_chi=1` on a
network whose bond dimension exceeds one, rather than failing with
`LinAlgError` inside Quimb's stabilized QR, and `test_fix.py` checks
`max_chi=1` against the real `QuimbBackend` ([next#442](https://github.com/rel-int/discopy-next/issues/442)).
