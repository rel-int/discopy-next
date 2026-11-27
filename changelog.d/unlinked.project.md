`test_Diagram_permutation` and `test_Kauffman` patch class attributes through
`monkeypatch`, so a failure no longer leaves them patched for the tests after
it, and `test/balanced.py` defines `test_functor_twist_dagger` once
