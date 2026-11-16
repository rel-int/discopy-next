> Review round on #569 (cubic-dev-ai[bot]), head d73cd6e2:
>
> 1. discopy/__init__.py:43 (thread 4592024271)
> [P2] Dropping `grammar` from this import list means `import discopy` no longer exposes `discopy.grammar`, so existing code that does `import discopy` and then uses `discopy.grammar.x` now fails with AttributeError. Before this PR that access worked. Consider exposing the deprecated alias lazily (for example a module-level `__getattr__` that imports `discopy.grammar`) so the warning fires only on use.
>
> 2. discopy/grammar.py:30 (thread 4592024288)
> [P2] The shim only aliases submodules, so top-level names the old `discopy.grammar` package exported are lost. `from discopy.grammar import Word` (and `Rule`) now raises ImportError. Re-export them from `discopy.lambek.thue` in this module.
>
> 3. discopy/grammar.py:36 (thread 4592024297)
> [P2] The neural alias only works if `discopy.lambek.neural` was imported earlier. Otherwise `import discopy.grammar.neural` and unpickling old neural objects raise ModuleNotFoundError, since `discopy.grammar` is not a package. This breaks the stated goal that legacy pickles keep loading. Either resolve `discopy.grammar.neural` lazily (for example through a `sys.meta_path` finder that imports `discopy.lambek.neural` on demand), or state the limitation in the docstring and tests.
>
> 4. pyproject.toml:65 (thread 4592024305)
> [P2] Removing the `grammar` extra breaks existing `pip install discopy[grammar]` commands. pip only warns about the unknown extra and silently skips nltk and torch. Consider keeping `grammar = ["discopy[lambek]"]` under `[project.optional-dependencies]` as a deprecated alias, and noting it in the CHANGELOG.
>
> 5. test/grammar.py:25 (thread 4592024309)
> [P2] This test imports `discopy.grammar` itself before calling `loads`, which hides a load failure. `discopy/__init__.py` does not import `grammar`, and `resolve_factory` only checks `sys.modules`, so a fresh `import discopy; loads(old)` of an old-name tree fails. Import the alias in a subprocess or otherwise run the load without the pre-import, so the test covers a fresh process.
>
> 6. test/biclosed.py:385 (thread 4592024315)
> [P2] This assertion was the only check that legacy `grammar.categorial.Ty` JSON still loads, and the rename changed it to `lambek.categorial.Ty`. Restore the old factory name here so the compatibility path stays covered.
>
> ```suggestion
>     assert loads('{"factory": "grammar.categorial.Ty", "inside": ['
> ```

- [x] discopy/__init__.py:43 Dropping `grammar` from this import list means `import discopy` no longer exposes `discopy.grammar`, so exist…
- [x] discopy/grammar.py:30 The shim only aliases submodules, so top-level names the old `discopy.grammar` package exported are lost.
- [x] discopy/grammar.py:36 The neural alias only works if `discopy.lambek.neural` was imported earlier.
- [x] pyproject.toml:65 Removing the `grammar` extra breaks existing `pip install discopy[grammar]` commands.
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:46 test/grammar.py:25 This test imports `discopy.grammar` itself before calling `loads`, which hides a load failure.
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:47 test/biclosed.py:385 This assertion was the only check that legacy `grammar.categorial.Ty` JSON still loads, and the rename change…
