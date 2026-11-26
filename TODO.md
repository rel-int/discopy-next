> USER answered "yes": shorten the class references too, on #746. This covers `:class:` references to discopy.* in discopy/ and docs/api/, plus `:meth:`, `:func:`, `:obj:`, `:attr:`, `:exc:`, each with the `~` so only the last part shows. Leave references that are already short or relative alone. Extend test/api_names.py so it fails on these roles too, the same way it does for `:mod:`. Check a local build.

- [x] Shorten :class:, :meth:, :func:, :obj:, :attr:, :exc: references to discopy.* with ~, checked by test/api_names.py and a local build
- [ ] Changelog entry
