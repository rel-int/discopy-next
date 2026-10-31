# TODO

🌙 Evening's prompt for the night of 2026-10-31, verbatim:

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

Two boxes of #314 in `discopy/optyx/utils/perceval_conversion.py`, both reproduced on `main`
`adcd06b6`: a feed-forward configurator paired its symbols with its values through a set, so the
circuit it converted to depended on `PYTHONHASHSEED`, and `_feedforward_common` carried a flag
nothing read whose branch was commented out.

- [x] `ff_configurator` zipped `U.free_symbols`, a set, with the values of the configuration dict,
      so a symbol silently got another symbol's value under half the hash seeds; it now substitutes
      by name, `{s: symbol_values[str(s)] for s in free_symbols}`, and raises `ValueError` when a
      symbol has no value rather than pairing it with whatever came next
- [x] `use_provider_dom` was declared on `_feedforward_common`, passed `True` by
      `ff_circuit_provider` and `False` by `ff_configurator`, and read nowhere, its branch
      commented out; the parameter, the branch and both arguments are gone, since the signature
      promised a provider domain the code never computed and which way that domain should be read
      is a design decision, not a repair
