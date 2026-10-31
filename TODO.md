# TODO

🌙 Evening's prompt for the night of 2026-10-31, verbatim:

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

This head takes the four documentation boxes of
[#314](https://github.com/rel-int/discopy-next/issues/314) — the three functions the `compiler`
autosummary renders as classes, and the snippets of `discopy/optyx/README.md`, of which 14 of the
19 code blocks did not run against `main` `adcd06b6`. Each box's oracle is the code its snippet
claims to demonstrate; `test/optyx/readme.py` now executes every block, the way `test/readme.py`
already executes the root README's, so a snippet that stops running fails a check rather than
being read.

- [x] The `compiler` autosummary applied `:template: class.rst` to `compile_to_semm`,
      `decompile_from_semm` and `pattern_satisfies_order`, so the API page documented three
      functions as classes. They now have their own `:template: function.rst` block under an
      `.. admonition:: Functions`, which is how `quantum/channel.py` and `quantum/gates.py` do it.
- [x] The snippets used four names they never imported and imported one that does not exist:
      `Channel` (now `from discopy.optyx import Channel, qubit`), `H` (now
      `from discopy.optyx.qubits import H, Measure`), `FusionTypeII` (now
      `from discopy.optyx.photonic import DualRail, FusionTypeII`), `qubits.Ket` where only `Ket`
      was imported (now called `Ket`, bare, like every other generator in this README), and
      `AddN`, which `classical` has never had: the sum of two measured photon numbers is
      `Add(2)`.
- [x] `state_array`, the amplitudes sent through the perceval processor, was used twice and
      defined nowhere. It is the state the next snippet transmits, so it is now built and
      normalised where it is first used: `(Ket("+") >> Z(1, 1, 0.3)).eval().tensor.array`.
- [x] `np.isclose(result[(1, 1)], theoretical_result, 3)` passed 3 as `rtol`, a tolerance of
      300%, which made the distinguishability assertion hold whatever the simulation said; it now
      passes at numpy's default, the two agreeing to 4e-17. `np.allclose(a, b, Id(1).double()
      .to_tensor().eval().array)` passed the identity channel as `rtol`: the third array is what
      the snippet means to compare against, both syntaxes being the identity CPTP map, so it is
      now a second `assert np.allclose` of its own.
- [x] Found on the way and fixed, as `AGENTS.md` asks: the graphix snippet called
      `simulator.run().psi`, where graphix 0.4's `run()` returns `None` and `psi` is flat, and
      then compared amplitudes directly, which is a coin flip — the pattern's random outcomes put
      a phase of ±1 on the state, measured −1 on 4 of 12 runs. It now reads the statevector off
      `simulator.backend.state` and compares up to a phase, as `test/optyx/test_qubit.py` and
      `docs/notebooks/optyx/readme_example.md` already do.
- [x] `uv run` `pflake8 discopy`, `pylint discopy`, `ruff check discopy`,
      `ty check --exit-zero-on-warning`, `pytest test/optyx -q`, and all 19 README blocks.
