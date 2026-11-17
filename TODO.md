> cubic reviewed the move of optyx (#276) and flagged 60 lines. 59 are optyx's own code, the same on optyx `main` up to the `optyx.` → `discopy.optyx.` rename, and the move changes no behaviour (#225), so they are filed here rather than fixed there. Each was checked against the code: most reproduced on #276's head (Python 3.14, numpy 2.4, perceval 1.3), the README snippets by reading. One box per bug, each linking cubic's thread.
>
> Filed elsewhere: `from_pyzx`'s output wiring, which is in `discopy.quantum.zx` too, as #313, and the six that already had an issue: #234, #254, #292, #296, #297, #298. Not a bug: `compare_arrays_of_different_sizes` ignoring the tail of the longer array, which is what it is for, comparing two truncations.
>
> ## Channels, classical, photonic and qubit boxes
>
> - [ ] `BitFlipError(0)` and `DephasingError(0)` raise `ZeroDivisionError` ([qubits.py:906](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369491))
> - [ ] `classical.And(n)` declares a `bit**2` domain, so `And(3)` fails an assertion; `bit**n` ([classical.py:480](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369549))
> - [ ] `photonic.Select()` stores `()` where `zw.Select()` defaults to `(1,)`: its path matrix has domain 0 and no selection while the channel's domain is `qmode` ([photonic.py:283](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369562))
> - [ ] `box ** 0`, and `**` with any n ≤ 0 on `core.diagram`, `channel.Diagram` and `CQMap`, recurses forever: `zw.W(2) ** 0`, `BS ** 0` ([core/diagram.py:919](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369612), [channel.py:1482](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369844))
> - [ ] `qubits.Z(0, 0)._decomp()` is 0 → 1, its codomain `qubit` ([qubits.py:702](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369658))
> - [ ] `Channel.dagger` fails an assertion with a non-empty environment: `PhotonLoss(.1).dagger()` ([channel.py:1566](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369678))
> - [ ] `ControlledPhaseShift.conjugate` keeps its phases: the truncation of `conjugate()` equals the original's; negate them ([core/control.py:313](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369691))
> - [ ] `ControlledPhaseShift` with two or more control modes raises on reshape; `array[tuple(i)]` ([core/control.py:285](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369722))
> - [ ] `TBS` and `MZI` decompose ignoring `is_conj`, so `TBS(x).conjugate().grad(x) == TBS(x).grad(x)`, which finite differences disagree with ([photonic.py:706](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369703))
> - [ ] `TBS.dagger` and `MZI.dagger` are wrong when conjugated, `g >> g.dagger()` is not the identity: the global phase ORs the flags where it should XOR them ([photonic.py:721](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369861))
> - [ ] `BBS.dagger` drops `is_conj`: for a conjugated `BBS`, `dagger()` equals `g`; `BBS(0.5 - self.bias, self.is_conj)` ([photonic.py:618](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369764))
> - [ ] `PostselectDigit()` and `Digit()` declare zero modes and fail an assertion; `len(x or (1,))` ([classical.py:432](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369817), [classical.py:669](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369827))
> - [ ] `PostselectBit()` with no bits raises `IndexError` ([classical.py:408](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369900))
> - [ ] the scalar `sqrt(n)` of `classical.Xor(n)` is only right for n = 2: the truth table of `Xor(3)` has entries 0.866; `sqrt(2 ** (n - 1))` ([classical.py:463](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369875))
> - [ ] `BitControlledGate` of an empty diagram leaves `layer` unbound, `UnboundLocalError`; read `diag.cod` ([classical.py:233](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369945))
> - [ ] the `classical` docstring names `XorBit` and `AddN`, which do not exist (`Add` does) ([classical.py:91](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369926))
> - [ ] `power_fix` with `n_steps > max_steps` contracts past `max_steps`, warns that it stayed within it, and returns the deeper result ([channel.py:990](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369838))
>
> ## Path matrices, ZW and `utils.misc`
>
> - [ ] `path.Matrix.dagger` leaves `scalar` unconjugated: `Matrix(eye(1), 1, 1, scalar=1j).dagger().scalar == 1j` ([core/path.py:298](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369500))
> - [ ] `path.Matrix.dilate` drops `scalar` and `normalisation`: `dilate().eval(2) != eval(2)` ([core/path.py:350](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369789))
> - [ ] `path.Matrix.eval(0)` with no selections calls `np.stack([])`: `zw.Id(1).to_path().eval()` raises `ValueError` ([core/path.py:376](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369513))
> - [ ] `npperm` of a 0 × 0 matrix is 2, where the permanent is 1 ([core/path.py:167](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369728))
> - [ ] `IndexableAmplitudes.__eq__` compares bytecode only: `ZBox(1, 1, lambda i: .5 ** i) == ZBox(1, 1, lambda i: .9 ** i)`, and `a == a.conjugate()` ([core/zw.py:205](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369667))
> - [ ] `ZBox.__eq__` raises `ValueError` on lists of amplitudes of different lengths ([core/zw.py:388](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369850))
> - [ ] `zw.W` and `ZBox` define `__eq__` without `__hash__`, so `hash(zw.W(2))` raises, and so does a `Functor` with `W` in a dict `ar_map` ([core/zw.py:262](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369896))
> - [ ] `matrix_to_zw` strides columns by the row count: every 2 × 3 and 3 × 2 matrix raises "Expected a permutation"; `U.shape[1] * r + c` ([utils/misc.py:38](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369887))
> - [ ] `matrix_to_zw` on a symbolic matrix with zero entries, `[[x, 0], [0, x]]`, raises `AxiomError` ([utils/misc.py:43](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369801))
> - [ ] the zero-count prefix sum in `matrix_to_zw` is quadratic (performance only) ([utils/misc.py:72](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369935))
> - [ ] `basis_vector_from_kets([-1], [3])` returns -1: negative indices pass the check ([utils/misc.py:148](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369807))
>
> ## Backends and the perceval conversion
>
> - [ ] `PercevalBackend` sums the probabilities of the terms of a sum, so interference is lost: `Create(1) + Create(1) @ Scalar(-1)` gives `{(1,): 1.0}` where quimb finds zero; sum amplitudes, then take the Born rule ([core/backends.py:647](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369406))
> - [ ] `PercevalBackend` ignores `Matrix.scalar` and the normalisation `dilate()` adds: the amplitude of `Create(1) @ Scalar(1j)` is 1, quimb's 1j ([core/backends.py:1037](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369591))
> - [ ] in the feed-forward conversion, each configuration entry applies `default_action` when it does not match, so the default is applied once per non-matching entry: with default `PERM([1, 0])` and two identity configurations, `|1,1,0>` gives `(1,0,1)` in optyx and `|1,1,0>` in perceval ([utils/perceval_conversion.py:79](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369398))
> - [ ] `ff_configurator` zips a set of symbols with dict values, so the pairing depends on `PYTHONHASHSEED`; `{s: symbol_values[str(s)] for s in free_symbols}` ([utils/perceval_conversion.py:164](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369474))
> - [ ] `use_provider_dom` is never read, its branch commented out ([utils/perceval_conversion.py:93](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369915))
>
> ## Compiler
>
> - [ ] `fn_to_semm` skips `photon += 1` for an output node: `FusionNetwork([0, 1, 2, 3], {0, 2}, [Fusion(0, 2)])` raises `KeyError: 3` ([compiler/semm.py:158](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369417))
> - [ ] `fn_to_semm` and its measurement times index by node label arrays sized by path position: labels `{10, 20, 30}` raise `IndexError`, path `[2, 0, 1]` gives a delay of 3 ([compiler/semm.py:227](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369426))
> - [ ] `fn_to_semm` always emits X fusions, a Y fusion included (a known limitation its `NOTE` says) ([compiler/semm.py:154](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369601))
> - [ ] `find_min_path_cover` raises "unreachable!" on one node or an edgeless graph; `range(1, n + 1)` ([compiler/path_cover.py:31](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369527))
> - [ ] the SEMM decompiler misses a double measurement through a delay: `[Next(0), Measure(3, m1), Measure(0, m2)]` decompiles silently and keeps one ([compiler/semm_decompiler.py:87](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369740))
> - [ ] the SEMM decompiler drops a delayed fusion nobody consumes: `[Next(0), Fusion(5), Measure]` decompiles with no fusion and no error ([compiler/semm_decompiler.py:99](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369867))
> - [ ] the SEMM decompiler raises `IndexError` rather than `ValidationError` on an operation before `next_node` ([compiler/semm_decompiler.py:69](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369909))
> - [ ] `Fusion == None` raises `AttributeError` ([compiler/mbqc.py:67](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369952))
> - [ ] the `compiler` autosummary applies `class.rst` to the functions `compile_to_semm`, `decompile_from_semm` and `pattern_satisfies_order`, which the API page of #276 now renders ([compiler/__init__.py:25](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369958))
>
> ## `discopy/optyx/README.md` snippets
>
> - [ ] names used but never imported: `Channel`, `H`, `FusionTypeII`, `qubits.Ket` where only `Ket` is imported; line 174 also imports `AddN`, which does not exist ([discopy/optyx/README.md:28](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369628), [discopy/optyx/README.md:63](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369754), [discopy/optyx/README.md:242](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369639), [discopy/optyx/README.md:325](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369746))
> - [ ] `state_array` is undefined ([discopy/optyx/README.md:384](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369634))
> - [ ] `np.isclose(..., 3)` and `np.allclose(a, b, Id)` pass 3 and the identity as `rtol` ([discopy/optyx/README.md:204](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369569), [discopy/optyx/README.md:93](https://github.com/rel-int/discopy-next/pull/276#discussion_r4353369616))

Prompt: rel-int/discopy-next#314, quoted above. This branch takes three boxes that still reproduce on main 89467e62.

- [x] Channel.dagger with an environment: the adjoint CP map, Kraus map K† with the environment bent back by a cap, so double(c†) = double(c)†
- [WIP] @session_01MEJxy4kqJ6YhajqkcZRUXm-2026-11-17 00:21 power_fix with n_steps > max_steps raises ValueError instead of contracting past max_steps
- [WIP] @session_01MEJxy4kqJ6YhajqkcZRUXm-2026-11-17 00:21 fn_to_semm emits each fusion's own type, so a Y fusion decompiles back as Y
- [ ] CHANGELOG.md bullet
