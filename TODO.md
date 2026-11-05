> I got some spare tokens to spend: let’s use them to polish all things discopy-next. Make a deep review of the code and the infra around it, summarising your findings and suggestions in an issue, fixing any bugs that you can fix without requiring a design decisions from my side.
>
> go ahead with the fixes once workflow A lands

The fixes of rel-int/discopy-next#445 in `optyx`: bugs the review confirmed that need no design decision, one commit and one regression test each.

- [x] **S13-1** PercevalBackend gives the wrong outcome labels for any diagram with a Select or a dilation once more than one outcome is non-zero:
- [x] **S13-2** QuimbBackend with a hyperoptimiser sorts output index names as strings (out10 before out2), so a result with 11 or more input or output wires has its …
- [x] **S13-3** classical.BitControlledPhaseShift declares its classical control wires last, but its Kraus box reads the first wires as control:
- [x] **S13-4** PermanentBackend sums the arrays of terms with different photon totals by numpy broadcasting:
- [x] **S13-5** ZXDiagram.to_pyzx exports zx.ZBox(n, m, w) as an X spider of phase 2w:
- [x] **S13-6** path.Matrix inherits matrix.Matrix.__eq__, which ignores creations, selections, normalisation and scalar:
- [x] **S13-8** A zw.ZBox with no inputs truncates a finite amplitude list at dimension 2.
- [x] **S13-9** Since cfc59baa (#87), Matrix.permutation and Matrix.copy fill a Matrix[Expr] with Python bools, so a symbolic path matrix crashes whenever …
- [x] **S13-11** deferred to #445: needs a repr on 14 classes in three files, past 40 lines
- [x] **S13-12** diagram.Scalar and diagram.Sum fall through to Box.inflate's d-fold tensor power:
- [x] **S13-14** zx.ZBox(0, 0, w) evaluates to 1 instead of 1 + w (its own P_0 + w P_1, pyzx's legless Z_BOX and spider fusion all give 1 + w):
- [x] **S13-15** BitControlledBox.photon_number_transform sets self.photon_preservation_behaviour = NON_LO during to_tensor, so evaluating a diagram permanently …
- [x] **S13-16** zx's autosummary lists zx_to_path, decomp and zx2path, all deleted in optyx#137, so Sphinx warns that it failed to import each one;
- [x] **S14-1** qubits.QubitChannel.from_discopy, a stale copy of quantum.zx.gate2zx from before 53d949fc, converts gates wrongly:
- [x] **S14-2** CQMap.inflate inflates the doubled density matrix in place, so a qmode's wires come out as (m^1..m^d, m̄^1..m̄^d) where dom.inflate(d).double() is …
- [x] **S14-4** classical.Sub computes first − second, against its docstring and test_subn (second − first);
- [x] **S14-6** channel.Diagram.grad (tensor.Diagram.grad) recurses forever on any channel box with free symbols and no grad override (qubits.Z/X/Scalar, …
- [x] **S14-7** Channel.lambdify returns the box itself, so subs and lambdify are silent no-ops on photonic.Gate and qubits.Scalar, whose free symbols survive …
- [x] **S14-8** Diagram.fix warns that the result is an approximation truncated to max_chi even when the backend never compresses (DiscopyBackend, or QuimbBackend …
- [x] **S14-10** The qubits module's teleportation doctest is skipped as if the library were at fault, but its circuit is wrong:
- [x] **S14-11** ZMeasurementDR's docstring says 'Z basis after a phase shift' but leaves out the HadamardBS:
- [x] **S14-12** BBS's docstring matrix is the array with its columns swapped, left stale when optyx redefined BS (8ee5a9ac).
- [x] **S14-15** qubits.X.dual_rail maps X(1,1,.25) to BS.dagger(), which is not even proportional to it.
- [x] **S14-16** classical's autosummary lists `Select`, a name the module does not define, so Sphinx cannot import it;
- [x] **S14-17** Diagram.is_pure's Encode branch tests the codomain for classical types, which an Encode never has, so it never fires.
- [x] **S15-2** Diagram.from_perceval applies a PostSelect condition to the wrong modes when a herald precedes them:
- [x] **S15-3** compile_postselect cannot tokenise '!=', so Diagram.from_perceval raises on a processor whose PostSelect uses '!=':
- [x] **S15-4** Diagram.from_perceval creates the input side from p.heralds (perceval's output heralds) instead of p.in_heralds, so an OUTPUT-only or INPUT-only …
- [x] **S15-5** Optyx reads every diagram as one box per layer (`(box, offset), = layer.inside[0].boxes_and_offsets`), so a foliated feedback state, Kraus map or …
- [x] **S15-7** SemmDecompiler.delay_then_fuse's docstring says it applies a unitary and measures, but it only puts the current node's photon in the delay loop, to …
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 full suite and lint gates on the branch
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 one `CHANGELOG.md` line per user-facing fix
