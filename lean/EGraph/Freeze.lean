import EGraph.Drain

/-!
# Freezing

What a builder with the invariant freezes into: a valid e-graph whose classes
relate the spiders of one root. So `merge` and `rebuild` on the builder are
`merge` and `rebuild` on the e-graph it freezes into.
-/

namespace EGraph

namespace Builder

theorem classes_fold {rank : Nat → Nat} :
    ∀ (spiders : List Nat) (b : Builder) (found : Array Nat), Forest b.label rank →
      (spiders.foldl (fun (state : Builder × Array Nat) spider =>
        ((state.1.find spider).1, state.2.push (state.1.find spider).2)) (b, found)).2.toList
        = found.toList
          ++ spiders.map fun x => b.least.getD (rep b.label x) (rep b.label x) := by
  intro spiders
  induction spiders with
  | nil => intro b found _; simp
  | cons spider spiders ih =>
    intro b found forest
    have ⟨value, label, eq, forest', _, same⟩ := b.root_spec forest spider
    have find_eq : b.find spider = ({ b with label := label },
        b.least.getD (rep b.label spider) (rep b.label spider)) := by
      simp only [find, eq, value]
    simp only [List.foldl_cons, find_eq, List.map_cons]
    rw [ih _ _ forest', funext same]
    simp

theorem classes_eq {b : Builder} {rank : Nat → Nat} (forest : Forest b.label rank) :
    b.classes = (List.range b.label.size).map
      fun x => b.least.getD (rep b.label x) (rep b.label x) := by
  have := classes_fold (List.range b.label.size) b #[] forest
  simpa [classes] using this

namespace Inv

theorem classes_length {b : Builder} (h : Inv b) : b.freeze.classes.length = b.label.size := by
  obtain ⟨rank, forest⟩ := h.forest
  show b.classes.length = _
  rw [classes_eq forest]; simp

theorem find_classes {b : Builder} (h : Inv b) (x : Nat) :
    EGraph.find b.freeze.classes x = if x < b.label.size
      then b.least.getD (rep b.label x) (rep b.label x) else x := by
  obtain ⟨rank, forest⟩ := h.forest
  show EGraph.find b.classes x = _
  split
  · rename_i lt
    apply find_of_some
    rw [classes_eq forest]
    simp [lt]
  · rename_i ge
    exact find_of_ge (by
      have : b.classes.length = b.label.size := h.classes_length
      omega)

/-- Two spiders are in the same class of the e-graph when they have one root. -/
theorem same_iff {b : Builder} (h : Inv b) (x y : Nat) :
    Same b.freeze.classes x y ↔ rep b.label x = rep b.label y := by
  obtain ⟨rank, forest⟩ := h.forest
  have t := h.tables
  unfold Same
  rw [h.find_classes, h.find_classes]
  have least_lt : ∀ z, z < b.label.size →
      b.least.getD (rep b.label z) (rep b.label z) < b.label.size :=
    fun z lt => Nat.lt_of_le_of_lt (t.least_le z lt) lt
  have back : ∀ z, z < b.label.size →
      rep b.label (b.least.getD (rep b.label z) (rep b.label z)) = rep b.label z :=
    fun z lt => t.least_class _ (forest.rep_lt lt) (forest.rep_rep z)
  by_cases lx : x < b.label.size <;> by_cases ly : y < b.label.size
  · simp only [lx, ly, ite_true]
    exact ⟨fun eq => by rw [← back x lx, eq, back y ly], fun eq => by rw [eq]⟩
  · simp only [lx, ly, ite_true, ite_false]
    have := least_lt x lx
    have := forest.rep_lt lx
    rw [forest.rep_of_ge (Nat.le_of_not_lt ly)]
    constructor <;> intro eq <;> omega
  · simp only [lx, ly, ite_true, ite_false]
    have := least_lt y ly
    have := forest.rep_lt ly
    rw [forest.rep_of_ge (Nat.le_of_not_lt lx)]
    constructor <;> intro eq <;> omega
  · simp only [lx, ly, ite_false]
    rw [forest.rep_of_ge (Nat.le_of_not_lt lx), forest.rep_of_ge (Nat.le_of_not_lt ly)]

theorem normal {b : Builder} (h : Inv b) : Normal b.freeze.classes := by
  obtain ⟨rank, forest⟩ := h.forest
  have t := h.tables
  have value : ∀ i c, b.freeze.classes[i]? = some c →
      i < b.label.size ∧ c = b.least.getD (rep b.label i) (rep b.label i) := by
    intro i c at_i
    have lt : i < b.label.size := by
      have := (List.getElem?_eq_some_iff.1 at_i).1
      rwa [h.classes_length] at this
    have := find_of_some at_i
    rw [h.find_classes, if_pos lt] at this
    exact ⟨lt, this.symm⟩
  refine ⟨fun i c at_i => ?_, fun i c at_i => ?_⟩
  · have ⟨lt, eq⟩ := value i c at_i
    rw [eq]; exact t.least_le i lt
  · have ⟨lt, eq⟩ := value i c at_i
    have lt' : c < b.label.size := by
      rw [eq]; exact Nat.lt_of_le_of_lt (t.least_le i lt) lt
    have : EGraph.find b.freeze.classes c = c := by
      rw [h.find_classes, if_pos lt']
      have root : rep b.label c = rep b.label i := by
        rw [eq]; exact t.least_class _ (forest.rep_lt lt) (forest.rep_rep i)
      rw [root, ← eq]
    have at_c := some_find (classes := b.freeze.classes) (i := c)
      (by rw [h.classes_length]; exact lt')
    rw [this] at at_c
    exact at_c

theorem valid {b : Builder} (h : Inv b) : b.freeze.Valid :=
  ⟨h.normal, fun node mem s out => by
    rw [h.classes_length]; exact (h.tables.wires node mem).2 s out, h.tables.hashed.arity⟩

/-- With nothing on the worklist, the classes are congruent. -/
theorem congruent {b : Builder} (h : Inv b) (empty : b.worklist = []) :
    Congruent b.nodes.toList fun x y => rep b.label x = rep b.label y := by
  intro a a_mem c c_mem da dc head inputs
  have hashed := h.tables.hashed
  have maps : a.inputs.map (rep b.label) = c.inputs.map (rep b.label) := by
    have ⟨length, at_i⟩ := pointwise_iff.1 inputs
    apply List.ext_getElem?
    intro i
    simp only [List.getElem?_map]
    cases ha : a.inputs[i]? with
    | none =>
      have : c.inputs[i]? = none := by
        rw [List.getElem?_eq_none_iff] at ha ⊢; omega
      rw [this]
    | some x =>
      have lt := (List.getElem?_eq_some_iff.1 ha).1
      rw [List.getElem?_eq_getElem (length ▸ lt)]
      simp only [Option.map_some]
      exact congrArg some (at_i i x _ ha (List.getElem?_eq_getElem (length ▸ lt)))
  have same_key : key (rep b.label) a = key (rep b.label) c := by
    simp [key, da, dc, head, maps]
  have ne : key (rep b.label) a ≠ [] := by simp [key, da]
  obtain ⟨i, at_i⟩ := List.mem_iff_getElem?.1 a_mem
  obtain ⟨j, at_j⟩ := List.mem_iff_getElem?.1 c_mem
  rw [Array.getElem?_toList] at at_i at_j
  have key_i := hashed.keys i a (fun f => f) at_i
  have key_j := hashed.keys j c (fun f => f) at_j
  rw [← same_key] at key_j
  cases entry : b.hashcons[key (rep b.label) a]? with
  | none => exact (hashed.hash_none i _ (fun f => f) key_i ne entry).elim
  | some first =>
    have eqv : Equivalence fun x y => rep b.label x = rep b.label y :=
      ⟨fun _ => rfl, Eq.symm, Eq.trans⟩
    have to (x : Nat) (node : Node) (at_x : b.nodes[x]? = some node)
        (key_x : b.keys[x]? = some (key (rep b.label) a)) :
        Pointwise (fun x y => rep b.label x = rep b.label y) (b.outputs first) node.outputs := by
      rw [← outputs_of at_x]
      exact (hashed.linked x _ first (fun f => f) key_x entry).mono
        fun _ _ pending => pending.of_empty empty
    exact Pointwise.trans eqv (Pointwise.symm eqv (to i a at_i key_i)) (to j c at_j key_j)

end Inv

/-- Merging on the builder is merging on the e-graph. -/
theorem merge_spec {b : Builder} (h : Inv b) (spiders : List Nat)
    (lt : ∀ s ∈ spiders, s < b.label.size) :
    Inv (b.merge spiders) ∧ (b.merge spiders).freeze = b.freeze.merge spiders := by
  have ⟨inv, nodes, size, _, _, extensive, related, least⟩ :=
    unions_spec (Graph.chain spiders) b h (Graph.chain_lt lt)
  have inv : Inv (b.merge spiders) := inv
  have size : (b.merge spiders).label.size = b.label.size := size
  have lt' : ∀ s ∈ spiders, s < b.freeze.classes.length := by
    rw [h.classes_length]; exact lt
  have ⟨length, normal, _⟩ := Graph.merge_spec b.freeze h.normal spiders lt'
  refine ⟨inv, Graph.ext (congrArg Array.toList nodes) ?_⟩
  refine Normal.ext inv.normal normal ?_ fun x y => ?_
  · rw [inv.classes_length, length, h.classes_length]; exact size
  · rw [inv.same_iff]
    constructor
    · exact least _ (Same.equivalence _)
        (fun x y eq => Graph.merge_extensive _ h.normal _ lt' ((h.same_iff x y).2 eq))
        (Graph.merge_chain _ h.normal _ lt') x y
    · exact Graph.merge_least _ h.normal _ lt'
        (r := fun x y => rep (b.merge spiders).label x = rep (b.merge spiders).label y)
        ⟨fun _ => rfl, Eq.symm, Eq.trans⟩
        (fun x y same => extensive x y ((h.same_iff x y).1 same)) related

/-- Rebuilding on the builder is rebuilding on the e-graph. -/
theorem rebuild_spec {b : Builder} (h : Inv b) :
    Inv b.rebuild ∧ b.rebuild.freeze = b.freeze.rebuild := by
  have enough : b.todo b.total ≤ b.fuel := by
    unfold todo fuel count
    have := List.countP_le_length (p := fun x => rep b.label x == x)
      (l := List.range b.label.size)
    rw [List.length_range] at this
    exact Nat.add_le_add_left (Nat.mul_le_mul_right _ this) _
  have ⟨inv, empty, nodes, size, extensive, least⟩ := drain_spec b.fuel b h enough
  have inv : Inv b.rebuild := inv
  have size : b.rebuild.label.size = b.label.size := size
  have valid := h.valid
  refine ⟨inv, Graph.ext (congrArg Array.toList nodes) ?_⟩
  refine Normal.ext inv.normal (Graph.rebuild_valid valid).normal ?_ fun x y => ?_
  · rw [inv.classes_length, (Graph.rebuild_round valid).length, h.classes_length]; exact size
  · rw [inv.same_iff]
    constructor
    · exact least _ (Same.equivalence _) (Graph.rebuild_congruent valid)
        (fun x y eq => Graph.rebuild_extensive valid ((h.same_iff x y).2 eq)) x y
    · have congruent := inv.congruent empty
      rw [show (Builder.rebuild b).nodes = b.nodes from nodes] at congruent
      exact Graph.rebuild_least valid
        (r := fun x y => rep b.rebuild.label x = rep b.rebuild.label y)
        ⟨fun _ => rfl, Eq.symm, Eq.trans⟩ congruent
        (fun x y same => extensive x y ((h.same_iff x y).1 same))

end Builder

end EGraph
