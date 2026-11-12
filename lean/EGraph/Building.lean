import EGraph.Add

/-!
# A builder freezes into what its operations give in turn

The law `EGraph.building` of the Python: for every history that the Python
accepts, `History.build` is `History.fold`.
-/

namespace EGraph

namespace Builder

theorem map_find_iff {b : Builder} (h : Inv b) (xs ys : List Nat) :
    xs.map (EGraph.find b.freeze.classes) = ys.map (EGraph.find b.freeze.classes)
      ↔ xs.map (rep b.label) = ys.map (rep b.label) := by
  constructor
  · intro eq
    exact map_of_pointwise (Pointwise.mono (fun x y => (h.same_iff x y).1)
      (Graph.pointwise_same.2 eq))
  · intro eq
    exact Graph.pointwise_same.1 (Pointwise.mono (fun x y => (h.same_iff x y).2)
      (pointwise_map eq))

/-- A node is one the e-graph reuses when it has the key. -/
theorem reads_iff {b : Builder} (h : Inv b) (head : Nat) (inputs : List Nat) (node : Node) :
    Graph.reads b.freeze.classes head
        (inputs.map (EGraph.find b.freeze.classes)) node = true
      ↔ key (rep b.label) node = head :: inputs.map (rep b.label) := by
  unfold Graph.reads key
  cases node.det
  · simp
  · simp only [Bool.true_and, Bool.and_eq_true, beq_iff_eq, ite_true, List.cons.injEq]
    rw [map_find_iff h]

/-- Adding on the builder is adding on the e-graph. -/
theorem add_spec {b : Builder} (h : Inv b) (head : Nat) (det : Bool) (arity : Nat)
    (inputs : List Nat) (lt : ∀ s ∈ inputs, s < b.label.size)
    (ar : ∀ node ∈ b.nodes.toList, node.head = head → node.outputs.length = arity) :
    Inv (b.add head det arity inputs).1
      ∧ ((b.add head det arity inputs).1.freeze, (b.add head det arity inputs).2)
        = b.freeze.add head det arity inputs := by
  obtain ⟨rank, forest⟩ := h.forest
  have length := h.classes_length
  cases det with
  | false =>
    have ⟨inv, frozen⟩ := append_spec h head false arity inputs lt ar
    have empty : key (rep b.label) ⟨head, false, inputs, List.range' b.label.size arity⟩ = [] :=
      rfl
    rw [empty] at inv frozen
    have unfolded : b.add head false arity inputs = (Builder.place { b with
        label := b.label ++ (List.range' b.label.size arity).toArray
        least := b.least ++ (List.range' b.label.size arity).toArray
        parents := b.parents ++ Array.replicate arity [] }
        ⟨head, false, inputs, List.range' b.label.size arity⟩ [],
        List.range' b.label.size arity) := by
      simp [add]
    rw [unfolded, Graph.add_none (by simp), length, frozen]
    exact ⟨inv, rfl⟩
  | true =>
    have ⟨values, label, eq, forest', size, same⟩ := roots_spec (rank := rank) inputs b forest
    have inv : Inv { b with label := label } := h.relabel forest' size same
    have frozen := freeze_relabel forest forest' size same
    have hashed := h.tables.hashed
    cases entry : b.hashcons[head :: inputs.map (rep b.label)]? with
    | some first =>
      have unfolded : b.add head true arity inputs
          = ({ b with label := label }, ((b.nodes[first]?).map (·.outputs)).getD []) := by
        simp only [add, ite_true, values, eq, entry]
      have ⟨_, _, keyed, least⟩ := hashed.hash_some _ first entry
      obtain ⟨node, at_first, key_eq⟩ := hashed.node_of_key (fun f => f) keyed
      have found : b.freeze.nodes.find? (fun node => true && Graph.reads b.freeze.classes head
          (inputs.map (EGraph.find b.freeze.classes)) node) = some node := by
        refine find?_least _ b.nodes.toList first node (by rw [Array.getElem?_toList]; exact at_first)
          (by simpa using (reads_iff h head inputs node).2 key_eq.symm) fun i y at_i holds => ?_
        rw [Array.getElem?_toList] at at_i
        have := (reads_iff h head inputs y).1 (by simpa using holds)
        exact least i (fun f => f) (by rw [← this]; exact hashed.keys i y (fun f => f) at_i)
      rw [unfolded, Graph.add_some found, frozen]
      simp [at_first]
      exact inv
    | none =>
      have ⟨inv', frozen'⟩ := append_spec inv head true arity inputs
        (fun s mem => by show s < label.size; rw [size]; exact lt s mem) ar
      have key_eq : key (rep label) ⟨head, true, inputs, List.range' label.size arity⟩
          = head :: inputs.map (rep b.label) := by
        simp only [key, ite_true, List.cons.injEq, true_and]
        exact List.map_congr_left fun x _ => same x
      have unfolded : b.add head true arity inputs = (Builder.place { b with
          label := label ++ (List.range' label.size arity).toArray
          least := b.least ++ (List.range' label.size arity).toArray
          parents := b.parents ++ Array.replicate arity [] }
          ⟨head, true, inputs, List.range' label.size arity⟩
          (head :: inputs.map (rep b.label)), List.range' label.size arity) := by
        simp only [add, ite_true, values, eq, entry]
      have missing : b.freeze.nodes.find? (fun node => true && Graph.reads b.freeze.classes head
          (inputs.map (EGraph.find b.freeze.classes)) node) = none := by
        refine List.find?_eq_none.2 fun y mem holds => ?_
        obtain ⟨i, at_i⟩ := List.mem_iff_getElem?.1 mem
        have at_i : b.nodes[i]? = some y := by rw [← Array.getElem?_toList]; exact at_i
        have := (reads_iff h head inputs y).1 (by simpa using holds)
        exact hashed.hash_none i _ (fun f => f)
          (by rw [← this]; exact hashed.keys i y (fun f => f) at_i) (by simp) entry
      rw [unfolded, Graph.add_none missing, length]
      show Inv _ ∧ _
      rw [← key_eq]
      rw [frozen, size] at frozen'
      rw [size]
      rw [size] at inv'
      exact ⟨inv', by rw [frozen']⟩

/-! ## Starting from an e-graph -/

/-- The builder of an e-graph has the invariant and freezes into it. -/
theorem start_spec (g : Graph) (valid : g.Valid)
    (inputs : ∀ node ∈ g.nodes, ∀ s ∈ node.inputs, s < g.classes.length) :
    Inv (start g) ∧ (start g).freeze = g := by
  let n := g.classes.length
  -- the first builder, with no node
  have parent_eq : ∀ i, parent g.classes.toArray i = EGraph.find g.classes i := by
    intro i; simp [parent, EGraph.find, Array.getD_eq_getD_getElem?, List.getD_eq_getElem?_getD]
  have forest : Forest g.classes.toArray (fun i => n - i) := by
    refine ⟨fun i lt => ?_, fun i up => ?_⟩
    · rw [parent_eq]
      have := valid.normal.find_le i
      simp at lt ⊢; omega
    · rw [parent_eq] at up ⊢
      have := valid.normal.find_le i
      have lt : i < n := by
        rcases Nat.lt_or_ge i n with lt | ge
        · exact lt
        · exact absurd (find_of_ge ge) up
      show n - i < n - EGraph.find g.classes i
      omega
  have roots : ∀ x, rep g.classes.toArray x = EGraph.find g.classes x := by
    intro x
    refine forest.rep_eq ?_
    by_cases is_root : parent g.classes.toArray x = x
    · rw [parent_eq] at is_root; rw [is_root]; exact .root (by rw [parent_eq]; exact is_root)
    · have root : parent g.classes.toArray (EGraph.find g.classes x)
          = EGraph.find g.classes x := by rw [parent_eq]; exact valid.normal.find_find x
      exact .step is_root (by rw [parent_eq]; exact .root root)
  have range_get : ∀ r, (List.range n).toArray.getD r r = r := by
    intro r
    simp only [Array.getD_eq_getD_getElem?, List.getElem?_toArray]
    rcases Nat.lt_or_ge r n with lt | ge
    · rw [List.getElem?_range lt]; rfl
    · rw [List.getElem?_eq_none (by simpa using ge)]; rfl
  -- placing the nodes of a list in turn
  have step : ∀ (todo : List Node) (b : Builder), Inv b → b.label.size = n →
      (∀ x, rep b.label x = EGraph.find g.classes x) → b.least = (List.range n).toArray →
      (∀ node ∈ todo, (∀ s ∈ node.inputs, s < n) ∧ (∀ s ∈ node.outputs, s < n)
        ∧ ∀ a ∈ b.nodes.toList ++ todo, a.head = node.head →
          a.outputs.length = node.outputs.length) →
      let result := todo.foldl (fun b node =>
        (if node.det then b.roots node.inputs else (b, [])).1.place node
          (if node.det then node.head :: (if node.det then b.roots node.inputs else (b, [])).2
            else [])) b
      Inv result ∧ result.label.size = n
        ∧ (∀ x, rep result.label x = EGraph.find g.classes x)
        ∧ result.least = (List.range n).toArray
        ∧ result.nodes.toList = b.nodes.toList ++ todo := by
    intro todo
    induction todo with
    | nil => intro b h size roots least _; exact ⟨h, size, roots, least, by simp⟩
    | cons node todo ih =>
      intro b h size roots least wires
      obtain ⟨rank, forest⟩ := h.forest
      have ⟨inputs, outputs, arity⟩ := wires node (List.mem_cons_self ..)
      have ⟨values, label, eq, forest', size', same⟩ :=
        roots_spec (rank := rank) node.inputs b forest
      have inv : Inv { b with label := label } := h.relabel forest' size' same
      have key_eq : (if node.det then node.head
            :: (if node.det then b.roots node.inputs else (b, [])).2 else [])
          = key (rep label) node := by
        unfold key
        cases node.det
        · rfl
        · simp only [ite_true, values, List.cons.injEq, true_and]
          exact List.map_congr_left fun x _ => (same x).symm
      have placed_eq : (if node.det then b.roots node.inputs else (b, [])).1.place node
            (if node.det then node.head
              :: (if node.det then b.roots node.inputs else (b, [])).2 else [])
          = Builder.place (if node.det then { b with label := label } else b) node
            (key (rep label) node) := by
        rw [key_eq]
        cases node.det
        · rfl
        · simp only [ite_true, eq]
      -- a sample is placed on the labels as they are, which have the same roots
      have result : ∀ b' : Builder, b' = (if node.det then { b with label := label } else b) →
          ∃ label', b' = { b with label := label' } ∧ Forest label' rank
            ∧ label'.size = b.label.size ∧ ∀ x, rep label' x = rep b.label x := by
        intro b' eq
        cases det : node.det
        · rw [det] at eq
          exact ⟨b.label, eq, forest, rfl, fun _ => rfl⟩
        · rw [det] at eq
          exact ⟨label, eq, forest', size', same⟩
      obtain ⟨label', from_eq, forest'', size'', same''⟩ := result _ rfl
      have key_eq' : key (rep label) node = key (rep label') node := by
        unfold key; split
        · simp only [List.cons.injEq, true_and]
          exact List.map_congr_left fun x _ => (same x).trans (same'' x).symm
        · rfl
      have inv'' : Inv (Builder.place { b with label := label' } node (key (rep label') node)) := by
        have relabeled : Inv { b with label := label' } := h.relabel forest'' size'' same''
        have := relabeled.tables.place node (by simpa [size'', size] using inputs)
          (by simpa [size'', size] using outputs)
          (fun s mem => by
            show rep label' s < label'.size
            exact forest''.rep_lt (by rw [size'', size]; exact inputs s mem))
          (fun a mem eq => arity a (List.mem_append_left _ mem) eq)
        exact ⟨⟨rank, by rw [place_label]; exact forest''⟩, by rw [place_label]; exact this⟩
      simp only [List.foldl_cons]
      rw [placed_eq, from_eq, key_eq']
      have next := ih _ inv'' (by rw [place_label]; exact size''.trans size)
        (fun x => by rw [place_label]; exact (same'' x).trans (roots x))
        (by rw [place_least]; exact least) fun node' mem => by
          have ⟨inputs', outputs', arity'⟩ := wires node' (List.mem_cons_of_mem _ mem)
          refine ⟨inputs', outputs', fun a a_mem => arity' a ?_⟩
          rcases List.mem_append.1 a_mem with left | rest
          · rw [place_nodes, Array.toList_push] at left
            rcases List.mem_append.1 left with old | new
            · exact List.mem_append_left _ old
            · rw [List.mem_singleton.1 new]
              exact List.mem_append_right _ (List.mem_cons_self ..)
          · exact List.mem_append_right _ (List.mem_cons_of_mem _ rest)
      obtain ⟨h', size_r, roots_r, least_r, nodes_r⟩ := next
      refine ⟨h', size_r, roots_r, least_r, ?_⟩
      rw [nodes_r, place_nodes]
      simp
  have first : Inv {
      label := g.classes.toArray
      least := (List.range n).toArray
      nodes := #[]
      keys := #[]
      hashcons := {}
      parents := Array.replicate n []
      worklist := [] } := by
    have no_node : ∀ (box : Nat) (node : Node), (#[] : Array Node)[box]? = some node → False :=
      fun box node at_box => by simp at at_box
    have no_key : ∀ (box : Nat) (k : List Nat),
        (#[] : Array (List Nat))[box]? = some k → False :=
      fun box k at_box => by simp at at_box
    have no_entry : ∀ (k : List Nat) (first : Nat),
        ({} : Std.HashMap (List Nat) Nat)[k]? = some first → False :=
      fun k first entry => by simp at entry
    refine ⟨⟨_, forest⟩, ⟨⟨rfl, fun a mem => (by simp at mem),
        fun box node _ at_box => (no_node box node at_box).elim,
        fun k first entry => (no_entry k first entry).elim,
        fun box k _ at_box => (no_key box k at_box).elim, fun p mem => (by cases mem),
        fun box k first _ at_box => (no_key box k at_box).elim⟩,
      (by simp [n]), (by simp [n]), fun r lt is_root => ?_, fun x lt => ?_,
      fun r _ _ _ => range_get r, fun node mem => (by simp at mem),
      fun box k r at_box => (no_key box k at_box).elim, fun r box mem => ?_⟩⟩
    · show rep g.classes.toArray ((List.range n).toArray.getD r r) = r
      rw [range_get]; exact is_root
    · show (List.range n).toArray.getD (rep g.classes.toArray x) (rep g.classes.toArray x) ≤ x
      rw [range_get, roots]; exact valid.normal.find_le x
    · have mem : box ∈ (Array.replicate n ([] : List Nat)).getD r [] := mem
      rw [Array.getD_eq_getD_getElem?] at mem
      cases found : (Array.replicate n ([] : List Nat))[r]? with
      | none => simp [found] at mem
      | some v =>
        obtain ⟨_, eq⟩ := Array.getElem?_eq_some_iff.1 found
        simp at eq
        subst eq
        rw [found] at mem
        cases mem
  have ⟨inv, size, roots', least, nodes⟩ := step g.nodes _ first (by simp [n]) roots rfl
    fun node mem => ⟨inputs node mem, valid.outputs node mem,
      fun a a_mem eq => valid.arity a (by simpa using a_mem) node mem eq⟩
  have inv : Inv (start g) := inv
  have size : (start g).label.size = n := size
  have least : (start g).least = (List.range n).toArray := least
  have roots' : ∀ x, rep (start g).label x = EGraph.find g.classes x := roots'
  refine ⟨inv, Graph.ext (nodes.trans (List.nil_append _)) ?_⟩
  obtain ⟨rank, forest'⟩ := inv.forest
  show (start g).classes = g.classes
  rw [classes_eq forest']
  rw [size, least]
  rw [Graph.eq_map_find g.classes]
  refine List.map_congr_left fun x _ => ?_
  rw [range_get]
  exact roots' x

end Builder

/-! ## The law -/

namespace Graph

/-- An operation the Python accepts: spiders that are there, and a box with
as many outputs as the nodes of its head. -/
def Allows (g : Graph) : Operation → Prop
  | .add head _ arity inputs => (∀ s ∈ inputs, s < g.classes.length)
    ∧ ∀ node ∈ g.nodes, node.head = head → node.outputs.length = arity
  | .merge spiders => ∀ s ∈ spiders, s < g.classes.length
  | .rebuild => True

/-- A history the Python accepts. -/
def Accepts (g : Graph) : List Operation → Prop
  | [] => True
  | operation :: operations => g.Allows operation ∧ (g.apply operation).Accepts operations

end Graph

namespace Builder

theorem apply_spec {b : Builder} (h : Inv b) (operation : Operation)
    (allowed : b.freeze.Allows operation) :
    Inv (b.apply operation) ∧ (b.apply operation).freeze = b.freeze.apply operation := by
  cases operation with
  | add head det arity inputs =>
    have ⟨inv, eq⟩ := add_spec h head det arity inputs
      (by rw [← h.classes_length]; exact allowed.1) allowed.2
    exact ⟨inv, congrArg Prod.fst eq⟩
  | merge spiders => exact merge_spec h spiders (by rw [← h.classes_length]; exact allowed)
  | rebuild => exact rebuild_spec h

theorem fold_spec : ∀ (operations : List Operation) (b : Builder), Inv b →
    b.freeze.Accepts operations →
    Inv (operations.foldl Builder.apply b)
      ∧ (operations.foldl Builder.apply b).freeze = b.freeze.fold operations := by
  intro operations
  induction operations with
  | nil => intro b h _; exact ⟨h, rfl⟩
  | cons operation operations ih =>
    intro b h accepted
    have ⟨inv, eq⟩ := apply_spec h operation accepted.1
    have ⟨inv', eq'⟩ := ih _ inv (eq ▸ accepted.2)
    simp only [List.foldl_cons, Graph.fold]
    exact ⟨inv', by rw [eq', eq]; rfl⟩

/-- A builder freezes into what its operations give in turn. -/
theorem building (g : Graph) (valid : g.Valid)
    (inputs : ∀ node ∈ g.nodes, ∀ s ∈ node.inputs, s < g.classes.length)
    (operations : List Operation) (accepted : g.Accepts operations) :
    build g operations = g.fold operations := by
  have ⟨inv, frozen⟩ := start_spec g valid inputs
  have := (fold_spec operations (start g) inv (by rw [frozen]; exact accepted)).2
  rw [frozen] at this
  exact this

/-- After a history, adding a box on the builder gives the outputs that
adding it on the e-graph gives. -/
theorem adding (g : Graph) (valid : g.Valid)
    (inputs : ∀ node ∈ g.nodes, ∀ s ∈ node.inputs, s < g.classes.length)
    (operations : List Operation) (accepted : g.Accepts operations)
    (head : Nat) (det : Bool) (arity : Nat) (spiders : List Nat)
    (allowed : (g.fold operations).Allows (.add head det arity spiders)) :
    ((operations.foldl Builder.apply (start g)).add head det arity spiders).2
      = ((g.fold operations).add head det arity spiders).2 := by
  have ⟨inv, frozen⟩ := start_spec g valid inputs
  have ⟨inv', eq⟩ := fold_spec operations (start g) inv (by rw [frozen]; exact accepted)
  rw [frozen] at eq
  have := add_spec inv' head det arity spiders
    (by rw [← inv'.classes_length, eq]; exact allowed.1) (by
      have := allowed.2
      rw [← eq] at this
      exact this)
  rw [eq] at this
  exact congrArg Prod.snd this.2

end Builder

end EGraph
