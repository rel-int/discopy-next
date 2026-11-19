import EGraph.Invariant

/-!
# The hash-cons between two methods

The part of the invariant that `cons` keeps: what the keys, the hash-cons and
the worklist say of the nodes that are not stale, a node being stale while
its key waits to be entered again.
-/

namespace EGraph

theorem pointwise_iff {r : Nat → Nat → Prop} {xs ys : List Nat} :
    Pointwise r xs ys ↔ xs.length = ys.length
      ∧ ∀ (i x y : Nat), xs[i]? = some x → ys[i]? = some y → r x y := by
  constructor
  · intro ⟨length, h⟩
    refine ⟨length, fun i x y hx hy => h (x, y) ?_⟩
    exact List.mem_iff_getElem?.2 ⟨i, List.getElem?_zip_eq_some.2 ⟨hx, hy⟩⟩
  · intro ⟨length, h⟩
    refine ⟨length, fun p mem => ?_⟩
    obtain ⟨i, at_i⟩ := List.mem_iff_getElem?.1 mem
    have := List.getElem?_zip_eq_some.1 at_i
    exact h i _ _ this.1 this.2

theorem Pointwise.symm {r : Nat → Nat → Prop} (eqv : Equivalence r)
    {xs ys : List Nat} (h : Pointwise r xs ys) : Pointwise r ys xs := by
  have ⟨length, h⟩ := pointwise_iff.1 h
  exact pointwise_iff.2 ⟨length.symm, fun i x y hx hy => eqv.symm (h i y x hy hx)⟩

theorem Pointwise.trans {r : Nat → Nat → Prop}
    (eqv : Equivalence r) {xs ys zs : List Nat}
    (h : Pointwise r xs ys) (h' : Pointwise r ys zs) : Pointwise r xs zs := by
  have ⟨length, h⟩ := pointwise_iff.1 h
  have ⟨length', h'⟩ := pointwise_iff.1 h'
  refine pointwise_iff.2 ⟨length.trans length', fun i x z hx hz => ?_⟩
  have lt : i < ys.length := by
    have := (List.getElem?_eq_some_iff.1 hx).1; omega
  exact eqv.trans (h i x _ hx (List.getElem?_eq_getElem lt))
    (h' i _ z (List.getElem?_eq_getElem lt) hz)

namespace Builder

theorem Hashed.node_of_key {ρ : Nat → Nat} {stale : Nat → Prop} {b : Builder}
    (h : Hashed ρ stale b) {box : Nat} {k : List Nat} (fresh : ¬ stale box)
    (some : b.keys[box]? = some k) :
    ∃ node, b.nodes[box]? = .some node ∧ k = key ρ node := by
  obtain ⟨lt, _⟩ := Array.getElem?_eq_some_iff.1 some
  rw [h.keys_size] at lt
  refine ⟨b.nodes[box], Array.getElem?_eq_getElem lt, ?_⟩
  have := h.keys box _ fresh (Array.getElem?_eq_getElem lt)
  rw [some] at this
  exact Option.some.inj this

theorem Hashed.stale_iff {ρ : Nat → Nat} {stale stale' : Nat → Prop} {b : Builder}
    (h : Hashed ρ stale b) (iff : ∀ x, stale' x ↔ stale x) : Hashed ρ stale' b :=
  have : stale' = stale := funext fun x => propext (iff x)
  this ▸ h

theorem mem_nodes {b : Builder} {box : Nat} {node : Node}
    (some : b.nodes[box]? = some node) : node ∈ b.nodes.toList := by
  rw [← Array.getElem?_toList] at some
  exact List.mem_of_getElem? some

theorem outputs_of {b : Builder} {box : Nat} {node : Node}
    (some : b.nodes[box]? = some node) : b.outputs box = node.outputs := by
  simp [outputs, some]

/-- Entering a node under its key keeps the invariant, the node no longer
stale. -/
theorem Hashed.cons {ρ : Nat → Nat} {stale : Nat → Prop} {b : Builder}
    (h : Hashed ρ stale b) {box : Nat} {node : Node}
    (node_at : b.nodes[box]? = some node)
    (keyed : b.keys[box]? = some (key ρ node)) (det : node.det = true) :
    Hashed ρ (fun x => stale x ∧ x ≠ box) (b.cons box) := by
  have ne : key ρ node ≠ [] := by simp [key, det]
  have getD : b.keys.getD box [] = key ρ node := by
    simp [Array.getD_eq_getD_getElem?, keyed]
  have grow {b' : Builder} (work : ∀ p ∈ b.worklist, p ∈ b'.worklist) {i j : Nat}
      (pending : Pending ρ b i j) (nodes : b'.nodes = b.nodes := by rfl) :
      Pending ρ b' i j :=
    pending.mono (fun _ _ eq => .of_class eq) fun p mem q zip =>
      .of_pair (work p mem) (by simpa only [outputs, nodes] using zip)
  cases found : b.hashcons[key ρ node]? with
  | none =>
    have eq : b.cons box
        = { b with hashcons := b.hashcons.insert (key ρ node) box } := by
      simp only [Builder.cons, getD, found]
    have alone : ∀ x : Nat, ¬ stale x → b.keys[x]? = some (key ρ node) → False :=
      fun x fresh some => h.hash_none x _ fresh some ne found
    rw [eq]
    refine ⟨h.keys_size, h.arity, fun x node' fresh at_x => ?_,
      fun k first some => ?_, fun x k fresh some ne' => ?_, h.work,
      fun x k first fresh some entry => ?_⟩
    · by_cases is_box : x = box
      · subst is_box; rw [node_at] at at_x; cases at_x; exact keyed
      · exact h.keys x node' (fun s => fresh ⟨s, is_box⟩) at_x
    · simp only [Std.HashMap.getElem?_insert, beq_iff_eq] at some
      split at some
      · rename_i is_key; subst is_key; cases some
        refine ⟨ne, fun s => s.2 rfl, keyed, fun x fresh some => ?_⟩
        by_cases is_box : x = box
        · exact Nat.le_of_eq is_box.symm
        · exact (alone x (fun s => fresh ⟨s, is_box⟩) some).elim
      · have ⟨ne', fresh, keyed', least⟩ := h.hash_some k first some
        refine ⟨ne', fun s => fresh s.1, keyed', fun x fresh' some' => ?_⟩
        by_cases is_box : x = box
        · subst is_box; rw [keyed] at some'; cases some'; rename_i ne_key; exact (ne_key rfl).elim
        · exact least x (fun s => fresh' ⟨s, is_box⟩) some'
    · simp only [Std.HashMap.getElem?_insert, beq_iff_eq]
      split
      · simp
      · by_cases is_box : x = box
        · subst is_box; rw [keyed] at some; cases some; rename_i ne_key; exact (ne_key rfl).elim
        · exact h.hash_none x k (fun s => fresh ⟨s, is_box⟩) some ne'
    · simp only [Std.HashMap.getElem?_insert, beq_iff_eq] at entry
      split at entry
      · rename_i is_key; subst is_key; cases entry
        by_cases is_box : x = box
        · subst is_box
          exact Pointwise.refl' (Pending.equivalence ρ _).refl _
        · exact (alone x (fun s => fresh ⟨s, is_box⟩) some).elim
      · by_cases is_box : x = box
        · subst is_box; rw [keyed] at some; cases some; rename_i ne_key; exact (ne_key rfl).elim
        · exact (h.linked x k first (fun s => fresh ⟨s, is_box⟩) some entry).mono
            fun _ _ pending => grow (fun _ mem => mem) pending
  | some first =>
    have ⟨_, first_fresh, first_key, least⟩ := h.hash_some _ first found
    by_cases same : first = box
    · have eq : b.cons box = b := by simp [Builder.cons, getD, found, same]
      rw [eq]
      exact h.stale_iff fun x => ⟨fun s => s.1, fun s => ⟨s, fun is_box =>
        first_fresh (same ▸ is_box ▸ s)⟩⟩
    · have eq : b.cons box = { b with
          hashcons := b.hashcons.insert (key ρ node) (min first box)
          worklist := (first, box) :: b.worklist } := by
        simp [Builder.cons, getD, found, same]
      obtain ⟨first_node, first_at, first_eq⟩ := h.node_of_key first_fresh first_key
      have ⟨first_det, _, head, inputs⟩ := key_eq first_eq.symm (first_eq ▸ ne)
      have pair : Pointwise (Pending ρ (b.cons box)) (b.outputs first) (b.outputs box) := by
        rw [eq]
        refine ⟨?_, fun q zip => .of_pair (p := (first, box))
          (List.mem_cons_self ..) zip⟩
        rw [outputs_of first_at, outputs_of node_at]
        exact h.arity _ (mem_nodes first_at) _ (mem_nodes node_at) head
      have old {x : Nat} (fresh : ¬ stale x) (some : b.keys[x]? = some (key ρ node)) :
          Pointwise (Pending ρ (b.cons box)) (b.outputs first) (b.outputs x) := by
        rw [eq]
        exact (h.linked x _ first fresh some found).mono
          fun _ _ pending => grow (fun _ mem => List.mem_cons_of_mem _ mem) pending
      have eqv := Pending.equivalence ρ (b.cons box)
      rw [eq] at pair old eqv ⊢
      refine ⟨h.keys_size, h.arity, fun x node' fresh at_x => ?_,
        fun k entry some => ?_, fun x k fresh some ne' => ?_, fun p mem => ?_,
        fun x k entry fresh some found' => ?_⟩
      · by_cases is_box : x = box
        · subst is_box; rw [node_at] at at_x; cases at_x; exact keyed
        · exact h.keys x node' (fun s => fresh ⟨s, is_box⟩) at_x
      · simp only [Std.HashMap.getElem?_insert, beq_iff_eq] at some
        split at some
        · rename_i is_key; subst is_key; cases some
          refine ⟨ne, fun s => ?_, ?_, fun x fresh some => ?_⟩
          · rcases Nat.le_total first box with le | le
            · rw [Nat.min_eq_left le] at s; exact first_fresh s.1
            · rw [Nat.min_eq_right le] at s; exact s.2 rfl
          · rcases Nat.le_total first box with le | le
            · rw [Nat.min_eq_left le]; exact first_key
            · rw [Nat.min_eq_right le]; exact keyed
          · by_cases is_box : x = box
            · subst is_box; exact Nat.min_le_right _ _
            · exact Nat.le_trans (Nat.min_le_left _ _)
                (least x (fun s => fresh ⟨s, is_box⟩) some)
        · have ⟨ne', fresh, keyed', least'⟩ := h.hash_some k entry some
          refine ⟨ne', fun s => fresh s.1, keyed', fun x fresh' some' => ?_⟩
          by_cases is_box : x = box
          · subst is_box; rw [keyed] at some'; cases some'; rename_i ne_key; exact (ne_key rfl).elim
          · exact least' x (fun s => fresh' ⟨s, is_box⟩) some'
      · simp only [Std.HashMap.getElem?_insert, beq_iff_eq]
        split
        · simp
        · by_cases is_box : x = box
          · subst is_box; rw [keyed] at some; cases some; rename_i ne_key; exact (ne_key rfl).elim
          · exact h.hash_none x k (fun s => fresh ⟨s, is_box⟩) some ne'
      · rcases List.mem_cons.1 mem with rfl | mem
        · exact ⟨first_node, node, first_at, node_at, first_det, det, head, inputs⟩
        · exact h.work p mem
      · simp only [Std.HashMap.getElem?_insert, beq_iff_eq] at found'
        split at found'
        · rename_i is_key; subst is_key; cases found'
          have to (x : Nat) (fresh : ¬ (stale x ∧ x ≠ box))
              (some : b.keys[x]? = .some (key ρ node)) :
              Pointwise (Pending ρ { b with
                hashcons := b.hashcons.insert (key ρ node) (min first box)
                worklist := (first, box) :: b.worklist }) (b.outputs first) (b.outputs x) := by
            by_cases is_box : x = box
            · subst is_box; exact pair
            · exact old (fun s => fresh ⟨s, is_box⟩) some
          rcases Nat.le_total first box with le | le
          · rw [Nat.min_eq_left le]; exact to x fresh some
          · rw [Nat.min_eq_right le]
            exact Pointwise.trans eqv (Pointwise.symm eqv pair) (to x fresh some)
        · by_cases is_box : x = box
          · subst is_box; rw [keyed] at some; cases some; rename_i ne_key; exact (ne_key rfl).elim
          · exact (h.linked x k entry (fun s => fresh ⟨s, is_box⟩) some found').mono
              fun _ _ pending => grow (fun _ mem => List.mem_cons_of_mem _ mem) pending

end Builder

end EGraph
