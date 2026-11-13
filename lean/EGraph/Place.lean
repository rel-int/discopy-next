import EGraph.Hashed

/-!
# Placing a node

`Builder.place` keeps the invariant: a deterministic node it appends is
entered in the hash-cons and becomes a parent of each root of its key, and a
sample, whose key is empty, is only appended.
-/

namespace EGraph

namespace Builder

theorem cons_nodes (b : Builder) (box : Nat) : (b.cons box).nodes = b.nodes := by
  simp only [Builder.cons]; split <;> (try split) <;> rfl

theorem cons_keys (b : Builder) (box : Nat) : (b.cons box).keys = b.keys := by
  simp only [Builder.cons]; split <;> (try split) <;> rfl

theorem cons_least (b : Builder) (box : Nat) : (b.cons box).least = b.least := by
  simp only [Builder.cons]; split <;> (try split) <;> rfl

theorem cons_parents (b : Builder) (box : Nat) : (b.cons box).parents = b.parents := by
  simp only [Builder.cons]; split <;> (try split) <;> rfl

theorem cons_label (b : Builder) (box : Nat) : (b.cons box).label = b.label := by
  simp only [Builder.cons]; split <;> (try split) <;> rfl

theorem Hashed.reparent {ρ : Nat → Nat} {stale : Nat → Prop} {b : Builder}
    (h : Hashed ρ stale b) (parents : Array (List Nat)) :
    Hashed ρ stale { b with parents := parents } :=
  ⟨h.1, h.2, h.3, h.4, h.5, h.6, h.7⟩

/-- A builder with one more node and its key. -/
abbrev pushed (b : Builder) (node : Node) (k : List Nat) : Builder :=
  { b with nodes := b.nodes.push node, keys := b.keys.push k }

/-- Appending a node with its key: it is stale when it is deterministic,
until it is entered in the hash-cons. -/
theorem Hashed.push {ρ : Nat → Nat} {b : Builder} (h : Hashed ρ (fun _ => False) b)
    (node : Node) (arity : ∀ a ∈ b.nodes.toList, a.head = node.head →
      a.outputs.length = node.outputs.length) :
    Hashed ρ (fun x => x = b.nodes.size ∧ node.det = true)
      { b with nodes := b.nodes.push node, keys := b.keys.push (key ρ node) } := by
  have nodes_get : ∀ (i : Nat) (x : Node), (b.nodes.push node)[i]? = some x →
      (i = b.nodes.size ∧ x = node) ∨ b.nodes[i]? = some x := by
    intro i x at_i
    rw [Array.getElem?_push] at at_i
    split at at_i
    · exact .inl ⟨by assumption, (Option.some.inj at_i).symm⟩
    · exact .inr at_i
  have keys_get : ∀ (i : Nat) (k : List Nat), (b.keys.push (key ρ node))[i]? = some k →
      (i = b.nodes.size ∧ k = key ρ node) ∨ b.keys[i]? = some k := by
    intro i k at_i
    rw [Array.getElem?_push, h.keys_size] at at_i
    split at at_i
    · exact .inl ⟨by assumption, (Option.some.inj at_i).symm⟩
    · exact .inr at_i
  have keys_old : ∀ (i : Nat) (k : List Nat), b.keys[i]? = some k →
      (b.keys.push (key ρ node))[i]? = some k := by
    intro i k at_i
    obtain ⟨lt, _⟩ := Array.getElem?_eq_some_iff.1 at_i
    rw [Array.getElem?_push]; simp [Nat.ne_of_lt lt, at_i]
  have nodes_old : ∀ (i : Nat) (x : Node), b.nodes[i]? = some x →
      (b.nodes.push node)[i]? = some x := by
    intro i x at_i
    obtain ⟨lt, _⟩ := Array.getElem?_eq_some_iff.1 at_i
    rw [Array.getElem?_push]; simp [Nat.ne_of_lt lt, at_i]
  have key_lt : ∀ (i : Nat) (k : List Nat), b.keys[i]? = some k → i < b.nodes.size := by
    intro i k at_i
    obtain ⟨lt, _⟩ := Array.getElem?_eq_some_iff.1 at_i
    rw [h.keys_size] at lt; exact lt
  have sample : ∀ i : Nat, ¬ (i = b.nodes.size ∧ node.det = true) → i = b.nodes.size →
      key ρ node = [] := by
    intro i fresh eq
    have : node.det = false := by
      cases det : node.det
      · rfl
      · exact (fresh ⟨eq, det⟩).elim
    simp [key, this]
  have grow (b' : Builder) (work : b'.worklist = b.worklist)
      (outs : ∀ i, i < b.nodes.size → b'.outputs i = b.outputs i) {i j : Nat}
      (pending : Pending ρ b i j) : Pending ρ b' i j := by
    refine Pending.mono (b' := b') (fun _ _ eq => .of_class eq) (fun p mem q zip => ?_) pending
    obtain ⟨a, c, at_a, at_c, _⟩ := h.work p mem
    refine .of_pair (p := p) (work ▸ mem) ?_
    rw [outs _ (Array.getElem?_eq_some_iff.1 at_a).1,
      outs _ (Array.getElem?_eq_some_iff.1 at_c).1]
    exact zip
  refine ⟨by simp [h.keys_size], fun a a_mem c c_mem head => ?_,
    fun x node' fresh at_x => ?_, fun k first entry => ?_,
    fun x k fresh at_x ne => ?_, fun p mem => ?_, fun x k first fresh at_x entry => ?_⟩
  · simp only [Array.toList_push, List.mem_append, List.mem_singleton] at a_mem c_mem
    rcases a_mem with a_mem | rfl <;> rcases c_mem with c_mem | rfl
    · exact h.arity a a_mem c c_mem head
    · exact arity a a_mem head
    · exact (arity c c_mem head.symm).symm
    · rfl
  · rcases nodes_get x node' at_x with ⟨eq, rfl⟩ | old
    · subst eq
      show (b.keys.push (key ρ node'))[b.nodes.size]? = _
      rw [Array.getElem?_push, h.keys_size]; simp
    · exact keys_old _ _ (h.keys x node' (fun f => f) old)
  · have ⟨ne, _, keyed, least⟩ := h.hash_some k first entry
    refine ⟨ne, fun s => Nat.lt_irrefl _ (s.1 ▸ key_lt _ _ keyed), keys_old _ _ keyed,
      fun x fresh at_x => ?_⟩
    rcases keys_get x k at_x with ⟨eq, rfl⟩ | old
    · exact (ne (sample x fresh eq)).elim
    · exact least x (fun f => f) old
  · rcases keys_get x k at_x with ⟨eq, rfl⟩ | old
    · exact (ne (sample x fresh eq)).elim
    · exact h.hash_none x k (fun f => f) old ne
  · obtain ⟨a, c, at_a, at_c, rest⟩ := h.work p mem
    exact ⟨a, c, nodes_old _ _ at_a, nodes_old _ _ at_c, rest⟩
  · have ⟨ne, _, keyed, _⟩ := h.hash_some k first entry
    rcases keys_get x k at_x with ⟨eq, rfl⟩ | old
    · exact (ne (sample x fresh eq)).elim
    · rw [outputs_push _ _ _ (key_lt _ _ keyed), outputs_push _ _ _ (key_lt _ _ old)]
      refine Pointwise.mono (fun _ _ pending => ?_) (h.linked x k first (fun f => f) old entry)
      exact grow (b.pushed node (key ρ node)) rfl (fun i lt => outputs_push _ _ _ lt) pending

/-! ## Parents -/

theorem parents_fold (box : Nat) : ∀ (l : List Nat) (b : Builder),
    l.foldl (fun b root => { b with parents := b.parents.modify root (· ++ [box]) }) b
      = { b with parents :=
          l.foldl (fun parents root => parents.modify root (· ++ [box])) b.parents } := by
  intro l
  induction l with
  | nil => intro b; rfl
  | cons r l ih => intro b; simp only [List.foldl_cons]; rw [ih]

theorem mem_modify (parents : Array (List Nat)) (a box r x : Nat) :
    x ∈ (parents.modify a (· ++ [box])).getD r []
      ↔ x ∈ parents.getD r [] ∨ (x = box ∧ r = a ∧ r < parents.size) := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_modify]
  rcases Nat.lt_or_ge r parents.size with lt | ge
  · rw [Array.getElem?_eq_getElem lt]
    by_cases eq : a = r
    · subst eq; simp [lt]
    · simp [eq, Ne.symm eq]
  · rw [Array.getElem?_eq_none ge]
    simp; omega

theorem mem_modify_fold (box : Nat) : ∀ (l : List Nat) (parents : Array (List Nat)),
    (l.foldl (fun parents root => parents.modify root (· ++ [box])) parents).size
        = parents.size
      ∧ ∀ r x : Nat,
        x ∈ (l.foldl (fun parents root => parents.modify root (· ++ [box])) parents).getD r []
          ↔ x ∈ parents.getD r [] ∨ (x = box ∧ r ∈ l ∧ r < parents.size) := by
  intro l
  induction l with
  | nil => intro parents; simp
  | cons a l ih =>
    intro parents
    have ⟨size, mem⟩ := ih (parents.modify a (· ++ [box]))
    simp only [List.foldl_cons]
    refine ⟨by rw [size]; simp, fun r x => ?_⟩
    rw [mem, mem_modify]
    simp only [Array.size_modify, List.mem_cons]
    constructor
    · rintro ((old | ⟨rfl, rfl, lt⟩) | ⟨rfl, tail, lt⟩)
      · exact .inl old
      · exact .inr ⟨rfl, .inl rfl, lt⟩
      · exact .inr ⟨rfl, .inr tail, lt⟩
    · rintro (old | ⟨rfl, rfl | tail, lt⟩)
      · exact .inl (.inl old)
      · exact .inl (.inr ⟨rfl, rfl, lt⟩)
      · exact .inr ⟨rfl, tail, lt⟩

/-! ## Placing -/

theorem place_sample (b : Builder) (node : Node) :
    b.place node [] = { b with nodes := b.nodes.push node, keys := b.keys.push [] } := by
  simp [place]

theorem place_det (b : Builder) (node : Node) (k : List Nat) (ne : k ≠ []) :
    b.place node k = { ((b.pushed node k).cons b.nodes.size) with
      parents := k.tail.foldl
        (fun parents root => parents.modify root (· ++ [b.nodes.size])) b.parents } := by
  unfold place
  simp only [ne, ite_false]
  rw [parents_fold, cons_parents]

/-- The rest of the invariant once a node is appended and entered. -/
theorem Tables.of_pushed {n : Nat} {ρ : Nat → Nat} {b : Builder} (h : Tables n ρ b)
    (node : Node) (inputs : ∀ s ∈ node.inputs, s < n)
    (outputs : ∀ s ∈ node.outputs, s < n) (roots : ∀ s ∈ node.inputs, ρ s < n)
    (x : Builder) (hashed : Hashed ρ (fun _ => False) x)
    (nodes_eq : x.nodes = b.nodes.push node)
    (keys_eq : x.keys = b.keys.push (key ρ node)) (least_eq : x.least = b.least) :
    Tables n ρ { x with
      parents := (key ρ node).tail.foldl
        (fun parents root => parents.modify root (· ++ [b.nodes.size])) b.parents } := by
  have keys_size := h.hashed.keys_size
  have ⟨size, mem⟩ := mem_modify_fold b.nodes.size (key ρ node).tail b.parents
  have keys_get : ∀ (i : Nat) (k : List Nat), (b.keys.push (key ρ node))[i]? = some k →
      (i = b.nodes.size ∧ k = key ρ node) ∨ b.keys[i]? = some k := by
    intro i k at_i
    rw [Array.getElem?_push, keys_size] at at_i
    split at at_i
    · exact .inl ⟨by assumption, (Option.some.inj at_i).symm⟩
    · exact .inr at_i
  have keys_old : ∀ (i : Nat) (k : List Nat), b.keys[i]? = some k →
      (b.keys.push (key ρ node))[i]? = some k := by
    intro i k at_i
    obtain ⟨lt, _⟩ := Array.getElem?_eq_some_iff.1 at_i
    rw [Array.getElem?_push]; simp [Nat.ne_of_lt lt, at_i]
  refine ⟨hashed.reparent _, least_eq ▸ h.least_size, by rw [← h.parents_size]; exact size,
    least_eq ▸ h.least_class, least_eq ▸ h.least_le, fun r lt is_root empty => ?_,
    fun y y_mem => ?_, fun y k r at_y r_mem => ?_, fun r y y_mem => ?_⟩
  · show x.least.getD r r = r
    rw [least_eq]
    refine h.least_plain r lt is_root (List.eq_nil_iff_forall_not_mem.2 fun y y_mem => ?_)
    have := (mem r y).2 (.inl y_mem)
    rw [show _ = [] from empty] at this
    cases this
  · have y_mem : y ∈ x.nodes.toList := y_mem
    rw [nodes_eq] at y_mem
    simp only [Array.toList_push, List.mem_append, List.mem_singleton] at y_mem
    rcases y_mem with old | rfl
    · exact h.wires y old
    · exact ⟨inputs, outputs⟩
  · have at_y : x.keys[y]? = some k := at_y
    rw [keys_eq] at at_y
    refine (mem r y).2 ?_
    rcases keys_get y k at_y with ⟨rfl, rfl⟩ | old
    · obtain ⟨s, s_mem, rfl⟩ := key_tail r_mem
      exact .inr ⟨rfl, r_mem, h.parents_size ▸ roots s s_mem⟩
    · exact .inl (h.parents_mem y k r old r_mem)
  · show ∃ k, x.keys[y]? = some k ∧ _
    rw [keys_eq]
    rcases (mem r y).1 y_mem with old | ⟨rfl, tail, _⟩
    · obtain ⟨k, at_y, r_mem⟩ := h.parents_key r y old
      exact ⟨k, keys_old _ _ at_y, r_mem⟩
    · exact ⟨key ρ node, by rw [Array.getElem?_push, keys_size]; simp, tail⟩

/-- Placing a node on spiders with its key keeps the invariant. -/
theorem Tables.place {n : Nat} {ρ : Nat → Nat} {b : Builder} (h : Tables n ρ b)
    (node : Node) (inputs : ∀ s ∈ node.inputs, s < n)
    (outputs : ∀ s ∈ node.outputs, s < n) (roots : ∀ s ∈ node.inputs, ρ s < n)
    (arity : ∀ a ∈ b.nodes.toList, a.head = node.head →
      a.outputs.length = node.outputs.length) :
    Tables n ρ (b.place node (key ρ node)) := by
  have pushed := h.hashed.push node arity
  cases det : node.det with
  | false =>
    have empty : key ρ node = [] := by simp [key, det]
    have := h.of_pushed node inputs outputs roots _
      (pushed.stale_iff fun x => ⟨False.elim, fun s => by simp [det] at s⟩) rfl rfl rfl
    rw [empty] at this ⊢
    rw [place_sample]
    exact this
  | true =>
    have ne : key ρ node ≠ [] := by simp [key, det]
    rw [place_det _ _ _ ne]
    refine h.of_pushed node inputs outputs roots
      ((b.pushed node (key ρ node)).cons b.nodes.size) ?_ (cons_nodes _ _) (cons_keys _ _)
      (cons_least _ _)
    have := pushed.cons (box := b.nodes.size) (node := node)
      (by simp) (by
        show (b.keys.push (key ρ node))[b.nodes.size]? = _
        rw [Array.getElem?_push, h.hashed.keys_size]; simp) det
    exact this.stale_iff fun x => ⟨False.elim, fun s => s.2 s.1.1⟩

end Builder

end EGraph
