import EGraph.Place

/-!
# A class joins another

What `Builder.union` does to the tables when the root `d` of a class joins
the root `s` of another: the roots become `swap d s ∘ ρ`.
-/

namespace EGraph

namespace Builder

/-- The root `d` becomes `s`. -/
def swap (d s r : Nat) : Nat := if r = d then s else r

theorem key_swap_tail (ρ : Nat → Nat) (d s : Nat) (node : Node) :
    (key (fun x => swap d s (ρ x)) node).tail = (key ρ node).tail.map (swap d s) := by
  unfold key; split <;> simp

/-- A key that does not read `d` stays as it is. -/
theorem key_swap {ρ : Nat → Nat} {d s : Nat} {node : Node}
    (fresh : d ∉ (key ρ node).tail) : key (fun x => swap d s (ρ x)) node = key ρ node := by
  unfold key at fresh ⊢
  split
  · simp only [List.tail_cons, List.mem_map, not_exists, not_and, ‹node.det = true›,
      ite_true] at fresh
    simp only [List.cons.injEq, true_and]
    refine List.map_congr_left fun x mem => ?_
    simp only [swap]
    split
    · rename_i eq; exact (fresh x mem eq).elim
    · rfl
  · rfl

theorem swap_ne {d s : Nat} (ne : s ≠ d) (r : Nat) : swap d s r ≠ d := by
  unfold swap; split
  · exact ne
  · assumption

/-- The quick way: a root with no parent joins a lesser root. -/
theorem Tables.link_plain {n : Nat} {ρ : Nat → Nat} {b : Builder} (h : Tables n ρ b)
    {s d : Nat} (hs : ρ s = s) (hd : ρ d = d) (lt : s < d) (ld : d < n)
    (plain : b.parents.getD d [] = []) :
    Tables n (fun x => swap d s (ρ x)) b := by
  have fresh : ∀ (box : Nat) (node : Node), b.nodes[box]? = some node →
      d ∉ (key ρ node).tail := by
    intro box node at_box mem
    have := h.parents_mem box _ d (h.hashed.keys box node (fun f => f) at_box) mem
    rw [plain] at this; cases this
  have root : ∀ r, swap d s (ρ r) = r → ρ r = r ∧ r ≠ d := by
    intro r eq
    unfold swap at eq
    split at eq
    · rename_i is_d; subst eq; rw [hs] at is_d; exact absurd is_d (Nat.ne_of_lt lt)
    · exact ⟨eq, fun is_d => by subst is_d; simp_all⟩
  have keep : ∀ r, r ≠ d → swap d s r = r := fun r ne => by simp [swap, ne]
  refine ⟨⟨h.hashed.keys_size, h.hashed.arity, fun box node _ at_box => ?_,
      h.hashed.hash_some, h.hashed.hash_none, fun p mem => ?_,
      fun box k first _ at_box entry => ?_⟩,
    h.least_size, h.parents_size, fun r lt' is_root => ?_, fun x lt' => ?_,
    fun r lt' is_root empty => ?_, h.wires, h.parents_mem, h.parents_key⟩
  · rw [key_swap (fresh box node at_box)]
    exact h.hashed.keys box node (fun f => f) at_box
  · obtain ⟨a, c, at_a, at_c, da, dc, head, inputs⟩ := h.hashed.work p mem
    refine ⟨a, c, at_a, at_c, da, dc, head, ?_⟩
    have := congrArg (List.map (swap d s)) inputs
    simpa only [List.map_map, Function.comp_def] using this
  · exact (h.hashed.linked box k first (fun f => f) at_box entry).mono fun _ _ pending =>
      pending.mono (fun _ _ eq => .of_class (congrArg (swap d s) eq))
        fun p mem q zip => .of_pair mem zip
  · have ⟨old, ne⟩ := root r is_root
    show swap d s (ρ (b.least.getD r r)) = r
    rw [h.least_class r lt' old, keep r ne]
  · show b.least.getD (swap d s (ρ x)) (swap d s (ρ x)) ≤ x
    by_cases is_d : ρ x = d
    · have least_s := h.least_le s (Nat.lt_trans lt ld)
      have least_x := h.least_le x lt'
      rw [hs] at least_s
      rw [is_d, h.least_plain d ld hd plain] at least_x
      simp only [swap, is_d, ite_true]
      omega
    · rw [keep _ is_d]; exact h.least_le x lt'
  · exact h.least_plain r lt' (root r is_root).1 empty

/-! ## Moving the parents of a class -/

theorem erase_fold (keys : Array (List Nat)) :
    ∀ (moved : List Nat) (hashcons : Std.HashMap (List Nat) Nat) (k : List Nat),
      (moved.foldl (fun hashcons box => hashcons.erase (keys.getD box [])) hashcons)[k]?
        = if ∃ box ∈ moved, keys.getD box [] = k then none else hashcons[k]? := by
  intro moved
  induction moved with
  | nil => intro hashcons k; simp
  | cons x xs ih =>
    intro hashcons k
    simp only [List.foldl_cons, ih, Std.HashMap.getElem?_erase, beq_iff_eq, List.mem_cons,
      exists_eq_or_imp]
    split <;> split <;> simp_all

/-- A node is stale when its key still reads the root `d`. -/
def Stale (d : Nat) (b : Builder) (x : Nat) : Prop :=
  ∃ k, b.keys[x]? = some k ∧ d ∈ k.tail

/-- The invariant while the parents of `d` move to their new keys: the
hash-cons is right for the roots `ρ'` but for the stale nodes, which keep
their key for the roots `ρ`. -/
structure Moving (ρ ρ' : Nat → Nat) (d : Nat) (b : Builder) : Prop where
  hashed : Hashed ρ' (Stale d b) b
  old : ∀ (box : Nat) (node : Node), b.nodes[box]? = some node → Stale d b box →
    b.keys[box]? = some (key ρ node)

theorem getElem?_set_keys (keys : Array (List Nat)) (box : Nat) (k : List Nat) (x : Nat)
    (lt : box < keys.size) :
    (keys.setIfInBounds box k)[x]? = if x = box then some k else keys[x]? := by
  rw [Array.getElem?_setIfInBounds]
  by_cases eq : box = x
  · subst eq; simp [lt]
  · simp [eq, Ne.symm eq]

/-- Moving one node to its new key. -/
theorem Moving.rekey {ρ : Nat → Nat} {d s : Nat} {b : Builder} {rank : Nat → Nat}
    (m : Moving ρ (fun x => swap d s (ρ x)) d b) (ne : s ≠ d) (hs : ρ s = s)
    (idem : ∀ x, ρ (ρ x) = ρ x) (forest : Forest b.label rank)
    (roots : ∀ x, rep b.label x = swap d s (ρ x))
    {box : Nat} {node : Node} (node_at : b.nodes[box]? = some node)
    (det : node.det = true) :
    Moving ρ (fun x => swap d s (ρ x)) d (b.rekey box)
      ∧ (∀ x, Stale d (b.rekey box) x → Stale d b x ∧ x ≠ box)
      ∧ (b.rekey box).nodes = b.nodes ∧ (b.rekey box).least = b.least
      ∧ (b.rekey box).parents = b.parents
      ∧ Forest (b.rekey box).label rank
      ∧ (b.rekey box).label.size = b.label.size
      ∧ (∀ x, rep (b.rekey box).label x = swap d s (ρ x))
      ∧ (b.rekey box).worklist.length ≤ b.worklist.length + 1 := by
  have lt : box < b.keys.size := by
    rw [m.hashed.keys_size]; exact (Array.getElem?_eq_some_iff.1 node_at).1
  have twice : ∀ x, swap d s (ρ (swap d s (ρ x))) = swap d s (ρ x) := by
    intro x; unfold swap
    by_cases is_d : ρ x = d
    · simp [is_d, hs, ne]
    · simp [is_d, idem]
  -- the key of the box reads the old roots or the new ones
  have keyed : ∃ σ : Nat → Nat, b.keys[box]? = some (node.head :: node.inputs.map σ)
      ∧ ∀ x, swap d s (ρ (σ x)) = swap d s (ρ x) := by
    by_cases stale : Stale d b box
    · exact ⟨ρ, by simpa [key, det] using m.old box node node_at stale,
        fun x => by rw [idem]⟩
    · exact ⟨fun x => swap d s (ρ x),
        by simpa [key, det] using m.hashed.keys box node stale node_at, twice⟩
  obtain ⟨σ, at_box, absorb⟩ := keyed
  have ⟨values, label, eq, forest', size, same⟩ :=
    roots_spec (rank := rank) (node.inputs.map σ) b forest
  have new_key : node.head :: (b.roots (node.inputs.map σ)).2
      = key (fun x => swap d s (ρ x)) node := by
    rw [values]
    simp only [key, det, ite_true, List.map_map, List.cons.injEq, true_and]
    exact List.map_congr_left fun x _ => by simp [roots, absorb]
  have unfolded : b.rekey box = Builder.cons { b with
      label := label
      keys := b.keys.setIfInBounds box (key (fun x => swap d s (ρ x)) node) } box := by
    simp only [Builder.rekey, Array.getD_eq_getD_getElem?, at_box, Option.getD_some]
    rw [new_key, eq]
  have fresh_key : d ∉ (key (fun x => swap d s (ρ x)) node).tail := by
    rw [key_swap_tail]
    intro mem
    obtain ⟨r, _, is_d⟩ := List.mem_map.1 mem
    exact swap_ne ne r is_d
  -- updating the key changes nothing for the nodes that are not stale
  have keys_same : ∀ x, ¬ Stale d b x →
      (b.keys.setIfInBounds box (key (fun x => swap d s (ρ x)) node))[x]? = b.keys[x]? := by
    intro x fresh
    rw [getElem?_set_keys _ _ _ _ lt]
    split
    · rename_i is_box; subst is_box
      exact (m.hashed.keys x node fresh node_at).symm
    · rfl
  have updated : Hashed (fun x => swap d s (ρ x)) (Stale d b) { b with
      label := label
      keys := b.keys.setIfInBounds box (key (fun x => swap d s (ρ x)) node) } := by
    have h := m.hashed
    refine ⟨by simpa using h.keys_size, h.arity, fun x node' fresh at_x => ?_,
      fun k first entry => ?_, fun x k fresh at_x ne' => ?_, h.work,
      fun x k first fresh at_x entry => ?_⟩
    · show (b.keys.setIfInBounds _ _)[x]? = _
      rw [keys_same x fresh]; exact h.keys x node' fresh at_x
    · have ⟨ne', fresh, keyed', least⟩ := h.hash_some k first entry
      refine ⟨ne', fresh, (keys_same first fresh).trans keyed', fun x fresh' at_x => ?_⟩
      exact least x fresh' ((keys_same x fresh').symm.trans at_x)
    · exact h.hash_none x k fresh ((keys_same x fresh).symm.trans at_x) ne'
    · exact (h.linked x k first fresh ((keys_same x fresh).symm.trans at_x) entry).mono
        fun _ _ pending => pending.mono (fun _ _ eq => .of_class eq)
          fun p mem q zip => .of_pair mem zip
  have consed := updated.cons (box := box) (node := node) node_at
    (by show (b.keys.setIfInBounds _ _)[box]? = _; rw [getElem?_set_keys _ _ _ _ lt]; simp) det
  have stale_iff : ∀ x, Stale d (b.rekey box) x ↔ Stale d b x ∧ x ≠ box := by
    intro x
    rw [unfolded]
    unfold Stale
    rw [cons_keys]
    show (∃ k, (b.keys.setIfInBounds _ _)[x]? = some k ∧ _) ↔ _
    rw [getElem?_set_keys _ _ _ _ lt]
    by_cases is_box : x = box
    · simp only [is_box, ite_true, Option.some.injEq, ne_eq, not_true, and_false, iff_false]
      rintro ⟨k, rfl, mem⟩
      exact fresh_key mem
    · simp [is_box]
  refine ⟨⟨?_, fun x node' at_x stale => ?_⟩, fun x stale => (stale_iff x).1 stale, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_⟩
  · rw [unfolded] at stale_iff ⊢
    exact consed.stale_iff stale_iff
  · have ⟨was, ne_box⟩ := (stale_iff x).1 stale
    rw [unfolded, cons_keys]
    rw [unfolded, cons_nodes] at at_x
    show (b.keys.setIfInBounds _ _)[x]? = _
    rw [getElem?_set_keys _ _ _ _ lt]
    simp only [ne_box, ite_false]
    exact m.old x node' at_x was
  · rw [unfolded, cons_nodes]
  · rw [unfolded, cons_least]
  · rw [unfolded, cons_parents]
  · rw [unfolded, cons_label]; exact forest'
  · rw [unfolded, cons_label]; exact size
  · intro x; rw [unfolded, cons_label]; exact (same x).trans (roots x)
  · rw [unfolded]
    simp only [Builder.cons]
    split
    · exact Nat.le_succ _
    · split
      · exact Nat.le_succ _
      · simp

/-- Moving the nodes of a list to their new keys, in turn. -/
theorem Moving.rekeys {ρ : Nat → Nat} {d s : Nat} {rank : Nat → Nat} (ne : s ≠ d)
    (hs : ρ s = s) (idem : ∀ x, ρ (ρ x) = ρ x) :
    ∀ (todo : List Nat) (b : Builder), Moving ρ (fun x => swap d s (ρ x)) d b →
      Forest b.label rank → (∀ x, rep b.label x = swap d s (ρ x)) →
      (∀ box ∈ todo, ∃ node, b.nodes[box]? = some node ∧ node.det = true) →
      Moving ρ (fun x => swap d s (ρ x)) d (todo.foldl Builder.rekey b)
        ∧ (∀ x, Stale d (todo.foldl Builder.rekey b) x → Stale d b x ∧ x ∉ todo)
        ∧ (todo.foldl Builder.rekey b).nodes = b.nodes
        ∧ (todo.foldl Builder.rekey b).least = b.least
        ∧ (todo.foldl Builder.rekey b).parents = b.parents
        ∧ Forest (todo.foldl Builder.rekey b).label rank
        ∧ (todo.foldl Builder.rekey b).label.size = b.label.size
        ∧ (∀ x, rep (todo.foldl Builder.rekey b).label x = swap d s (ρ x))
        ∧ (todo.foldl Builder.rekey b).worklist.length ≤ b.worklist.length + todo.length := by
  intro todo
  induction todo with
  | nil =>
    intro b m forest roots _
    exact ⟨m, fun x stale => ⟨stale, List.not_mem_nil⟩, rfl, rfl, rfl, forest, rfl, roots,
      Nat.le_refl _⟩
  | cons box rest ih =>
    intro b m forest roots nodes
    obtain ⟨node, node_at, det⟩ := nodes box (List.mem_cons_self ..)
    have ⟨m', stale', nodes', least', parents', forest', size', roots', work'⟩ :=
      m.rekey ne hs idem forest roots node_at det
    have ⟨m'', stale'', nodes'', least'', parents'', forest'', size'', roots'', work''⟩ :=
      ih (b.rekey box) m' forest' roots' fun x mem => by
        rw [nodes']; exact nodes x (List.mem_cons_of_mem _ mem)
    simp only [List.foldl_cons]
    refine ⟨m'', fun x stale => ?_, nodes''.trans nodes', least''.trans least',
      parents''.trans parents', forest'', size''.trans size', roots'', ?_⟩
    · have ⟨was, not_rest⟩ := stale'' x stale
      have ⟨was', ne_box⟩ := stale' x was
      exact ⟨was', fun mem => by
        rcases List.mem_cons.1 mem with rfl | mem
        · exact ne_box rfl
        · exact not_rest mem⟩
    · simp only [List.length_cons]; omega

/-- Before the parents of `d` move: their keys are out of the hash-cons. -/
theorem Tables.moving {n : Nat} {ρ : Nat → Nat} {b : Builder} (h : Tables n ρ b)
    (d s : Nat) (label least : Array Nat) (parents : Array (List Nat)) :
    Moving ρ (fun x => swap d s (ρ x)) d { b with
      parents := parents
      hashcons := (b.parents.getD d []).foldl
        (fun hashcons box => hashcons.erase (b.keys.getD box [])) b.hashcons
      label := label
      least := least } := by
  have moved : ∀ (k : List Nat), (∃ box ∈ b.parents.getD d [], b.keys.getD box [] = k) →
      d ∈ k.tail := by
    rintro k ⟨box, mem, rfl⟩
    obtain ⟨k, at_box, tail⟩ := h.parents_key d box mem
    simpa [Array.getD_eq_getD_getElem?, at_box] using tail
  have erased : ∀ (k : List Nat) (first : Nat), b.hashcons[k]? = some first → d ∈ k.tail →
      ∃ box ∈ b.parents.getD d [], b.keys.getD box [] = k := by
    intro k first entry tail
    have ⟨_, _, keyed, _⟩ := h.hashed.hash_some k first entry
    exact ⟨first, h.parents_mem first k d keyed tail,
      by simp [Array.getD_eq_getD_getElem?, keyed]⟩
  refine ⟨⟨h.hashed.keys_size, h.hashed.arity, fun x node fresh at_x => ?_,
      fun k first entry => ?_, fun x k fresh at_x ne => ?_, fun p mem => ?_,
      fun x k first fresh at_x entry => ?_⟩,
    fun x node at_x _ => h.hashed.keys x node (fun f => f) at_x⟩
  · have old := h.hashed.keys x node (fun f => f) at_x
    show b.keys[x]? = _
    rw [key_swap fun tail => fresh ⟨_, old, tail⟩]
    exact old
  · have entry : (List.foldl _ b.hashcons (b.parents.getD d []))[k]? = some first := entry
    rw [erase_fold] at entry
    split at entry
    · cases entry
    · rename_i kept
      have ⟨ne, _, keyed, least⟩ := h.hashed.hash_some k first entry
      exact ⟨ne, fun ⟨k', at_first, tail⟩ => kept (erased k first entry (by
          have : b.keys[first]? = some k' := at_first
          rw [keyed] at this; cases this; exact tail)),
        keyed, fun x _ at_x => least x (fun f => f) at_x⟩
  · show (List.foldl _ b.hashcons (b.parents.getD d []))[k]? ≠ none
    rw [erase_fold]
    split
    · rename_i gone; exact (fresh ⟨k, at_x, moved k gone⟩).elim
    · exact h.hashed.hash_none x k (fun f => f) at_x ne
  · obtain ⟨a, c, at_a, at_c, da, dc, head, inputs⟩ := h.hashed.work p mem
    refine ⟨a, c, at_a, at_c, da, dc, head, ?_⟩
    have := congrArg (List.map (swap d s)) inputs
    simpa only [List.map_map, Function.comp_def] using this
  · have entry : (List.foldl _ b.hashcons (b.parents.getD d []))[k]? = some first := entry
    rw [erase_fold] at entry
    split at entry
    · cases entry
    · exact (h.hashed.linked x k first (fun f => f) at_x entry).mono fun _ _ pending =>
        pending.mono (fun _ _ eq => .of_class (congrArg (swap d s) eq))
          fun p mem q zip => .of_pair mem zip

theorem getElem?_set {α : Type} (xs : Array α) (i : Nat) (v : α) (j : Nat)
    (lt : i < xs.size) :
    (xs.setIfInBounds i v)[j]? = if j = i then some v else xs[j]? := by
  rw [Array.getElem?_setIfInBounds]
  by_cases eq : i = j
  · subst eq; simp [lt]
  · simp [eq, Ne.symm eq]

theorem parents_linked (parents : Array (List Nat)) {s d : Nat} (ne : s ≠ d)
    (ls : s < parents.size) (ld : d < parents.size) (r : Nat) :
    ((parents.setIfInBounds d []).modify s (· ++ parents.getD d [])).getD r []
      = if r = s then parents.getD s [] ++ parents.getD d []
        else if r = d then [] else parents.getD r [] := by
  obtain ⟨v, at_s⟩ : ∃ v, parents[s]? = some v := ⟨_, Array.getElem?_eq_getElem ls⟩
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_modify, getElem?_set _ _ _ _ ld]
  by_cases is_s : r = s
  · simp [is_s, ne, at_s]
  · simp only [Ne.symm is_s, is_s, ite_false]
    split <;> simp

/-- The tables once the parents of `d` have moved to `s`. -/
theorem Tables.linked {n : Nat} {ρ : Nat → Nat} {b : Builder} (h : Tables n ρ b)
    {s d : Nat} (hs : ρ s = s) (hd : ρ d = d) (ne : s ≠ d) (ls : s < n) (ld : d < n)
    (side : b.parents.getD s [] ≠ [] ∨ b.parents.getD d [] ≠ [])
    (c : Builder) (hashed : Hashed (fun x => swap d s (ρ x)) (fun _ => False) c)
    (nodes : c.nodes = b.nodes)
    (least : c.least = if b.least.getD d d < b.least.getD s s
      then b.least.setIfInBounds s (b.least.getD d d) else b.least)
    (parents : c.parents = b.parents.setIfInBounds d []) :
    Tables n (fun x => swap d s (ρ x))
      { c with parents := c.parents.modify s (· ++ b.parents.getD d []) } := by
  have root : ∀ r, swap d s (ρ r) = r → ρ r = r ∧ r ≠ d := by
    intro r eq
    unfold swap at eq
    split at eq
    · rename_i is_d; subst eq; rw [hs] at is_d; exact absurd is_d ne
    · exact ⟨eq, fun is_d => by subst is_d; simp_all⟩
  have keep : ∀ r, r ≠ d → swap d s r = r := fun r ne => by simp [swap, ne]
  have swap_d : swap d s d = s := by simp [swap]
  have least_get : ∀ r, c.least.getD r r = if r = s
      then min (b.least.getD s s) (b.least.getD d d) else b.least.getD r r := by
    intro r
    rw [least]
    split
    · rename_i less
      rw [Array.getD_eq_getD_getElem?, getElem?_set _ _ _ _ (by rw [h.least_size]; exact ls)]
      by_cases is_s : r = s
      · simp only [is_s, ite_true, Option.getD_some]
        exact (Nat.min_eq_right (Nat.le_of_lt less)).symm
      · simp only [is_s, ite_false]
        rw [← Array.getD_eq_getD_getElem?]
    · rename_i more
      by_cases is_s : r = s
      · simp only [is_s, ite_true]
        exact (Nat.min_eq_left (Nat.le_of_not_lt more)).symm
      · simp only [is_s, ite_false]
  have parents_get : ∀ r, (c.parents.modify s (· ++ b.parents.getD d [])).getD r []
      = if r = s then b.parents.getD s [] ++ b.parents.getD d []
        else if r = d then [] else b.parents.getD r [] := by
    intro r
    rw [parents]
    exact parents_linked b.parents ne (h.parents_size ▸ ls) (h.parents_size ▸ ld) r
  have key_of : ∀ (x : Nat) (node : Node), b.nodes[x]? = some node →
      b.keys[x]? = some (key ρ node)
        ∧ c.keys[x]? = some (key (fun x => swap d s (ρ x)) node) :=
    fun x node at_x => ⟨h.hashed.keys x node (fun f => f) at_x,
      hashed.keys x node (fun f => f) (nodes ▸ at_x)⟩
  refine ⟨hashed.reparent _, ?_, ?_, fun r lt is_root => ?_, fun x lt => ?_,
    fun r lt is_root empty => ?_, nodes ▸ h.wires, fun x k r at_x mem => ?_,
    fun r x mem => ?_⟩
  · show c.least.size = n
    rw [least]; split <;> simp [h.least_size]
  · show (c.parents.modify _ _).size = n
    rw [parents]; simp [h.parents_size]
  · have ⟨old, ne_d⟩ := root r is_root
    show swap d s (ρ (c.least.getD r r)) = r
    rw [least_get]
    split
    · rename_i is_s; subst is_s
      rcases Nat.le_total (b.least.getD r r) (b.least.getD d d) with le | le
      · rw [Nat.min_eq_left le, h.least_class r lt old, keep r ne_d]
      · rw [Nat.min_eq_right le, h.least_class d ld hd, swap_d]
    · rw [h.least_class r lt old, keep r ne_d]
  · show c.least.getD (swap d s (ρ x)) (swap d s (ρ x)) ≤ x
    have := h.least_le x lt
    rw [least_get]
    by_cases is_d : ρ x = d
    · rw [is_d] at this
      simp only [swap, is_d, ite_true]
      omega
    · rw [keep _ is_d]
      split
      · rename_i is_s; rw [is_s] at this; omega
      · exact this
  · have ⟨old, ne_d⟩ := root r is_root
    have empty : (c.parents.modify s (· ++ b.parents.getD d [])).getD r [] = [] := empty
    rw [parents_get] at empty
    show c.least.getD r r = r
    rw [least_get]
    by_cases is_s : r = s
    · rw [if_pos is_s] at empty
      rcases side with side | side
      · exact (side (List.append_eq_nil_iff.1 empty).1).elim
      · exact (side (List.append_eq_nil_iff.1 empty).2).elim
    · rw [if_neg is_s, if_neg ne_d] at empty
      rw [if_neg is_s]
      exact h.least_plain r lt old empty
  · show x ∈ (c.parents.modify s (· ++ b.parents.getD d [])).getD r []
    obtain ⟨node, at_x', rfl⟩ := hashed.node_of_key (fun f => f) at_x
    have ⟨old, _⟩ := key_of x node (nodes ▸ at_x')
    rw [key_swap_tail] at mem
    obtain ⟨r0, mem0, rfl⟩ := List.mem_map.1 mem
    have was := h.parents_mem x _ r0 old mem0
    rw [parents_get]
    by_cases is_d : r0 = d
    · rw [is_d] at was ⊢
      rw [swap_d, if_pos rfl]
      exact List.mem_append_right _ was
    · rw [keep r0 is_d]
      by_cases is_s : r0 = s
      · rw [if_pos is_s]; rw [is_s] at was; exact List.mem_append_left _ was
      · rw [if_neg is_s, if_neg is_d]; exact was
  · have mem : x ∈ (c.parents.modify s (· ++ b.parents.getD d [])).getD r [] := mem
    rw [parents_get] at mem
    have found : ∀ r0, x ∈ b.parents.getD r0 [] → swap d s r0 = r →
        ∃ k, c.keys[x]? = some k ∧ r ∈ k.tail := by
      intro r0 was eq
      obtain ⟨k, at_x, tail⟩ := h.parents_key r0 x was
      obtain ⟨node, at_node, rfl⟩ := h.hashed.node_of_key (fun f => f) at_x
      refine ⟨_, (key_of x node at_node).2, ?_⟩
      rw [key_swap_tail]
      exact List.mem_map.2 ⟨r0, tail, eq⟩
    by_cases is_s : r = s
    · rw [if_pos is_s] at mem
      rcases List.mem_append.1 mem with was | was
      · exact found s was (by rw [keep s ne, is_s])
      · exact found d was (by rw [swap_d, is_s])
    · rw [if_neg is_s] at mem
      by_cases is_d : r = d
      · rw [if_pos is_d] at mem; cases mem
      · rw [if_neg is_d] at mem; exact found r mem (keep r is_d)

end Builder

end EGraph
