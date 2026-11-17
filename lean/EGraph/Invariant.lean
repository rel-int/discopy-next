import EGraph.Builder

/-!
# The invariant of the builder

What holds of a builder between two of its methods, in terms of the root of
each spider: the tables say of the roots what the e-graph it freezes into
says of the classes.
-/

namespace EGraph

namespace Builder

/-- The key of a node: its head then the roots of its inputs when it is
deterministic, nothing for a sample. -/
def key (ρ : Nat → Nat) (node : Node) : List Nat :=
  if node.det then node.head :: node.inputs.map ρ else []

/-- What the classes and the pairs of the worklist generate: two spiders
related by every equivalence containing the classes and relating the outputs
of each pair. -/
def Pending (ρ : Nat → Nat) (b : Builder) (i j : Nat) : Prop :=
  ∀ r : Nat → Nat → Prop, Equivalence r → (∀ i j, ρ i = ρ j → r i j) →
    (∀ p ∈ b.worklist, ∀ q ∈ (b.outputs p.1).zip (b.outputs p.2), r q.1 q.2) → r i j

/-- The invariant of the hash-cons, but for the stale nodes. -/
structure Hashed (ρ : Nat → Nat) (stale : Nat → Prop) (b : Builder) : Prop where
  keys_size : b.keys.size = b.nodes.size
  arity : ∀ a ∈ b.nodes.toList, ∀ c ∈ b.nodes.toList, a.head = c.head →
    a.outputs.length = c.outputs.length
  keys : ∀ (box : Nat) (node : Node), ¬ stale box → b.nodes[box]? = some node →
    b.keys[box]? = some (key ρ node)
  hash_some : ∀ (k : List Nat) (first : Nat), b.hashcons[k]? = some first →
    k ≠ [] ∧ ¬ stale first ∧ b.keys[first]? = some k
      ∧ ∀ box : Nat, ¬ stale box → b.keys[box]? = some k → first ≤ box
  hash_none : ∀ (box : Nat) (k : List Nat), ¬ stale box → b.keys[box]? = some k →
    k ≠ [] → b.hashcons[k]? ≠ none
  work : ∀ p ∈ b.worklist, ∃ a c, b.nodes[p.1]? = some a ∧ b.nodes[p.2]? = some c
    ∧ a.det = true ∧ c.det = true ∧ a.head = c.head
    ∧ a.inputs.map ρ = c.inputs.map ρ
  linked : ∀ (box : Nat) (k : List Nat) (first : Nat), ¬ stale box →
    b.keys[box]? = some k → b.hashcons[k]? = some first →
    Pointwise (Pending ρ b) (b.outputs first) (b.outputs box)

/-- The invariant of the tables, for `n` spiders with roots `ρ`. -/
structure Tables (n : Nat) (ρ : Nat → Nat) (b : Builder) : Prop where
  hashed : Hashed ρ (fun _ => False) b
  least_size : b.least.size = n
  parents_size : b.parents.size = n
  least_class : ∀ r : Nat, r < n → ρ r = r → ρ (b.least.getD r r) = r
  least_le : ∀ x : Nat, x < n → b.least.getD (ρ x) (ρ x) ≤ x
  least_plain : ∀ r : Nat, r < n → ρ r = r → b.parents.getD r [] = [] → b.least.getD r r = r
  wires : ∀ node ∈ b.nodes.toList,
    (∀ s ∈ node.inputs, s < n) ∧ ∀ s ∈ node.outputs, s < n
  parents_mem : ∀ (box : Nat) (k : List Nat) (r : Nat), b.keys[box]? = some k → r ∈ k.tail →
    box ∈ b.parents.getD r []
  parents_key : ∀ r box : Nat, box ∈ b.parents.getD r [] →
    ∃ k, b.keys[box]? = some k ∧ r ∈ k.tail

/-- The invariant of a builder: its labels are a forest and its tables say of
the roots what they should. -/
structure Inv (b : Builder) : Prop where
  forest : ∃ rank, Forest b.label rank
  tables : Tables b.label.size (rep b.label) b

/-- The tables do not read the labels. -/
theorem Hashed.relabel {ρ : Nat → Nat} {stale : Nat → Prop} {b : Builder}
    (h : Hashed ρ stale b) (label : Array Nat) :
    Hashed ρ stale { b with label := label } :=
  ⟨h.1, h.2, h.3, h.4, h.5, h.6, h.7⟩

theorem Tables.relabel {n : Nat} {ρ : Nat → Nat} {b : Builder} (h : Tables n ρ b)
    (label : Array Nat) : Tables n ρ { b with label := label } :=
  ⟨h.1.relabel label, h.2, h.3, h.4, h.5, h.6, h.7, h.8, h.9⟩

/-- Labels with the same roots keep the invariant. -/
theorem Inv.relabel {b : Builder} (h : Inv b) {label : Array Nat} {rank : Nat → Nat}
    (forest : Forest label rank) (size : label.size = b.label.size)
    (same : ∀ x, rep label x = rep b.label x) : Inv { b with label := label } := by
  refine ⟨⟨rank, forest⟩, ?_⟩
  have : rep label = rep b.label := funext same
  show Tables label.size (rep label) _
  rw [size, this]
  exact h.tables.relabel label

/-! ## Roots -/

theorem root_spec (b : Builder) {rank : Nat → Nat} (forest : Forest b.label rank)
    (spider : Nat) :
    (b.root spider).2 = rep b.label spider
      ∧ ∃ label, (b.root spider).1 = { b with label := label } ∧ Forest label rank
        ∧ label.size = b.label.size ∧ ∀ x, rep label x = rep b.label x := by
  have ⟨_, forest', size, same⟩ :=
    forest.root_spec b.label.size b.label spider (above_le ..)
  exact ⟨rfl, _, rfl, forest', size, fun x => forest.rep_congr forest' same x⟩

theorem roots_spec {rank : Nat → Nat} :
    ∀ (spiders : List Nat) (b : Builder), Forest b.label rank →
      (b.roots spiders).2 = spiders.map (rep b.label)
        ∧ ∃ label, (b.roots spiders).1 = { b with label := label } ∧ Forest label rank
          ∧ label.size = b.label.size ∧ ∀ x, rep label x = rep b.label x := by
  intro spiders
  induction spiders with
  | nil => exact fun b forest => ⟨rfl, b.label, rfl, forest, rfl, fun _ => rfl⟩
  | cons spider spiders ih =>
    intro b forest
    have ⟨value, label, eq, forest', size, same⟩ := b.root_spec forest spider
    have ⟨values, label', eq', forest'', size', same'⟩ :=
      ih (b.root spider).1 (by rw [eq]; exact forest')
    simp only [roots]
    rw [eq] at values eq' size' same'
    refine ⟨?_, label', ?_, forest'', size'.trans size, fun x => (same' x).trans (same x)⟩
    · rw [value, eq, values]
      simp only [List.map_cons, List.cons.injEq, true_and]
      exact List.map_congr_left fun x _ => same x
    · rw [eq, eq']

/-! ## Keys and pending pairs -/

theorem key_eq {ρ : Nat → Nat} {a c : Node} (eq : key ρ a = key ρ c)
    (ne : key ρ a ≠ []) :
    a.det = true ∧ c.det = true ∧ a.head = c.head
      ∧ a.inputs.map ρ = c.inputs.map ρ := by
  unfold key at eq ne
  cases ha : a.det <;> cases hc : c.det <;> simp_all

theorem key_tail {ρ : Nat → Nat} {node : Node} {r : Nat}
    (mem : r ∈ (key ρ node).tail) : ∃ s ∈ node.inputs, ρ s = r := by
  unfold key at mem
  split at mem
  · simpa using mem
  · cases mem

theorem outputs_push (b : Builder) (node : Node) (keys : Array (List Nat))
    {box : Nat} (lt : box < b.nodes.size) :
    outputs { b with nodes := b.nodes.push node, keys := keys } box = b.outputs box := by
  simp [outputs, Array.getElem?_push, Nat.ne_of_lt lt]

namespace Pending

theorem equivalence (ρ : Nat → Nat) (b : Builder) : Equivalence (Pending ρ b) :=
  ⟨fun _ r eqv _ _ => eqv.refl _, fun h r eqv c p => eqv.symm (h r eqv c p),
    fun h h' r eqv c p => eqv.trans (h r eqv c p) (h' r eqv c p)⟩

theorem of_class {ρ : Nat → Nat} {b : Builder} {i j : Nat} (eq : ρ i = ρ j) :
    Pending ρ b i j := fun _ _ classes _ => classes i j eq

theorem of_pair {ρ : Nat → Nat} {b : Builder} {p q : Nat × Nat}
    (mem : p ∈ b.worklist) (zip : q ∈ (b.outputs p.1).zip (b.outputs p.2)) :
    Pending ρ b q.1 q.2 := fun _ _ _ pairs => pairs p mem q zip

theorem mono {ρ ρ' : Nat → Nat} {b b' : Builder}
    (classes : ∀ i j, ρ i = ρ j → Pending ρ' b' i j)
    (pairs : ∀ p ∈ b.worklist, ∀ q ∈ (b.outputs p.1).zip (b.outputs p.2),
      Pending ρ' b' q.1 q.2) {i j : Nat} (h : Pending ρ b i j) : Pending ρ' b' i j :=
  h _ (equivalence ρ' b') classes pairs

theorem of_empty {ρ : Nat → Nat} {b : Builder} (empty : b.worklist = []) {i j : Nat}
    (h : Pending ρ b i j) : ρ i = ρ j :=
  h (fun i j => ρ i = ρ j) ⟨fun _ => rfl, Eq.symm, Eq.trans⟩ (fun _ _ eq => eq)
    (by rw [empty]; exact fun _ mem => nomatch mem)

end Pending

theorem Pointwise.refl' {r : Nat → Nat → Prop} (refl : ∀ i, r i i) (xs : List Nat) :
    Pointwise r xs xs := by
  refine ⟨rfl, fun p mem => ?_⟩
  have : p.1 = p.2 := by
    clear refl
    induction xs with
    | nil => cases mem
    | cons x xs ih =>
      simp only [List.zip_cons_cons, List.mem_cons] at mem
      rcases mem with rfl | mem
      · rfl
      · exact ih mem
  rw [this]; exact refl _

end Builder

end EGraph
