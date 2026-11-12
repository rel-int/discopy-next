import EGraph.Freeze

/-!
# Adding a box

`Builder.add` keeps the invariant, and is `add` on the e-graph the builder
freezes into: the hash-cons holds the first node of each key, which is the
node the e-graph finds by going through them all.
-/

namespace EGraph

theorem find?_least {α : Type} (p : α → Bool) :
    ∀ (l : List α) (first : Nat) (x : α), l[first]? = some x → p x = true →
      (∀ (i : Nat) (y : α), l[i]? = some y → p y = true → first ≤ i) →
      l.find? p = some x := by
  intro l
  induction l with
  | nil => intro first x at_first; simp at at_first
  | cons y ys ih =>
    intro first x at_first holds least
    cases first with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at at_first
      subst at_first
      simp [holds]
    | succ first =>
      have not_y : p y = false := by
        cases py : p y
        · rfl
        · exact absurd (least 0 y rfl py) (Nat.not_succ_le_zero _)
      rw [List.find?_cons_of_neg (by simp [not_y])]
      exact ih first x (by simpa using at_first) holds fun i z at_i pz => by
        have := least (i + 1) z (by simpa using at_i) pz
        omega

theorem map_of_pointwise {f : Nat → Nat} {xs ys : List Nat}
    (h : Pointwise (fun i j => f i = f j) xs ys) : xs.map f = ys.map f := by
  have ⟨length, at_i⟩ := pointwise_iff.1 h
  apply List.ext_getElem?
  intro i
  simp only [List.getElem?_map]
  cases hx : xs[i]? with
  | none =>
    have : ys[i]? = none := by rw [List.getElem?_eq_none_iff] at hx ⊢; omega
    rw [this]
  | some x =>
    have lt := (List.getElem?_eq_some_iff.1 hx).1
    rw [List.getElem?_eq_getElem (length ▸ lt)]
    exact congrArg some (at_i i x _ hx (List.getElem?_eq_getElem (length ▸ lt)))

namespace Builder

theorem place_label (b : Builder) (node : Node) (k : List Nat) :
    (b.place node k).label = b.label := by
  by_cases empty : k = []
  · subst empty; rw [place_sample]
  · rw [place_det _ _ _ empty]; exact cons_label _ _

theorem place_least (b : Builder) (node : Node) (k : List Nat) :
    (b.place node k).least = b.least := by
  by_cases empty : k = []
  · subst empty; rw [place_sample]
  · rw [place_det _ _ _ empty]; exact cons_least _ _

theorem place_nodes (b : Builder) (node : Node) (k : List Nat) :
    (b.place node k).nodes = b.nodes.push node := by
  by_cases empty : k = []
  · subst empty; rw [place_sample]
  · rw [place_det _ _ _ empty]; exact cons_nodes _ _

/-- New spiders, each its own class with no parent, keep the tables. -/
theorem Tables.extend {n : Nat} {ρ : Nat → Nat} {b : Builder} (h : Tables n ρ b)
    (k : Nat) (label : Array Nat) (fix : ∀ r, n ≤ r → ρ r = r) :
    Tables (n + k) ρ { b with
      label := label
      least := b.least ++ (List.range' n k).toArray
      parents := b.parents ++ Array.replicate k [] } := by
  have least_get : ∀ r, (b.least ++ (List.range' n k).toArray).getD r r = b.least.getD r r := by
    intro r
    have := parent_extend b.least k r
    rw [h.least_size] at this
    exact this
  have parents_get : ∀ r, (b.parents ++ Array.replicate k ([] : List Nat)).getD r []
      = b.parents.getD r [] := by
    intro r
    simp only [Array.getD_eq_getD_getElem?, Array.getElem?_append]
    split
    · rfl
    · rename_i ge
      rw [Array.getElem?_eq_none (Nat.le_of_not_lt ge)]
      cases found : (Array.replicate k ([] : List Nat))[r - b.parents.size]? with
      | none => rfl
      | some v =>
        obtain ⟨_, eq⟩ := Array.getElem?_eq_some_iff.1 found
        simp at eq
        simp [eq]
  have beyond : ∀ r, n ≤ r → b.least.getD r r = r := by
    intro r ge
    simp [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (h.least_size ▸ ge)]
  have hashed := h.hashed
  refine ⟨⟨hashed.1, hashed.2, hashed.3, hashed.4, hashed.5, hashed.6, hashed.7⟩,
    by simp [h.least_size], by simp [h.parents_size], fun r lt is_root => ?_, fun x lt => ?_,
    fun r lt is_root empty => ?_, fun node mem => ?_, fun box key r at_box mem => ?_,
    fun r box mem => ?_⟩
  · show ρ ((b.least ++ (List.range' n k).toArray).getD r r) = r
    rw [least_get]
    by_cases old : r < n
    · exact h.least_class r old is_root
    · rw [beyond r (Nat.le_of_not_lt old)]; exact is_root
  · show (b.least ++ (List.range' n k).toArray).getD (ρ x) (ρ x) ≤ x
    rw [least_get]
    by_cases old : x < n
    · exact h.least_le x old
    · rw [fix x (Nat.le_of_not_lt old), beyond x (Nat.le_of_not_lt old)]
      exact Nat.le_refl x
  · show (b.least ++ (List.range' n k).toArray).getD r r = r
    have empty : (b.parents ++ Array.replicate k ([] : List Nat)).getD r [] = [] := empty
    rw [least_get]
    rw [parents_get] at empty
    by_cases old : r < n
    · exact h.least_plain r old is_root empty
    · exact beyond r (Nat.le_of_not_lt old)
  · have ⟨inputs, outputs⟩ := h.wires node mem
    exact ⟨fun s s_mem => Nat.lt_add_right k (inputs s s_mem),
      fun s s_mem => Nat.lt_add_right k (outputs s s_mem)⟩
  · show box ∈ (b.parents ++ Array.replicate k ([] : List Nat)).getD r []
    rw [parents_get]; exact h.parents_mem box key r at_box mem
  · have mem : box ∈ (b.parents ++ Array.replicate k ([] : List Nat)).getD r [] := mem
    rw [parents_get] at mem; exact h.parents_key r box mem

/-- A builder with halved paths freezes into the same e-graph. -/
theorem freeze_relabel {b : Builder} {rank : Nat → Nat} (forest : Forest b.label rank)
    {label : Array Nat} (forest' : Forest label rank) (size : label.size = b.label.size)
    (same : ∀ x, rep label x = rep b.label x) :
    Builder.freeze { b with label := label } = b.freeze := by
  refine Graph.ext rfl ?_
  show Builder.classes { b with label := label } = b.classes
  rw [classes_eq forest, classes_eq (b := { b with label := label }) forest']
  simp only [size, funext same]

/-- Appending a node on new spiders. -/
theorem append_spec {b : Builder} (h : Inv b) (head : Nat) (det : Bool) (arity : Nat)
    (inputs : List Nat) (lt : ∀ s ∈ inputs, s < b.label.size)
    (ar : ∀ node ∈ b.nodes.toList, node.head = head → node.outputs.length = arity) :
    Inv (Builder.place { b with
        label := b.label ++ (List.range' b.label.size arity).toArray
        least := b.least ++ (List.range' b.label.size arity).toArray
        parents := b.parents ++ Array.replicate arity [] }
        ⟨head, det, inputs, List.range' b.label.size arity⟩
        (key (rep b.label) ⟨head, det, inputs, List.range' b.label.size arity⟩))
      ∧ (Builder.place { b with
        label := b.label ++ (List.range' b.label.size arity).toArray
        least := b.least ++ (List.range' b.label.size arity).toArray
        parents := b.parents ++ Array.replicate arity [] }
        ⟨head, det, inputs, List.range' b.label.size arity⟩
        (key (rep b.label) ⟨head, det, inputs, List.range' b.label.size arity⟩)).freeze
      = ⟨b.freeze.nodes ++ [⟨head, det, inputs, List.range' b.label.size arity⟩],
        b.freeze.classes ++ List.range' b.label.size arity⟩ := by
  obtain ⟨rank, forest⟩ := h.forest
  have ⟨forest', same⟩ := forest.extend arity
  have extended := h.tables.extend arity (b.label ++ (List.range' b.label.size arity).toArray)
    fun r ge => forest.rep_of_ge ge
  have placed := extended.place ⟨head, det, inputs, List.range' b.label.size arity⟩
    (fun s mem => Nat.lt_add_right arity (lt s mem))
    (fun s mem => by have := List.mem_range'_1.1 mem; omega)
    (fun s mem => Nat.lt_add_right arity (forest.rep_lt (lt s mem)))
    (fun a mem eq => by simpa using ar a mem eq)
  have least_get : ∀ r, (b.least ++ (List.range' b.label.size arity).toArray).getD r r
      = b.least.getD r r := by
    intro r
    have := parent_extend b.least arity r
    rw [h.tables.least_size] at this
    exact this
  refine ⟨⟨⟨rank, by rw [place_label]; exact forest'⟩, ?_⟩, Graph.ext ?_ ?_⟩
  · rw [place_label]
    show Tables (b.label ++ (List.range' b.label.size arity).toArray).size
      (rep (b.label ++ (List.range' b.label.size arity).toArray)) _
    rw [funext same]
    simpa using placed
  · show (Builder.place _ _ _).nodes.toList = _
    rw [place_nodes]; simp [Builder.freeze]
  · show Builder.classes _ = b.classes ++ _
    rw [classes_eq (rank := rank) (by rw [place_label]; exact forest'), classes_eq forest,
      place_label, place_least]
    simp only [funext same, least_get]
    have beyond : ∀ j, b.least.getD (b.label.size + j) (b.label.size + j)
        = b.label.size + j := by
      intro j
      simp [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none
        (show b.least.size ≤ b.label.size + j by rw [h.tables.least_size]; omega)]
    simp only [Array.size_append, List.size_toArray, List.length_range', List.range_add,
      List.map_append, List.map_map, List.range'_eq_map_range, List.length_map,
      List.length_range]
    congr 1
    refine List.map_congr_left fun j _ => ?_
    simp only [Function.comp_apply]
    rw [forest.rep_of_ge (Nat.le_add_right _ _), beyond]

end Builder

end EGraph
