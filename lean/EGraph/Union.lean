import EGraph.Link

/-!
# Union keeps the invariant

`Builder.union` keeps the invariant, relates the two spiders and no more,
and brings down what `rebuild` has left to do.
-/

namespace EGraph

theorem sum_range_set (f : Nat → Nat) (v : Nat) :
    ∀ (n a : Nat), a < n →
      ((List.range n).map fun r => if r = a then v else f r).sum + f a
        = ((List.range n).map f).sum + v := by
  intro n
  induction n with
  | zero => intro a lt; exact absurd lt (Nat.not_lt_zero _)
  | succ n ih =>
    intro a lt
    simp only [List.range_succ, List.map_append, List.sum_append, List.map_cons,
      List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
    rcases Nat.lt_succ_iff_lt_or_eq.1 lt with lt | rfl
    · have := ih a lt
      simp only [Nat.ne_of_gt lt, ite_false]
      omega
    · have : ((List.range a).map fun r => if r = a then v else f r)
          = (List.range a).map f :=
        List.map_congr_left fun r mem => by
          simp [Nat.ne_of_lt (List.mem_range.1 mem)]
      rw [this]; simp only [ite_true]; omega

namespace Builder

/-- How many classes there are. -/
def count (b : Builder) : Nat :=
  (List.range b.label.size).countP fun x => rep b.label x == x

/-- What `rebuild` has left to do, for at most `total` parents in all. -/
def todo (total : Nat) (b : Builder) : Nat :=
  b.worklist.length + b.count * (total + 1)

/-- The classes after a union: as they were, or a root `d` joined a root `s`,
the roots of the two spiders. -/
def Joined (ρ ρ' : Nat → Nat) (left right : Nat) : Prop :=
  (ρ left = ρ right ∧ ∀ x, ρ' x = ρ x)
    ∨ ∃ s d, s ≠ d ∧ ((s = ρ left ∧ d = ρ right) ∨ (s = ρ right ∧ d = ρ left))
      ∧ ∀ x, ρ' x = swap d s (ρ x)

theorem count_swap {b b' : Builder} {s d : Nat} {rank : Nat → Nat}
    (forest : Forest b.label rank) (hs : rep b.label s = s) (hd : rep b.label d = d)
    (ne : s ≠ d) (ld : d < b.label.size) (size : b'.label.size = b.label.size)
    (roots : ∀ x, rep b'.label x = swap d s (rep b.label x)) : b'.count + 1 ≤ b.count := by
  unfold count
  rw [size]
  refine countP_lt (a := d) (fun x _ is_root => ?_) (List.mem_range.2 ld) (by simpa using hd) ?_
  · simp only [beq_iff_eq, roots] at is_root ⊢
    unfold swap at is_root
    split at is_root
    · rename_i is_d; rw [← is_root, hs] at is_d; exact absurd is_d ne
    · exact is_root
  · simp only [roots, hd, swap, ite_true]
    simpa using ne

theorem link_spec {b : Builder} (h : Inv b) {s d : Nat} (hs : rep b.label s = s)
    (hd : rep b.label d = d) (ne : s ≠ d) (ls : s < b.label.size) (ld : d < b.label.size)
    (side : b.parents.getD s [] ≠ [] ∨ b.parents.getD d [] ≠ []) :
    Inv (b.link s d) ∧ (b.link s d).nodes = b.nodes
      ∧ (b.link s d).label.size = b.label.size
      ∧ (∀ x, rep (b.link s d).label x = swap d s (rep b.label x))
      ∧ (b.link s d).worklist.length ≤ b.worklist.length + (b.parents.getD d []).length
      ∧ (b.link s d).total = b.total := by
  obtain ⟨rank, forest⟩ := h.forest
  have t := h.tables
  have root_s : parent b.label s = s := by have := forest.rep_root s; rwa [hs] at this
  have root_d : parent b.label d = d := by have := forest.rep_root d; rwa [hd] at this
  obtain ⟨rank', forest', roots'⟩ := forest.link root_s root_d ne ls ld
  have moving := t.moving d s (b.label.setIfInBounds d s)
    (if b.least.getD d d < b.least.getD s s
      then b.least.setIfInBounds s (b.least.getD d d) else b.least)
    (b.parents.setIfInBounds d [])
  have ⟨m, stale, nodes, least, parents, forest'', size, roots'', work⟩ :=
    Moving.rekeys (rank := rank') ne hs forest.rep_rep (b.parents.getD d []) _ moving forest'
      (fun x => by rw [roots' x]; rfl) fun box mem => by
        obtain ⟨k, at_box, tail⟩ := t.parents_key d box mem
        obtain ⟨node, at_node, rfl⟩ := t.hashed.node_of_key (fun f => f) at_box
        refine ⟨node, at_node, ?_⟩
        cases det : node.det
        · simp [key, det] at tail
        · rfl
  have tables := t.linked hs hd ne ls ld side _
    (m.hashed.stale_iff fun x => ⟨False.elim, fun is_stale =>
      let ⟨⟨k, at_x, tail⟩, not_moved⟩ := stale x is_stale
      not_moved (t.parents_mem x k d at_x tail)⟩) nodes least parents
  have size' : (b.link s d).label.size = b.label.size := by
    show (List.foldl Builder.rekey _ (b.parents.getD d [])).label.size = _
    rw [size]; simp
  have roots''' : ∀ x, rep (b.link s d).label x = swap d s (rep b.label x) := roots''
  refine ⟨⟨⟨rank', forest''⟩, ?_⟩, nodes, size', roots''', ?_, ?_⟩
  · show Tables (b.link s d).label.size (rep (b.link s d).label) _
    rw [size', funext roots''']
    exact tables
  · exact work
  · unfold total
    rw [size']
    have formula : ∀ r, ((b.link s d).parents.getD r []).length
        = if r = s then (b.parents.getD s []).length + (b.parents.getD d []).length
          else if r = d then 0 else (b.parents.getD r []).length := by
      intro r
      show ((Array.modify _ s (· ++ b.parents.getD d [])).getD r []).length = _
      rw [parents, parents_linked b.parents ne (t.parents_size ▸ ls) (t.parents_size ▸ ld)]
      split
      · simp
      · split <;> rfl
    rw [List.map_congr_left fun r _ => formula r]
    have first := sum_range_set
      (fun r => if r = d then 0 else (b.parents.getD r []).length)
      ((b.parents.getD s []).length + (b.parents.getD d []).length) b.label.size s ls
    have second := sum_range_set (fun r => (b.parents.getD r []).length) 0 b.label.size d ld
    simp only [ne, ite_false] at first
    omega

/-- Joining two roots keeps the invariant. -/
theorem join_spec {b : Builder} (h : Inv b) {left right : Nat}
    (hl : rep b.label left = left) (hr : rep b.label right = right)
    (ll : left < b.label.size) (lr : right < b.label.size) :
    Inv (b.join left right) ∧ (b.join left right).nodes = b.nodes
      ∧ (b.join left right).label.size = b.label.size
      ∧ Joined (rep b.label) (rep (b.join left right).label) left right
      ∧ (b.join left right).total = b.total
      ∧ (b.join left right).todo b.total ≤ b.todo b.total := by
  obtain ⟨rank, forest⟩ := h.forest
  simp only [join]
  split
  · rename_i same
    exact ⟨h, rfl, rfl, .inl ⟨by rw [hl, hr, same], fun _ => rfl⟩, rfl, Nat.le_refl _⟩
  · rename_i ne
    have joined : ∀ {s d : Nat} {ρ' : Nat → Nat}, (s = min left right ∧ d = max left right)
        ∨ (s = max left right ∧ d = min left right) → (∀ x, ρ' x = swap d s (rep b.label x)) →
        Joined (rep b.label) ρ' left right := by
      intro s d ρ' which roots
      refine .inr ⟨s, d, ?_, ?_, roots⟩
      · rcases which with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> omega
      · rw [hl, hr]
        rcases Nat.le_total left right with le | le
        · rw [Nat.min_eq_left le, Nat.max_eq_right le] at which
          rcases which with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
          · exact .inl ⟨rfl, rfl⟩
          · exact .inr ⟨rfl, rfl⟩
        · rw [Nat.min_eq_right le, Nat.max_eq_left le] at which
          rcases which with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
          · exact .inr ⟨rfl, rfl⟩
          · exact .inl ⟨rfl, rfl⟩
    have root_min : rep b.label (min left right) = min left right := by
      rcases Nat.le_total left right with le | le
      · rw [Nat.min_eq_left le]; exact hl
      · rw [Nat.min_eq_right le]; exact hr
    have root_max : rep b.label (max left right) = max left right := by
      rcases Nat.le_total left right with le | le
      · rw [Nat.max_eq_right le]; exact hr
      · rw [Nat.max_eq_left le]; exact hl
    have lt_min : min left right < b.label.size := by omega
    have lt_max : max left right < b.label.size := by omega
    have lt : min left right < max left right := by omega
    have less : ∀ {b' : Builder} {s d : Nat}, rep b.label s = s → rep b.label d = d → s ≠ d →
        d < b.label.size → b'.label.size = b.label.size →
        (∀ x, rep b'.label x = swap d s (rep b.label x)) →
        b'.worklist.length ≤ b.worklist.length + b.total →
        b'.todo b.total ≤ b.todo b.total := by
      intro b' s d hs hd ne ld size roots work
      have := count_swap forest hs hd ne ld size roots
      have := Nat.mul_le_mul_right (b.total + 1) this
      rw [Nat.add_mul, Nat.one_mul] at this
      unfold todo
      omega
    split
    · rename_i plain
      have root_min' : parent b.label (min left right) = min left right := by
        have := forest.rep_root (min left right); rwa [root_min] at this
      have root_max' : parent b.label (max left right) = max left right := by
        have := forest.rep_root (max left right); rwa [root_max] at this
      obtain ⟨rank', forest', roots'⟩ :=
        forest.link root_min' root_max' (Nat.ne_of_lt lt) lt_min lt_max
      have roots'' : ∀ x, rep (b.label.setIfInBounds (max left right) (min left right)) x
          = swap (max left right) (min left right) (rep b.label x) :=
        fun x => by rw [roots' x]; rfl
      refine ⟨⟨⟨rank', forest'⟩, ?_⟩, rfl, by simp, joined (.inl ⟨rfl, rfl⟩) roots'', ?_, ?_⟩
      · show Tables (b.label.setIfInBounds _ _).size (rep (b.label.setIfInBounds _ _)) _
        rw [funext roots'']
        simp only [Array.size_setIfInBounds]
        exact (h.tables.link_plain root_min root_max lt lt_max plain).relabel _
      · simp [total]
      · exact less root_min root_max (Nat.ne_of_lt lt) lt_max (by simp) roots''
          (Nat.le_add_right _ _)
    · rename_i busy
      have length_le : ∀ r, r < b.label.size → (b.parents.getD r []).length ≤ b.total := by
        intro r lt
        have := sum_range_set (fun r => (b.parents.getD r []).length) 0 b.label.size r lt
        unfold total; omega
      by_cases fewer : (b.parents.getD (min left right) []).length
          < (b.parents.getD (max left right) []).length
      · simp only [fewer, decide_true, ite_true]
        have ⟨inv, nodes, size, roots, work, total⟩ := link_spec h root_max root_min
          (Ne.symm (Nat.ne_of_lt lt)) lt_max lt_min (.inl busy)
        exact ⟨inv, nodes, size, joined (.inr ⟨rfl, rfl⟩) roots, total,
          less root_max root_min (Ne.symm (Nat.ne_of_lt lt)) lt_min size roots
            (Nat.le_trans work (Nat.add_le_add_left (length_le _ lt_min) _))⟩
      · simp only [fewer, decide_false, Bool.false_eq_true, ite_false]
        have ⟨inv, nodes, size, roots, work, total⟩ := link_spec h root_min root_max
          (Nat.ne_of_lt lt) lt_min lt_max (.inr busy)
        exact ⟨inv, nodes, size, joined (.inl ⟨rfl, rfl⟩) roots, total,
          less root_min root_max (Nat.ne_of_lt lt) lt_max size roots
            (Nat.le_trans work (Nat.add_le_add_left (length_le _ lt_max) _))⟩


namespace Joined

theorem extensive {ρ ρ' : Nat → Nat} {left right : Nat} (h : Joined ρ ρ' left right)
    {x y : Nat} (eq : ρ x = ρ y) : ρ' x = ρ' y := by
  rcases h with ⟨_, same⟩ | ⟨s, d, _, _, roots⟩
  · rw [same, same, eq]
  · rw [roots, roots, eq]

theorem related {ρ ρ' : Nat → Nat} {left right : Nat} (h : Joined ρ ρ' left right) :
    ρ' left = ρ' right := by
  rcases h with ⟨eq, same⟩ | ⟨s, d, ne, which, roots⟩
  · rw [same, same, eq]
  · rw [roots, roots]
    rcases which with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> simp [swap, ne]

theorem least {ρ ρ' : Nat → Nat} {left right : Nat} (h : Joined ρ ρ' left right)
    {r : Nat → Nat → Prop} (eqv : Equivalence r) (sub : ∀ x y, ρ x = ρ y → r x y)
    (rel : r left right) {x y : Nat} (eq : ρ' x = ρ' y) : r x y := by
  rcases h with ⟨_, same⟩ | ⟨s, d, ne, which, roots⟩
  · rw [same, same] at eq; exact sub x y eq
  · rw [roots, roots] at eq
    unfold swap at eq
    have cross : ∀ a c, ρ a = d → ρ c = s → r a c := by
      intro a c ha hc
      rcases which with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
      · exact eqv.trans (sub a right ha) (eqv.trans (eqv.symm rel) (eqv.symm (sub c left hc)))
      · exact eqv.trans (sub a left ha) (eqv.trans rel (eqv.symm (sub c right hc)))
    by_cases hx : ρ x = d <;> by_cases hy : ρ y = d
    · exact sub x y (hx.trans hy.symm)
    · simp only [hx, hy, ite_true, ite_false] at eq
      exact cross x y hx eq.symm
    · simp only [hx, hy, ite_true, ite_false] at eq
      exact eqv.symm (cross y x hy eq)
    · simp only [hx, hy, ite_false] at eq
      exact sub x y eq

end Joined

/-- `union` keeps the invariant and relates its two spiders. -/
theorem union_spec {b : Builder} (h : Inv b) {left right : Nat}
    (ll : left < b.label.size) (lr : right < b.label.size) :
    Inv (b.union left right) ∧ (b.union left right).nodes = b.nodes
      ∧ (b.union left right).label.size = b.label.size
      ∧ Joined (rep b.label) (rep (b.union left right).label) left right
      ∧ (b.union left right).total = b.total
      ∧ (b.union left right).todo b.total ≤ b.todo b.total := by
  obtain ⟨rank, forest⟩ := h.forest
  have ⟨value, label, eq, forest', size, same⟩ := b.root_spec forest left
  have ⟨value', label', eq', forest'', size', same'⟩ :=
    Builder.root_spec (rank := rank) (b.root left).1 (by rw [eq]; exact forest') right
  rw [eq] at value' eq' size' same'
  have same2 : ∀ x, rep label' x = rep b.label x := fun x => (same' x).trans (same x)
  have size2 : label'.size = b.label.size := size'.trans size
  have inv : Inv { b with label := label' } := h.relabel forest'' size2 same2
  have roots : rep label' = rep b.label := funext same2
  have ⟨inv', nodes, size3, joined, total, todo⟩ := join_spec inv
    (left := rep b.label left) (right := rep b.label right)
    (by show rep label' _ = _; rw [roots]; exact forest.rep_rep left)
    (by show rep label' _ = _; rw [roots]; exact forest.rep_rep right)
    (by show _ < label'.size; rw [size2]; exact forest.rep_lt ll)
    (by show _ < label'.size; rw [size2]; exact forest.rep_lt lr)
  have unfolded : b.union left right
      = Builder.join { b with label := label' } (rep b.label left) (rep b.label right) := by
    show Builder.join ((b.root left).1.root right).1 (b.root left).2
      ((b.root left).1.root right).2 = _
    rw [value, eq, eq', value']
    show Builder.join { b with label := label' } (rep b.label left) (rep label right) = _
    rw [same right]
  have same_total : Builder.total { b with label := label' } = b.total := by
    simp only [Builder.total, size2]
  have same_todo : Builder.todo b.total { b with label := label' } = b.todo b.total := by
    simp only [Builder.todo, Builder.count, size2, roots]
  rw [unfolded]
  refine ⟨inv', nodes, size3.trans size2, ?_, total.trans same_total, ?_⟩
  · rw [show rep ({ b with label := label' } : Builder).label = rep b.label from roots]
      at joined
    rcases joined with ⟨eq, same⟩ | ⟨s, d, ne, which, roots⟩
    · exact .inl ⟨by rwa [forest.rep_rep, forest.rep_rep] at eq, same⟩
    · exact .inr ⟨s, d, ne, by rwa [forest.rep_rep, forest.rep_rep] at which, roots⟩
  · rw [same_total] at todo; rw [← same_todo]; exact todo

/-- Unions in turn keep the invariant and relate what the pairs generate. -/
theorem unions_spec :
    ∀ (pairs : List (Nat × Nat)) (b : Builder), Inv b →
      (∀ p ∈ pairs, p.1 < b.label.size ∧ p.2 < b.label.size) →
      Inv (pairs.foldl (fun b p => b.union p.1 p.2) b)
        ∧ (pairs.foldl (fun b p => b.union p.1 p.2) b).nodes = b.nodes
        ∧ (pairs.foldl (fun b p => b.union p.1 p.2) b).label.size = b.label.size
        ∧ (pairs.foldl (fun b p => b.union p.1 p.2) b).total = b.total
        ∧ (pairs.foldl (fun b p => b.union p.1 p.2) b).todo b.total ≤ b.todo b.total
        ∧ (∀ x y, rep b.label x = rep b.label y →
          rep (pairs.foldl (fun b p => b.union p.1 p.2) b).label x
            = rep (pairs.foldl (fun b p => b.union p.1 p.2) b).label y)
        ∧ (∀ p ∈ pairs, rep (pairs.foldl (fun b p => b.union p.1 p.2) b).label p.1
          = rep (pairs.foldl (fun b p => b.union p.1 p.2) b).label p.2)
        ∧ ∀ r : Nat → Nat → Prop, Equivalence r →
          (∀ x y, rep b.label x = rep b.label y → r x y) → (∀ p ∈ pairs, r p.1 p.2) →
          ∀ x y, rep (pairs.foldl (fun b p => b.union p.1 p.2) b).label x
            = rep (pairs.foldl (fun b p => b.union p.1 p.2) b).label y → r x y := by
  intro pairs
  induction pairs with
  | nil =>
    intro b h _
    exact ⟨h, rfl, rfl, rfl, Nat.le_refl _, fun _ _ eq => eq,
      fun _ mem => absurd mem List.not_mem_nil,
      fun _ _ sub _ x y eq => sub x y eq⟩
  | cons pair pairs ih =>
    intro b h bound
    have ⟨lt, lt'⟩ := bound pair (List.mem_cons_self ..)
    have ⟨inv, nodes, size, joined, total, todo⟩ := union_spec h lt lt'
    have ⟨inv', nodes', size', total', todo', extensive, related, least⟩ :=
      ih (b.union pair.1 pair.2) inv fun p mem => by
        rw [size]; exact bound p (List.mem_cons_of_mem _ mem)
    simp only [List.foldl_cons]
    refine ⟨inv', nodes'.trans nodes, size'.trans size, total'.trans total, ?_,
      fun x y eq => extensive x y (joined.extensive eq), fun p mem => ?_,
      fun r eqv sub rel x y eq => ?_⟩
    · rw [total] at todo'; exact Nat.le_trans todo' todo
    · rcases List.mem_cons.1 mem with rfl | mem
      · exact extensive _ _ joined.related
      · exact related p mem
    · exact least r eqv
        (fun x y eq => joined.least eqv sub (rel pair (List.mem_cons_self ..)) eq)
        (fun p mem => rel p (List.mem_cons_of_mem _ mem)) x y eq

end Builder

end EGraph
