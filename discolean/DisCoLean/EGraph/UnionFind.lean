/-!
# The union-find of the builder

The model of `Builder.label` and `Builder.root`: an array sending each spider
to its parent, a root to itself, and the root of a spider found by halving
the path there. The loop of the Python runs on fuel here, the size of the
array, which is enough when the array is a forest.
-/

namespace EGraph

/-- The parent of a spider, itself when it is not there. -/
def parent (label : Array Nat) (i : Nat) : Nat := label.getD i i

/-- The root of a spider and the labels with the path there halved. -/
def root (label : Array Nat) : Nat → Nat → Array Nat × Nat
  | 0, spider => (label, spider)
  | fuel + 1, spider =>
    if parent label spider = spider then (label, spider)
    else
      let next := parent label (parent label spider)
      root (label.setIfInBounds spider next) fuel next

/-- The root of a spider. -/
def rep (label : Array Nat) (i : Nat) : Nat := (root label label.size i).2

/-- A spider leads to a root by its parents. -/
inductive Leads (label : Array Nat) : Nat → Nat → Prop
  | root {r} : parent label r = r → Leads label r r
  | step {i r} : parent label i ≠ i → Leads label (parent label i) r → Leads label i r

/-- Labels are a forest when a rank goes up from each spider to its parent. -/
structure Forest (label : Array Nat) (rank : Nat → Nat) : Prop where
  lt : ∀ i, i < label.size → parent label i < label.size
  up : ∀ i, parent label i ≠ i → rank i < rank (parent label i)

theorem parent_of_ge {label : Array Nat} {i : Nat} (ge : label.size ≤ i) :
    parent label i = i := by
  simp [parent, Array.getD, Nat.not_lt.2 ge]

theorem parent_set (label : Array Nat) (i v j : Nat) (lt : i < label.size) :
    parent (label.setIfInBounds i v) j = if j = i then v else parent label j := by
  simp only [parent, Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds]
  by_cases eq : i = j
  · subst eq; simp [lt]
  · simp [eq, Ne.symm eq]

namespace Leads

theorem is_root {label : Array Nat} {i r : Nat} (h : Leads label i r) :
    parent label r = r := by
  induction h with
  | root h => exact h
  | step _ _ ih => exact ih

theorem unique {label : Array Nat} {i r r' : Nat} (h : Leads label i r)
    (h' : Leads label i r') : r = r' := by
  induction h with
  | root h =>
    cases h' with
    | root => rfl
    | step ne => exact absurd h ne
  | step ne _ ih =>
    cases h' with
    | root h => exact absurd h ne
    | step _ h' => exact ih h'

end Leads

/-! ## Fuel -/

theorem countP_lt {l : List Nat} {p q : Nat → Bool}
    (sub : ∀ x ∈ l, p x = true → q x = true) {a : Nat} (mem : a ∈ l)
    (qa : q a = true) (pa : p a = false) : l.countP p < l.countP q := by
  induction l with
  | nil => cases mem
  | cons x xs ih =>
    have le := List.countP_mono_left (l := xs) fun x mem => sub x (List.mem_cons_of_mem _ mem)
    have head := sub x (List.mem_cons_self ..)
    have step : (if p x = true then 1 else 0) ≤ (if q x = true then 1 else 0) := by
      by_cases px : p x = true
      · simp [px, head px]
      · simp [px]
    simp only [List.countP_cons]
    rcases List.mem_cons.1 mem with rfl | mem
    · simp only [qa, pa]; simp; omega
    · have := ih (fun x mem => sub x (List.mem_cons_of_mem _ mem)) mem
      omega

/-- How many spiders rank above a spider: the steps to its root at most. -/
def above (rank : Nat → Nat) (n i : Nat) : Nat :=
  (List.range n).countP fun j => rank i < rank j

theorem above_le (rank : Nat → Nat) (n i : Nat) : above rank n i ≤ n := by
  have := List.countP_le_length (p := fun j => decide (rank i < rank j)) (l := List.range n)
  simpa [above] using this

theorem above_lt {rank : Nat → Nat} {n i j : Nat} (lt : j < n)
    (up : rank i < rank j) : above rank n j < above rank n i := by
  refine countP_lt (a := j) (fun x _ h => ?_) (List.mem_range.2 lt) (by simpa using up) (by simp)
  simp only [decide_eq_true_eq] at h ⊢
  omega

namespace Forest

/-- Every spider of a forest leads to a root. -/
theorem leads {label : Array Nat} {rank : Nat → Nat} (h : Forest label rank) (i : Nat) :
    ∃ r, Leads label i r := by
  generalize eq : above rank label.size i = k
  induction k using Nat.strongRecOn generalizing i with
  | _ k ih =>
    by_cases is_root : parent label i = i
    · exact ⟨i, .root is_root⟩
    · have lt : i < label.size := by
        rcases Nat.lt_or_ge i label.size with lt | ge
        · exact lt
        · exact absurd (parent_of_ge ge) is_root
      have ⟨r, leads⟩ := ih _ (eq ▸ above_lt (h.lt i lt) (h.up i is_root)) _ rfl
      exact ⟨r, .step is_root leads⟩

/-- Halving the path at a spider that is not a root. -/
theorem halve {label : Array Nat} {rank : Nat → Nat} (h : Forest label rank)
    {i : Nat} (lt : i < label.size) (ne : parent label i ≠ i) :
    Forest (label.setIfInBounds i (parent label (parent label i))) rank
      ∧ ∀ j r, Leads (label.setIfInBounds i (parent label (parent label i))) j r
        ↔ Leads label j r := by
  have up : rank i < rank (parent label (parent label i)) := by
    have := h.up i ne
    by_cases top : parent label (parent label i) = parent label i
    · rw [top]; exact this
    · exact Nat.lt_trans this (h.up _ top)
  have forest : Forest (label.setIfInBounds i (parent label (parent label i))) rank := by
    refine ⟨fun j lt' => ?_, fun j ne' => ?_⟩
    · rw [parent_set _ _ _ _ lt]
      simp only [Array.size_setIfInBounds] at lt' ⊢
      split
      · exact h.lt _ (h.lt i lt)
      · exact h.lt j lt'
    · rw [parent_set _ _ _ _ lt] at ne' ⊢
      split
      · rename_i eq; subst eq; exact up
      · rename_i neq; simp [neq] at ne'; exact h.up j ne'
  have forward : ∀ j r, Leads label j r →
      Leads (label.setIfInBounds i (parent label (parent label i))) j r := by
    intro j r leads
    induction leads with
    | @root r is_root =>
      refine .root ?_
      rw [parent_set _ _ _ _ lt]
      split
      · rename_i eq; subst eq; exact absurd is_root ne
      · exact is_root
    | @step j r ne' leads ih =>
      by_cases eq : j = i
      · subst eq
        have skip : parent (label.setIfInBounds j (parent label (parent label j))) j
            = parent label (parent label j) := by
          rw [parent_set _ _ _ _ lt]; simp
        have skip_ne : parent label (parent label j) ≠ j := by
          intro eq; rw [eq] at up; exact Nat.lt_irrefl _ up
        have same : parent (label.setIfInBounds j (parent label (parent label j)))
            (parent label j) = parent label (parent label j) := by
          rw [parent_set _ _ _ _ lt]; simp [ne']
        refine .step (by rw [skip]; exact skip_ne) ?_
        rw [skip]
        cases ih with
        | root is_root =>
          exact Eq.mpr (congrArg (fun x => Leads (label.setIfInBounds j
            (parent label (parent label j))) x (parent label j))
              (same.symm.trans is_root)) (.root is_root)
        | step _ leads => rw [same] at leads; exact leads
      · have same : parent (label.setIfInBounds i (parent label (parent label i))) j
            = parent label j := by
          rw [parent_set _ _ _ _ lt]; simp [eq]
        exact .step (by rw [same]; exact ne') (by rw [same]; exact ih)
  refine ⟨forest, fun j r => ⟨fun leads => ?_, forward j r⟩⟩
  have ⟨r', leads'⟩ := h.leads j
  exact (forward j r' leads').unique leads ▸ leads'

/-- `root` gives the root of a spider and keeps the forest and its roots. -/
theorem root_spec {rank : Nat → Nat} :
    ∀ (fuel : Nat) (label : Array Nat) (i : Nat), Forest label rank →
      above rank label.size i ≤ fuel →
      Leads label i (root label fuel i).2 ∧ Forest (root label fuel i).1 rank
        ∧ (root label fuel i).1.size = label.size
        ∧ ∀ j r, Leads (root label fuel i).1 j r ↔ Leads label j r := by
  intro fuel
  induction fuel with
  | zero =>
    intro label i h le
    have is_root : parent label i = i := by
      apply Classical.byContradiction; intro ne
      have lt : i < label.size := by
        rcases Nat.lt_or_ge i label.size with lt | ge
        · exact lt
        · exact absurd (parent_of_ge ge) ne
      have := above_lt (h.lt i lt) (h.up i ne)
      omega
    exact ⟨.root is_root, h, rfl, fun _ _ => Iff.rfl⟩
  | succ fuel ih =>
    intro label i h le
    unfold root
    split
    · rename_i is_root
      exact ⟨.root is_root, h, rfl, fun _ _ => Iff.rfl⟩
    · rename_i ne
      have lt : i < label.size := by
        rcases Nat.lt_or_ge i label.size with lt | ge
        · exact lt
        · exact absurd (parent_of_ge ge) ne
      have ⟨forest, same⟩ := h.halve lt ne
      have up : rank i < rank (parent label (parent label i)) := by
        have := h.up i ne
        by_cases top : parent label (parent label i) = parent label i
        · rw [top]; exact this
        · exact Nat.lt_trans this (h.up _ top)
      have := above_lt (h.lt _ (h.lt i lt)) up
      have ⟨leads, forest', size, same'⟩ := ih _ (parent label (parent label i)) forest
        (by simp only [Array.size_setIfInBounds]; omega)
      refine ⟨?_, forest', by simpa using size, fun j r => (same' j r).trans (same j r)⟩
      refine (same _ _).1 (.step ?_ ?_)
      · rw [parent_set _ _ _ _ lt]; simp
        intro eq; rw [eq] at up; exact Nat.lt_irrefl _ up
      · rw [parent_set _ _ _ _ lt]; simp; exact leads

theorem leads_rep {label : Array Nat} {rank : Nat → Nat} (h : Forest label rank) (i : Nat) :
    Leads label i (rep label i) :=
  (root_spec _ label i h (above_le ..)).1

theorem rep_eq {label : Array Nat} {rank : Nat → Nat} (h : Forest label rank)
    {i r : Nat} (leads : Leads label i r) : rep label i = r :=
  (h.leads_rep i).unique leads

theorem rep_root {label : Array Nat} {rank : Nat → Nat} (h : Forest label rank)
    (i : Nat) : parent label (rep label i) = rep label i :=
  (h.leads_rep i).is_root

theorem rep_of_root {label : Array Nat} {rank : Nat → Nat} (h : Forest label rank)
    {r : Nat} (is_root : parent label r = r) : rep label r = r :=
  h.rep_eq (.root is_root)

theorem rep_rep {label : Array Nat} {rank : Nat → Nat} (h : Forest label rank)
    (i : Nat) : rep label (rep label i) = rep label i :=
  h.rep_of_root (h.rep_root i)

theorem leads_lt {label : Array Nat} {rank : Nat → Nat} (h : Forest label rank)
    {i r : Nat} (leads : Leads label i r) (lt : i < label.size) : r < label.size := by
  induction leads with
  | root => exact lt
  | step _ _ ih => exact ih (h.lt _ lt)

theorem rep_lt {label : Array Nat} {rank : Nat → Nat} (h : Forest label rank)
    {i : Nat} (lt : i < label.size) : rep label i < label.size :=
  h.leads_lt (h.leads_rep i) lt

theorem rep_of_ge {label : Array Nat} {rank : Nat → Nat} (h : Forest label rank)
    {i : Nat} (ge : label.size ≤ i) : rep label i = i :=
  h.rep_of_root (parent_of_ge ge)

/-- Forests with the same roots have the same `rep`. -/
theorem rep_congr {label label' : Array Nat} {rank rank' : Nat → Nat}
    (h : Forest label rank) (h' : Forest label' rank')
    (same : ∀ j r, Leads label' j r ↔ Leads label j r) (i : Nat) :
    rep label' i = rep label i :=
  h'.rep_eq ((same _ _).2 (h.leads_rep i))

/-- A root joins another: its class goes to the root of the other. -/
theorem link {label : Array Nat} {rank : Nat → Nat} (h : Forest label rank)
    {s d : Nat} (hs : parent label s = s) (hd : parent label d = d) (ne : s ≠ d)
    (ls : s < label.size) (ld : d < label.size) :
    ∃ rank', Forest (label.setIfInBounds d s) rank'
      ∧ ∀ x, rep (label.setIfInBounds d s) x
        = if rep label x = d then s else rep label x := by
  have forest : Forest (label.setIfInBounds d s)
      (fun x => if rep label x = s then rank x + rank d + 1 else rank x) := by
    refine ⟨fun j lt => ?_, fun j up => ?_⟩
    · rw [parent_set _ _ _ _ ld]
      simp only [Array.size_setIfInBounds] at lt ⊢
      split
      · exact ls
      · exact h.lt j lt
    · rw [parent_set _ _ _ _ ld] at up ⊢
      by_cases eq : j = d
      · subst eq
        simp only [ite_true, h.rep_of_root hd, h.rep_of_root hs, Ne.symm ne, ite_false]
        omega
      · simp only [eq, ite_false] at up ⊢
        have same : rep label (parent label j) = rep label j :=
          h.rep_eq (Leads.step up (h.leads_rep _)) ▸ rfl
        have := h.up j up
        rw [same]
        split <;> omega
  have leads : ∀ x r, Leads label x r →
      Leads (label.setIfInBounds d s) x (if r = d then s else r) := by
    intro x r leads
    induction leads with
    | @root r is_root =>
      by_cases eq : r = d
      · subst eq
        simp only [ite_true]
        refine .step (by rw [parent_set _ _ _ _ ld]; simpa using ne) ?_
        rw [parent_set _ _ _ _ ld]; simp only [ite_true]
        exact .root (by rw [parent_set _ _ _ _ ld]; simp [ne, hs])
      · simp only [eq, ite_false]
        exact .root (by rw [parent_set _ _ _ _ ld]; simp [eq, is_root])
    | @step i r up _ ih =>
      have ne' : i ≠ d := fun eq => up (eq ▸ hd)
      have same : parent (label.setIfInBounds d s) i = parent label i := by
        rw [parent_set _ _ _ _ ld]; simp [ne']
      exact .step (by rw [same]; exact up) (by rw [same]; exact ih)
  exact ⟨_, forest, fun x => forest.rep_eq (leads x _ (h.leads_rep x))⟩

end Forest

theorem parent_extend (label : Array Nat) (k i : Nat) :
    parent (label ++ (List.range' label.size k).toArray) i = parent label i := by
  simp only [parent, Array.getD_eq_getD_getElem?, Array.getElem?_append]
  split
  · rfl
  · rename_i ge
    rw [Array.getElem?_eq_none (Nat.le_of_not_lt ge)]
    cases found : (List.range' label.size k).toArray[i - label.size]? with
    | none => rfl
    | some v =>
      obtain ⟨_, eq⟩ := Array.getElem?_eq_some_iff.1 found
      simp at eq ⊢
      omega

namespace Forest

/-- New spiders, each its own root, leave the forest as it is. -/
theorem extend {label : Array Nat} {rank : Nat → Nat} (h : Forest label rank) (k : Nat) :
    Forest (label ++ (List.range' label.size k).toArray) rank
      ∧ ∀ x, rep (label ++ (List.range' label.size k).toArray) x = rep label x := by
  have same := parent_extend label k
  have forest : Forest (label ++ (List.range' label.size k).toArray) rank := by
    refine ⟨fun i lt => ?_, fun i up => ?_⟩
    · rw [same]
      rcases Nat.lt_or_ge i label.size with lt' | ge
      · have := h.lt i lt'; simp; omega
      · rw [parent_of_ge ge]; exact lt
    · rw [same] at up ⊢; exact h.up i up
  have leads : ∀ x r, Leads label x r →
      Leads (label ++ (List.range' label.size k).toArray) x r := by
    intro x r leads
    induction leads with
    | root is_root => exact .root (by rw [same]; exact is_root)
    | step up _ ih => exact .step (by rw [same]; exact up) (by rw [same]; exact ih)
  exact ⟨forest, fun x => forest.rep_eq (leads x _ (h.leads_rep x))⟩

end Forest

end EGraph
