import EGraph.Union

/-!
# Rebuilding keeps the invariant

`Builder.rebuild` takes a pair off the worklist before it relates its
outputs, when the invariant still needs the pair. No method reads the
worklist, they only append to it, so the pair may as well leave after.
-/

namespace EGraph

namespace Builder

/-- A builder with another worklist. -/
abbrev setWork (b : Builder) (work : List (Nat × Nat)) : Builder :=
  { b with worklist := work }

/-- A method that does not read the worklist and only appends to it. -/
def Appends (f : Builder → Builder) : Prop :=
  ∀ (b : Builder) (work : List (Nat × Nat)), f (b.setWork work)
    = (f (b.setWork [])).setWork ((f (b.setWork [])).worklist ++ work)

namespace Appends

theorem of_frame {f : Builder → Builder}
    (h : ∀ (b : Builder) (work : List (Nat × Nat)), f (b.setWork work) = (f b).setWork work) :
    Appends f := by
  intro b work
  rw [h b work, h b []]
  simp

theorem comp {f g : Builder → Builder} (hf : Appends f) (hg : Appends g) :
    Appends fun b => g (f b) := by
  intro b work
  have first := hf b work
  have second := hg (f (b.setWork [])) ((f (b.setWork [])).worklist ++ work)
  have third : g (f (b.setWork [])) = _ := hg (f (b.setWork [])) (f (b.setWork [])).worklist
  show g (f (b.setWork work)) = (g (f (b.setWork []))).setWork
    ((g (f (b.setWork []))).worklist ++ work)
  rw [first, second, third]
  simp [List.append_assoc]

theorem foldl {α : Type} {g : Builder → α → Builder} (h : ∀ x, Appends (g · x)) :
    ∀ l : List α, Appends fun b => l.foldl g b := by
  intro l
  induction l with
  | nil => exact of_frame fun _ _ => rfl
  | cons x l ih => exact comp (h x) ih

end Appends

theorem appends_cons (box : Nat) : Appends (·.cons box) := by
  intro b work
  simp only [Builder.cons]
  cases b.hashcons[b.keys.getD box []]? with
  | none => simp
  | some first => by_cases same : first = box <;> simp [same]

theorem roots_setWork (work : List (Nat × Nat)) :
    ∀ (spiders : List Nat) (b : Builder),
      (b.setWork work).roots spiders
        = ((b.roots spiders).1.setWork work, (b.roots spiders).2) := by
  intro spiders
  induction spiders with
  | nil => intro b; rfl
  | cons spider spiders ih =>
    intro b
    simp only [roots]
    have : (b.setWork work).root spider = ((b.root spider).1.setWork work, (b.root spider).2) :=
      rfl
    rw [this, ih]

theorem appends_rekey (box : Nat) : Appends (·.rekey box) := by
  intro b work
  simp only [Builder.rekey]
  cases b.keys.getD box [] with
  | nil => simp
  | cons head inputs =>
    simp only [roots_setWork]
    exact appends_cons box
      { (b.roots inputs).1 with
        keys := (b.roots inputs).1.keys.setIfInBounds box (head :: (b.roots inputs).2) } work

theorem appends_link (left right : Nat) : Appends (·.link left right) := by
  intro b work
  have := Appends.foldl appends_rekey (b.parents.getD right [])
    { b with
      parents := b.parents.setIfInBounds right []
      hashcons := (b.parents.getD right []).foldl
        (fun hashcons box => hashcons.erase (b.keys.getD box [])) b.hashcons
      label := b.label.setIfInBounds right left
      least := if b.least.getD right right < b.least.getD left left
        then b.least.setIfInBounds left (b.least.getD right right) else b.least } work
  simp only [link]
  simp only [setWork] at this ⊢
  rw [this]

theorem appends_join (left right : Nat) : Appends (·.join left right) := by
  intro b work
  simp only [join]
  split
  · simp
  · split
    · simp
    · exact appends_link _ _ b work

theorem appends_union (left right : Nat) : Appends (·.union left right) := by
  intro b work
  exact appends_join _ _ ((b.root left).1.root right).1 work

/-! ## One pair of the worklist -/

theorem pointwise_map {f : Nat → Nat} {xs ys : List Nat} (eq : xs.map f = ys.map f) :
    Pointwise (fun i j => f i = f j) xs ys := by
  refine pointwise_iff.2 ⟨by simpa using congrArg List.length eq, fun i x y hx hy => ?_⟩
  have := congrArg (·[i]?) eq
  simpa [List.getElem?_map, hx, hy] using this

theorem outputs_lt {b : Builder} (h : Inv b) (box : Nat) :
    ∀ s ∈ b.outputs box, s < b.label.size := by
  intro s mem
  unfold outputs at mem
  cases at_box : b.nodes[box]? with
  | none => simp [at_box] at mem
  | some node =>
    simp only [at_box, Option.map_some, Option.getD_some] at mem
    exact (h.tables.wires node (mem_nodes at_box)).2 s mem

/-- A pair leaves the worklist once its outputs are related. -/
theorem Inv.drop {b : Builder} (h : Inv b) {init rest : List (Nat × Nat)} {pair : Nat × Nat}
    (work : b.worklist = init ++ [pair] ++ rest)
    (related : ∀ q ∈ (b.outputs pair.1).zip (b.outputs pair.2),
      rep b.label q.1 = rep b.label q.2) : Inv (b.setWork (init ++ rest)) := by
  have t := h.tables
  have hashed := t.hashed
  have sub : ∀ p ∈ init ++ rest, p ∈ b.worklist := by
    intro p mem
    rw [work]
    rcases List.mem_append.1 mem with mem | mem
    · exact List.mem_append_left _ (List.mem_append_left _ mem)
    · exact List.mem_append_right _ mem
  refine ⟨h.forest, ⟨⟨hashed.keys_size, hashed.arity, hashed.keys, hashed.hash_some,
    hashed.hash_none, fun p mem => hashed.work p (sub p mem),
    fun box k first fresh at_box entry => ?_⟩, t.2, t.3, t.4, t.5, t.6, t.7, t.8, t.9⟩⟩
  refine (hashed.linked box k first fresh at_box entry).mono fun _ _ pending => ?_
  refine pending.mono (fun _ _ eq => .of_class eq) fun p mem q zip => ?_
  rw [work] at mem
  rcases List.mem_append.1 mem with mem | mem
  · rcases List.mem_append.1 mem with mem | mem
    · exact .of_pair (p := p) (List.mem_append_left _ mem) zip
    · cases List.mem_singleton.1 mem
      exact .of_class (related q zip)
  · exact .of_pair (p := p) (List.mem_append_right _ mem) zip

/-- One round of `rebuild`: a pair leaves the worklist and its outputs are
related. -/
theorem drain_step {b : Builder} (h : Inv b) {init : List (Nat × Nat)} {pair : Nat × Nat}
    (work : b.worklist = pair :: init) :
    Inv (((b.outputs pair.1).zip (b.outputs pair.2)).foldl
        (fun b p => b.union p.1 p.2) (b.setWork init))
      ∧ (((b.outputs pair.1).zip (b.outputs pair.2)).foldl
        (fun b p => b.union p.1 p.2) (b.setWork init)).nodes = b.nodes
      ∧ (((b.outputs pair.1).zip (b.outputs pair.2)).foldl
        (fun b p => b.union p.1 p.2) (b.setWork init)).label.size = b.label.size
      ∧ (((b.outputs pair.1).zip (b.outputs pair.2)).foldl
        (fun b p => b.union p.1 p.2) (b.setWork init)).total = b.total
      ∧ (((b.outputs pair.1).zip (b.outputs pair.2)).foldl
        (fun b p => b.union p.1 p.2) (b.setWork init)).todo b.total + 1 ≤ b.todo b.total
      ∧ (∀ x y, rep b.label x = rep b.label y →
        rep (((b.outputs pair.1).zip (b.outputs pair.2)).foldl
          (fun b p => b.union p.1 p.2) (b.setWork init)).label x
        = rep (((b.outputs pair.1).zip (b.outputs pair.2)).foldl
          (fun b p => b.union p.1 p.2) (b.setWork init)).label y)
      ∧ ∀ r : Nat → Nat → Prop, Equivalence r → Congruent b.nodes.toList r →
        (∀ x y, rep b.label x = rep b.label y → r x y) →
        ∀ x y, rep (((b.outputs pair.1).zip (b.outputs pair.2)).foldl
          (fun b p => b.union p.1 p.2) (b.setWork init)).label x
        = rep (((b.outputs pair.1).zip (b.outputs pair.2)).foldl
          (fun b p => b.union p.1 p.2) (b.setWork init)).label y → r x y := by
  generalize zip_eq : (b.outputs pair.1).zip (b.outputs pair.2) = zip
  have bound : ∀ p ∈ zip, p.1 < b.label.size ∧ p.2 < b.label.size := by
    intro p mem
    rw [← zip_eq] at mem
    have := List.of_mem_zip mem
    exact ⟨outputs_lt h _ _ this.1, outputs_lt h _ _ this.2⟩
  have ⟨inv, nodes, size, total, todo, extensive, related, least⟩ := unions_spec zip b h bound
  have appends := Appends.foldl (g := fun b (p : Nat × Nat) => b.union p.1 p.2)
    (fun p => appends_union p.1 p.2) zip
  have short : zip.foldl (fun b p => b.union p.1 p.2) (b.setWork init) = _ := appends b init
  have full : zip.foldl (fun b p => b.union p.1 p.2) (b.setWork (pair :: init)) = _ :=
    appends b (pair :: init)
  have whole : b.setWork (pair :: init) = b := by rw [← work]
  rw [whole] at full
  have full_work : (zip.foldl (fun b p => b.union p.1 p.2) b).worklist
      = (zip.foldl (fun b p => b.union p.1 p.2) (b.setWork [])).worklist ++ [pair] ++ init := by
    have := congrArg Builder.worklist full
    simpa using this
  have same : zip.foldl (fun b p => b.union p.1 p.2) (b.setWork init)
      = (zip.foldl (fun b p => b.union p.1 p.2) b).setWork
        ((zip.foldl (fun b p => b.union p.1 p.2) (b.setWork [])).worklist ++ init) := by
    rw [short, full]
  have dropped := inv.drop full_work fun q mem => related q (by
    rw [← zip_eq]
    simpa only [outputs, nodes] using mem)
  rw [same]
  refine ⟨dropped, nodes, size, total, ?_, extensive, fun r eqv congruent sub x y eq => ?_⟩
  · have : (Builder.setWork (zip.foldl (fun b p => b.union p.1 p.2) b)
          ((zip.foldl (fun b p => b.union p.1 p.2) (b.setWork [])).worklist ++ init)).todo b.total
        + 1 = (zip.foldl (fun b p => b.union p.1 p.2) b).todo b.total := by
      show ((zip.foldl (fun b p => b.union p.1 p.2) (b.setWork [])).worklist ++ init).length
          + (zip.foldl (fun b p => b.union p.1 p.2) b).count * (b.total + 1) + 1
        = (zip.foldl (fun b p => b.union p.1 p.2) b).worklist.length
          + (zip.foldl (fun b p => b.union p.1 p.2) b).count * (b.total + 1)
      rw [full_work]
      simp only [List.length_append, List.length_singleton]
      omega
    omega
  · refine least r eqv sub (fun q mem => ?_) x y eq
    obtain ⟨a, c, at_a, at_c, da, dc, head, inputs⟩ :=
      h.tables.hashed.work pair (by rw [work]; simp)
    rw [← zip_eq, outputs_of at_a, outputs_of at_c] at mem
    exact (congruent a (mem_nodes at_a) c (mem_nodes at_c) da dc head
      (Pointwise.mono sub (pointwise_map inputs))).2 q mem

/-- With enough fuel, `drain` empties the worklist, keeps the invariant and
relates no more than a congruent equivalence containing the classes. -/
theorem drain_spec :
    ∀ (fuel : Nat) (b : Builder), Inv b → b.todo b.total ≤ fuel →
      Inv (drain fuel b) ∧ (drain fuel b).worklist = [] ∧ (drain fuel b).nodes = b.nodes
        ∧ (drain fuel b).label.size = b.label.size
        ∧ (∀ x y, rep b.label x = rep b.label y →
          rep (drain fuel b).label x = rep (drain fuel b).label y)
        ∧ ∀ r : Nat → Nat → Prop, Equivalence r → Congruent b.nodes.toList r →
          (∀ x y, rep b.label x = rep b.label y → r x y) →
          ∀ x y, rep (drain fuel b).label x = rep (drain fuel b).label y → r x y := by
  intro fuel
  induction fuel with
  | zero =>
    intro b h le
    have : b.worklist.length = 0 := by unfold todo at le; omega
    exact ⟨h, List.length_eq_zero_iff.1 this, rfl, rfl, fun _ _ eq => eq,
      fun _ _ _ sub x y eq => sub x y eq⟩
  | succ fuel ih =>
    intro b h le
    unfold drain
    cases work : b.worklist with
    | nil =>
      exact ⟨h, work, rfl, rfl, fun _ _ eq => eq, fun _ _ _ sub x y eq => sub x y eq⟩
    | cons pair rest =>
      have ⟨inv, nodes, size, total, todo, extensive, least⟩ := drain_step h work
      have ⟨inv', empty, nodes', size', extensive', least'⟩ :=
        ih _ inv (by rw [total]; omega)
      refine ⟨inv', empty, nodes'.trans nodes, size'.trans size,
        fun x y eq => extensive' x y (extensive x y eq), fun r eqv congruent sub x y eq => ?_⟩
      exact least' r eqv (nodes ▸ congruent) (least r eqv congruent sub) x y eq

end Builder

end EGraph
