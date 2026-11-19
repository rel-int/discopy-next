import EGraph.Graph

/-!
# Rebuilding

The model of `EGraph.rebuild`: relate the outputs of each deterministic node
with those of the first node of its head on its classes, until nothing
changes. The sum of the classes goes down at each step that changes them,
which is the fuel of the loop and the proof that it ends.
-/

namespace EGraph

/-- Two lists of one length related entry by entry. -/
def Pointwise (r : Nat → Nat → Prop) (xs ys : List Nat) : Prop :=
  xs.length = ys.length ∧ ∀ p ∈ xs.zip ys, r p.1 p.2

theorem Pointwise.mono {r r' : Nat → Nat → Prop} (sub : ∀ i j, r i j → r' i j)
    {xs ys : List Nat} (h : Pointwise r xs ys) : Pointwise r' xs ys :=
  ⟨h.1, fun p mem => sub _ _ (h.2 p mem)⟩

/-- A relation is congruent when two deterministic nodes of one head on
related inputs have related outputs. -/
def Congruent (nodes : List Node) (r : Nat → Nat → Prop) : Prop :=
  ∀ a ∈ nodes, ∀ b ∈ nodes, a.det = true → b.det = true → a.head = b.head →
    Pointwise r a.inputs b.inputs → Pointwise r a.outputs b.outputs

namespace Graph

/-- The outputs of each deterministic node paired with those of the first
node of its head on its classes. -/
def congruences (nodes : List Node) (classes : List Nat) : List (Nat × Nat) :=
  nodes.flatMap fun node =>
    match node.det, nodes.find?
      (reads classes node.head (node.inputs.map (find classes))) with
    | true, some first => first.outputs.zip node.outputs
    | _, _ => []

/-- One round of rebuilding. -/
def step (nodes : List Node) (classes : List Nat) : List Nat :=
  partition classes.length (links classes ++ congruences nodes classes)

/-- Rounds of rebuilding until one changes nothing, at most `fuel` of them. -/
def close (nodes : List Node) : Nat → List Nat → List Nat
  | 0, classes => classes
  | fuel + 1, classes =>
    let next := step nodes classes
    if next = classes then classes else close nodes fuel next

/-- The congruence closure. -/
def rebuild (g : Graph) : Graph :=
  { g with classes := close g.nodes g.classes.sum g.classes }

/-- What the theorems ask of an e-graph: normal classes, outputs that are
spiders, and as many outputs for each node of a head. -/
structure Valid (g : Graph) : Prop where
  normal : Normal g.classes
  outputs : ∀ node ∈ g.nodes, ∀ s ∈ node.outputs, s < g.classes.length
  arity : ∀ a ∈ g.nodes, ∀ b ∈ g.nodes, a.head = b.head →
    a.outputs.length = b.outputs.length

/-! ## Pointwise relations -/

theorem pointwise_same {classes xs ys : List Nat} :
    Pointwise (Same classes) xs ys
      ↔ xs.map (find classes) = ys.map (find classes) := by
  induction xs generalizing ys with
  | nil => cases ys <;> simp [Pointwise]
  | cons x xs ih =>
    cases ys with
    | nil => simp [Pointwise]
    | cons y ys =>
      have := ih (ys := ys)
      simp only [Pointwise, List.length_cons, List.zip_cons_cons, List.mem_cons,
        List.map_cons, List.cons.injEq, Nat.add_right_cancel_iff] at this ⊢
      constructor
      · intro ⟨length, h⟩
        exact ⟨h _ (.inl rfl), this.1 ⟨length, fun p mem => h p (.inr mem)⟩⟩
      · intro ⟨head, tail⟩
        have ⟨length, h⟩ := this.2 tail
        refine ⟨length, fun p mem => ?_⟩
        rcases mem with rfl | mem
        · exact head
        · exact h p mem

/-! ## One round -/

theorem mem_congruences {nodes : List Node} {classes : List Nat} {p : Nat × Nat}
    (mem : p ∈ congruences nodes classes) :
    ∃ node ∈ nodes, ∃ first ∈ nodes, node.det = true ∧ first.det = true
      ∧ first.head = node.head
      ∧ first.inputs.map (find classes) = node.inputs.map (find classes)
      ∧ nodes.find? (reads classes node.head (node.inputs.map (find classes)))
        = some first
      ∧ p ∈ first.outputs.zip node.outputs := by
  obtain ⟨node, node_mem, mem⟩ := List.mem_flatMap.1 mem
  split at mem
  · rename_i first det found
    have := List.find?_some found
    simp only [reads, Bool.and_eq_true, beq_iff_eq] at this
    exact ⟨node, node_mem, first, List.mem_of_find?_eq_some found, det,
      this.1.1, this.1.2, this.2, found, mem⟩
  · cases mem

/-- The nodes and classes of the rounds: normal classes on `n` spiders, the
outputs of the nodes among them. -/
structure Round (nodes : List Node) (n : Nat) (classes : List Nat) : Prop where
  length : classes.length = n
  normal : Normal classes
  outputs : ∀ node ∈ nodes, ∀ s ∈ node.outputs, s < n

theorem step_spec {nodes : List Node} {n : Nat} {classes : List Nat}
    (h : Round nodes n classes) :
    Round nodes n (step nodes classes)
      ∧ ∀ i j, Same (step nodes classes) i j
        ↔ Related (links classes ++ congruences nodes classes) i j := by
  have ⟨length, normal, same⟩ := partition_normal classes.length
    (links classes ++ congruences nodes classes) fun p mem => by
      rcases List.mem_append.1 mem with mem | mem
      · exact h.normal.lt mem
      · obtain ⟨node, node_mem, first, first_mem, _, _, _, _, _, mem⟩ :=
          mem_congruences mem
        have := List.of_mem_zip mem
        rw [h.length]
        exact ⟨h.outputs first first_mem _ this.1, h.outputs node node_mem _ this.2⟩
  exact ⟨⟨length.trans h.length, normal, h.outputs⟩, same⟩

theorem step_extensive {nodes : List Node} {n : Nat} {classes : List Nat}
    (h : Round nodes n classes) {i j : Nat} (same : Same classes i j) :
    Same (step nodes classes) i j :=
  ((step_spec h).2 i j).2
    ((h.normal.related_links.2 same).mono fun _ => List.mem_append_left _)

theorem step_le {nodes : List Node} {n : Nat} {classes : List Nat}
    (h : Round nodes n classes) (i : Nat) :
    find (step nodes classes) i ≤ find classes i := by
  have : find (step nodes classes) i = find (step nodes classes) (find classes i) :=
    step_extensive h (h.normal.find_find i).symm
  rw [this]
  exact (step_spec h).1.normal.find_le _

/-! ## The sum of the classes -/

theorem sum_range {n : Nat} {f g : Nat → Nat} (le : ∀ i, i < n → f i ≤ g i) :
    ((List.range n).map f).sum ≤ ((List.range n).map g).sum
      ∧ (((List.range n).map f).sum = ((List.range n).map g).sum →
        ∀ i, i < n → f i = g i) := by
  induction n with
  | zero => exact ⟨Nat.le_refl _, fun _ _ lt => absurd lt (Nat.not_lt_zero _)⟩
  | succ n ih =>
    have ⟨sum_le, eq⟩ := ih fun i lt => le i (Nat.lt_succ_of_lt lt)
    have := le n (Nat.lt_succ_self n)
    simp only [List.range_succ, List.map_append, List.sum_append, List.map_cons,
      List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
    refine ⟨by omega, fun sum i lt => ?_⟩
    rcases Nat.lt_succ_iff_lt_or_eq.1 lt with lt | rfl
    · exact eq (by omega) i lt
    · omega

theorem eq_map_find (classes : List Nat) :
    classes = (List.range classes.length).map (find classes) := by
  apply List.ext_getElem
  · simp
  · intro i lt _
    simp only [List.getElem_map, List.getElem_range]
    exact (find_of_some (List.getElem?_eq_getElem lt)).symm

theorem step_sum {nodes : List Node} {n : Nat} {classes : List Nat}
    (h : Round nodes n classes) :
    (step nodes classes).sum ≤ classes.sum
      ∧ ((step nodes classes).sum = classes.sum → step nodes classes = classes) := by
  have length := (step_spec h).1.length
  have ⟨le, eq⟩ := sum_range (n := n) fun i _ => step_le h i
  have left : step nodes classes
      = (List.range n).map (find (step nodes classes)) := by
    have := eq_map_find (step nodes classes); rwa [length] at this
  have right : classes = (List.range n).map (find classes) := by
    have := eq_map_find classes; rwa [h.length] at this
  have left_sum := congrArg List.sum left
  have right_sum := congrArg List.sum right
  refine ⟨by omega, fun sum => left.trans (.trans ?_ right.symm)⟩
  exact List.map_congr_left fun i mem => eq (by omega) i (List.mem_range.1 mem)

/-! ## The rounds -/

theorem close_succ (nodes : List Node) (fuel : Nat) (classes : List Nat) :
    close nodes (fuel + 1) classes = if step nodes classes = classes then classes
      else close nodes fuel (step nodes classes) := rfl

theorem close_induct {nodes : List Node} {P : List Nat → Prop}
    (preserved : ∀ classes, P classes → P (step nodes classes)) :
    ∀ fuel classes, P classes → P (close nodes fuel classes) := by
  intro fuel
  induction fuel with
  | zero => exact fun _ h => h
  | succ fuel ih =>
    intro classes h
    rw [close_succ]
    split
    · exact h
    · exact ih _ (preserved _ h)

/-- With the sum of the classes as fuel, the rounds end where a round changes
nothing. -/
theorem close_fixed {nodes : List Node} {n : Nat} :
    ∀ fuel classes, Round nodes n classes → classes.sum ≤ fuel →
      step nodes (close nodes fuel classes) = close nodes fuel classes := by
  intro fuel
  induction fuel with
  | zero =>
    intro classes h sum
    have := step_sum h
    exact this.2 (by omega)
  | succ fuel ih =>
    intro classes h sum
    have := step_sum h
    rw [close_succ]
    split
    · rename_i eq; exact eq
    · rename_i ne
      refine ih _ (step_spec h).1 ?_
      have : (step nodes classes).sum ≠ classes.sum := fun eq => ne (this.2 eq)
      omega

theorem fixed_same {nodes : List Node} {n : Nat} {classes : List Nat}
    (h : Round nodes n classes) (fixed : step nodes classes = classes) :
    ∀ p ∈ congruences nodes classes, Same classes p.1 p.2 := by
  intro p mem
  have := ((step_spec h).2 p.1 p.2).2 (.pair (List.mem_append_right _ mem))
  rwa [fixed] at this

theorem fixed_congruent {nodes : List Node} {n : Nat} {classes : List Nat}
    (h : Round nodes n classes) (fixed : step nodes classes = classes)
    (arity : ∀ a ∈ nodes, ∀ b ∈ nodes, a.head = b.head →
      a.outputs.length = b.outputs.length) :
    Congruent nodes (Same classes) := by
  intro a a_mem b b_mem a_det b_det head inputs
  have key := pointwise_same.1 inputs
  cases found : nodes.find?
      (reads classes a.head (a.inputs.map (find classes))) with
  | none =>
    have := List.find?_eq_none.1 found a a_mem
    simp [reads, a_det] at this
  | some first =>
    have first_mem := List.mem_of_find?_eq_some found
    have first_head : first.head = a.head := by
      have := List.find?_some found
      simp only [reads, Bool.and_eq_true, beq_iff_eq] at this
      exact this.1.2
    have to (node : Node) (node_mem : node ∈ nodes) (det : node.det = true)
        (found : nodes.find? (reads classes node.head
          (node.inputs.map (find classes))) = some first)
        (head : first.head = node.head) :
        first.outputs.map (find classes) = node.outputs.map (find classes) := by
      refine pointwise_same.1 ⟨arity _ first_mem _ node_mem head, fun p mem => ?_⟩
      refine fixed_same h fixed p (List.mem_flatMap.2 ⟨node, node_mem, ?_⟩)
      simp only [det, found]
      exact mem
    refine pointwise_same.2 ((to a a_mem a_det found first_head).symm.trans ?_)
    exact to b b_mem b_det (by rw [← head, ← key]; exact found) (first_head.trans head)

theorem step_least {nodes : List Node} {n : Nat} {classes : List Nat}
    (h : Round nodes n classes) {r : Nat → Nat → Prop} (eqv : Equivalence r)
    (congruent : Congruent nodes r) (sub : ∀ i j, Same classes i j → r i j)
    (i j : Nat) (same : Same (step nodes classes) i j) : r i j := by
  refine Related.iff_forall.1 (((step_spec h).2 i j).1 same) r eqv fun p mem => ?_
  rcases List.mem_append.1 mem with mem | mem
  · exact sub _ _ (h.normal.related_links.1 (.pair mem))
  · obtain ⟨node, node_mem, first, first_mem, det, first_det, head, key, _, mem⟩ :=
      mem_congruences mem
    exact (congruent first first_mem node node_mem first_det det head
      (Pointwise.mono sub (pointwise_same.2 key))).2 p mem

/-! ## Rebuilding -/

theorem Valid.round {g : Graph} (h : Valid g) :
    Round g.nodes g.classes.length g.classes :=
  ⟨rfl, h.normal, h.outputs⟩

theorem rebuild_round {g : Graph} (h : Valid g) :
    Round g.nodes g.classes.length g.rebuild.classes :=
  close_induct (P := Round g.nodes g.classes.length)
    (fun _ h => (step_spec h).1) _ _ h.round

theorem rebuild_valid {g : Graph} (h : Valid g) : Valid g.rebuild :=
  have round := rebuild_round h
  ⟨round.normal, fun node mem s out => round.length ▸ h.outputs node mem s out,
    h.arity⟩

theorem rebuild_fixed {g : Graph} (h : Valid g) :
    step g.nodes g.rebuild.classes = g.rebuild.classes :=
  close_fixed _ _ h.round (Nat.le_refl _)

/-- What rebuilding relates is congruent. -/
theorem rebuild_congruent {g : Graph} (h : Valid g) :
    Congruent g.nodes (Same g.rebuild.classes) :=
  fixed_congruent (rebuild_round h) (rebuild_fixed h) h.arity

/-- Rebuilding relates what was related. -/
theorem rebuild_extensive {g : Graph} (h : Valid g) {i j : Nat}
    (same : Same g.classes i j) : Same g.rebuild.classes i j :=
  (close_induct (P := fun classes => Round g.nodes g.classes.length classes
      ∧ Same classes i j)
    (fun _ h => ⟨(step_spec h.1).1, step_extensive h.1 h.2⟩) _ _ ⟨h.round, same⟩).2

/-- Rebuilding relates no more than any congruent equivalence containing what
was related. -/
theorem rebuild_least {g : Graph} (h : Valid g) {r : Nat → Nat → Prop}
    (eqv : Equivalence r) (congruent : Congruent g.nodes r)
    (sub : ∀ i j, Same g.classes i j → r i j) {i j : Nat}
    (same : Same g.rebuild.classes i j) : r i j :=
  (close_induct (P := fun classes => Round g.nodes g.classes.length classes
      ∧ ∀ i j, Same classes i j → r i j)
    (fun _ h => ⟨(step_spec h.1).1, step_least h.1 eqv congruent h.2⟩)
    _ _ ⟨h.round, sub⟩).2 i j same

theorem rebuild_of_fixed {g : Graph} (fixed : step g.nodes g.classes = g.classes) :
    g.rebuild = g := by
  refine ext rfl ?_
  show close g.nodes g.classes.sum g.classes = g.classes
  cases g.classes.sum <;> simp [close, fixed]

/-- Rebuilding twice is rebuilding once. -/
theorem rebuild_idempotent {g : Graph} (h : Valid g) :
    g.rebuild.rebuild = g.rebuild :=
  rebuild_of_fixed (rebuild_fixed h)

theorem merge_valid {g : Graph} (h : Valid g) (spiders : List Nat)
    (lt : ∀ s ∈ spiders, s < g.classes.length) : Valid (g.merge spiders) :=
  have ⟨length, normal, _⟩ := merge_spec g h.normal spiders lt
  ⟨normal, fun node mem s out => length ▸ h.outputs node mem s out, h.arity⟩

/-- Rebuilding is confluent: a closure is closed, relates the spiders merged,
and is reached as well by rebuilding before merging. -/
theorem rebuild_closure {g : Graph} (h : Valid g) (s t : Nat)
    (hs : s < g.classes.length) (ht : t < g.classes.length) :
    (g.merge [s, t]).rebuild.rebuild = (g.merge [s, t]).rebuild
      ∧ (g.merge [s, t]).rebuild.merge [s, t] = (g.merge [s, t]).rebuild
      ∧ (g.rebuild.merge [s, t]).rebuild = (g.merge [s, t]).rebuild := by
  have lt {n : Nat} (hs : s < n) (ht : t < n) : ∀ x ∈ [s, t], x < n := by
    simp [hs, ht]
  have merged := merge_valid h [s, t] (lt hs ht)
  have closed := rebuild_valid merged
  have length := (rebuild_round merged).length.trans
    (merge_spec g h.normal [s, t] (lt hs ht)).1
  have rebuilt := rebuild_valid h
  have length' := (rebuild_round h).length
  have lt' := lt (length' ▸ hs) (length' ▸ ht)
  have merged' := merge_valid rebuilt [s, t] lt'
  have related : ∀ p ∈ chain [s, t],
      Same (g.merge [s, t]).rebuild.classes p.1 p.2 := fun p mem =>
    rebuild_extensive merged (merge_chain g h.normal [s, t] (lt hs ht) p mem)
  have congruent : Congruent g.nodes (Same (g.merge [s, t]).rebuild.classes) :=
    rebuild_congruent merged
  have congruent' : Congruent g.nodes
      (Same (g.rebuild.merge [s, t]).rebuild.classes) := rebuild_congruent merged'
  refine ⟨rebuild_idempotent merged,
    merge_of_same _ closed.normal _ (lt (length ▸ hs) (length ▸ ht)) related,
    ext rfl (Normal.ext (rebuild_valid merged').normal closed.normal ?_ fun i j => ⟨?_, ?_⟩)⟩
  · exact (rebuild_round merged').length.trans
      ((merge_spec _ rebuilt.normal [s, t] lt').1.trans (length'.trans length.symm))
  · refine rebuild_least merged' (Same.equivalence _) congruent ?_
    intro i j
    refine merge_least _ rebuilt.normal _ lt' (Same.equivalence _) (fun i j => ?_) related
    refine rebuild_least h (Same.equivalence _) congruent ?_
    exact fun i j same =>
      rebuild_extensive merged (merge_extensive g h.normal _ (lt hs ht) same)
  · refine rebuild_least merged (Same.equivalence _) congruent' ?_
    intro i j
    refine merge_least g h.normal _ (lt hs ht) (Same.equivalence _) (fun i j same => ?_)
      fun p mem => rebuild_extensive merged'
        (merge_chain _ rebuilt.normal [s, t] lt' p mem)
    exact rebuild_extensive merged' (merge_extensive _ rebuilt.normal _ lt'
      (rebuild_extensive h same))

end Graph

end EGraph
