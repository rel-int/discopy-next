import EGraph.Classes

/-!
# The pure e-graph

The model of `discopy.egraph.EGraph` without its types: the spiders are the
indices of a list of classes, a node is a box on input and output spiders,
the box given by a number, its head, and whether it is deterministic.
-/

namespace EGraph

/-- A box on its wires: Python's `boxes[j]` and `box_wires[j]`. -/
structure Node where
  head : Nat
  det : Bool
  inputs : List Nat
  outputs : List Nat
deriving DecidableEq, Repr

/-- An e-graph: its nodes and the class of each spider. -/
structure Graph where
  nodes : List Node
  classes : List Nat
deriving DecidableEq, Repr

namespace Graph

theorem ext {g h : Graph} (nodes : g.nodes = h.nodes)
    (classes : g.classes = h.classes) : g = h := by
  cases g; cases h; simp_all

/-- Each spider paired with the next, Python's `zip(spiders, spiders[1:])`. -/
def chain (spiders : List Nat) : List (Nat × Nat) := spiders.zip spiders.tail

/-- Relate spiders, without closing under congruence. -/
def merge (g : Graph) (spiders : List Nat) : Graph :=
  { g with classes :=
      partition g.classes.length (links g.classes ++ chain spiders) }

/-- Whether a node is a deterministic box of a head reading some classes. -/
def reads (classes : List Nat) (head : Nat) (key : List Nat) (node : Node) : Bool :=
  node.det && node.head == head && node.inputs.map (find classes) == key

/-- Add a box reading some spiders and give its outputs: the next spiders, or
those of the first deterministic node of that head on their classes. -/
def add (g : Graph) (head : Nat) (det : Bool) (arity : Nat)
    (inputs : List Nat) : Graph × List Nat :=
  let key := inputs.map (find g.classes)
  match g.nodes.find? fun node => det && reads g.classes head key node with
  | some node => (g, node.outputs)
  | none =>
    let outputs := List.range' g.classes.length arity
    ({ nodes := g.nodes ++ [⟨head, det, inputs, outputs⟩],
       classes := g.classes ++ outputs }, outputs)

/-! ## Merging -/

theorem chain_lt {n : Nat} {spiders : List Nat} (lt : ∀ s ∈ spiders, s < n) :
    ∀ p ∈ chain spiders, p.1 < n ∧ p.2 < n := by
  intro (a, b) mem
  have := List.of_mem_zip mem
  exact ⟨lt a this.1, lt b (List.mem_of_mem_tail this.2)⟩

/-- Merging keeps the classes normal and relates what they and the spiders
generate. -/
theorem merge_spec (g : Graph) (normal : Normal g.classes) (spiders : List Nat)
    (lt : ∀ s ∈ spiders, s < g.classes.length) :
    (g.merge spiders).classes.length = g.classes.length
      ∧ Normal (g.merge spiders).classes
      ∧ ∀ i j, Same (g.merge spiders).classes i j
        ↔ Related (links g.classes ++ chain spiders) i j :=
  partition_normal _ _ fun p mem => by
    rcases List.mem_append.1 mem with mem | mem
    · exact normal.lt mem
    · exact chain_lt lt p mem

/-- Merging spiders already related changes nothing. -/
theorem merge_of_same (g : Graph) (normal : Normal g.classes)
    (spiders : List Nat) (lt : ∀ s ∈ spiders, s < g.classes.length)
    (same : ∀ p ∈ chain spiders, Same g.classes p.1 p.2) :
    g.merge spiders = g := by
  have ⟨length, merged, spec⟩ := merge_spec g normal spiders lt
  refine ext rfl (Normal.ext merged normal length fun i j => (spec i j).trans ?_)
  refine ⟨fun related => normal.related_links.1 (related.sub fun p mem => ?_),
    fun same => (normal.related_links.2 same).mono fun _ => List.mem_append_left _⟩
  rcases List.mem_append.1 mem with mem | mem
  · exact .pair mem
  · exact normal.related_links.2 (same p mem)

/-- Merging relates no more than an equivalence containing what was related
and the spiders. -/
theorem merge_least (g : Graph) (normal : Normal g.classes) (spiders : List Nat)
    (lt : ∀ s ∈ spiders, s < g.classes.length) {r : Nat → Nat → Prop}
    (eqv : Equivalence r) (sub : ∀ i j, Same g.classes i j → r i j)
    (chained : ∀ p ∈ chain spiders, r p.1 p.2) {i j : Nat}
    (same : Same (g.merge spiders).classes i j) : r i j := by
  refine Related.iff_forall.1 (((merge_spec g normal spiders lt).2.2 i j).1 same)
    r eqv fun p mem => ?_
  rcases List.mem_append.1 mem with mem | mem
  · exact sub _ _ (normal.related_links.1 (.pair mem))
  · exact chained p mem

theorem merge_extensive (g : Graph) (normal : Normal g.classes)
    (spiders : List Nat) (lt : ∀ s ∈ spiders, s < g.classes.length) {i j : Nat}
    (same : Same g.classes i j) : Same (g.merge spiders).classes i j :=
  ((merge_spec g normal spiders lt).2.2 i j).2
    ((normal.related_links.2 same).mono fun _ => List.mem_append_left _)

theorem merge_chain (g : Graph) (normal : Normal g.classes)
    (spiders : List Nat) (lt : ∀ s ∈ spiders, s < g.classes.length) :
    ∀ p ∈ chain spiders, Same (g.merge spiders).classes p.1 p.2 :=
  fun p mem => ((merge_spec g normal spiders lt).2.2 p.1 p.2).2
    (.pair (List.mem_append_right _ mem))

theorem merge_symm (g : Graph) (normal : Normal g.classes) (s t : Nat)
    (hs : s < g.classes.length) (ht : t < g.classes.length) :
    g.merge [s, t] = g.merge [t, s] := by
  have lt (a b : Nat) (ha : a < g.classes.length) (hb : b < g.classes.length) :
      ∀ x ∈ [a, b], x < g.classes.length := by simp [ha, hb]
  have ⟨length, merged, spec⟩ := merge_spec g normal [s, t] (lt s t hs ht)
  have ⟨length', merged', spec'⟩ := merge_spec g normal [t, s] (lt t s ht hs)
  have swap (a b : Nat) {i j : Nat}
      (related : Related (links g.classes ++ chain [a, b]) i j) :
      Related (links g.classes ++ chain [b, a]) i j := by
    refine related.sub fun p mem => ?_
    rcases List.mem_append.1 mem with mem | mem
    · exact .pair (List.mem_append_left _ mem)
    · cases List.mem_singleton.1 mem
      exact (Related.pair (List.mem_append_right _ (List.mem_singleton.2 rfl))).symm
  exact ext rfl (Normal.ext merged merged' (length.trans length'.symm) fun i j =>
    (spec i j).trans (Iff.trans ⟨swap s t, swap t s⟩ (spec' i j).symm))

theorem merge_trans (g : Graph) (normal : Normal g.classes) (s t u : Nat)
    (hs : s < g.classes.length) (ht : t < g.classes.length)
    (hu : u < g.classes.length) :
    ((g.merge [s, t]).merge [t, u]).merge [s, u] = (g.merge [s, t]).merge [t, u] := by
  have lt {n : Nat} (a b : Nat) (ha : a < n) (hb : b < n) :
      ∀ x ∈ [a, b], x < n := by simp [ha, hb]
  have ⟨length, merged, spec⟩ := merge_spec g normal [s, t] (lt s t hs ht)
  have ⟨length', merged', spec'⟩ := merge_spec (g.merge [s, t]) merged [t, u]
    (lt t u (length ▸ ht) (length ▸ hu))
  have last {l : List (Nat × Nat)} {a b : Nat} : Related (l ++ chain [a, b]) a b :=
    .pair (List.mem_append_right _ (List.mem_singleton.2 rfl))
  refine merge_of_same _ merged' _ (lt s u (by omega) (by omega)) fun p mem => ?_
  cases List.mem_singleton.1 mem
  refine (spec' s u).2 (Related.trans ?_ last)
  exact (merged.related_links.2 ((spec s t).2 last)).mono fun _ => List.mem_append_left _

/-! ## Adding -/

theorem find_append_range (classes : List Nat) (k i : Nat) :
    find (classes ++ List.range' classes.length k) i = find classes i := by
  simp only [find, List.getD_eq_getElem?_getD, List.getElem?_append]
  split
  · rfl
  · rename_i ge
    rw [List.getElem?_eq_none (Nat.le_of_not_lt ge)]
    cases h : (List.range' classes.length k)[i - classes.length]? with
    | none => rfl
    | some v =>
      obtain ⟨_, eq⟩ := List.getElem?_eq_some_iff.1 h
      simp at eq ⊢
      omega

theorem add_some {g : Graph} {head arity : Nat} {det : Bool} {inputs : List Nat}
    {node : Node}
    (found : g.nodes.find? (fun node => det
      && reads g.classes head (inputs.map (find g.classes)) node) = some node) :
    g.add head det arity inputs = (g, node.outputs) := by
  simp [add, found]

theorem add_none {g : Graph} {head arity : Nat} {det : Bool} {inputs : List Nat}
    (missing : g.nodes.find? (fun node => det
      && reads g.classes head (inputs.map (find g.classes)) node) = none) :
    g.add head det arity inputs = (
      { nodes := g.nodes
          ++ [⟨head, det, inputs, List.range' g.classes.length arity⟩],
        classes := g.classes ++ List.range' g.classes.length arity },
      List.range' g.classes.length arity) := by
  simp [add, missing]

/-- Adding a deterministic box again on the same classes changes nothing and
gives the same outputs. -/
theorem add_hashcons (g : Graph) (head arity : Nat) (inputs again : List Nat)
    (same : again.map (find (g.add head true arity inputs).1.classes)
      = inputs.map (find (g.add head true arity inputs).1.classes)) :
    (g.add head true arity inputs).1.add head true arity again
      = g.add head true arity inputs := by
  cases found : g.nodes.find? (fun node => true
      && reads g.classes head (inputs.map (find g.classes)) node) with
  | some node =>
    rw [add_some found] at same ⊢
    exact add_some (by rw [same]; exact found)
  | none =>
    have find_eq : find (g.classes ++ List.range' g.classes.length arity)
        = find g.classes := funext (find_append_range _ _)
    rw [add_none found] at same ⊢
    simp only [find_eq] at same
    have reads_eq : reads (g.classes ++ List.range' g.classes.length arity)
        = reads g.classes := by
      funext head key node; simp only [reads, find_eq]
    refine add_some (node := ⟨head, true, inputs, _⟩) ?_
    simp only [find_eq, reads_eq]
    rw [same, List.find?_append, found]
    simp [reads]

end Graph

end EGraph
