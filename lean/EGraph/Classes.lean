import EGraph.Partition

/-!
# Lists of classes

A list of classes relates two spiders when it sends them to the same one, and
it is normal when it sends each spider to the least one it relates it to.
Normal lists are equal when they relate the same spiders, so that an equation
between e-graphs is an equivalence between what their classes relate.
-/

namespace EGraph

/-- The class of a spider, itself when it is not there. -/
def find (classes : List Nat) (i : Nat) : Nat := classes.getD i i

/-- Two spiders are in the same class. -/
def Same (classes : List Nat) (i j : Nat) : Prop := find classes i = find classes j

/-- Each spider paired with its class, Python's `enumerate(classes)`. -/
def links (classes : List Nat) : List (Nat × Nat) :=
  classes.zipIdx.map fun entry => (entry.2, entry.1)

theorem find_of_some {classes : List Nat} {i c : Nat}
    (h : classes[i]? = some c) : find classes i = c := by
  simp [find, List.getD_eq_getElem?_getD, h]

theorem find_of_ge {classes : List Nat} {i : Nat}
    (h : classes.length ≤ i) : find classes i = i := by
  simp [find, List.getD_eq_getElem?_getD, List.getElem?_eq_none h]

theorem some_find {classes : List Nat} {i : Nat}
    (h : i < classes.length) : classes[i]? = some (find classes i) := by
  simp [find, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]

theorem mem_links {classes : List Nat} {i c : Nat} :
    (i, c) ∈ links classes ↔ classes[i]? = some c := by
  simp only [links, List.mem_map, Prod.exists, List.mem_zipIdx_iff_getElem?,
    Prod.mk.injEq]
  exact ⟨fun ⟨_, _, h, hi, hc⟩ => hi ▸ hc ▸ h, fun h => ⟨_, _, h, rfl, rfl⟩⟩

theorem Same.equivalence (classes : List Nat) : Equivalence (Same classes) :=
  ⟨fun _ => rfl, Eq.symm, Eq.trans⟩

/-- A list sending each spider to the least spider of its class. -/
structure Normal (classes : List Nat) : Prop where
  below : ∀ i c : Nat, classes[i]? = some c → c ≤ i
  idem : ∀ i c : Nat, classes[i]? = some c → classes[c]? = some c

namespace Normal

theorem find_le {classes : List Nat} (h : Normal classes) (i : Nat) :
    find classes i ≤ i := by
  rcases Nat.lt_or_ge i classes.length with lt | ge
  · exact h.below i _ (some_find lt)
  · rw [find_of_ge ge]; exact Nat.le_refl i

theorem find_find {classes : List Nat} (h : Normal classes) (i : Nat) :
    find classes (find classes i) = find classes i := by
  rcases Nat.lt_or_ge i classes.length with lt | ge
  · exact find_of_some (h.idem i _ (some_find lt))
  · rw [find_of_ge ge, find_of_ge ge]

theorem lt {classes : List Nat} (h : Normal classes) {i c : Nat}
    (mem : (i, c) ∈ links classes) : i < classes.length ∧ c < classes.length := by
  have some := mem_links.1 mem
  have lt : i < classes.length := by
    rcases Nat.lt_or_ge i classes.length with lt | ge
    · exact lt
    · simp [List.getElem?_eq_none ge] at some
  exact ⟨lt, Nat.lt_of_le_of_lt (h.below i c some) lt⟩

/-- The pairs of a normal list generate what it relates. -/
theorem related_links {classes : List Nat} (h : Normal classes) {i j : Nat} :
    Related (links classes) i j ↔ Same classes i j := by
  constructor
  · intro related
    refine Related.iff_forall.1 related _ (Same.equivalence classes) ?_
    intro (i, c) mem
    have some := mem_links.1 mem
    show find classes i = find classes c
    rw [find_of_some some, find_of_some (h.idem i c some)]
  · intro same
    have link (k : Nat) : Related (links classes) k (find classes k) := by
      rcases Nat.lt_or_ge k classes.length with lt | ge
      · exact .pair (mem_links.2 (some_find lt))
      · rw [find_of_ge ge]; exact .refl k
    have := link j
    rw [← show find classes i = find classes j from same] at this
    exact (link i).trans this.symm

/-- Normal lists of one length relating the same spiders are equal. -/
theorem ext {a b : List Nat} (ha : Normal a) (hb : Normal b)
    (length : a.length = b.length) (same : ∀ i j, Same a i j ↔ Same b i j) :
    a = b := by
  have le {a b : List Nat} (ha : Normal a) (hb : Normal b)
      (same : ∀ i j, Same a i j → Same b i j) (i : Nat) :
      find b i ≤ find a i := by
    have : find b i = find b (find a i) := same i _ (ha.find_find i).symm
    rw [this]
    exact hb.find_le _
  apply List.ext_getElem? ; intro i
  rcases Nat.lt_or_ge i a.length with lt | ge
  · rw [some_find lt, some_find (length ▸ lt)]
    exact congrArg some (Nat.le_antisymm
      (le hb ha (fun i j => (same i j).2) i) (le ha hb (fun i j => (same i j).1) i))
  · rw [List.getElem?_eq_none ge, List.getElem?_eq_none (length ▸ ge)]

end Normal

/-- `partition` is normal and relates what its pairs generate. -/
theorem partition_normal (n : Nat) (pairs : List (Nat × Nat))
    (bound : ∀ p ∈ pairs, p.1 < n ∧ p.2 < n) :
    (partition n pairs).length = n ∧ Normal (partition n pairs)
      ∧ ∀ i j, Same (partition n pairs) i j ↔ Related pairs i j := by
  have ⟨c, canonical, length, holds⟩ := partition_spec n pairs bound
  have fixed (k : Nat) : c (c k) = c k :=
    (canonical.equal _ _ (canonical.related k)).symm
  have lt {i v : Nat} (some : (partition n pairs)[i]? = some v) : i < n := by
    rcases Nat.lt_or_ge i n with lt | ge
    · exact lt
    · rw [List.getElem?_eq_none (by rw [length]; exact ge)] at some; cases some
  have find_eq (k : Nat) : find (partition n pairs) k = c k := by
    rcases Nat.lt_or_ge k n with lt | ge
    · exact find_of_some (holds k lt)
    · rw [find_of_ge (by rw [length]; exact ge)]
      exact Related.eq_of_ge bound (canonical.related k) ge
  refine ⟨length, ⟨fun i v some => ?_, fun i v some => ?_⟩, fun i j => ?_⟩
  · have := holds i (lt some)
    have := canonical.below i
    simp_all
  · have hi := holds i (lt some)
    have : v = c i := by simp_all
    subst this
    have := holds (c i) (Nat.lt_of_le_of_lt (canonical.below i) (lt some))
    rw [fixed] at this
    exact this
  · show find _ i = find _ j ↔ _
    rw [find_eq, find_eq]
    constructor
    · intro eq
      have := canonical.related j
      rw [← eq] at this
      exact (canonical.related i).trans this.symm
    · exact canonical.equal i j

end EGraph
