import DisCoLean.Premonoidal.Basic

/-!
# The terms of `discopy.cat`

The kappa calculus of a category, as `discopy.cat.TermBase` has it: a term
has one input, either its free variable `x : a ⊢ e : b` (`Open`) or its stack
`⊢ e : a → b` (`Closed`), and denotes a path of boxes. A term is built from

- `var a`, the variable, an identity
- `call f e`, the box `f` applied to `e`, i.e. `Application(Constant(f), e)`
- `letIn e body`, `let x = e in body`, i.e. `Application(Kappa(x, body), e)`
- `const f` and `kappa e`, the two terms with a stack

In a category a context has one variable, so the model is nameless: a
variable is the only one in scope, alpha-equivalent terms are equal, and
substitution captures nothing. The arrows of the free category are `Path`,
and `toPath` is what `TermBase.eval()` computes with no functor.

The model mirrors `discopy/cat.py` as of `b66485d`.
-/

namespace Terms.Cat

/-- A signature of a category, i.e. a graph: generating objects, boxes, and
for each box a domain and a codomain. -/
structure Signature where
  Ob : Type
  Box : Type
  dom : Box → Ob
  cod : Box → Ob

variable {S : Signature} {a b c d : S.Ob}

/-- A path of boxes, the arrows of the free category: `cons f p` is `p`
then `f`. -/
inductive Path (S : Signature) : S.Ob → S.Ob → Type
  | nil (a : S.Ob) : Path S a a
  | cons {a : S.Ob} (f : S.Box) : Path S a (S.dom f) → Path S a (S.cod f)

namespace Path

/-- Composition of paths, `p` then `q`. -/
def comp {a b c : S.Ob} : Path S a b → Path S b c → Path S a c
  | p, .nil _ => p
  | p, .cons f q => .cons f (comp p q)

@[simp] theorem comp_nil (p : Path S a b) : comp p (.nil b) = p := rfl

@[simp] theorem comp_cons (p : Path S a b) (f : S.Box) (q : Path S b (S.dom f)) :
    comp p (.cons f q) = .cons f (comp p q) := rfl

@[simp] theorem nil_comp (p : Path S a b) : comp (.nil a) p = p := by
  induction p with
  | nil => rfl
  | cons f p ih => simp [ih]

theorem comp_assoc (p : Path S a b) (q : Path S b c) (r : Path S c d) :
    comp (comp p q) r = comp p (comp q r) := by
  induction r with
  | nil => rfl
  | cons f r ih => simp [ih]

end Path

/-- `x : a ⊢ e : b`, a term of its one free variable, with no stack. -/
inductive Open (S : Signature) : S.Ob → S.Ob → Type
  | var (a : S.Ob) : Open S a a
  | call {a : S.Ob} (f : S.Box) : Open S a (S.dom f) → Open S a (S.cod f)
  | letIn {a b c : S.Ob} : Open S a b → Open S b c → Open S a c

/-- `⊢ e : a → b`, a term with a stack and no free variable. -/
inductive Closed (S : Signature) : S.Ob → S.Ob → Type
  | const (f : S.Box) : Closed S (S.dom f) (S.cod f)
  | kappa {a b : S.Ob} : Open S a b → Closed S a b

/-- A term with a stack applied to an argument, `Application(func, arg)`:
a call of a constant, a `let` of a kappa. -/
def Closed.apply {a b c : S.Ob} : Closed S a b → Open S c a → Open S c b
  | .const f, e => .call f e
  | .kappa body, e => .letIn e body

/-! ## Evaluation into the free category -/

/-- The arrow of a term: a variable is an identity, a call composes its
argument then its box, a `let` its expression then its body. -/
def Open.toPath {a b : S.Ob} : Open S a b → Path S a b
  | .var a => .nil a
  | .call f e => .cons f e.toPath
  | .letIn e body => e.toPath.comp body.toPath

/-- The arrow of a term with a stack: a constant is its box, a kappa does
nothing. -/
def Closed.toPath {a b : S.Ob} : Closed S a b → Path S a b
  | .const f => .cons f (.nil _)
  | .kappa e => e.toPath

/-! ## Beta reduction -/

/-- The term with `x` substituted for its variable. -/
def Open.subst {a b c : S.Ob} : Open S b c → Open S a b → Open S a c
  | .var _, x => x
  | .call f e, x => .call f (e.subst x)
  | .letIn e body, x => .letIn (e.subst x) body

/-- The beta normal form: every `let x = e in body` contracted to
`body[e / x]`. -/
def Open.normalForm {a b : S.Ob} : Open S a b → Open S a b
  | .var x => .var x
  | .call f e => .call f e.normalForm
  | .letIn e body => body.normalForm.subst e.normalForm

/-- The beta normal form of a term with a stack, that of its body. -/
def Closed.normalForm {a b : S.Ob} : Closed S a b → Closed S a b
  | .const f => .const f
  | .kappa e => .kappa e.normalForm

/-- A term in normal form: a variable under a sequence of calls. -/
inductive Open.Normal : {a b : S.Ob} → Open S a b → Prop
  | var (a : S.Ob) : Normal (.var a)
  | call {a : S.Ob} (f : S.Box) {e : Open S a (S.dom f)} : Normal e → Normal (.call f e)

/-! ## Reading an arrow, eta expansion -/

/-- The variable under the boxes of a path, in order. -/
def Path.spine {a b : S.Ob} : Path S a b → Open S a b
  | .nil _ => .var _
  | .cons f p => .call f p.spine

/-- The term of an arrow, `Arrow.to_term`: the kappa of its spine. -/
def Path.read (p : Path S a b) : Closed S a b := .kappa p.spine

/-- The eta expansion of a term with a stack, `TermBase.expand`: its
application to the variable, under its kappa. -/
def Closed.expand (t : Closed S a b) : Closed S a b := .kappa (t.apply (.var a))

/-! ## Theorems -/

namespace Open

theorem toPath_subst (e : Open S b c) (x : Open S a b) :
    (e.subst x).toPath = x.toPath.comp e.toPath := by
  induction e generalizing a with
  | var => rfl
  | call f e ih => simp [subst, toPath, ih]
  | letIn e body ih _ => simp [subst, toPath, ih, Path.comp_assoc]

/-- Beta reduction is sound, on the nose. -/
theorem toPath_normalForm (e : Open S a b) : e.normalForm.toPath = e.toPath := by
  induction e with
  | var => rfl
  | call f e ih => simp [normalForm, toPath, ih]
  | letIn e body ih ih' => simp [normalForm, toPath, toPath_subst, ih, ih']

theorem Normal.subst {e : Open S b c} {x : Open S a b} (he : e.Normal)
    (hx : x.Normal) : (e.subst x).Normal := by
  induction he with
  | var => exact hx
  | call f _ ih => exact .call f ih

theorem normal_normalForm (e : Open S a b) : e.normalForm.Normal := by
  induction e with
  | var a => exact .var a
  | call f _ ih => exact .call f ih
  | letIn _ _ ih ih' => exact ih'.subst ih

theorem Normal.normalForm_eq {e : Open S a b} (h : e.Normal) : e.normalForm = e := by
  induction h with
  | var => rfl
  | call f _ ih => simp [normalForm, ih]

theorem normalForm_normalForm (e : Open S a b) :
    e.normalForm.normalForm = e.normalForm :=
  (normal_normalForm e).normalForm_eq

theorem Normal.spine_toPath {e : Open S a b} (h : e.Normal) : e.toPath.spine = e := by
  induction h with
  | var => rfl
  | call f _ ih => simp [toPath, Path.spine, ih]

/-- Completeness: the normal form of a term is the spine of its arrow. -/
theorem normalForm_eq_spine (e : Open S a b) : e.normalForm = e.toPath.spine := by
  rw [← (normal_normalForm e).spine_toPath, toPath_normalForm]

end Open

namespace Path

theorem toPath_spine (p : Path S a b) : p.spine.toPath = p := by
  induction p with
  | nil => rfl
  | cons f p ih => simp [spine, Open.toPath, ih]

theorem normal_spine (p : Path S a b) : p.spine.Normal := by
  induction p with
  | nil => exact .var _
  | cons f _ ih => exact .call f ih

/-- Reading is a section of evaluation. -/
theorem toPath_read (p : Path S a b) : p.read.toPath = p := toPath_spine p

end Path

theorem Open.normal_iff (e : Open S a b) : e.Normal ↔ ∃ p : Path S a b, e = p.spine :=
  ⟨fun h => ⟨e.toPath, h.spine_toPath.symm⟩, fun ⟨p, h⟩ => h ▸ p.normal_spine⟩

/-- Two terms have one normal form exactly when they have one arrow. -/
theorem Open.normalForm_eq_iff (e e' : Open S a b) :
    e.normalForm = e'.normalForm ↔ e.toPath = e'.toPath := by
  constructor
  · intro h; rw [← toPath_normalForm e, h, toPath_normalForm]
  · intro h; rw [normalForm_eq_spine, normalForm_eq_spine, h]

namespace Closed

theorem toPath_apply (t : Closed S a b) (e : Open S c a) :
    (t.apply e).toPath = e.toPath.comp t.toPath := by
  cases t <;> rfl

theorem toPath_normalForm (t : Closed S a b) : t.normalForm.toPath = t.toPath := by
  cases t with
  | const => rfl
  | kappa e => exact e.toPath_normalForm

theorem normalForm_normalForm (t : Closed S a b) :
    t.normalForm.normalForm = t.normalForm := by
  cases t with
  | const => rfl
  | kappa e => exact congrArg kappa e.normalForm_normalForm

/-- Eta expansion on the nose. -/
theorem toPath_expand (t : Closed S a b) : t.expand.toPath = t.toPath := by
  simp [expand, Closed.toPath, toPath_apply, Open.toPath]

/-- Completeness: the normal form of the eta expansion of a term is the
reading of its arrow. -/
theorem normalForm_expand (t : Closed S a b) : t.expand.normalForm = t.toPath.read := by
  show kappa _ = kappa _
  rw [Open.normalForm_eq_spine, ← toPath_expand]; rfl

end Closed

/-- Completeness for a term with a free variable: the normal form of its
kappa is the reading of its arrow. -/
theorem Open.normalForm_kappa (e : Open S a b) :
    (Closed.kappa e).normalForm = e.toPath.read :=
  congrArg Closed.kappa e.normalForm_eq_spine

/-! ## Evaluation into any category -/

section Eval

universe u v
variable (C : Premonoidal.Category.{u, v}) (ob : S.Ob → C.Ob) (ar : S.Box → C.Hom)

/-- A path goes to the composite of the images of its boxes. -/
def Path.eval : {a b : S.Ob} → Path S a b → C.Hom
  | _, _, .nil x => C.id (ob x)
  | _, _, .cons f p => C.comp p.eval (ar f)

@[simp] theorem Path.eval_nil (x : S.Ob) :
    (Path.nil x).eval C ob ar = C.id (ob x) := by simp [Path.eval]

@[simp] theorem Path.eval_cons (f : S.Box) (p : Path S a (S.dom f)) :
    (Path.cons f p).eval C ob ar = C.comp (p.eval C ob ar) (ar f) := by
  simp [Path.eval]

/-- The image of a term under a functor, `TermBase.eval(functor)`. -/
def Open.eval : {a b : S.Ob} → Open S a b → C.Hom
  | _, _, .var x => C.id (ob x)
  | _, _, .call f e => C.comp e.eval (ar f)
  | _, _, .letIn e body => C.comp e.eval body.eval

/-- The image of a term with a stack under a functor. -/
def Closed.eval {a b : S.Ob} : Closed S a b → C.Hom
  | .const f => ar f
  | .kappa e => e.eval C ob ar

variable (hdom : ∀ f, C.dom (ar f) = ob (S.dom f))
  (hcod : ∀ f, C.cod (ar f) = ob (S.cod f))
include hdom hcod

theorem Path.dom_cod_eval (p : Path S a b) :
    C.dom (p.eval C ob ar) = ob a ∧ C.cod (p.eval C ob ar) = ob b := by
  induction p with
  | nil => rw [eval_nil]; exact ⟨C.dom_id _, C.cod_id _⟩
  | cons f p ih =>
    have h : C.cod (p.eval C ob ar) = C.dom (ar f) := by rw [ih.2, hdom]
    exact ⟨by rw [eval_cons, C.dom_comp _ _ h, ih.1],
      by rw [eval_cons, C.cod_comp _ _ h, hcod]⟩

theorem Path.eval_comp (p : Path S a b) (q : Path S b c) :
    (p.comp q).eval C ob ar = C.comp (p.eval C ob ar) (q.eval C ob ar) := by
  induction q with
  | nil =>
    have h := C.comp_id (p.eval C ob ar)
    rw [(p.dom_cod_eval C ob ar hdom hcod).2] at h
    rw [comp_nil, eval_nil, h]
  | cons f q ih =>
    have pq : C.cod (p.eval C ob ar) = C.dom (q.eval C ob ar) := by
      rw [(p.dom_cod_eval C ob ar hdom hcod).2, (q.dom_cod_eval C ob ar hdom hcod).1]
    have qf : C.cod (q.eval C ob ar) = C.dom (ar f) := by
      rw [(q.dom_cod_eval C ob ar hdom hcod).2, hdom]
    simp only [comp_cons, eval_cons, ih, C.assoc _ _ _ pq qf]

omit hdom hcod in
/-- `Path` is the free category: a map of paths that preserves identities
and composition and agrees with `ar` on the boxes is `Path.eval`. -/
theorem Path.eval_unique (F : ∀ {a b : S.Ob}, Path S a b → C.Hom)
    (map_nil : ∀ a, F (.nil a) = C.id (ob a))
    (map_comp : ∀ {a b c : S.Ob} (p : Path S a b) (q : Path S b c),
      F (p.comp q) = C.comp (F p) (F q))
    (map_box : ∀ f, F (.cons f (.nil _)) = ar f) (p : Path S a b) :
    F p = p.eval C ob ar := by
  induction p with
  | nil => rw [eval_nil]; exact map_nil _
  | cons f p ih =>
    show F (p.comp (.cons f (.nil _))) = _
    rw [map_comp, ih, map_box, eval_cons]

/-- A term evaluates as its arrow does: evaluation factors through the free
category. -/
theorem Open.eval_eq (e : Open S a b) : e.eval C ob ar = e.toPath.eval C ob ar := by
  induction e with
  | var => simp [eval, toPath]
  | call f e ih => simp [eval, toPath, ih]
  | letIn e body ih ih' =>
    simp [eval, toPath, Path.eval_comp C ob ar hdom hcod, ih, ih']

theorem Closed.eval_eq (t : Closed S a b) : t.eval C ob ar = t.toPath.eval C ob ar := by
  cases t with
  | const f =>
    have h := C.id_comp (ar f)
    rw [hdom] at h
    simp [eval, toPath, h]
  | kappa e => exact e.eval_eq C ob ar hdom hcod

/-- Beta reduction is sound under every functor. -/
theorem Open.eval_normalForm (e : Open S a b) :
    e.normalForm.eval C ob ar = e.eval C ob ar := by
  rw [eval_eq C ob ar hdom hcod, eval_eq C ob ar hdom hcod, toPath_normalForm]

end Eval

end Terms.Cat
