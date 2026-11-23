import DisCoLean.Terms

/-!
# What is proved of the terms of `discopy.cat`

The model is the kappa calculus of `discopy.cat.TermBase`, nameless since a
context has one variable: `Open S a b` is `x : a ⊢ e : b`, `Closed S a b` is
`⊢ e : a → b`. The definitions are restated by `rfl` and each theorem names
one broken implementation it rules out. `Audit.lean` covers this file. The
model is not linked to the Python by a test: it mirrors `discopy/cat.py`.
-/

namespace Statements.Terms
open _root_.Terms.Cat

variable {S : Signature} {a b c : S.Ob}

/-! ## Definitions -/

/-- A term evaluates to a path: a variable is the identity, a call
composes its argument then its box, a `let` its expression then its body. -/
theorem toPath_def (f : S.Box) (e : Open S a (S.dom f)) (e' : Open S a b)
    (body : Open S b c) :
    (Open.var a).toPath = Path.nil a ∧ (Open.call f e).toPath = .cons f e.toPath
    ∧ (Open.letIn e' body).toPath = e'.toPath.comp body.toPath := ⟨rfl, rfl, rfl⟩

/-- Beta reduction substitutes the expression of each `let` for the
variable of its body. -/
theorem normalForm_def (f : S.Box) (e : Open S a (S.dom f)) (e' : Open S a b)
    (body : Open S b c) :
    (Open.var a).normalForm = .var a ∧ (Open.call f e).normalForm = .call f e.normalForm
    ∧ (Open.letIn e' body).normalForm = body.normalForm.subst e'.normalForm :=
  ⟨rfl, rfl, rfl⟩

/-- An arrow reads as the kappa of the variable under its boxes, and eta
expansion applies a term to its variable under a kappa. -/
theorem read_def (p : Path S a b) (t : Closed S a b) :
    p.read = .kappa p.spine ∧ t.expand = .kappa (t.apply (.var a)) := ⟨rfl, rfl⟩

/-! ## Theorems: the five laws of `cat.TermBase` -/

/-- `eta`: a term and its eta expansion have one arrow.
Rules out: an `expand` that applies the term to a variable of its codomain. -/
theorem eta (t : Closed S a b) : t.expand.toPath = t.toPath := t.toPath_expand

/-- `normal_form_soundness`: a term and its normal form have one arrow.
Rules out: a substitution that puts the expression after the body. -/
theorem normal_form_soundness (e : Open S a b) : e.normalForm.toPath = e.toPath :=
  e.toPath_normalForm

/-- `normal_form_idempotence`.
Rules out: a normal form that contracts only the outermost `let`. -/
theorem normal_form_idempotence (e : Open S a b) :
    e.normalForm.normalForm = e.normalForm := e.normalForm_normalForm

/-- `normal_form_completeness`: the normal form of the eta expansion of a
term is the reading of its arrow, so two terms are equal up to beta and eta
exactly when their arrows are.
Rules out: a normal form that keeps a `let`, e.g. one whose variable shadows
a free one, as the Python's does. -/
theorem normal_form_completeness (t : Closed S a b) (e e' : Open S a b) :
    t.expand.normalForm = t.toPath.read
    ∧ (e.normalForm = e'.normalForm ↔ e.toPath = e'.toPath) :=
  ⟨t.normalForm_expand, Open.normalForm_eq_iff e e'⟩

/-- `reading`: reading is a section of evaluation.
Rules out: a reading that drops the identity boxes of a path. -/
theorem reading (p : Path S a b) : p.read.toPath = p := p.toPath_read

/-- A term in normal form is a variable under a sequence of calls, i.e. a path.
Rules out: a normal form that leaves a `let` behind. -/
theorem normal_iff (e : Open S a b) : e.Normal ↔ ∃ p : Path S a b, e = p.spine :=
  e.normal_iff

/-- Evaluation under any functor, i.e. into any category with images of the
boxes of the right boundary, factors through the path of the term, and so
beta reduction is sound under every functor.
Rules out: an evaluation that composes a `let` body then its expression. -/
theorem eval_factors (C : _root_.Premonoidal.Category) (ob : S.Ob → C.Ob)
    (ar : S.Box → C.Hom) (hdom : ∀ f, C.dom (ar f) = ob (S.dom f))
    (hcod : ∀ f, C.cod (ar f) = ob (S.cod f)) (e : Open S a b) (t : Closed S a b) :
    e.eval C ob ar = e.toPath.eval C ob ar ∧ t.eval C ob ar = t.toPath.eval C ob ar
    ∧ e.normalForm.eval C ob ar = e.eval C ob ar :=
  ⟨e.eval_eq C ob ar hdom hcod, t.eval_eq C ob ar hdom hcod,
    e.eval_normalForm C ob ar hdom hcod⟩

end Statements.Terms
