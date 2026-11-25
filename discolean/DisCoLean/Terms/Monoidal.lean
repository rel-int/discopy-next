import DisCoLean.Premonoidal.Diagram

/-!
# The terms of `discopy.monoidal`

The planar linear kappa calculus of `discopy.monoidal.TermBase`, restricted to
terms with no stack and to boxes with one output: a term `Open S Γ b` has the
list of wires `Γ` as context, each variable used once and in order, and one
output `b`. It is nameless, a variable being where it stands in the context,
so that alpha-equivalent terms are equal. A term is built from

- `var x`, a variable, the identity on one wire
- `call f args`, the box `f` applied to one term for each of its inputs, the
  contexts of the arguments side by side, i.e. `Constant(f)(*args)`
- `letIn e body`, `let x = e in body`, where `x` stands between `Γ₁` and
  `Γ₂` in the context of `body` and the context of `e` takes its place

and evaluates into Diagram(S) as `TermBase.eval()` does with no functor, the
arguments of a call from the last one to the first, as the Python does.

Reading a diagram with one output gives a `let` for each box, so that the term
evaluates back to the diagram, box for box: the Python gives no `let` to the
last box, whose output is the output of the term, which changes nothing of
its diagram.

The model mirrors `discopy/monoidal.py` as of `b66485d`.
-/

namespace Terms.Monoidal
open Premonoidal

/-- A monoidal signature whose boxes have one output each. -/
structure Signature where
  Ob : Type
  Box : Type
  dom : Box → List Ob
  cod : Box → Ob

/-- The signature of the diagrams of the boxes, each of one output. -/
abbrev Signature.diagrams (S : Signature) : Premonoidal.Signature :=
  ⟨S.Ob, S.Box, S.dom, fun f => [S.cod f]⟩

mutual
/-- `Γ ⊢ e : b`, a term with no stack. -/
inductive Open (S : Signature) : List S.Ob → S.Ob → Type
  | var (x : S.Ob) : Open S [x] x
  | call {Γ : List S.Ob} (f : S.Box) : Args S Γ (S.dom f) → Open S Γ (S.cod f)
  | letIn {Γ₁ Γ₂ Δ : List S.Ob} {a b : S.Ob} :
      Open S Δ a → Open S (Γ₁ ++ [a] ++ Γ₂) b → Open S (Γ₁ ++ Δ ++ Γ₂) b

/-- A list of terms, one for each wire of `Δ`, their contexts side by side. -/
inductive Args (S : Signature) : List S.Ob → List S.Ob → Type
  | nil : Args S [] []
  | cons {Γ Γ' : List S.Ob} {a : S.Ob} {Δ : List S.Ob} :
      Open S Γ a → Args S Γ' Δ → Args S (Γ ++ Γ') (a :: Δ)
end

variable {S : Signature} [DecidableEq S.Ob]

mutual
/-- The diagram of a term: a variable is an identity, a call evaluates its
arguments then its box, a `let` its expression, between the wires of `Γ₁`
and `Γ₂`, then its body. -/
def Open.eval : {Γ : List S.Ob} → {b : S.Ob} → Open S Γ b → Diagram S.diagrams
  | _, _, .var x => Diagram.id [x]
  | _, _, .call f args => args.eval.comp (Diagram.box (S := S.diagrams) f)
  | _, _, @Open.letIn _ Γ₁ Γ₂ _ _ _ e body =>
    (Diagram.whiskerLeft Γ₁ (e.eval.whiskerRight Γ₂)).comp body.eval

/-- The diagram of arguments side by side, the last one first. -/
def Args.eval : {Γ Δ : List S.Ob} → Args S Γ Δ → Diagram S.diagrams
  | _, _, .nil => Diagram.id []
  | _, _, @Args.cons _ Γ _ _ Δ e rest =>
    (Diagram.whiskerLeft Γ rest.eval).comp (e.eval.whiskerRight Δ)
end

mutual
/-- A term evaluates to a diagram from its context to its output. -/
theorem Open.boundary : ∀ {Γ : List S.Ob} {b : S.Ob} (e : Open S Γ b),
    e.eval.dom = Γ ∧ e.eval.cod = [b]
  | _, _, .var _ => ⟨rfl, rfl⟩
  | _, _, .call f args => by
    have h := args.boundary
    have hc : args.eval.cod = (Diagram.box (S := S.diagrams) f).dom := h.2
    simp only [Open.eval]
    exact ⟨by rw [Diagram.dom_comp hc, h.1], by rw [Diagram.cod_comp hc]; rfl⟩
  | _, _, @Open.letIn _ Γ₁ Γ₂ _ _ _ e body => by
    have h := e.boundary
    have h' := body.boundary
    have hc : (Diagram.whiskerLeft Γ₁ (e.eval.whiskerRight Γ₂)).cod = body.eval.dom := by
      rw [Diagram.cod_whiskerLeft, Diagram.cod_whiskerRight, h.2, h'.1,
        List.append_assoc]
    simp only [Open.eval]
    refine ⟨?_, by rw [Diagram.cod_comp hc, h'.2]⟩
    rw [Diagram.dom_comp hc]
    simp [Diagram.whiskerLeft, Diagram.whiskerRight, h.1]

/-- Arguments evaluate to a diagram from their contexts to their outputs. -/
theorem Args.boundary : ∀ {Γ Δ : List S.Ob} (args : Args S Γ Δ),
    args.eval.dom = Γ ∧ args.eval.cod = Δ
  | _, _, .nil => ⟨rfl, rfl⟩
  | _, _, @Args.cons _ Γ _ _ Δ e rest => by
    have h := e.boundary
    have h' := rest.boundary
    have hc : (Diagram.whiskerLeft Γ rest.eval).cod = (e.eval.whiskerRight Δ).dom := by
      rw [Diagram.cod_whiskerLeft, h'.2]
      show Γ ++ Δ = e.eval.dom ++ Δ
      rw [h.1]
    simp only [Args.eval]
    refine ⟨?_, ?_⟩
    · rw [Diagram.dom_comp hc]; simp [Diagram.whiskerLeft, h'.1]
    · rw [Diagram.cod_comp hc, Diagram.cod_whiskerRight, h.2]; rfl
end

/-! ## Reading a diagram -/

/-- A variable for each wire, in order. -/
def vars : (X : List S.Ob) → Args S X X
  | [] => .nil
  | x :: X => .cons (.var x) (vars X)

theorem id_comp_id (X : List S.Ob) :
    (Diagram.id (S := S.diagrams) X).comp (Diagram.id X) = Diagram.id X := by
  have h : (Diagram.id (S := S.diagrams) X).cod = (Diagram.id X).dom := rfl
  ext
  · rw [Diagram.dom_comp h]
  · rw [Diagram.layers_comp h]; rfl

/-- Variables side by side are the identity. -/
theorem eval_vars (X : List S.Ob) : (vars (S := S) X).eval = Diagram.id X := by
  induction X with
  | nil => rfl
  | cons x X ih =>
    simp only [vars, Args.eval, Open.eval, ih]
    exact id_comp_id (x :: X)

/-- Eta for a box: a box applied to a variable for each of its inputs
evaluates to the box. -/
theorem eval_call_vars (f : S.Box) :
    (Open.call f (vars (S := S) (S.dom f))).eval = Diagram.box (S := S.diagrams) f := by
  simp only [Open.eval, eval_vars]
  have h : (Diagram.id (S := S.diagrams) (S.dom f)).cod
      = (Diagram.box (S := S.diagrams) f).dom := rfl
  ext
  · rw [Diagram.dom_comp h]; rfl
  · rw [Diagram.layers_comp h]; rfl

/-- The term of layers starting from `X` and ending on one wire: a `let` of
the box of each layer, applied to the variables of its inputs. -/
def read {b : S.Ob} : (X : List S.Ob) → (ls : List (Layer S.diagrams)) →
    Composable X ls → codFrom X ls = [b] → Open S X b
  | _, [], _, rfl => .var b
  | _, l :: ls, ⟨rfl, hc⟩, h =>
    .letIn (Γ₁ := l.left) (Γ₂ := l.right) (.call l.box (vars _)) (read l.cod ls hc h)

/-- The term of a diagram with one output, `Diagram.to_term`. -/
def toTerm {b : S.Ob} (d : Diagram S.diagrams) (h : d.cod = [b]) : Open S d.dom b :=
  Terms.Monoidal.read d.dom d.layers d.composable h

theorem eval_read {b : S.Ob} : ∀ (X : List S.Ob) (ls : List (Layer S.diagrams))
    (hc : Composable X ls) (h : codFrom X ls = [b]),
    (read X ls hc h).eval = ⟨X, ls, hc⟩
  | _, [], _, rfl => rfl
  | _, l :: ls, ⟨rfl, hc⟩, h => by
    show (Diagram.whiskerLeft l.left ((Open.call l.box (vars _)).eval.whiskerRight
      l.right)).comp (read l.cod ls hc h).eval = _
    rw [eval_call_vars, eval_read l.cod ls hc h]
    have hd : (Diagram.whiskerLeft l.left
        ((Diagram.box (S := S.diagrams) l.box).whiskerRight l.right)).cod
        = (⟨l.cod, ls, hc⟩ : Diagram S.diagrams).dom := by
      rw [Diagram.cod_whiskerLeft, Diagram.cod_whiskerRight]
      show l.left ++ ([S.cod l.box] ++ l.right) = l.left ++ [S.cod l.box] ++ l.right
      rw [List.append_assoc]
    apply Diagram.ext
    · rw [Diagram.dom_comp hd]
      simp [Diagram.whiskerLeft, Diagram.whiskerRight, Diagram.box, Layer.dom]
    · rw [Diagram.layers_comp hd]
      simp [Diagram.whiskerLeft, Diagram.whiskerRight, Diagram.box,
        Layer.whiskerLeft, Layer.whiskerRight]

/-- Reading is a section of evaluation: a diagram with one output reads as a
term that evaluates back to it, box for box. -/
theorem eval_toTerm {b : S.Ob} (d : Diagram S.diagrams) (h : d.cod = [b]) :
    (toTerm d h).eval = d :=
  Terms.Monoidal.eval_read d.dom d.layers d.composable h

end Terms.Monoidal
