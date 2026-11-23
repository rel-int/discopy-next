import DisCoLean.ABC
import DisCoLean.Premonoidal.Functor

/-!
# The coherence of `discopy.monoidal`

`DisCoLean.ABC.monoidal` is the theory of `monoidal.Diagram`: the laws of
`abc.MonoidalCategory` but bifunctoriality, which the Python declares broken.
Its models are the strict premonoidal categories: `toABC` makes a strict
premonoidal category a model, with the tensor of two morphisms the first one
first, and `ofABC` makes a model a strict premonoidal category, with
whiskering the tensor with an identity.

The coherence theorem `Diagram.free` follows: Diagram(S) is the free model,
i.e. a functor of `abc.MonoidalCategory` from Diagram(S) to a model is given
by where it sends the generating objects and the boxes, and by nothing else.
-/

universe u v

namespace Premonoidal.StrictPremonoidal

variable (C : StrictPremonoidal)

theorem tensorHom_eq (f g : C.Hom) : C.tensorHom f g
    = C.comp (C.whiskerRight f (C.dom g)) (C.whiskerLeft (C.cod f) g) := rfl

theorem composable_tensorHom (f g : C.Hom) :
    C.cod (C.whiskerRight f (C.dom g)) = C.dom (C.whiskerLeft (C.cod f) g) := by
  rw [C.cod_whiskerRight, C.dom_whiskerLeft]

theorem dom_tensorHom (f g : C.Hom) :
    C.dom (C.tensorHom f g) = C.tensor (C.dom f) (C.dom g) := by
  rw [tensorHom, C.dom_comp _ _ (C.composable_tensorHom f g), C.dom_whiskerRight]

theorem cod_tensorHom (f g : C.Hom) :
    C.cod (C.tensorHom f g) = C.tensor (C.cod f) (C.cod g) := by
  rw [tensorHom, C.cod_comp _ _ (C.composable_tensorHom f g), C.cod_whiskerLeft]

theorem id_tensorHom (X : C.Ob) (f : C.Hom) :
    C.tensorHom (C.id X) f = C.whiskerLeft X f := by
  have h := C.id_comp (C.whiskerLeft X f)
  rw [C.dom_whiskerLeft] at h
  rw [tensorHom, C.id_whiskerRight, C.cod_id, h]

theorem tensorHom_id (f : C.Hom) (X : C.Ob) :
    C.tensorHom f (C.id X) = C.whiskerRight f X := by
  have h := C.comp_id (C.whiskerRight f X)
  rw [C.cod_whiskerRight] at h
  rw [tensorHom, C.whiskerLeft_id, C.dom_id, h]

theorem tensorHom_assoc (f g h : C.Hom) :
    C.tensorHom (C.tensorHom f g) h = C.tensorHom f (C.tensorHom g h) := by
  have first : C.cod (C.whiskerRight f (C.tensor (C.dom g) (C.dom h)))
      = C.dom (C.whiskerLeft (C.cod f) (C.whiskerRight g (C.dom h))) := by
    rw [C.cod_whiskerRight, C.dom_whiskerLeft, C.dom_whiskerRight]
  have second : C.cod (C.whiskerLeft (C.cod f) (C.whiskerRight g (C.dom h)))
      = C.dom (C.whiskerLeft (C.cod f) (C.whiskerLeft (C.cod g) h)) := by
    rw [C.cod_whiskerLeft, C.dom_whiskerLeft, C.cod_whiskerRight, C.dom_whiskerLeft]
  rw [C.tensorHom_eq (C.tensorHom f g) h, cod_tensorHom, C.tensorHom_eq f g,
    C.comp_whiskerRight _ _ _ (C.composable_tensorHom f g), ← C.whiskerRight_tensor,
    C.whisker_assoc, C.tensor_whiskerLeft, C.tensorHom_eq f (C.tensorHom g h), dom_tensorHom,
    C.tensorHom_eq g h, C.whiskerLeft_comp _ _ _ (C.composable_tensorHom g h),
    C.assoc _ _ _ first second]

/-- A strict premonoidal category as an `abc.MonoidalCategory`, the tensor
of morphisms `f ⋉ dom g` then `cod f ⋊ g`. -/
def toABC : DisCoLean.ABC.MonoidalCategory C.Ob C.Hom where
  dom := C.dom
  cod := C.cod
  id := C.id
  comp := C.comp
  ax X := ⟨C.dom_id X, C.cod_id X⟩
  cut _ _ _ f g dom_f cod_f dom_g cod_g :=
    ⟨(C.dom_comp f g (cod_f.trans dom_g.symm)).trans dom_f,
     (C.cod_comp f g (cod_f.trans dom_g.symm)).trans cod_g⟩
  tensor := C.tensorHom
  tensorOb := C.tensor
  unitOb := C.unit
  mix _ _ _ _ f g dom_f cod_f dom_g cod_g := by
    rw [C.dom_tensorHom, C.cod_tensorHom, dom_f, cod_f, dom_g, cod_g]
    exact ⟨rfl, rfl⟩

/-- A strict premonoidal category has the laws of `monoidal.Diagram`. -/
theorem toABC_monoidal : DisCoLean.ABC.monoidal C.toABC := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact fun _ _ f _ _ => ⟨C.id_comp f, (C.comp_id f).symm⟩
  · exact fun _ _ _ _ f g h _ cod_f dom_g cod_g dom_h _ =>
      C.assoc f g h (cod_f.trans dom_g.symm) (cod_g.trans dom_h.symm)
  · exact fun X Y => (C.id_tensorHom X (C.id Y)).trans (C.whiskerLeft_id X Y)
  · exact fun _ _ _ _ _ _ f g h _ _ _ _ _ _ => C.tensorHom_assoc f g h
  · exact fun _ _ f _ _ => ⟨(C.id_tensorHom _ f).trans (C.unit_whiskerLeft f),
      ((C.tensorHom_id f _).trans (C.whiskerRight_unit f)).symm⟩
  · intro X _ _ _ f g _ cod_f dom_g _
    show C.tensorHom (C.id X) (C.comp f g)
      = C.comp (C.tensorHom (C.id X) f) (C.tensorHom (C.id X) g)
    rw [C.id_tensorHom, C.id_tensorHom, C.id_tensorHom,
      C.whiskerLeft_comp _ _ _ (cod_f.trans dom_g.symm)]
  · intro X _ _ _ f g _ cod_f dom_g _
    show C.tensorHom (C.comp f g) (C.id X)
      = C.comp (C.tensorHom f (C.id X)) (C.tensorHom g (C.id X))
    rw [C.tensorHom_id, C.tensorHom_id, C.tensorHom_id,
      C.comp_whiskerRight _ _ _ (cod_f.trans dom_g.symm)]
  · intro _ _ _ _ f g _ _ _ _
    show C.tensorHom f g = C.comp (C.tensorHom f (C.id (C.dom g)))
      (C.tensorHom (C.id (C.cod f)) g)
    rw [C.tensorHom_id, C.id_tensorHom, tensorHom]

end Premonoidal.StrictPremonoidal

namespace DisCoLean.ABC.MonoidalCategory

variable {Ob : Type u} {Hom : Type v} (M : MonoidalCategory Ob Hom)

/-- Two objects with the same identity are equal. -/
theorem ob_eq {X Y : Ob} (h : M.id X = M.id Y) : X = Y :=
  (M.ax X).1.symm.trans ((congrArg M.dom h).trans (M.ax Y).1)

theorem dom_tensor_id (X : Ob) (f : Hom) :
    M.dom (M.tensor (M.id X) f) = M.tensorOb X (M.dom f) :=
  (M.mix _ _ _ _ _ f (M.ax X).1 (M.ax X).2 rfl rfl).1

theorem cod_tensor_id (X : Ob) (f : Hom) :
    M.cod (M.tensor (M.id X) f) = M.tensorOb X (M.cod f) :=
  (M.mix _ _ _ _ _ f (M.ax X).1 (M.ax X).2 rfl rfl).2

theorem dom_id_tensor (f : Hom) (X : Ob) :
    M.dom (M.tensor f (M.id X)) = M.tensorOb (M.dom f) X :=
  (M.mix _ _ _ _ f _ rfl rfl (M.ax X).1 (M.ax X).2).1

theorem cod_id_tensor (f : Hom) (X : Ob) :
    M.cod (M.tensor f (M.id X)) = M.tensorOb (M.cod f) X :=
  (M.mix _ _ _ _ f _ rfl rfl (M.ax X).1 (M.ax X).2).2

variable {M} (laws : monoidal M)
include laws

theorem tensor_id_id (X Y : Ob) :
    M.tensor (M.id X) (M.id Y) = M.id (M.tensorOb X Y) := laws.2.2.1 X Y

theorem tensor_assoc (f g h : Hom) :
    M.tensor (M.tensor f g) h = M.tensor f (M.tensor g h) :=
  laws.2.2.2.1 _ _ _ _ _ _ f g h rfl rfl rfl rfl rfl rfl

theorem tensor_unit_left (f : Hom) : M.tensor (M.id M.unitOb) f = f :=
  (laws.2.2.2.2.1 _ _ f rfl rfl).1

theorem tensor_unit_right (f : Hom) : M.tensor f (M.id M.unitOb) = f :=
  (laws.2.2.2.2.1 _ _ f rfl rfl).2.symm

/-- A model of the theory of `monoidal.Diagram` as a strict premonoidal
category, whiskering being the tensor with an identity. -/
def ofABC : Premonoidal.StrictPremonoidal.{u, v} where
  Ob := Ob
  Hom := Hom
  dom := M.dom
  cod := M.cod
  id := M.id
  comp := M.comp
  dom_id X := (M.ax X).1
  cod_id X := (M.ax X).2
  dom_comp f g h := (M.cut _ _ _ f g rfl rfl h.symm rfl).1
  cod_comp f g h := (M.cut _ _ _ f g rfl rfl h.symm rfl).2
  id_comp f := (laws.1 _ _ f rfl rfl).1
  comp_id f := (laws.1 _ _ f rfl rfl).2.symm
  assoc f g h fg gh := laws.2.1 _ _ _ _ f g h rfl rfl fg.symm rfl gh.symm rfl
  unit := M.unitOb
  tensor := M.tensorOb
  unit_tensor X := M.ob_eq <| (tensor_id_id laws _ X).symm.trans (tensor_unit_left laws _)
  tensor_unit X := M.ob_eq <| (tensor_id_id laws X _).symm.trans (tensor_unit_right laws _)
  tensor_assoc X Y Z := M.ob_eq <| by
    rw [← tensor_id_id laws, ← tensor_id_id laws, ← tensor_id_id laws,
      ← tensor_id_id laws, tensor_assoc laws]
  whiskerLeft X f := M.tensor (M.id X) f
  whiskerRight f X := M.tensor f (M.id X)
  dom_whiskerLeft := M.dom_tensor_id
  cod_whiskerLeft := M.cod_tensor_id
  dom_whiskerRight := M.dom_id_tensor
  cod_whiskerRight := M.cod_id_tensor
  whiskerLeft_id := tensor_id_id laws
  id_whiskerRight := tensor_id_id laws
  whiskerLeft_comp X f g h := laws.2.2.2.2.2.1 X _ _ _ f g rfl rfl h.symm rfl
  comp_whiskerRight f g X h := laws.2.2.2.2.2.2.1 X _ _ _ f g rfl rfl h.symm rfl
  unit_whiskerLeft := tensor_unit_left laws
  whiskerRight_unit := tensor_unit_right laws
  tensor_whiskerLeft X Y f := by rw [← tensor_id_id laws, tensor_assoc laws]
  whiskerRight_tensor f X Y := by rw [← tensor_id_id laws, tensor_assoc laws]
  whisker_assoc X f Y := tensor_assoc laws _ _ _

/-- In a model, the tensor of two morphisms is the first one first. -/
theorem tensorHom_ofABC (f g : Hom) : (ofABC laws).tensorHom f g = M.tensor f g :=
  (laws.2.2.2.2.2.2.2 _ _ _ _ f g rfl rfl rfl rfl).symm

end DisCoLean.ABC.MonoidalCategory

namespace Premonoidal.Diagram

open DisCoLean.ABC

variable (S : Signature) [DecidableEq S.Ob]

/-- Diagram(S) has the laws of `monoidal.Diagram`. -/
theorem monoidal : DisCoLean.ABC.monoidal (premonoidal S).toABC :=
  (premonoidal S).toABC_monoidal

variable {S} {Ob : Type u} {Hom : Type v} {M : MonoidalCategory Ob Hom}

/-- A functor of `abc.MonoidalCategory` from Diagram(S) to a model is a
strict premonoidal functor. -/
def Functor.toPremonoidal (laws : DisCoLean.ABC.monoidal M)
    (F : MonoidalCategory.Functor (premonoidal S).toABC M) :
    Premonoidal.Functor (premonoidal S) (MonoidalCategory.ofABC laws) where
  ob := F.ob
  map := F.map
  dom_map f := (F.ob_dom f).symm
  cod_map f := (F.ob_cod f).symm
  map_id := F.map_id
  map_comp f g h := F.map_comp _ _ _ f g rfl rfl h.symm rfl
  ob_unit := F.ob_unitOb
  ob_tensor := F.ob_tensorOb
  map_whiskerLeft X f := by
    have h := F.map_tensor _ _ _ _ ((premonoidal S).id X) f rfl rfl rfl rfl
    rw [show (premonoidal S).toABC.tensor = (premonoidal S).tensorHom from rfl,
      StrictPremonoidal.id_tensorHom,
      show F.map ((premonoidal S).id X) = M.id (F.ob X) from F.map_id X] at h
    exact h
  map_whiskerRight f X := by
    have h := F.map_tensor _ _ _ _ f ((premonoidal S).id X) rfl rfl rfl rfl
    rw [show (premonoidal S).toABC.tensor = (premonoidal S).tensorHom from rfl,
      StrictPremonoidal.tensorHom_id,
      show F.map ((premonoidal S).id X) = M.id (F.ob X) from F.map_id X] at h
    exact h

/-- **Coherence of `discopy.monoidal`**: Diagram(S) is the free model of the
theory of `monoidal.Diagram`. A functor of `abc.MonoidalCategory` to a model
that sends each generating object and each box where given exists, and it
is unique. -/
theorem free (laws : DisCoLean.ABC.monoidal M) (ob : S.Ob → Ob) (ar : S.Box → Hom)
    (hdom : ∀ b, M.dom (ar b) = Lift.obs (MonoidalCategory.ofABC laws) ob (S.dom b))
    (hcod : ∀ b, M.cod (ar b) = Lift.obs (MonoidalCategory.ofABC laws) ob (S.cod b)) :
    ∃ F : MonoidalCategory.Functor (premonoidal S).toABC M,
      (∀ x, F.ob [x] = ob x) ∧ (∀ b, F.map (box b) = ar b) ∧
      ∀ G : MonoidalCategory.Functor (premonoidal S).toABC M,
        (∀ x, G.ob [x] = ob x) → (∀ b, G.map (box b) = ar b) → G = F := by
  let C := MonoidalCategory.ofABC laws
  let L := Lift.functor C ob ar hdom hcod
  refine ⟨{
    ob := L.ob
    map := L.map
    map_id := L.map_id
    map_comp := fun _ _ _ f g _ cod_f dom_g _ => L.map_comp f g (cod_f.trans dom_g.symm)
    ob_dom := fun f => (L.dom_map f).symm
    ob_cod := fun f => (L.cod_map f).symm
    map_tensor := fun _ _ _ _ f g _ _ _ _ => by
      have mid : C.comp (L.map ((premonoidal S).whiskerRight f ((premonoidal S).dom g)))
          (L.map ((premonoidal S).whiskerLeft ((premonoidal S).cod f) g))
          = C.tensorHom (L.map f) (L.map g) := by
        rw [L.map_whiskerRight, L.map_whiskerLeft, ← L.dom_map, ← L.cod_map]; rfl
      exact (L.map_comp _ _ ((premonoidal S).composable_tensorHom f g)).trans
        (mid.trans (MonoidalCategory.tensorHom_ofABC laws _ _))
    ob_tensorOb := L.ob_tensor
    ob_unitOb := L.ob_unit }, Lift.obs_singleton C ob, Lift.map_box C ob ar hcod, ?_⟩
  intro G ob_G ar_G
  have h := Premonoidal.Functor.unique (Functor.toPremonoidal laws G) L
    (fun x => (ob_G x).trans (Lift.obs_singleton C ob x).symm)
    (fun b => (ar_G b).trans (Lift.map_box C ob ar hcod b).symm)
  have ob_eq : G.ob = L.ob := congrArg Premonoidal.Functor.ob h
  have map_eq : G.map = L.map := congrArg Premonoidal.Functor.map h
  cases G
  rename_i G _ _ _
  cases G
  cases ob_eq
  cases map_eq
  rfl

end Premonoidal.Diagram
