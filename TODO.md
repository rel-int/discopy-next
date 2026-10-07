In `discopy/pattern.py` at lines 451-454:
```python
    head: str = OBJECTS
    size: int | str | None = None
    count: bool = False
    side: bool = False
```
doesnt make sense to have all four fields for all sorts. just make a Sort class then four subclasses for each possible head, instead of enumerating heads as shitty string toplevel constants like OBJECTS/ARROWS/SELF.

In `discopy/pattern.py` at line 468:
```python
            return cls("bool", side=True)
```
don't use strings to represent a type?? use Self, Count and bool as actual types


switch to the new scheme with Diagram (or Ty)-owned search strategies as opposed to the current external ones. cleanup. I was thinking, would it make sense to use Obj[Self] = Self.ob and Hom[Self, A, B] = Self.ar, instead of Obj[Ty] and Hom[Diagram, A, B]? would it help to avoid the metaprogramming fuckery? would it be still ty-compatible?

## Sorts
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 14:50 `Sort` with one subclass per head — the objects of a class or type parameter, `Self`, `Count` and `bool` — each head an actual type
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 14:50 heads resolved by type: `Self` the category, a type parameter by its position in the class declaring it, a class by the category's own subclass of it; `OBJECTS`, `ARROWS`, `SELF` and `heads` go

## Strategies owned by the category
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 14:50 the search draws objects from `cls.ob.strategy` and free boxes from the category itself: the `types` and `free` parameters go
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 14:50 cleanup: `Declaration.scope`, `name`, the leftovers of the unified search

## Question
- [WIP] @session_019jj2d1JBnEGRPmjqSAP37A-2026-10-07 14:50 probe `Obj[Self]` and `Hom[Self, A, B]` with `ty` and report
