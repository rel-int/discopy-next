> toumix, in https://ampcode.com/threads/T-48aec84b-09cc-47fa-bf08-e9067aece821:
>
> > so the dagger of a box with args should not be equal to the box with the dagger of its args but it should still be drawn as such, here it's not the case (also there's a weird bug with the two left and right rotations being the same, are you drawing pivotal examples or rigid?)
> > also there's the distinction between arguments drawn side by side or on top of one another, in general we want to keep the internal representation the same but the drawing should be such that the dagger really is the vertical reflection (except for labels ofc) and similarly for rotation

- [x] Draw the args of a box's dagger, transpose and conjugate left to right as the reflection, half turn and reflection of those of the box: slots in reverse order, contents reflected or turned
- [ ] Draw the transposes of a box with args top to bottom as the half turn of the box: args turned, side by side in reverse order
