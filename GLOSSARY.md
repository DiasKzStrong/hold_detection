# Climbing Wall Reconstruction

This context describes the physical climbing wall, its captured observations, and
the reconstructed model used to identify routes and match the same holds across
multiple views.

## Physical Wall

**Physical Hold**:
A distinct grip or foothold mounted on the physical climbing wall and preserved
as one identifiable object in the reconstructed wall model.
_Avoid_: Detection, crop, frame-local hold

**Volume**:
A large wall-mounted climbing feature that can support Physical Holds and is
identified separately from them.
_Avoid_: Large hold

**Route**:
A set of Physical Holds intended to be climbed together.
_Avoid_: Color, grip type

**Color Group**:
A set of Physical Holds on one wall that share the same observed route color.
_Avoid_: Route, color name

## Capture And Reconstruction

**Wall Capture**:
The complete set of overlapping views supplied by a user for one physical wall.
_Avoid_: Polycam export, image folder

**Hold Observation**:
The visible evidence for a possible Physical Hold in one view of a Wall Capture.
_Avoid_: Physical Hold, GLB hold

**Reconstructed Wall Model**:
The generated three-dimensional representation of one physical wall in which
each Physical Hold and Volume has a stable identity.
_Avoid_: 3D image, scan

**Hold Correspondence**:
An asserted relationship between a Hold Observation and a Physical Hold in the
Reconstructed Wall Model.
_Avoid_: Classification, color match
