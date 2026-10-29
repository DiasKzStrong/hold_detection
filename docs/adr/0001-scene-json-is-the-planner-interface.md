# Scene JSON is the planner interface

The route planner consumes a versioned `scene.json`, while GLB remains a
geometric exchange and visualization artifact. Encoding planner semantics only
in GLB nodes would make validation and schema evolution difficult, whereas JSON
keeps observations, confidence, review state, and future 3D properties explicit.
