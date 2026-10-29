# Route perception demo

The demo is an offline-first vertical slice from a wall capture to reviewable
route candidates and metric RGB-D spatial evidence. It does not claim that the
capture coordinate frame is registered to the GLB.

## Colab run

1. Open `notebooks/DeMoSearch_SAM3_Colab.ipynb` in a GPU runtime.
2. Add an approved Hugging Face read token as the `HF_TOKEN` Colab secret.
3. Select `drive` input and point `DRIVE_INPUT` at a photo directory or ZIP.
4. Leave grip classification disabled; the notebook no longer exposes it in the main flow.
5. Run YOLO first, then SAM 3. Start with three images.
6. Download the offline report before presenting.

Every inference run contains:

- `scene.json`: draft planner input with frame-local observations and route candidates;
- `index.html`: portable review gallery;
- `<frame>/routes.jpg`: color-grouped route visualization;
- `<frame>/holds.json`: full frame-local evidence;

SAM 3 runs additionally contain `<frame>/masks/` with lossless instance masks.
Separate repeatable prompts find climbing holds and climbing volumes; cross-prompt
NMS removes duplicate instances.

The route candidates use mask median color transformed to CIE Lab and a default
Delta E 76 threshold of 22. They are diagnostics requiring human review, not
measured route accuracy. `physical_holds` remains empty for ordinary photos and
box-only runs.

Color is not a unique route identifier. In the collected dataset, wall A has two
green routes. The pipeline therefore emits color-based `route_candidates`; route
notes or a human selection must resolve final route membership.

When matching Polycam camera/depth files are present, each masked observation is
unprojected into metric camera and capture coordinates. Observations with the
same route-color candidate are fused using sampled 3D point overlap plus a
bounded center distance. Confirmed `physical_holds` provide
`position_3d_m`, `estimated_size_m`, and `protrusion_m`. These candidates require
review. The scene records `glb_registration = not_registered`; it never presents
capture coordinates as GLB coordinates without a verified transform.

When masks are unavailable, the YOLO fallback estimates color from each bounding
box and records `bbox_median_rgb_fallback`. This is presentation-safe but less
reliable because wall pixels contaminate the estimate; every such observation is
forced into review.

## Offline fallback

The current demo capture contains 38 matched RGB/camera/depth frames. Prepare it
and run three frames locally with:

```bash
uv run holds prepare-polycam data/demo/wallA_1_raw_data.zip \
  --output data/demo/wallA_1
uv run holds fetch-models --model detector
uv run holds predict data/demo/wallA_1/keyframes/corrected_images \
  --output runs/wallA_demo_yolo --device cpu --rotate 90 --limit 3 --conf 0.25
```

Open `runs/wallA_demo_yolo/index.html`. The source data, model checkpoints, and
run artifacts are ignored by Git and must not be committed.
