# Route perception demo

The demo is an offline-first vertical slice from wall photographs to reviewable
route candidates. It does not claim multi-view identity or 3D reconstruction.

## Colab run

1. Open `notebooks/DeMoSearch_SAM3_Colab.ipynb` in a GPU runtime.
2. Add an approved Hugging Face read token as the `HF_TOKEN` Colab secret.
3. Select `drive` input and point `DRIVE_INPUT` at a photo directory or ZIP.
4. Keep `CLASSIFIER = 'none'`; grip classification is outside the current scope.
5. Run YOLO first, then SAM 3. Start with three images.
6. Download the offline report before presenting.

Every inference run contains:

- `scene.json`: draft planner input with frame-local observations and route candidates;
- `index.html`: portable review gallery;
- `<frame>/routes.jpg`: color-grouped route visualization;
- `<frame>/holds.json`: full frame-local evidence;

SAM 3 runs additionally contain `<frame>/masks/` with lossless instance masks.
The current SAM 3 prompt finds climbing holds only; separate Volume detection is
a planned milestone and is not claimed in this demo.

The route candidates use mask median color transformed to CIE Lab and a default
Delta E 76 threshold of 22. They are diagnostics requiring human review, not
measured route accuracy. `physical_holds` remains empty and reconstruction status
is `not_run` until multi-view fusion is implemented.

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
