# Initial validation

Input: the user-provided `wallA_1_raw_data.zip`, with 38 matched RGB-D/camera
keyframes. This is one scan of one wall, not 38 independent scenes.

## Real-image inference

The frozen CPU environment was installed and the reference model downloader was
run twice. The second run verified and reused both checkpoints. No credentials
or Polycam service calls were needed.

```bash
holds prepare-polycam /path/to/wallA_1_raw_data.zip --output data/wallA_1
holds predict data/wallA_1/keyframes/corrected_images \
  --output runs/wallA_1-detector --device cpu --rotate 90
```

All 38 frames completed, with 1,814 candidate observations in about 10 seconds
after model initialization on this CPU instance. Observations include repeat
views of the same holds and false positives; this is not a count of unique or
correctly detected holds. 472 observations fell below the default type-score
threshold of 0.5 and were marked `unknown`/for review.

All JSON boxes were checked against their processed-image dimensions, scores
were finite and in [0,1], crop files matched the exported records, and all 38
frame JSON files and annotated images were generated. Without SAM, mask paths
and 3D positions are null. No detection AP or gym-specific type accuracy is
available because these frames have not been labeled.

Additional runs completed on the first three sorted frames:

| Mode | Observations | Marked for review |
| --- | ---: | ---: |
| Whole-image YOLO, confidence 0.35 | 130 | 33 |
| Whole-image + 640px overlapping tiles | 156 | 45 |
| Whole-image YOLO boxes + crop CNN | 130 | 93 |

More detections do not establish better recall without labels. Visual review
found route tags detected as holds, overly broad boxes, and type disagreements.
The tiled path is available but is not the default. Review in the CNN mode
includes disagreement with the detector, even when the CNN softmax is high.

## CNN evaluation on upstream samples

```bash
holds evaluate-crops /path/to/reference/Sample_Processed_Dataset/test \
  --output runs/cnn-upstream-samples.json
```

The provided 100 labeled test-sample crops produced **69/100 correct** and
**macro-F1 0.5149** over all six classes. Supports were 57 crimp, 26 jug, 8 pinch,
3 pocket, 4 sloper, and 2 volume. This small, imbalanced sample comes from the
checkpoint's own repository; its independence from training has not been
verified. These results must not be reported as performance on our gym.

## Automated checks and remaining limitation

`pytest -q`: **15 passed**. A second frozen dependency sync made no changes and
the same tests passed again. `git diff --check` passed.

The optional SAM 2 adapter's export contract is tested using a controlled mask,
but actual SAM weights/inference were not validated. The official checkpoint
host `dl.fbaipublicfiles.com` returned HTTP 403 in both ordinary and escalated
connectivity checks. Real segmentation requires access to that host or a local
copy of the official checkpoint. RGB-D registration and reconstruction are not
part of this baseline.
