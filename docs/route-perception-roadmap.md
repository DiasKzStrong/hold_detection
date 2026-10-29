# Route perception roadmap

The six-week research MVP turns multi-view wall captures into spatial hold data
for a graph-based climbing planner. Each milestone remains independently
measurable so reconstruction errors cannot hide perception errors.

## Milestones

1. **Instance perception:** compare SAM 3 masks with the legacy YOLO proposal
   baseline; export reviewable evidence and color-route candidates.
2. **Data and evaluation:** train a YOLO segmentation candidate and measure mask
   AP/recall on a human-verified, wall-separated test set.
3. **Multi-view identity:** associate observations of the same Physical Hold and
   preserve ambiguity rather than forcing a match.
4. **3D association:** project verified observations onto a registered wall mesh
   and compare against stable hold IDs supplied with ground truth.
5. **Spatial scene:** populate metric position, surface normal, size, protrusion,
   and volume mounting relationships in `scene.json`.
6. **Hardening:** evaluate end-to-end coverage, prepare portable reports, and
   document failure modes for the planner team.

Animation, service deployment, authentication, and a general-purpose
photogrammetry backend are outside this MVP.

## Dataset policy

- [Climbing Hold Detection v26](https://universe.roboflow.com/climbtag/climbing-hold-detection-7uehq/dataset/26)
  is the initial CC BY 4.0 instance-segmentation source.
- [ClimbInst](https://huggingface.co/datasets/ClimbVision/ClimbInst) may be used
  for university research experiments, but its constituent-source licensing must
  be resolved before redistribution or commercial use.
- [Blackcreed v7](https://universe.roboflow.com/blackcreed-xpgxh/climbing-holds-and-volumes/dataset/7)
  supplies CC BY 4.0 boxes that may seed SAM-assisted annotations.
- Pseudo-labels are never benchmark ground truth. Test masks, route labels, and
  cross-view physical-hold IDs require human verification.
- Dataset splits are by physical wall/capture session, never random frames.

The project currently has no public dataset that directly maps a hold across
multiple images to a stable node in a 3D wall model. Team-authored wall captures
and ground-truth associations are therefore required for milestones 3–5.
