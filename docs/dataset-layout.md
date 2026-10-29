# Collected wall dataset

The Drive dataset contains six wall captures: two wall A scans and one scan each
for walls B through E. Every capture currently has a Polycam raw-data ZIP, GLB,
OBJ backup, and route notes. Route photos are partial and currently stored as
top-level HEIC files rather than consistently inside `photos/`.

Preferred portable layout:

```text
wallC_overhang_right/
├── wallC_scan.glb
├── wallC_obj.zip
├── wallC_raw_data.zip
├── photos/
│   ├── yellow_v3.heic
│   ├── green_v1.heic
│   └── blue_v5.heic
└── route_notes.txt
```

`holds inspect-wall-bundle` also accepts the current layout, including top-level
route photos, raw ZIPs without a `.zip` suffix, and extracted `*_raw_data/`
directories. For Colab, add the shared dataset folder as a shortcut under My
Drive and set `DRIVE_INPUT` to one complete wall folder.

The raw capture is the inference source because it contains synchronized RGB,
depth, confidence, camera intrinsics, and camera poses. The GLB is retained as
the wall mesh. Route photos supply a cleaner overview and route selection but do
not carry the raw capture camera calibration.

Route notes are weak supervision, not per-hold ground truth. In particular, wall
A contains two green V3 routes with opposite directions. Consequently, color
clustering proposes route candidates but cannot assign final route identity by
itself.
