"""Notebook-first figures and an optional interactive, side-by-side viewer."""

from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .benchmark import _read_mask, load_run, validate_shared_frames
from .geometry import TYPES
from .pipeline import COLORS


def processed_image(frame):
    with Image.open(frame["source_image"]) as original:
        return original.convert("RGB").rotate(-frame["rotation_clockwise_degrees"], expand=True)


def plot_comparison(runs, frame_id=None, thresholds=None, boxes=True, masks=True, labels=False,
                    quality=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch, Rectangle

    loaded = {name: load_run(Path(path)) for name, path in runs.items()}
    validate_shared_frames(loaded)
    reference = next(iter(loaded.values()))["frames"]
    first = next((f for f in reference if f["frame_id"] == frame_id), reference[0])
    image = processed_image(first)
    figure, axes = plt.subplots(1, len(runs) + 1, figsize=(6 * (len(runs) + 1), 9), squeeze=False)
    axes = axes[0]
    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=15, fontweight="bold", pad=14)
    for ax, (name, run) in zip(axes[1:], loaded.items()):
        frame = next(f for f in run["frames"] if f["source_image"] == first["source_image"])
        selected = [h for h in frame["holds"] if h["detector_score"] >= (thresholds or {}).get(name, 0)]
        background = ax.imshow(image)
        overlay = np.asarray(image).astype(np.float32) if masks else None
        evaluated = (quality or {}).get(name, {}).get("matches", {}).get(frame["frame_id"])
        tp_ids = set(evaluated["tp"]) if evaluated else set()
        fp_ids = set(evaluated["fp"]) if evaluated else set()
        for hold in selected:
            color = ("#16a34a" if hold["hold_id"] in tp_ids else "#ef4444" if hold["hold_id"] in fp_ids
                     else COLORS.get(hold["predicted_type"], "#3b82f6"))
            if masks and hold.get("mask_path"):
                mask = _read_mask(frame, hold)
                overlay[mask] = 0.65 * overlay[mask] + 0.35 * 255 * np.asarray(to_rgb(color))
                if mask.any() and not mask.all():
                    ax.contour(mask, levels=[0.5], colors=[color], linewidths=0.7)
            x1, y1, x2, y2 = hold["bbox_xyxy"]
            if boxes:
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=color, linewidth=1.4))
            if labels:
                ax.text(x1, max(0, y1 - 3), hold["hold_id"], fontsize=7, color="white",
                        bbox={"facecolor": color, "alpha": 0.85, "pad": 1, "edgecolor": "none"})
        if masks:
            # One RGB raster per panel, not one full-size RGBA artist per hold.
            background.set_data(np.clip(overlay, 0, 255).astype(np.uint8))
        if evaluated:
            for x1, y1, x2, y2 in evaluated["fn_boxes"]:
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor="#f59e0b", linestyle="--", linewidth=1.6))
        ax.set_title(f"{name}\n{len(selected)} candidates · threshold {(thresholds or {}).get(name, 0):.2f}",
                     fontsize=14, fontweight="bold", pad=12)
    for ax in axes:
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        ax.axis("off")
    figure.suptitle(Path(first["source_image"]).name, fontsize=12, color="#64748b")
    legend_colors = ({"TP": "#16a34a", "FP": "#ef4444", "Missed GT": "#f59e0b"} if quality
                     else {**COLORS, "unknown": "#3b82f6"})
    figure.legend(handles=[Patch(facecolor=color, label=label) for label, color in legend_colors.items()],
                  loc="lower center", ncol=len(legend_colors), frameon=False, fontsize=9)
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    return figure


def plot_crop_grid(run_path, frame_id=None, threshold=0, maximum=18):
    import matplotlib.pyplot as plt

    frames = load_run(Path(run_path))["frames"]
    frame = next((f for f in frames if f["frame_id"] == frame_id), frames[0])
    holds = sorted((h for h in frame["holds"] if h["detector_score"] >= threshold),
                   key=lambda h: h["detector_score"], reverse=True)[:maximum]
    columns = 6
    rows = max(1, int(np.ceil(len(holds) / columns)))
    figure, axes = plt.subplots(rows, columns, figsize=(16, rows * 2.7), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, hold in zip(axes.flat, holds):
        with Image.open(Path(frame["directory"]) / hold["crop_path"]) as crop:
            ax.imshow(crop.copy())
        label = hold["object_kind"]
        mask_score = hold.get("segmentation_score")
        mask_text = "—" if mask_score is None else f"{mask_score:.2f}"
        ax.set_title(f"{hold['hold_id']} · {label}\ndet {hold['detector_score']:.2f} · mask {mask_text}",
                     fontsize=9, color="#b45309" if hold["needs_review"] else "#334155")
    if not holds:
        axes[0, 0].text(0.5, 0.5, "No candidates at this threshold", ha="center")
    figure.tight_layout()
    return figure


def plot_diagnostics(paths, report):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    names = [r["model"] for r in report["models"]]
    colors = ["#2563eb", "#8b5cf6", "#14b8a6"]
    axes[0].bar(names, [r["detections_at_threshold"] for r in report["models"]], color=colors[:len(names)])
    axes[0].set_title("Candidates at selected thresholds")
    axes[0].set_ylabel("Observations across frames, not unique holds")
    latency = [r["model_ms_p50"] for r in report["models"]]
    if all(v is not None for v in latency):
        axes[1].bar(names, latency, color=colors[:len(names)])
        axes[1].set_ylabel("Median model milliseconds / image")
    else:
        axes[1].text(0.5, 0.5, "Stage timing absent in legacy run", ha="center", transform=axes[1].transAxes)
    axes[1].set_title("Measured latency (load excluded)")
    for name, row in zip(names, report["models"]):
        frames = load_run(Path(paths[name]))["frames"]
        scores = [h["detector_score"] for f in frames for h in f["holds"]]
        axes[2].hist(scores, bins=np.linspace(0, 1, 21), alpha=0.55, label=name)
        axes[2].axvline(row["threshold"], linestyle="--", linewidth=1)
    axes[2].set_title("Raw detection scores — not accuracy")
    axes[2].set_xlabel("Scores are not calibrated across models")
    axes[2].legend()
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    figure.tight_layout()
    return figure


def plot_quality(report):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    evaluated = [row for row in report["models"] if row["quality"] is not None]
    for row in evaluated:
        quality = row["quality"]
        if quality["box_ap"] is not None:
            curve = quality["box_ap"]["pr_curve_iou50"]
            axes[0].plot(curve["recall"], [np.nan if p is None else p for p in curve["precision"]],
                         label=row["model"], linewidth=2)
    axes[0].set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
                title="Class-agnostic box PR · IoU 0.50")
    if axes[0].lines:
        axes[0].legend()
    if evaluated:
        width = 0.25
        x = np.arange(len(evaluated))
        for offset, (key, title) in enumerate([("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]):
            axes[1].bar(x + (offset - 1) * width,
                        [np.nan if r["quality"][key] is None else r["quality"][key] for r in evaluated],
                        width, label=title)
        axes[1].set_xticks(x, [r["model"] for r in evaluated])
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()
        axes[1].set_title("Operating point · configured thresholds")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Ground truth required", transform=ax.transAxes, ha="center", fontsize=15)
    figure.tight_layout()
    return figure


def plot_type_confusions(report):
    import matplotlib.pyplot as plt

    rows = [r for r in report["models"] if r.get("quality") and r["quality"]["classification"]]
    if not rows:
        return None
    figure, axes = plt.subplots(1, len(rows), figsize=(7 * len(rows), 5), squeeze=False)
    for ax, row in zip(axes[0], rows):
        data = row["quality"]["classification"]
        matrix = np.asarray(data["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(7), data["predicted_labels"], rotation=45, ha="right")
        ax.set_yticks(range(6), data["true_labels"])
        ax.set(xlabel="Predicted raw top-1", ylabel="Ground truth",
               title=f"{row['model']} · types on IoU-matched GT")
        for (y, x), value in np.ndenumerate(matrix):
            ax.text(x, y, str(value), ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "#334155", fontsize=9)
    figure.tight_layout()
    return figure


def metric_rows(report):
    rows = []
    for model in report["models"]:
        quality = model["quality"]
        types = quality["classification"] if quality else None
        rows.append({"Model": model["model"], "Images": model["images"],
                     "Candidates": model["detections_at_threshold"], "Threshold": model["threshold"],
                     "Model p50 ms": model["model_ms_p50"],
                     "Model p95 ms": model["model_ms_p95"],
                     "Precision": quality["precision"] if quality else None,
                     "Recall": quality["recall"] if quality else None,
                     "F1": quality["f1"] if quality else None,
                     "Box AP50": quality["box_ap"]["ap50"] if quality and quality["box_ap"] else None,
                     "Box AP50–95": quality["box_ap"]["ap50_95"] if quality and quality["box_ap"] else None,
                     "Mask AP50–95": quality["mask_ap"]["ap50_95"] if quality and quality["mask_ap"] else None,
                     "Mask IoU · matched": quality["mean_mask_iou_on_box_matched"] if quality else None,
                     "Type acc · matched": types["accuracy_on_matched"] if types else None,
                     "Type macro-F1 · matched": types["macro_f1_on_matched_all_six"] if types else None,
                     "Type coverage": types["typed_gt_detection_coverage"] if types else None,
                     "Correct type recall · all GT": types["end_to_end_correct_type_recall"] if types else None})
    return rows


class ComparisonViewer:
    def __init__(self, paths, thresholds):
        import ipywidgets as widgets
        from IPython.display import clear_output, display
        import matplotlib.pyplot as plt

        self.paths = paths
        loaded = {name: load_run(Path(path)) for name, path in paths.items()}
        validate_shared_frames(loaded)
        frames = next(iter(loaded.values()))["frames"]
        self.frame = widgets.Dropdown(options=[(Path(f["source_image"]).name, f["frame_id"]) for f in frames],
                                      description="Frame:", layout=widgets.Layout(width="60%"))
        self.sliders = {name: widgets.FloatSlider(value=thresholds[name], min=loaded[name]["summary"]["settings"]["confidence"], max=1, step=0.01,
                        description=name, continuous_update=False, readout_format=".2f",
                        style={"description_width": "initial"}) for name in paths}
        self.boxes = widgets.Checkbox(value=True, description="Boxes")
        self.masks = widgets.Checkbox(value=True, description="Masks")
        self.labels = widgets.Checkbox(value=False, description="IDs (can clutter)")
        self.output = widgets.Output()

        def update(*_):
            with self.output:
                clear_output(wait=True)
                figure = plot_comparison(paths, self.frame.value, self.thresholds, self.boxes.value,
                                         self.masks.value, self.labels.value)
                display(figure)
                plt.close(figure)

        for control in [self.frame, *self.sliders.values(), self.boxes, self.masks, self.labels]:
            control.observe(update, names="value")
        self.widget = widgets.VBox([
            self.frame, widgets.HBox(list(self.sliders.values())),
            widgets.HBox([self.boxes, self.masks, self.labels]), self.output,
            widgets.HTML("<small>Thresholds change the viewer. Rerun the metrics cell to evaluate the new thresholds.</small>"),
        ])
        update()

    @property
    def thresholds(self):
        return {name: slider.value for name, slider in self.sliders.items()}


def write_report(paths, report, output: Path, maximum_frames=3):
    """Portable static figures plus machine-readable metrics; never replace a report."""
    import csv
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rows = metric_rows(report)
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures = [("diagnostics.png", plot_diagnostics(paths, report))]
    if report["quality_status"] == "evaluated":
        figures.append(("quality.png", plot_quality(report)))
        confusion = plot_type_confusions(report)
        if confusion is not None:
            figures.append(("types.png", confusion))
    reference = load_run(Path(next(iter(paths.values()))))["frames"]
    thresholds = {r["model"]: r["threshold"] for r in report["models"]}
    quality = {r["model"]: r["quality"] for r in report["models"] if r["quality"] is not None}
    for index, frame in enumerate(reference[:maximum_frames]):
        figures.append((f"frame_{index:03d}.jpg", plot_comparison(paths, frame["frame_id"], thresholds)))
        if quality:
            figures.append((f"errors_{index:03d}.jpg", plot_comparison(paths, frame["frame_id"], thresholds,
                                                                    quality=quality)))
    for name, figure in figures:
        figure.savefig(output / name, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(figure)
    table = "<tr>" + "".join(f"<th>{escape(k)}</th>" for k in rows[0]) + "</tr>"
    for row in rows:
        table += "<tr>" + "".join("<td>" + ("—" if v is None else f"{v:.3f}" if isinstance(v, float)
                                               else escape(str(v))) + "</td>" for v in row.values()) + "</tr>"
    html = ('<!doctype html><meta charset="utf-8"><title>DeMoSearch · comparison</title>'
            '<style>body{font:15px system-ui;max-width:1400px;margin:30px auto;padding:0 20px;color:#172033}'
            'img{max-width:100%}table{border-collapse:collapse;font-size:12px}td,th{padding:9px;border-bottom:1px solid #ddd}'
            '.table{overflow:auto}p{color:#64748b}</style><h1>YOLO vs SAM 3</h1>'
            '<p>Candidate observations, not unique holds. Scores are not accuracy. '
            'Box metrics are class-agnostic; maxDets and thresholds are recorded in comparison.json.</p>'
            '<div class="table"><table>' + table + '</table></div>')
    if report["quality_status"] != "evaluated":
        html += '<p><strong>Ground truth not supplied: quality metrics are unavailable.</strong></p>'
    html += ''.join(f'<h2>{escape(name)}</h2><img loading="lazy" src="{name}">' for name, _ in figures)
    html += '<p>Errors: green TP · red FP · dashed orange FN. Type accuracy is conditional on matched GT.</p>'
    (output / "index.html").write_text(html, encoding="utf-8")
    return output
