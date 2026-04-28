from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HIGHER_BETTER = {"mAP@0.5", "mAP@0.5:0.95", "fps", "speedup_vs_frcnn"}
LOWER_BETTER = {"mean_inf_time_s", "false_positives", "false_negatives"}
PLOT_MODELS = ["yolo", "frcnn", "hybrid"]
DISPLAY_NAMES = {"yolo": "YOLO", "frcnn": "Faster R-CNN", "hybrid": "Hybrid"}


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def flatten_run(run: dict) -> list[dict]:
    rows = []
    for model_name, metrics in run["results"].items():
        stage_timings = metrics.get("stage_timings", {})
        row = {
            "run_id": run["run_id"],
            "dataset": run["dataset"],
            "evaluation_scope": run["evaluation_scope"],
            "model": model_name,
            "hybrid_strategy": run["hybrid_config"].get("strategy", "") if model_name == "hybrid" else "",
            "proposal_conf": run["hybrid_config"].get("proposal_conf", "") if model_name == "hybrid" else "",
            "proposal_iou": run["hybrid_config"].get("proposal_iou", "") if model_name == "hybrid" else "",
            "max_proposals": run["hybrid_config"].get("max_proposals", "") if model_name == "hybrid" else "",
            "frcnn_batch_size": run["hybrid_config"].get("frcnn_batch_size", "") if model_name == "hybrid" else "",
            "agreement_iou": run["hybrid_config"].get("agreement_iou", "") if model_name == "hybrid" else "",
            "unmatched_frcnn_score_thresh": run["hybrid_config"].get("unmatched_frcnn_score_thresh", "") if model_name == "hybrid" else "",
            "unmatched_yolo_score_thresh": run["hybrid_config"].get("unmatched_yolo_score_thresh", "") if model_name == "hybrid" else "",
            "score_fusion": run["hybrid_config"].get("score_fusion", "") if model_name == "hybrid" else "",
            "final_score_thresh": run["hybrid_config"].get("final_score_thresh", "") if model_name == "hybrid" else "",
            "final_nms_iou": run["hybrid_config"].get("final_nms_iou", "") if model_name == "hybrid" else "",
            "stage_yolo_s": stage_timings.get("yolo", 0.0),
            "stage_crop_s": stage_timings.get("crop", 0.0),
            "stage_frcnn_s": stage_timings.get("frcnn", 0.0),
            "stage_post_s": stage_timings.get("post", 0.0),
            "stage_total_s": stage_timings.get("total", 0.0),
        }
        row.update(metrics)
        row.pop("stage_timings", None)
        rows.append(row)
    return rows


def normalize_metric(values: pd.Series, metric_name: str) -> pd.Series:
    if values.nunique() == 1:
        return pd.Series(np.ones(len(values)), index=values.index)

    min_v = values.min()
    max_v = values.max()
    scaled = (values - min_v) / (max_v - min_v)
    if metric_name in LOWER_BETTER:
        scaled = 1.0 - scaled
    return scaled


def save_csv_tables(manifest: dict, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for run in manifest["runs"]:
        all_rows.extend(flatten_run(run))
    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(output_dir / "experiment_runs.csv", index=False)

    selected = next(run for run in manifest["runs"] if run["run_id"] == manifest["selected_final_run_id"])
    final_df = pd.DataFrame(flatten_run(selected))
    final_df.to_csv(output_dir / "final_model_comparison.csv", index=False)

    timing_rows = []
    for model_name, metrics in selected["results"].items():
        row = {"model": model_name}
        row.update(metrics["stage_timings"])
        timing_rows.append(row)
    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(output_dir / "stage_timing_breakdown.csv", index=False)

    return all_df, final_df


def make_bar_plot(df: pd.DataFrame, metric: str, title: str, ylabel: str, output_path: Path) -> None:
    ordered = df.set_index("model").loc[PLOT_MODELS].reset_index()
    labels = [DISPLAY_NAMES[name] for name in ordered["model"]]
    values = ordered[metric].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#2563eb", "#059669", "#dc2626"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}" if value < 10 else f"{value:.1f}",
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_grouped_accuracy_plot(df: pd.DataFrame, output_path: Path) -> None:
    ordered = df.set_index("model").loc[PLOT_MODELS].reset_index()
    labels = [DISPLAY_NAMES[name] for name in ordered["model"]]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, ordered["mAP@0.5"], width, label="mAP@0.5", color="#2563eb")
    ax.bar(x + width / 2, ordered["mAP@0.5:0.95"], width, label="mAP@0.5:0.95", color="#7c3aed")

    ax.set_title("Accuracy Comparison")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_grouped_error_plot(df: pd.DataFrame, output_path: Path) -> None:
    ordered = df.set_index("model").loc[PLOT_MODELS].reset_index()
    labels = [DISPLAY_NAMES[name] for name in ordered["model"]]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, ordered["false_positives"], width, label="False Positives", color="#f59e0b")
    ax.bar(x + width / 2, ordered["false_negatives"], width, label="False Negatives", color="#ef4444")

    ax.set_title("Detection Error Comparison")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def annotate_heatmap(ax, data: np.ndarray, xlabels: list[str], ylabels: list[str], fmt: str = ".2f") -> None:
    im = ax.imshow(data, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_yticklabels(ylabels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", color="black", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def make_performance_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    ordered = df.set_index("model").loc[PLOT_MODELS].reset_index()
    metrics = [
        "mAP@0.5",
        "mAP@0.5:0.95",
        "fps",
        "speedup_vs_frcnn",
        "mean_inf_time_s",
        "false_positives",
        "false_negatives",
    ]
    normalized = pd.DataFrame({"model": ordered["model"]})
    for metric in metrics:
        normalized[metric] = normalize_metric(ordered[metric], metric)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    annotate_heatmap(
        ax=ax,
        data=normalized[metrics].to_numpy(),
        xlabels=metrics,
        ylabels=[DISPLAY_NAMES[m] for m in normalized["model"]],
        fmt=".2f",
    )
    ax.set_title("Normalized Performance Heatmap (Higher Is Better)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_stage_timing_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    ordered = df.set_index("model").loc[PLOT_MODELS].reset_index()
    metrics = ["yolo", "crop", "frcnn", "post", "total"]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    annotate_heatmap(
        ax=ax,
        data=ordered[metrics].to_numpy(),
        xlabels=metrics,
        ylabels=[DISPLAY_NAMES[m] for m in ordered["model"]],
        fmt=".3f",
    )
    ax.set_title("Stage Timing Heatmap (seconds/image)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_radar_plot(df: pd.DataFrame, output_path: Path) -> None:
    ordered = df.set_index("model").loc[PLOT_MODELS].reset_index()
    metrics = ["mAP@0.5", "mAP@0.5:0.95", "fps", "speedup_vs_frcnn", "false_positives", "false_negatives"]
    normalized = pd.DataFrame({"model": ordered["model"]})
    for metric in metrics:
        normalized[metric] = normalize_metric(ordered[metric], metric)

    labels = metrics
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors = {"yolo": "#2563eb", "frcnn": "#059669", "hybrid": "#dc2626"}
    for _, row in normalized.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=DISPLAY_NAMES[row["model"]], color=colors[row["model"]])
        ax.fill(angles, values, alpha=0.15, color=colors[row["model"]])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Overall Performance Radar (Normalized)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_plots(final_df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    make_bar_plot(final_df, "mean_inf_time_s", "Inference Time Comparison", "Seconds / image", figures_dir / "inference_time_comparison.png")
    make_bar_plot(final_df, "fps", "Speed Comparison", "Frames per second", figures_dir / "fps_comparison.png")
    make_grouped_accuracy_plot(final_df, figures_dir / "accuracy_comparison.png")
    make_grouped_error_plot(final_df, figures_dir / "error_comparison.png")
    make_performance_heatmap(final_df, figures_dir / "performance_heatmap.png")
    make_radar_plot(final_df, figures_dir / "performance_radar.png")

    timing_df = final_df[
        ["model", "stage_yolo_s", "stage_crop_s", "stage_frcnn_s", "stage_post_s", "stage_total_s"]
    ].rename(
        columns={
            "stage_yolo_s": "yolo",
            "stage_crop_s": "crop",
            "stage_frcnn_s": "frcnn",
            "stage_post_s": "post",
            "stage_total_s": "total",
        }
    )
    make_stage_timing_heatmap(timing_df, figures_dir / "stage_timing_heatmap.png")


def format_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[columns].copy()
    for col in ["mAP@0.5", "mAP@0.5:0.95", "mean_inf_time_s", "fps", "speedup_vs_frcnn"]:
        if col in table.columns:
            table[col] = table[col].map(lambda x: f"{x:.4f}")
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def generate_report(manifest: dict, all_df: pd.DataFrame, final_df: pd.DataFrame, output_path: Path) -> None:
    selected_run = next(run for run in manifest["runs"] if run["run_id"] == manifest["selected_final_run_id"])
    best_hybrid = final_df[final_df["model"] == "hybrid"].iloc[0]
    frcnn = final_df[final_df["model"] == "frcnn"].iloc[0]
    yolo = final_df[final_df["model"] == "yolo"].iloc[0]

    report = f"""# Hybrid Object Detection Report

## Project Overview

This project evaluates a hybrid object detector that combines YOLO and Faster R-CNN. The original crop-refinement cascade was implemented first, but evaluation showed a structural problem: it reduced false positives while sharply increasing false negatives and made inference substantially slower than Faster R-CNN alone. The project was then redesigned around a detector agreement-fusion strategy, where YOLO and Faster R-CNN both run on the full image and the final detections are fused or filtered based on cross-detector agreement.

## Objective

The target was to achieve a better speed-accuracy trade-off than standalone detectors:

- `YOLO` as the high-speed detector
- `Faster R-CNN` as the high-accuracy detector
- `Hybrid` as a precision-improving model that preserves most of Faster R-CNN's accuracy without the severe runtime penalty of the crop cascade

## Methodology

### 1. Standalone baselines

- `YOLO` was evaluated directly on the full image.
- `Faster R-CNN` was evaluated directly on the full image.

### 2. Initial hybrid design: crop refinement

The first hybrid design used YOLO to generate proposals, cropped those regions, and then ran Faster R-CNN on each crop. This design had two major flaws:

- repeated Faster R-CNN execution over many crops made runtime much worse than standalone Faster R-CNN
- tight or missing proposals caused recall loss, which translated directly into higher false negatives and lower `mAP`

### 3. Final hybrid design: agreement fusion

The final hybrid design uses:

- YOLO full-image detections
- Faster R-CNN full-image detections
- class-aware agreement matching based on IoU
- score fusion and optional YOLO fallback for unmatched confident detections
- final class-aware NMS

This preserves full-image context and avoids re-running Faster R-CNN for every proposal crop.

## Evaluation Scope

- Dataset: `{selected_run["dataset"]}`
- Scope: `{selected_run["evaluation_scope"]}`
- Selected final run: `{selected_run["run_id"]}`

## Final Selected Hybrid Configuration

```json
{json.dumps(selected_run["hybrid_config"], indent=2)}
```

## Final Model Comparison

{format_table(final_df, ["model", "mAP@0.5", "mAP@0.5:0.95", "mean_inf_time_s", "fps", "false_positives", "false_negatives", "speedup_vs_frcnn"])}

## Experimental Progression

{format_table(all_df, ["run_id", "model", "mAP@0.5", "mAP@0.5:0.95", "mean_inf_time_s", "fps", "false_positives", "false_negatives"])}

## Analysis

### YOLO

- Highest speed: `{yolo["fps"]:.2f} FPS`
- Lowest latency: `{yolo["mean_inf_time_s"]:.4f}s/image`
- Accuracy lower than Faster R-CNN and hybrid
- Very high false-positive count because it is optimized for detection throughput and broad recall

### Faster R-CNN

- Best `mAP@0.5`: `{frcnn["mAP@0.5"]:.4f}`
- Strong `mAP@0.5:0.95`: `{frcnn["mAP@0.5:0.95"]:.4f}`
- Better recall than the hybrid
- Slower than YOLO, but still the strongest pure accuracy baseline

### Final Hybrid

- `mAP@0.5`: `{best_hybrid["mAP@0.5"]:.4f}`
- `mAP@0.5:0.95`: `{best_hybrid["mAP@0.5:0.95"]:.4f}`
- `FPS`: `{best_hybrid["fps"]:.2f}`
- False positives reduced sharply to `{int(best_hybrid["false_positives"])}`
- False negatives remained above Faster R-CNN at `{int(best_hybrid["false_negatives"])}`

The final hybrid significantly improved over the original crop-refinement cascade. It no longer suffered the extreme runtime penalty of repeated crop processing, and it preserved localization quality much better. The best hybrid run slightly exceeded Faster R-CNN on `mAP@0.5:0.95`, which suggests that the agreement-fusion strategy improved box ranking and localization stability. However, Faster R-CNN still remained better on `mAP@0.5`, which indicates that recall was still slightly lower in the hybrid system.

## Key Findings

1. The original crop-refinement hybrid was not a good production design.
   It was structurally too slow and too sensitive to proposal recall loss.
2. The agreement-fusion redesign is the correct hybrid direction.
   It keeps latency close to Faster R-CNN while retaining the hybrid's precision advantage.
3. The hybrid is best interpreted as a precision-oriented detector.
   It suppresses false positives much better than Faster R-CNN, but still loses some true detections.
4. The final hybrid is a valid research result even without winning every metric.
   It demonstrates a meaningful trade-off rather than a universal improvement.

## Limitations

- The current final comparison is based on a VOC subset rather than a full COCO evaluation.
- COCO was not finalized in the current experiment set because the Kaggle input was not mounted correctly during the reported runs.
- False-positive and false-negative counts depend on thresholding choices and should be interpreted alongside `mAP`, not in isolation.

## Recommended Final Conclusion

The final agreement-fusion hybrid is the strongest hybrid design explored in this project. It substantially improves over the original crop-refinement approach by preserving full-image context and avoiding repeated Faster R-CNN execution on cropped proposals. Compared with standalone Faster R-CNN, the final hybrid offers a large reduction in false positives and competitive `mAP@0.5:0.95`, with only a small runtime penalty. Compared with YOLO, it sacrifices throughput but gains a much stronger precision-accuracy profile. For applications where false positives are costly and near-Faster-R-CNN latency is acceptable, the redesigned hybrid provides a useful trade-off.

## Generated Assets

- `reports/tables/final_model_comparison.csv`
- `reports/tables/experiment_runs.csv`
- `reports/tables/stage_timing_breakdown.csv`
- `reports/figures/inference_time_comparison.png`
- `reports/figures/fps_comparison.png`
- `reports/figures/accuracy_comparison.png`
- `reports/figures/error_comparison.png`
- `reports/figures/performance_heatmap.png`
- `reports/figures/performance_radar.png`
- `reports/figures/stage_timing_heatmap.png`
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CSV tables, plots, and a markdown report from experiment results.")
    parser.add_argument("--manifest", type=Path, default=Path("reports/data/experiment_runs.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    report_path = args.output_dir / "FINAL_REPORT.md"

    all_df, final_df = save_csv_tables(manifest, tables_dir)
    make_plots(final_df, figures_dir)
    generate_report(manifest, all_df, final_df, report_path)

    print(f"Generated report assets in {args.output_dir}")


if __name__ == "__main__":
    main()
