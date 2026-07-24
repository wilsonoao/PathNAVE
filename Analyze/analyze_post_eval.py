from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path("/work/Agent_benchmark")
OUT = BASE / "Analyze"
METHODS = ["slide_seek", "Pathology-CoT", "PathAgent", "CPathAgent"]
TASKS = ["CAMELYON16_detection", "TCGA_BRCA_subtype", "TCGA_LUNG_Classification"]
TASK_LABELS = {
    "CAMELYON16_detection": "CAMELYON16\ndetection",
    "TCGA_BRCA_subtype": "TCGA-BRCA\nsubtype",
    "TCGA_LUNG_Classification": "TCGA-LUNG\nclassification",
}
METHOD_LABELS = {
    "slide_seek": "slide_seek",
    "Pathology-CoT": "Pathology\nCoT",
    "PathAgent": "PathAgent",
    "CPathAgent": "CPathAgent",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def read_summary_value(path: Path, metric: str) -> float | None:
    if not path.exists():
        return None
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("metric") == metric:
                try:
                    return float(row["value"])
                except (KeyError, ValueError):
                    return None
    return None


def load_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    threshold_rows = []
    for method in METHODS:
        for task in TASKS:
            roi_dir = BASE / method / "post_eval" / task / "ROI"
            mrfh_path = roi_dir / "mrfh_by_threshold.csv"
            summary_path = roi_dir / "roi_overlap_summary.csv"

            if not mrfh_path.exists():
                continue

            curve = pd.read_csv(mrfh_path)
            threshold_col = None
            if "precision_threshold" in curve.columns:
                threshold_col = "precision_threshold"
            elif "recall_threshold" in curve.columns:
                threshold_col = "recall_threshold"
                curve = curve.rename(columns={"recall_threshold": "precision_threshold"})
            else:
                print(f"Skipping {mrfh_path}: no threshold column")
                continue
            curve["threshold_type"] = threshold_col.replace("_threshold", "")
            curve["method"] = method
            curve["task"] = task
            threshold_rows.append(curve)

            row0 = curve[curve["precision_threshold"].round(8) == 0]
            if row0.empty:
                row0 = curve.iloc[[0]]
            row0 = row0.iloc[0]

            rows.append(
                {
                    "method": method,
                    "task": task,
                    "mrfh_at_thr0": float(row0.get("mrfh", float("nan"))),
                    "hit_rate_at_thr0": float(row0["hit_rate_pct"]),
                    "hit_cases_at_thr0": int(row0["hit_cases"]),
                    "case_micro_precision": read_summary_value(summary_path, "case_micro_precision"),
                    "per_roi_micro_precision": read_summary_value(summary_path, "per_roi_micro_precision"),
                    "conditional_hit_cases": read_summary_value(summary_path, "conditional_hit_cases"),
                    "conditional_hit_rate": read_summary_value(summary_path, "conditional_hit_rate"),
                    "conditional_mrfh": read_summary_value(summary_path, "conditional_mrfh"),
                    "processed_cases": read_summary_value(summary_path, "processed_cases"),
                    "per_roi_total": read_summary_value(summary_path, "per_roi_total"),
                    "total_16x16_subpatches": read_summary_value(summary_path, "total_16x16_subpatches"),
                    "coverage_16x16_multiscale": read_summary_value(summary_path, "coverage_16x16_multiscale"),
                    "efficiency": read_summary_value(summary_path, "efficiency"),
                    "efficiency_avg_per_case": read_summary_value(summary_path, "efficiency_avg_per_case"),
                    "efficiency_cases_with_total_size": read_summary_value(summary_path, "efficiency_cases_with_total_size"),
                    "has_explicit_hit_rate_file": (roi_dir / "hit_rate_by_threshold.csv").exists(),
                }
            )

    return pd.DataFrame(rows), pd.concat(threshold_rows, ignore_index=True)


def plot_task_comparison(metrics: pd.DataFrame) -> None:
    metric_specs = [
        ("mrfh_at_hit_threshold", "MRfH\nthreshold 0.05", "#3b6fb6", 1.0),
        ("case_micro_precision", "Case micro\nprecision", "#c75146", 1.0),
        ("hit_rate_at_hit_threshold", "Hit rate\nthreshold 0.05", "#3f8f58", 100.0),
    ]

    fig, axes = plt.subplots(
        len(metric_specs),
        len(TASKS),
        figsize=(16, 9.5),
        sharey="row",
        constrained_layout=True,
    )
    x = list(range(len(METHODS)))

    for row, (key, metric_label, color, divisor) in enumerate(metric_specs):
        for col, task in enumerate(TASKS):
            ax = axes[row, col]
            sub = metrics[metrics["task"] == task].set_index("method").reindex(METHODS)
            vals = sub[key].astype(float) / divisor
            bars = ax.bar(x, vals, width=0.64, color=color)

            if row == 0:
                ax.set_title(TASK_LABELS.get(task, task), pad=8)
            if col == 0:
                ax.set_ylabel(metric_label)

            ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.bar_label(bars, labels=[f"{v:.2f}" if pd.notna(v) else "" for v in vals], padding=3, fontsize=8)

    fig.suptitle("MRfH Precision Hit Rate", fontsize=15)
    fig.savefig(OUT / "task_method_mrfh_precision_hitrate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_metrics_at_threshold(metrics: pd.DataFrame, curves: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for _, base in metrics.iterrows():
        curve = curves[(curves["method"] == base["method"]) & (curves["task"] == base["task"])]
        if curve.empty:
            continue
        exact = curve[curve["precision_threshold"].round(8) == round(threshold, 8)]
        if exact.empty:
            idx = (curve["precision_threshold"] - threshold).abs().idxmin()
            row = curve.loc[idx]
        else:
            row = exact.iloc[0]
        out = base.copy()
        out["hit_threshold"] = float(row["precision_threshold"])
        out["mrfh_at_hit_threshold"] = float(row["mrfh"])
        out["hit_rate_at_hit_threshold"] = float(row["hit_rate_pct"])
        out["hit_cases_at_hit_threshold"] = int(row["hit_cases"])
        out["threshold_type"] = row.get("threshold_type", "precision")
        rows.append(out)
    return pd.DataFrame(rows)


def plot_task_comparison_at_threshold(metrics_thr: pd.DataFrame, threshold: float) -> None:
    metric_specs = [
        ("mrfh_at_hit_threshold", f"MRfH\nthreshold {threshold:.2f}", "#3b6fb6", 1.0),
        ("case_micro_precision", "Case micro\nprecision", "#c75146", 1.0),
        ("hit_rate_at_hit_threshold", f"Hit rate\nthreshold {threshold:.2f}", "#3f8f58", 100.0),
    ]

    fig, axes = plt.subplots(
        len(metric_specs),
        len(TASKS),
        figsize=(16, 9.5),
        sharey="row",
        constrained_layout=True,
    )
    x = list(range(len(METHODS)))

    for row_idx, (key, metric_label, color, divisor) in enumerate(metric_specs):
        for col, task in enumerate(TASKS):
            ax = axes[row_idx, col]
            sub = metrics_thr[metrics_thr["task"] == task].set_index("method").reindex(METHODS)
            vals = sub[key].astype(float) / divisor
            bars = ax.bar(x, vals, width=0.64, color=color)

            if row_idx == 0:
                ax.set_title(TASK_LABELS.get(task, task), pad=8)
            if col == 0:
                ax.set_ylabel(metric_label)

            ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.bar_label(bars, labels=[f"{v:.2f}" if pd.notna(v) else "" for v in vals], padding=3, fontsize=8)

    fig.suptitle("MRfH Precision Hit Rate", fontsize=15)
    suffix = str(threshold).replace(".", "")
    fig.savefig(OUT / f"task_method_mrfh_precision_hitrate_thr{suffix}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_curves(curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2,
        len(TASKS),
        figsize=(17, 8.8),
        sharex=True,
        constrained_layout=True,
    )
    legend_handles = []
    legend_labels = []
    for col, task in enumerate(TASKS):
        sub = curves[curves["task"] == task]
        for method in METHODS:
            m = sub[sub["method"] == method]
            if m.empty:
                continue
            threshold_type = str(m["threshold_type"].iloc[0])
            label = METHOD_LABELS.get(method, method).replace("\n", " ")
            linestyle = "-"
            if threshold_type != "precision":
                label = f"{label} ({threshold_type} thr.)"
                linestyle = "--"
            line0 = axes[0, col].plot(
                m["precision_threshold"],
                m["hit_rate_pct"],
                marker="o",
                ms=3,
                lw=1.8,
                linestyle=linestyle,
                label=label,
            )[0]
            axes[1, col].plot(
                m["precision_threshold"],
                m["mrfh"],
                marker="o",
                ms=3,
                lw=1.8,
                linestyle=linestyle,
                label=label,
            )
            if label not in legend_labels:
                legend_handles.append(line0)
                legend_labels.append(label)
        axes[0, col].set_title(TASK_LABELS.get(task, task), pad=8)
        axes[0, col].set_ylabel("Hit rate (%)")
        axes[1, col].set_ylabel("MRfH")
        axes[1, col].set_xlabel("Threshold")
        for ax in axes[:, col]:
            ax.grid(alpha=0.25)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelrotation=0)

    fig.legend(legend_handles, legend_labels, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("Threshold Sensitivity", fontsize=15)
    fig.savefig(OUT / "threshold_sensitivity_by_task.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_conditional_metrics(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2,
        len(TASKS),
        figsize=(17, 7.8),
        sharey="row",
        constrained_layout=True,
    )
    x = list(range(len(METHODS)))

    for col, task in enumerate(TASKS):
        sub = metrics[metrics["task"] == task].set_index("method").reindex(METHODS)
        hit_rate = sub["hit_rate_at_hit_threshold"].astype(float)
        mrfh = sub["mrfh_at_hit_threshold"].astype(float)

        bars0 = axes[0, col].bar(x, hit_rate, color="#3f8f58", width=0.64)
        bars1 = axes[1, col].bar(x, mrfh, color="#3b6fb6", width=0.64)

        axes[0, col].set_title(TASK_LABELS.get(task, task), pad=8)
        axes[0, col].set_ylabel("Hit rate threshold 0.05")
        axes[1, col].set_ylabel("MRfH threshold 0.05")
        axes[1, col].set_xlabel("Method")

        for ax in axes[:, col]:
            ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)

        axes[0, col].bar_label(
            bars0,
            labels=[f"{v:.1f}" if pd.notna(v) else "" for v in hit_rate],
            padding=3,
            fontsize=8,
        )
        axes[1, col].bar_label(
            bars1,
            labels=[f"{v:.3f}" if pd.notna(v) else "" for v in mrfh],
            padding=3,
            fontsize=8,
        )
        axes[0, col].set_ylim(0, max(hit_rate.max() * 1.2, 5.0))
        axes[1, col].set_ylim(0, max(mrfh.max() * 1.2, 0.05))

    fig.suptitle("Hit Rate MRfH Threshold 0.05", fontsize=15)
    fig.savefig(OUT / "conditional_hit_rate_mrfh_by_task.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_cost_efficiency(metrics: pd.DataFrame) -> pd.DataFrame:
    cost = metrics.copy()
    if "mrfh_at_hit_threshold" in cost.columns:
        cost["mrfh_at_thr0"] = cost["mrfh_at_hit_threshold"]
    if "hit_rate_at_hit_threshold" in cost.columns:
        cost["hit_rate_at_thr0"] = cost["hit_rate_at_hit_threshold"]
    cost["avg_roi_per_case"] = cost["per_roi_total"] / cost["processed_cases"]
    cost["avg_16x16_subpatches_per_case"] = cost["total_16x16_subpatches"] / cost["processed_cases"]
    cost["coverage_pct"] = cost["coverage_16x16_multiscale"] * 100.0
    cost["efficiency_pct"] = cost["efficiency"] * 100.0
    cost["efficiency_avg_per_case_pct"] = cost["efficiency_avg_per_case"] * 100.0
    cols = [
        "method",
        "task",
        "mrfh_at_hit_threshold",
        "hit_rate_at_hit_threshold",
        "coverage_pct",
        "efficiency_pct",
        "efficiency_avg_per_case_pct",
        "avg_roi_per_case",
        "avg_16x16_subpatches_per_case",
        "per_roi_total",
        "total_16x16_subpatches",
        "processed_cases",
        "efficiency_cases_with_total_size",
    ]
    return cost[cols].sort_values(["task", "coverage_pct"])


def plot_cost_efficiency(cost: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        3,
        len(TASKS),
        figsize=(17, 11.2),
        constrained_layout=True,
    )
    x = list(range(len(METHODS)))

    for col, task in enumerate(TASKS):
        sub = cost[cost["task"] == task].set_index("method").reindex(METHODS)
        coverage = sub["coverage_pct"].astype(float)
        total_roi = sub["per_roi_total"].astype(float)
        efficiency = sub["efficiency_pct"].astype(float)

        bars0 = axes[0, col].bar(x, coverage, color="#6f7f8f", width=0.64)
        bars1 = axes[1, col].bar(x, total_roi, color="#b27a3c", width=0.64)
        bars2 = axes[2, col].bar(x, efficiency, color="#4f8f72", width=0.64)

        axes[0, col].set_title(TASK_LABELS.get(task, task), pad=8)
        axes[0, col].set_ylabel("Cost: searched coverage (%)")
        axes[1, col].set_ylabel("Cost: total ROI patches")
        axes[2, col].set_ylabel("Efficiency field: unique ROI / slide area (%)")
        axes[2, col].set_xlabel("Method")

        for ax in axes[:, col]:
            ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)

        axes[0, col].bar_label(
            bars0,
            labels=[f"{v:.2f}" if pd.notna(v) else "" for v in coverage],
            padding=3,
            fontsize=8,
        )
        axes[1, col].bar_label(
            bars1,
            labels=[f"{v:,.0f}" if pd.notna(v) else "" for v in total_roi],
            padding=3,
            fontsize=8,
        )
        axes[2, col].bar_label(
            bars2,
            labels=[f"{v:.2f}" if pd.notna(v) else "" for v in efficiency],
            padding=3,
            fontsize=8,
        )
        axes[0, col].set_ylim(0, max(coverage.max() * 1.25, 0.05))
        axes[1, col].set_ylim(0, max(total_roi.max() * 1.25, 1.0))
        axes[2, col].set_ylim(0, max(efficiency.max() * 1.25, 0.05))

    fig.suptitle("Cost Efficiency", fontsize=15)
    fig.savefig(OUT / "cost_efficiency_by_task.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_method_drops(curves: pd.DataFrame, baseline: float = 0.05) -> pd.DataFrame:
    precision_curves = curves[curves["threshold_type"] == "precision"].copy()
    base = precision_curves[precision_curves["precision_threshold"].round(8) == round(baseline, 8)][
        ["method", "task", "hit_rate_pct", "mrfh"]
    ].rename(columns={"hit_rate_pct": "hit_rate_base", "mrfh": "mrfh_base"})
    drops = precision_curves.merge(base, on=["method", "task"], how="left")
    drops["baseline_threshold"] = baseline
    drops["hit_rate_drop_pp"] = drops["hit_rate_base"] - drops["hit_rate_pct"]
    drops["mrfh_drop"] = drops["mrfh_base"] - drops["mrfh"]
    return drops.sort_values(["task", "method", "precision_threshold"])


def plot_method_drops(method_drops: pd.DataFrame) -> None:
    plot_data = method_drops[method_drops["precision_threshold"] >= 0.05].copy()
    fig, axes = plt.subplots(
        len(TASKS),
        2,
        figsize=(15, 12),
        sharex=True,
        constrained_layout=True,
    )

    for row, task in enumerate(TASKS):
        sub = plot_data[plot_data["task"] == task]
        for method in METHODS:
            m = sub[sub["method"] == method]
            if m.empty:
                continue
            label = METHOD_LABELS.get(method, method).replace("\n", " ")
            axes[row, 0].plot(
                m["precision_threshold"],
                m["hit_rate_drop_pp"],
                marker="o",
                ms=3,
                lw=1.8,
                label=label,
            )
            axes[row, 1].plot(
                m["precision_threshold"],
                m["mrfh_drop"],
                marker="o",
                ms=3,
                lw=1.8,
                label=label,
            )

        axes[row, 0].set_ylabel(f"{TASK_LABELS.get(task, task)}\nHit-rate drop pp")
        axes[row, 1].set_ylabel("MRfH drop")
        for ax in axes[row, :]:
            ax.grid(alpha=0.25)
            ax.set_axisbelow(True)

    axes[0, 0].set_title("Hit-rate drop")
    axes[0, 1].set_title("MRfH drop")
    axes[-1, 0].set_xlabel("Precision threshold")
    axes[-1, 1].set_xlabel("Precision threshold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Method Drop by Threshold", fontsize=15)
    fig.savefig(OUT / "method_drop_by_task_threshold.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_method_drops(method_drops: pd.DataFrame) -> pd.DataFrame:
    nonzero = method_drops[method_drops["precision_threshold"] >= 0.05].copy()
    return (
        nonzero.groupby(["method", "task"], as_index=False)
        .agg(
            mean_hit_rate_drop_pp=("hit_rate_drop_pp", "mean"),
            max_hit_rate_drop_pp=("hit_rate_drop_pp", "max"),
            mean_mrfh_drop=("mrfh_drop", "mean"),
            max_mrfh_drop=("mrfh_drop", "max"),
        )
        .sort_values(["task", "max_hit_rate_drop_pp"], ascending=[True, False])
    )


def choose_thresholds(curves: pd.DataFrame, baseline: float = 0.05) -> pd.DataFrame:
    curves = curves[curves["threshold_type"] == "precision"].copy()
    base = curves[curves["precision_threshold"].round(8) == round(baseline, 8)][
        ["method", "task", "hit_rate_pct", "mrfh"]
    ].rename(columns={"hit_rate_pct": "hit_rate_base", "mrfh": "mrfh_base"})
    merged = curves.merge(base, on=["method", "task"], how="left")
    merged["hit_rate_drop_pp"] = merged["hit_rate_base"] - merged["hit_rate_pct"]
    merged["mrfh_drop"] = merged["mrfh_base"] - merged["mrfh"]

    rows = []
    thresholds = sorted(merged["precision_threshold"].unique())
    for thr in thresholds:
        if thr < 0.05:
            continue
        s = merged[merged["precision_threshold"] == thr]
        rows.append(
            {
                "precision_threshold": thr,
                "mean_hit_rate_pct": s["hit_rate_pct"].mean(),
                "min_hit_rate_pct": s["hit_rate_pct"].min(),
                "mean_hit_rate_drop_pp": s["hit_rate_drop_pp"].mean(),
                "max_hit_rate_drop_pp": s["hit_rate_drop_pp"].max(),
                "mean_mrfh": s["mrfh"].mean(),
                "mean_mrfh_drop": s["mrfh_drop"].mean(),
                "max_mrfh_drop": s["mrfh_drop"].max(),
            }
        )
    table = pd.DataFrame(rows)
    table["reviewer_friendly_score"] = (
        table["precision_threshold"] * 10
        - table["mean_hit_rate_drop_pp"]
        - 2 * table["max_hit_rate_drop_pp"]
        - 20 * table["mean_mrfh_drop"]
    )
    return table.sort_values(["reviewer_friendly_score", "precision_threshold"], ascending=[False, False])


def plot_threshold_recommendation(threshold_table: pd.DataFrame, curves: pd.DataFrame | None = None) -> None:
    if curves is None or curves.empty:
        return

    precision_curves = curves[curves["threshold_type"] == "precision"].copy()
    base = precision_curves[precision_curves["precision_threshold"].round(8) == 0.05][
        ["method", "task", "hit_rate_pct", "mrfh"]
    ].rename(columns={"hit_rate_pct": "hit_rate_base", "mrfh": "mrfh_base"})
    merged = precision_curves.merge(base, on=["method", "task"], how="left")
    merged["hit_rate_drop_pp"] = merged["hit_rate_base"] - merged["hit_rate_pct"]
    merged["mrfh_drop"] = merged["mrfh_base"] - merged["mrfh"]
    merged = merged[merged["precision_threshold"] >= 0.05]

    fig, axes = plt.subplots(
        1,
        len(TASKS),
        figsize=(17, 5.6),
        sharex=True,
        constrained_layout=True,
    )

    for ax, task in zip(axes, TASKS):
        sub = merged[merged["task"] == task]
        if sub.empty:
            ax.set_title(TASK_LABELS.get(task, task), pad=8)
            ax.text(0.5, 0.5, "No precision-threshold data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        task_summary = (
            sub.groupby("precision_threshold", as_index=False)
            .agg(
                mean_hit_rate_drop_pp=("hit_rate_drop_pp", "mean"),
                max_hit_rate_drop_pp=("hit_rate_drop_pp", "max"),
                mean_mrfh=("mrfh", "mean"),
            )
            .sort_values("precision_threshold")
        )

        ax.plot(
            task_summary["precision_threshold"],
            task_summary["mean_hit_rate_drop_pp"],
            marker="o",
            lw=1.9,
            color="#c75146",
            label="Mean hit-rate drop (pp)",
        )
        ax.plot(
            task_summary["precision_threshold"],
            task_summary["max_hit_rate_drop_pp"],
            marker="o",
            lw=1.9,
            color="#8f2d27",
            label="Max hit-rate drop (pp)",
        )
        ax.set_title(TASK_LABELS.get(task, task), pad=8)
        ax.set_xlabel("Precision threshold")
        ax.set_ylabel("Hit-rate drop pp")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

        ax2 = ax.twinx()
        ax2.plot(
            task_summary["precision_threshold"],
            task_summary["mean_mrfh"],
            marker="s",
            lw=1.8,
            color="#3b6fb6",
            label="Mean MRfH",
        )
        ax2.set_ylabel("Mean MRfH")

        if task == TASKS[0]:
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()

    fig.legend(handles1 + handles2, labels1 + labels2, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Threshold Choice", fontsize=15)
    fig.savefig(OUT / "threshold_choice_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def get_nested_score(obj: dict) -> float | None:
    candidates = [
        obj.get("final_score"),
        obj.get("score"),
        obj.get("matching", {}).get("final_score") if isinstance(obj.get("matching"), dict) else None,
    ]
    for value in candidates:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def normalize_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "correct"}
    if value is None:
        return None
    return bool(value)


def load_cpathagent_camelyon_output_correct() -> dict[str, bool]:
    output_dir = BASE / "CPathAgent" / "output" / "CAMELYON16"
    correct_by_case = {}
    if not output_dir.exists():
        return correct_by_case
    for path in sorted(output_dir.glob("*.json")):
        try:
            obj = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON: {path}")
            continue
        if not isinstance(obj, dict):
            continue
        case_id = obj.get("case_id") or path.stem.split("_DetectionLocalization")[0]
        correct = normalize_bool(obj.get("correct"))
        if case_id and correct is not None:
            correct_by_case[str(case_id)] = correct
    return correct_by_case


def load_evidence_scores() -> pd.DataFrame:
    rows = []
    cpath_camelyon_correct = load_cpathagent_camelyon_output_correct()
    for method in METHODS:
        for task in TASKS:
            match_dir = BASE / method / "post_eval" / task / "Evidence" / "Matching"
            if not match_dir.exists():
                continue
            for path in sorted(match_dir.glob("*.json")):
                try:
                    obj = json.loads(path.read_text())
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON: {path}")
                    continue
                if not isinstance(obj, dict):
                    continue
                final_score = get_nested_score(obj)
                case_id = obj.get("case_id")
                correct_norm = normalize_bool(obj.get("correct"))
                correct_source = "matching_json"
                if method == "CPathAgent" and task == "CAMELYON16_detection" and case_id in cpath_camelyon_correct:
                    correct_norm = cpath_camelyon_correct[case_id]
                    correct_source = "CPathAgent/output/CAMELYON16"
                rows.append(
                    {
                        "method": method,
                        "task": task,
                        "file": str(path),
                        "case_id": case_id,
                        "gt_key": obj.get("gt_key"),
                        "correct": correct_norm,
                        "correct_source": correct_source,
                        "final_score": final_score,
                        "facts_count": obj.get("facts_count"),
                    }
                )
    return pd.DataFrame(rows)


def summarize_evidence_scores(evidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = evidence.dropna(subset=["final_score"]).copy()
    summary = (
        valid.groupby(["method", "task"], as_index=False)
        .agg(
            n=("final_score", "size"),
            mean_final_score=("final_score", "mean"),
            median_final_score=("final_score", "median"),
            std_final_score=("final_score", "std"),
            min_final_score=("final_score", "min"),
            max_final_score=("final_score", "max"),
            correct_rate=("correct", "mean"),
        )
        .sort_values(["task", "method"])
    )
    by_correct = (
        valid.groupby(["method", "task", "correct"], dropna=False, as_index=False)
        .agg(
            n=("final_score", "size"),
            mean_final_score=("final_score", "mean"),
            median_final_score=("final_score", "median"),
            std_final_score=("final_score", "std"),
        )
        .sort_values(["task", "method", "correct"])
    )
    return summary, by_correct


def format_score_label(score: float) -> str:
    if pd.isna(score):
        return "NA"
    rounded = round(float(score), 4)
    if abs(rounded - round(rounded)) < 1e-8:
        return str(int(round(rounded)))
    return f"{rounded:g}"


def compute_evidence_score_distribution(evidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = evidence.dropna(subset=["final_score"]).copy()
    valid["score_label"] = valid["final_score"].map(format_score_label)
    totals = valid.groupby(["method", "task"], as_index=False).size().rename(columns={"size": "total"})

    score_dist = (
        valid.groupby(["method", "task", "score_label"], as_index=False)
        .size()
        .merge(totals, on=["method", "task"], how="left")
    )
    score_dist["proportion"] = score_dist["size"] / score_dist["total"]

    score_correct_dist = (
        valid.dropna(subset=["correct"])
        .groupby(["method", "task", "score_label", "correct"], as_index=False)
        .size()
        .merge(totals, on=["method", "task"], how="left")
    )
    score_correct_dist["proportion_of_total"] = score_correct_dist["size"] / score_correct_dist["total"]
    return score_dist.sort_values(["task", "method", "score_label"]), score_correct_dist.sort_values(["task", "method", "score_label", "correct"])


def sorted_score_labels(score_dist: pd.DataFrame) -> list[str]:
    labels = list(score_dist["score_label"].dropna().unique())
    def key(label: str):
        try:
            return (0, float(label))
        except ValueError:
            return (1, label)
    return sorted(labels, key=key)


def plot_evidence_score_distribution(score_dist: pd.DataFrame) -> None:
    labels = sorted_score_labels(score_dist)
    palette = ["#d0d7de", "#8ea6c8", "#4f6f9f", "#2f4c75", "#a3b18a", "#6f8f72"]
    colors = {label: palette[i % len(palette)] for i, label in enumerate(labels)}

    fig, axes = plt.subplots(1, len(TASKS), figsize=(17, 5.0), sharey=True, constrained_layout=True)
    x = list(range(len(METHODS)))
    for ax, task in zip(axes, TASKS):
        bottoms = pd.Series([0.0] * len(METHODS), index=METHODS)
        sub_task = score_dist[score_dist["task"] == task]
        for label in labels:
            vals = (
                sub_task[sub_task["score_label"] == label]
                .set_index("method")
                .reindex(METHODS)["proportion"]
                .fillna(0.0)
            )
            ax.bar(x, vals, bottom=bottoms.values, width=0.64, color=colors[label], label=f"score={label}")
            bottoms = bottoms + vals
        ax.set_title(TASK_LABELS.get(task, task), pad=8)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[0].set_ylabel("Share of evidence matching cases")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=min(len(labels), 6), frameon=False)
    fig.suptitle("Evidence Score Distribution", fontsize=15)
    fig.savefig(OUT / "evidence_score_distribution_by_method.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_evidence_score_correct_distribution(score_correct_dist: pd.DataFrame) -> None:
    labels = sorted_score_labels(score_correct_dist)
    colors = {False: "#c75146", True: "#3f8f58"}
    names = {False: "incorrect", True: "correct"}

    fig, axes = plt.subplots(len(TASKS), len(METHODS), figsize=(18, 10.5), sharey=True, constrained_layout=True)
    if len(TASKS) == 1:
        axes = [axes]

    xpos = list(range(len(labels)))
    width = 0.36
    for row, task in enumerate(TASKS):
        for col, method in enumerate(METHODS):
            ax = axes[row][col]
            sub = score_correct_dist[(score_correct_dist["task"] == task) & (score_correct_dist["method"] == method)]
            for i, correct_value in enumerate([False, True]):
                vals = (
                    sub[sub["correct"] == correct_value]
                    .set_index("score_label")
                    .reindex(labels)["proportion_of_total"]
                    .fillna(0.0)
                )
                offsets = [x + (i - 0.5) * width for x in xpos]
                ax.bar(
                    offsets,
                    vals,
                    width=width,
                    color=colors[correct_value],
                    edgecolor="white",
                    linewidth=0.4,
                    label=names[correct_value],
                )
            if row == 0:
                ax.set_title(METHOD_LABELS.get(method, method).replace("\n", " "), pad=8)
            if col == 0:
                ax.set_ylabel(f"{TASK_LABELS.get(task, task)}\nShare of total")
            ax.set_xticks(xpos, labels)
            ax.set_xlabel("final_score")
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Evidence Score by Correctness", fontsize=15)
    fig.savefig(OUT / "evidence_score_correct_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_evidence_overall(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(TASKS), figsize=(17, 4.8), sharey=True, constrained_layout=True)
    x = list(range(len(METHODS)))
    for ax, task in zip(axes, TASKS):
        sub = summary[summary["task"] == task].set_index("method").reindex(METHODS)
        vals = sub["mean_final_score"].astype(float)
        bars = ax.bar(x, vals, color="#5c6f9f", width=0.64)
        ax.set_title(TASK_LABELS.get(task, task), pad=8)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
        ax.set_ylim(0, max(vals.max() * 1.2, 1.0))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.bar_label(
            bars,
            labels=[f"{v:.2f}" if pd.notna(v) else "" for v in vals],
            padding=3,
            fontsize=8,
        )
    axes[0].set_ylabel("Mean evidence final_score")
    fig.suptitle("Evidence Matching Score", fontsize=15)
    fig.savefig(OUT / "evidence_final_score_by_task_method.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_evidence_score_by_correct(by_correct: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(TASKS), figsize=(17, 5.2), sharey=True, constrained_layout=True)
    x = list(range(len(METHODS)))
    width = 0.34
    colors = {False: "#c75146", True: "#3f8f58"}
    labels = {False: "incorrect", True: "correct"}
    for ax, task in zip(axes, TASKS):
        sub_task = by_correct[by_correct["task"] == task]
        for i, correct_value in enumerate([False, True]):
            sub = sub_task[sub_task["correct"] == correct_value].set_index("method").reindex(METHODS)
            vals = sub["mean_final_score"].astype(float)
            offsets = [v + (i - 0.5) * width for v in x]
            bars = ax.bar(offsets, vals, width=width, color=colors[correct_value], label=labels[correct_value])
            ax.bar_label(
                bars,
                labels=[f"{v:.2f}" if pd.notna(v) else "" for v in vals],
                padding=3,
                fontsize=7,
            )
        ax.set_title(TASK_LABELS.get(task, task), pad=8)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean evidence final_score")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Evidence Score by Correctness", fontsize=15)
    fig.savefig(OUT / "evidence_final_score_by_correct.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_evidence_correct_counts(evidence: pd.DataFrame) -> None:
    valid = evidence.dropna(subset=["correct"]).copy()
    counts = valid.groupby(["method", "task", "correct"], as_index=False).size()
    fig, axes = plt.subplots(1, len(TASKS), figsize=(17, 4.8), sharey=True, constrained_layout=True)
    x = list(range(len(METHODS)))
    for ax, task in zip(axes, TASKS):
        sub_task = counts[counts["task"] == task]
        incorrect = sub_task[sub_task["correct"] == False].set_index("method").reindex(METHODS)["size"].fillna(0)
        correct = sub_task[sub_task["correct"] == True].set_index("method").reindex(METHODS)["size"].fillna(0)
        ax.bar(x, incorrect, color="#c75146", width=0.64, label="incorrect")
        ax.bar(x, correct, bottom=incorrect, color="#3f8f58", width=0.64, label="correct")
        ax.set_title(TASK_LABELS.get(task, task), pad=8)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Evidence matching JSON count")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Evidence Correct Counts", fontsize=15)
    fig.savefig(OUT / "evidence_correct_counts_by_method.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_score_by_category(evidence: pd.DataFrame) -> None:
    """Plot 1: mean final_score per gt_key category, split by correct/incorrect.

    Color encodes correct (green) vs incorrect (red).
    Hatch pattern encodes method so bars remain distinguishable in greyscale.
    """
    valid = evidence.dropna(subset=["final_score", "correct"]).copy()
    multi_cat_tasks = ["TCGA_BRCA_subtype", "TCGA_LUNG_Classification"]
    sub = valid[valid["task"].isin(multi_cat_tasks)]
    if sub.empty:
        return

    grouped = (
        sub.groupby(["task", "method", "gt_key", "correct"], as_index=False)
        .agg(mean_score=("final_score", "mean"), n=("final_score", "size"))
    )

    # Per-method visual encoding: hatch + edge colour so bars are distinct even when same fill color
    method_style = {
        "slide_seek":    {"hatch": "",     "edgecolor": "#222222", "lw": 0.8},
        "Pathology-CoT": {"hatch": "///",  "edgecolor": "#1a1aff", "lw": 0.8},
        "PathAgent":     {"hatch": "xxx",  "edgecolor": "#cc6600", "lw": 0.8},
        "CPathAgent":    {"hatch": "...",  "edgecolor": "#880088", "lw": 0.8},
    }
    colors_correct = {False: "#e07070", True: "#5cb87a"}

    n_tasks = len(multi_cat_tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(8 * n_tasks, 6), constrained_layout=True)
    if n_tasks == 1:
        axes = [axes]

    for ax, task in zip(axes, multi_cat_tasks):
        task_data = grouped[grouped["task"] == task]
        categories = sorted(task_data["gt_key"].dropna().unique())
        n_methods = len(METHODS)
        bar_width = 0.09
        gap_between_pairs = 0.02   # gap between incorrect/correct within same method
        gap_between_methods = 0.05 # gap between methods
        gap_between_cats = 0.3     # gap between categories
        xticks, xlabels = [], []

        for ci, cat in enumerate(categories):
            cat_data = task_data[task_data["gt_key"] == cat]
            # start x for this category
            pair_width = bar_width * 2 + gap_between_pairs
            group_w = n_methods * pair_width + (n_methods - 1) * gap_between_methods
            cat_start = ci * (group_w + gap_between_cats)

            for mi, method in enumerate(METHODS):
                method_data = cat_data[cat_data["method"] == method]
                style = method_style.get(method, {"hatch": "", "edgecolor": "#000000", "lw": 0.8})
                x_base = cat_start + mi * (pair_width + gap_between_methods)

                for bi, correct_val in enumerate([False, True]):
                    row = method_data[method_data["correct"] == correct_val]
                    score = float(row["mean_score"].iloc[0]) if not row.empty else 0.0
                    x_pos = x_base + bi * (bar_width + gap_between_pairs)
                    bar = ax.bar(
                        x_pos, score, width=bar_width,
                        color=colors_correct[correct_val],
                        hatch=style["hatch"],
                        edgecolor=style["edgecolor"],
                        linewidth=style["lw"],
                        alpha=0.88,
                    )
                    if score > 0:
                        ax.text(x_pos, score + 0.01, f"{score:.2f}", ha="center", va="bottom",
                                fontsize=5.5, rotation=90)

            # x-tick at center of method group
            group_center = cat_start + group_w / 2
            xticks.append(group_center)
            xlabels.append(cat)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Mean evidence final_score")
        ax.set_title(TASK_LABELS.get(task, task), pad=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        # Add note if a method has no data for a category
        for cat in categories:
            for method in METHODS:
                subset = grouped[(grouped["task"] == task) & (grouped["method"] == method) & (grouped["gt_key"] == cat)]
                if subset.empty:
                    ax.annotate(f"{METHOD_LABELS[method].replace(chr(10),' ')} has no\n{cat} evidence",
                                xy=(0.02, 0.97), xycoords="axes fraction",
                                fontsize=6, va="top", color="#888888",
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

    # Legend section 1: correct / incorrect fill colour
    from matplotlib.patches import Patch
    color_handles = [
        Patch(facecolor=colors_correct[False], edgecolor="#555", label="Incorrect"),
        Patch(facecolor=colors_correct[True],  edgecolor="#555", label="Correct"),
    ]
    # Legend section 2: method hatch
    method_handles = [
        Patch(facecolor="#cccccc",
              hatch=method_style[m]["hatch"],
              edgecolor=method_style[m]["edgecolor"],
              linewidth=method_style[m]["lw"],
              label=METHOD_LABELS[m].replace("\n", " "))
        for m in METHODS
    ]
    fig.legend(handles=color_handles + method_handles,
               loc="outside lower center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle("Evidence Score by Category", fontsize=13)
    fig.savefig(OUT / "evidence_score_by_category_correct.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_score_correct_ratio(evidence: pd.DataFrame) -> None:
    """Plot 2: for each discrete score value, correct/incorrect ratio per method/task."""
    valid = evidence.dropna(subset=["final_score", "correct"]).copy()
    valid["score_bin"] = valid["final_score"].round(2).astype(str)
    score_order = ["0.0", "0.5", "1.0"]

    counts = (
        valid.groupby(["task", "method", "score_bin", "correct"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    totals = counts.groupby(["task", "method", "score_bin"])["n"].sum().reset_index(name="total")
    counts = counts.merge(totals, on=["task", "method", "score_bin"])
    counts["ratio"] = counts["n"] / counts["total"]

    fig, axes = plt.subplots(
        len(TASKS), len(METHODS),
        figsize=(4.5 * len(METHODS), 3.5 * len(TASKS)),
        constrained_layout=True, sharey=True,
    )

    colors = {False: "#c75146", True: "#3f8f58"}
    x = list(range(len(score_order)))

    for ri, task in enumerate(TASKS):
        for ci, method in enumerate(METHODS):
            ax = axes[ri, ci]
            sub = counts[(counts["task"] == task) & (counts["method"] == method)]
            bottom = [0.0] * len(score_order)
            for correct_val, label in [(False, "Incorrect"), (True, "Correct")]:
                vals = []
                for s in score_order:
                    row = sub[(sub["score_bin"] == s) & (sub["correct"] == correct_val)]
                    vals.append(float(row["ratio"].iloc[0]) if not row.empty else 0.0)
                bars = ax.bar(x, vals, bottom=bottom, color=colors[correct_val],
                              width=0.6, label=label)
                bottom = [b + v for b, v in zip(bottom, vals)]
            # total n labels
            for xi, s in enumerate(score_order):
                total_row = sub[sub["score_bin"] == s]["total"]
                n_total = int(total_row.iloc[0]) if not total_row.empty else 0
                ax.text(xi, 1.02, f"n={n_total}", ha="center", va="bottom", fontsize=6)

            ax.set_xticks(x, score_order)
            ax.set_ylim(0, 1.12)
            ax.set_xlabel("Evidence score")
            if ci == 0:
                ax.set_ylabel(f"{TASK_LABELS.get(task, task)}\nRatio")
            if ri == 0:
                ax.set_title(METHOD_LABELS.get(method, method))
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)

    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Correct Ratio by Score", fontsize=13)
    fig.savefig(OUT / "correct_ratio_by_score_value.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_acc_score1_vs_all(evidence: pd.DataFrame) -> None:
    """Plot 3: accuracy at evidence_score=1 vs overall accuracy."""
    valid = evidence.dropna(subset=["final_score", "correct"]).copy()

    rows = []
    for method in METHODS:
        for task in TASKS:
            sub = valid[(valid["method"] == method) & (valid["task"] == task)]
            if sub.empty:
                continue
            overall_acc = sub["correct"].mean()
            filtered = sub[sub["final_score"] == 1.0]
            filtered_acc = filtered["correct"].mean() if not filtered.empty else float("nan")
            filtered_n = len(filtered)
            total_n = len(sub)
            rows.append({
                "method": method,
                "task": task,
                "overall_acc": overall_acc,
                "score1_acc": filtered_acc,
                "total_n": total_n,
                "score1_n": filtered_n,
            })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(TASKS), figsize=(5.5 * len(TASKS), 5.2),
                             sharey=True, constrained_layout=True)
    x = list(range(len(METHODS)))
    width = 0.35
    colors_acc = {"overall": "#5c6f9f", "score1": "#e8a838"}

    for ax, task in zip(axes, TASKS):
        sub = df[df["task"] == task].set_index("method").reindex(METHODS)
        overall = sub["overall_acc"].astype(float)
        score1 = sub["score1_acc"].astype(float)
        score1_n = sub["score1_n"].fillna(0).astype(int)
        total_n = sub["total_n"].fillna(0).astype(int)

        b1 = ax.bar([v - width / 2 for v in x], overall, width=width,
                    color=colors_acc["overall"], label="All cases")
        b2 = ax.bar([v + width / 2 for v in x], score1, width=width,
                    color=colors_acc["score1"], label="Score = 1 only")

        score1_ratio = (score1_n / total_n.replace(0, float("nan"))).fillna(float("nan"))
        for xi, (ov, s1, ratio) in enumerate(zip(overall, score1, score1_ratio)):
            if pd.notna(ov):
                ax.text(xi - width / 2, ov + 0.01, f"{ov:.2f}",
                        ha="center", va="bottom", fontsize=7.5)
            if pd.notna(s1):
                ax.text(xi + width / 2, s1 + 0.01, f"{s1:.2f}\n({ratio:.0%})",
                        ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Accuracy")
        ax.set_title(TASK_LABELS.get(task, task), pad=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Accuracy by Score 1", fontsize=13)
    fig.savefig(OUT / "acc_score1_vs_all.png", dpi=220, bbox_inches="tight")
    plt.close(fig)



def roi_overlap_results_path(method: str, task: str) -> Path | None:
    roi_dir = BASE / method / "post_eval" / task / "ROI"
    for name in ["roi_overlap_results.csv", "roi_overlap_result.csv"]:
        path = roi_dir / name
        if path.exists():
            return path
    return None



def roi_case_ids(method: str, task: str) -> set[str]:
    path = roi_overlap_results_path(method, task)
    if path is None:
        return set()
    try:
        roi = pd.read_csv(path, usecols=["case"])
    except Exception:
        return set()
    return set(roi["case"].dropna().astype(str))


def filter_matching_to_roi_cases(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data.empty or "case_id" not in data.columns:
        return data, pd.DataFrame()
    rows = []
    summaries = []
    for method in METHODS:
        for task in TASKS:
            sub = data[(data["method"] == method) & (data["task"] == task)].copy()
            if sub.empty:
                continue
            roi_cases = roi_case_ids(method, task)
            sub["case_id"] = sub["case_id"].astype(str)
            kept = sub[sub["case_id"].isin(roi_cases)].copy()
            rows.append(kept)
            summaries.append(
                {
                    "method": method,
                    "task": task,
                    "roi_cases": len(roi_cases),
                    "matching_json_before": len(sub),
                    "matching_unique_cases_before": sub["case_id"].nunique(),
                    "matching_json_after": len(kept),
                    "matching_unique_cases_after": kept["case_id"].nunique(),
                }
            )
    if not rows:
        return pd.DataFrame(columns=data.columns), pd.DataFrame(summaries)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries).sort_values(["task", "method"])

def load_roi_case_scores_with_matching_correct(evidence: pd.DataFrame) -> pd.DataFrame:
    matching_correct = (
        evidence.dropna(subset=["case_id", "correct"])
        .assign(case_id=lambda df: df["case_id"].astype(str))
        .drop_duplicates(["method", "task", "case_id"], keep="last")
        [["method", "task", "case_id", "correct"]]
        .rename(columns={"case_id": "case"})
    )

    rows = []
    for method in METHODS:
        for task in TASKS:
            path = roi_overlap_results_path(method, task)
            if path is None:
                continue
            roi = pd.read_csv(path)
            if roi.empty or "case" not in roi.columns:
                continue

            for col in ["precision", "recall", "iou", "hit", "agg_precision", "agg_recall", "agg_iou"]:
                if col in roi.columns:
                    roi[col] = pd.to_numeric(roi[col], errors="coerce")

            aggregations = {
                "roi_rows": ("roi", "size") if "roi" in roi.columns else ("case", "size"),
                "max_precision": ("precision", "max"),
                "mean_precision": ("precision", "mean"),
                "max_recall": ("recall", "max"),
                "mean_recall": ("recall", "mean"),
                "max_iou": ("iou", "max"),
                "mean_iou": ("iou", "mean"),
                "any_hit": ("hit", "max"),
                "agg_precision": ("agg_precision", "max"),
                "agg_recall": ("agg_recall", "max"),
                "agg_iou": ("agg_iou", "max"),
            }
            existing = {
                out_col: spec
                for out_col, spec in aggregations.items()
                if spec[0] in roi.columns
            }
            case_scores = roi.groupby("case", as_index=False).agg(**existing)
            case_scores["method"] = method
            case_scores["task"] = task
            case_scores["roi_source"] = str(path)
            rows.append(case_scores)

    if not rows:
        return pd.DataFrame()

    scores = pd.concat(rows, ignore_index=True)
    scores["case"] = scores["case"].astype(str)
    scores = scores.merge(matching_correct, on=["method", "task", "case"], how="left")
    scores["has_matching_correct"] = scores["correct"].notna()
    return scores.sort_values(["task", "method", "case"])


def summarize_roi_scores_by_matching_correct(roi_case_scores: pd.DataFrame) -> pd.DataFrame:
    if roi_case_scores.empty:
        return pd.DataFrame()
    valid = roi_case_scores.dropna(subset=["correct"]).copy()
    score_cols = [
        col
        for col in ["agg_precision", "agg_recall", "agg_iou", "max_precision", "mean_precision", "any_hit"]
        if col in valid.columns
    ]
    if valid.empty or not score_cols:
        return pd.DataFrame()

    grouped = (
        valid.groupby(["method", "task", "correct"], as_index=False)
        .agg(
            n=("case", "size"),
            **{f"mean_{col}": (col, "mean") for col in score_cols},
            **{f"median_{col}": (col, "median") for col in score_cols},
        )
    )

    if "mean_agg_precision" in grouped.columns:
        wide = grouped.pivot(index=["method", "task"], columns="correct", values="mean_agg_precision").reset_index()
        if False in wide.columns and True in wide.columns:
            wide["mean_agg_precision_diff_correct_minus_incorrect"] = wide[True] - wide[False]
            wide = wide[["method", "task", "mean_agg_precision_diff_correct_minus_incorrect"]]
            grouped = grouped.merge(wide, on=["method", "task"], how="left")
    return grouped.sort_values(["task", "method", "correct"])


def plot_roi_score_by_matching_correct(summary: pd.DataFrame) -> None:
    if summary.empty or "mean_agg_precision" not in summary.columns:
        return

    fig, axes = plt.subplots(
        1,
        len(TASKS),
        figsize=(17, 5.4),
        sharey=True,
        constrained_layout=True,
    )
    width = 0.34
    x = list(range(len(METHODS)))
    colors = {False: "#c75146", True: "#3f8f58"}
    labels = {False: "incorrect", True: "correct"}

    for ax, task in zip(axes, TASKS):
        sub_task = summary[summary["task"] == task]
        for i, correct_value in enumerate([False, True]):
            sub = sub_task[sub_task["correct"] == correct_value].set_index("method").reindex(METHODS)
            vals = sub["mean_agg_precision"].astype(float)
            offsets = [v + (i - 0.5) * width for v in x]
            bars = ax.bar(offsets, vals, width=width, color=colors[correct_value], label=labels[correct_value])
            ax.bar_label(
                bars,
                labels=[f"{v:.3f}" if pd.notna(v) else "" for v in vals],
                padding=3,
                fontsize=7,
            )

        if "mean_agg_precision_diff_correct_minus_incorrect" in sub_task.columns:
            diff = (
                sub_task[["method", "mean_agg_precision_diff_correct_minus_incorrect"]]
                .dropna()
                .drop_duplicates("method")
                .set_index("method")
                .reindex(METHODS)["mean_agg_precision_diff_correct_minus_incorrect"]
            )
            for xi, method in enumerate(METHODS):
                value = diff.get(method)
                if pd.notna(value):
                    ax.text(xi, 1.02, f"diff {value:+.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(TASK_LABELS.get(task, task), pad=8)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean ROI agg_precision")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("ROI Score by Correctness", fontsize=15)
    fig.savefig(OUT / "roi_score_by_matching_correct.png", dpi=220, bbox_inches="tight")
    plt.close(fig)




def compute_roi_aligned_accuracy(roi_case_scores: pd.DataFrame) -> pd.DataFrame:
    if roi_case_scores.empty:
        return pd.DataFrame()
    valid = roi_case_scores.dropna(subset=["correct"]).copy()
    if valid.empty:
        return pd.DataFrame()
    out = (
        valid.groupby(["method", "task"], as_index=False)
        .agg(
            cases=("case", "size"),
            correct_cases=("correct", "sum"),
            acc=("correct", "mean"),
        )
        .sort_values(["task", "method"])
    )
    return out



def plot_roi_aligned_accuracy(acc: pd.DataFrame) -> None:
    if acc.empty:
        return
    task_order = TASKS
    method_order = ["PathAgent", "Pathology-CoT", "CPathAgent", "slide_seek"]
    method_names = {
        "PathAgent": "PathAgent",
        "Pathology-CoT": "Pathology-CoT",
        "CPathAgent": "CPathAgent",
        "slide_seek": "SlideSeek",
    }
    task_names = {
        "CAMELYON16_detection": "Tumor Detection",
        "TCGA_LUNG_Classification": "Tumor Classification",
        "TCGA_BRCA_subtype": "Subtype Classification",
    }
    colors = {
        "PathAgent": "#1f77b4",
        "Pathology-CoT": "#ff7f0e",
        "CPathAgent": "#2ca02c",
        "slide_seek": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    x = list(range(len(task_order)))
    width = 0.18
    for i, method in enumerate(method_order):
        sub = acc[acc["method"] == method].set_index("task").reindex(task_order)
        vals = sub["acc"].astype(float)
        offsets = [v + (i - (len(method_order) - 1) / 2) * width for v in x]
        bars = ax.bar(offsets, vals, width=width, color=colors[method], label=method_names[method])
        ax.bar_label(bars, labels=[f"{v:.3f}" if pd.notna(v) else "" for v in vals], padding=3, fontsize=7)

    ax.set_title("Accuracy", fontsize=13, pad=8)
    ax.text(0.99, 0.98, "higher is better", transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x, [task_names[t] for t in task_order])
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    fig.legend(loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    fig.savefig(OUT / "accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_roi_hit_rate(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        return
    task_order = [
        "CAMELYON16_detection",
        "TCGA_LUNG_Classification",
        "TCGA_BRCA_subtype",
    ]
    method_order = ["PathAgent", "Pathology-CoT", "CPathAgent", "slide_seek"]
    method_names = {
        "PathAgent": "PathAgent",
        "Pathology-CoT": "Pathology-CoT",
        "CPathAgent": "CPathAgent",
        "slide_seek": "SlideSeek",
    }
    task_names = {
        "CAMELYON16_detection": "Tumor Detection",
        "TCGA_LUNG_Classification": "Tumor Classification",
        "TCGA_BRCA_subtype": "Subtype Classification",
    }
    colors = {
        "PathAgent": "#1f77b4",
        "Pathology-CoT": "#ff7f0e",
        "CPathAgent": "#2ca02c",
        "slide_seek": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    x = list(range(len(task_order)))
    width = 0.18
    for i, method in enumerate(method_order):
        sub = (
            metrics[metrics["method"] == method]
            .drop_duplicates("task")
            .set_index("task")
            .reindex(task_order)
        )
        vals = sub["hit_rate_at_hit_threshold"].astype(float) / 100.0
        offsets = [v + (i - (len(method_order) - 1) / 2) * width for v in x]
        bars = ax.bar(
            offsets,
            vals,
            width=width,
            color=colors[method],
            label=method_names[method],
        )
        ax.bar_label(
            bars,
            labels=[f"{v:.3f}" if pd.notna(v) else "" for v in vals],
            padding=3,
            fontsize=7,
        )

    ax.set_title("ROI Hit Rate: Frequent ROI Visitation", fontsize=13, pad=8)
    ax.text(
        0.99,
        0.98,
        "higher is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )
    ax.set_ylabel("ROI Hit Rate")
    ax.set_xticks(x, [task_names[t] for t in task_order])
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    fig.legend(loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    fig.savefig(OUT / "roi_hit_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_hitrate_precision_accuracy_scatter(
    metrics: pd.DataFrame,
    accuracy: pd.DataFrame,
) -> None:
    required_metrics = {
        "method",
        "task",
        "hit_rate_at_hit_threshold",
        "case_micro_precision",
    }
    required_accuracy = {"method", "task", "acc"}
    if (
        metrics.empty
        or accuracy.empty
        or not required_metrics.issubset(metrics.columns)
        or not required_accuracy.issubset(accuracy.columns)
    ):
        return

    data = metrics[list(required_metrics)].merge(
        accuracy[list(required_accuracy)],
        on=["method", "task"],
        how="inner",
    )
    data = data.dropna(
        subset=["hit_rate_at_hit_threshold", "case_micro_precision", "acc"]
    ).copy()
    if data.empty:
        return

    data["hit_rate"] = data["hit_rate_at_hit_threshold"].astype(float) / 100.0
    data["roi_precision"] = data["case_micro_precision"].astype(float)
    data["final_accuracy"] = data["acc"].astype(float)
    hit_median = data["hit_rate"].median()
    precision_median = data["roi_precision"].median()

    markers = {
        "slide_seek": "o",
        "Pathology-CoT": "s",
        "PathAgent": "^",
        "CPathAgent": "D",
    }
    method_names = {
        "slide_seek": "SlideSeek",
        "Pathology-CoT": "Pathology-CoT",
        "PathAgent": "PathAgent",
        "CPathAgent": "CPathAgent",
    }
    task_short = {
        "CAMELYON16_detection": "Detection",
        "TCGA_BRCA_subtype": "BRCA",
        "TCGA_LUNG_Classification": "LUNG",
    }
    annotation_offsets = {
        "CAMELYON16_detection": (5, 5),
        "TCGA_BRCA_subtype": (5, -12),
        "TCGA_LUNG_Classification": (5, 5),
    }

    fig, ax = plt.subplots(figsize=(10.2, 7.2), constrained_layout=True)
    norm = plt.Normalize(0.0, 1.0)
    cmap = plt.get_cmap("viridis")
    for method in METHODS:
        sub = data[data["method"] == method]
        if sub.empty:
            continue
        ax.scatter(
            sub["hit_rate"],
            sub["roi_precision"],
            c=sub["final_accuracy"],
            cmap=cmap,
            norm=norm,
            marker=markers[method],
            s=125,
            edgecolors="black",
            linewidths=0.7,
            alpha=0.95,
            label=method_names[method],
            zorder=3,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                task_short.get(row["task"], row["task"]),
                (row["hit_rate"], row["roi_precision"]),
                xytext=annotation_offsets.get(row["task"], (5, 5)),
                textcoords="offset points",
                fontsize=8,
                color="#303030",
            )

    ax.axvline(hit_median, color="#555555", linestyle="--", linewidth=1.2, zorder=1)
    ax.axhline(precision_median, color="#555555", linestyle="--", linewidth=1.2, zorder=1)

    x_min, x_max = 0.0, min(1.0, max(0.95, data["hit_rate"].max() + 0.06))
    y_min, y_max = 0.0, min(1.0, max(0.72, data["roi_precision"].max() + 0.08))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    label_style = dict(
        fontsize=9,
        fontweight="bold",
        color="#3f3f3f",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
    )
    ax.text(0.98, 0.97, "frequent-focused localization", transform=ax.transAxes, ha="right", va="top", **label_style)
    ax.text(0.98, 0.03, "broad exploration", transform=ax.transAxes, ha="right", va="bottom", **label_style)
    ax.text(0.02, 0.97, "sparse-focused localization", transform=ax.transAxes, ha="left", va="top", **label_style)
    ax.text(0.02, 0.03, "weak localization", transform=ax.transAxes, ha="left", va="bottom", **label_style)

    ax.set_title("ROI Hit Rate vs ROI Precision", fontsize=14, pad=10)
    ax.set_xlabel("ROI Hit Rate")
    ax.set_ylabel("ROI Precision")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(title="Method", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.13, fraction=0.045)
    colorbar.set_label("Final Accuracy")
    colorbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )

    fig.savefig(
        OUT / "hitrate_precision_accuracy_scatter.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_navigation_pairwise_strategy_scatter(
    metrics: pd.DataFrame,
    accuracy: pd.DataFrame,
) -> None:
    metric_columns = {
        "Hit Rate": "hit_rate_at_hit_threshold",
        "ROI Precision": "case_micro_precision",
        "MRFH": "mrfh_at_hit_threshold",
        "Cost": "coverage_16x16_multiscale",
    }
    required = {"method", "task", *metric_columns.values()}
    if (
        metrics.empty
        or accuracy.empty
        or not required.issubset(metrics.columns)
        or not {"method", "task", "acc"}.issubset(accuracy.columns)
    ):
        return

    data = metrics[["method", "task", *metric_columns.values()]].merge(
        accuracy[["method", "task", "acc"]],
        on=["method", "task"],
        how="inner",
    )
    data = data.rename(
        columns={
            "hit_rate_at_hit_threshold": "Hit Rate",
            "case_micro_precision": "ROI Precision",
            "mrfh_at_hit_threshold": "MRFH",
            "coverage_16x16_multiscale": "Cost",
            "acc": "Final Accuracy",
        }
    )
    data["Hit Rate"] = data["Hit Rate"].astype(float) / 100.0
    data["Cost"] = data["Cost"].astype(float) * 100.0
    data = data.dropna(
        subset=["Hit Rate", "ROI Precision", "MRFH", "Cost", "Final Accuracy"]
    ).copy()
    if data.empty:
        return

    related = OUT / "related"
    related.mkdir(exist_ok=True)
    data.to_csv(related / "navigation_pairwise_strategy_data.csv", index=False)

    pairs = [
        ("Hit Rate", "ROI Precision"),
        ("Hit Rate", "MRFH"),
        ("Hit Rate", "Cost"),
        ("ROI Precision", "MRFH"),
        ("ROI Precision", "Cost"),
        ("MRFH", "Cost"),
    ]
    markers = {
        "slide_seek": "o",
        "Pathology-CoT": "s",
        "PathAgent": "^",
        "CPathAgent": "D",
    }
    method_names = {
        "slide_seek": "SlideSeek",
        "Pathology-CoT": "Pathology-CoT",
        "PathAgent": "PathAgent",
        "CPathAgent": "CPathAgent",
    }
    task_short = {
        "CAMELYON16_detection": "Detection",
        "TCGA_BRCA_subtype": "BRCA",
        "TCGA_LUNG_Classification": "LUNG",
    }
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0.0, 1.0)

    def axis_formatter(metric: str):
        if metric in {"Hit Rate", "ROI Precision"}:
            return plt.FuncFormatter(lambda value, _: f"{value:.0%}")
        if metric == "Cost":
            return plt.FuncFormatter(lambda value, _: f"{value:.0f}%")
        return plt.FuncFormatter(lambda value, _: f"{value:.1f}")

    def limits(values: pd.Series, metric: str) -> tuple[float, float]:
        upper = float(values.max())
        if metric in {"Hit Rate", "ROI Precision", "MRFH"}:
            return 0.0, min(1.0, max(0.1, upper * 1.14))
        return 0.0, max(1.0, upper * 1.14)

    def quadrant_labels(x_metric: str, y_metric: str) -> tuple[str, str, str, str]:
        if x_metric == "Hit Rate" and y_metric == "ROI Precision":
            return (
                "frequent-focused localization",
                "broad exploration",
                "sparse-focused localization",
                "weak localization",
            )
        return (
            f"high {x_metric} / high {y_metric}",
            f"high {x_metric} / low {y_metric}",
            f"low {x_metric} / high {y_metric}",
            f"low {x_metric} / low {y_metric}",
        )

    def draw_panel(ax, x_metric: str, y_metric: str, compact: bool) -> None:
        for method in METHODS:
            sub = data[data["method"] == method]
            if sub.empty:
                continue
            ax.scatter(
                sub[x_metric],
                sub[y_metric],
                c=sub["Final Accuracy"],
                cmap=cmap,
                norm=norm,
                marker=markers[method],
                s=82 if compact else 125,
                edgecolors="black",
                linewidths=0.6,
                alpha=0.95,
                label=method_names[method],
                zorder=3,
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    task_short.get(row["task"], row["task"]),
                    (row[x_metric], row[y_metric]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6 if compact else 8,
                    color="#303030",
                )

        x_median = data[x_metric].median()
        y_median = data[y_metric].median()
        ax.axvline(x_median, color="#555555", linestyle="--", linewidth=1.1, zorder=1)
        ax.axhline(y_median, color="#555555", linestyle="--", linewidth=1.1, zorder=1)
        ax.set_xlim(*limits(data[x_metric], x_metric))
        ax.set_ylim(*limits(data[y_metric], y_metric))
        ax.xaxis.set_major_formatter(axis_formatter(x_metric))
        ax.yaxis.set_major_formatter(axis_formatter(y_metric))
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(f"{x_metric} vs {y_metric}", pad=8)
        ax.grid(alpha=0.18, zorder=0)
        ax.set_axisbelow(True)

        top_right, bottom_right, top_left, bottom_left = quadrant_labels(
            x_metric, y_metric
        )
        style = dict(
            fontsize=6.5 if compact else 8.5,
            fontweight="bold",
            color="#404040",
            bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.5),
        )
        ax.text(0.98, 0.97, top_right, transform=ax.transAxes, ha="right", va="top", **style)
        ax.text(0.98, 0.03, bottom_right, transform=ax.transAxes, ha="right", va="bottom", **style)
        ax.text(0.02, 0.97, top_left, transform=ax.transAxes, ha="left", va="top", **style)
        ax.text(0.02, 0.03, bottom_left, transform=ax.transAxes, ha="left", va="bottom", **style)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)
    for ax, (x_metric, y_metric) in zip(axes.flat, pairs):
        draw_panel(ax, x_metric, y_metric, compact=True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Method", loc="outside lower center", ncol=4, frameon=False)
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=axes, location="right", fraction=0.018, pad=0.025)
    colorbar.set_label("Final Accuracy")
    colorbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    fig.suptitle("Navigation Strategy Pairwise Maps", fontsize=16)
    fig.savefig(OUT / "navigation_pairwise_strategy_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    for x_metric, y_metric in pairs:
        fig, ax = plt.subplots(figsize=(10.2, 7.2), constrained_layout=True)
        draw_panel(ax, x_metric, y_metric, compact=False)
        ax.legend(title="Method", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.13, fraction=0.045)
        colorbar.set_label("Final Accuracy")
        colorbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        filename = f"{x_metric}_vs_{y_metric}_strategy.png".lower().replace(" ", "_")
        fig.savefig(related / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_roi_aligned_accuracy_original(acc: pd.DataFrame) -> None:
    if acc.empty:
        return
    fig, axes = plt.subplots(1, len(TASKS), figsize=(17, 5.0), sharey=True, constrained_layout=True)
    x = list(range(len(METHODS)))
    for ax, task in zip(axes, TASKS):
        sub = acc[acc["task"] == task].set_index("method").reindex(METHODS)
        vals = sub["acc"].astype(float)
        bars = ax.bar(x, vals, width=0.64, color="#4f7f9f")
        ax.bar_label(bars, labels=[f"{v:.1%}" if pd.notna(v) else "" for v in vals], padding=3, fontsize=8)
        ax.set_title(TASK_LABELS.get(task, task), pad=8)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Accuracy", fontsize=15)
    fig.savefig(OUT / "accuracy_original.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def compute_hit_rate_zero_nonzero_case_stats(
    roi_case_scores: pd.DataFrame,
    evidence: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if roi_case_scores.empty or "max_precision" not in roi_case_scores.columns:
        return pd.DataFrame(), pd.DataFrame()

    cases = roi_case_scores.copy()
    cases["hit_at_thr005"] = cases["max_precision"].fillna(0).astype(float).ge(0.05)
    cases["hit_rate_group"] = cases["hit_at_thr005"].map(
        {False: "hit_rate=0", True: "hit_rate!=0"}
    )

    count_rows = []
    for method in METHODS:
        for task in TASKS:
            curve_path = BASE / method / "post_eval" / task / "ROI" / "mrfh_by_threshold.csv"
            if not curve_path.exists():
                sub_cases = cases[(cases["method"] == method) & (cases["task"] == task)]
                total_cases = len(sub_cases)
                hit_cases = int(sub_cases["hit_at_thr005"].sum())
            else:
                curve = pd.read_csv(curve_path)
                row = curve[curve["precision_threshold"].round(8).eq(0.05)]
                if row.empty:
                    row = curve.iloc[[0]]
                row = row.iloc[0]
                hit_cases = int(row["hit_cases"])
                processed = read_summary_value(
                    BASE / method / "post_eval" / task / "ROI" / "roi_overlap_summary.csv",
                    "processed_cases",
                )
                total_cases = int(processed) if processed is not None else int(hit_cases)
            count_rows.extend(
                [
                    {"method": method, "task": task, "hit_rate_group": "hit_rate=0", "cases": max(total_cases - hit_cases, 0), "total_cases": total_cases},
                    {"method": method, "task": task, "hit_rate_group": "hit_rate!=0", "cases": hit_cases, "total_cases": total_cases},
                ]
            )

    summary = pd.DataFrame(count_rows)
    acc = (
        cases.dropna(subset=["correct"])
        .groupby(["method", "task", "hit_rate_group"], as_index=False)
        .agg(
            acc=("correct", "mean"),
            acc_cases=("case", "size"),
            mean_max_precision=("max_precision", "mean"),
            mean_agg_precision=("agg_precision", "mean"),
        )
    )
    summary = summary.merge(acc, on=["method", "task", "hit_rate_group"], how="left")
    summary["case_share"] = summary["cases"] / summary["total_cases"].replace(0, pd.NA)
    order = {"hit_rate=0": 0, "hit_rate!=0": 1}
    summary["_order"] = summary["hit_rate_group"].map(order)
    summary = summary.sort_values(["task", "method", "_order"]).drop(columns="_order")
    return cases.sort_values(["task", "method", "case"]), summary

def plot_hit_rate_zero_nonzero_case_stats(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    groups = ["hit_rate=0", "hit_rate!=0"]
    colors = {"hit_rate=0": "#c75146", "hit_rate!=0": "#3f8f58"}
    fig, axes = plt.subplots(
        2,
        len(TASKS),
        figsize=(17, 8.2),
        sharey="row",
        constrained_layout=True,
    )
    x = list(range(len(METHODS)))
    width = 0.34
    specs = [
        ("case_share", "Case proportion", "{:.1%}"),
        ("acc", "Accuracy from Matching correct", "{:.1%}"),
    ]

    for row, (metric, ylabel, fmt) in enumerate(specs):
        for col, task in enumerate(TASKS):
            ax = axes[row, col]
            sub_task = summary[summary["task"] == task]
            for i, group in enumerate(groups):
                sub = sub_task[sub_task["hit_rate_group"] == group].set_index("method").reindex(METHODS)
                vals = sub[metric].astype(float)
                offsets = [v + (i - 0.5) * width for v in x]
                bars = ax.bar(offsets, vals, width=width, color=colors[group], label=group)
                ax.bar_label(
                    bars,
                    labels=[fmt.format(v) if pd.notna(v) else "" for v in vals],
                    padding=3,
                    fontsize=7,
                )
            if row == 0:
                ax.set_title(TASK_LABELS.get(task, task), pad=8)
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.12)
            ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Hit-rate Groups Threshold 0.05", fontsize=15)
    fig.savefig(OUT / "hit_rate_zero_nonzero_case_share_acc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_correct_only_hit_rate_distribution(hit_rate_group_cases: pd.DataFrame) -> pd.DataFrame:
    if hit_rate_group_cases.empty:
        return pd.DataFrame()
    correct_cases = hit_rate_group_cases[hit_rate_group_cases["correct"] == True].copy()
    if correct_cases.empty:
        return pd.DataFrame()
    totals = correct_cases.groupby(["method", "task"], as_index=False).agg(correct_cases=("case", "size"))
    dist = (
        correct_cases.groupby(["method", "task", "hit_rate_group"], as_index=False)
        .agg(cases=("case", "size"))
        .merge(totals, on=["method", "task"], how="left")
    )
    dist["share_among_correct"] = dist["cases"] / dist["correct_cases"]
    order = {"hit_rate=0": 0, "hit_rate!=0": 1}
    dist["_order"] = dist["hit_rate_group"].map(order)
    return dist.sort_values(["task", "method", "_order"]).drop(columns="_order")


def plot_correct_only_hit_rate_distribution(dist: pd.DataFrame) -> None:
    if dist.empty:
        return
    groups = ["hit_rate=0", "hit_rate!=0"]
    colors = {"hit_rate=0": "#c75146", "hit_rate!=0": "#3f8f58"}
    fig, axes = plt.subplots(1, len(TASKS), figsize=(17, 5.2), sharey=True, constrained_layout=True)
    x = list(range(len(METHODS)))
    width = 0.34

    for ax, task in zip(axes, TASKS):
        sub_task = dist[dist["task"] == task]
        for i, group in enumerate(groups):
            sub = sub_task[sub_task["hit_rate_group"] == group].set_index("method").reindex(METHODS)
            vals = sub["share_among_correct"].astype(float)
            offsets = [v + (i - 0.5) * width for v in x]
            bars = ax.bar(offsets, vals, width=width, color=colors[group], label=group)
            ax.bar_label(
                bars,
                labels=[f"{v:.1%}" if pd.notna(v) else "" for v in vals],
                padding=3,
                fontsize=7,
            )
        ax.set_title(TASK_LABELS.get(task, task), pad=8)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Share among correct cases")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Correct Case Hit-rate Groups", fontsize=15)
    fig.savefig(OUT / "hit_rate_zero_nonzero_correct_case_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)



def compute_all_vs_hit_nonzero_accuracy(hit_rate_group_cases: pd.DataFrame) -> pd.DataFrame:
    if hit_rate_group_cases.empty:
        return pd.DataFrame()
    valid = hit_rate_group_cases.dropna(subset=["correct"]).copy()
    if valid.empty:
        return pd.DataFrame()
    valid["correct_with_hit"] = valid["correct"].astype(bool) & valid["hit_rate_group"].eq("hit_rate!=0")
    out = (
        valid.groupby(["method", "task"], as_index=False)
        .agg(
            all_cases=("case", "size"),
            all_acc=("correct", "mean"),
            hit_nonzero_cases=("correct_with_hit", "sum"),
            hit_nonzero_acc=("correct_with_hit", "mean"),
        )
    )
    hit_cases = (
        valid[valid["hit_rate_group"] == "hit_rate!=0"]
        .groupby(["method", "task"], as_index=False)
        .agg(hit_nonzero_case_count=("case", "size"))
    )
    out = out.merge(hit_cases, on=["method", "task"], how="left")
    out["hit_nonzero_case_count"] = out["hit_nonzero_case_count"].fillna(0).astype(int)
    out["hit_nonzero_case_share"] = out["hit_nonzero_case_count"] / out["all_cases"]
    out["acc_delta"] = out["hit_nonzero_acc"] - out["all_acc"]
    return out.sort_values(["task", "method"])


def plot_all_vs_hit_nonzero_accuracy(comp: pd.DataFrame) -> None:
    if comp.empty:
        return
    fig, axes = plt.subplots(1, len(TASKS), figsize=(17, 5.2), sharey=True, constrained_layout=True)
    x = list(range(len(METHODS)))
    width = 0.34
    for ax, task in zip(axes, TASKS):
        sub = comp[comp["task"] == task].set_index("method").reindex(METHODS)
        all_acc = sub["all_acc"].astype(float)
        hit_acc = sub["hit_nonzero_acc"].astype(float)
        share = sub["hit_nonzero_case_share"].astype(float)
        bars0 = ax.bar([v - width / 2 for v in x], all_acc, width=width, color="#6f7f8f", label="Original acc")
        bars1 = ax.bar([v + width / 2 for v in x], hit_acc, width=width, color="#3f8f58", label="Hit-gated acc")
        ax.bar_label(bars0, labels=[f"{v:.1%}" if pd.notna(v) else "" for v in all_acc], padding=3, fontsize=7)
        ax.bar_label(
            bars1,
            labels=[f"{acc:.1%}\nhit {shr:.1%}" if pd.notna(acc) and pd.notna(shr) else "" for acc, shr in zip(hit_acc, share)],
            padding=3,
            fontsize=7,
        )
        ax.set_title(TASK_LABELS.get(task, task), pad=8)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Accuracy With Hit Gate", fontsize=15)
    fig.savefig(OUT / "all_vs_hit_nonzero_accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def load_roi_navigation_thr005_with_matching_correct(evidence: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    matching_correct = (
        evidence.dropna(subset=["case_id", "correct"])
        .assign(case_id=lambda df: df["case_id"].astype(str))
        .drop_duplicates(["method", "task", "case_id"], keep="last")
        [["method", "task", "case_id", "correct"]]
        .rename(columns={"case_id": "case"})
    )

    rows = []
    for method in METHODS:
        for task in TASKS:
            path = roi_overlap_results_path(method, task)
            if path is None:
                continue
            roi = pd.read_csv(path)
            if roi.empty or "case" not in roi.columns or "precision" not in roi.columns:
                continue
            roi = roi.copy()
            roi["case"] = roi["case"].astype(str)
            for col in ["precision", "recall", "iou", "agg_precision", "agg_recall", "agg_iou"]:
                if col in roi.columns:
                    roi[col] = pd.to_numeric(roi[col], errors="coerce")
            roi["roi_rank"] = roi.groupby("case").cumcount() + 1
            roi["hit_thr005"] = roi["precision"] >= threshold

            case_rows = []
            for case, sub in roi.groupby("case", sort=False):
                hit_rows = sub[sub["hit_thr005"]]
                case_rows.append(
                    {
                        "case": case,
                        "method": method,
                        "task": task,
                        "threshold": threshold,
                        "roi_rows": len(sub),
                        "hit_at_thr005": bool(len(hit_rows) > 0),
                        "first_hit_rank_thr005": float(hit_rows["roi_rank"].iloc[0]) if len(hit_rows) else float("nan"),
                        "max_precision": sub["precision"].max(),
                        "mean_precision": sub["precision"].mean(),
                        "max_precision_ge_thr005": hit_rows["precision"].max() if len(hit_rows) else float("nan"),
                        "mean_precision_ge_thr005": hit_rows["precision"].mean() if len(hit_rows) else float("nan"),
                        "agg_precision": sub["agg_precision"].max() if "agg_precision" in sub.columns else float("nan"),
                        "agg_recall": sub["agg_recall"].max() if "agg_recall" in sub.columns else float("nan"),
                        "agg_iou": sub["agg_iou"].max() if "agg_iou" in sub.columns else float("nan"),
                        "roi_source": str(path),
                    }
                )
            rows.append(pd.DataFrame(case_rows))

    if not rows:
        return pd.DataFrame()
    nav = pd.concat(rows, ignore_index=True)
    nav = nav.merge(matching_correct, on=["method", "task", "case"], how="left")
    nav["has_matching_correct"] = nav["correct"].notna()
    return nav.sort_values(["task", "method", "case"])


def summarize_roi_navigation_thr005(nav: pd.DataFrame) -> pd.DataFrame:
    if nav.empty:
        return pd.DataFrame()
    valid = nav.dropna(subset=["correct"]).copy()
    if valid.empty:
        return pd.DataFrame()
    summary = (
        valid.groupby(["method", "task", "correct"], as_index=False)
        .agg(
            n=("case", "size"),
            hit_rate_thr005=("hit_at_thr005", "mean"),
            mean_first_hit_rank_thr005=("first_hit_rank_thr005", "mean"),
            median_first_hit_rank_thr005=("first_hit_rank_thr005", "median"),
            mean_max_precision=("max_precision", "mean"),
            mean_agg_precision=("agg_precision", "mean"),
        )
    )

    for metric in ["hit_rate_thr005", "mean_first_hit_rank_thr005", "mean_max_precision", "mean_agg_precision"]:
        wide = summary.pivot(index=["method", "task"], columns="correct", values=metric).reset_index()
        if False in wide.columns and True in wide.columns:
            wide[f"{metric}_diff_correct_minus_incorrect"] = wide[True] - wide[False]
            summary = summary.merge(
                wide[["method", "task", f"{metric}_diff_correct_minus_incorrect"]],
                on=["method", "task"],
                how="left",
            )
    return summary.sort_values(["task", "method", "correct"])


def plot_roi_navigation_thr005(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(
        2,
        len(TASKS),
        figsize=(17, 8.2),
        constrained_layout=True,
    )
    width = 0.34
    x = list(range(len(METHODS)))
    colors = {False: "#c75146", True: "#3f8f58"}
    labels = {False: "incorrect", True: "correct"}
    specs = [
        ("hit_rate_thr005", "Hit rate", 1.0, "{:.1%}"),
        ("mean_first_hit_rank_thr005", "Mean first-hit rank", None, "{:.2f}"),
    ]

    for row, (metric, ylabel, fixed_ylim, fmt) in enumerate(specs):
        for col, task in enumerate(TASKS):
            ax = axes[row, col]
            sub_task = summary[summary["task"] == task]
            max_val = 0.0
            for i, correct_value in enumerate([False, True]):
                sub = sub_task[sub_task["correct"] == correct_value].set_index("method").reindex(METHODS)
                vals = sub[metric].astype(float)
                max_val = max(max_val, vals.max(skipna=True) if vals.notna().any() else 0.0)
                offsets = [v + (i - 0.5) * width for v in x]
                bars = ax.bar(offsets, vals, width=width, color=colors[correct_value], label=labels[correct_value])
                ax.bar_label(
                    bars,
                    labels=[fmt.format(v) if pd.notna(v) else "" for v in vals],
                    padding=3,
                    fontsize=7,
                )
            if row == 0:
                ax.set_title(TASK_LABELS.get(task, task), pad=8)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
                ax.set_ylim(0, 1.12)
            else:
                ax.set_ylim(0, max(max_val * 1.25, 1.0))
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("ROI Navigation Threshold 0.05", fontsize=15)
    fig.savefig(OUT / "roi_navigation_metrics_thr005.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def load_uncertainty_scores() -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for task in TASKS:
            match_dir = BASE / method / "post_eval" / task / "Evidence" / "Uncertainty_matching"
            if not match_dir.exists():
                continue
            for path in sorted(match_dir.glob("*.json")):
                try:
                    obj = json.loads(path.read_text())
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON: {path}")
                    continue
                if not isinstance(obj, dict):
                    continue
                rows.append(
                    {
                        "method": method,
                        "task": task,
                        "file": str(path),
                        "case_id": obj.get("case_id"),
                        "gt_key": obj.get("gt_key"),
                        "correct": normalize_bool(obj.get("correct")),
                        "final_score": get_nested_score(obj),
                        "facts_count": obj.get("facts_count"),
                    }
                )
    return pd.DataFrame(rows)


def available_task_method_lists(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    tasks = [task for task in TASKS if task in set(data["task"])]
    methods = [method for method in METHODS if method in set(data["method"])]
    return tasks, methods


def save_empty_uncertainty_note() -> None:
    (OUT / "Uncertainty_no_data.txt").write_text("No Uncertainty_matching JSON files found.\n")


def cpath_camelyon_output_accuracy() -> tuple[float | None, int, int]:
    out_dir = BASE / "CPathAgent" / "output" / "CAMELYON16"
    vals = []
    if not out_dir.exists():
        return None, 0, 0
    for path in sorted(out_dir.glob("*.json")):
        try:
            obj = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("correct") is not None:
            vals.append(bool(obj.get("correct")))
    if not vals:
        return None, 0, 0
    return sum(vals) / len(vals), sum(vals), len(vals)


def compute_task_accuracy() -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for task in TASKS:
            summary_path = BASE / method / "post_eval" / task / "ROI" / "roi_overlap_summary.csv"
            acc = read_summary_value(summary_path, "acc_annotated")
            correct = read_summary_value(summary_path, "qa_correct_cases")
            total = read_summary_value(summary_path, "qa_has_result")
            source = str(summary_path)
            if method == "CPathAgent" and task == "CAMELYON16_detection":
                acc, correct, total = cpath_camelyon_output_accuracy()
                source = str(BASE / "CPathAgent" / "output" / "CAMELYON16")
            rows.append(
                {
                    "method": method,
                    "task": task,
                    "acc": acc,
                    "acc_3dp": None if acc is None else round(acc, 3),
                    "qa_correct_cases": correct,
                    "qa_has_result": total,
                    "source": source,
                }
            )
    return pd.DataFrame(rows)


def compute_score_accuracy_comparison(data: pd.DataFrame, task_accuracy: pd.DataFrame | None = None) -> pd.DataFrame:
    valid = data.dropna(subset=["final_score", "correct"]).copy()
    if valid.empty:
        return pd.DataFrame()
    valid["score_label"] = valid["final_score"].map(format_score_label)
    matching_totals = (
        valid.groupby(["method", "task"], as_index=False)
        .agg(matching_n=("correct", "size"), matching_acc=("correct", "mean"))
    )
    # Any plot labelled "Original acc" must use the same matching JSON correct field
    # as the score-filtered accuracy. Do not mix in ROI-summary task accuracy here.
    overall = matching_totals.copy()
    overall["original_n"] = overall["matching_n"]
    overall["original_acc"] = overall["matching_acc"]
    by_score = (
        valid.groupby(["method", "task", "score_label"], as_index=False)
        .agg(score_n=("correct", "size"), score_acc=("correct", "mean"))
        .merge(overall, on=["method", "task"], how="left")
    )
    by_score["score_share_of_total"] = by_score["score_n"] / by_score["matching_n"]
    by_score["acc_delta_vs_original"] = by_score["score_acc"] - by_score["original_acc"]

    def score_key(label: str):
        try:
            return float(label)
        except (TypeError, ValueError):
            return float("inf")

    by_score["_score_sort"] = by_score["score_label"].map(score_key)
    by_score = by_score.sort_values(["task", "method", "_score_sort"]).drop(columns=["_score_sort"])
    return by_score[
        [
            "method",
            "task",
            "score_label",
            "score_n",
            "score_share_of_total",
            "score_acc",
            "matching_n",
            "matching_acc",
            "original_n",
            "original_acc",
            "acc_delta_vs_original",
        ]
    ]


def plot_score1_accuracy_comparison(score1: pd.DataFrame, prefix: str) -> None:
    if score1.empty:
        return
    tasks = [task for task in TASKS if task in set(score1["task"])]
    methods = [method for method in METHODS if method in set(score1["method"])]

    fig, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(max(6.5, 5.8 * len(tasks)), 5.2),
        sharey=True,
        constrained_layout=True,
    )
    if len(tasks) == 1:
        axes = [axes]

    width = 0.28
    for ax, task in zip(axes, tasks):
        sub = score1[score1["task"] == task].set_index("method").reindex(methods)
        x = list(range(len(methods)))
        original = sub["original_acc"].astype(float)
        score_acc = sub["score_acc"].astype(float)
        share = sub["score_share_of_total"].astype(float)

        bars0 = ax.bar([v - width / 2 for v in x], original, width=width, color="#6f7f8f", label="Original acc")
        bars1 = ax.bar([v + width / 2 for v in x], score_acc, width=width, color="#3f8f58", label="score=1 acc")
        ax.set_title(TASK_LABELS.get(task, task), pad=8)
        ax.set_ylim(0, 1.12)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in methods])
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.bar_label(bars0, labels=[f"{v:.1%}" if pd.notna(v) else "" for v in original], padding=3, fontsize=8)
        ax.bar_label(
            bars1,
            labels=[f"{acc:.1%}\nshare {shr:.1%}" if pd.notna(acc) and pd.notna(shr) else "" for acc, shr in zip(score_acc, share)],
            padding=3,
            fontsize=8,
        )

    axes[0].set_ylabel("Accuracy")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Score 1 Accuracy", fontsize=15)
    safe_prefix = prefix.replace(" ", "_")
    fig.savefig(OUT / f"{safe_prefix}_score1_accuracy_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_overall(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    tasks, methods = available_task_method_lists(summary)
    fig, axes = plt.subplots(1, len(tasks), figsize=(max(5.5, 5.5 * len(tasks)), 4.8), sharey=True, constrained_layout=True)
    if len(tasks) == 1:
        axes = [axes]
    x = list(range(len(methods)))
    for ax, task in zip(axes, tasks):
        sub = summary[summary["task"] == task].set_index("method").reindex(methods)
        vals = sub["mean_final_score"].astype(float)
        bars = ax.bar(x, vals, color="#5c6f9f", width=0.64)
        ax.set_title(TASK_LABELS.get(task, task), pad=8)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in methods])
        ax.set_ylim(0, max(vals.max() * 1.2, 1.0))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.bar_label(bars, labels=[f"{v:.2f}" if pd.notna(v) else "" for v in vals], padding=3, fontsize=8)
    axes[0].set_ylabel("Mean uncertainty final_score")
    fig.suptitle("Uncertainty Score", fontsize=15)
    fig.savefig(OUT / "Uncertainty_evidence_final_score_by_task_method.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_score_by_correct(by_correct: pd.DataFrame) -> None:
    if by_correct.empty:
        return
    tasks, methods = available_task_method_lists(by_correct)
    fig, axes = plt.subplots(1, len(tasks), figsize=(max(5.8, 5.8 * len(tasks)), 5.0), sharey=True, constrained_layout=True)
    if len(tasks) == 1:
        axes = [axes]
    x = list(range(len(methods)))
    width = 0.34
    colors = {False: "#c75146", True: "#3f8f58"}
    labels = {False: "incorrect", True: "correct"}
    for ax, task in zip(axes, tasks):
        sub_task = by_correct[by_correct["task"] == task]
        for i, correct_value in enumerate([False, True]):
            sub = sub_task[sub_task["correct"] == correct_value].set_index("method").reindex(methods)
            vals = sub["mean_final_score"].astype(float)
            offsets = [v + (i - 0.5) * width for v in x]
            bars = ax.bar(offsets, vals, width=width, color=colors[correct_value], label=labels[correct_value])
            ax.bar_label(bars, labels=[f"{v:.2f}" if pd.notna(v) else "" for v in vals], padding=3, fontsize=7)
        ax.set_title(TASK_LABELS.get(task, task), pad=8)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in methods])
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean uncertainty final_score")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Uncertainty Score by Correctness", fontsize=15)
    fig.savefig(OUT / "Uncertainty_evidence_final_score_by_correct.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_correct_counts(data: pd.DataFrame) -> None:
    valid = data.dropna(subset=["correct"]).copy()
    if valid.empty:
        return
    counts = valid.groupby(["method", "task", "correct"], as_index=False).size()
    tasks, methods = available_task_method_lists(valid)
    fig, axes = plt.subplots(1, len(tasks), figsize=(max(5.8, 5.8 * len(tasks)), 4.8), sharey=True, constrained_layout=True)
    if len(tasks) == 1:
        axes = [axes]
    x = list(range(len(methods)))
    for ax, task in zip(axes, tasks):
        sub_task = counts[counts["task"] == task]
        incorrect = sub_task[sub_task["correct"] == False].set_index("method").reindex(methods)["size"].fillna(0)
        correct = sub_task[sub_task["correct"] == True].set_index("method").reindex(methods)["size"].fillna(0)
        ax.bar(x, incorrect, color="#c75146", width=0.64, label="incorrect")
        ax.bar(x, correct, bottom=incorrect, color="#3f8f58", width=0.64, label="correct")
        ax.set_title(TASK_LABELS.get(task, task), pad=8)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in methods])
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Uncertainty matching JSON count")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Uncertainty Correct Counts", fontsize=15)
    fig.savefig(OUT / "Uncertainty_evidence_correct_counts_by_method.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_score_distribution(score_dist: pd.DataFrame) -> None:
    if score_dist.empty:
        return
    tasks, methods = available_task_method_lists(score_dist)
    score_labels = sorted_score_labels(score_dist)
    palette = ["#d0d7de", "#8ea6c8", "#4f6f9f", "#2f4c75", "#a3b18a", "#6f8f72"]
    colors = {label: palette[i % len(palette)] for i, label in enumerate(score_labels)}
    fig, axes = plt.subplots(1, len(tasks), figsize=(max(5.8, 5.8 * len(tasks)), 4.8), sharey=True, constrained_layout=True)
    if len(tasks) == 1:
        axes = [axes]
    x = list(range(len(methods)))
    for ax, task in zip(axes, tasks):
        bottoms = pd.Series([0.0] * len(methods), index=methods)
        sub_task = score_dist[score_dist["task"] == task]
        for label in score_labels:
            vals = sub_task[sub_task["score_label"] == label].set_index("method").reindex(methods)["proportion"].fillna(0.0)
            ax.bar(x, vals, bottom=bottoms.values, width=0.64, color=colors[label], label=f"score={label}")
            bottoms = bottoms + vals
        ax.set_title(TASK_LABELS.get(task, task), pad=8)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in methods])
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[0].set_ylabel("Share of uncertainty matching cases")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=min(len(score_labels), 6), frameon=False)
    fig.suptitle("Uncertainty Score Distribution", fontsize=15)
    fig.savefig(OUT / "Uncertainty_evidence_score_distribution_by_method.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_score_correct_distribution(score_correct_dist: pd.DataFrame) -> None:
    if score_correct_dist.empty:
        return
    tasks, methods = available_task_method_lists(score_correct_dist)
    score_labels = sorted_score_labels(score_correct_dist)
    colors = {False: "#c75146", True: "#3f8f58"}
    names = {False: "incorrect", True: "correct"}
    fig, axes = plt.subplots(len(tasks), len(methods), figsize=(max(5.8, 5.8 * len(methods)), max(4.8, 4.8 * len(tasks))), sharey=True, constrained_layout=True)
    if len(tasks) == 1 and len(methods) == 1:
        axes = [[axes]]
    elif len(tasks) == 1:
        axes = [axes]
    elif len(methods) == 1:
        axes = [[ax] for ax in axes]
    xpos = list(range(len(score_labels)))
    width = 0.36
    for row, task in enumerate(tasks):
        for col, method in enumerate(methods):
            ax = axes[row][col]
            sub = score_correct_dist[(score_correct_dist["task"] == task) & (score_correct_dist["method"] == method)]
            for i, correct_value in enumerate([False, True]):
                vals = sub[sub["correct"] == correct_value].set_index("score_label").reindex(score_labels)["proportion_of_total"].fillna(0.0)
                offsets = [x + (i - 0.5) * width for x in xpos]
                ax.bar(offsets, vals, width=width, color=colors[correct_value], edgecolor="white", linewidth=0.4, label=names[correct_value])
            ax.set_title(f"{TASK_LABELS.get(task, task).replace(chr(10), ' ')} / {METHOD_LABELS.get(method, method).replace(chr(10), ' ')}", pad=8)
            ax.set_xticks(xpos, score_labels)
            ax.set_xlabel("final_score")
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[0][0].set_ylabel("Share of total")
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Uncertainty Score by Correctness", fontsize=15)
    fig.savefig(OUT / "Uncertainty_evidence_score_correct_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "None."
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for value in row:
            if pd.isna(value):
                vals.append("")
            elif isinstance(value, float):
                vals.append(format(value, floatfmt))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(metrics: pd.DataFrame, curves: pd.DataFrame, threshold_table: pd.DataFrame) -> None:
    best = threshold_table.iloc[0]
    robust = threshold_table[
        (threshold_table["precision_threshold"] >= 0.1)
        & (threshold_table["max_hit_rate_drop_pp"] <= 5.0)
        & (threshold_table["mean_mrfh_drop"] <= 0.03)
    ].sort_values("precision_threshold", ascending=False)
    conservative = robust.iloc[0] if not robust.empty else best
    non_precision = curves[curves["threshold_type"] != "precision"][
        ["method", "task", "threshold_type"]
    ].drop_duplicates()

    lines = [
        "# Post-eval Analysis",
        "",
        "## Files generated",
        "- `task_method_mrfh_precision_hitrate.png`: required comparison plot, one panel per task.",
        "- `threshold_sensitivity_by_task.png`: hit-rate and MRfH curves over precision thresholds.",
        "- `threshold_choice_tradeoff.png`: threshold trade-off summary.",
        "- `summary_metrics.csv`: method/task metrics at precision threshold 0.05.",
        "- `threshold_recommendation.csv`: threshold-level robustness table.",
        "",
        "## Threshold recommendation",
        (
            f"I would report a non-zero precision threshold of **{conservative['precision_threshold']:.2f}** "
            f"as the reviewer-friendly default. Across all available method/task pairs, its mean hit rate is "
            f"{conservative['mean_hit_rate_pct']:.2f}%, mean hit-rate drop from threshold 0.05 is "
            f"{conservative['mean_hit_rate_drop_pp']:.2f} pp, worst drop is "
            f"{conservative['max_hit_rate_drop_pp']:.2f} pp, and mean MRfH drop is "
            f"{conservative['mean_mrfh_drop']:.4f}."
        ),
        "",
        "Rationale: threshold 0.05 requires measurable overlap while preserving most of the ranking signal.",
        "",
        "## Method/task summary at threshold 0.05",
        df_to_markdown(metrics.sort_values(["task", "method"]), floatfmt=".4f"),
        "",
        "## Top threshold candidates",
        df_to_markdown(threshold_table.head(8), floatfmt=".4f"),
        "",
        "## Missing-data note",
        "CPathAgent/CAMELYON16_detection has `mrfh_by_threshold.csv` but no separate `hit_rate_by_threshold.csv`; the hit-rate values were read from `mrfh_by_threshold.csv`, which contains the same hit-rate columns.",
        "",
        "The following curve files do not use `precision_threshold`, so they are included in threshold-0 comparison but excluded from the precision-threshold recommendation:",
        df_to_markdown(non_precision) if not non_precision.empty else "None.",
    ]
    (OUT / "analysis_report.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    metrics, curves = load_metrics()
    task_accuracy = compute_task_accuracy()
    task_accuracy.to_csv(OUT / "acc_by_method_task.csv", index=False)
    task_accuracy.pivot(index="method", columns="task", values="acc_3dp").reindex(METHODS)[TASKS].to_csv(OUT / "acc_by_method_task_pivot.csv")
    curves.to_csv(OUT / "all_threshold_curves.csv", index=False)
    metrics_thr005 = make_metrics_at_threshold(metrics, curves, 0.05)
    summary_metrics = metrics_thr005.drop(columns=["mrfh_at_thr0", "hit_rate_at_thr0", "hit_cases_at_thr0"], errors="ignore")
    summary_metrics.to_csv(OUT / "summary_metrics.csv", index=False)
    metrics_thr005.to_csv(OUT / "summary_metrics_thr005.csv", index=False)
    cost_efficiency = compute_cost_efficiency(metrics_thr005)
    cost_efficiency.to_csv(OUT / "cost_efficiency_metrics.csv", index=False)
    threshold_table = choose_thresholds(curves, 0.05)
    threshold_table.to_csv(OUT / "threshold_recommendation.csv", index=False)
    method_drops = compute_method_drops(curves, 0.05)
    method_drops.to_csv(OUT / "method_drops_by_threshold.csv", index=False)
    summarize_method_drops(method_drops).to_csv(OUT / "method_drop_summary.csv", index=False)
    evidence_raw = load_evidence_scores()
    evidence_raw.to_csv(OUT / "evidence_final_scores_all_matching.csv", index=False)
    evidence, evidence_roi_filter_summary = filter_matching_to_roi_cases(evidence_raw)
    evidence.to_csv(OUT / "evidence_final_scores.csv", index=False)
    evidence_roi_filter_summary.to_csv(OUT / "evidence_roi_filter_summary.csv", index=False)
    evidence_summary, evidence_by_correct = summarize_evidence_scores(evidence)
    evidence_summary.to_csv(OUT / "evidence_score_summary.csv", index=False)
    evidence_by_correct.to_csv(OUT / "evidence_score_by_correct_summary.csv", index=False)
    evidence_score_dist, evidence_score_correct_dist = compute_evidence_score_distribution(evidence)
    evidence_score_dist.to_csv(OUT / "evidence_score_distribution.csv", index=False)
    evidence_score_correct_dist.to_csv(OUT / "evidence_score_correct_distribution.csv", index=False)
    roi_case_scores = load_roi_case_scores_with_matching_correct(evidence)
    roi_case_scores.to_csv(OUT / "roi_case_scores_with_matching_correct.csv", index=False)
    roi_score_summary = summarize_roi_scores_by_matching_correct(roi_case_scores)
    roi_score_summary.to_csv(OUT / "roi_score_by_matching_correct_summary.csv", index=False)
    roi_aligned_accuracy = compute_roi_aligned_accuracy(roi_case_scores)
    roi_aligned_accuracy.to_csv(OUT / "accuracy.csv", index=False)
    hit_rate_group_cases, hit_rate_group_summary = compute_hit_rate_zero_nonzero_case_stats(roi_case_scores, evidence)
    hit_rate_group_cases.to_csv(OUT / "hit_rate_zero_nonzero_case_metrics.csv", index=False)
    hit_rate_group_summary.to_csv(OUT / "hit_rate_zero_nonzero_case_share_acc.csv", index=False)
    correct_hit_rate_group_dist = compute_correct_only_hit_rate_distribution(hit_rate_group_cases)
    correct_hit_rate_group_dist.to_csv(OUT / "hit_rate_zero_nonzero_correct_case_distribution.csv", index=False)
    all_vs_hit_nonzero_acc = compute_all_vs_hit_nonzero_accuracy(hit_rate_group_cases)
    all_vs_hit_nonzero_acc.to_csv(OUT / "all_vs_hit_nonzero_accuracy.csv", index=False)
    roi_nav_thr005 = load_roi_navigation_thr005_with_matching_correct(evidence, 0.05)
    roi_nav_thr005.to_csv(OUT / "roi_navigation_case_metrics_thr005.csv", index=False)
    roi_nav_thr005_summary = summarize_roi_navigation_thr005(roi_nav_thr005)
    roi_nav_thr005_summary.to_csv(OUT / "roi_navigation_metrics_thr005_summary.csv", index=False)
    uncertainty_raw = load_uncertainty_scores()
    if uncertainty_raw.empty:
        uncertainty = uncertainty_raw
        save_empty_uncertainty_note()
    else:
        uncertainty_raw.to_csv(OUT / "Uncertainty_evidence_final_scores_all_matching.csv", index=False)
        uncertainty, uncertainty_roi_filter_summary = filter_matching_to_roi_cases(uncertainty_raw)
        uncertainty.to_csv(OUT / "Uncertainty_evidence_final_scores.csv", index=False)
        uncertainty_roi_filter_summary.to_csv(OUT / "Uncertainty_evidence_roi_filter_summary.csv", index=False)
        uncertainty_summary, uncertainty_by_correct = summarize_evidence_scores(uncertainty)
        uncertainty_summary.to_csv(OUT / "Uncertainty_evidence_score_summary.csv", index=False)
        uncertainty_by_correct.to_csv(OUT / "Uncertainty_evidence_score_by_correct_summary.csv", index=False)
        uncertainty_score_dist, uncertainty_score_correct_dist = compute_evidence_score_distribution(uncertainty)
        uncertainty_score_dist.to_csv(OUT / "Uncertainty_evidence_score_distribution.csv", index=False)
        uncertainty_score_correct_dist.to_csv(OUT / "Uncertainty_evidence_score_correct_distribution.csv", index=False)
        uncertainty_acc_by_score = compute_score_accuracy_comparison(uncertainty)
        uncertainty_acc_by_score.to_csv(OUT / "Uncertainty_score_accuracy_comparison.csv", index=False)
        uncertainty_score1_acc = uncertainty_acc_by_score[pd.to_numeric(uncertainty_acc_by_score["score_label"], errors="coerce").eq(1.0)].copy()
        uncertainty_score1_acc.to_csv(OUT / "Uncertainty_score1_accuracy_comparison.csv", index=False)

    plot_task_comparison(metrics_thr005)
    plot_task_comparison_at_threshold(metrics_thr005, 0.05)
    plot_conditional_metrics(metrics_thr005)
    plot_cost_efficiency(cost_efficiency)
    plot_threshold_curves(curves)
    plot_threshold_recommendation(threshold_table, curves)
    plot_method_drops(method_drops)
    plot_evidence_overall(evidence_summary)
    plot_evidence_score_by_correct(evidence_by_correct)
    plot_evidence_correct_counts(evidence)
    plot_evidence_score_distribution(evidence_score_dist)
    plot_evidence_score_correct_distribution(evidence_score_correct_dist)
    plot_roi_score_by_matching_correct(roi_score_summary)
    plot_roi_aligned_accuracy(roi_aligned_accuracy)
    plot_roi_hit_rate(metrics_thr005)
    plot_hitrate_precision_accuracy_scatter(metrics_thr005, roi_aligned_accuracy)
    plot_navigation_pairwise_strategy_scatter(metrics_thr005, roi_aligned_accuracy)
    plot_roi_aligned_accuracy_original(roi_aligned_accuracy)
    plot_hit_rate_zero_nonzero_case_stats(hit_rate_group_summary)
    plot_correct_only_hit_rate_distribution(correct_hit_rate_group_dist)
    plot_all_vs_hit_nonzero_accuracy(all_vs_hit_nonzero_acc)
    plot_roi_navigation_thr005(roi_nav_thr005_summary)
    if not uncertainty.empty:
        plot_uncertainty_overall(uncertainty_summary)
        plot_uncertainty_score_by_correct(uncertainty_by_correct)
        plot_uncertainty_correct_counts(uncertainty)
        plot_uncertainty_score_distribution(uncertainty_score_dist)
        plot_uncertainty_score_correct_distribution(uncertainty_score_correct_dist)
        plot_score1_accuracy_comparison(uncertainty_score1_acc, "Uncertainty")
    plot_score_by_category(evidence)
    plot_score_correct_ratio(evidence)
    plot_acc_score1_vs_all(evidence)
    write_report(summary_metrics, curves, threshold_table)


if __name__ == "__main__":
    main()
