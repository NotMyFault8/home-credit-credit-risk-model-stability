"""Generate an evidence-linked model validation report from run artifacts."""

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(value, digits=6):
    if value is None or pd.isna(value):
        return "unavailable"
    return f"{float(value):.{digits}f}"


def _load_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _line_svg(series, destination, title, x_domain, y_domain, diagonal=False):
    """Write a compact dependency-free SVG from named x/y series."""
    width, height, left, top = 760, 440, 70, 45
    plot_width, plot_height = 665, 340

    def point(x, y):
        px = left + (x - x_domain[0]) / (x_domain[1] - x_domain[0]) * plot_width
        py = top + plot_height - (y - y_domain[0]) / (y_domain[1] - y_domain[0]) * plot_height
        return px, py

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2"]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="380" y="25" text-anchor="middle" font-family="sans-serif" '
        f'font-size="18">{escape(title)}</text>',
        f'<path d="M {left} {top} V {top + plot_height} H {left + plot_width}" '
        'fill="none" stroke="#111827"/>',
    ]
    if diagonal:
        x1, y1 = point(0, 0)
        x2, y2 = point(1, 1)
        svg.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            'stroke="#9ca3af" stroke-dasharray="5 5"/>'
        )
    for index, (name, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        points = [point(float(x), float(y)) for x, y in values if pd.notna(x) and pd.notna(y)]
        if not points:
            continue
        coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        svg.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        svg.append(
            f'<text x="{left + 10}" y="{top + 15 * (index + 1)}" '
            f'font-family="sans-serif" font-size="11" fill="{color}">{escape(str(name))}</text>'
        )
    for value in np.linspace(x_domain[0], x_domain[1], 6):
        x, _ = point(value, y_domain[0])
        svg.append(
            f'<text x="{x:.1f}" y="405" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11">{value:.2f}</text>'
        )
    for value in np.linspace(y_domain[0], y_domain[1], 6):
        _, y = point(x_domain[0], value)
        svg.append(
            f'<text x="60" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{value:.2f}</text>'
        )
    svg.append("</svg>")
    destination.write_text("\n".join(svg) + "\n")


def generate_validation_charts(run_root, destination):
    """Generate reliability and weekly AUC charts from aggregate evidence."""
    root, destination = Path(run_root), Path(destination)
    calibration = _load_csv(root / "bi/calibration_weekly.csv")
    if {"mean_score", "default_rate"} <= set(calibration) and not calibration.empty:
        keys = [column for column in ("mode", "score_type") if column in calibration]
        grouped = calibration.groupby(keys) if keys else [("all", calibration)]
        series = {
            str(key): list(zip(group.mean_score, group.default_rate)) for key, group in grouped
        }
        _line_svg(
            series,
            destination / "validation-calibration.svg",
            "Reliability Diagram",
            (0, 1),
            (0, 1),
            diagonal=True,
        )
    performance = _load_csv(root / "bi/model_performance_weekly.csv")
    if {"WEEK_NUM", "auc"} <= set(performance) and performance.auc.notna().any():
        grouped = performance.groupby("mode") if "mode" in performance else [("all", performance)]
        series = {str(key): list(zip(group.WEEK_NUM, group.auc)) for key, group in grouped}
        weeks = performance.WEEK_NUM.dropna()
        domain = (float(weeks.min()), float(weeks.max()))
        domain = (domain[0] - 1, domain[1] + 1) if domain[0] == domain[1] else domain
        _line_svg(
            series,
            destination / "validation-weekly-auc.svg",
            "Weekly AUC Trend",
            domain,
            (0, 1),
        )


def generate_validation_report(run_root, destination=None):
    """Build Markdown only from persisted JSON/CSV evidence, never temporary numbers."""
    root = Path(run_root)
    destination = Path(destination or root / "MODEL_VALIDATION_REPORT.md")
    evaluation_path = root / "evaluation.json"
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Missing evaluation evidence: {evaluation_path}")
    evaluation = json.loads(evaluation_path.read_text())
    manifest_path = root / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    summary_path = root / "dataset-summary.json"
    if not manifest and summary_path.exists():
        manifest = json.loads(summary_path.read_text())
    synthetic = bool(evaluation.get("synthetic", manifest.get("synthetic", False)))
    extension_available = bool(evaluation.get("calibration_protocols"))
    evidence_kind = "SYNTHETIC ENGINEERING DEMO" if synthetic else "RETROSPECTIVE REAL-DATA RUN"
    calibration = _load_csv(root / "bi/calibration_comparison.csv")
    segments = _load_csv(root / "bi/segment_performance.csv")
    alerts = _load_csv(root / "bi/stability_alerts.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    generate_validation_charts(root, destination.parent)
    lines = [
        "# Model Validation Report",
        "",
        f"> Evidence classification: **{evidence_kind}**. This report is generated from run artifacts.",
        "",
        "## 1. Executive Summary",
        "",
        "The validation framework covers discrimination, calibration, temporal behavior, segment "
        "robustness, data/score drift, and operational score parity; unavailable evidence is identified "
        "below. It is a research and engineering review, "
        "not approval for lending, a production-readiness statement, or a regulatory certification.",
        "",
        "## 2. Data Provenance & Quality",
        "",
        f"- Data version: `{manifest.get('data_version', 'unavailable')}`",
        f"- Feature version: `{manifest.get('feature_version', 'unavailable')}`",
        f"- Source type: {'fictional synthetic fixtures' if synthetic else 'historical Kaggle competition data'}",
        "- Point-in-time feature availability and real label maturity: **not assessed / unavailable**.",
        "",
        "## 3. Model & Feature Methodology",
        "",
        "Logistic regression and LightGBM candidates use train-only preprocessing. Full and internal "
        "fallback models are selected and versioned separately. The legacy `score` remains the raw "
        "model probability; `calibrated_score` is an additional, reversible output.",
        "",
        "## 4. Validation Design",
        "",
        "Three expanding temporal folds are used for model selection and out-of-time calibration "
        "predictions. Calibration method selection stays within development data. The final four weeks "
        "are a retrospective, previously inspected evaluation period, not a newly untouched independent test.",
        "",
        "## 5. Discrimination & Calibration",
        "",
        "| Mode | AUC | Gini | Brier | Log loss | ECE | O/E | Stability |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in evaluation.get("holdout", {}).items():
        gini = metrics.get("gini")
        if gini is None and metrics.get("auc") is not None:
            gini = 2 * metrics["auc"] - 1
        lines.append(
            f"| {mode} | {_fmt(metrics.get('auc'))} | {_fmt(gini)} | "
            f"{_fmt(metrics.get('brier'))} | {_fmt(metrics.get('log_loss'))} | "
            f"{_fmt(metrics.get('ece'))} | {_fmt(metrics.get('oe_ratio'))} | "
            f"{_fmt(metrics.get('stability'))} |"
        )
    if calibration.empty:
        lines.extend(["", "Calibration comparison evidence: unavailable."])
    else:
        lines.extend(
            [
                "",
                "Calibration comparison is recorded in `bi/calibration_comparison.csv`; improvements "
                "must be judged by the numeric Brier/log-loss/ECE comparison, not chart appearance.",
                "",
                "| Mode | Score | AUC | Brier | Log loss | ECE | O/E | Intercept | Slope |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in calibration.to_dict(orient="records"):
            lines.append(
                f"| {row.get('mode', 'unavailable')} | {row.get('score_type', 'unavailable')} | "
                f"{_fmt(row.get('auc'))} | {_fmt(row.get('brier'))} | "
                f"{_fmt(row.get('log_loss'))} | {_fmt(row.get('ece'))} | "
                f"{_fmt(row.get('oe_ratio'))} | {_fmt(row.get('calibration_intercept'))} | "
                f"{_fmt(row.get('calibration_slope'))} |"
            )
    if (destination.parent / "validation-calibration.svg").exists():
        lines.extend(["", "![Reliability diagram](validation-calibration.svg)"])
    lines.extend(
        [
            "",
            "## 6. Segment Performance",
            "",
            (
                f"`bi/segment_performance.csv` contains {len(segments):,} overall/segment/time rows "
                "with explicit non-estimable states. This is segment robustness analysis, not fairness certification."
                if not segments.empty
                else "Segment evidence is unavailable; fairness was not assessed."
            ),
            "",
            "## 7. Stability & Drift",
            "",
            (
                "Weekly performance, feature PSI (numeric and categorical), missingness, source coverage, "
                "raw/calibrated score distributions, and score PSI are written under `bi/`."
                if extension_available
                else "The legacy artifact includes weekly performance, numeric feature PSI, missingness and "
                "coverage; categorical PSI and raw/calibrated score PSI are unavailable until a new run."
            ),
            "PSI > 0.2 is an explanatory example threshold, not an automatic approval or retraining rule.",
            f"The run produced {len(alerts):,} traceable alert rows.",
            "",
            (
                "![Weekly AUC trend](validation-weekly-auc.svg)"
                if (destination.parent / "validation-weekly-auc.svg").exists()
                else "Weekly AUC chart unavailable."
            ),
            "",
            "## 8. Operational Verification",
            "",
            (
                "Batch and API use the same bound model/calibrator transformation. Model metadata records "
                "data, code, feature, seed, mode, base-model and calibrator versions. Activation/rollback "
                "continues to operate on the paired Full/Internal bundles."
                if extension_available
                else "The committed legacy run verifies raw-score batch/API parity and paired Full/Internal "
                "activation/rollback. Calibrator parity and binding are unavailable for this pre-extension "
                "artifact; they are exercised by the current synthetic test suite."
            ),
            "",
            "## 9. Limitations & Risks",
            "",
            "- Historical feature arrival times cannot be verified from `WEEK_NUM` alone.",
            "- Label delay is simulated and must not be interpreted as observed production maturity.",
            "- The curated feature set is limited; protected attributes and fairness mechanisms are absent.",
            "- The final weeks have already been inspected and reported; a new independent test requires later data.",
            "- Scores are not validated 12-month PDs and are not suitable for real loan approval.",
            "",
            "## 10. Validation Conclusion & Action Items",
            "",
            "Engineering checks are reproducible from the cited artifacts. Statistical results remain "
            "conditional on this historical/synthetic scope. Before real-world use, obtain a genuinely new "
            "time period, point-in-time lineage, real label-maturity evidence, independent governance review, "
            "and an appropriate fairness assessment.",
            "",
            "## Evidence Index",
            "",
            "- `evaluation.json`: versions, temporal experiments, retrospective holdout metrics",
            "- `data/manifest.json`: data hashes, schemas and provenance (when retained with the run)",
            "- `bi/calibration_comparison.csv`: raw versus calibrated metrics",
            "- `bi/calibration_weekly.csv`: fixed-bin reliability evidence and event counts",
            "- `bi/segment_performance.csv`: frozen segment definitions and metrics",
            "- `bi/model_performance_weekly.csv`: weekly performance and calibration",
            "- `bi/data_quality_weekly.csv`: feature drift, missingness and coverage",
            "- `bi/score_stability_weekly.csv`: score distribution drift",
            "- `bi/stability_alerts.csv`: threshold, week, model and reference lineage",
        ]
    )
    destination.write_text("\n".join(lines) + "\n")
    return destination
