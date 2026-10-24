"""Create a portable Tableau workbook from pipeline CSV summaries."""

import argparse
import copy
import csv
import json
import os
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path


def node(parent, tag, text=None, **attrs):
    child = ET.SubElement(parent, tag, {k.replace("_", "-"): str(v) for k, v in attrs.items()})
    child.text = text
    return child


def build(source, destination, label="SYNTHETIC DEMO"):
    source, destination = Path(source), Path(destination)
    if (source.parent / "evaluation.json").exists():
        evaluation = json.loads((source.parent / "evaluation.json").read_text())
        label += " | Stability " + " / ".join(
            f"{mode}: {metrics['stability']:.3f}"
            for mode, metrics in evaluation["holdout"].items()
            if metrics["stability"] is not None
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element(
        "workbook",
        {
            "original-version": "18.1",
            "source-build": "2024.1.0",
            "source-platform": "mac",
            "version": "18.1",
        },
    )
    node(root, "preferences")
    datasources = node(root, "datasources")
    definitions = {}
    names = [
        "risk_summary_weekly",
        "model_performance_weekly",
        "data_quality_weekly",
        "pipeline_runs",
        "calibration_weekly",
    ]
    for name in names:
        path = source / f"{name}.csv"
        with path.open() as stream:
            reader = csv.DictReader(stream)
            rows, fields = list(reader), reader.fieldnames
        ds = node(
            datasources,
            "datasource",
            caption=name.replace("_", " ").title(),
            inline="true",
            name=name,
            version="18.1",
        )
        connection = node(
            ds,
            "connection",
            **{
                "class": "textscan",
                "directory": os.path.relpath(source.resolve(), destination.parent.resolve()),
                "filename": path.name,
                "password": "",
                "server": "",
            },
        )
        node(connection, "relation", **{"name": name, "table": f"[{path.name}]", "type": "table"})
        node(
            connection,
            "textscan-format",
            separator=",",
            header="yes",
            locale="en_US",
            character_set="UTF-8",
        )
        cols = {}
        for field in fields:
            dimension = field in [
                "WEEK_NUM",
                "model_version",
                "mode",
                "as_of_week",
                "label_mature",
                "sample_scope",
                "status",
                "feature",
                "reference_scope",
                "alert",
                "run_id",
                "started_at",
                "finished_at",
                "hardware",
                "data_version",
                "error",
            ]
            values = [r[field] for r in rows if r[field]]
            try:
                for value in values:
                    float(value)
                datatype = "integer" if field in ["WEEK_NUM", "as_of_week"] else "real"
            except ValueError:
                datatype = "string"
            column = node(
                ds,
                "column",
                name=f"[{field}]",
                datatype=datatype,
                role="dimension" if dimension else "measure",
                type="ordinal"
                if datatype == "integer" and dimension
                else "nominal"
                if dimension
                else "quantitative",
            )
            if not dimension and any(k in field for k in ["rate", "score", "coverage"]):
                column.set("default-format", "p1%")
            cols[field] = column
        definitions[name] = cols
    worksheets = node(root, "worksheets")
    charts = [
        ("Applications", names[0], "WEEK_NUM", "applications", "mode", "Bar"),
        ("Predicted risk", names[0], "WEEK_NUM", "mean_score", "mode", "Line"),
        ("Mature default rate", names[0], "WEEK_NUM", "default_rate", "mode", "Line"),
        ("High risk band", names[0], "WEEK_NUM", "high_risk", "mode", "Bar"),
        ("Weekly Gini", names[1], "WEEK_NUM", "gini", "mode", "Line"),
        ("Weekly AUC", names[1], "WEEK_NUM", "auc", "mode", "Line"),
        ("Brier score", names[1], "WEEK_NUM", "brier", "mode", "Line"),
        ("Calibration", names[4], "mean_score", "default_rate", "mode", "Circle"),
        ("Missing features", names[2], "WEEK_NUM", "missing_rate", "feature", "Line"),
        ("Feature drift PSI", names[2], "WEEK_NUM", "psi", "feature", "Line"),
        ("Source coverage", names[2], "WEEK_NUM", "coverage", "feature", "Line"),
        (
            "Missingness reference",
            names[2],
            "WEEK_NUM",
            "reference_missing_rate",
            "feature",
            "Line",
        ),
        ("Run duration", names[3], "run_id", "duration_seconds", "status", "Bar"),
        ("Rows processed", names[3], "run_id", "rows", "status", "Bar"),
        ("Peak memory MB", names[3], "run_id", "peak_rss_mb", "status", "Bar"),
        ("Scoring volume", names[0], "WEEK_NUM", "applications", "mode", "Bar"),
    ]
    for title, ds, x, y, color, mark in charts:
        worksheet = node(worksheets, "worksheet", name=title)
        table = node(worksheet, "table")
        view = node(table, "view")
        node(node(view, "datasources"), "datasource", caption=ds.replace("_", " ").title(), name=ds)
        deps = node(view, "datasource-dependencies", datasource=ds)
        for field in [x, y, color]:
            deps.append(copy.deepcopy(definitions[ds][field]))
            quantitative = field == y
            inst = f"[avg:{field}:qk]" if quantitative else f"[none:{field}:nk]"
            node(
                deps,
                "column-instance",
                column=f"[{field}]",
                derivation="Avg" if quantitative else "None",
                name=inst,
                pivot="key",
                type="quantitative" if quantitative else "nominal",
            )
        node(view, "aggregation", value="true")
        style = node(table, "style")
        node(
            node(style, "style-rule", element="worksheet"),
            "format",
            attr="font-family",
            value="Tableau Book",
        )
        pane = node(
            node(table, "panes"), "pane", selection_relaxation_option="selection-relaxation-allow"
        )
        node(node(pane, "view"), "breakdown", value="auto")
        node(pane, "mark", **{"class": mark})
        node(node(pane, "encodings"), "color", column=f"[{ds}].[none:{color}:nk]")
        node(table, "rows", f"[{ds}].[avg:{y}:qk]")
        node(table, "cols", f"[{ds}].[none:{x}:nk]")
        node(worksheet, "simple-id", uuid="{" + str(uuid.uuid5(uuid.NAMESPACE_URL, title)) + "}")
    dashboards = node(root, "dashboards")
    pages = [
        "01 Risk overview",
        "02 Model stability",
        "03 Data quality and drift",
        "04 Pipeline operations",
    ]
    for i, title in enumerate(pages):
        dashboard = node(dashboards, "dashboard", name=title)
        node(
            node(node(node(dashboard, "layout-options"), "title"), "formatted-text"),
            "run",
            "Credit Risk & Model Stability | " + title,
            fontsize="20",
            fontcolor="#143d52",
        )
        node(dashboard, "size", maxheight="850", maxwidth="1250", minheight="850", minwidth="1250")
        zones = node(dashboard, "zones")
        node(zones, "zone", h="9000", id="1", type_v2="title", w="96000", x="2000", y="0")
        text = node(zones, "zone", h="7000", id="2", type_v2="text", w="96000", x="2000", y="9000")
        node(
            node(text, "formatted-text"),
            "run",
            label
            + " | Final holdout only | Labels mature after 2 simulated weeks. Blank metrics mean unavailable labels, not zero.",
            fontsize="11",
            fontcolor="#557080",
        )
        for j, (chart, *_) in enumerate(charts[i * 4 : i * 4 + 4]):
            node(
                zones,
                "zone",
                h="39000",
                id=str(j + 3),
                name=chart,
                w="47000",
                x=str(2000 + (j % 2) * 49000),
                y=str(17000 + 41000 * (j // 2)),
            )
        node(dashboard, "simple-id", uuid="{" + str(uuid.uuid5(uuid.NAMESPACE_URL, title)) + "}")
    windows = node(root, "windows", saved_dpi_scale_factor="1.0", source_height="1000")
    for title in pages:
        window = node(windows, "window", **{"class": "dashboard", "name": title})
        node(window, "viewpoints")
        node(window, "active", id="-1")
    node(
        node(
            root, "explain-data", enabled_for_viewer="false", extreme_values_enabled_for_all="false"
        ),
        "explanation-types",
    )
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="reports/demo/bi")
    parser.add_argument("--output", default="bi/credit-risk-stability.twb")
    parser.add_argument("--label", default="SYNTHETIC DEMO")
    args = parser.parse_args()
    build(args.source, args.output, args.label)
