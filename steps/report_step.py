"""
ReportStep - Pipeline step for generating an HTML design report.

Produces a modern, minimal HTML report summarising all pipeline stages,
key metric distributions (histograms), the top-ranked design structure
(Mol* 3-D viewer), and a paginated table of every ranked design.
"""

import base64
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Metric columns per gentype  (used for AF3score / ReFold histograms)
# ---------------------------------------------------------------------------

_HIST_METRICS: Dict[str, List[str]] = {
    "binder":   ["chain_A_ptm", "iptm"],
    "nanobody": ["chain_A_ptm", "iptm"],
    "antibody": ["chain_A_ptm", "chain_B_ptm", "iptm_A_C", "iptm_B_C"],
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Return (fieldnames, rows) or ([], []) when the file is absent."""
    if not os.path.isfile(path):
        return [], []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _count_pdb_files(directory: str) -> int:
    """Count *.pdb files (including symlinks) directly in *directory*."""
    if not os.path.isdir(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if f.lower().endswith(".pdb"))


def _count_csv_rows(csv_path: str) -> int:
    """Return number of data rows in a CSV file (0 if missing)."""
    _, rows = _read_csv(csv_path)
    return len(rows)


def _pdb_to_base64(pdb_path: str) -> str:
    """Read a PDB file and return its content as a base-64 ASCII string."""
    if not os.path.isfile(pdb_path):
        return ""
    with open(pdb_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _build_histogram_data(
    rows: List[Dict[str, str]],
    metrics: List[str],
) -> Dict[str, List[float]]:
    """Extract numeric value lists for each requested metric column."""
    data: Dict[str, List[float]] = {m: [] for m in metrics}
    for row in rows:
        for m in metrics:
            val = row.get(m, "")
            try:
                data[m].append(float(val))
            except (TypeError, ValueError):
                pass
    return data


# ---------------------------------------------------------------------------
# ReportStep
# ---------------------------------------------------------------------------


class ReportStep:
    """Pipeline step that generates an HTML design report.

    Parameters
    ----------
    gentype : str
        Generation type: ``"binder"``, ``"nanobody"``, or ``"antibody"``.
    name : str
        Human-readable design name shown in the report header.
    output_dir : str
        Directory where the report HTML will be written
        (typically ``design_output``).
    stage1_dir : str
        Path to the stage-1 pipeline directory.
    stage2_dir : str
        Path to the stage-2 pipeline directory.
    af3score_stage1_csv : str
        Path to ``af3score_metrics.csv`` from AF3scoreStep_stage1.
    af3score_stage2_csv : str
        Path to ``af3score_metrics.csv`` from AF3scoreStep_stage2.
    af3_refold_csv : str
        Path to ``af3_iptm@1.csv`` from ReFoldStep.
    ranked_csv : str
        Path to the ranked designs CSV produced by RankStep.
    output_csv_name : str
        Filename of the ranked designs CSV (default: ``"ranked_designs.csv"``).
    report_filename : str
        Output HTML filename (default: ``"design_report.html"``).
    """

    def __init__(
        self,
        gentype: str,
        name: str,
        output_dir: str,
        stage1_dir: str,
        stage2_dir: str,
        af3score_stage1_csv: str,
        af3score_stage2_csv: str,
        af3_refold_csv: str,
        ranked_csv: str,
        output_csv_name: str = "ranked_designs.csv",
        report_filename: str = "design_report.html",
    ):
        self.gentype = gentype
        self.name = name
        self.output_dir = output_dir
        self.stage1_dir = stage1_dir
        self.stage2_dir = stage2_dir
        self.af3score_stage1_csv = af3score_stage1_csv
        self.af3score_stage2_csv = af3score_stage2_csv
        self.af3_refold_csv = af3_refold_csv
        self.ranked_csv = ranked_csv
        self.output_csv_name = output_csv_name
        self.report_filename = report_filename

        self._hist_metrics = _HIST_METRICS.get(gentype, ["iptm"])

    # ------------------------------------------------------------------
    # Stage counts
    # ------------------------------------------------------------------

    def _gather_stage_counts(self) -> Dict[str, int]:
        """Return design counts at each filtered stage."""
        s1 = self.stage1_dir
        s2 = self.stage2_dir

        ppiflow_out = os.path.join(s1, "mpnn_pdbs")
        af3s1_csv   = os.path.join(s1, "af3score_output", "af3score_metrics.csv")
        filtered_s1 = os.path.join(s1, "filtered_iptm07")

        partial_out = os.path.join(s2, "mpnn_pdbs")
        af3s2_csv   = os.path.join(s2, "af3score_output", "af3score_metrics.csv")
        filtered_s2 = os.path.join(s2, "filtered_iptm08")

        # Count partial_output: may be flat or one sub-dir per target
        partial_count = 0
        if os.path.isdir(partial_out):
            flat = _count_pdb_files(partial_out)
            if flat:
                partial_count = flat
            else:
                partial_count = sum(
                    _count_pdb_files(os.path.join(partial_out, d))
                    for d in os.listdir(partial_out)
                    if os.path.isdir(os.path.join(partial_out, d))
                )

        return {
            "PPIFlowStep":         _count_pdb_files(ppiflow_out),
            "AF3scoreStep_stage1": _count_csv_rows(af3s1_csv),
            "FilterStep_stage1":   _count_pdb_files(filtered_s1),
            "PartialStep":         partial_count,
            "AF3scoreStep_stage2": _count_csv_rows(af3s2_csv),
            "FilterStep_stage2":   _count_pdb_files(filtered_s2),
            "RankStep":            _count_csv_rows(self.ranked_csv),
        }

    # ------------------------------------------------------------------
    # Top-ranked PDB
    # ------------------------------------------------------------------

    def _get_top_pdb_path(self) -> Optional[str]:
        """Return path to the top-ranked design PDB (highest rank_score)."""
        _, rows = _read_csv(self.ranked_csv)
        if not rows:
            return None

        def _rank_score(row: Dict[str, str]) -> float:
            try:
                return float(row.get("rank_score", 0))
            except (TypeError, ValueError):
                return 0.0

        top_row = max(rows, key=_rank_score)
        # ranked_csv uses pdb_name; fall back to target_name
        pdb_name = (
            top_row.get("pdb_name") or top_row.get("target_name") or ""
        ).strip()
        if not pdb_name:
            return None

        pdb_path = os.path.join(self.output_dir, "pdbs", f"{pdb_name}.pdb")
        if os.path.isfile(pdb_path):
            return pdb_path
        return None

    # ------------------------------------------------------------------
    # Histogram datasets for AF3score / ReFold sections
    # ------------------------------------------------------------------

    def _load_af3_histogram_datasets(self) -> List[Dict]:
        """Build histogram dataset descriptors for AF3score + ReFold sources."""
        datasets = []
        sources = [
            ("AF3score Stage 1", self.af3score_stage1_csv, self._hist_metrics),
            ("AF3score Stage 2", self.af3score_stage2_csv, self._hist_metrics),
            ("ReFold",           self.af3_refold_csv,      self._hist_metrics),
        ]
        for label, csv_path, metrics in sources:
            _, rows = _read_csv(csv_path)
            if not rows:
                continue
            # Use only metrics that actually exist in this CSV
            available = [m for m in metrics if m in rows[0]]
            if not available:
                continue
            datasets.append({
                "label":   label,
                "metrics": available,
                "data":    _build_histogram_data(rows, available),
            })
        return datasets

    # ------------------------------------------------------------------
    # Histogram datasets for ranked_designs.csv
    # ------------------------------------------------------------------

    def _load_ranked_histogram_datasets(self) -> List[Dict]:
        """Build histogram datasets from ranked_designs.csv.

        Shows interface_score and all refold metric columns that are present.
        """
        _, ranked_rows = _read_csv(self.ranked_csv)
        if not ranked_rows:
            return []

        # Columns to plot: interface_score  +  refold metrics that exist
        refold_metrics = [
            m for m in self._hist_metrics
            if m in ranked_rows[0]
        ]
        plot_cols = []
        if "interface_score" in ranked_rows[0]:
            plot_cols.append("interface_score")
        plot_cols.extend(refold_metrics)

        if not plot_cols:
            return []

        return [{
            "label":   "Ranked Designs",
            "metrics": plot_cols,
            "data":    _build_histogram_data(ranked_rows, plot_cols),
        }]

    # ------------------------------------------------------------------
    # Table data
    # ------------------------------------------------------------------

    def _get_refold_metric_cols(self) -> List[str]:
        """Return the gentype-specific refold metric columns that exist in af3_refold_csv.

        Columns are drawn from ``_HIST_METRICS`` (already keyed by gentype) so
        only the intended metrics are shown:
          binder/nanobody : chain_A_ptm, iptm
          antibody        : chain_A_ptm, chain_B_ptm, iptm_A_C, iptm_B_C
        """
        _, rows = _read_csv(self.af3_refold_csv)
        available = set(rows[0].keys()) if rows else set()
        return [m for m in self._hist_metrics if m in available]

    def _build_table_data(self) -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Build table columns and rows for the ranked designs table.

        Columns shown (in order):
          pdb_name, rank_score, interface_score, dockq_mean,
          binder_seq / heavy_seq / light_seq  (whichever present),
          gentype-specific refold metric columns (from _HIST_METRICS).
        """
        _, ranked_rows    = _read_csv(self.ranked_csv)
        _, af3_refold_rows = _read_csv(self.af3_refold_csv)

        if not ranked_rows:
            return [], []

        # Index refold rows by pdb_name
        af3_index: Dict[str, Dict[str, str]] = {}
        for row in af3_refold_rows:
            key = (row.get("pdb_name") or "").strip()
            if key:
                af3_index[key] = row

        refold_extra_cols = self._get_refold_metric_cols()

        # Core columns always shown
        core_cols = ["pdb_name", "rank_score", "interface_score", "dockq_mean"]

        # Sequence columns (present only for relevant gentype)
        seq_cols: List[str] = []
        first = ranked_rows[0]
        for col in ("binder_seq", "heavy_seq", "light_seq"):
            if col in first:
                seq_cols.append(col)

        all_cols = core_cols + seq_cols + refold_extra_cols

        merged: List[Dict[str, str]] = []
        for row in ranked_rows:
            # Resolve pdb_name: ranked_csv may use pdb_name or target_name
            pdb_name_val = (
                row.get("pdb_name") or row.get("target_name") or ""
            ).strip()
            af3_row = af3_index.get(pdb_name_val, {})

            merged_row: Dict[str, str] = {}
            for col in all_cols:
                if col == "pdb_name":
                    merged_row[col] = pdb_name_val
                else:
                    merged_row[col] = row.get(col) or af3_row.get(col) or ""
            merged.append(merged_row)

        return all_cols, merged

    # ------------------------------------------------------------------
    # Funnel HTML helper
    # ------------------------------------------------------------------

    @staticmethod
    def _funnel_rows_html(steps: List[Tuple[str, int]]) -> str:
        parts = []
        for step_name, count in steps:
            parts.append(
                f'<div class="funnel-card">'
                f'<div class="step-name">{step_name}</div>'
                f'<div class="step-count">{count}</div>'
                f'<div class="step-label">designs</div>'
                f'</div>'
            )
        return "\n        ".join(parts)

    # ------------------------------------------------------------------
    # HTML generation
    # ------------------------------------------------------------------

    def _render_html(self) -> str:
        stage_counts      = self._gather_stage_counts()
        af3_hist_datasets = self._load_af3_histogram_datasets()
        ranked_hist_datasets = self._load_ranked_histogram_datasets()
        top_pdb_path      = self._get_top_pdb_path()
        table_cols, table_rows = self._build_table_data()

        top_pdb_b64  = _pdb_to_base64(top_pdb_path) if top_pdb_path else ""
        top_pdb_name = os.path.basename(top_pdb_path) if top_pdb_path else "N/A"

        stage1_rows_html = self._funnel_rows_html([
            ("PPIFlowStep",          stage_counts.get("PPIFlowStep", 0)),
            ("AF3scoreStep stage 1", stage_counts.get("AF3scoreStep_stage1", 0)),
            ("FilterStep stage 1",   stage_counts.get("FilterStep_stage1", 0)),
        ])
        stage2_rows_html = self._funnel_rows_html([
            ("PartialStep",          stage_counts.get("PartialStep", 0)),
            ("AF3scoreStep stage 2", stage_counts.get("AF3scoreStep_stage2", 0)),
            ("FilterStep stage 2",   stage_counts.get("FilterStep_stage2", 0)),
            ("RankStep",             stage_counts.get("RankStep", 0)),
        ])

        af3_hist_json    = json.dumps(af3_hist_datasets,    ensure_ascii=False, default=str)
        ranked_hist_json = json.dumps(ranked_hist_datasets, ensure_ascii=False, default=str)
        table_json       = json.dumps(
            {"columns": table_cols, "rows": table_rows},
            ensure_ascii=False,
            default=str,
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PPIFlow Design Report — {self.name}</title>

  <!-- Chart.js -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

  <!-- Mol* (Molstar) viewer -->
  <script src="https://cdn.jsdelivr.net/npm/molstar@4.5.0/build/viewer/molstar.js"></script>
  <link  rel="stylesheet" href="https://cdn.jsdelivr.net/npm/molstar@4.5.0/build/viewer/molstar.css" />

  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:         #f4f6fa;
      --surface:    #ffffff;
      --border:     #dde1e8;
      --text:       #1a1d23;
      --text-muted: #6b7280;
      --accent:     #3b5bdb;
      --accent-lt:  #eef2ff;
      --radius:     10px;
      --shadow:     0 2px 14px rgba(0,0,0,.07);
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 14px;
      line-height: 1.6;
    }}

    /* ── header ── */
    header {{
      background: var(--accent);
      color: #fff;
      padding: 26px 5vw;
    }}
    header h1 {{ font-size: 1.55rem; font-weight: 700; letter-spacing: -.4px; }}
    header p  {{ opacity: .82; margin-top: 3px; font-size: .9rem; }}

    /* ── main fluid layout ── */
    main {{
      width: 100%;
      padding: 28px 5vw;
      display: grid;
      gap: 26px;
    }}

    section {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 22px 26px;
    }}

    h2 {{
      font-size: .95rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .6px;
      color: var(--text-muted);
      margin-bottom: 16px;
      padding-bottom: 9px;
      border-bottom: 1px solid var(--border);
    }}

    /* ── funnel ── */
    .funnel-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 11px;
    }}
    .funnel-card {{
      background: var(--accent-lt);
      border: 1px solid #c5d3ff;
      border-radius: var(--radius);
      padding: 14px 16px;
    }}
    .funnel-card .step-name  {{ font-weight: 600; font-size: .82rem; color: var(--accent); }}
    .funnel-card .step-count {{ font-size: 1.7rem; font-weight: 700; color: var(--text); margin-top: 3px; }}
    .funnel-card .step-label {{ font-size: .75rem; color: var(--text-muted); }}
    .stage-label {{
      font-size: .75rem; font-weight: 700;
      text-transform: uppercase; letter-spacing: .5px;
      color: var(--accent); margin-bottom: 9px;
    }}
    .stage-block + .stage-block {{ margin-top: 20px; }}

    /* ── histogram grid ── */
    .hist-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 18px;
    }}
    .hist-card {{
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px;
      cursor: zoom-in;
      transition: box-shadow .15s;
    }}
    .hist-card:hover {{ box-shadow: 0 4px 18px rgba(0,0,0,.1); }}
    .hist-card h3 {{
      font-size: .82rem; font-weight: 600;
      margin-bottom: 9px; color: var(--text-muted);
    }}
    .hist-card canvas {{ width: 100% !important; }}

    /* ── Mol* viewer ── */
    #viewer-wrap {{
      position: relative;
      width: 100%;
      height: 480px;
      background: #1b1b2f;
      border-radius: var(--radius);
      overflow: hidden;
      cursor: zoom-in;
    }}
    #molstar-app {{ width: 100%; height: 100%; }}
    .viewer-badge {{
      position: absolute; bottom: 10px; left: 14px;
      background: rgba(0,0,0,.45);
      color: rgba(255,255,255,.85);
      font-size: .74rem; font-family: monospace;
      padding: 3px 8px; border-radius: 4px;
      pointer-events: none;
    }}

    /* ── data table ── */
    .table-wrap {{ overflow-x: auto; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: .82rem;
    }}
    thead th {{
      background: var(--accent);
      color: #fff;
      padding: 8px 11px;
      text-align: left;
      font-weight: 600;
      white-space: nowrap;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    tbody tr:nth-child(even) {{ background: var(--bg); }}
    tbody td {{
      padding: 6px 11px;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }}
    .seq-cell {{
      max-width: 200px;
      overflow: hidden;
      text-overflow: ellipsis;
      font-family: monospace;
      font-size: .76rem;
      cursor: pointer;
    }}

    /* ── pagination ── */
    .pager {{
      display: flex; gap: 5px; align-items: center;
      margin-top: 12px; flex-wrap: wrap;
    }}
    .pager button {{
      border: 1px solid var(--border);
      background: var(--surface);
      border-radius: 6px;
      padding: 4px 10px;
      font-size: .8rem;
      cursor: pointer;
      transition: background .12s;
    }}
    .pager button:hover  {{ background: var(--accent-lt); }}
    .pager button:disabled {{ opacity: .4; cursor: default; }}
    .pager button.active {{
      background: var(--accent); color: #fff;
      border-color: var(--accent);
    }}
    .page-info {{ color: var(--text-muted); font-size: .8rem; margin-left: 6px; }}

    /* ── lightbox ── */
    #lightbox {{
      display: none;
      position: fixed; inset: 0;
      background: rgba(0,0,0,.78);
      z-index: 9999;
      align-items: center; justify-content: center;
    }}
    #lightbox.open {{ display: flex; }}
    #lb-inner {{
      background: #fff;
      border-radius: var(--radius);
      padding: 18px;
      position: relative;
      max-width: 92vw;
      max-height: 92vh;
      overflow: auto;
    }}
    #lb-canvas {{ display: none; max-width: 78vw; max-height: 78vh; }}
    #lb-mol-wrap {{
      display: none;
      width: 76vw; height: 74vh;
      background: #1b1b2f;
      border-radius: var(--radius);
      overflow: hidden;
    }}
    #lb-mol-app {{ width: 100%; height: 100%; }}
    #lb-close {{
      position: absolute; top: 9px; right: 13px;
      font-size: 1.35rem; line-height: 1;
      cursor: pointer; color: var(--text-muted);
      background: none; border: none;
    }}
  </style>
</head>
<body>

<header>
  <h1>PPIFlow Design Report</h1>
  <p>Design: <strong>{self.name}</strong> &nbsp;|&nbsp; Type: <strong>{self.gentype}</strong></p>
</header>

<main>

  <!-- ========== FUNNEL ========== -->
  <section>
    <h2>Pipeline Funnel — Design Counts</h2>
    <div class="stage-block">
      <div class="stage-label">Stage 1</div>
      <div class="funnel-grid">{stage1_rows_html}</div>
    </div>
    <div class="stage-block">
      <div class="stage-label">Stage 2</div>
      <div class="funnel-grid">{stage2_rows_html}</div>
    </div>
  </section>

  <!-- ========== AF3 / REFOLD HISTOGRAMS ========== -->
  <section>
    <h2>Key Metric Distributions — AF3score &amp; ReFold</h2>
    <div class="hist-grid" id="af3-hist-grid"></div>
  </section>

  <!-- ========== RANKED DESIGNS HISTOGRAMS ========== -->
  <section>
    <h2>Ranked Designs — Score Distributions</h2>
    <div class="hist-grid" id="ranked-hist-grid"></div>
  </section>

  <!-- ========== MOL* VIEWER ========== -->
  <section>
    <h2>Top-Ranked Design Structure</h2>
    <p style="color:var(--text-muted);font-size:.83rem;margin-bottom:11px;">
      <code>{top_pdb_name}</code> &nbsp;— click to enlarge
    </p>
    <div id="viewer-wrap">
      <div id="molstar-app"></div>
      <span class="viewer-badge">{top_pdb_name}</span>
    </div>
  </section>

  <!-- ========== RANKED TABLE ========== -->
  <section>
    <h2>All Ranked Designs</h2>
    <div class="table-wrap">
      <table>
        <thead id="rank-thead"></thead>
        <tbody id="rank-tbody"></tbody>
      </table>
    </div>
    <div class="pager" id="pager"></div>
    <span class="page-info" id="page-info"></span>
  </section>

</main>

<!-- ========== LIGHTBOX ========== -->
<div id="lightbox">
  <div id="lb-inner">
    <button id="lb-close" title="Close">&times;</button>
    <canvas id="lb-canvas"></canvas>
    <div id="lb-mol-wrap"><div id="lb-mol-app"></div></div>
  </div>
</div>

<!-- ========== EMBEDDED DATA ========== -->
<script>
  const AF3_HIST_DATASETS    = {af3_hist_json};
  const RANKED_HIST_DATASETS = {ranked_hist_json};
  const TABLE_DATA           = {table_json};
  const PDB_B64              = "{top_pdb_b64}";
  const PDB_NAME             = "{top_pdb_name}";
</script>

<script>
// ====================================================================
// Shared histogram builder
// ====================================================================
(function () {{
  const PALETTE = [
    "#3b5bdb","#2f9e44","#e03131","#f08c00",
    "#9c36b5","#1098ad","#c92a2a","#5c7cfa",
  ];

  function histBins(values, nBins) {{
    if (!values.length) return {{ labels: [], counts: [] }};
    const mn = Math.min(...values), mx = Math.max(...values);
    if (mn === mx) return {{ labels: [mn.toFixed(3)], counts: [values.length] }};
    const step = (mx - mn) / nBins;
    const counts = new Array(nBins).fill(0);
    values.forEach(v => {{
      let i = Math.floor((v - mn) / step);
      if (i >= nBins) i = nBins - 1;
      counts[i]++;
    }});
    const labels = Array.from({{length: nBins}}, (_, i) =>
      (mn + i * step + step / 2).toFixed(3)
    );
    return {{ labels, counts }};
  }}

  function buildHistCharts(gridId, datasets) {{
    const grid = document.getElementById(gridId);
    if (!grid) return;

    datasets.forEach((ds, dsIdx) => {{
      ds.metrics.forEach(metric => {{
        const vals              = ds.data[metric] || [];
        const {{ labels, counts }} = histBins(vals, 20);

        const card   = document.createElement("div");
        card.className = "hist-card";
        card.innerHTML = `<h3>${{ds.label}} — ${{metric}} (n=${{vals.length}})</h3><canvas></canvas>`;
        grid.appendChild(card);

        const canvas = card.querySelector("canvas");
        const color  = PALETTE[dsIdx % PALETTE.length];

        const chartCfg = (titleText) => ({{
          type: "bar",
          data: {{
            labels,
            datasets: [{{
              label: metric,
              data: counts,
              backgroundColor: color + "bb",
              borderColor: color,
              borderWidth: 1,
            }}],
          }},
          options: {{
            responsive: true,
            plugins: {{
              legend: {{ display: false }},
              title: {{ display: !!titleText, text: titleText, font: {{ size: 14 }} }},
            }},
            scales: {{
              x: {{ ticks: {{ maxRotation: 45, font: {{ size: 10 }} }} }},
              y: {{ ticks: {{ font: {{ size: 10 }} }} }},
            }},
            animation: false,
          }},
        }});

        new Chart(canvas, chartCfg(""));

        // click → lightbox
        card.addEventListener("click", () => {{
          const lb       = document.getElementById("lightbox");
          const lbCanvas = document.getElementById("lb-canvas");
          const lbMol    = document.getElementById("lb-mol-wrap");

          lbMol.style.display    = "none";
          lbCanvas.style.display = "block";
          lb.classList.add("open");

          lbCanvas.width  = 900;
          lbCanvas.height = 520;
          if (window._lbChart) window._lbChart.destroy();
          window._lbChart = new Chart(lbCanvas, chartCfg(`${{ds.label}} — ${{metric}}`));
        }});
      }});
    }});
  }}

  buildHistCharts("af3-hist-grid",    AF3_HIST_DATASETS);
  buildHistCharts("ranked-hist-grid", RANKED_HIST_DATASETS);
}})();

// ====================================================================
// Mol* viewer  (molstar.Viewer.create + loadStructureFromData API)
// ====================================================================
(function () {{
  if (!PDB_B64) return;
  if (typeof molstar === "undefined" || typeof molstar.Viewer === "undefined") {{
    console.warn("Molstar bundle not loaded — structure viewer unavailable.");
    return;
  }}

  const pdbStr = atob(PDB_B64);

  async function initMolstar(containerId) {{
    const viewer = await molstar.Viewer.create(containerId, {{
      layoutIsExpanded:          false,
      layoutShowControls:        false,
      layoutShowSequence:        false,
      layoutShowLog:             false,
      collapseLeftPanel:         true,
      collapseRightPanel:        true,
      viewportShowExpand:        false,
      viewportShowSelectionMode: false,
    }});
    await viewer.loadStructureFromData(pdbStr, "pdb", {{ dataLabel: PDB_NAME }});
    return viewer;
  }}

  // Inline viewer
  initMolstar("molstar-app").catch(e => console.warn("Molstar inline:", e));

  // Click → lightbox viewer
  document.getElementById("viewer-wrap").addEventListener("click", () => {{
    const lb       = document.getElementById("lightbox");
    const lbCanvas = document.getElementById("lb-canvas");
    const lbMol    = document.getElementById("lb-mol-wrap");

    lbCanvas.style.display = "none";
    lbMol.style.display    = "block";
    lb.classList.add("open");

    // Wipe the container so Mol* can attach cleanly
    const app = document.getElementById("lb-mol-app");
    app.innerHTML = "";
    setTimeout(() => {{
      initMolstar("lb-mol-app").catch(e => console.warn("Molstar lightbox:", e));
    }}, 80);
  }});
}})();

// ====================================================================
// Ranked designs table (paginated)
// ====================================================================
(function () {{
  const PAGE_SIZE = 20;
  let currentPage = 1;

  const {{ columns: cols, rows }} = TABLE_DATA;
  if (!cols || !cols.length) return;

  const SEQ_COLS   = new Set(["binder_seq", "heavy_seq", "light_seq"]);
  const FLOAT_COLS = new Set(["rank_score", "interface_score", "dockq_mean"]);

  // Build header
  const thead = document.getElementById("rank-thead");
  const hRow  = document.createElement("tr");
  cols.forEach(c => {{
    const th = document.createElement("th");
    th.textContent = c;
    hRow.appendChild(th);
  }});
  thead.appendChild(hRow);

  function renderPage(page) {{
    currentPage = page;
    const tbody = document.getElementById("rank-tbody");
    tbody.innerHTML = "";

    const start = (page - 1) * PAGE_SIZE;
    const end   = Math.min(start + PAGE_SIZE, rows.length);

    for (let i = start; i < end; i++) {{
      const row = rows[i];
      const tr  = document.createElement("tr");
      cols.forEach(c => {{
        const td  = document.createElement("td");
        let val   = row[c] ?? "";
        // Format floats
        if (FLOAT_COLS.has(c) && val !== "") {{
          const n = parseFloat(val);
          if (!isNaN(n)) val = n.toFixed(4);
        }} else if (!SEQ_COLS.has(c) && val !== "") {{
          const n = parseFloat(val);
          if (!isNaN(n) && isFinite(n)) val = n.toFixed(4);
        }}
        td.textContent = val;
        if (SEQ_COLS.has(c)) {{
          td.className = "seq-cell";
          td.title     = row[c] ?? "";
        }}
        tr.appendChild(td);
      }});
      tbody.appendChild(tr);
    }}

    renderPager(page);
    document.getElementById("page-info").textContent =
      `Showing ${{start + 1}}–${{end}} of ${{rows.length}} designs`;
  }}

  function renderPager(page) {{
    const total = Math.ceil(rows.length / PAGE_SIZE);
    const pg    = document.getElementById("pager");
    pg.innerHTML = "";

    const mkBtn = (label, targetPage, disabled) => {{
      const b = document.createElement("button");
      b.textContent = label;
      b.disabled    = disabled;
      if (targetPage === page) b.classList.add("active");
      b.addEventListener("click", () => renderPage(targetPage));
      return b;
    }};

    pg.appendChild(mkBtn("‹ Prev", page - 1, page === 1));

    const WING  = 3;
    let lo = Math.max(1, page - WING);
    let hi = Math.min(total, page + WING);
    if (hi - lo < WING * 2) {{
      lo = Math.max(1, hi - WING * 2);
      hi = Math.min(total, lo + WING * 2);
    }}
    for (let p = lo; p <= hi; p++) pg.appendChild(mkBtn(p, p, false));

    pg.appendChild(mkBtn("Next ›", page + 1, page === total));
  }}

  renderPage(1);
}})();

// ====================================================================
// Lightbox close
// ====================================================================
document.getElementById("lb-close").addEventListener("click", () => {{
  document.getElementById("lightbox").classList.remove("open");
}});
document.getElementById("lightbox").addEventListener("click", e => {{
  if (e.target === e.currentTarget)
    e.currentTarget.classList.remove("open");
}});
</script>
</body>
</html>
"""
        return html

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> str:
        """Generate the HTML report and write it to *output_dir*.

        Returns
        -------
        str
            Absolute path to the written HTML file.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        html = self._render_html()

        output_path = os.path.join(self.output_dir, self.report_filename)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        print(f"[ReportStep] Report written to: {output_path}")
        return output_path
