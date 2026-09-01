"""HI-VIS — LLM/VLM vs. our trained YOLO detector, presented as two slides.

Slide 1 (ALL RUNS) is built live from every `runs/llm/*/run_manifest.json`
on disk, plus any `prompt_comparison.csv` runs that have no manifest (the
qualitative, unscored ones) — this is a survey of the whole comparison
effort, not just the final numbers. Slide 2 (LATEST RUN) scores the most
recent scored run (by manifest `created_at`) live from its `presence.csv`
against `data/merged/labels_long.csv` + `data.yaml`, falling back to the
baked-in numbers from `reports/llm_vs_yolo_comparison.md` if the ground
truth files aren't present on this checkout (they're git-ignored, like the
rest of `data/`) — same pattern `pages/demo.py` uses for missing weights:
degrade with an explanation, never fake a result.

Color: sequential black->white for the F1 heatmap (magnitude, one hue);
categorical black-vs-muted-grey for "ours vs. every VLM" in the bar chart,
with red reserved for negative-class bars specifically (matches this app's
existing non-compliant=red convention) — never a rainbow, never color-only
identity (every bar/cell is also directly labeled).
"""

import glob
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml
from PIL import Image

import view_helpers as vh

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_LLM_ROOT = REPO_ROOT / "runs" / "llm"
MERGED_ROOT = REPO_ROOT / "data" / "merged"
MERGED_TEST_IMAGES = MERGED_ROOT / "test" / "images"


def image_path_for(file_stem):
    """Same lookup scripts/compare_models.py uses — duplicated locally (it's
    three lines) rather than importing across into scripts/, which isn't on
    this page's Python path and pulls in heavier module-level deps."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = MERGED_TEST_IMAGES / f"{file_stem}{ext}"
        if p.exists():
            return p
    return None


st.markdown(vh.HV_STYLE_CSS, unsafe_allow_html=True)
st.markdown(vh.header_html("LLM vs YOLO COMPARISON", "runs/llm/ — every comparison run"), unsafe_allow_html=True)

INK = "#141414"
MUTED = "#71736D"
FAINT = "#C4C6C0"
NEGATIVE_RED = "#B02A20"
CHART_FONT = "IBM Plex Sans, sans-serif"
alt.themes.enable("none")

MODEL_LABEL = {
    "yolo": "YOLO26 (ours)", "gemini": "Gemini 3.6 Flash", "qwen3-vl": "Qwen3-VL", "gemma4": "Gemma 4",
    "minicpm-v": "MiniCPM-V", "ollama": "LLaVA (Ollama)", "florence2": "Florence-2", "yoloe": "YOLO-E",
}
CLASS_ORDER = ["person", "helmet", "gloves", "boots", "vest", "no-helmet", "no-gloves", "no-boots", "no-vest"]
POSITIVE_CLASSES = {"person", "helmet", "gloves", "boots", "vest"}

# Fallback macro-level metrics — reports/llm_vs_yolo_comparison.md, used only
# if data/merged/labels_long.csv (git-ignored) isn't present on this
# checkout to score the latest run's presence.csv live.
# (tp, fp, fn, tn) per class.
_FALLBACK_RAW = {
    "yolo":      {"person": (49, 0, 0, 51), "helmet": (47, 1, 2, 50), "gloves": (31, 0, 2, 67), "boots": (32, 0, 3, 65), "vest": (29, 2, 3, 66),
                  "no-helmet": (31, 2, 4, 63), "no-gloves": (17, 1, 1, 81), "no-boots": (16, 0, 0, 84), "no-vest": (22, 1, 2, 75)},
    "gemini":    {"person": (48, 51, 1, 0), "helmet": (45, 2, 4, 49), "gloves": (27, 10, 6, 57), "boots": (25, 10, 10, 55), "vest": (28, 0, 4, 68),
                  "no-helmet": (33, 33, 2, 32), "no-gloves": (13, 58, 5, 24), "no-boots": (12, 29, 4, 55), "no-vest": (20, 59, 4, 17)},
}


def _base_config(chart):
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(labelFont=CHART_FONT, titleFont=CHART_FONT, labelColor=MUTED, titleColor=INK,
                         grid=False, domainColor=FAINT, tickColor=FAINT, labelFontSize=11.5, titleFontSize=11.5)
        .configure_legend(labelFont=CHART_FONT, titleFont=CHART_FONT, labelColor=INK, titleColor=INK,
                           labelFontSize=11.5, titleFontSize=11.5, orient="top", symbolType="square")
        .configure_text(font=CHART_FONT)
    )


def model_chip(name):
    label = MODEL_LABEL.get(name, name)
    bg = INK if name == "yolo" else "#FFFFFF"
    fg = "#FFFFFF" if name == "yolo" else INK
    border = INK if name == "yolo" else "#C4C6C0"
    return (f'<span class="hv-mono" style="display:inline-block;font-size:10.5px;padding:3px 8px;'
            f'background:{bg};color:{fg};border:1px solid {border};margin:2px 4px 2px 0;white-space:nowrap">{label}</span>')


# ---------------------------------------------------------------------------
# load every run on disk
# ---------------------------------------------------------------------------

@st.cache_data
def load_all_runs():
    scored_runs = []
    for f in sorted(glob.glob(str(RUNS_LLM_ROOT / "*" / "run_manifest.json"))):
        m = json.loads(Path(f).read_text())
        scored_runs.append({
            "run_name": m["run_name"], "kind": "scored", "created_at": pd.Timestamp(m["created_at"]),
            "n_images": m["n_images_sampled"], "models": m["models"],
            "parse_failures": sum(m.get("parse_failures", {}).values()),
        })
    scored_dirs = {Path(f).parent.name for f in glob.glob(str(RUNS_LLM_ROOT / "*" / "run_manifest.json"))}
    qualitative_runs = []
    for f in sorted(glob.glob(str(RUNS_LLM_ROOT / "*" / "prompt_comparison.csv"))):
        run_dir = Path(f).parent
        if run_dir.name in scored_dirs:
            continue
        df = pd.read_csv(f)
        # run_name embeds a UTC "YYYYMMDD_HHMMSS" prefix — parse it back out
        # for a real timestamp rather than falling back to file mtime, which
        # only reflects when this checkout last touched the file.
        ts = pd.to_datetime(run_dir.name[:15], format="%Y%m%d_%H%M%S", utc=True, errors="coerce")
        qualitative_runs.append({
            "run_name": run_dir.name, "kind": "qualitative (prompt-style, unscored)",
            "created_at": ts if pd.notna(ts) else pd.Timestamp(run_dir.stat().st_mtime, unit="s", tz="UTC"),
            "n_images": df["file"].nunique(), "models": ["gemini"], "parse_failures": 0,
        })
    runs = pd.DataFrame(scored_runs + qualitative_runs)
    return runs.sort_values("created_at").reset_index(drop=True)


all_runs = load_all_runs()

# ---------------------------------------------------------------------------
# live scoring of the latest scored run, with a baked-in fallback
# ---------------------------------------------------------------------------

@st.cache_data
def score_latest_run():
    scored_only = all_runs[all_runs["kind"] == "scored"]
    if scored_only.empty:
        return None, None
    latest_run_name = scored_only.iloc[-1]["run_name"]
    run_dir = RUNS_LLM_ROOT / latest_run_name
    presence_path = run_dir / "presence.csv"
    labels_path = MERGED_ROOT / "labels_long.csv"
    data_yaml_path = MERGED_ROOT / "data.yaml"
    if not (presence_path.exists() and labels_path.exists() and data_yaml_path.exists()):
        return None, latest_run_name

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    presence = pd.read_csv(presence_path)
    labels_long = pd.read_csv(labels_path)
    class_names = yaml.safe_load(data_yaml_path.read_text())["names"]
    id_to_name = {i: n for i, n in enumerate(class_names)}
    labels_long = labels_long.copy()
    labels_long["class_name"] = labels_long["class_id"].map(id_to_name)
    sampled_files = manifest["sampled_files"]
    gt = labels_long[labels_long["file"].isin(sampled_files)]
    gt_by_file = gt.groupby("file")["class_name"].apply(set).to_dict()

    rows = []
    for model in manifest["models"]:
        sub = presence[(presence["model"] == model) & (~presence["parse_error"])]
        sub_by_key = {(r.file, r.class_name): bool(r.present) for r in sub.itertuples()}
        for cls in sorted(sub["class_name"].unique()):
            tp = fp = fn = tn = 0
            for f in sampled_files:
                if (f, cls) not in sub_by_key:
                    continue
                pred = sub_by_key[(f, cls)]
                actual = cls in gt_by_file.get(f, set())
                if pred and actual:
                    tp += 1
                elif pred and not actual:
                    fp += 1
                elif not pred and actual:
                    fn += 1
                else:
                    tn += 1
            rows.append({"model": model, "class_name": cls, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return pd.DataFrame(rows), latest_run_name


def metrics_from_raw(raw_dict):
    rows = []
    for model, classes in raw_dict.items():
        for cls, (tp, fp, fn, tn) in classes.items():
            rows.append({"model": model, "class_name": cls, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return pd.DataFrame(rows)


def add_prf1(df):
    df = df.copy()
    total = df["tp"] + df["tn"] + df["fp"] + df["fn"]
    # Accuracy and recall are our main metrics. Precision/F1 stay as
    # secondary diagnostic columns (e.g. explaining why accuracy is high but
    # recall is low), not the headline numbers.
    df["accuracy"] = (df["tp"] + df["tn"]) / total.replace(0, pd.NA)
    df["recall"] = df["tp"] / (df["tp"] + df["fn"]).replace(0, pd.NA)
    df["precision"] = df["tp"] / (df["tp"] + df["fp"]).replace(0, pd.NA)
    for col in ("accuracy", "recall", "precision"):
        df[col] = df[col].astype(float)
    denom = df["precision"] + df["recall"]
    df["f1"] = (2 * df["precision"] * df["recall"] / denom).where(denom > 0)
    df["model_label"] = df["model"].map(lambda m: MODEL_LABEL.get(m, m))
    df["group"] = df["class_name"].map(lambda c: "Positive (present)" if c in POSITIVE_CLASSES else "Negative (absent)")
    return df


live_raw, latest_run_name = score_latest_run()
using_live_data = live_raw is not None
metrics = add_prf1(live_raw if using_live_data else metrics_from_raw(_FALLBACK_RAW))
if latest_run_name is None:
    latest_run_name = "runs/llm/20260831_merged_n100_seed42_yolo-gemini"

model_order_present = list(dict.fromkeys(metrics.sort_values("model")["model"]))
MODEL_ORDER = sorted(model_order_present, key=lambda m: (m != "yolo", m != "gemini", m))
MODEL_LABEL_ORDER = [MODEL_LABEL.get(m, m) for m in MODEL_ORDER]

# Macro (mean-of-classes) accuracy and recall — our two main metrics — split
# by positive/negative class group, plus an overall (all-9-class) mean of
# each. Precision/F1 aren't tabled at this macro level; they're still on the
# per-class heatmap below for anyone who wants the detail.
def macro_table(metric):
    t = (
        metrics.groupby(["model", "model_label", "group"])[metric].mean().reset_index()
        .pivot(index=["model", "model_label"], columns="group", values=metric).reset_index()
    )
    t["overall"] = metrics.groupby("model")[metric].mean().reindex(t["model"]).values
    return t.set_index("model").loc[MODEL_ORDER].reset_index()


macro = macro_table("accuracy")
macro_recall = macro_table("recall")

# ---------------------------------------------------------------------------
# slide switcher
# ---------------------------------------------------------------------------

st.markdown("""
<style>
div[role="radiogroup"] { gap: 0 !important; }
</style>
""", unsafe_allow_html=True)

slide = st.radio(
    "Slide", ["① ALL RUNS", "② LATEST RUN"], horizontal=True, label_visibility="collapsed", key="llm_slide",
)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ===========================================================================
# SLIDE 1 — every run, at a glance
# ===========================================================================

if slide == "① ALL RUNS":
    n_runs = len(all_runs)
    all_models_ever = sorted({m for models in all_runs["models"] for m in models} - {"yolo"})
    total_images_scored = int(all_runs.loc[all_runs["kind"] == "scored", "n_images"].sum())
    date_span = f'{all_runs["created_at"].min():%b %d} – {all_runs["created_at"].max():%b %d}'

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:22px">
      <div style="background:#141414;color:#FFFFFF;padding:16px 20px 14px">
        <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#9B9D97">COMPARISON RUNS</div>
        <div class="hv-h1" style="font-size:48px;line-height:1;color:#FFFFFF">{n_runs}</div>
        <div style="font-size:12px;color:#9B9D97">{date_span}, 2026</div>
      </div>
      <div style="background:#FFFFFF;color:#141414;border:1px solid #C4C6C0;padding:16px 20px 14px">
        <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#71736D">VLMS TRIED AGAINST YOLO</div>
        <div class="hv-h1" style="font-size:48px;line-height:1">{len(all_models_ever)}</div>
        <div style="font-size:12px;color:#71736D">{", ".join(MODEL_LABEL.get(m, m) for m in all_models_ever)}</div>
      </div>
      <div style="background:#FFFFFF;color:#141414;border:1px solid #C4C6C0;padding:16px 20px 14px">
        <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#71736D">IMAGES SCORED (SCORED RUNS)</div>
        <div class="hv-h1" style="font-size:48px;line-height:1">{total_images_scored}</div>
        <div style="font-size:12px;color:#71736D">across {(all_runs["kind"] == "scored").sum()} scored runs</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hv-h1" style="font-size:22px;margin-bottom:2px">EVERY RUN, IN ORDER</div>', unsafe_allow_html=True)
    st.caption("Bar length = images sampled. The lineup grew (and dropped experiments) run over run — see the model chips below each bar.")

    timeline_df = all_runs.copy()
    timeline_df["row_label"] = timeline_df.apply(
        lambda r: f'{r["created_at"]:%b %d, %H:%M} UTC — {"scored" if r["kind"] == "scored" else "prompt-style"}', axis=1)
    timeline_df["row_order"] = range(len(timeline_df))

    timeline_df["models_str"] = timeline_df["models"].apply(lambda ms: ", ".join(MODEL_LABEL.get(m, m) for m in ms))
    row_h = 42
    chart_h = row_h * len(timeline_df)

    tl_bar = (
        alt.Chart(timeline_df)
        .mark_bar(size=row_h - 14, cornerRadiusTopRight=2, cornerRadiusBottomRight=2)
        .encode(
            y=alt.Y("row_label:N", title=None, sort=alt.SortField("row_order"),
                    scale=alt.Scale(paddingInner=0.3, paddingOuter=0.2)),
            x=alt.X("n_images:Q", title="images sampled"),
            color=alt.Color("kind:N", title=None,
                             scale=alt.Scale(domain=["scored", "qualitative (prompt-style, unscored)"], range=[INK, FAINT])),
            tooltip=[alt.Tooltip("run_name:N", title="Run"), alt.Tooltip("n_images:Q", title="Images"),
                     alt.Tooltip("models_str:N", title="Models")],
        )
        .properties(height=chart_h)
    )
    tl_text = (
        alt.Chart(timeline_df)
        .mark_text(align="left", dx=6, font=CHART_FONT, fontSize=11, color=INK)
        .encode(y=alt.Y("row_label:N", sort=alt.SortField("row_order"), scale=alt.Scale(paddingInner=0.3, paddingOuter=0.2)),
                x="n_images:Q", text="n_images:Q")
        .properties(height=chart_h)
    )
    st.altair_chart(_base_config(tl_bar + tl_text), width="stretch")

    for _, r in timeline_df.sort_values("row_order", ascending=False).iterrows():
        chips = "".join(model_chip(m) for m in r["models"])
        note = f' · {r["parse_failures"]} parse failures' if r["parse_failures"] else ""
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #C4C6C0;padding:8px 12px;margin-bottom:6px;
             display:flex;align-items:center;gap:14px;flex-wrap:wrap">
          <span class="hv-mono" style="font-size:10.5px;color:#71736D;white-space:nowrap">{r["created_at"]:%b %d %H:%M}</span>
          <span style="font-size:11px;color:#4A4B47;white-space:nowrap">{r["n_images"]} img · {r["kind"]}{note}</span>
          <span>{chips}</span>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Florence-2 and YOLO-E (Aug 27) were tried as alternative grounding-capable baselines alongside YOLO, then "
        "dropped from later runs in favor of the settled 5-VLM lineup (ollama/qwen3-vl/gemma4/minicpm-v) plus, "
        "most recently, Gemini. Source: every run_manifest.json + prompt_comparison.csv under runs/llm/."
    )

# ===========================================================================
# SLIDE 2 — the latest run, in depth
# ===========================================================================

else:
    if not using_live_data:
        st.markdown(
            '<div style="background:#FFFFFF;border:1px dashed #9B9D97;padding:10px 16px;margin-bottom:16px;'
            'font-size:12px;color:#71736D">Showing the last known-good numbers from '
            '<code>reports/llm_vs_yolo_comparison.md</code> — <code>data/merged/labels_long.csv</code> isn\'t on '
            'this checkout (it\'s git-ignored) so the latest run couldn\'t be scored live.</div>',
            unsafe_allow_html=True,
        )

    yolo_overall = macro.loc[macro["model"] == "yolo", "overall"].iloc[0] if "yolo" in macro["model"].values else None
    best_vlm_row = macro[macro["model"] != "yolo"].assign(_o=lambda d: d["overall"]).sort_values("_o", ascending=False)
    best_vlm = best_vlm_row.iloc[0] if not best_vlm_row.empty else None

    yolo_recall = macro_recall.loc[macro_recall["model"] == "yolo", "overall"].iloc[0] if "yolo" in macro_recall["model"].values else None
    best_vlm_recall = macro_recall.loc[macro_recall["model"] == best_vlm["model"], "overall"].iloc[0] if best_vlm is not None else None

    tiles = ['<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:22px">']
    if yolo_overall is not None:
        tiles.append(f"""
          <div style="background:#141414;color:#FFFFFF;padding:16px 20px 14px">
            <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#9B9D97">YOLO26 (OURS) — ACCURACY / RECALL</div>
            <div class="hv-h1" style="font-size:48px;line-height:1;color:#FFFFFF">{yolo_overall:.2f} / {yolo_recall:.2f}</div>
            <div style="font-size:12px;color:#9B9D97">macro-averaged across all 9 classes</div>
          </div>""")
    if best_vlm is not None:
        tiles.append(f"""
          <div style="background:#FFFFFF;color:#141414;border:1px solid #C4C6C0;padding:16px 20px 14px">
            <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#71736D">BEST VLM ({best_vlm["model_label"].upper()}) — ACCURACY / RECALL</div>
            <div class="hv-h1" style="font-size:48px;line-height:1">{best_vlm["overall"]:.2f} / {best_vlm_recall:.2f}</div>
            <div style="font-size:12px;color:#71736D">strongest non-YOLO model in this run</div>
          </div>""")
    if yolo_recall is not None and best_vlm_recall is not None:
        yolo_neg_recall = macro_recall.loc[macro_recall["model"] == "yolo", "Negative (absent)"].iloc[0]
        vlm_neg_recall = macro_recall.loc[macro_recall["model"] == best_vlm["model"], "Negative (absent)"].iloc[0]
        gap_x = yolo_neg_recall / vlm_neg_recall if vlm_neg_recall else float("nan")
        tiles.append(f"""
          <div style="background:#EFE600;color:#141414;padding:16px 20px 14px">
            <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#3A3B30">RECALL GAP ON ABSENCE DETECTION</div>
            <div class="hv-h1" style="font-size:48px;line-height:1">{gap_x:.1f}×</div>
            <div style="font-size:12px;color:#3A3B30">YOLO's negative-class recall ({yolo_neg_recall:.2f}) vs. best VLM's ({vlm_neg_recall:.2f})</div>
          </div>""")
    tiles.append("</div>")
    st.markdown("".join(tiles), unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-size:12.5px;color:#4A4B47;margin:-8px 0 20px">Latest scored run: '
        f'<code>{latest_run_name}</code> — {"scored live from presence.csv + ground truth" if using_live_data else "baked-in fallback"}. '
        f'Full write-up: <code>reports/llm_vs_yolo_comparison.md</code>.</div>',
        unsafe_allow_html=True,
    )

    def macro_bar(table, metric_label):
        """One row of grouped bars (positive vs. negative class group) per
        model, for a single macro metric — used twice below (accuracy, then
        recall), so the shape lives here once."""
        df = table.melt(id_vars=["model", "model_label"], value_vars=["Positive (present)", "Negative (absent)"],
                         var_name="group", value_name="value").dropna(subset=["value"])
        bar = (
            alt.Chart()
            .mark_bar(size=16, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
            .encode(
                x=alt.X("group:N", title=None, axis=None),
                y=alt.Y("value:Q", title=f"Macro {metric_label}", scale=alt.Scale(domain=[0, 1.02])),
                color=alt.Color("group:N", title=None,
                                 scale=alt.Scale(domain=["Positive (present)", "Negative (absent)"], range=[INK, NEGATIVE_RED])),
                tooltip=[alt.Tooltip("model_label:N", title="Model"), alt.Tooltip("group:N", title="Class group"),
                         alt.Tooltip("value:Q", title=f"Macro {metric_label}", format=".3f")],
            )
        )
        text = (
            alt.Chart()
            .mark_text(dy=-6, font=CHART_FONT, fontSize=10.5, color=INK)
            .encode(x=alt.X("group:N", axis=None), y=alt.Y("value:Q"), text=alt.Text("value:Q", format=".2f"))
        )
        return (
            alt.layer(bar, text, data=df)
            .properties(width=70, height=220)
            .facet(column=alt.Column("model_label:N", title=None, sort=MODEL_LABEL_ORDER,
                                      header=alt.Header(labelFont=CHART_FONT, labelFontSize=12.5, labelColor=INK, labelOrient="bottom")))
        )

    st.markdown('<div class="hv-h1" style="font-size:22px;margin-bottom:2px">MACRO ACCURACY BY MODEL</div>', unsafe_allow_html=True)
    st.caption("Negative classes (no-helmet, no-boots, …) are where every VLM falls apart — YOLO barely notices the difference.")
    st.altair_chart(_base_config(macro_bar(macro, "accuracy")), width="stretch")

    st.markdown('<div class="hv-h1" style="font-size:22px;margin:26px 0 2px">MACRO RECALL BY MODEL</div>', unsafe_allow_html=True)
    st.caption("Recall = of the cases actually present, how many the model caught — the number that matters most for a safety product.")
    st.altair_chart(_base_config(macro_bar(macro_recall, "recall")), width="stretch")

    st.markdown('<div class="hv-h1" style="font-size:22px;margin:26px 0 2px">PER-CLASS ACCURACY — EVERY MODEL, EVERY CLASS</div>',
                unsafe_allow_html=True)
    st.caption("Darker = higher accuracy. Reads left→right as PPE-present classes, then the four absence classes.")

    heat_df = metrics.copy()
    heat_df["accuracy_display"] = heat_df["accuracy"].apply(lambda v: "—" if pd.isna(v) else f"{v:.2f}")

    cells = (
        alt.Chart(heat_df)
        .mark_rect(stroke="#E4E5E2", strokeWidth=2)
        .encode(
            x=alt.X("class_name:N", title=None, sort=CLASS_ORDER, axis=alt.Axis(labelAngle=-40, labelFontSize=11)),
            y=alt.Y("model_label:N", title=None, sort=MODEL_LABEL_ORDER),
            color=alt.Color("accuracy:Q", title="Accuracy", scale=alt.Scale(scheme="greys", domain=[0, 1]),
                             legend=alt.Legend(orient="right", gradientLength=140)),
            tooltip=[alt.Tooltip("model_label:N", title="Model"), alt.Tooltip("class_name:N", title="Class"),
                     alt.Tooltip("accuracy:Q", title="Accuracy", format=".3f"), alt.Tooltip("recall:Q", title="Recall", format=".3f"),
                     alt.Tooltip("tp:Q", title="tp"), alt.Tooltip("fp:Q", title="fp"),
                     alt.Tooltip("fn:Q", title="fn"), alt.Tooltip("tn:Q", title="tn")],
        )
        .properties(height=max(120, 40 * len(MODEL_ORDER)))
    )
    labels = (
        alt.Chart(heat_df)
        .mark_text(font=CHART_FONT, fontSize=11)
        .encode(
            x=alt.X("class_name:N", sort=CLASS_ORDER), y=alt.Y("model_label:N", sort=MODEL_LABEL_ORDER),
            text="accuracy_display:N",
            color=alt.condition(alt.datum.accuracy > 0.55, alt.value("#FFFFFF"), alt.value(INK)),
        )
        .properties(height=max(120, 40 * len(MODEL_ORDER)))
    )
    st.altair_chart(_base_config((cells + labels)), width="stretch")

    if "person" in metrics["class_name"].values:
        st.markdown('<div class="hv-h1" style="font-size:22px;margin:26px 0 8px">WHY "PERSON" LOOKS WORSE THAN IT IS</div>',
                    unsafe_allow_html=True)
        gt_col1, gt_col2 = st.columns([3, 2])
        with gt_col1:
            person_row = metrics[(metrics["class_name"] == "person") & (metrics["model"] != "yolo")].iloc[0] \
                if (metrics["class_name"] == "person").any() and (metrics["model"] != "yolo").any() else None
            if person_row is not None:
                n_total = int(person_row["tp"] + person_row["fp"])
                gt_df = pd.DataFrame([
                    {"label": "Has a person box", "count": int(person_row["tp"])},
                    {"label": "PPE boxed, but no person box", "count": int(person_row["fp"])},
                ])
                gt_bar = (
                    alt.Chart(gt_df).mark_bar(size=44, cornerRadiusTopRight=2, cornerRadiusBottomRight=2)
                    .encode(
                        y=alt.Y("label:N", title=None, sort=["Has a person box", "PPE boxed, but no person box"]),
                        x=alt.X("count:Q", title=f"of {n_total} sampled test images", scale=alt.Scale(domain=[0, n_total])),
                        color=alt.Color("label:N", scale=alt.Scale(domain=["Has a person box", "PPE boxed, but no person box"],
                                                                     range=[INK, FAINT]), legend=None),
                        tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("count:Q", title="images")],
                    ).properties(height=110)
                )
                gt_text = (
                    alt.Chart(gt_df).mark_text(align="left", dx=6, font=CHART_FONT, fontSize=13, fontWeight="bold", color=INK)
                    .encode(y=alt.Y("label:N", sort=["Has a person box", "PPE boxed, but no person box"]), x="count:Q", text="count:Q")
                    .properties(height=110)
                )
                st.altair_chart(_base_config(gt_bar + gt_text), width="stretch")
        with gt_col2:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px dashed #9B9D97;padding:16px;height:100%;box-sizing:border-box">
              <div style="font-size:13px;line-height:1.5;color:#141414">
              Every non-YOLO model in this run reports the same false-positive count on <code>person</code> — that's
              not a coincidence; it's a labeling gap in the test set. Every VLM correctly says <i>"yes, there's a
              person"</i> and gets marked wrong by an incomplete ground-truth box.
              </div>
              <div style="font-size:11.5px;color:#71736D;margin-top:10px">YOLO scores near-perfectly here only because it was
              <i>trained</i> on this same gapped label set — consistency with the labels, not a stronger read on the photo.</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:30px 0 18px'/>", unsafe_allow_html=True)
    st.markdown('<div class="hv-h1" style="font-size:18px;margin-bottom:2px;color:#71736D">EXTRA — GEMINI: DESCRIPTIVE VS. STRUCTURED PROMPT</div>',
                unsafe_allow_html=True)
    st.caption(
        "Unscored, from a separate qualitative run — there's no ground truth for prose. Same images, two prompts: "
        "a free-text \"site record\" description, and the strict-JSON prompt everything above is scored on."
    )
    candidates = sorted(glob.glob(str(RUNS_LLM_ROOT / "*" / "prompt_comparison.csv")))
    if not candidates:
        st.markdown(
            '<div style="background:#FFFFFF;border:1px solid #C4C6C0;padding:20px;color:#71736D;font-size:13px">'
            'No <code>prompt_comparison.csv</code> found under <code>runs/llm/</code> yet — run '
            '<code>scripts/gemini_prompt_comparison.py</code> to populate this section.</div>',
            unsafe_allow_html=True,
        )
    else:
        prompt_df = pd.read_csv(candidates[-1])
        pivot = prompt_df.pivot(index="file", columns="prompt_name", values="response_text")
        sample_files = pivot.index[:3]
        cols = st.columns(len(sample_files))
        for col, file in zip(cols, sample_files):
            with col:
                img_path = image_path_for(file)
                if img_path is not None:
                    thumb_b64 = vh.b64_image(Image.open(img_path).convert("RGB"), max_dim=360)
                    st.markdown(f'<img src="data:image/jpeg;base64,{thumb_b64}" '
                                f'style="width:100%;aspect-ratio:4/3;object-fit:cover;display:block;border:1px solid #C4C6C0"/>',
                                unsafe_allow_html=True)
                descriptive = pivot.loc[file].get("1_new_descriptive", "—")
                structured = pivot.loc[file].get("2_existing_structured", "—")
                st.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #C4C6C0;border-top:none;padding:12px 14px 14px">
                  <div style="font-size:12.5px;line-height:1.45;color:#141414">{descriptive}</div>
                  <div class="hv-mono" style="font-size:10px;color:#71736D;background:#F0F1EC;padding:6px 8px;margin-top:10px;
                       white-space:pre-wrap;word-break:break-word">{structured}</div>
                </div>
                """, unsafe_allow_html=True)

    st.caption("Source data: every run_manifest.json / presence.csv / prompt_comparison.csv under runs/llm/.")
