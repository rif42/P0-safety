"""HI-VIS — LLM/VLM vs. our trained YOLO detector, presented visually.

A read-only showcase page, not a live tool: every number here is baked in
from the scored comparison run (see `reports/llm_vs_yolo_comparison.md` for
the full write-up and `runs/llm/20260831_merged_n100_seed42_yolo-gemini/`
+ `runs/llm/20260828_035813_..._yolo-ollama-qwen3-vl-gemma4-minicpm-v/` for
the underlying data) — same 100 held-out test images, seed 42, for every
model. The one live-ish bit is the descriptive-vs-structured prompt section
at the bottom, which reads `runs/llm/*/prompt_comparison.csv` off disk if
present and degrades gracefully if it isn't (that run directory isn't
committed — see scripts/gemini_prompt_comparison.py to regenerate it).

Color: sequential black->white for the F1 heatmap (magnitude, one hue);
categorical black-vs-muted-grey for "ours vs. every VLM" in the bar chart,
with red reserved for negative-class bars specifically (matches this app's
existing non-compliant=red convention) — never a rainbow, never color-only
identity (every bar/cell is also directly labeled).
"""

import glob
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

import view_helpers as vh

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_TEST_IMAGES = REPO_ROOT / "data" / "merged" / "test" / "images"


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
st.markdown(vh.header_html("LLM vs YOLO COMPARISON", "n=100 · seed 42 · presence/absence"), unsafe_allow_html=True)

INK = "#141414"
MUTED = "#71736D"
FAINT = "#C4C6C0"
NEGATIVE_RED = "#B02A20"

MODEL_ORDER = ["yolo", "gemini", "qwen3-vl", "gemma4", "minicpm-v", "ollama"]
MODEL_LABEL = {
    "yolo": "YOLO26 (ours)", "gemini": "Gemini 3.6 Flash", "qwen3-vl": "Qwen3-VL",
    "gemma4": "Gemma 4", "minicpm-v": "MiniCPM-V", "ollama": "LLaVA (Ollama)",
}
CLASS_ORDER = ["person", "helmet", "gloves", "boots", "vest", "no-helmet", "no-gloves", "no-boots", "no-vest"]

# tp/fp/fn straight from reports/llm_vs_yolo_comparison.md — two runs, same
# 100-image/seed-42 sample (YOLO landed byte-identical across both, as a
# deterministic checkpoint should).
_RAW = {
    "yolo":      {"person": (49, 0, 0), "helmet": (47, 1, 2), "gloves": (31, 0, 2), "boots": (32, 0, 3), "vest": (29, 2, 3),
                  "no-helmet": (31, 2, 4), "no-gloves": (17, 1, 1), "no-boots": (16, 0, 0), "no-vest": (22, 1, 2)},
    "gemini":    {"person": (48, 51, 1), "helmet": (45, 2, 4), "gloves": (27, 10, 6), "boots": (25, 10, 10), "vest": (28, 0, 4),
                  "no-helmet": (33, 33, 2), "no-gloves": (13, 58, 5), "no-boots": (12, 29, 4), "no-vest": (20, 59, 4)},
    "qwen3-vl":  {"person": (48, 51, 1), "helmet": (46, 2, 3), "gloves": (24, 15, 9), "boots": (23, 16, 12), "vest": (26, 5, 6),
                  "no-helmet": (21, 13, 14), "no-gloves": (5, 35, 13), "no-boots": (9, 27, 7), "no-vest": (8, 23, 16)},
    "gemma4":    {"person": (49, 51, 0), "helmet": (46, 22, 3), "gloves": (29, 37, 4), "boots": (35, 55, 0), "vest": (31, 45, 1),
                  "no-helmet": (17, 10, 18), "no-gloves": (3, 31, 15), "no-boots": (2, 9, 14), "no-vest": (9, 13, 15)},
    "minicpm-v": {"person": (49, 51, 0), "helmet": (45, 11, 4), "gloves": (24, 38, 9), "boots": (34, 53, 1), "vest": (30, 24, 2),
                  "no-helmet": (11, 9, 24), "no-gloves": (5, 34, 13), "no-boots": (0, 14, 16), "no-vest": (9, 22, 15)},
    "ollama":    {"person": (49, 51, 0), "helmet": (34, 28, 15), "gloves": (19, 33, 14), "boots": (21, 39, 14), "vest": (21, 34, 11),
                  "no-helmet": (17, 31, 18), "no-gloves": (12, 40, 6), "no-boots": (11, 37, 5), "no-vest": (9, 27, 15)},
}
POSITIVE_CLASSES = {"person", "helmet", "gloves", "boots", "vest"}


@st.cache_data
def build_metrics_df():
    rows = []
    for model, classes in _RAW.items():
        for cls, (tp, fp, fn) in classes.items():
            precision = tp / (tp + fp) if (tp + fp) else float("nan")
            recall = tp / (tp + fn) if (tp + fn) else float("nan")
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) and precision == precision and recall == recall and (precision + recall) > 0
                  else float("nan"))
            rows.append({
                "model": model, "model_label": MODEL_LABEL[model], "class_name": cls,
                "group": "Positive (present)" if cls in POSITIVE_CLASSES else "Negative (absent)",
                "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            })
    return pd.DataFrame(rows)


metrics = build_metrics_df()
macro = (
    metrics.groupby(["model", "model_label", "group"])["f1"].mean().reset_index()
    .pivot(index=["model", "model_label"], columns="group", values="f1").reset_index()
)
macro["overall"] = metrics.groupby("model")["f1"].mean().reindex(macro["model"]).values
macro = macro.set_index("model").loc[MODEL_ORDER].reset_index()

alt.themes.enable("none")
CHART_FONT = "IBM Plex Sans, sans-serif"


def _base_config(chart):
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(labelFont=CHART_FONT, titleFont=CHART_FONT, labelColor=MUTED, titleColor=INK,
                         grid=False, domainColor=FAINT, tickColor=FAINT, labelFontSize=11.5, titleFontSize=11.5)
        .configure_legend(labelFont=CHART_FONT, titleFont=CHART_FONT, labelColor=INK, titleColor=INK,
                           labelFontSize=11.5, titleFontSize=11.5, orient="top", symbolType="square")
        .configure_text(font=CHART_FONT)
    )


# ---------------------------------------------------------------------------
# stat tiles
# ---------------------------------------------------------------------------

yolo_overall = macro.loc[macro["model"] == "yolo", "overall"].iloc[0]
gemini_overall = macro.loc[macro["model"] == "gemini", "overall"].iloc[0]
yolo_neg = macro.loc[macro["model"] == "yolo", "Negative (absent)"].iloc[0]
gemini_neg = macro.loc[macro["model"] == "gemini", "Negative (absent)"].iloc[0]
gap_x = yolo_neg / gemini_neg

st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:22px">
  <div style="background:#141414;color:#FFFFFF;padding:16px 20px 14px">
    <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#9B9D97">YOLO26 (OURS) — OVERALL F1</div>
    <div class="hv-h1" style="font-size:48px;line-height:1;color:#FFFFFF">{yolo_overall:.2f}</div>
    <div style="font-size:12px;color:#9B9D97">macro-averaged across all 9 classes</div>
  </div>
  <div style="background:#FFFFFF;color:#141414;border:1px solid #C4C6C0;padding:16px 20px 14px">
    <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#71736D">BEST VLM (GEMINI) — OVERALL F1</div>
    <div class="hv-h1" style="font-size:48px;line-height:1">{gemini_overall:.2f}</div>
    <div style="font-size:12px;color:#71736D">strongest of 5 general-purpose VLMs tried</div>
  </div>
  <div style="background:#EFE600;color:#141414;padding:16px 20px 14px">
    <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#3A3B30">GAP ON ABSENCE DETECTION</div>
    <div class="hv-h1" style="font-size:48px;line-height:1">{gap_x:.1f}×</div>
    <div style="font-size:12px;color:#3A3B30">YOLO's negative-class F1 ({yolo_neg:.2f}) vs. Gemini's ({gemini_neg:.2f})</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div style="font-size:12.5px;color:#4A4B47;margin:-8px 0 20px">100 held-out test images, seed 42, the identical '
    'presence/absence prompt sent to every model — this is a read-only snapshot of a scored comparison run, not live '
    'inference. Full write-up: <code>reports/llm_vs_yolo_comparison.md</code>.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# grouped bar — macro F1 by model, positive vs. negative classes
# ---------------------------------------------------------------------------

st.markdown('<div class="hv-h1" style="font-size:22px;margin-bottom:2px">MACRO F1 BY MODEL</div>', unsafe_allow_html=True)
st.caption("Negative classes (no-helmet, no-boots, …) are where every VLM falls apart — YOLO barely notices the difference.")

bar_df = macro.melt(id_vars=["model", "model_label"], value_vars=["Positive (present)", "Negative (absent)"],
                     var_name="group", value_name="f1")
MODEL_LABEL_ORDER = [MODEL_LABEL[m] for m in MODEL_ORDER]

bar = (
    alt.Chart()
    .mark_bar(size=16, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
    .encode(
        x=alt.X("group:N", title=None, axis=None),
        y=alt.Y("f1:Q", title="Macro F1", scale=alt.Scale(domain=[0, 1.02])),
        color=alt.Color("group:N", title=None,
                         scale=alt.Scale(domain=["Positive (present)", "Negative (absent)"], range=[INK, NEGATIVE_RED])),
        tooltip=[alt.Tooltip("model_label:N", title="Model"), alt.Tooltip("group:N", title="Class group"),
                 alt.Tooltip("f1:Q", title="Macro F1", format=".3f")],
    )
)
text = (
    alt.Chart()
    .mark_text(dy=-6, font=CHART_FONT, fontSize=10.5, color=INK)
    .encode(x=alt.X("group:N", axis=None), y=alt.Y("f1:Q"), text=alt.Text("f1:Q", format=".2f"))
)
grouped_bar = (
    alt.layer(bar, text, data=bar_df)
    .properties(width=70, height=220)
    .facet(column=alt.Column("model_label:N", title=None, sort=MODEL_LABEL_ORDER,
                              header=alt.Header(labelFont=CHART_FONT, labelFontSize=12.5, labelColor=INK, labelOrient="bottom")))
)
st.altair_chart(_base_config(grouped_bar), width="stretch")

# ---------------------------------------------------------------------------
# heatmap — full per-class F1, every model
# ---------------------------------------------------------------------------

st.markdown('<div class="hv-h1" style="font-size:22px;margin:26px 0 2px">PER-CLASS F1 — EVERY MODEL, EVERY CLASS</div>',
            unsafe_allow_html=True)
st.caption("Darker = higher F1. Reads left→right as PPE-present classes, then the four absence classes.")

heat_df = metrics.copy()
heat_df["f1_display"] = heat_df["f1"].apply(lambda v: "—" if v != v else f"{v:.2f}")

cells = (
    alt.Chart(heat_df)
    .mark_rect(stroke="#E4E5E2", strokeWidth=2)
    .encode(
        x=alt.X("class_name:N", title=None, sort=CLASS_ORDER,
                axis=alt.Axis(labelAngle=-40, labelFontSize=11)),
        y=alt.Y("model_label:N", title=None, sort=MODEL_LABEL_ORDER),
        color=alt.Color("f1:Q", title="F1", scale=alt.Scale(scheme="greys", domain=[0, 1]),
                         legend=alt.Legend(orient="right", gradientLength=140)),
        tooltip=[alt.Tooltip("model_label:N", title="Model"), alt.Tooltip("class_name:N", title="Class"),
                 alt.Tooltip("f1:Q", title="F1", format=".3f"), alt.Tooltip("tp:Q", title="tp"),
                 alt.Tooltip("fp:Q", title="fp"), alt.Tooltip("fn:Q", title="fn")],
    )
    .properties(height=230)
)
labels = (
    alt.Chart(heat_df)
    .mark_text(font=CHART_FONT, fontSize=11)
    .encode(
        x=alt.X("class_name:N", sort=CLASS_ORDER), y=alt.Y("model_label:N", sort=MODEL_LABEL_ORDER),
        text="f1_display:N",
        color=alt.condition(alt.datum.f1 > 0.55, alt.value("#FFFFFF"), alt.value(INK)),
    )
    .properties(height=230)
)
st.altair_chart(_base_config((cells + labels)), width="stretch")

# ---------------------------------------------------------------------------
# the ground-truth "person" gap, confirmed
# ---------------------------------------------------------------------------

st.markdown('<div class="hv-h1" style="font-size:22px;margin:26px 0 8px">WHY "PERSON" LOOKS WORSE THAN IT IS</div>',
            unsafe_allow_html=True)

gt_col1, gt_col2 = st.columns([3, 2])
with gt_col1:
    gt_df = pd.DataFrame([
        {"label": "Has a person box", "count": 49},
        {"label": "PPE boxed, but no person box", "count": 51},
    ])
    gt_bar = (
        alt.Chart(gt_df)
        .mark_bar(size=44, cornerRadiusTopRight=2, cornerRadiusBottomRight=2)
        .encode(
            y=alt.Y("label:N", title=None, sort=["Has a person box", "PPE boxed, but no person box"]),
            x=alt.X("count:Q", title="of 100 sampled test images", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("label:N", scale=alt.Scale(domain=["Has a person box", "PPE boxed, but no person box"],
                                                         range=[INK, FAINT]), legend=None),
            tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("count:Q", title="images")],
        )
        .properties(height=110)
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
      All five VLMs report the identical <b>51 false positives</b> on <code>person</code> — every time, across
      completely different model families. That's not five coincidental mistakes; it's a labeling gap in the
      test set itself. Every VLM correctly says <i>"yes, there's a person"</i> and gets marked wrong by an
      incomplete ground-truth box.
      </div>
      <div style="font-size:11.5px;color:#71736D;margin-top:10px">YOLO scores 49/49 here only because it was
      <i>trained</i> on this same gapped label set — consistency with the labels, not a stronger read on the photo.</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# descriptive vs. structured prompt — qualitative, reads from disk if present
# ---------------------------------------------------------------------------

st.markdown('<div class="hv-h1" style="font-size:22px;margin:26px 0 2px">ONE MORE THING GEMINI CAN DO: WRITE THE RECORD</div>',
            unsafe_allow_html=True)
st.caption(
    "Unscored — there's no ground truth for prose. Same images, two prompts: a free-text \"site record\" "
    "description, and the strict-JSON prompt everything above is scored on."
)

candidates = sorted(glob.glob(str(REPO_ROOT / "runs" / "llm" / "*" / "prompt_comparison.csv")))
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

st.caption(
    "Source data: reports/llm_vs_yolo_comparison.md · runs/llm/20260831_merged_n100_seed42_yolo-gemini/ · "
    "runs/llm/20260828_035813_merged_n100_seed42_yolo-ollama-qwen3-vl-gemma4-minicpm-v/"
)
