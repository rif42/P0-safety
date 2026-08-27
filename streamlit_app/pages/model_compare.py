"""HI-VIS — side-by-side comparison of the trained runs. Deliberately minimal
for now: no upload box of its own and no detail view (tiles aren't
clickable) — it reuses whatever batch is already loaded on the Demo page and
runs every model that CAN produce a compliance verdict over it, showing the
three core stat tiles (Photos Processed / PPE Exceptions Flagged / Image
Compliance) plus the actual photos with their overlays, verdict and
confidence, per model, side by side, so the runs can be judged on identical
input.

One model — yolo26s_Altec_PPE_100e — has no Person class in its training
data at all (confirmed via its confusion matrix), so it structurally cannot
produce a per-person compliance verdict. Per policy this app never fakes a
person box to paper over that, so it gets its own card explaining why
instead of a live result. See detector.ALTEC_NO_PERSON_NOTE.
"""

import streamlit as st

import detector
import view_helpers as vh

st.markdown(vh.HV_STYLE_CSS, unsafe_allow_html=True)
st.markdown(vh.header_html("MODEL COMPARISON"), unsafe_allow_html=True)

# Same batch cache the Demo page fills — shared via st.session_state, so
# whatever's already uploaded there is what gets compared here.
cache = st.session_state.get("_detections", {})
batch_order = st.session_state.get("batch_order", [])
items = [{"key": k, **cache[k]} for k in batch_order if k in cache]

if not items:
    st.info(
        "No photos loaded yet. Upload a batch on the **Demo** page first — this page "
        "runs every model that can produce a compliance verdict over whatever batch is "
        "currently loaded, so there's nothing to compare until one exists."
    )
    st.stop()

threshold = st.session_state.get("threshold", 0.35)
required = tuple(
    s for s, dflt in (
        ("hardhat", True), ("vest", True), ("mask", True), ("gloves", False), ("boots", False),
    )
    if st.session_state.get(f"require_{s}", dflt)
)

st.markdown(
    f'<div style="font-size:12.5px;color:#4A4B47;margin-bottom:18px">Comparing runs on the current '
    f'batch of <b>{len(items)}</b> photo{"s" if len(items) != 1 else ""}, at threshold '
    f'<b>{threshold:.2f}</b>, using the same WHAT COUNTS AS COMPLIANT rule set as the Demo page. '
    f'Change either on the Demo page to see this update here.</div>',
    unsafe_allow_html=True,
)

MODELS = [
    {"kind": "live", "key": "v8", "weights": detector.V8_WEIGHTS, "label": detector.V8_LABEL},
    {"kind": "live", "key": "v26", "weights": detector.V26_WEIGHTS, "label": detector.V26_LABEL},
    {"kind": "live", "key": "merged", "weights": detector.MERGED_WEIGHTS, "label": detector.MERGED_LABEL},
    {"kind": "no_person", "key": "altec", "weights": detector.ALTEC_WEIGHTS, "label": detector.ALTEC_LABEL,
     "note": detector.ALTEC_NO_PERSON_NOTE},
]

# Raw detections per model are cached here (keyed by model + photo) so
# flipping the threshold slider on the Demo page — which reruns this page
# too — never re-runs inference, only the cheap pure-python assess() step.
st.session_state.setdefault("_compare_raw", {})


def render_live(col, model_key, weights_path, label):
    with col:
        st.markdown(f'<div class="hv-h1" style="font-size:20px;margin-bottom:10px">{label}</div>',
                    unsafe_allow_html=True)

        model = detector.load_model(weights_path)
        if model is None:
            st.warning(f"Weights not found at `{weights_path}`.")
            return

        raw_cache = st.session_state._compare_raw.setdefault(model_key, {})
        for it in items:
            if it["key"] not in raw_cache:
                raw_cache[it["key"]] = detector.detect_raw(model, it["image"])

        results = [detector.assess(raw_cache[it["key"]], threshold, required=required) for it in items]
        assessed = [r for r in results if r["verdict"] != "none"]
        non_items = [r for r in results if r["verdict"] == "non"]
        rate = round((len(assessed) - len(non_items)) / len(assessed) * 100) if assessed else 0
        exc_bg = "#EFE600" if non_items else "#FFFFFF"

        st.markdown(f"""
        <div style="display:flex;flex-direction:column;gap:12px">
          <div style="background:#141414;color:#FFFFFF;padding:16px 20px 14px">
            <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#9B9D97">PHOTOS PROCESSED</div>
            <div class="hv-h1" style="font-size:44px;line-height:1;color:#FFFFFF">{len(items)}</div>
            <div style="font-size:12px;color:#9B9D97">{len(assessed)} assessed · {len(items) - len(assessed)} no person detected</div>
          </div>
          <div style="background:{exc_bg};color:#141414;border:1px solid #141414;padding:16px 20px 14px">
            <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#3A3B30">PPE EXCEPTIONS FLAGGED</div>
            <div class="hv-h1" style="font-size:44px;line-height:1">{len(non_items)}</div>
            <div style="font-size:12px;color:#3A3B30">{"photos with at least one finding" if non_items else "none at this threshold"}</div>
          </div>
          <div style="background:#FFFFFF;color:#141414;border:1px solid #C4C6C0;padding:16px 20px 14px">
            <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#71736D">IMAGE COMPLIANCE</div>
            <div class="hv-h1" style="font-size:44px;line-height:1">{rate}%</div>
            <div style="font-size:12px;color:#71736D">{len(assessed)} of {len(items)} photos assessed at this threshold</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="hv-h1" style="font-size:15px;margin:18px 0 8px">PHOTOS</div>', unsafe_allow_html=True)
        photo_cols = st.columns(2)
        for i, (it, r) in enumerate(zip(items, results)):
            with photo_cols[i % 2]:
                thumb_b64 = vh.b64_image(vh.draw_overlay(it["image"], r["persons"], show_boxes=True), max_dim=360)
                fc = vh.flag_confidence(r, required)
                conf_label = f"{fc:.2f} conf" if fc is not None else "— conf"
                st.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #C4C6C0;margin-bottom:10px" title="{it['name']}">
                  <img src="data:image/jpeg;base64,{thumb_b64}" style="width:100%;aspect-ratio:4/3;object-fit:cover;display:block"/>
                  <div style="display:flex;align-items:center;gap:8px;padding:6px 8px">
                    {vh.verdict_badge(r["verdict"])}
                    <span class="hv-mono" style="font-size:10.5px;color:#4A4B47;white-space:nowrap">{conf_label}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)


def render_no_person(col, label, note):
    with col:
        st.markdown(f'<div class="hv-h1" style="font-size:20px;margin-bottom:10px">{label}</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#E4E5E2;border:2px dashed #9B9D97;padding:20px;display:flex;flex-direction:column;gap:10px">
          <div class="hv-h1" style="font-size:32px;color:#71736D">N/A</div>
          <div style="font-size:12.5px;color:#4A4B47;line-height:1.5">{note}</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Training metrics for this run are on the Model Performance page.")


# 2×2 grid — two models per row, so each column stays wide enough for its
# photo grid to read comfortably.
for row_start in range(0, len(MODELS), 2):
    row_cols = st.columns(2)
    for col, m in zip(row_cols, MODELS[row_start:row_start + 2]):
        if m["kind"] == "live":
            render_live(col, m["key"], m["weights"], m["label"])
        else:
            render_no_person(col, m["label"], m["note"])
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

st.caption(
    "Tiles aren't clickable yet — no detail view or drop box of its own on this page for now. "
    "It's read-only against whatever's loaded on the Demo page."
)
