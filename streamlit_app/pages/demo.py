"""HI-VIS — PPE compliance detection demo (real YOLO inference).

Upload site photos, run them through the trained detector (see detector.py —
swap models there, no edits needed here), and review compliance: a results
gallery, a filterable exception log with CSV export, and a per-photo detail
view with detection overlays. Visual language follows the HI-VIS design spec
(Aug 2026): Barlow Condensed / IBM Plex, black + safety-yellow, dense mono
labels.
"""

import base64
import hashlib
import io

import streamlit as st

import detector
import view_helpers as vh

# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "threshold": 0.35,
    "view": "results",
    "detail_key": None,
    "selected_person": None,
    "show_boxes": True,
    "gallery_filter": "All photos",
    "gallery_limit": 12,
    "type_filter": "All exception types",
    "verdict_filter": "Non-compliant only",
    "row_limit": 25,
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("_detections", {})

TYPE_FILTER_MAP = {
    "All exception types": "all",
    "Hardhat — absence detected": "hardhat",
    "Hi-vis vest — absence detected": "vest",
}
VERDICT_FILTER_MAP = {
    "Non-compliant only": "non-compliant",
    "Compliant only": "compliant",
    "All assessed persons": "all",
}
RULE_TEXT = "Hardhat and hi-vis vest required; boots advisory (not tracked by this model)."

# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; }
.stApp { background:#E4E5E2; }
#MainMenu, footer { visibility:hidden; }
.block-container { padding-top:1.5rem; max-width:1400px; }
.hv-mono { font-family:'IBM Plex Mono',monospace; }
.hv-h1 { font-family:'Barlow Condensed',sans-serif; font-weight:800; letter-spacing:.5px; color:#141414; }
div[data-testid="stFileUploaderDropzone"] { background:#FFFFFF; border:2px dashed #141414 !important; border-radius:0; }
div[role="radiogroup"] label { border:1px solid #141414; padding:4px 12px; margin-right:0 !important; background:#FFFFFF; }
.stButton>button { border-radius:0; border:1px solid #141414; font-weight:600; }
.stDownloadButton>button { border-radius:0; background:#141414; color:#FFFFFF; border:1px solid #141414; font-weight:600; }
hr { border-color:#C4C6C0; }
</style>
""", unsafe_allow_html=True)


def icon_svg(kind, color, size=12, stroke=3):
    paths = {
        "check": '<path d="M4 12.5l5 5L20 6.5"/>',
        "cross": '<path d="M5 5l14 14M19 5L5 19"/>',
        "warn": '<path d="M12 3L2 21h20L12 3z"/><path d="M12 10v5"/>',
    }
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke}" style="vertical-align:-2px;flex:none">'
            f'{paths.get(kind, "")}</svg>')


def verdict_badge(verdict):
    meta = vh.badge_for(verdict)
    kind = "check" if verdict == "ok" else "warn" if verdict == "non" else "cross"
    icon = icon_svg(kind, meta["fg"], 11, 2.6)
    return (f'<span style="display:inline-flex;align-items:center;gap:5px;font-size:10.5px;'
            f'font-weight:700;letter-spacing:.5px;padding:2px 7px;background:{meta["bg"]};'
            f'color:{meta["fg"]};border:1px solid {meta["border"]};white-space:nowrap">'
            f'{icon}{meta["label"]}</span>')


def b64_image(img, max_dim=480, quality=82):
    im = img.copy()
    im.thumbnail((max_dim, max_dim))
    if im.mode != "RGB":
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------

st.markdown(f"""
<div style="background:#141414;color:#FFFFFF;display:flex;align-items:center;gap:16px;
     padding:14px 24px;margin:-1rem 0 20px 0;flex-wrap:wrap">
  <div style="background:#EFE600;color:#141414;font-family:'Barlow Condensed',sans-serif;
       font-weight:800;font-size:24px;letter-spacing:1px;padding:2px 10px 4px;line-height:1">HI-VIS</div>
  <div style="font-family:'Barlow Condensed',sans-serif;font-weight:600;font-size:15px;
       letter-spacing:2.5px">PPE COMPLIANCE DETECTION</div>
  <div class="hv-mono" style="font-size:11px;color:#8D8F8A;border-left:1px solid #3A3B38;
       padding-left:14px">{detector.MODEL_LABEL}</div>
</div>
""", unsafe_allow_html=True)

model = detector.load_model()
if model is None:
    st.error(
        f"No trained weights found at `{detector.WEIGHTS_PATH}`.\n\n"
        "This file is git-ignored (weights aren't committed) — restore it from wherever "
        "training happened (Colab/Kaggle download, teammate's copy, shared drive), or point "
        "`HIVIS_MODEL_PATH` at a `.pt` file you do have, e.g.:\n\n"
        "`HIVIS_MODEL_PATH=runs/scratch/yolov8n_scratch/weights/best.pt streamlit run app.py`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------

with st.container(border=True):
    st.markdown('<div class="hv-h1" style="font-size:26px">DROP SITE PHOTOS HERE</div>', unsafe_allow_html=True)
    st.caption("Multiple images or a whole folder. Each photo is checked for hardhats and hi-vis vests, person by person.")
    uploaded_files = st.file_uploader(
        "Select photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
        label_visibility="collapsed",
    )

if not uploaded_files:
    st.info("Waiting for photos to analyse. Results, the exception log and CSV export appear here once you upload a batch.")
    st.stop()

# ---------------------------------------------------------------------------
# process uploads (cached: inference runs once per unique file, not per rerun)
# ---------------------------------------------------------------------------

cache = st.session_state._detections
items = []
new_count = 0
for f in uploaded_files:
    raw_bytes = f.getvalue()
    key = hashlib.md5(raw_bytes).hexdigest()
    if key not in cache:
        new_count += 1
        img = vh.load_image(raw_bytes)
        raw = detector.detect_raw(model, img)
        dt = vh.exif_datetime(raw_bytes)
        cache[key] = {"name": f.name, "image": img, "raw": raw, "datetime": dt}
    items.append({"key": key, **cache[key]})

if new_count:
    st.toast(f"Analysed {new_count} new photo{'s' if new_count != 1 else ''}.")

threshold = st.session_state.threshold
for it in items:
    it["assessment"] = detector.assess(it["raw"], threshold)


def go(**state):
    for k, v in state.items():
        st.session_state[k] = v
    st.rerun()


# ---------------------------------------------------------------------------
# results view
# ---------------------------------------------------------------------------

if st.session_state.view == "results":
    assessed = [it for it in items if it["assessment"]["verdict"] != "none"]
    non_items = [it for it in items if it["assessment"]["verdict"] == "non"]
    rate = round((len(assessed) - len(non_items)) / len(assessed) * 100) if assessed else 0
    exc_bg = "#EFE600" if non_items else "#FFFFFF"

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:18px">
      <div style="background:#141414;color:#FFFFFF;padding:16px 20px 14px">
        <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#9B9D97">PHOTOS PROCESSED</div>
        <div class="hv-h1" style="font-size:48px;line-height:1;color:#FFFFFF">{len(items)}</div>
        <div style="font-size:12px;color:#9B9D97">{len(assessed)} assessed · {len(items) - len(assessed)} no person detected</div>
      </div>
      <div style="background:{exc_bg};color:#141414;border:1px solid #141414;padding:16px 20px 14px">
        <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#3A3B30">PPE EXCEPTIONS FLAGGED</div>
        <div class="hv-h1" style="font-size:48px;line-height:1">{len(non_items)}</div>
        <div style="font-size:12px;color:#3A3B30">{"photos with at least one finding" if non_items else "none at this threshold"}</div>
      </div>
      <div style="background:#FFFFFF;color:#141414;border:1px solid #C4C6C0;padding:16px 20px 14px">
        <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#71736D">IMAGE COMPLIANCE</div>
        <div class="hv-h1" style="font-size:48px;line-height:1">{rate}%</div>
        <div style="font-size:12px;color:#71736D">{len(assessed)} of {len(items)} photos assessed at this threshold</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown('<div class="hv-h1" style="font-size:18px">DETECTION CONFIDENCE THRESHOLD</div>', unsafe_allow_html=True)
            st.slider("Threshold", min_value=0.10, max_value=0.90, step=0.01, key="threshold", label_visibility="collapsed")
        with c2:
            st.markdown(
                f'<div style="font-size:12.5px;color:#4A4B47;padding-top:28px">Detections below this confidence are ignored. '
                f'Lower catches more missing PPE but raises false alarms; higher reports only what the model is sure of. '
                f'Results update live — currently <b>{len(non_items)} exception{"s" if len(non_items) != 1 else ""}</b> at {threshold:.2f}.</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="hv-h1" style="font-size:24px;margin-top:24px">RESULTS</div>', unsafe_allow_html=True)
    gcol1, gcol2 = st.columns([3, 1])
    with gcol1:
        st.radio("Gallery filter", ["All photos", "Exceptions only"], key="gallery_filter",
                  horizontal=True, label_visibility="collapsed")
    filtered = items if st.session_state.gallery_filter == "All photos" else non_items

    if not filtered:
        st.markdown('<div style="background:#FFFFFF;border:1px solid #C4C6C0;padding:36px;text-align:center;'
                     'color:#71736D;font-size:13px">No photos in this batch yet.</div>', unsafe_allow_html=True)
    else:
        shown = filtered[:st.session_state.gallery_limit]
        cols = st.columns(4)
        for i, it in enumerate(shown):
            a = it["assessment"]
            with cols[i % 4]:
                b64 = b64_image(vh.draw_overlay(it["image"], a["persons"], show_boxes=True))
                st.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #C4C6C0">
                  <img src="data:image/jpeg;base64,{b64}" style="width:100%;aspect-ratio:4/3;object-fit:cover;display:block"/>
                  <div style="display:flex;align-items:center;gap:8px;padding:7px 9px">
                    {verdict_badge(a["verdict"])}
                    <span class="hv-mono" style="font-size:11px;color:#4A4B47;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{it["name"]}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Open →", key=f"open_{it['key']}", width="stretch"):
                    go(view="detail", detail_key=it["key"], selected_person=None)
        if len(filtered) > st.session_state.gallery_limit:
            if st.button(f"Show more photos ({len(filtered) - st.session_state.gallery_limit} remaining)"):
                st.session_state.gallery_limit += 12
                st.rerun()

    st.markdown('<div class="hv-h1" style="font-size:24px;margin-top:28px">EXCEPTION LOG</div>', unsafe_allow_html=True)
    fcol1, fcol2, fcol3 = st.columns([1, 1, 1])
    with fcol1:
        st.selectbox("Exception type", list(TYPE_FILTER_MAP), key="type_filter", label_visibility="collapsed")
    with fcol2:
        st.selectbox("Verdict", list(VERDICT_FILTER_MAP), key="verdict_filter", label_visibility="collapsed")

    all_rows = vh.build_rows(
        [{"name": it["name"], "datetime": it["datetime"], "assessment": it["assessment"]} for it in items],
        threshold, RULE_TEXT,
    )
    type_key = TYPE_FILTER_MAP[st.session_state.type_filter]
    verdict_key = VERDICT_FILTER_MAP[st.session_state.verdict_filter]
    rows = all_rows
    if type_key != "all":
        rows = [r for r in rows if r["type"] == type_key]
    if verdict_key != "all":
        rows = [r for r in rows if r["verdict"] == verdict_key]

    with fcol3:
        st.download_button(
            "⬇ Export CSV", vh.rows_to_csv(rows),
            file_name=f"hivis-exception-log-t{threshold:.2f}.csv", mime="text/csv",
            width="stretch",
        )
    st.caption("Export includes filename, timestamp, person, finding, confidence, verdict, threshold and the rule set applied.")

    if not rows:
        st.markdown('<div style="background:#FFFFFF;border:1px solid #C4C6C0;padding:28px;text-align:center;'
                     'color:#71736D;font-size:13px">No rows match these filters.</div>', unsafe_allow_html=True)
    else:
        shown_rows = rows[:st.session_state.row_limit]
        head = ("<tr style='background:#141414;color:#FFFFFF;text-align:left'>"
                "<th style='padding:8px 12px;font-size:10.5px' class='hv-mono'>FILE</th>"
                "<th style='padding:8px 12px;font-size:10.5px' class='hv-mono'>WHEN</th>"
                "<th style='padding:8px 12px;font-size:10.5px' class='hv-mono'>PERSON</th>"
                "<th style='padding:8px 12px;font-size:10.5px' class='hv-mono'>FINDING</th>"
                "<th style='padding:8px 12px;font-size:10.5px;text-align:right' class='hv-mono'>CONF</th></tr>")
        body = ""
        for r in shown_rows:
            fg = "#B02A20" if r["verdict"] == "non-compliant" else "#1B7A3D"
            conf = f'{r["confidence"]:.2f}' if r["confidence"] is not None else "—"
            body += (f"<tr style='border-top:1px solid #E4E5E2'>"
                     f"<td class='hv-mono' style='padding:7px 12px;white-space:nowrap'>{r['file']}</td>"
                     f"<td style='padding:7px 12px;white-space:nowrap'>{r['datetime']}</td>"
                     f"<td style='padding:7px 12px'>{r['person']}</td>"
                     f"<td style='padding:7px 12px;font-weight:600;color:{fg}'>{r['finding']}</td>"
                     f"<td class='hv-mono' style='padding:7px 12px;text-align:right'>{conf}</td></tr>")
        st.markdown(f"<div style='background:#FFFFFF;border:1px solid #C4C6C0;overflow-x:auto'>"
                     f"<table style='width:100%;border-collapse:collapse;font-size:12.5px'>{head}{body}</table></div>",
                     unsafe_allow_html=True)
        if len(rows) > st.session_state.row_limit:
            if st.button(f"Show more rows ({len(rows) - st.session_state.row_limit} remaining)"):
                st.session_state.row_limit += 25
                st.rerun()

    st.caption(f"Rule set: **{RULE_TEXT}** A person is non-compliant when the model makes a positive absence "
               f"finding (NO-Hardhat / NO-Safety Vest) at or above the threshold. {detector.UNTRACKED_NOTE}")

# ---------------------------------------------------------------------------
# detail view
# ---------------------------------------------------------------------------

else:
    idx_by_key = {it["key"]: i for i, it in enumerate(items)}
    if st.session_state.detail_key not in idx_by_key:
        go(view="results")
    idx = idx_by_key[st.session_state.detail_key]
    it = items[idx]
    a = it["assessment"]

    top1, top2, top3 = st.columns([1, 4, 2])
    with top1:
        if st.button("← RESULTS"):
            go(view="results")
    with top2:
        st.markdown(f'<div class="hv-h1" style="font-size:24px">{it["name"]} '
                     f'<span class="hv-mono" style="font-size:13px;color:#4A4B47;font-weight:400">'
                     f'· {it["datetime"] or "no capture time"}</span> {verdict_badge(a["verdict"])}</div>',
                     unsafe_allow_html=True)
    with top3:
        n1, n2, n3 = st.columns(3)
        if n1.button("◀ PREV"):
            go(view="detail", detail_key=items[(idx - 1) % len(items)]["key"], selected_person=None)
        if n2.button("NEXT ▶"):
            go(view="detail", detail_key=items[(idx + 1) % len(items)]["key"], selected_person=None)
        if n3.button("⚠ NEXT EXC."):
            found = next((items[(idx + k) % len(items)] for k in range(1, len(items) + 1)
                          if items[(idx + k) % len(items)]["assessment"]["verdict"] == "non"), None)
            if found:
                go(view="detail", detail_key=found["key"], selected_person=None)
            else:
                st.toast("No exceptions in the current batch.")

    left, right = st.columns([3, 2])
    with left:
        st.session_state.show_boxes = st.checkbox("Show detection boxes", value=st.session_state.show_boxes)
        big = vh.draw_overlay(it["image"], a["persons"], selected_idx=st.session_state.selected_person,
                               show_boxes=st.session_state.show_boxes)
        b64 = b64_image(big, max_dim=1400, quality=90)
        st.markdown(f'<div style="background:#141414;padding:10px"><img src="data:image/jpeg;base64,{b64}" '
                     f'style="width:100%;display:block"/></div>', unsafe_allow_html=True)

        if a["persons"]:
            st.caption("Isolate:")
            pc = st.columns(len(a["persons"]) + 1)
            if pc[0].button("All persons", type="primary" if st.session_state.selected_person is None else "secondary"):
                st.session_state.selected_person = None
                st.rerun()
            for pi in range(len(a["persons"])):
                if pc[pi + 1].button(f"Person {pi + 1}",
                                      type="primary" if st.session_state.selected_person == pi else "secondary"):
                    st.session_state.selected_person = pi
                    st.rerun()

        legend = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px;white-space:nowrap">'
            f'<span style="width:12px;height:12px;border:2px solid {meta["color"]};display:inline-block"></span>{meta["label"]}</span>'
            for meta in detector.CLASS_META.values()
        )
        st.markdown(f'<div style="background:#FFFFFF;border:1px solid #C4C6C0;padding:8px 14px;font-size:11.5px;'
                     f'margin-top:10px"><span class="hv-mono" style="font-size:10px;color:#71736D;margin-right:10px">'
                     f'CLASS COLOURS</span>{legend}</div>', unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div style="background:#141414;color:#FFFFFF;padding:12px 16px;margin-bottom:10px">
          <div class="hv-mono" style="font-size:10px;letter-spacing:1.5px;color:#9B9D97">RULE SET APPLIED</div>
          <div class="hv-h1" style="font-size:17px;color:#FFFFFF">Hardhat and hi-vis vest required. Boots advisory.</div>
          <div style="font-size:11.5px;color:#B9BBB4;margin-top:6px">Non-compliant = a positive absence finding at or above threshold {threshold:.2f}. {detector.UNTRACKED_NOTE}</div>
        </div>
        """, unsafe_allow_html=True)

        if not a["persons"]:
            note = " (detections exist below the current threshold — try lowering it)" if a.get("recoverable") else ""
            st.markdown(f'<div style="background:#FFFFFF;border:2px solid #141414;padding:16px;">'
                         f'<div class="hv-h1" style="font-size:20px">{icon_svg("warn", "#141414", 20, 2)} NO PERSON DETECTIONS</div>'
                         f'<div style="font-size:13px;margin-top:8px">No Person boxes returned for this photo{note}. '
                         f'Excluded from the compliance rate — review manually.</div></div>', unsafe_allow_html=True)
        for pi, p in enumerate(a["persons"]):
            border = "2px solid #141414" if st.session_state.selected_person == pi else "1px solid #C4C6C0"
            rows_html = ""
            for slot in ("hardhat", "vest", "boots"):
                st_ = p["status"][slot]
                if st_["state"] == "notvisible":
                    label, fg = "not visible", "#9B9D97"
                elif st_["state"] == "present":
                    label, fg = "present", "#1B7A3D"
                else:
                    label, fg = "missing", "#B02A20"
                conf = f'{st_["conf"]:.2f}' if st_["conf"] is not None else "—"
                rows_html += (f"<div style='display:grid;grid-template-columns:80px 1fr 50px;gap:10px;"
                              f"padding:6px 14px;border-bottom:1px solid #F0F1EC;font-size:12.5px'>"
                              f"<div style='font-weight:600'>{slot.capitalize()}</div>"
                              f"<div style='color:{fg};font-weight:600'>{label}</div>"
                              f"<div class='hv-mono' style='text-align:right;color:{fg}'>{conf}</div></div>")
            st.markdown(f"""
            <div style="background:#FFFFFF;border:{border};margin-bottom:10px">
              <div style="display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-bottom:1px solid #E4E5E2">
                <span class="hv-h1" style="font-size:17px">Person {pi + 1} <span class="hv-mono" style="font-size:11px;color:#71736D;font-weight:400">conf {p['conf']:.2f}</span></span>
                {verdict_badge(p['verdict'])}
              </div>
              {rows_html}
            </div>
            """, unsafe_allow_html=True)
