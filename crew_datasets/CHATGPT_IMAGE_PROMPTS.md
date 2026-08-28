# ChatGPT image-generation prompts for Crew A / Crew B datasets

Generates the same two-crew CCTV story as `generate_synthetic_crew_data.py`
(see `CREW_A_STORY.md` / `CREW_B_STORY.md`), but via ChatGPT's image tool
instead of PIL — for a more photorealistic look. **Same events, same
trend, same numbers** — only the rendering method differs, so the story
stays consistent no matter which dataset you actually use in the pitch.

## How this has to work in practice

ChatGPT generates one image per turn, and it can't guarantee pixel-identical
backgrounds across separate generations from a text prompt alone. So this is
a **two-step, repeated workflow**, not a single prompt:

1. **Generate one base background once per camera** (Crew A's camera, Crew
   B's camera) — an empty site, no people.
2. **Re-upload that same background image** in every subsequent turn and ask
   ChatGPT to *edit* it (its image tool supports editing an uploaded/attached
   image, not just generating fresh ones) — add today's workers, keep the
   background/lighting/angle unchanged, update the timestamp watermark. This
   is what keeps the "same camera, different day" illusion intact.

Doing this for all 30 weekdays x 2 crews is 60 manual generations. If that's
too many, the **recommended shoot list** below picks 8-9 per crew — one per
week plus every event day — which is enough to tell the whole story on a
trend chart while staying manageable by hand.

**Say so wherever these are shown**: these are AI-generated mockups for a
pitch demo, not real security footage. Don't caption or present them as
genuine camera footage.

---

## Step 1 — Base background prompts (run once per crew)

### Crew A — CAM-A1 (industrial fabrication yard)

> A still frame from an elevated industrial security camera, mounted about
> 5 metres up on a warehouse wall, looking down at a slight angle over a
> construction/fabrication yard. Concrete floor with visible seams and
> stains, scattered structural steel beams and rebar stacked to one side, a
> small workbench with tools, a large yellow site vehicle parked in the
> background, chain-link fencing with a strip of sky visible at the top.
> No people in the shot. Grainy, slightly desaturated CCTV footage look,
> mild green-gray colour cast, visible compression artifacts, low dynamic
> range, security-camera lens distortion at the edges. Square or 4:3
> aspect ratio. Daytime, overcast lighting.

### Crew B — CAM-B1 (materials laydown / staging area)

> A still frame from an elevated industrial security camera, mounted about
> 5 metres up, looking down at a slight angle over a construction site
> materials laydown yard. Wet/damp concrete with visible puddles, stacked
> pallets of building materials, a portable site cabin/porta-cabin to one
> side, coiled hoses and cabling on the ground, a chain-link fence with
> site signage. No people in the shot. Grainy, slightly desaturated CCTV
> footage look, mild green-gray colour cast, visible compression artifacts,
> low dynamic range, security-camera lens distortion at the edges. Square
> or 4:3 aspect ratio. Overcast, slightly dim lighting (looks like it's
> been raining).

Save both outputs — you'll re-upload them for every frame of that crew.

---

## Step 2 — Reusable per-frame edit template

Attach the crew's saved background image, then send:

> Using this exact image as the background — same camera angle, same
> lighting, same objects, do not change anything about the scene itself —
> add {N} construction workers scattered naturally across the yard, each
> mid-task (walking, carrying material, talking in a small group, crouched
> working). Do not add more or fewer than {N} people. Render each worker's
> PPE exactly as specified, no exceptions:
>
> - Worker 1: hard hat {yes/no}, hi-vis vest {yes/no}, work gloves {yes/no}, safety boots {yes/no}
> - Worker 2: hard hat {yes/no}, hi-vis vest {yes/no}, work gloves {yes/no}, safety boots {yes/no}
> - Worker 3: hard hat {yes/no}, hi-vis vest {yes/no}, work gloves {yes/no}, safety boots {yes/no}
> - Worker 4: hard hat {yes/no}, hi-vis vest {yes/no}, work gloves {yes/no}, safety boots {yes/no}
> - Worker 5: hard hat {yes/no}, hi-vis vest {yes/no}, work gloves {yes/no}, safety boots {yes/no}
>
> A "no" on hard hat means bare-headed, not just no colour on it. A "no" on
> hi-vis vest means an ordinary plain shirt, not a dull-coloured vest. Keep
> the same grainy, desaturated CCTV look as the background.
>
> Overlay a security-camera timestamp watermark, top-left, blocky white
> digital font with a black outline/shadow, reading:
> **"{DD-MM-YYYY} {Day} {HH:MM:SS AM/PM}"**
> and a small camera-ID watermark, bottom-right, same style, reading:
> **"{CAMERA_ID}"**

Fill in `{N}`, the per-worker PPE list, the date/time, and `{CAMERA_ID}`
(`CAM-A1` or `CAM-B1`) from the tables below.

---

## Recommended shoot list — Crew A (CAM-A1), "the slow fade"

One frame per week plus both event days = 8 frames. Regular interval: every
frame at **10:30 AM**, starting Monday 2026-06-01.

| # | Date | Day-idx | Workers | Hard hat | Vest | Gloves | Boots | Note (tell the presenter this) |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-06-01 Mon | 0 | 5 | 5/5 | 5/5 | 5/5 | 5/5 | Baseline — strong start |
| 2 | 2026-06-08 Mon | 5 | 5 | 5/5 | 4/5 | 4/5 | 5/5 | Week 2 — first small slips |
| 3 | 2026-06-17 Wed | **12** | 5 | 4/5 | **2/5** | 3/5 | 4/5 | **Heatwave, peak day** — vests coming off in the heat |
| 4 | 2026-06-19 Fri | 14 | 5 | 4/5 | 4/5 | 4/5 | 4/5 | Heatwave over — vest bounces back |
| 5 | 2026-06-26 Fri | 19 | 5 | 4/5 | 4/5 | 4/5 | 4/5 | Week 4 — drift resumes |
| 6 | 2026-07-03 Fri | 24 | 5 | 4/5 | 3/5 | 4/5 | 4/5 | Week 5 — steady decline continues |
| 7 | 2026-07-06 Mon | **25** | **6** | **3/6** | **3/6** | **3/6** | **4/6** | **Apprentices join** — 2 new faces, everything dips one day |
| 8 | 2026-07-10 Fri | 29 | 5 | 4/5 | 3/5 | 3/5 | 4/5 | Week 6 end — new, lower normal |

**Example ready-to-run prompt (row 3, the heatwave peak):**

> Using this exact image as the background — same camera angle, same
> lighting, same objects, do not change anything about the scene itself —
> add 5 construction workers scattered naturally across the yard, each
> mid-task. Do not add more or fewer than 5 people. Render each worker's
> PPE exactly as specified:
> - Worker 1: hard hat yes, hi-vis vest no, work gloves no, safety boots yes
> - Worker 2: hard hat yes, hi-vis vest yes, work gloves yes, safety boots yes
> - Worker 3: hard hat yes, hi-vis vest no, work gloves yes, safety boots yes
> - Worker 4: hard hat no, hi-vis vest no, work gloves no, safety boots yes
> - Worker 5: hard hat yes, hi-vis vest no, work gloves yes, safety boots no
>
> One or two workers visibly wiping their brow / look overheated, to sell
> a heatwave. A "no" on hard hat means bare-headed. A "no" on hi-vis vest
> means an ordinary plain shirt. Keep the same grainy, desaturated CCTV
> look as the background.
>
> Overlay a timestamp watermark, top-left: "17-06-2026 Wed 10:30:00 AM".
> Overlay a camera-ID watermark, bottom-right: "CAM-A1".

---

## Recommended shoot list — Crew B (CAM-B1), "the turnaround"

Same cadence: one frame per week plus every event day = 9 frames, every
frame at **10:30 AM**.

| # | Date | Day-idx | Workers | Hard hat | Vest | Gloves | Boots | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-06-01 Mon | **0** | 5 | 2/5 | 3/5 | 2/5 | **2/5** | **Rainy week, day 1** — visible puddles, mud on boots |
| 2 | 2026-06-08 Mon | 5 | 5 | 2/5 | 3/5 | 2/5 | 3/5 | Week 2 — rain's gone, boots/gloves still poor |
| 3 | 2026-06-15 Mon | 10 | 5 | 2/5 | 3/5 | 2/5 | 3/5 | Week 3 day 1 — still flat, day before the near-miss |
| 4 | 2026-06-16 Tue | **11** | 5 | **4/5** | 3/5 | 2/5 | 3/5 | **Near-miss, next morning** — helmet compliance jumps, nothing else does |
| 5 | 2026-06-19 Fri | 14 | 5 | 4/5 | 3/5 | 2/5 | 3/5 | Near-miss effect holding through the week |
| 6 | 2026-06-22 Mon | **15** | 5 | **4/5** | **4/5** | **4/5** | **4/5** | **Toolbox talk, day 1** — every metric jumps at once, not just one |
| 7 | 2026-07-03 Fri | 24 | 5 | 4/5 | 4/5 | 4/5 | 4/5 | Week 5 — intervention holding |
| 8 | 2026-07-10 Fri | 29 | 5 | 5/5 | 5/5 | 4/5 | 5/5 | Week 6 end — new, higher normal |

**Example ready-to-run prompt (row 6, the toolbox talk):**

> Using this exact image as the background — same camera angle, same
> lighting, same objects, do not change anything about the scene itself —
> add 5 construction workers scattered naturally across the yard. Do not
> add more or fewer than 5 people. Render each worker's PPE exactly as
> specified:
> - Worker 1: hard hat yes, hi-vis vest yes, work gloves yes, safety boots yes
> - Worker 2: hard hat yes, hi-vis vest yes, work gloves yes, safety boots yes
> - Worker 3: hard hat yes, hi-vis vest yes, work gloves no, safety boots yes
> - Worker 4: hard hat yes, hi-vis vest no, work gloves yes, safety boots yes
> - Worker 5: hard hat no, hi-vis vest yes, work gloves yes, safety boots no
>
> Compose the scene as a small group gathered loosely around one person
> holding a clipboard, as if a safety briefing just wrapped up. A "no" on
> hard hat means bare-headed. A "no" on hi-vis vest means an ordinary plain
> shirt. Keep the same grainy, desaturated CCTV look as the background.
>
> Overlay a timestamp watermark, top-left: "22-06-2026 Mon 10:30:00 AM".
> Overlay a camera-ID watermark, bottom-right: "CAM-B1".

---

## If you want the full 30-day series instead of the 8-9 frame shoot list

Use the exact daily probabilities below (identical to
`generate_synthetic_crew_data.py`'s model) and convert each into a
worker-by-worker roster yourself: for N workers and a target probability p,
make `round(p × N)` of them compliant on that item and vary *which* workers
those are each day (don't always pick the same ones — real crews don't).

### Crew A daily targets (helmet / vest / gloves / boots)

| Day | Date | Helmet | Vest | Gloves | Boots | Event |
|---|---|---|---|---|---|---|
| 0 | 06-01 Mon | .95 | .95 | .93 | .96 | |
| 1 | 06-02 Tue | .94 | .94 | .92 | .95 | |
| 2 | 06-03 Wed | .93 | .93 | .91 | .95 | |
| 3 | 06-04 Thu | .93 | .92 | .91 | .94 | |
| 4 | 06-05 Fri | .92 | .91 | .90 | .94 | |
| 5 | 06-08 Mon | .91 | .90 | .89 | .93 | |
| 6 | 06-09 Tue | .90 | .89 | .88 | .93 | |
| 7 | 06-10 Wed | .89 | .88 | .87 | .92 | |
| 8 | 06-11 Thu | .89 | .88 | .87 | .92 | |
| 9 | 06-12 Fri | .88 | .87 | .86 | .91 | |
| 10 | 06-15 Mon | .87 | .51 | .70 | .90 | Heatwave starts |
| 11 | 06-16 Tue | .86 | .50 | .69 | .90 | Heatwave |
| 12 | 06-17 Wed | .85 | .49 | .68 | .89 | Heatwave peak |
| 13 | 06-18 Thu | .85 | .83 | .83 | .89 | Heatwave over |
| 14 | 06-19 Fri | .84 | .82 | .82 | .88 | |
| 15 | 06-22 Mon | .83 | .81 | .81 | .88 | |
| 16 | 06-23 Tue | .82 | .80 | .80 | .87 | |
| 17 | 06-24 Wed | .82 | .79 | .80 | .87 | |
| 18 | 06-25 Thu | .81 | .78 | .79 | .86 | |
| 19 | 06-26 Fri | .80 | .77 | .78 | .86 | |
| 20 | 06-29 Mon | .79 | .76 | .77 | .85 | |
| 21 | 06-30 Tue | .78 | .75 | .76 | .84 | |
| 22 | 07-01 Wed | .78 | .75 | .76 | .84 | |
| 23 | 07-02 Thu | .77 | .74 | .75 | .83 | |
| 24 | 07-03 Fri | .76 | .73 | .74 | .83 | |
| 25 | 07-06 Mon | .55 | .52 | .48 | .62 | Apprentices join (+1-2 workers this day only) |
| 26 | 07-07 Tue | .74 | .71 | .72 | .82 | |
| 27 | 07-08 Wed | .74 | .70 | .72 | .81 | |
| 28 | 07-09 Thu | .73 | .69 | .71 | .81 | |
| 29 | 07-10 Fri | .72 | .68 | .70 | .80 | |

### Crew B daily targets (helmet / vest / gloves / boots)

| Day | Date | Helmet | Vest | Gloves | Boots | Event |
|---|---|---|---|---|---|---|
| 0 | 06-01 Mon | .45 | .50 | .30 | .35 | Rainy week |
| 1 | 06-02 Tue | .45 | .50 | .30 | .35 | Rainy week |
| 2 | 06-03 Wed | .45 | .50 | .30 | .35 | Rainy week |
| 3 | 06-04 Thu | .46 | .51 | .31 | .36 | Rainy week |
| 4 | 06-05 Fri | .46 | .51 | .31 | .36 | Rainy week |
| 5 | 06-08 Mon | .46 | .51 | .46 | .56 | |
| 6 | 06-09 Tue | .46 | .51 | .46 | .56 | |
| 7 | 06-10 Wed | .46 | .51 | .46 | .56 | |
| 8 | 06-11 Thu | .46 | .51 | .46 | .56 | |
| 9 | 06-12 Fri | .47 | .52 | .47 | .57 | |
| 10 | 06-15 Mon | .47 | .52 | .47 | .57 | |
| 11 | 06-16 Tue | .77 | .52 | .47 | .57 | Near-miss (next-day spike) |
| 12 | 06-17 Wed | .77 | .52 | .47 | .57 | Near-miss |
| 13 | 06-18 Thu | .77 | .52 | .47 | .57 | Near-miss fading |
| 14 | 06-19 Fri | .77 | .52 | .47 | .57 | Near-miss fading |
| 15 | 06-22 Mon | .88 | .88 | .83 | .88 | Toolbox talk starts |
| 16 | 06-23 Tue | .88 | .88 | .83 | .88 | Toolbox talk |
| 17 | 06-24 Wed | .88 | .88 | .83 | .88 | Toolbox talk |
| 18 | 06-25 Thu | .88 | .88 | .83 | .88 | Toolbox talk |
| 19 | 06-26 Fri | .88 | .88 | .83 | .88 | Toolbox talk |
| 20 | 06-29 Mon | .88 | .88 | .83 | .88 | Toolbox talk holding |
| 21 | 06-30 Tue | .89 | .89 | .84 | .89 | Toolbox talk holding |
| 22 | 07-01 Wed | .89 | .89 | .84 | .89 | Toolbox talk holding |
| 23 | 07-02 Thu | .89 | .89 | .84 | .89 | Toolbox talk holding |
| 24 | 07-03 Fri | .89 | .89 | .84 | .89 | Toolbox talk holding |
| 25 | 07-06 Mon | .89 | .89 | .84 | .89 | Toolbox talk holding |
| 26 | 07-07 Tue | .89 | .89 | .84 | .89 | Toolbox talk holding |
| 27 | 07-08 Wed | .90 | .90 | .85 | .90 | Toolbox talk holding |
| 28 | 07-09 Thu | .90 | .90 | .85 | .90 | Toolbox talk holding |
| 29 | 07-10 Fri | .90 | .90 | .85 | .90 | Toolbox talk holding |

---

## After generating

There's no automatic ground-truth label here (unlike the PIL pipeline, which
knows every box because it drew it) — if you want YOLO-format labels for
these images too, they'd need manual annotation or a pass through the
trained detector. For the pitch itself, the images plus this table's
`{N, compliant-count}` numbers are enough to build the same trend chart as
`CREW_A_STORY.md` / `CREW_B_STORY.md`.
