# Workshop: Comparing AI Vision Models on Image Metadata
## 60-minute instructor guide (mixed audience: educators + CS)

**Goal:** Participants learn how to send the same image to multiple vision-language models (VLMs), collect structured metadata, and measure whether the models are *saying the same thing* — using both human judgment and automated similarity metrics.

**Audiences:** Classroom practitioners (ag, bio, STEM educators) and CS folks (students, developers, data engineers) in the same room. Same core demo; **dual tracks** during hands-on and closing (see below).

**Participant handout:** Print from `handouts/` (`practitioner_*_2page.md`, `cs_supplement_1page.md`, or run `./handouts/make_pdfs.sh`)

| Who | File |
|-----|------|
| Educators (2 pp, one image) | `handouts/practitioner_7109_2page.md` · `_7225_` · `_7149_` |
| CS add-on | `handouts/cs_supplement_1page.md` |
| Full PDF layout | `handouts/worksheet_pdf_full.md` |

**Core message:** AI can help students observe and describe the world, but agreement between models is **confidence**, not **truth**. The workshop builds data-literacy and critical thinking, not blind trust in AI labels.

---

## Learning objectives

### Everyone

1. Name 4–6 metadata fields useful for ecological / trail-camera images (species, scene, time, lighting, weather, background).
2. Interpret a three-model comparison on one image (with or without running code).
3. Distinguish **lexical agreement** (same words) from **semantic agreement** (same meaning, different words).
4. Explain why “all three models agree” does not prove correctness.

### Practitioners (educators)

5. Sketch one classroom activity where students compare model outputs to their own observations.
6. Name one privacy/ethics constraint for using cloud VLMs with student-generated images.

### CS participants

5. Read `field_results` / `semantic_trust` in the agreement JSON and relate them to pairwise metrics.
6. Articulate why `min_pairwise_semantic` is used instead of mean, and one failure mode of consensus-based scoring.

---

## Dual-track design (same room)

| Moment | Practitioners | CS participants |
|--------|---------------|-----------------|
| **0:08 metadata intro** | “Questions students could answer from the photo” | Same + optional: JSON schema / MCP fields |
| **0:28 hands-on** | Worksheet Part 3 — human Y/Partial/N | Part 3 **plus** Part 6B–C — run script, open JSON |
| **0:40 metrics block** | 7225 vs 7149 stories, trust / review queue | Same + TF-IDF vs SBERT, threshold tuning |
| **0:52 close** | Classroom activity ideas | Pipeline sketch: inference → metrics → HITL queue |

**Facilitation tip:** Assign image IDs in pairs (7109 / 7225 / 7149) so each table has mixed backgrounds; CS folks can coach lexical vs semantic without dominating.

---

## What you need before the workshop

### Instructor (required)

| Item | Notes |
|------|--------|
| Laptop + projector | Test HDMI/adapters day before |
| Python 3.10+ | `pip install pandas scikit-learn` minimum; SBERT optional |
| This repo | Clone [vlm-metadata-workshop](https://github.com/AIFARMS/vlm-metadata-workshop); `cd` into repo root |
| Pre-built coyote CSV | `coyote_metadata_comparison.csv` (334 images × 3 models) |
| **SBERT metrics (bundled)** | `coyote_sbert_metrics/` + `coyote_sbert_report.json` |
| **3 demo images** | `workshopImages/{7109,7225,7149}.jpg` |
| Slides (optional) | 5–8 slides: agenda, metadata fields, one results table |

### Participants

| Tier | What they need |
|------|----------------|
| **Tier A — full hands-on** | Laptop, Python, repo clone; instructor provides API keys on lab VM *or* uses pre-computed CSV |
| **Tier B — follow along** | Laptop optional; **`PARTICIPANT_WORKSHEET.md`** (print or PDF) |
| **Tier C — observe** | No laptop; discuss in pairs using projected tables |

**Recommendation:** Plan for Tier B. Run live API calls yourself (Tier A instructor demo); students analyze pre-computed rows so no one needs keys.

### API keys (instructor only, optional for live inference)

- Copy `.env.example` → `.env`, `chmod 600 .env`
- Never share keys in chat or slides
- Alternative: skip live inference entirely; use `coyote_metadata_comparison.csv`

---

## 60-minute agenda

| Time | Block | What happens |
|------|--------|----------------|
| 0:00–0:08 | **Welcome & framing** | Intros, goals, ethics (see talking points below) |
| 0:08–0:18 | **Metadata from images** | What VLMs return; show MCP-style fields; trail-cam example |
| 0:18–0:28 | **Live: three models, one image** | Project `7109.jpg`; show three columns of text side by side |
| 0:28–0:40 | **Hands-on: “Do they agree?”** | Worksheet Part 3 (all); CS: Part 6 + optional `--limit 20` run |
| 0:40–0:52 | **Metrics: strings vs meaning** | 7225 (paraphrase) vs 7149 (real disagreement); SBERT story |
| 0:52–1:00 | **Applications & Q&A** | Practitioners: classroom ideas; CS: pipeline / metrics Q&A |

*Optional:* skip live API inference to save ~5 min; use pre-computed CSV only. No scheduled break — offer a quick stretch between blocks if the room needs it.

---

## Mixed-audience talking points

### For practitioners (keep concrete)

- Parallel to **peer review** or **triangulation** in lab reports.
- **Disagreement flags** = teachable moment, not failure.
- **7109** → typical class discussion (“same night scene, different sentences”).
- **7149** → “when the image is hard, AI is hard too” (fog, blur).

### For CS (keep precise, not gatekeeping)

- **No ground-truth labels** in this dataset — inter-model agreement only.
- **TF-IDF** = bag-of-words; **SBERT** = embedding cosine; paraphrase gap is the whole lesson.
- **`best_model` in metrics CSV** is consensus-heuristic, not accuracy — do not present as winner.
- **Auto-trust 0.6%** → strict `min_pairwise` + ≥4/5 fields; good default for HITL pipelines. See **Min vs mean** box in Block 5; relax with `--auto-trust-min-attrs 2` for demos.
- Extension: swap coyote CSV for your own wide CSV; same `compare_model_agreement.py` interface.

### Bridge sentence (use after hands-on)

> “Educators: you just did the rubric we’d ask students to do. CS folks: the script automates that rubric with lexical + semantic scores — but the human column is still the anchor.”

---

## Block 1 (8 min): Welcome & framing

### Opening script (≈2 min)

> “Today we’re not learning to *trust* AI to label nature. We’re learning to *use* AI as one source of evidence — like three students describing the same photograph. Sometimes they agree because they’re right. Sometimes they agree because they’re all guessing the same wrong thing. Our job is to compare, question, and verify.”

### Ground rules

- Models can hallucinate species, weather, and time.
- Empty answers (“not visible”) are valid and often correct.
- Disagreement is **information** — it tells you what needs human review.
- No student PII in images; use public-domain or course-owned trail-cam data.

### Poll (optional, 2 min)

Ask by show of hands:

1. Have you used ChatGPT or similar with students?
2. Have students analyzed photos as data (not just “look at this picture”)?

---

## Block 2 (10 min): Metadata from images

### Teach the field set (use trail-cam photo on screen)

| Field | Student-friendly question |
|-------|---------------------------|
| **species** | What animal or plant is this? |
| **scene / setting** | Where is this — field, forest, backyard? |
| **time** | Day or night? What time if timestamp visible? |
| **lighting** | Sunlight, flash, infrared night vision? |
| **weather** | Rain, fog, clear — or can’t tell? |
| **background** | What’s behind the main subject? |

### Why three models?

- Single model → single bias / single failure mode.
- Three models → **consensus** when they align; **flag for review** when they don’t.
- Pedagogy parallel: peer review, triangulation in science class.

### Optional live inference (5 min)

If you have keys and network:

```bash
python3 classroom_vlm_comparison.py --image workshopImages/7109.jpg --models gpt4o,gemini,claude
python3 compare_model_agreement.py --input output/*_comparison.jsonl --output demo_report.json
```

If not, skip to pre-computed CSV — same learning outcome.

---

## Block 3 (10 min): Live walkthrough — `7109.jpg` (typical mixed case)

### Show the image first (30 sec)

Night trail camera: coyote in grass, distant lights. Ask: *“What would you write for species? Time? Lighting?”*  
*(Images are from a coyote dataset; the CSV does not include species columns from each model — see [README.md](README.md).)*

### Side-by-side table (project this)

| Field | GPT-4o | GPT-4.5 | Llama |
|-------|--------|---------|-------|
| **lighting** | infrared trail camera | infrared night vision | low light, artificial |
| **time** | Nighttime, early morning (4:21 AM) | Nighttime (early morning, timestamp) | Night |
| **scene** | grass, distant lights, near buildings? | natural grassy open field | outdoor grassy field |
| **background** | distant lights, darkness | dark, dim structures | dark sky, distant lights |
| **weather** | Not visible | not visible | Not visible |

*(No species columns in CSV — ask the room what species they see.)*

### Facilitated discussion (5 min)

1. **Which fields feel unanimous to you?** (weather — all “not visible”)
2. **Which feel like “same idea, different words”?** (lighting, time)
3. **Where do models genuinely diverge?** (GPT-4o mentions “buildings”; others say “natural field”)

### Reveal the scores (2 min)

From the full coyote SBERT run (334 images):

- Lexical overall agreement on **scene fields in CSV**: **~42%** → verdict **mixed**
- SBERT auto-trust: **no** on 7109 (2 of 5 attributes trusted)
- **Do not cite “100% species”** — that comes from `--species-hint coyote`, not model outputs (see README)

**Talking point:** *“String matching says ‘mixed’ on almost everything. That’s normal for free-text captions. Next we’ll ask: are they actually describing different scenes, or just writing differently?”*

---

## Block 4 (12 min): Hands-on exercise — human agreement first

### Instructions for pairs (5 min)

Hand out **`PARTICIPANT_WORKSHEET.md`** (print double-sided if possible).

Give each pair **one image ID**: `7109`, `7225`, or `7149` (mix practitioner + CS in each pair if you can).

- **Everyone:** Worksheet Parts 1–3 (human agreement before scores).
- **CS with laptops:** Part 6C while others finish Part 3.

Data is on the worksheet; optional source: `coyote_metadata_comparison.csv`.

### Run the script (instructor, 5 min)

**Smoke test (20 rows)** — full dataset is 334 images; use bundled SBERT for the metrics block:

```bash
python3 compare_model_agreement.py \
  --csv coyote_metadata_comparison.csv \
  --species-hint coyote \
  --output workshop_report.json \
  --limit 20 \
  --no-sbert
```

Open JSON → find their image → compare `field_results` and `divergences` to pair answers.

**Without Python:** use pre-extracted rows in the “Answer key” section below.

### Debrief (5 min)

- How many pairs marked “same meaning” for lighting on 7109? (Usually most.)
- **Species side note:** CSV has no model species columns; in MCP JSON, ~80% of descriptions hedge (“coyote or fox”, “canine”, etc.) while structured species was set to coyote.
- **Key insight:** Human “partial” often matches script “split” or “mixed.”

---

## Block 5 (12 min): When agreement metrics lie and tell the truth

### Story A — `7225.jpg` (“looks broken, actually best case”)

**Lexical:** mixed (46%) — every field “split”  
**SBERT auto-trust:** **yes** — 4 of 5 attributes (one of only 2 images in 334)

| Field | Sample outputs | SBERT tier |
|-------|----------------|------------|
| lighting | all say infrared / night vision | **high** |
| scene | grass, bushes, outdoor | **high** |
| background | bushes + distant lights | **high** |
| time | Nighttime / 01:52 AM / Night | low |
| weather | all not visible | absent ✓ |

**Script (90 sec):**

> “If you only did Ctrl+F string matching, you’d say these models fight on every line. SBERT compares *meaning*. Here, they’re describing the same night trail-cam scene. Only time-of-day wording is weak. This is our pipeline working: high semantic agreement → we might auto-fill those fields; low time → human glance at timestamp.”

### Min vs mean pairwise agreement *(instructor reference — project or hand out)*

For each **image × field**, three models produce **three pairwise** similarity scores (GPT-4o↔GPT-4.5, GPT-4o↔Llama, GPT-4.5↔Llama):

| Summary | Formula | Question it answers |
|---------|---------|---------------------|
| **Mean pairwise** | average of the 3 scores | “On average, how similar are the models?” |
| **Min pairwise** | **minimum** of the 3 scores | “Did **every** pair agree well enough?” |

**Why min for auto-trust (not mean):** Mean is pulled up by strong pairs even when one pair clearly disagrees.

Example (hypothetical lighting scores):

- Pairs: **0.85**, **0.81**, **0.38** → mean ≈ **0.68** (looks “partial”) but **min = 0.38** → tier **low** → do not auto-publish.

**7225 time field (real case):** GPT-4o says “Nighttime,” GPT-4.5 says “Night time, 01:52 AM,” Llama says “Night.” Humans agree it is night; SBERT **min** is still **low** because one pair’s wording diverges enough. Lighting/scene/background on the same image have **high min** — that is why 7225 passes **4 of 5** fields, not 5 of 5.

**Image-level gate (two steps):**

1. **Per field:** tier = **high** if min ≥ threshold (default 0.72 SBERT), or **absent** if all models say “not visible.”
2. **Per image:** **auto_trust** if enough fields pass — default **≥4 of 5** in `add_evaluation_metrics.py`. For exploratory/demo use, relax with `--auto-trust-min-attrs 2` (at least two fields where every pair agrees).

**Plain-language analogy:** Mean = “the class mostly agrees.” Min = “would **every** pair of students give the same answer if asked separately?” For metadata you might publish without review, use min.

**Note:** With only **two** models, there is one pair per field — min and mean are the same.

### Story B — `7149.jpg` (“real disagreement — fog”)

**Lexical:** high_divergence (35%)  
**SBERT auto-trust:** no (1 of 5)

| Field | Issue |
|-------|--------|
| weather | GPT-4o: fog/haze; GPT-4.5: *empty*; Llama: foggy |
| background/scene | fog obscures details — three different emphases |
| lighting | oddly **high** SBERT (all say low light / night) |

**Script (90 sec):**

> “Here both humans and metrics say: be careful. Missing weather from one model, fog confusing the scene — this goes to a review queue, not into a database unchecked. Five images in our set look like this; they’re valuable teaching examples of *epistemic humility*.”

### String match vs semantic match (2 min diagram)

```
Same meaning, different words     →  Lexical: LOW   |  SBERT: HIGH
Different facts                   →  Lexical: LOW   |  SBERT: LOW
Same empty answer ("not visible") →  Lexical: HIGH  |  tier: absent
```

### Dataset headline stats (1 slide)

| Metric | Value |
|--------|-------|
| Images | 334 |
| Fields in CSV | weather, background, lighting, time, scene |
| Mean lexical overall (scene fields) | **~42%** |
| Auto-trust (strict rule) | **2 images (0.6%)** |
| High-divergence outliers | **5 images** |

**Takeaway:** This workshop compares **scene metadata** paraphrases and disagreements. Species ID is harder and was **not** exported per model in this CSV.

---

## Block 6 (8 min): Applications & close

### Activity ideas — practitioners (pick 2)

1. **Triangulation lab** — Students write metadata, then compare to three model outputs; discuss largest gaps.
2. **Skepticism journal** — One trail-cam image per week: “What did AI get wrong that I could see in the photo?”
3. **Consensus rubric** — Class defines what “agree” means per field before seeing metrics.
4. **Ethics mini-case** — Model says “coyote” on blurry image; two others say “dog.” Who decides?
5. **Data pipeline role-play** — Roles: photographer, AI worker, reviewer, database curator.

### Extension prompts — CS (pick 1 if time)

1. **Threshold sweep** — How would auto-trust rate change if `semantic-high` were 0.65 vs 0.72?
2. **Fourth model** — What breaks in the wide CSV format if you add Claude? (column naming, pairwise count)
3. **Ground truth** — Design an eval set with expert labels; which metric would you trust then?
4. **Latency/cost** — Three API calls × 334 images; where does batch + cache fit?

### Limitations to state clearly

- Models were not scored against biological ground truth — only against each other.
- Trail-cam infrared images are hard; fog/rain breaks everyone.
- API costs and FERPA/privacy if using student photos.

### Commands cheat sheet (handout)

Run from **repo root** after `git clone` (no `classroom_demo/` prefix):

```bash
# 1) Compare models on images (instructor, needs API keys in .env)
python3 classroom_vlm_comparison.py --image workshopImages/7109.jpg --models auto

# 2) Regenerate agreement + metrics from CSV (optional — SBERT outputs are in coyote_sbert_metrics/)
python3 compare_model_agreement.py \
  --csv coyote_metadata_comparison.csv \
  --species-hint coyote \
  --output report.json \
  --metrics-dir metrics_out/

# 3) No GPU / no SBERT (lexical + TF-IDF only)
python3 compare_model_agreement.py \
  --csv coyote_metadata_comparison.csv \
  --species-hint coyote \
  --output report.json \
  --no-sbert

# 4) Demo mode without keys
python3 classroom_vlm_comparison.py --demo --image workshopImages/7109.jpg
```

### Closing questions (split room or popcorn)

- **Practitioners:** “Where in your curriculum could students compare *multiple sources* — not just AI — the way we did today?”
- **CS:** “What would you add between `[3 VLMs]` and `[database]` that we did *not* build today?”

---

## Answer key for hands-on (instructor)

### 7109.jpg — typical **mixed**

- Human: weather Y; lighting partial; time partial; scene partial; background partial
- Script: verdict **mixed**; SBERT trusted 2/5 (background high, weather absent)

### 7225.jpg — **best semantic agreement**

- Human: lighting partial→Y; scene Y; background Y; time partial; weather Y (absent)
- Script: verdict **mixed** lexically; SBERT **auto_trust true** (4/5)

### 7149.jpg — **high divergence**

- Human: weather partial (one empty); scene partial; background N/partial; time partial
- Script: verdict **high_divergence**; SBERT 1/5 trusted; review required

---

## Instructor prep checklist (week before)

- [ ] Print handouts from **`handouts/`** — mix of `practitioner_*_2page.md` by table assignment + `cs_supplement_1page.md` for CS (~4×7109, 3×7225, 3×7149, 8–10 CS supplements)
- [x] Demo images in `workshopImages/` (7109, 7225, 7149)
- [ ] Confirm **`coyote_sbert_metrics/`** and **`coyote_sbert_report.json`** are in the repo (or run `./copy_sbert_from_taiga.sh` from Taiga)
- [ ] Test `compare_model_agreement.py --limit 5 --no-sbert` on your laptop (optional)
- [ ] Print or share CSV excerpt (3 rows) for Tier B participants
- [ ] Decide: live API demo yes/no
- [ ] Prepare `.env` on lab machine OR use CSV-only path
- [ ] Optional: export one slide per image with photo + 3-column table
- [ ] Skim `coyote_sbert_metrics/image_trust_summary.json` for 7109 / 7225 / 7149 / 7221 (backup if live demo fails)

---

## How you can extend this workshop

| Extension | Time | Description |
|-----------|------|-------------|
| Student images | +30 min | Bring 5 course-safe photos; run `--demo` or instructor batch |
| Build a wide CSV | +20 min | Export spreadsheet from three model JSON outputs |
| Rubric design | +15 min | Groups write “when is partial agreement OK?” per field |
| Research connection | +10 min | Link to MCP / biodiversity data standards in your institution |

---

## Files in this repo

| File | Role |
|------|------|
| `classroom_vlm_comparison.py` | Call 2–3 VLMs on an image; emit JSONL |
| `compare_model_agreement.py` | Agreement report + optional SBERT metrics |
| `add_evaluation_metrics.py` | Semantic similarity per attribute |
| `coyote_metadata_comparison.csv` | Pre-computed 334×3 workshop dataset |
| `coyote_sbert_metrics/` | Pre-computed SBERT per-attribute CSVs + `image_trust_summary.json` |
| `coyote_sbert_report.json` | Full 334-image agreement + semantic report |
| `copy_sbert_from_taiga.sh` | Copy metrics from Taiga into repo |
| `handouts/` | PDF-friendly 2-page practitioner sheets, CS supplement, full print doc |
| `PARTICIPANT_WORKSHEET.md` | All-in-one markdown handout |
| `WORKSHOP_60MIN.md` | This instructor guide |
| `.env.example` | API key template (instructor only) |

---

*Workshop version 1.0 — coyote trail-camera dataset, GPT-4o / GPT-4.5 / Llama comparison pipeline.*
