# Participant Worksheet
## Comparing AI Vision Models on Image Metadata (60 min)

**Name:** _________________________ **Image assigned:** ☐ 7109 ☐ 7225 ☐ 7149  
**Background:** ☐ Educator / practitioner ☐ CS / technical ☐ Other __________

---

## Part 1 — Look, then read (8 min)

Your instructor will show a **night trail-camera** photo. Before seeing model text, jot your own answers:

| Field | *Your observation* |
|-------|---------------------|
| Species | |
| Scene / setting | |
| Time of day | |
| Lighting | |
| Weather | |
| Background | |

---

## Part 2 — Three models, same image (10 min)

Read the three model outputs for **your assigned image** (below). Do **not** look at automated scores yet.

### If you have **7109.jpg**

| Field | GPT-4o | GPT-4.5 | Llama |
|-------|--------|---------|-------|
| Weather | Not visible | not visible | Not visible |
| Background | Distant lights, buildings?, rest in darkness | Dark, dim structures or lights | Dark sky, distant lights |
| Lighting | Low light, infrared trail camera | Low artificial infrared / night vision | Low light, artificial illumination |
| Time | Nighttime, early morning (4:21 AM) | Nighttime (early morning, timestamp) | Night |
| Scene | Outdoor, grass, distant lights, near buildings? | Natural environment, grassy open field | Outdoor grassy field, distant lights |

### If you have **7225.jpg**

| Field | GPT-4o | GPT-4.5 | Llama |
|-------|--------|---------|-------|
| Weather | Not visible | not visible | Not visible |
| Background | Sparse bushes, small lights in distance | Bushes, distant lights or reflective objects | Trees, bushes, distant lights |
| Lighting | Low light, infrared night vision | Low-light, infrared / night vision camera | Low light, flash or infrared |
| Time | Nighttime | Night time, 01:52 AM (timestamp) | Night |
| Scene | Outdoor, dry grass and bushes | Natural outdoor, grassy area with bushes | Grassy area, trees and bushes |

### If you have **7149.jpg**

| Field | GPT-4o | GPT-4.5 | Llama |
|-------|--------|---------|-------|
| Weather | Fog or haze, misty / humid | *(empty / not visible)* | Foggy or misty |
| Background | Obscured by fog; grass shapes lower frame | Blurred grass, open outdoor space | Fog or heavy mist obscuring details |
| Lighting | Low light, nighttime | Low-light, infrared / night-vision | Low light, nighttime or early morning |
| Time | Nighttime (timestamp, low light) | Early morning (3:30 AM) | Early morning (~3:30 AM) |
| Scene | Open outdoor, field, grass/vegetation | Outdoor, natural grassy area | Outdoor, fog or heavy mist |

---

## Part 3 — Human agreement (do this first!) 

For **your image**, mark each field. Use **your judgment**, not keyword matching.

| Field | Same meaning? | Notes (what differs?) |
|-------|---------------|------------------------|
| Species *(all rows: coyote)* | ☐ Y ☐ Partial ☐ N | |
| Weather | ☐ Y ☐ Partial ☐ N | |
| Lighting | ☐ Y ☐ Partial ☐ N | |
| Time | ☐ Y ☐ Partial ☐ N | |
| Scene | ☐ Y ☐ Partial ☐ N | |
| Background | ☐ Y ☐ Partial ☐ N | |

**Quick definitions**

- **Y** — Same factual claim (even if wording differs)
- **Partial** — Overlap, but one model adds or omits something important
- **N** — Meaningfully different claims

---

## Part 4 — Compare to the pipeline (instructor reveals)

After the instructor runs the agreement report, fill in:

| Question | Your answer |
|----------|-------------|
| Script verdict for your image (mixed / high_divergence / other) | |
| Did lexical/string matching match your “Y / Partial / N”? | |
| Which field had the biggest gap between your judgment and the score? | |
| Would you auto-save this row to a database without review? ☐ Yes ☐ No ☐ Some fields only | |

---

## Part 5 — Debrief (all participants)

Discuss in pairs, then share with the room:

1. **When all three models agree, is that enough to trust the label?** Why or why not?

2. **Where did models use different words for the same idea?** (Example: “infrared trail cam” vs “night vision.”)

3. **Where did models actually disagree on facts?** (Example: buildings vs natural field; fog described vs missing.)

4. **For your classroom or project:** One activity where students compare AI output to their own observation.

5. **Ethics:** What image types should *not* go through a cloud VLM in a K–12 or university class?

---

## Part 6 — CS / technical extension (optional)

*Skip if you are focused on classroom use; discuss with your pair if time allows.*

### A. Metrics vocabulary

| Term | In one sentence, what does it measure? |
|------|----------------------------------------|
| Lexical / Jaccard agreement | |
| TF-IDF cosine | |
| SBERT cosine | |
| `min_pairwise_semantic` | |
| `semantic_agreement` tier (high / partial / low / absent) | |

### B. Design question — min vs mean pairwise

For each **image × field**, three models → **three pairwise** scores (e.g. GPT-4o↔Llama, GPT-4o↔4.5, 4.5↔Llama):

| Summary | What it is | Good for |
|---------|------------|----------|
| **Mean pairwise** | Average of the 3 scores | “How aligned are they overall?” (summaries) |
| **Min pairwise** | **Smallest** of the 3 scores | “Did **every** pair agree?” (auto-publish gate) |

**Why min, not mean?** One weak pair can hide inside a high average.

Example: scores **0.85, 0.81, 0.38** → mean ≈ 0.68 (looks OK) but **min = 0.38** → block auto-trust.

**7225.jpg example:** Lighting/scene/background → high min (all say night / infrared / grass). **Time** → low min (“Nighttime” vs “01:52 AM” vs “Night”) even though humans agree it is night. Image passes **4 of 5** fields, not 5.

**Image rule:** Count fields with tier **high** or **absent**. Default production gate: **≥4 of 5**. Relaxed demo gate: **≥2 of 5** (`--auto-trust-min-attrs 2`).

**Your answers:**

Why use **min** instead of **mean**? _______________________________________________

When is **min** too strict? _______________________________________________________

### C. Try it (laptop + repo)

```bash
python3 classroom_demo/compare_model_agreement.py \
  --csv classroom_demo/coyote_metadata_comparison.csv \
  --species-hint coyote \
  --output /tmp/workshop_report.json \
  --limit 20 \
  --no-sbert
```

Open `/tmp/workshop_report.json` → find your image under `"assessments"` → compare `field_results` and `semantic_trust` to Part 3.

**Stretch:** Re-run one image mentally with SBERT — which fields would move from partial → high? (Hint: lighting/time paraphrases.)

### D. Implementation sketch (5 min)

Draw or describe a minimal pipeline:

```
[image] → [3 VLMs] → [???] → [trust / review queue]
```

What belongs in `[???]` for a production system that is *not* just “pick the longest answer”?

---

## Reference — dataset headlines (instructor slide)

| Stat | Value |
|------|-------|
| Trail-cam images | 334 |
| Models | GPT-4o, GPT-4.5, Llama |
| Species agreement | 100% |
| Mean lexical overall | ~42% |
| Auto-trust (strict rule, ≥4 of 5 fields) | 2 images (0.6%) |
| High-divergence outliers | 5 images |

**Remember:** Agreement = confidence among models, **not** ground truth.

---

*Workshop materials: `classroom_demo/handouts/` — short educator sheets: `practitioner_*_2page.md` · CS: `cs_supplement_1page.md`*
