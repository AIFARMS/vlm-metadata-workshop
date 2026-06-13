<style>
@media print {
  .pagebreak { page-break-after: always; break-after: page; height: 0; margin: 0; border: 0; }
}
</style>

# Technical reference (1 page)
### Optional — same session, extra depth for repo / metrics work

**Name:** _________________________ **Image:** ☐ 7109 ☐ 7225 ☐ 7149

Print this **with** [PARTICIPANT_WORKSHEET.md](../PARTICIPANT_WORKSHEET.md) if you want a compact metrics cheat sheet. Everyone still does Parts 1–5 first.

---

## A. Vocabulary *(one sentence each)*

**Lexical / Jaccard agreement:** ____________________________________________

**TF-IDF cosine:** _________________________________________________________

**SBERT cosine:** ____________________________________________________________

**min_pairwise_semantic:** __________________________________________________

**semantic_agreement tier:** __________________________________________________

---

## B. Min vs mean (three models → three pairs per field)

| | Mean pairwise | Min pairwise |
|---|---------------|--------------|
| **Formula** | average of 3 pair scores | **smallest** of 3 |
| **Asks** | “Mostly similar?” | “**Every** pair similar?” |
| **Risk** | One dissenting pair averaged away | Can fail on paraphrase (7225 **time**) |

**7225:** lighting/scene/background = high min; time = low min → **4/5** auto-trust, not 5/5.

**Image gate:** count fields **high** or **absent**; default **≥4/5** (strict), or **≥2/5** for demos.

Why **min** for automation? ___________________________________________________

When is min **too** strict? ____________________________________________________

---

## C. Run it *(laptop + repo)*

**Quick smoke test** (first 20 images only — fast, no SBERT):

```
pip install -r requirements.txt

python3 compare_model_agreement.py \
  --csv coyote_metadata_comparison.csv \
  --species-hint coyote \
  --output /tmp/workshop_report.json \
  --limit 20 --no-sbert
```

The CSV has **334** coyote images; `--limit 20` keeps the run under ~30s on a laptop. Omit `--limit` to process all 334 (still TF-IDF-only with `--no-sbert`).

In JSON → `assessments` → your image → compare `field_results` and `semantic_trust` to **Part 3** of the worksheet.

**Full SBERT (334 images, no script needed):** `coyote_sbert_report.json` and `coyote_sbert_metrics/image_trust_summary.json`. Filter any `*_metrics.csv` on `image_id == 7225.jpg`.

**Which fields would SBERT likely upgrade from partial → high?** ________________

---

## D. Pipeline sketch

`[image] → [3 VLMs] → [ ??? ] → [trust / review queue]`

What belongs in `[ ??? ]` besides “pick the longest answer”?

___________________________________________________________________________

**Note:** `best_model` in metrics CSV is consensus-heuristic, not ground-truth accuracy.

---

*Main handout: [PARTICIPANT_WORKSHEET.md](../PARTICIPANT_WORKSHEET.md)*
