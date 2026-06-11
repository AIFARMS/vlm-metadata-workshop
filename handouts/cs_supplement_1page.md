<style>
@media print {
  .pagebreak { page-break-after: always; break-after: page; height: 0; margin: 0; border: 0; }
}
</style>

# CS supplement (1 page)
### Same workshop — technical track

**Name:** _________________________ **Image:** ☐ 7109 ☐ 7225 ☐ 7149

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

```
python3 classroom_demo/compare_model_agreement.py \
  --csv classroom_demo/coyote_metadata_comparison.csv \
  --species-hint coyote \
  --output /tmp/workshop_report.json \
  --limit 20 --no-sbert
```

In JSON → `assessments` → your image → compare `field_results` and `semantic_trust` to the educator rubric.

**Which fields would SBERT likely upgrade from partial → high?** ________________

---

## D. Pipeline sketch

`[image] → [3 VLMs] → [ ??? ] → [trust / review queue]`

What belongs in `[ ??? ]` besides “pick the longest answer”?

___________________________________________________________________________

**Note:** `best_model` in metrics CSV is consensus-heuristic, not ground-truth accuracy.

---

*Educator handout: `practitioner_<image>_2page.md` · Full print doc: `worksheet_pdf_full.md`*
