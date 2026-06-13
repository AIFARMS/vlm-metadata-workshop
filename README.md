# VLM metadata workshop

**60-minute hands-on session:** send the same trail-camera image to multiple vision-language models (VLMs), collect structured metadata, and measure whether outputs **agree** — lexically (same words) and semantically (same meaning, different words).

Built for **educators, developers, and mixed rooms** — one core worksheet (human judgment first), optional repo hands-on for those with laptops. No MCP server required.

| Resource | Purpose |
|----------|---------|
| [PARTICIPANT_WORKSHEET.md](PARTICIPANT_WORKSHEET.md) | Main handout — **everyone** (Parts 1–5 core; Part 6 optional depth) |
| [handouts/technical_reference_1page.md](handouts/technical_reference_1page.md) | Optional one-page metrics cheat sheet (print with worksheet) |
| [workshopImages/](workshopImages/) | Demo JPEGs: `7109.jpg`, `7225.jpg`, `7149.jpg` |
| [coyote_metadata_comparison.csv](coyote_metadata_comparison.csv) | Pre-computed outputs: **334 images × 3 models** |
| [coyote_sbert_metrics/](coyote_sbert_metrics/) | Pre-computed **SBERT** agreement (334 images) |
| [compare_model_agreement.py](compare_model_agreement.py) | Agreement report + optional SBERT metrics |
| [classroom_vlm_comparison.py](classroom_vlm_comparison.py) | Live multi-model inference (optional; needs API keys) |

**Related (separate project):** [species.aifarms.org](https://species.aifarms.org) — public search over AIFARMS image metadata · [mcp.aifarms.org](https://mcp.aifarms.org) — MCP API. Full server codebase: *model-context-protocol* (AIFARMS, when published).

---

## Quick start (no API keys)

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt

python3 compare_model_agreement.py \
  --csv coyote_metadata_comparison.csv \
  --output coyote_report.json \
  --metrics-dir coyote_metrics \
  --no-sbert
```

Open `workshopImages/7109.jpg` while reading the report. Session handout: [PARTICIPANT_WORKSHEET.md](PARTICIPANT_WORKSHEET.md).

### Pre-computed SBERT (bundled — no GPU)

The repo ships **`coyote_sbert_metrics/`** and **`coyote_sbert_report.json`** so participants can explore semantic tiers without running SBERT.

```bash
# Lookup auto-trust images (expect 7221.jpg, 7225.jpg)
python3 -c "
import json
t=json.load(open('coyote_sbert_metrics/image_trust_summary.json'))
print([k for k,v in t.items() if v.get('auto_trust')])
"
```

See [coyote_sbert_metrics/README.md](coyote_sbert_metrics/README.md) for column definitions and demo-image notes.

### Regenerate SBERT yourself (Delta / GPU machine)

```bash
pip install -r requirements-sbert.txt
python3 compare_model_agreement.py \
  --csv coyote_metadata_comparison.csv \
  --output coyote_report.json \
  --metrics-dir coyote_metrics
```

### Optional: live inference (instructor only)

```bash
cp .env.example .env   # add keys; never commit .env
pip install -r requirements-live.txt
python3 classroom_vlm_comparison.py \
  --image workshopImages/7109.jpg --models auto
```

---

## What is in the CSV?

**Five scene fields × three models (GPT-4o, GPT-4.5, Llama):**

Weather · Background · Lighting · Time of Day · Setting (scene)

**Note:** The middle Azure column is labeled **GPT-4.5** in CSV and JSON exports, but the deployment that generated this dataset was **GPT-4.1** (an Azure preview that was later retired and released as GPT-4.1—the export header was not updated).

**There are no per-model species columns.** This dataset compares **scene/context** attributes, not species ID from each VLM.

---

## Important: species and “100% agreement”

If you run with `--species-hint coyote`, the script **injects** `"species": "coyote"` for all models on every row. That yields **100% species agreement by construction** — not evidence that every VLM identified coyote correctly.

In related AIFARMS MCP data, structured `species` is often `coyote` while free-text descriptions frequently hedge (“coyote or fox”, etc.). See `load_wide_csv` in `compare_model_agreement.py` and [handouts/technical_reference_1page.md](handouts/technical_reference_1page.md) for min-vs-mean pairwise semantics.

---

## Print handouts

```bash
cd handouts && ./make_pdfs.sh
```

Or in Cursor: install **Markdown PDF** extension → open a `.md` handout → `Cmd+Shift+P` → **Markdown PDF: Export (pdf)**.

---

## Repository layout

```
├── README.md
├── PARTICIPANT_WORKSHEET.md
├── requirements.txt           # core (TF-IDF metrics)
├── requirements-live.txt      # optional VLM APIs
├── requirements-sbert.txt     # optional SBERT
├── coyote_metadata_comparison.csv
├── coyote_sbert_metrics/      # pre-computed SBERT metrics (334 images)
├── coyote_sbert_report.json   # full agreement report
├── compare_model_agreement.py
├── add_evaluation_metrics.py
├── classroom_vlm_comparison.py
├── handouts/
├── workshopImages/
└── samples/demo_comparison.json
```

---

## License

- **Software and docs:** [Apache License 2.0](LICENSE)
- **Trail-camera demo images:** educational use; confirm your institution’s policy before redistributing. Coyote images derive from camera-trap collections used in the AIFARMS MCP project (see [LILA Idaho Camera Traps](https://lila.science/datasets/idaho-camera-traps/) for source licensing context).

---

## Citation (optional)

If you use these materials in a course or publication, you can cite the repo URL and note: pairwise TF-IDF + optional SBERT (`all-MiniLM-L6-v2`) for inter-model agreement on structured image metadata.
