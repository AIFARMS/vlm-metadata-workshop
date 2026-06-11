# VLM metadata workshop

**60-minute hands-on session:** send the same trail-camera image to multiple vision-language models (VLMs), collect structured metadata, and measure whether outputs **agree** — lexically (same words) and semantically (same meaning, different words).

Built for educators and CS instructors. No MCP server required; runs from this repo alone.

| Resource | Purpose |
|----------|---------|
| [WORKSHOP_60MIN.md](WORKSHOP_60MIN.md) | Instructor guide (educators + CS tracks) |
| [PARTICIPANT_WORKSHEET.md](PARTICIPANT_WORKSHEET.md) | All-in-one participant handout |
| [handouts/](handouts/) | Printable 2-page sheets + CS supplement |
| [workshopImages/](workshopImages/) | Demo JPEGs: `7109.jpg`, `7225.jpg`, `7149.jpg` |
| [coyote_metadata_comparison.csv](coyote_metadata_comparison.csv) | Pre-computed outputs: **334 images × 3 models** |
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

Open `workshopImages/7109.jpg` while reading the report. Full agenda: [WORKSHOP_60MIN.md](WORKSHOP_60MIN.md).

### Optional: semantic (SBERT) metrics

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

**There are no per-model species columns.** This dataset compares **scene/context** attributes, not species ID from each VLM.

---

## Important: species and “100% agreement”

If you run with `--species-hint coyote`, the script **injects** `"species": "coyote"` for all models on every row. That yields **100% species agreement by construction** — not evidence that every VLM identified coyote correctly.

**For the classroom, say:**

> “These images come from a coyote trail-camera set. We compare scene metadata across three models. Species ID is harder — and this CSV does not include three model species columns.”

In related AIFARMS MCP data, structured `species` is often `coyote` while free-text descriptions frequently hedge (“coyote or fox”, etc.). See `load_wide_csv` in `compare_model_agreement.py` and the min-vs-mean discussion in [WORKSHOP_60MIN.md](WORKSHOP_60MIN.md) Block 5.

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
├── WORKSHOP_60MIN.md
├── PARTICIPANT_WORKSHEET.md
├── requirements.txt           # core (TF-IDF metrics)
├── requirements-live.txt      # optional VLM APIs
├── requirements-sbert.txt     # optional SBERT
├── coyote_metadata_comparison.csv
├── compare_model_agreement.py
├── add_evaluation_metrics.py
├── classroom_vlm_comparison.py
├── handouts/
├── workshopImages/
└── samples/demo_comparison.json
```

Large outputs (`coyote_metrics/`, agreement JSON) are gitignored; regenerate with the commands above.

---

## License

- **Software and docs:** [Apache License 2.0](LICENSE)
- **Trail-camera demo images:** educational use; confirm your institution’s policy before redistributing. Coyote images derive from camera-trap collections used in the AIFARMS MCP project (see [LILA Idaho Camera Traps](https://lila.science/datasets/idaho-camera-traps/) for source licensing context).
- **Do not commit API keys** — use `.env` (see `.env.example`).

---

## Citation (optional)

If you use these materials in a course or publication, you can cite the repo URL and note: pairwise TF-IDF + optional SBERT (`all-MiniLM-L6-v2`) for inter-model agreement on structured image metadata.
