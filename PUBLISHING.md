# Publishing this folder as its own GitHub repository

Use this when you want a **small, public workshop repo** separate from the full MCP server codebase.

## 1. Create the repo on GitHub

In the [AIFARMS](https://github.com/AIFARMS) org (or your account):

- **Name suggestion:** `vlm-metadata-workshop`
- **Description:** 90-minute workshop — compare vision-language models on trail-camera metadata
- **Public**, no README/license template (this folder already has them)

## 2. Push from this directory

If `classroom_demo/` is still inside `model-context-protocol`:

```bash
cd /path/to/model-context-protocol/classroom_demo

git init
git add .
git status   # confirm no .env, __pycache__, or huge generated JSON slipped in
git commit -m "Initial public workshop materials for VLM metadata comparison"

git branch -M main
git remote add origin git@github.com:AIFARMS/vlm-metadata-workshop.git
git push -u origin main
```

If the remote repo already exists with a README, use `git pull origin main --rebase` first or force-push only if you intend to replace it.

## 3. What gets published (checklist)

| Include | Skip |
|---------|------|
| `README.md`, `WORKSHOP_90MIN.md`, `PARTICIPANT_WORKSHEET.md` | `.env`, `__pycache__/` |
| `handouts/*.md`, `handouts/make_pdfs.sh`, `handouts/PRINTING.md` | `coyote_agreement_report.json`, `coyote_metrics/` |
| `workshopImages/*.jpg` | `demo_tutorial.txt` (internal) |
| `coyote_metadata_comparison.csv` | `submit_coyote_sbert_eval.slurm` (optional HPC) |
| `compare_model_agreement.py`, `add_evaluation_metrics.py` | Parent repo paths / Taiga paths in commits |
| `classroom_vlm_comparison.py`, `samples/` | API keys in any file |

Optional: commit `handouts/*.pdf` for instructors who won't run pandoc — or leave them out and point to `PRINTING.md`.

## 4. After publish

- Add topics on GitHub: `education`, `computer-vision`, `vlm`, `trail-camera`, `metadata`
- Link from workshop slides to the repo URL
- Link **to** [species.aifarms.org](https://species.aifarms.org) and note full MCP stack lives in a separate repo when public
- Pin `WORKSHOP_90MIN.md` or `README.md` in the repo description

## 5. Keeping in sync with `model-context-protocol`

**Option A — this folder is the source of truth:** develop here, copy or push to the standalone repo when releasing.

**Option B — monorepo is source:** when you update `classroom_demo/` in `model-context-protocol`, re-copy or subtree-push to `vlm-metadata-workshop`.

**Option C — git subtree split** (preserve history for this subfolder only):

```bash
cd /path/to/model-context-protocol
git subtree split --prefix=classroom_demo -b classroom-demo-only
git push git@github.com:AIFARMS/vlm-metadata-workshop.git classroom-demo-only:main
```

After subtree split, the standalone repo root *is* the former `classroom_demo/` contents (no extra nesting).
