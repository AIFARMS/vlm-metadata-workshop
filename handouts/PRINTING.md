# Printing guide — workshop handouts

## Quick pick

| Audience | File | Pages (approx.) |
|----------|------|-----------------|
| Educators, one image | `practitioner_7109_2page.md` (or `_7225_`, `_7149_`) | **2** |
| CS (add-on) | `cs_supplement_1page.md` | **1** |
| Everyone, full + PDF | `worksheet_pdf_full.md` | **6–7** |
| CS only, full | `worksheet_pdf_full.md` (print all pages) | **6–7** |

## How to convert Markdown → PDF

### Option A — One command (recommended on your Mac)

You already have **pandoc** and **LaTeX** installed. From the repo root:

```bash
cd classroom_demo/handouts
./make_pdfs.sh
```

That writes `.pdf` next to each `.md` handout in this folder. To convert a single file:

```bash
cd classroom_demo/handouts
pandoc cs_supplement_1page.md -o cs_supplement_1page.pdf \
  --pdf-engine=xelatex -V geometry:margin=0.75in
```

Open the PDF in Preview (double-click) or print from there.

### Option B — Cursor / VS Code extension (no terminal)

1. Install extension **“Markdown PDF”** (yzane.markdown-pdf).
2. Open the `.md` file (e.g. `practitioner_7109_2page.md`).
3. Command Palette (`Cmd+Shift+P`) → **Markdown PDF: Export (pdf)**.
4. PDF appears in the same folder as the markdown file.

### Option C — Browser print (quick & dirty)

1. In Cursor, open the `.md` file and use **Markdown Preview** (preview pane).
2. `Cmd+P` → **Save as PDF**.
3. Enable “Background graphics” if checkboxes look faint.

Page breaks in split handouts may not work in the browser; use Option A or B for 2-page practitioner sheets.

### Option D — Google Docs

1. Copy markdown text into a new Google Doc (tables paste OK).
2. File → Download → **PDF**.

---

## How to print

Page breaks use `<div class="pagebreak"></div>`. If your tool ignores them, print from the **split files** (one 2-page file per image).

## Room setup

- **Educator pairs:** one 2-page sheet matching assigned image (7109 / 7225 / 7149).
- **CS with laptops:** same 2-page sheet **plus** `cs_supplement_1page.md`.
- **Mixed pair:** share one 2-page sheet; CS person keeps supplement.

### Print counts (~20 participants)

Pairs share one sheet (~10 sheets total):

| Handout | Qty | Role |
|---------|-----|------|
| `practitioner_7109_2page.md` | **4** | Typical mixed case; live walkthrough |
| `practitioner_7225_2page.md` | **3** | Semantic-agreement “surprise” |
| `practitioner_7149_2page.md` | **3** | Fog / high-divergence |
| `cs_supplement_1page.md` | **8–10** | CS folks or anyone with a laptop |
| Extra `7109` | **+2** | Spares / instructor copy |

Assign images round-robin at tables (7109 → 7225 → 7149 → …).

## Instructor copies

Keep `worksheet_pdf_full.md` (or `WORKSHOP_60MIN.md` answer key) for verdicts: 7109 mixed, 7225 auto-trust, 7149 high_divergence.

## Workshop demo images (7109, 7225, 7149)

**In this repo:**

```
classroom_demo/workshop_images/7109.jpg   # typical mixed — live walkthrough
classroom_demo/workshop_images/7225.jpg   # semantic auto-trust surprise
classroom_demo/workshop_images/7149.jpg   # fog / high-divergence
```

Live inference example:

```bash
python3 classroom_demo/classroom_vlm_comparison.py \
  --image classroom_demo/workshop_images/7109.jpg --models auto
```

**Original Taiga source** (if you need more coyote images):

```
/taiga/ncsa/radiant/bbgp/rgpu02/owodd/coco_datasets/animals/coyote/
```
