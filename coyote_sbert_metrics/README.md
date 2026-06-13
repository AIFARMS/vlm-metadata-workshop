# Pre-computed SBERT metrics (334 coyote images)

Full **semantic agreement** outputs from the Delta/Taiga run (`all-MiniLM-L6-v2`, min pairwise SBERT, default thresholds high ≥ 0.72 / partial ≥ 0.55).

## Files

| File | Contents |
|------|----------|
| `background_metrics.csv` | Per-image background: model text, `sbert_cos_*`, `min_pairwise_semantic`, `semantic_agreement`, `trusted_for_auto_use` |
| `lighting_metrics.csv` | Same for lighting |
| `time_metrics.csv` | Same for time |
| `setting_metrics.csv` | Same for scene/setting |
| `weather_metrics.csv` | Same for weather |
| `metrics_summary.json` | Rollups per attribute + thresholds |
| `image_trust_summary.json` | Per-image auto-trust (≥4/5 fields high or absent) |

**`../coyote_sbert_report.json`** — full agreement report with `assessments` + `semantic_metrics`.

## Workshop use (no SBERT install required)

**Instructor — auto-trust images (2/334):**

```bash
python3 - <<'PY'
import json
t = json.load(open("coyote_sbert_metrics/image_trust_summary.json"))
for img, info in sorted(t.items()):
    if info.get("auto_trust"):
        print(img, info["trusted_attribute_count"], "/", info["total_attributes"])
PY
```

Expected: **`7221.jpg`**, **`7225.jpg`** (4/5 each).

**Demo images:**

| Image | Role | Typical trusted fields |
|-------|------|------------------------|
| `7109.jpg` | Typical mixed | ~2/5 |
| `7225.jpg` | SBERT auto-trust surprise | 4/5 |
| `7149.jpg` | Fog / high divergence | ~1/5 |

**CS track — one row in Excel or Python:**

```python
import pandas as pd
df = pd.read_csv("coyote_sbert_metrics/lighting_metrics.csv")
row = df[df["image_id"] == "7225.jpg"].iloc[0]
print(row[["min_pairwise_semantic", "semantic_agreement", "trusted_for_auto_use"]])
```

## Regenerate (maintainers, GPU / PyTorch ≥ 2.4)

```bash
pip install -r requirements-sbert.txt
python3 compare_model_agreement.py \
  --csv coyote_metadata_comparison.csv \
  --species-hint coyote \
  --output coyote_sbert_report.json \
  --metrics-dir coyote_sbert_metrics/
```

Set `SBERT_CACHE` or `HF_HOME` if you need a non-default Hugging Face cache directory.
