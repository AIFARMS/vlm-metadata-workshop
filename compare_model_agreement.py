#!/usr/bin/env python3
"""
Assess agreement vs divergence among multi-model VLM outputs (MCP metadata).

Reads:
  - JSONL from classroom_vlm_comparison.py (one row per image × model)
  - Wide CSV with columns like "Weather (GPT-4o)", "Setting (Llama)", ...

Examples:
  python compare_model_agreement.py --input output/beetle_comparison.jsonl

  # Agreement + semantic metrics (SBERT/TF-IDF) in one command:
  python compare_model_agreement.py --csv coyote_metadata_comparison.csv \\
      --species-hint coyote --output coyote_agreement_report.json

  python compare_model_agreement.py --csv coyote_metadata_comparison.csv \\
      --species-hint coyote --output coyote_agreement_report.json --limit 20 --no-sbert

Optional semantic summary (needs OPENAI_API_KEY or GOOGLE_API_KEY):
  python compare_model_agreement.py --csv coyote_metadata_comparison.csv --narrative
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

STRUCTURED_FIELDS_DEFAULT = [
    "species",
    "common_name",
    "scientific_name",
    "scene",
    "time",
    "season",
    "weather",
    "lighting",
    "action",
]

LONG_TEXT_FIELDS = frozenset({"background", "description"})

CSV_FIELD_MAP = {
    "weather": "weather",
    "background": "background",
    "lighting": "lighting",
    "time of day": "time",
    "setting": "scene",
}

VAGUE_VALUES = frozenset({
    "", "unknown", "unclear", "not visible", "not visible.", "n/a", "none", "unsure",
})


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(".,;:")
    return text


def is_informative(value: Any) -> bool:
    return normalize_text(value) not in VAGUE_VALUES


def token_set(text: str) -> set[str]:
    text = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", text)
    return {t for t in tokens if len(t) > 2}


def jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def pairwise_text_similarity(texts: Sequence[str]) -> float:
    """Average pairwise similarity across all model pairs."""
    if len(texts) < 2:
        return 1.0
    scores = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            ti, tj = texts[i], texts[j]
            jac = jaccard(ti, tj)
            seq = SequenceMatcher(None, normalize_text(ti), normalize_text(tj)).ratio()
            scores.append(0.5 * jac + 0.5 * seq)
    return sum(scores) / len(scores) if scores else 1.0


def field_consensus(values: Dict[str, Any], *, long_text: bool = False) -> dict:
    """
    Given model_id -> value, return agreement stats.
    For long_text fields, agreement_ratio = average pairwise text similarity.
    """
    items = [(mid, normalize_text(v)) for mid, v in values.items()]
    informative = [(mid, v) for mid, v in items if is_informative(v)]
    if len(items) <= 1:
        return {
            "status": "single_model",
            "agreement_ratio": 1.0,
            "unique_values": len(set(v for _, v in items)),
            "values_by_model": values,
            "consensus_value": items[0][1] if items else "",
        }

    if long_text and len(items) >= 2:
        texts = [str(values[mid]) for mid, _ in items]
        sim = pairwise_text_similarity(texts)
        if sim >= 0.75:
            status = "unanimous"
        elif sim >= 0.5:
            status = "majority"
        elif sim >= 0.3:
            status = "mixed"
        else:
            status = "split"
        return {
            "status": status,
            "agreement_ratio": round(sim, 3),
            "unique_values": len(set(normalize_text(t) for t in texts if is_informative(t))) or 1,
            "consensus_value": "",
            "values_by_model": values,
            "similarity_based": True,
        }

    compare = informative if len(informative) >= 2 else items
    counts = Counter(v for _, v in compare)
    top_value, top_count = counts.most_common(1)[0]
    n = len(compare)
    ratio = top_count / n

    if len(counts) == 1:
        status = "unanimous"
    elif ratio > 0.5:
        status = "majority"
    elif len(informative) >= 2 and len(counts) == len(informative):
        status = "split"
    else:
        status = "mixed"

    return {
        "status": status,
        "agreement_ratio": round(ratio, 3),
        "unique_values": len(counts),
        "consensus_value": top_value if status in ("unanimous", "majority") else "",
        "values_by_model": values,
        "value_counts": dict(counts),
    }


@dataclass
class ImageAssessment:
    image: str
    models: List[str] = field(default_factory=list)
    field_results: Dict[str, dict] = field(default_factory=dict)
    description_similarity: float = 0.0
    structured_agreement: float = 0.0
    overall_agreement: float = 0.0
    verdict: str = ""
    divergences: List[str] = field(default_factory=list)


def load_rows(paths: List[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def parse_wide_csv_columns(fieldnames: Sequence[str]) -> tuple[str, Dict[str, Dict[str, str]]]:
    """
    Parse wide CSV headers like 'Weather (GPT-4o)' -> models['GPT-4o']['weather'].
    Returns (image_id_column, models_map).
    """
    image_col = ""
    for fn in fieldnames:
        if fn.strip().lower().replace("_", " ") in ("image id", "image", "filename", "file"):
            image_col = fn
            break
    if not image_col:
        image_col = fieldnames[0]

    models: Dict[str, Dict[str, str]] = defaultdict(dict)
    pattern = re.compile(r"^(.+?)\s+\((.+)\)\s*$")
    for fn in fieldnames:
        if fn == image_col:
            continue
        m = pattern.match(fn.strip())
        if not m:
            continue
        raw_field, model = m.group(1).strip(), m.group(2).strip()
        key = CSV_FIELD_MAP.get(raw_field.lower())
        if key:
            models[model][key] = fn
    return image_col, dict(models)


def load_wide_csv(
    path: Path,
    *,
    species_hint: str = "",
    limit: int = 0,
) -> Dict[str, List[dict]]:
    """Load wide comparison CSV into image -> model rows (same shape as JSONL groups)."""
    groups: Dict[str, List[dict]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header: {path}")
        image_col, model_cols = parse_wide_csv_columns(reader.fieldnames)
        if not model_cols:
            raise SystemExit(
                f"Could not parse model columns from CSV header. "
                f"Expected columns like 'Weather (GPT-4o)'. Got: {reader.fieldnames}"
            )

        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            image_id = (row.get(image_col) or "").strip()
            if not image_id:
                continue
            model_rows = []
            for model_name, col_map in model_cols.items():
                meta: Dict[str, Any] = {}
                if species_hint:
                    meta["species"] = species_hint
                for field_key, col_name in col_map.items():
                    meta[field_key] = (row.get(col_name) or "").strip()
                bg = meta.get("background", "")
                scene = meta.get("scene", "")
                meta["description"] = " ".join(p for p in (bg, scene) if p).strip()
                model_rows.append({
                    "image": image_id,
                    "model_id": model_name,
                    "provider": model_name,
                    "mcp_metadata": meta,
                    "parsed": meta,
                })
            groups[image_id] = model_rows
    return groups


def group_by_image(rows: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        image = row.get("image") or row.get("image_path") or "unknown"
        groups[image].append(row)
    return dict(groups)


def assess_image(
    image: str,
    model_rows: List[dict],
    structured_fields: Optional[List[str]] = None,
) -> ImageAssessment:
    fields = structured_fields or STRUCTURED_FIELDS_DEFAULT
    models = [r.get("model_id") or r.get("provider") or f"model_{i}" for i, r in enumerate(model_rows)]
    metas = [r.get("mcp_metadata") or r.get("parsed") or {} for r in model_rows]
    descriptions = [m.get("description", "") for m in metas]
    if not any(descriptions):
        descriptions = [
            " ".join(str(m.get(k, "")) for k in ("background", "scene") if m.get(k)).strip()
            for m in metas
        ]

    assessment = ImageAssessment(image=image, models=models)
    field_ratios: List[float] = []

    for fld in fields:
        values = {models[i]: metas[i].get(fld, "") for i in range(len(models))}
        if not any(is_informative(v) for v in values.values()):
            continue
        result = field_consensus(values, long_text=(fld in LONG_TEXT_FIELDS))
        assessment.field_results[fld] = result
        field_ratios.append(result["agreement_ratio"])

    assessment.description_similarity = round(pairwise_text_similarity(descriptions), 3)
    assessment.structured_agreement = round(
        sum(field_ratios) / len(field_ratios) if field_ratios else 1.0, 3
    )
    assessment.overall_agreement = round(
        0.6 * assessment.structured_agreement + 0.4 * assessment.description_similarity, 3
    )

    for fld, res in assessment.field_results.items():
        if res["status"] in ("split", "mixed") and res.get("unique_values", 1) > 1:
            vals = ", ".join(f"{mid}={repr(v)}" for mid, v in res["values_by_model"].items() if is_informative(v))
            if vals:
                assessment.divergences.append(f"{fld}: {vals}")
        elif res["status"] == "majority" and res.get("unique_values", 1) > 1:
            minority = [
                f"{mid}={repr(v)}"
                for mid, v in res["values_by_model"].items()
                if is_informative(v) and normalize_text(v) != res.get("consensus_value")
            ]
            if minority:
                assessment.divergences.append(
                    f"{fld} (majority={res.get('consensus_value')!r}; outliers: {', '.join(minority)})"
                )

    if assessment.description_similarity < 0.45:
        assessment.divergences.append(
            f"description text similarity low ({assessment.description_similarity:.2f})"
        )

    if assessment.overall_agreement >= 0.85:
        assessment.verdict = "high_agreement"
    elif assessment.overall_agreement >= 0.6:
        assessment.verdict = "partial_agreement"
    elif assessment.overall_agreement >= 0.35:
        assessment.verdict = "mixed"
    else:
        assessment.verdict = "high_divergence"

    return assessment


def narrative_summary(assessments: List[ImageAssessment]) -> str:
    """Optional LLM narrative; falls back to template if no API key."""
    bullet_lines = []
    for a in assessments:
        bullet_lines.append(
            f"- {Path(a.image).name}: overall={a.overall_agreement}, "
            f"structured={a.structured_agreement}, description={a.description_similarity}, "
            f"verdict={a.verdict}"
        )
    template = (
        "Multi-model agreement summary:\n"
        + "\n".join(bullet_lines)
        + "\n\nInterpretation: structured fields (species/scene/time) show whether models "
        "label the same organism and context; description similarity captures whether prose "
        "says the same thing in different words."
    )

    prompt = (
        "You are teaching a class on vision-language models for agricultural metadata.\n"
        "Summarize in 4-6 sentences how much these models agree vs diverge, "
        "what fields conflict most (species names, scene, etc.), and what that implies "
        "for using VLMs as metadata generators.\n\n"
        + template
    )

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            return (resp.choices[0].message.content or template).strip()
        except Exception:
            pass

    if os.environ.get("GOOGLE_API_KEY"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
            return (model.generate_content(prompt).text or template).strip()
        except Exception:
            pass

    return template


def print_report(assessments: List[ImageAssessment], narrative: str = "") -> None:
    print("=" * 72)
    print("MULTI-MODEL AGREEMENT REPORT")
    print("=" * 72)
    for a in assessments:
        print(f"\nImage: {a.image}")
        print(f"Models ({len(a.models)}): {', '.join(a.models)}")
        print(
            f"Overall agreement: {a.overall_agreement:.2f}  "
            f"[structured {a.structured_agreement:.2f} | description {a.description_similarity:.2f}]"
        )
        print(f"Verdict: {a.verdict.replace('_', ' ')}")

        print("\n  Field consensus:")
        for fld in sorted(a.field_results.keys()):
            if fld.startswith("_"):
                continue
            res = a.field_results.get(fld)
            if not res:
                continue
            informative_any = any(is_informative(v) for v in res["values_by_model"].values())
            if not informative_any and res["status"] != "split":
                continue
            print(
                f"    - {fld}: {res['status']} "
                f"(agreement={res['agreement_ratio']:.2f}, unique={res['unique_values']})"
            )
            for mid, val in res["values_by_model"].items():
                if is_informative(val):
                    print(f"        {mid}: {val}")

        if a.divergences:
            print("\n  Divergences:")
            for d in a.divergences:
                print(f"    * {d}")

    if len(assessments) > 1:
        avg = sum(a.overall_agreement for a in assessments) / len(assessments)
        print("\n" + "-" * 72)
        print(f"Batch mean overall agreement: {avg:.2f} across {len(assessments)} image(s)")
        # Verdict histogram for classroom summary
        verdicts = Counter(a.verdict for a in assessments)
        print("Verdict counts:", dict(verdicts))

    if narrative:
        print("\n" + "=" * 72)
        print("NARRATIVE ASSESSMENT")
        print("=" * 72)
        print(narrative)


def print_semantic_summary(metrics_summary: dict) -> None:
    attrs = metrics_summary.get("attributes") or {}
    if not attrs:
        return
    print("\n" + "=" * 72)
    print("SEMANTIC SIMILARITY (pairwise means + agreement tiers)")
    print("=" * 72)
    sbert_on = metrics_summary.get("sbert_enabled", False)
    thresholds = metrics_summary.get("agreement_thresholds") or {}
    if thresholds:
        print(f"Thresholds: high≥{thresholds.get('high')}, partial≥{thresholds.get('partial')}")
    print(f"SBERT: {'on' if sbert_on else 'off (TF-IDF only)'}")
    for attr, info in sorted(attrs.items()):
        sbert = info.get("mean_sbert_pairwise")
        tfidf = info.get("mean_tfidf_pairwise")
        parts = []
        if sbert_on and sbert is not None:
            parts.append(f"SBERT={sbert:.4f}")
        if tfidf is not None:
            parts.append(f"TF-IDF={tfidf:.4f}")
        tier_counts = info.get("tier_counts") or {}
        if tier_counts:
            tiers = ", ".join(f"{k}={v}" for k, v in sorted(tier_counts.items()))
            parts.append(f"tiers({tiers})")
        print(f"  {attr}: {', '.join(parts)}")
    stats = metrics_summary.get("auto_trust_stats") or {}
    if stats:
        print(
            f"\nAuto-trust (≥{metrics_summary.get('auto_trust_min_attributes', 4)} attributes "
            f"high/absent): {stats.get('auto_trusted_images', 0)}/{stats.get('total_images', 0)} images "
            f"({100 * stats.get('auto_trust_rate', 0):.1f}%)"
        )
    print(f"\nPer-image metrics: {metrics_summary.get('output_dir')}")


def print_auto_trust_summary(metrics_summary: dict, limit: int = 5) -> None:
    image_trust = metrics_summary.get("image_trust") or {}
    if not image_trust:
        return
    auto = [img for img, info in image_trust.items() if info.get("auto_trust")]
    review = [img for img, info in image_trust.items() if info.get("needs_review")]
    print("\n" + "=" * 72)
    print("AUTO-TRUST SUMMARY (semantic agreement → safe to use any model output)")
    print("=" * 72)
    print(f"Auto-trusted: {len(auto)} | Needs review: {len(review)} | Total: {len(image_trust)}")
    if auto:
        print(f"\nSample auto-trusted images (up to {limit}):")
        for img in auto[:limit]:
            info = image_trust[img]
            attrs = info.get("attributes") or {}
            high = [a for a, d in attrs.items() if d.get("semantic_agreement") == "high"]
            absent = [a for a, d in attrs.items() if d.get("semantic_agreement") == "absent"]
            print(f"  {img}: high={high}, absent={absent}")
    if review:
        print(f"\nSample needs-review images (up to {limit}):")
        for img in review[:limit]:
            info = image_trust[img]
            low = [
                f"{a}({d.get('semantic_agreement')})"
                for a, d in (info.get("attributes") or {}).items()
                if d.get("semantic_agreement") in ("low", "incomplete", "partial")
            ]
            print(f"  {img}: {', '.join(low)}")


def default_metrics_dir(input_path: Optional[Path], output_path: Optional[str]) -> Path:
    if output_path:
        out = Path(output_path)
        return out.parent / f"{out.stem.replace('_report', '')}_metrics"
    if input_path:
        return input_path.parent / f"{input_path.stem}_metrics"
    return SCRIPT_DIR / "metrics_output"


def run_semantic_metrics(
    *,
    csv_path: Optional[Path],
    groups: Dict[str, List[dict]],
    metrics_dir: Path,
    limit: int,
    use_sbert: bool,
    high_threshold: Optional[float] = None,
    partial_threshold: Optional[float] = None,
    min_trusted_attributes: int = 4,
) -> dict:
    from add_evaluation_metrics import process_model_groups, process_wide_csv, sbert_enabled, configure_hf_cache

    configure_hf_cache(os.environ.get("SBERT_CACHE", ""))

    if use_sbert and not sbert_enabled():
        print("WARNING: sentence-transformers/PyTorch not available; semantic metrics use TF-IDF only.")
        use_sbert = False

    print("\n" + "=" * 72)
    print("Computing semantic similarity metrics...")
    print("=" * 72)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "high_threshold": high_threshold,
        "partial_threshold": partial_threshold,
        "min_trusted_attributes": min_trusted_attributes,
    }
    if csv_path:
        return process_wide_csv(csv_path, metrics_dir, limit=limit, use_sbert=use_sbert, **kwargs)
    return process_model_groups(groups, metrics_dir, use_sbert=use_sbert, **kwargs)


def merge_image_trust_into_assessments(
    assessments: List[ImageAssessment],
    image_trust: Dict[str, dict],
) -> None:
    """Attach semantic trust rollup to each assessment (mutates in place)."""
    for a in assessments:
        key = a.image
        alt = Path(a.image).name
        trust = image_trust.get(key) or image_trust.get(alt) or {}
        a.field_results.setdefault("_semantic_trust", {})
        if trust:
            a.field_results["_semantic_trust"] = {
                "auto_trust": trust.get("auto_trust", False),
                "trusted_attribute_count": trust.get("trusted_attribute_count", 0),
                "total_attributes": trust.get("total_attributes", 0),
                "needs_review": trust.get("needs_review", True),
                "attributes": trust.get("attributes", {}),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess agreement among multi-model MCP outputs.")
    parser.add_argument("--input", action="append", default=[], help="Comparison JSONL file(s); glob ok")
    parser.add_argument("--input-dir", help="Directory containing *_comparison.jsonl files")
    parser.add_argument(
        "--csv",
        help="Wide comparison CSV (e.g. coyote_metadata_comparison.csv with 'Field (Model)' columns)",
    )
    parser.add_argument("--species-hint", default="", help="Species label for CSV rows (e.g. coyote)")
    parser.add_argument("--limit", type=int, default=0, help="Max images to process from CSV")
    parser.add_argument("--output", help="Write JSON report to this path")
    parser.add_argument("--narrative", action="store_true", help="Add LLM narrative (optional API key)")
    parser.add_argument(
        "--semantic-metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute SBERT/TF-IDF pairwise metrics (default: on for --csv, off for JSONL)",
    )
    parser.add_argument(
        "--metrics-dir",
        help="Directory for per-attribute metric CSVs (default: derived from --output or CSV name)",
    )
    parser.add_argument(
        "--no-sbert",
        action="store_true",
        help="Skip SBERT in semantic metrics (TF-IDF only; no torch required)",
    )
    parser.add_argument(
        "--semantic-high",
        type=float,
        default=None,
        help="Min pairwise cosine for 'high' agreement (default: 0.72 SBERT / 0.12 TF-IDF)",
    )
    parser.add_argument(
        "--semantic-partial",
        type=float,
        default=None,
        help="Min pairwise cosine for 'partial' agreement (default: 0.55 SBERT / 0.06 TF-IDF)",
    )
    parser.add_argument(
        "--auto-trust-min-attrs",
        type=int,
        default=4,
        help="Min attributes (high or absent) to auto-trust an image (default: 4 of 5)",
    )
    parser.add_argument(
        "--hf-cache",
        help="HF/SBERT model cache dir (default: SBERT_CACHE, HF_HOME, or /projects/.../hf_cache)",
    )
    args = parser.parse_args()

    if args.hf_cache:
        os.environ["SBERT_CACHE"] = args.hf_cache
    if args.csv and not args.no_sbert:
        from add_evaluation_metrics import configure_hf_cache
        cache = configure_hf_cache(os.environ.get("SBERT_CACHE", ""))
        if cache:
            print(f"Using HF cache: {cache}")

    csv_path: Optional[Path] = None
    structured_fields = list(STRUCTURED_FIELDS_DEFAULT)
    input_label = ""

    if args.csv:
        csv_path = Path(args.csv).resolve()
        if not csv_path.is_file():
            raise SystemExit(f"CSV not found: {csv_path}")
        groups = load_wide_csv(
            csv_path,
            species_hint=args.species_hint,
            limit=args.limit,
        )
        structured_fields = (
            ["species", "weather", "lighting", "time", "scene", "background"]
            if args.species_hint
            else ["weather", "lighting", "time", "scene", "background"]
        )
        input_label = str(csv_path)
        paths: List[Path] = []
    else:
        paths = []
        for pattern in args.input:
            paths.extend(Path(p) for p in glob.glob(pattern))
        if args.input_dir:
            paths.extend(sorted(Path(args.input_dir).glob("*_comparison.jsonl")))
            paths.extend(sorted(Path(args.input_dir).glob("*.jsonl")))
        paths = sorted(set(p.resolve() for p in paths if p.is_file()))

        if not paths:
            demo = SCRIPT_DIR / "samples" / "demo_comparison.json"
            if demo.is_file() and not args.input and not args.input_dir:
                data = json.loads(demo.read_text(encoding="utf-8"))
                rows = []
                for row in data.get("default", []):
                    meta = row.get("parsed") or {}
                    rows.append({
                        "image": "demo_image.jpg",
                        "model_id": row["model_id"],
                        "provider": row.get("provider"),
                        "mcp_metadata": meta,
                        "parsed": meta,
                        "error": row.get("error"),
                    })
                groups = {"demo_image.jpg": rows}
                input_label = str(demo)
            else:
                raise SystemExit("No input found. Pass --csv, --input, or --input-dir.")
        else:
            rows = load_rows(paths)
            if not rows:
                raise SystemExit("Input files contained no rows.")
            groups = group_by_image(rows)
            input_label = ", ".join(str(p) for p in paths)

    assessments = [
        assess_image(img, grp, structured_fields=structured_fields)
        for img, grp in sorted(groups.items())
    ]
    narrative = narrative_summary(assessments) if args.narrative else ""

    run_metrics = args.semantic_metrics if args.semantic_metrics is not None else bool(args.csv)

    # Batch-level field agreement (mean across images)
    field_means: Dict[str, float] = {}
    for fld in structured_fields:
        scores = [a.field_results[fld]["agreement_ratio"] for a in assessments if fld in a.field_results]
        if scores:
            field_means[fld] = round(sum(scores) / len(scores), 3)

    print_report(assessments, narrative)
    if field_means:
        print("\nMean agreement by field (all images):")
        for fld, score in sorted(field_means.items(), key=lambda x: -x[1]):
            print(f"  {fld}: {score:.2f}")

    metrics_summary: dict = {}
    if run_metrics:
        metrics_dir = Path(args.metrics_dir).resolve() if args.metrics_dir else default_metrics_dir(
            csv_path, args.output
        )
        metrics_summary = run_semantic_metrics(
            csv_path=csv_path,
            groups=groups,
            metrics_dir=metrics_dir,
            limit=args.limit,
            use_sbert=not args.no_sbert,
            high_threshold=args.semantic_high,
            partial_threshold=args.semantic_partial,
            min_trusted_attributes=args.auto_trust_min_attrs,
        )
        merge_image_trust_into_assessments(
            assessments, metrics_summary.get("image_trust") or {},
        )
        print_semantic_summary(metrics_summary)
        print_auto_trust_summary(metrics_summary)

    report = {
        "inputs": input_label,
        "images": len(assessments),
        "models": assessments[0].models if assessments else [],
        "mean_overall_agreement": round(
            sum(a.overall_agreement for a in assessments) / len(assessments), 3
        ) if assessments else 0,
        "mean_field_agreement": field_means,
        "semantic_metrics": metrics_summary or None,
        "auto_trust_stats": metrics_summary.get("auto_trust_stats") if metrics_summary else None,
        "assessments": [
            {
                "image": a.image,
                "models": a.models,
                "overall_agreement": a.overall_agreement,
                "structured_agreement": a.structured_agreement,
                "description_similarity": a.description_similarity,
                "verdict": a.verdict,
                "field_results": {
                    k: v for k, v in a.field_results.items() if not k.startswith("_")
                },
                "divergences": a.divergences,
                "semantic_trust": a.field_results.get("_semantic_trust"),
            }
            for a in assessments
        ],
        "narrative": narrative,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
