#!/usr/bin/env python3
"""
Add semantic (SBERT) and lexical (TF-IDF) evaluation metrics to multi-model attribute CSVs.

Supports two input formats:

1) Wide comparison CSV (classroom demo), e.g. coyote_metadata_comparison.csv:
   Image ID, Weather (GPT-4o), Background (GPT-4o), ..., Setting (Llama)

2) Long-form attribute CSVs (original pipeline), one file per attribute:
   image_id, gpt4o, 4.5_gpt, Llama4

Metrics per attribute × image row:
  - mean/min pairwise semantic similarity (SBERT or TF-IDF)
  - semantic_agreement tier: high | partial | low | absent | incomplete
  - trusted_for_auto_use: True when models paraphrase the same thing (or all absent)
  - SBERT pairwise cosine between model pairs
  - TF-IDF pairwise cosine between model pairs
  - coverage, length, combined score (best_model is optional/legacy)

Examples:
  # Coyote wide CSV (recommended for classroom):
  python add_evaluation_metrics.py \\
    --wide-csv classroom_demo/coyote_metadata_comparison.csv \\
    --output-dir classroom_demo/coyote_metrics

  # Legacy attributes/ tree on Taiga:
  python add_evaluation_metrics.py --attributes-dir /path/to/attributes

Install:
  pip install sentence-transformers torch scikit-learn pandas numpy
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = Path(__file__).resolve().parent

# Legacy Taiga defaults (override with --attributes-dir)
DEFAULT_PROJECT_ROOT = "/taiga/ncsa/radiant/bbgp/rgpu02/owodd"
DEFAULT_ATTRIBUTES_DIR = os.path.join(DEFAULT_PROJECT_ROOT, "attributes")
CATEGORIES = ["animals", "plants", "pests"]
ATTRIBUTES = ["background", "setting", "lighting", "time", "weather"]

LEGACY_MODEL_MAPPING = {
    "4.5_gpt": "gpt4_5",
    "gpt4o": "gpt4o",
    "Llama4": "llama4",
    "Qwen2.5VL": "qwen2_5vl",
}
LEGACY_EVAL_MODELS = ["4.5_gpt", "gpt4o", "Llama4"]

WIDE_FIELD_MAP = {
    "weather": "weather",
    "background": "background",
    "lighting": "lighting",
    "time of day": "time",
    "setting": "setting",
}

WEIGHT_SBERT = 0.4
WEIGHT_TFIDF = 0.3
WEIGHT_COVERAGE = 0.2
WEIGHT_LENGTH = 0.1

# Pairwise semantic agreement tiers (min cosine across all model pairs)
SBERT_HIGH_THRESHOLD = 0.72
SBERT_PARTIAL_THRESHOLD = 0.55
TFIDF_HIGH_THRESHOLD = 0.12
TFIDF_PARTIAL_THRESHOLD = 0.06
AUTO_TRUST_MIN_ATTRIBUTES = 4
SEMANTIC_PASS_TIERS = frozenset({"high", "absent"})
ATTR_REPORT_ALIAS = {"setting": "scene"}

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
_SBERT_MODEL = None
_SBERT_AVAILABLE: Optional[bool] = None

DEFAULT_HF_CACHE_CANDIDATES = (
    "/projects/bbqj/alucic2/hf_cache",
    "/taiga/ncsa/radiant/bbgp/rgpu02/owodd/alucic2/hf_cache",
)


def configure_hf_cache(preferred: str = "") -> str:
    """
    Use project/Taiga cache instead of ~/.cache (home quota on Delta login nodes).
    Set SBERT_CACHE or HF_HOME before running to override.
    """
    cache = (preferred or os.environ.get("SBERT_CACHE") or os.environ.get("HF_HOME") or "").strip()
    if not cache:
        for candidate in DEFAULT_HF_CACHE_CANDIDATES:
            try:
                Path(candidate).mkdir(parents=True, exist_ok=True)
                cache = candidate
                break
            except OSError:
                continue
    if not cache:
        return ""
    cache_path = Path(cache)
    hub = cache_path / "hub"
    cache_path.mkdir(parents=True, exist_ok=True)
    hub.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_path)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_path))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_path))
    return str(cache_path)


def sbert_enabled() -> bool:
    global _SBERT_AVAILABLE
    if _SBERT_AVAILABLE is None:
        try:
            import sentence_transformers  # noqa: F401
            _SBERT_AVAILABLE = True
        except Exception:
            _SBERT_AVAILABLE = False
    return bool(_SBERT_AVAILABLE)


def get_sbert_model():
    global _SBERT_MODEL
    if not sbert_enabled():
        raise RuntimeError(
            "SBERT unavailable. Install sentence-transformers + PyTorch>=2.4, "
            "or run with --no-sbert for TF-IDF-only metrics."
        )
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        cache = configure_hf_cache()
        print(f"Loading SBERT model '{SBERT_MODEL_NAME}' ...")
        if cache:
            print(f"  HF/SBERT cache: {cache}")
        kwargs = {"cache_folder": cache} if cache else {}
        _SBERT_MODEL = SentenceTransformer(SBERT_MODEL_NAME, **kwargs)
        print("SBERT model loaded.")
    return _SBERT_MODEL


def slug_model(name: str) -> str:
    """GPT-4o -> gpt_4o, Llama -> llama"""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return s or "model"


def clean_text(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip()
    if text.lower() in {"n/a", "nan", "none", "unknown", "not visible", "not visible.", ""}:
        return ""
    return re.sub(r"\s+", " ", text)


def compute_length(text) -> int:
    text = clean_text(text)
    return len(text.split()) if text else 0


def compute_coverage(text) -> float:
    return 1.0 if clean_text(text) else 0.0


def normalize_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    min_v, max_v = series.min(), series.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_v) / (max_v - min_v)


def internal_name(col: str, mapping: Dict[str, str]) -> str:
    return mapping.get(col, slug_model(col))


def calculate_tfidf_per_row(df: pd.DataFrame, model_cols: List[str], mapping: Dict[str, str]) -> pd.DataFrame:
    corpus = []
    for col in model_cols:
        corpus.extend([t for t in df[col].astype(str).apply(clean_text).tolist() if t])

    if not corpus:
        for col in model_cols:
            df[f"{internal_name(col, mapping)}_tfidf"] = 0.0
        return df

    vectorizer = TfidfVectorizer(
        min_df=1,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
        token_pattern=r"\b[a-z]+\b",
    )
    try:
        vectorizer.fit(corpus)
    except Exception:
        for col in model_cols:
            df[f"{internal_name(col, mapping)}_tfidf"] = 0.0
        return df

    for col in model_cols:
        internal = internal_name(col, mapping)
        sums = []
        for raw in df[col].astype(str).apply(clean_text):
            if not raw:
                sums.append(0.0)
                continue
            try:
                sums.append(float(vectorizer.transform([raw]).sum()))
            except Exception:
                sums.append(0.0)
        df[f"{internal}_tfidf"] = sums
    return df


def calculate_sbert_similarity_per_row(
    df: pd.DataFrame, model_cols: List[str], mapping: Dict[str, str], use_sbert: bool = True
) -> pd.DataFrame:
    for col in model_cols:
        internal = internal_name(col, mapping)
        df[f"{internal}_sbert_score"] = 0.0
        df[f"{internal}_sbert_score_norm"] = 0.5

    if not use_sbert:
        return df

    from sentence_transformers import util

    model = get_sbert_model()
    for col in model_cols:
        df[col] = df[col].astype(str).apply(clean_text)

    num_rows = len(df)
    if num_rows == 0:
        return df

    embeddings = {col: model.encode(df[col].tolist(), convert_to_tensor=True) for col in model_cols}
    for col in model_cols:
        df[f"{internal_name(col, mapping)}_sbert_score"] = 0.0

    for i in range(num_rows):
        valid = [embeddings[c][i] for c in model_cols if df.loc[i, c]]
        centroid = sum(valid) / len(valid) if valid else None
        for col in model_cols:
            internal = internal_name(col, mapping)
            if centroid is None:
                score = 0.0
            else:
                score = util.cos_sim(embeddings[col][i], centroid).item()
            df.at[i, f"{internal}_sbert_score"] = score

    for col in model_cols:
        internal = internal_name(col, mapping)
        df[f"{internal}_sbert_score_norm"] = normalize_series(df[f"{internal}_sbert_score"])
    return df


def _pair_col(prefix: str, m1: str, m2: str, mapping: Dict[str, str]) -> str:
    return f"{prefix}_{internal_name(m1, mapping)}_{internal_name(m2, mapping)}"


def calculate_tfidf_pairwise(df: pd.DataFrame, model_cols: List[str], mapping: Dict[str, str]) -> pd.DataFrame:
    all_texts = []
    for col in model_cols:
        all_texts.extend([t for t in df[col].astype(str).apply(clean_text).tolist() if t])

    pairs = list(combinations(model_cols, 2))
    for m1, m2 in pairs:
        df[_pair_col("cos", m1, m2, mapping)] = 0.0

    if not all_texts:
        return df

    vectorizer = TfidfVectorizer(
        min_df=1,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
        token_pattern=r"\b[a-z]+\b",
    )
    try:
        vectorizer.fit(all_texts)
    except Exception:
        return df

    def vec(text):
        t = clean_text(text)
        if not t:
            return None
        try:
            return vectorizer.transform([t])
        except Exception:
            return None

    for i in range(len(df)):
        vecs = {c: vec(df.loc[i, c]) for c in model_cols}
        for m1, m2 in pairs:
            v1, v2 = vecs.get(m1), vecs.get(m2)
            sim = float(cosine_similarity(v1, v2)[0][0]) if v1 is not None and v2 is not None else 0.0
            df.at[i, _pair_col("cos", m1, m2, mapping)] = sim
    return df


def calculate_sbert_pairwise(
    df: pd.DataFrame, model_cols: List[str], mapping: Dict[str, str], use_sbert: bool = True
) -> pd.DataFrame:
    pairs = list(combinations(model_cols, 2))
    for m1, m2 in pairs:
        df[_pair_col("sbert_cos", m1, m2, mapping)] = 0.0

    if not use_sbert or len(df) == 0:
        return df

    from sentence_transformers import util

    model = get_sbert_model()
    for col in model_cols:
        df[col] = df[col].astype(str).apply(clean_text)

    embeddings = {col: model.encode(df[col].tolist(), convert_to_tensor=True) for col in model_cols}
    for i in range(len(df)):
        emb_map = {c: embeddings[c][i] if df.loc[i, c] else None for c in model_cols}
        for m1, m2 in pairs:
            e1, e2 = emb_map.get(m1), emb_map.get(m2)
            sim = util.cos_sim(e1, e2).item() if e1 is not None and e2 is not None else 0.0
            df.at[i, _pair_col("sbert_cos", m1, m2, mapping)] = float(sim)
    return df


def _pairwise_sim_columns(df: pd.DataFrame, use_sbert: bool) -> List[str]:
    if use_sbert:
        cols = [c for c in df.columns if c.startswith("sbert_cos_")]
        if cols:
            return cols
    return [c for c in df.columns if c.startswith("cos_")]


def classify_semantic_agreement(
    min_sim: float,
    coverages: Sequence[float],
    *,
    use_sbert: bool,
    high_threshold: float,
    partial_threshold: float,
) -> Tuple[str, bool]:
    """Return (tier, trusted_for_auto_use). trusted = high semantic match or unanimous absent."""
    n_present = sum(1 for c in coverages if c >= 1.0)
    if n_present == 0:
        return "absent", True
    if n_present < len(coverages):
        return "incomplete", False
    if min_sim >= high_threshold:
        return "high", True
    if min_sim >= partial_threshold:
        return "partial", False
    return "low", False


def add_semantic_agreement_columns(
    df: pd.DataFrame,
    model_cols: List[str],
    mapping: Dict[str, str],
    use_sbert: bool,
    *,
    high_threshold: Optional[float] = None,
    partial_threshold: Optional[float] = None,
) -> pd.DataFrame:
    sim_cols = _pairwise_sim_columns(df, use_sbert)
    high = high_threshold if high_threshold is not None else (
        SBERT_HIGH_THRESHOLD if use_sbert else TFIDF_HIGH_THRESHOLD
    )
    partial = partial_threshold if partial_threshold is not None else (
        SBERT_PARTIAL_THRESHOLD if use_sbert else TFIDF_PARTIAL_THRESHOLD
    )

    if not sim_cols:
        df["mean_pairwise_semantic"] = 0.0
        df["min_pairwise_semantic"] = 0.0
        df["semantic_agreement"] = "low"
        df["trusted_for_auto_use"] = False
        return df

    df["mean_pairwise_semantic"] = df[sim_cols].mean(axis=1).round(4)
    df["min_pairwise_semantic"] = df[sim_cols].min(axis=1).round(4)

    tiers: List[str] = []
    trusted: List[bool] = []
    for i in range(len(df)):
        coverages = [float(df.loc[i, f"{internal_name(c, mapping)}_coverage"]) for c in model_cols]
        tier, ok = classify_semantic_agreement(
            float(df.loc[i, "min_pairwise_semantic"]),
            coverages,
            use_sbert=use_sbert,
            high_threshold=high,
            partial_threshold=partial,
        )
        tiers.append(tier)
        trusted.append(ok)
    df["semantic_agreement"] = tiers
    df["trusted_for_auto_use"] = trusted
    return df


def summarize_agreement_tiers(df: pd.DataFrame) -> dict:
    counts = df["semantic_agreement"].value_counts().to_dict() if "semantic_agreement" in df.columns else {}
    trusted = int(df["trusted_for_auto_use"].sum()) if "trusted_for_auto_use" in df.columns else 0
    return {
        "tier_counts": counts,
        "trusted_rows": trusted,
        "total_rows": len(df),
        "mean_min_pairwise": round(float(df["min_pairwise_semantic"].mean()), 4)
        if "min_pairwise_semantic" in df.columns else None,
    }


def build_image_trust_summary(
    scored_frames: Dict[str, pd.DataFrame],
    *,
    min_trusted_attributes: int = AUTO_TRUST_MIN_ATTRIBUTES,
) -> Dict[str, dict]:
    """Per-image rollup: which attributes are semantically trusted for auto-use."""
    by_image: Dict[str, Dict[str, dict]] = defaultdict(dict)

    for attr, df in scored_frames.items():
        report_attr = ATTR_REPORT_ALIAS.get(attr, attr)
        if "image_id" not in df.columns:
            continue
        for _, row in df.iterrows():
            image_id = str(row["image_id"])
            by_image[image_id][report_attr] = {
                "mean_pairwise_semantic": float(row.get("mean_pairwise_semantic", 0)),
                "min_pairwise_semantic": float(row.get("min_pairwise_semantic", 0)),
                "semantic_agreement": str(row.get("semantic_agreement", "low")),
                "trusted_for_auto_use": bool(row.get("trusted_for_auto_use", False)),
            }

    summary: Dict[str, dict] = {}
    for image_id, attrs in sorted(by_image.items()):
        pass_count = sum(1 for a in attrs.values() if a["semantic_agreement"] in SEMANTIC_PASS_TIERS)
        summary[image_id] = {
            "attributes": attrs,
            "trusted_attribute_count": pass_count,
            "total_attributes": len(attrs),
            "auto_trust": pass_count >= min_trusted_attributes,
            "needs_review": any(a["semantic_agreement"] in ("low", "incomplete") for a in attrs.values()),
        }
    return summary


def add_metrics_to_dataframe(
    df: pd.DataFrame,
    model_cols: List[str],
    mapping: Dict[str, str],
    use_sbert: bool = True,
    *,
    high_threshold: Optional[float] = None,
    partial_threshold: Optional[float] = None,
) -> pd.DataFrame:
    if len(model_cols) < 2:
        raise ValueError(f"Need at least 2 model columns, got {model_cols}")

    for col in model_cols:
        internal = internal_name(col, mapping)
        df[f"{internal}_length"] = df[col].apply(compute_length)
        df[f"{internal}_coverage"] = df[col].apply(compute_coverage)

    df = calculate_tfidf_per_row(df, model_cols, mapping)
    df = calculate_sbert_similarity_per_row(df, model_cols, mapping, use_sbert=use_sbert)
    df = calculate_sbert_pairwise(df, model_cols, mapping, use_sbert=use_sbert)
    df = calculate_tfidf_pairwise(df, model_cols, mapping)

    sbert_weight = WEIGHT_SBERT if use_sbert else 0.0
    tfidf_weight = WEIGHT_TFIDF + (WEIGHT_SBERT if not use_sbert else 0.0)
    score_cols = []
    for col in model_cols:
        internal = internal_name(col, mapping)
        df[f"{internal}_length_norm"] = normalize_series(df[f"{internal}_length"])
        df[f"{internal}_tfidf_norm"] = normalize_series(df[f"{internal}_tfidf"])
        sbert_norm = f"{internal}_sbert_score_norm"
        if sbert_norm not in df.columns:
            df[sbert_norm] = 0.5
        df[f"{internal}_score"] = (
            sbert_weight * df[sbert_norm]
            + tfidf_weight * df[f"{internal}_tfidf_norm"]
            + WEIGHT_COVERAGE * df[f"{internal}_coverage"]
            + WEIGHT_LENGTH * df[f"{internal}_length_norm"]
        )
        score_cols.append(f"{internal}_score")

    if score_cols:
        df["best_model"] = df[score_cols].idxmax(axis=1).str.replace("_score", "")

    df = add_semantic_agreement_columns(
        df, model_cols, mapping, use_sbert,
        high_threshold=high_threshold,
        partial_threshold=partial_threshold,
    )
    return df


def parse_wide_csv(path: Path) -> Tuple[str, Dict[str, Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        fieldnames = list(reader.fieldnames)

    image_col = ""
    for fn in fieldnames:
        if fn.strip().lower().replace("_", " ") in ("image id", "image", "filename", "file"):
            image_col = fn
            break
    if not image_col:
        image_col = fieldnames[0]

    pattern = re.compile(r"^(.+?)\s+\((.+)\)\s*$")
    models: Dict[str, Dict[str, str]] = {}
    for fn in fieldnames:
        if fn == image_col:
            continue
        m = pattern.match(fn.strip())
        if not m:
            continue
        raw_field, model = m.group(1).strip(), m.group(2).strip()
        attr = WIDE_FIELD_MAP.get(raw_field.lower())
        if attr:
            models.setdefault(model, {})[attr] = fn

    if not models:
        raise ValueError("Could not parse wide CSV columns like 'Weather (GPT-4o)'")
    return image_col, models


def wide_csv_to_attribute_frames(path: Path, limit: int = 0) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Convert wide CSV to {attribute: dataframe with image_id + model columns}."""
    image_col, model_cols = parse_wide_csv(path)
    model_names = sorted(model_cols.keys())
    mapping = {m: slug_model(m) for m in model_names}

    # Read all rows
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]

    frames: Dict[str, pd.DataFrame] = {}
    for attr in sorted({a for cols in model_cols.values() for a in cols}):
        records = []
        for row in rows:
            image_id = (row.get(image_col) or "").strip()
            if not image_id:
                continue
            rec = {"image_id": image_id}
            for model in model_names:
                col_name = model_cols[model].get(attr)
                rec[model] = (row.get(col_name) or "").strip() if col_name else ""
            records.append(rec)
        frames[attr] = pd.DataFrame(records)
    return frames, mapping


def summarize_pairwise_means(df: pd.DataFrame, model_cols: List[str], mapping: Dict[str, str]) -> dict:
    out = {"sbert_pairwise": {}, "tfidf_pairwise": {}}
    for m1, m2 in combinations(model_cols, 2):
        sbert_col = _pair_col("sbert_cos", m1, m2, mapping)
        cos_col = _pair_col("cos", m1, m2, mapping)
        if sbert_col in df.columns:
            out["sbert_pairwise"][f"{mapping.get(m1, m1)} vs {mapping.get(m2, m2)}"] = round(float(df[sbert_col].mean()), 4)
        if cos_col in df.columns:
            out["tfidf_pairwise"][f"{mapping.get(m1, m1)} vs {mapping.get(m2, m2)}"] = round(float(df[cos_col].mean()), 4)
    return out


def process_wide_csv(
    wide_path: Path,
    output_dir: Path,
    limit: int = 0,
    use_sbert: bool = True,
    **kwargs,
) -> dict:
    frames, mapping = wide_csv_to_attribute_frames(wide_path, limit=limit)
    return _write_attribute_metrics(frames, mapping, output_dir, wide_path, use_sbert=use_sbert, **kwargs)


JSONL_ATTR_FIELDS = {
    "weather": "weather",
    "background": "background",
    "lighting": "lighting",
    "time": "time",
    "scene": "scene",
}


def groups_to_attribute_frames(
    groups: Dict[str, List[dict]],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Build per-attribute frames from compare_model_agreement image groups."""
    model_names = sorted({
        row.get("model_id") or row.get("provider") or f"model_{i}"
        for grp in groups.values()
        for i, row in enumerate(grp)
    })
    mapping = {m: slug_model(m) for m in model_names}

    frames: Dict[str, pd.DataFrame] = {}
    for attr, meta_key in JSONL_ATTR_FIELDS.items():
        records = []
        for image_id, model_rows in sorted(groups.items()):
            rec: Dict[str, str] = {"image_id": image_id}
            for row in model_rows:
                model = row.get("model_id") or row.get("provider") or "unknown"
                meta = row.get("mcp_metadata") or row.get("parsed") or {}
                rec[model] = str(meta.get(meta_key, "") or "").strip()
            records.append(rec)
        frames[attr] = pd.DataFrame(records)
    return frames, mapping


def _write_attribute_metrics(
    frames: Dict[str, pd.DataFrame],
    mapping: Dict[str, str],
    output_dir: Path,
    input_label: Path | str,
    use_sbert: bool = True,
    *,
    high_threshold: Optional[float] = None,
    partial_threshold: Optional[float] = None,
    min_trusted_attributes: int = AUTO_TRUST_MIN_ATTRIBUTES,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    high = high_threshold if high_threshold is not None else (
        SBERT_HIGH_THRESHOLD if use_sbert else TFIDF_HIGH_THRESHOLD
    )
    partial = partial_threshold if partial_threshold is not None else (
        SBERT_PARTIAL_THRESHOLD if use_sbert else TFIDF_PARTIAL_THRESHOLD
    )
    summary = {
        "input": str(input_label),
        "output_dir": str(output_dir),
        "sbert_enabled": use_sbert,
        "sbert_model": SBERT_MODEL_NAME if use_sbert else None,
        "agreement_thresholds": {"high": high, "partial": partial},
        "auto_trust_min_attributes": min_trusted_attributes,
        "attributes": {},
    }
    scored_frames: Dict[str, pd.DataFrame] = {}

    for attr, df in frames.items():
        model_cols = [c for c in df.columns if c != "image_id"]
        if len(model_cols) < 2:
            continue
        print(f"  attribute: {attr} ({len(df)} rows, models: {model_cols})")
        scored = add_metrics_to_dataframe(
            df.copy(), model_cols, mapping, use_sbert=use_sbert,
            high_threshold=high, partial_threshold=partial,
        )
        scored_frames[attr] = scored
        out_path = output_dir / f"{attr}_metrics.csv"
        scored.to_csv(out_path, index=False)

        attr_summary = summarize_pairwise_means(scored, model_cols, mapping)
        attr_summary.update(summarize_agreement_tiers(scored))
        sbert_cols = [c for c in scored.columns if c.startswith("sbert_cos_")]
        if sbert_cols:
            attr_summary["mean_sbert_pairwise"] = round(float(scored[sbert_cols].mean().mean()), 4)
        cos_cols = [c for c in scored.columns if c.startswith("cos_")]
        if cos_cols:
            attr_summary["mean_tfidf_pairwise"] = round(float(scored[cos_cols].mean().mean()), 4)
        summary["attributes"][attr] = attr_summary

    image_trust = build_image_trust_summary(
        scored_frames, min_trusted_attributes=min_trusted_attributes,
    )
    summary["image_trust"] = image_trust
    auto_trusted = sum(1 for v in image_trust.values() if v.get("auto_trust"))
    summary["auto_trust_stats"] = {
        "auto_trusted_images": auto_trusted,
        "total_images": len(image_trust),
        "auto_trust_rate": round(auto_trusted / len(image_trust), 4) if image_trust else 0.0,
    }

    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    trust_path = output_dir / "image_trust_summary.json"
    trust_path.write_text(json.dumps(image_trust, indent=2) + "\n", encoding="utf-8")
    return summary


def process_model_groups(
    groups: Dict[str, List[dict]],
    output_dir: Path,
    use_sbert: bool = True,
    **kwargs,
) -> dict:
    """Semantic metrics from JSONL/demo groups (same shape as compare_model_agreement)."""
    frames, mapping = groups_to_attribute_frames(groups)
    return _write_attribute_metrics(frames, mapping, output_dir, "model_groups", use_sbert=use_sbert, **kwargs)


def add_metrics_to_csv(csv_path: Path, mapping: Optional[Dict[str, str]] = None) -> bool:
    """Legacy: long-form attribute CSV with columns like gpt4o, 4.5_gpt."""
    mapping = mapping or LEGACY_MODEL_MAPPING
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  ERROR reading {csv_path}: {exc}")
        return False

    available = [m for m in LEGACY_EVAL_MODELS if m in df.columns]
    if len(available) < 2:
        print(f"  WARNING: Not enough models in {csv_path} (found: {available})")
        return False

    df = add_metrics_to_dataframe(df, available, mapping)
    try:
        df.to_csv(csv_path, index=False)
        return True
    except Exception as exc:
        print(f"  ERROR saving {csv_path}: {exc}")
        return False


def process_attributes_tree(attributes_dir: Path) -> None:
    total = processed = 0
    for category in CATEGORIES:
        category_path = attributes_dir / category
        if not category_path.is_dir():
            continue
        print(f"\nCategory: {category}")
        for species_path in sorted(category_path.iterdir()):
            if not species_path.is_dir() or species_path.name in {"organized_species", "evaluation", "reports"}:
                continue
            print(f"  {category}/{species_path.name}")
            for attr in ATTRIBUTES:
                csv_path = species_path / f"{attr}.csv"
                if not csv_path.is_file():
                    continue
                total += 1
                if add_metrics_to_csv(csv_path):
                    processed += 1
    print(f"\nDone: {processed}/{total} files updated.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SBERT + TF-IDF metrics for multi-model attribute CSVs.")
    parser.add_argument(
        "--wide-csv",
        help="Wide comparison CSV (e.g. coyote_metadata_comparison.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "metrics_output"),
        help="Output directory for per-attribute metric CSVs (wide-csv mode)",
    )
    parser.add_argument(
        "--attributes-dir",
        default="",
        help=f"Legacy attributes/ tree (default Taiga: {DEFAULT_ATTRIBUTES_DIR})",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max rows from wide CSV")
    parser.add_argument(
        "--no-sbert",
        action="store_true",
        help="Skip SBERT (TF-IDF pairwise + coverage/length only; no torch required)",
    )
    parser.add_argument(
        "--hf-cache",
        help="HF/SBERT model cache dir (default: SBERT_CACHE, HF_HOME, or project/Taiga path)",
    )
    args = parser.parse_args()

    use_sbert = not args.no_sbert
    if use_sbert:
        cache = configure_hf_cache(args.hf_cache or os.environ.get("SBERT_CACHE", ""))
        if cache:
            print(f"HF/SBERT cache: {cache}")
    if use_sbert and not sbert_enabled():
        print("WARNING: sentence-transformers/PyTorch not available; falling back to TF-IDF only.")
        use_sbert = False

    print("=" * 70)
    print("Evaluation metrics (SBERT + TF-IDF)")
    print(f"  SBERT: {'on' if use_sbert else 'off (TF-IDF only)'}")
    print(f"  Weights: SBERT={WEIGHT_SBERT}, TF-IDF={WEIGHT_TFIDF}, "
          f"coverage={WEIGHT_COVERAGE}, length={WEIGHT_LENGTH}")
    print("=" * 70)

    if args.wide_csv:
        wide_path = Path(args.wide_csv).resolve()
        if not wide_path.is_file():
            raise SystemExit(f"File not found: {wide_path}")
        out_dir = Path(args.output_dir).resolve()
        print(f"\nProcessing wide CSV: {wide_path}")
        summary = process_wide_csv(wide_path, out_dir, limit=args.limit, use_sbert=use_sbert)
        print(f"\nWrote metrics to {out_dir}")
        print("Summary (mean pairwise semantic similarity by attribute):")
        for attr, info in summary.get("attributes", {}).items():
            sbert = info.get("mean_sbert_pairwise", "n/a")
            tfidf = info.get("mean_tfidf_pairwise", "n/a")
            print(f"  {attr}: SBERT={sbert}, TF-IDF={tfidf}")
        print(f"Full summary: {out_dir / 'metrics_summary.json'}")
        return

    attributes_dir = Path(args.attributes_dir or DEFAULT_ATTRIBUTES_DIR)
    if not attributes_dir.is_dir():
        raise SystemExit(
            "Provide --wide-csv for classroom demo, or --attributes-dir for the legacy tree.\n"
            f"  Example: python add_evaluation_metrics.py --wide-csv classroom_demo/coyote_metadata_comparison.csv"
        )
    process_attributes_tree(attributes_dir)


if __name__ == "__main__":
    main()
