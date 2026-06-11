#!/usr/bin/env python3
"""
Compare 2–3 vision-language APIs on the same image(s) and emit MCP-style metadata.

Designed for classroom demos:
  - Keys come ONLY from environment variables (never hard-code or commit keys).
  - --demo runs without any API keys using bundled sample outputs.
  - --models auto skips providers whose keys are missing.

Supported providers (enable any subset):
  - openai / gpt4o      OPENAI_API_KEY
  - azure / gpt4o       AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT
  - gemini              GOOGLE_API_KEY
  - claude              ANTHROPIC_API_KEY

Examples:
  # Instructor live demo (keys in env or .env):
  python classroom_vlm_comparison.py --image sample.jpg --models gpt4o,gemini,claude

  # Students without keys:
  python classroom_vlm_comparison.py --demo --image sample.jpg

  # Batch folder:
  python classroom_vlm_comparison.py --input-dir ./demo_images --models auto --output-dir ./out

Output:
  - comparison.jsonl   one row per (image, model) with raw + MCP metadata
  - mcp_<model>.json   MCP dataset snippet per model (for side-by-side review)
  - summary.txt        short text comparison

Classroom API-key guidance (for instructors):
  1. Do NOT paste keys into notebooks, GitHub, or Slack.
  2. Prefer a course VM / Taiga project dir with chmod 600 .env (instructor-only).
  3. Or run live yourself while students use --demo on the same images.
  4. Optional: university Azure OpenAI with per-student keys via IT.
  5. Rotate keys after the semester if shared on a lab machine.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = SCRIPT_DIR / "samples"

MCP_PROMPT = """Analyze this image for an agricultural / ecological metadata catalog (MCP format).

Return ONLY valid JSON (no markdown fences) with this structure:
{
  "description": "One or two sentences describing the main subject and setting.",
  "species": "common English name of the primary organism (animal, plant, pest, or crop)",
  "scientific_name": "Latin binomial if identifiable, else empty string",
  "common_name": "primary English common name if applicable",
  "scene": "short scene label (e.g. meadow, forest floor, garden, barn)",
  "time": "dawn|day|dusk|night|unclear",
  "season": "spring|summer|fall|winter|unknown",
  "weather": "visible weather or 'not visible'",
  "lighting": "lighting conditions",
  "action": "observable action/state of main subject (e.g. foraging, resting)",
  "date": "date if visible in image metadata, else 'unknown'",
  "confidence": "high|medium|low"
}

Use factual observations only. Use 'unclear', 'unknown', or 'not visible' when uncertain."""


@dataclass
class ModelResult:
    model_id: str
    provider: str
    raw_text: str
    parsed: Dict[str, Any]
    mcp_metadata: Dict[str, Any]
    mcp_assistant_text: str
    latency_sec: float
    error: Optional[str] = None


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


def encode_image_jpeg(path: Path, quality: int = 90) -> tuple[str, dict]:
    if Image is None:
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return b64, {"note": "PIL not installed; sent raw bytes as JPEG mime"}
    with Image.open(path) as img:
        if img.mode == "RGBA":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        info = {"original_size": img.size, "compressed_bytes": len(data)}
    return base64.b64encode(data).decode("ascii"), info


def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {"description": text}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError:
                pass
    return {"description": text}


def analysis_to_mcp_metadata(parsed: Dict[str, Any], image_path: Path, species_hint: str = "") -> Dict[str, Any]:
    meta = {
        "description": str(parsed.get("description") or "").strip(),
        "species": str(parsed.get("species") or species_hint or "").strip(),
        "scientific_name": str(parsed.get("scientific_name") or "").strip(),
        "common_name": str(parsed.get("common_name") or parsed.get("species") or "").strip(),
        "scene": str(parsed.get("scene") or "unclear").strip(),
        "time": str(parsed.get("time") or "unclear").strip(),
        "season": str(parsed.get("season") or "unknown").strip(),
        "weather": str(parsed.get("weather") or "not visible").strip(),
        "lighting": str(parsed.get("lighting") or "unclear").strip(),
        "action": str(parsed.get("action") or "unclear").strip(),
        "date": str(parsed.get("date") or "unknown").strip(),
        "original_filename": image_path.name,
        "original_id": image_path.stem,
        "confidence": str(parsed.get("confidence") or "").strip(),
    }
    return {k: v for k, v in meta.items() if v}


def build_mcp_assistant_text(metadata: Dict[str, Any]) -> str:
    desc = metadata.get("description", "")
    tag_keys = [
        ("common name", metadata.get("common_name")),
        ("scientific name", metadata.get("scientific_name")),
        ("species", metadata.get("species")),
        ("scene", metadata.get("scene")),
        ("time", metadata.get("time")),
        ("season", metadata.get("season")),
        ("weather", metadata.get("weather")),
        ("lighting", metadata.get("lighting")),
        ("action", metadata.get("action")),
        ("date", metadata.get("date")),
    ]
    lines = [f"- {k}: {v}" for k, v in tag_keys if v and str(v).lower() not in ("unknown", "unclear", "not visible", "")]
    parts = []
    if desc:
        parts.append(desc)
    if lines:
        parts.append("Attributes:\n" + "\n".join(lines))
    return "\n\n".join(parts).strip()


def call_openai(model: str, b64: str, prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        max_tokens=800,
    )
    return (resp.choices[0].message.content or "").strip()


def call_azure(b64: str, prompt: str) -> str:
    from openai import AzureOpenAI

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    if "/openai/" not in endpoint:
        endpoint = endpoint + "/"
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=endpoint,
        api_version=api_version,
    )
    resp = client.chat.completions.create(
        model=deployment,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        max_tokens=800,
    )
    return (resp.choices[0].message.content or "").strip()


def call_gemini(model: str, image_path: Path, prompt: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gm = genai.GenerativeModel(model)
    img = Image.open(image_path) if Image else image_path
    resp = gm.generate_content([prompt, img])
    return (resp.text or "").strip()


def call_claude(model: str, b64: str, prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model,
        max_tokens=800,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
            ],
        }],
    )
    parts = []
    for block in resp.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts).strip()


PROVIDERS: Dict[str, dict] = {
    "gpt4o": {
        "label": "OpenAI GPT-4o",
        "env": ["OPENAI_API_KEY"],
        "default_model": "gpt-4o",
    },
    "azure": {
        "label": "Azure OpenAI GPT-4o",
        "env": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
        "default_model": "gpt-4o",
    },
    "gemini": {
        "label": "Google Gemini",
        "env": ["GOOGLE_API_KEY"],
        "default_model": "gemini-2.0-flash",
    },
    "claude": {
        "label": "Anthropic Claude",
        "env": ["ANTHROPIC_API_KEY"],
        "default_model": "claude-3-5-sonnet-20241022",
    },
}


def provider_available(name: str) -> bool:
    cfg = PROVIDERS[name]
    return all(os.environ.get(k) for k in cfg["env"])


def resolve_models(request: str) -> List[str]:
    if request.strip().lower() == "auto":
        return [k for k in PROVIDERS if provider_available(k)]
    names = [m.strip().lower() for m in request.split(",") if m.strip()]
    out = []
    for n in names:
        if n in ("openai", "gpt-4o"):
            n = "gpt4o"
        if n not in PROVIDERS:
            raise SystemExit(f"Unknown model key: {n}. Choose from: {', '.join(PROVIDERS)}")
        out.append(n)
    return out


def run_provider(name: str, image_path: Path, b64: str, prompt: str, model_override: str = "") -> ModelResult:
    cfg = PROVIDERS[name]
    model_id = model_override or os.environ.get("GEMINI_MODEL", cfg["default_model"]) if name == "gemini" else (model_override or cfg["default_model"])
    t0 = time.time()
    try:
        if name == "gpt4o":
            raw = call_openai(model_id, b64, prompt)
        elif name == "azure":
            raw = call_azure(b64, prompt)
        elif name == "gemini":
            raw = call_gemini(model_id, image_path, prompt)
        elif name == "claude":
            raw = call_claude(model_id, b64, prompt)
        else:
            raise ValueError(f"unsupported provider {name}")
        parsed = extract_json(raw)
        err = None
    except Exception as exc:
        raw, parsed, err = "", {}, str(exc)
    meta = analysis_to_mcp_metadata(parsed, image_path)
    return ModelResult(
        model_id=f"{name}:{model_id}",
        provider=name,
        raw_text=raw,
        parsed=parsed,
        mcp_metadata=meta,
        mcp_assistant_text=build_mcp_assistant_text(meta),
        latency_sec=round(time.time() - t0, 2),
        error=err,
    )


def load_demo_results(image_path: Path) -> List[ModelResult]:
    sample_path = SAMPLES_DIR / "demo_comparison.json"
    if not sample_path.is_file():
        raise SystemExit(f"Demo sample missing: {sample_path}")
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    stem = image_path.stem
    rows = data.get("by_image", {}).get(stem) or data.get("default", [])
    results = []
    for row in rows:
        parsed = row.get("parsed") or extract_json(row.get("raw_text", ""))
        meta = analysis_to_mcp_metadata(parsed, image_path)
        results.append(ModelResult(
            model_id=row["model_id"],
            provider=row.get("provider", "demo"),
            raw_text=row.get("raw_text", json.dumps(parsed)),
            parsed=parsed,
            mcp_metadata=meta,
            mcp_assistant_text=build_mcp_assistant_text(meta),
            latency_sec=float(row.get("latency_sec", 0)),
            error=row.get("error"),
        ))
    return results


def write_outputs(image_path: Path, results: List[ModelResult], output_dir: Path, dataset_type: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = image_path.stem

    comparison_path = output_dir / f"{stem}_comparison.jsonl"
    with comparison_path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "image": str(image_path),
                "model_id": r.model_id,
                "provider": r.provider,
                "latency_sec": r.latency_sec,
                "error": r.error,
                "parsed": r.parsed,
                "mcp_metadata": r.mcp_metadata,
                "mcp_assistant_text": r.mcp_assistant_text,
                "raw_text": r.raw_text,
                "timestamp": stamp,
            }, ensure_ascii=False) + "\n")

    for r in results:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", r.model_id)
        mcp_doc = {
            "dataset_type": dataset_type,
            "description": f"Classroom demo output for {image_path.name} via {r.model_id}",
            "images": [{
                "id": f"{stem}_{safe}",
                "collection": r.mcp_metadata.get("species") or stem,
                "metadata": {
                    **r.mcp_metadata,
                    "mcp_id": f"{stem}_{safe}",
                    "source_model": r.model_id,
                },
            }],
        }
        out = output_dir / f"{stem}_mcp_{safe}.json"
        out.write_text(json.dumps(mcp_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_lines = [f"Image: {image_path}", f"Generated: {stamp}", ""]
    for r in results:
        summary_lines.append(f"=== {r.model_id} ({r.latency_sec}s) ===")
        if r.error:
            summary_lines.append(f"ERROR: {r.error}")
        else:
            summary_lines.append(r.mcp_assistant_text or r.raw_text[:500])
        summary_lines.append("")
    (output_dir / f"{stem}_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def collect_images(args: argparse.Namespace) -> List[Path]:
    if args.image:
        return [Path(args.image).resolve()]
    indir = Path(args.input_dir).resolve()
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(p for p in indir.iterdir() if p.suffix.lower() in exts)
    if args.limit:
        files = files[: args.limit]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VLM APIs and emit MCP-style metadata for classroom demos.")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--input-dir", help="Directory of images")
    parser.add_argument("--models", default="auto", help="Comma-separated: gpt4o,azure,gemini,claude or auto")
    parser.add_argument("--output-dir", default="classroom_demo/output", help="Output directory")
    parser.add_argument("--dataset-type", default="demo", help="MCP dataset_type field (livestock, pest, crop, demo)")
    parser.add_argument("--demo", action="store_true", help="Use bundled sample outputs (no API keys)")
    parser.add_argument("--env-file", default="", help="Optional .env path (not committed to git)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API calls")
    parser.add_argument("--limit", type=int, default=0, help="Max images from input-dir")
    args = parser.parse_args()

    if not args.image and not args.input_dir:
        parser.error("Provide --image or --input-dir")
    if args.image and args.input_dir:
        parser.error("Use only one of --image or --input-dir")

    env_path = Path(args.env_file).resolve() if args.env_file else SCRIPT_DIR.parent / ".env"
    load_dotenv(env_path)

    images = collect_images(args)
    if not images:
        raise SystemExit("No images found.")

    output_dir = Path(args.output_dir).resolve()
    model_keys: List[str] = [] if args.demo else resolve_models(args.models)
    if not args.demo and not model_keys:
        raise SystemExit(
            "No API keys found for requested models.\n"
            "Set env vars (see script docstring) or run with --demo."
        )

    print(f"Images: {len(images)} | Mode: {'demo' if args.demo else 'live'} | Models: {model_keys or 'bundled samples'}")

    for img_path in images:
        if not img_path.is_file():
            print(f"Skip missing: {img_path}")
            continue
        print(f"\n--- {img_path.name} ---")
        if args.demo:
            results = load_demo_results(img_path)
        else:
            b64, _ = encode_image_jpeg(img_path)
            results = []
            for key in model_keys:
                print(f"  calling {key} ...")
                results.append(run_provider(key, img_path, b64, MCP_PROMPT))
                time.sleep(max(0.0, args.delay))
        write_outputs(img_path, results, output_dir, args.dataset_type)
        for r in results:
            status = "OK" if not r.error else f"ERR: {r.error}"
            print(f"  {r.model_id}: {status} ({r.latency_sec}s)")

    print(f"\nWrote outputs under {output_dir}")


if __name__ == "__main__":
    main()

You're set for the workshop. Here's what's ready in one place:

Materials

classroom_demo/WORKSHOP_60MIN.md — 60-min instructor guide (mixed educator + CS)
classroom_demo/handouts/ — 2-page sheets per image + CS supplement + print guide
classroom_demo/workshop_images/ — 7109, 7225, 7149
classroom_demo/coyote_metadata_comparison.csv — 334×3 pre-computed model outputs
Taiga report — full SBERT run (optional deep dive)
Day-of flow (short)

Project 7109 → three model columns → human agree/partial/disagree
Reveal scores → 7225 (paraphrase surprise) → 7149 (fog outlier)
Close with: agreement ≠ truth; species solid; almost nothing auto-trusts
Print (~20 people): 4×7109, 3×7225, 3×7149 handouts + 8–10 CS supplements (see handouts/make_pdfs.sh).

Good luck with the session — it should land well with both practitioners and CS folks. If you want speaker notes on one slide or a 5-minute backup plan if Wi‑Fi fails, say the word.