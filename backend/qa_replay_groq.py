"""Replay extraction on frozen OCR/retrieval without changing baseline reports."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from qa_archive_invoices import ROOT, pipeline, save_json


def replay(source, output):
    output.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "expected.json"):
        shutil.copy2(source / name, output / name)
    scope = pipeline()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["selected"]:
        name = Path(item["entry"]).stem + ".result.json"
        baseline = source / name
        target = output / name
        if target.exists() or not baseline.exists():
            continue
        result = json.loads(baseline.read_text(encoding="utf-8"))
        if not result.get("completed"):
            continue
        if hashlib.sha256(Path(item["local_path"]).read_bytes()).hexdigest() != result["sha256"]:
            raise ValueError("Image changed since baseline")
        result["replay"] = {"baseline": str(baseline), "ocr_and_retrieval_reused": True,
                            "reason": "Corrected items-only Groq JSON schema"}
        if not result["classification"][0]:
            save_json(target, result)
            continue
        text, evidence, rules = result["ocr_text"], result["evidence"], result["rule_candidates"]
        start = time.monotonic()
        data, model, raw = scope["analyze_invoice_fields"](text, "groq", evidence, rules)
        normalized = scope["normalize_invoice_data"](data, text, rules)
        fields, proofs = scope["enforce_evidence_guard"](
            normalized, text, evidence, rules, data.additional_data.get("cloud_evidence") if data else None)
        result["replay"]["llm_seconds"] = round(time.monotonic() - start, 3)
        result.update(model_used=model, analysis_diagnostics=raw, llm=data.model_dump() if data else None,
                      fields=fields, proofs=proofs, issues=scope["validate_normalized_invoice_fields"](fields, proofs))
        save_json(target, result)
        print(json.dumps({"filename": result["filename"], "status": raw.get("llm", {}).get("status"),
                          "items": len(fields.get("items") or []), "seconds": result["replay"]["llm_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "qa-results/groq-multisource-20260912")
    parser.add_argument("--output", type=Path, default=ROOT / "qa-results/groq-multisource-20260912-replay")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    replay(args.source, args.output)
