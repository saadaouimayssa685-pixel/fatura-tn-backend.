"""Replay downstream production steps from measured Paddle OCR; no database writes."""
import argparse
import json
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from paddle_classifier import layout_decision
from qa_archive_invoices import ROOT, pipeline, save_json


def run(names, provider, revision, pause):
    baseline = ROOT / "qa-results/groq-multisource-20260912"
    source = ROOT / "qa-results/paddle-small-20260913"
    output = ROOT / f"qa-results/paddle-pipeline-{provider}-20260913-{revision}"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "expected.json"):
        shutil.copy2(baseline / name, output / name)
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    scope = pipeline()
    previous_call = None
    for item in manifest["selected"]:
        name, stem = Path(item["entry"]).name, Path(item["entry"]).stem
        if names and name not in names:
            continue
        target = output / (stem + ".result.json")
        ocr_file = source / (stem + ".ocr.json")
        if not ocr_file.exists() or target.exists():
            continue
        result = json.loads(ocr_file.read_text(encoding="utf-8"))
        if not result.get("completed"):
            continue
        original = json.loads((baseline / (stem + ".result.json")).read_text(encoding="utf-8"))
        if result["sha256"] != original["sha256"]:
            raise ValueError("Image hash mismatch")
        text = result["ocr_text"]
        classified = scope["classify_invoice_text"](text)
        boxes = original["classification"][2].get("layout_boxes", [])
        result["classification"] = layout_decision(text, boxes, classified)
        result["layout_boxes_reused_from_same_image"] = True
        result["ocr_reused_for_debugging"] = True
        result["database_writes"] = False
        result["timings"] = {"ocr": result["seconds"]}
        if not result["classification"][0]:
            result.update(fields={}, model_used="not_run_non_invoice")
            save_json(target, result)
            continue
        started = time.monotonic()
        evidence = scope["get_invoice_field_evidence"](text)
        result["timings"]["rag"] = round(time.monotonic() - started, 3)
        rules = scope["build_rule_candidate_pack"](text)
        result.update(evidence=evidence, rule_candidates=rules, retrieval_backend=scope["extraction_retriever"].backend)
        started = time.monotonic()
        if previous_call is not None and pause:
            time.sleep(max(0, pause - (started - previous_call)))
        previous_call = time.monotonic()
        data, model, raw = scope["analyze_invoice_fields"](text, provider, evidence, rules)
        normalized = scope["normalize_invoice_data"](data, text, rules)
        fields, proofs = scope["enforce_evidence_guard"](normalized, text, evidence, rules)
        result["timings"]["llm"] = round(time.monotonic() - started, 3)
        result.update(model_used=model, analysis_diagnostics=raw, llm=data.model_dump() if data else None,
                      fields=fields, proofs=proofs, issues=scope["validate_normalized_invoice_fields"](fields, proofs))
        save_json(target, result)
        print(json.dumps({"filename": name, "status": raw.get("llm", {}).get("status"),
                          "supplier": fields.get("vendor_name"), "items": fields.get("items"),
                          "subtotal": fields.get("subtotal"), "total": fields.get("total_amount")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--provider", choices=["groq", "fallback"], default="groq")
    parser.add_argument("--revision", default="verified")
    parser.add_argument("--pause", type=float, default=65)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    run(args.names, args.provider, args.revision, args.pause)
