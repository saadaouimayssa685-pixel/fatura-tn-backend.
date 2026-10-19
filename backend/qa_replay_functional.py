"""Replay extraction/validation on an audited frozen OCR and retrieval result."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pause-seconds", type=float, default=0,
                        help="QA-only spacing of live API calls to respect account quotas.")
    parser.add_argument("--reuse-generations-from", type=Path,
                        help="QA-only replay of recorded raw Groq batches outside --live-field.")
    parser.add_argument("--live-field", action="append", default=[])
    args = parser.parse_args()
    load_dotenv(Path(__file__).with_name(".env"))
    from qa_archive_invoices import pipeline
    scope = pipeline()
    previous = json.loads(args.input.read_text(encoding="utf-8"))
    text, evidence = previous["ocr_text"], previous["evidence"]
    analyzer = scope["groq_analyzer"]
    if not analyzer.is_available():
        raise RuntimeError("Groq is not configured")
    trace = []
    last_call = None
    recorded = json.loads(args.reuse_generations_from.read_text(encoding="utf-8")) if args.reuse_generations_from else None
    if recorded and (recorded["model_used"] != f"groq:{analyzer.model_name}"
                     or recorded["input_sha256"] != hashlib.sha256(args.input.read_bytes()).hexdigest()):
        raise ValueError("Recorded generation model or input differs")
    build, generate, ground = analyzer._build_prompt, analyzer._generate, analyzer._ground_fields

    def traced_build(*a, **kw):
        prompt = build(*a, **kw)
        context = kw["context"]
        trace.append({"fields": kw["fields"], "prompt": prompt,
                      "allowed_lines": sorted(context["allowed_lines"]), "sources": context["sources"]})
        return prompt

    def traced_generate(*a, **kw):
        nonlocal last_call
        reused = recorded is not None and not set(args.live_field).intersection(trace[-1]["fields"])
        if reused:
            prior = next(t for t in recorded["trace"] if t["fields"] == trace[-1]["fields"])
            assert prior["sources"] == trace[-1]["sources"] and prior["allowed_lines"] == trace[-1]["allowed_lines"]
            attempt = len(trace[-1].get("generations", []))
            value = copy.deepcopy(prior["generations"][attempt]["payload"])
            trace[-1]["generation"] = value
            trace[-1].setdefault("generations", []).append({"prompt": a[0], "payload": value, "reused": True})
            return value
        if last_call is not None and args.pause_seconds:
            time.sleep(max(0, args.pause_seconds - (time.monotonic() - last_call)))
        last_call = time.monotonic()
        print("Live Groq batch: " + ", ".join(trace[-1]["fields"]), flush=True)
        value = generate(*a, **kw)
        trace[-1]["generation"] = copy.deepcopy(value)
        trace[-1].setdefault("generations", []).append({"prompt": a[0], "payload": copy.deepcopy(value), "reused": False})
        return value

    def traced_ground(*a, **kw):
        trace[-1]["parsed_before_grounding"] = copy.deepcopy(a[0])
        attempt = {"before": copy.deepcopy(a[0])}
        value = ground(*a, **kw)
        trace[-1]["after_grounding"] = copy.deepcopy(value)
        attempt["after"] = copy.deepcopy(value)
        trace[-1].setdefault("grounding_attempts", []).append(attempt)
        return value

    analyzer._build_prompt, analyzer._generate, analyzer._ground_fields = traced_build, traced_generate, traced_ground
    rules = scope["build_rule_candidate_pack"](text)
    invoice, model, diagnostics = scope["analyze_invoice_fields"](text, "groq", evidence, rules, previous.get("image"))
    if invoice is None:
        raise RuntimeError(f"Extraction returned no invoice: {diagnostics}")
    normalized = scope["normalize_invoice_data"](invoice, text, rules)
    fields, proofs = scope["enforce_evidence_guard"](normalized, text, evidence, rules)
    issues = scope["validate_normalized_invoice_fields"](fields, proofs)
    delimiter = "Passages OCR retrouves (lignes globales du document):\n"
    context_identical = all(
        before["prompt"].split(delimiter)[1] == after["prompt"].split(delimiter)[1]
        and before["allowed_lines"] == after["allowed_lines"] and before["sources"] == after["sources"]
        for before, after in zip(previous["trace"], trace)) and len(trace) == len(previous["trace"])
    result = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "ocr_reused": True, "retrieval_reused": True, "database_writes": False,
        "reused_raw_generations_from": str(args.reuse_generations_from) if recorded else None,
        "context_identical": context_identical, "model_used": model, "analysis_diagnostics": diagnostics,
        "trace": trace, "llm": invoice.model_dump(), "before_final_guard": normalized,
        "fields": fields, "proofs": proofs, "issues": issues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "context_identical": context_identical,
                      "fields": fields, "issues": issues, "diagnostics": diagnostics}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
