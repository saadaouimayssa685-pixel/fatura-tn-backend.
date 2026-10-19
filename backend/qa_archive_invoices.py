"""Read-only real-document benchmark. Never imports application database startup."""
import argparse
import ast
import hashlib
import json
import os
import re
import time
import zipfile
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def evaluate(output):
    expected = json.loads((output / "expected.json").read_text(encoding="utf-8"))
    comparisons = {}
    def comparable(value):
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return round(float(value), 3)
        value = unicodedata.normalize("NFKD", str(value)).casefold()
        return "".join(c for c in value if c.isalnum())
    for filename, truth in expected.items():
        report = output / (Path(filename).stem + ".result.json")
        if not isinstance(truth, dict) or not report.exists():
            continue
        result = json.loads(report.read_text(encoding="utf-8"))
        if not result.get("completed"):
            comparisons[filename] = {"completed": False, "model_used": result.get("model_used"),
                                     "checks": [], "passed": 0, "total": 0}
            continue
        fields = result.get("fields") or {}
        checks = []
        for key, target in truth.items():
            if key in {"document_type", "requires_type_review"}:
                continue
            value = fields.get(key)
            if key == "is_invoice":
                value = (result.get("classification") or [None])[0]
            if key in {"date", "due_date"} and value:
                for pattern in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%b-%Y"):
                    try:
                        value = datetime.strptime(value, pattern).date().isoformat()
                        break
                    except (ValueError, TypeError):
                        pass
            if key == "items":
                actual_items = fields.get("items") or []
                checks.append({"field": "item_count", "expected": len(target), "actual": len(actual_items),
                               "passed": len(target) == len(actual_items)})
                for index, row in enumerate(target):
                    for cell, cell_target in row.items():
                        actual = actual_items[index].get(cell) if index < len(actual_items) else None
                        checks.append({"field": f"items[{index}].{cell}", "expected": cell_target, "actual": actual,
                                       "passed": comparable(actual) == comparable(cell_target)})
                continue
            if key == "payment_method" and value and "especes" in comparable(value):
                value = "especes"
            checks.append({"field": key, "expected": target, "actual": value,
                           "passed": comparable(value) == comparable(target)})
        comparisons[filename] = {"completed": result.get("completed"), "model_used": result.get("model_used"),
                                 "checks": checks, "passed": sum(c["passed"] for c in checks), "total": len(checks)}
    save_json(output / "evaluation.json", comparisons)
    print(json.dumps({name: {key: value for key, value in row.items() if key != "checks"}
                      for name, row in comparisons.items()}, ensure_ascii=False), flush=True)


def prepare(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    images = []
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            if Path(entry.filename).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            data = source.read(entry)
            from io import BytesIO
            try:
                with Image.open(BytesIO(data)) as image:
                    image.load()
                    size = image.size
                images.append({"entry": entry.filename, "bytes": len(data), "size": size,
                               "sha256": hashlib.sha256(data).hexdigest()})
            except Exception as error:
                images.append({"entry": entry.filename, "error": str(error)})
        # Use held-out validation images when available; a reproducible spread,
        # not a selection based on successful extraction results.
        pool = sorted([item for item in images if any(f"/{split}/" in item["entry"] for split in ("val", "test")) and "error" not in item],
                      key=lambda item: item["entry"])
        if not pool:
            pool = [item for item in images if "error" not in item]
        selected = [pool[round(i * (len(pool) - 1) / 5)] for i in range(min(6, len(pool)))]
        sheet = Image.new("RGB", (1200, 1100), "#e9eef5")
        draw = ImageDraw.Draw(sheet)
        for index, item in enumerate(selected):
            filename = Path(item["entry"]).name
            dest = output / filename
            dest.write_bytes(source.read(item["entry"]))
            item["local_path"] = str(dest)
            with Image.open(dest) as image:
                preview = ImageOps.exif_transpose(image).convert("RGB")
                preview.thumbnail((380, 500))
                x, y = (index % 3) * 400, (index // 3) * 550
                sheet.paste(preview, (x + (400 - preview.width) // 2, y + 30))
                draw.text((x + 10, y + 8), filename, fill="black")
        sheet.save(output / "contact-sheet.jpg", quality=90)
    train_hashes = {item["sha256"] for item in images if "/train/" in item["entry"] and "sha256" in item}
    overlap = [item["entry"] for item in images if any(f"/{split}/" in item["entry"] for split in ("val", "test"))
               and item.get("sha256") in train_hashes]
    manifest = {"archive": str(archive), "image_count": len(images), "train_evaluation_overlap": overlap,
                "unique_images": len({item.get("sha256") for item in images if "sha256" in item}),
                "images": images, "selected": selected}
    save_json(output / "manifest.json", manifest)
    print(json.dumps({"image_count": len(images), "unique_images": manifest["unique_images"],
                      "selected": [item["entry"] for item in selected]}, ensure_ascii=False), flush=True)


def pipeline():
    # Compile production pure functions without importing main, whose startup
    # initializes user databases. No copied extraction implementation.
    from models import InvoiceData
    from invoice_rules import extract_invoice_table_text, filter_invoice_line_items
    from ocr_extractor import OptimizedOCRExtractor
    from pdf_extractor import PDFExtractor
    from glm_ocr import GLMOCRExtractor
    from ollama_analyzer import OllamaInvoiceAnalyzer
    from groq_analyzer import GroqInvoiceAnalyzer
    from extraction_retrieval import ExtractionRetriever
    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8-sig"))
    excluded = {"project_path", "validate_file", "detect_signature_objects", "detect_document_objects",
                "normalize_email", "hash_password", "create_password_hash", "auth_connection", "public_user",
                "init_auth_database", "create_session", "get_user_from_token", "user_from_authorization"}
    constants = {"EVIDENCE_GUARDED_FIELDS", "FIELD_RETRIEVAL_QUERIES", "AGENT_POLICY"}
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and node.name not in excluded and not node.decorator_list
             or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)]
    scope = dict(globals(), InvoiceData=InvoiceData, extract_invoice_table_text=extract_invoice_table_text,
                 filter_invoice_line_items=filter_invoice_line_items)
    scope["ocr_extractor"] = OptimizedOCRExtractor()
    scope["pdf_extractor"] = PDFExtractor(scope["ocr_extractor"])
    scope["glm_ocr"] = GLMOCRExtractor()
    scope["ollama_analyzer"] = OllamaInvoiceAnalyzer()
    scope["groq_analyzer"] = GroqInvoiceAnalyzer()
    scope["extraction_retriever"] = ExtractionRetriever()
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "main.py", "exec"), scope)
    return scope


def benchmark(output, names, paddle, reuse_ocr=False, provider="local"):
    scope = pipeline()
    from paddle_classifier import PaddleDocumentClassifier
    classifier = PaddleDocumentClassifier() if paddle else None
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    selected = [item for item in manifest["selected"] if not names or Path(item["entry"]).name in names]
    for item in selected:
        result = {"filename": Path(item["entry"]).name, "sha256": item["sha256"], "database_writes": False,
                  "timings": {}}
        report = output / (Path(item["entry"]).stem + ".result.json")
        def stage(name, call):
            print(f"{result['filename']}: {name}", flush=True)
            result["current_stage"] = name
            save_json(report, result)
            start = time.monotonic()
            value = call()
            result["timings"][name] = round(time.monotonic() - start, 3)
            return value
        try:
            cached = json.loads(report.read_text(encoding="utf-8")) if reuse_ocr and report.exists() else {}
            if cached.get("sha256") == item["sha256"] and cached.get("ocr_text"):
                text, confidence, metadata = cached["ocr_text"], cached["ocr_score"], cached["ocr_metadata"]
                result["ocr_reused_for_debugging"] = True
                result["original_ocr_seconds"] = cached["timings"].get("ocr", cached.get("original_ocr_seconds"))
                text = stage("stamp_retry", lambda: scope["enrich_payment_ocr"](item["local_path"], text, metadata))
            else:
                text, confidence, metadata = stage("ocr", lambda: scope["extract_document_text"](item["local_path"]))
            result.update(ocr_text=text, ocr_score=confidence, ocr_metadata=metadata)
            save_json(report, result)
            classification = scope["classify_invoice_text"](text)
            if classifier:
                classification = stage("paddle", lambda: classifier.classify(item["local_path"], text, classification))
            result["classification"] = classification
            result["paddle_requested"] = paddle
            if not classification[0]:
                result.update(completed=True, model_used="not_run_non_invoice", fields={})
                save_json(report, result)
                continue
            evidence = stage("rag", lambda: scope["get_invoice_field_evidence"](text))
            result["retrieval_backend"] = scope["extraction_retriever"].backend
            rules = stage("rules", lambda: scope["build_rule_candidate_pack"](text))
            result.update(evidence=evidence, rule_candidates=rules)
            save_json(report, result)
            data, model, raw = stage("llm", lambda: scope["analyze_invoice_fields"](text, provider, evidence, rules))
            normalized = scope["normalize_invoice_data"](data, text, rules)
            fields, proofs = scope["enforce_evidence_guard"](normalized, text, evidence, rules,
                data.additional_data.get("cloud_evidence") if data else None)
            result.update(model_used=model, analysis_diagnostics=raw, llm=data.model_dump() if data else None, fields=fields, proofs=proofs,
                          issues=scope["validate_normalized_invoice_fields"](fields, proofs), completed=True)
        except Exception as error:
            result.update(completed=False, error=f"{type(error).__name__}: {error}")
        save_json(report, result)
        print(json.dumps({"filename": result["filename"], "completed": result.get("completed"),
                          "model": result.get("model_used"), "timings": result["timings"],
                          "error": result.get("error")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "qa-results" / "dataset-20260911")
    parser.add_argument("--run", nargs="*")
    parser.add_argument("--paddle", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--reuse-ocr", action="store_true", help="Reuse explicitly labelled OCR results only when SHA256 matches")
    parser.add_argument("--provider", choices=["local", "groq"], default="local")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if args.evaluate:
        evaluate(args.output)
    elif args.run is None:
        prepare(args.archive, args.output)
    else:
        benchmark(args.output, args.run, args.paddle, args.reuse_ocr, args.provider)
