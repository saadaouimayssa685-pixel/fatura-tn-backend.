"""Exercise real retrieval and Ollama on cached invoice OCR without opening the app database."""
import argparse
import faulthandler
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from extraction_retrieval import ExtractionRetriever, selected_context
from ollama_analyzer import OllamaInvoiceAnalyzer
from groq_analyzer import GroqInvoiceAnalyzer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ocr_result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--provider", choices=["local", "groq"], default="local")
    parser.add_argument("--fields", nargs="+", default=["subtotal", "tax_amount", "stamp_duty", "total_amount"])
    args = parser.parse_args()
    load_dotenv(Path(__file__).with_name(".env"))
    faulthandler.dump_traceback_later(180, repeat=True)
    cached = json.loads(args.ocr_result.read_text(encoding="utf-8"))
    text = cached["ocr_text"]
    report = {"filename": cached["filename"], "ocr_reused": True, "database_writes": False,
              "fields_requested": args.fields, "status": "retrieving"}

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    print("Retrieving real OCR chunks and embeddings...", flush=True)
    started = time.monotonic()
    retriever = ExtractionRetriever()
    evidence = retriever.retrieve(text)
    evidence = {field: evidence.get(field, []) for field in args.fields}
    report.update(retrieval_seconds=round(time.monotonic() - started, 3), retrieval_backend=retriever.backend,
                  retrieved_evidence=evidence)
    context = selected_context(text, evidence)
    report.update(selected_lines=len(context["allowed_lines"]), original_lines=len(text.splitlines()),
                  selected_sources=context["sources"], dropped_passages=context["dropped_passages"], status="generating")
    save()
    print(json.dumps({key: report[key] for key in ("status", "retrieval_seconds", "retrieval_backend", "selected_lines", "original_lines")}), flush=True)
    analyzer = GroqInvoiceAnalyzer() if args.provider == "groq" else OllamaInvoiceAnalyzer()
    if args.model:
        analyzer.model_name = args.model
    diagnostics = {}
    result = analyzer.analyze_invoice_text(text, evidence, diagnostics=diagnostics)
    report.update(status=diagnostics["status"], diagnostics=diagnostics,
                  result=result.model_dump(mode="json") if result else None)
    save()
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps({key: report[key] for key in ("status", "retrieval_backend", "selected_lines", "original_lines")}), flush=True)


if __name__ == "__main__":
    main()
