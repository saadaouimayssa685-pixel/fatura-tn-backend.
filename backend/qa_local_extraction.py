"""Explicit local-only benchmark on a supplied invoice, without database writes."""
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("invoice")
    parser.add_argument("--models", nargs="+", default=["qwen2.5:1.5b", "qwen3:4b-instruct-2507-q4_K_M"])
    args = parser.parse_args()
    load_dotenv(Path(__file__).with_name(".env"))
    os.environ["OLLAMA_TIMEOUT_SECONDS"] = "180"
    os.environ["EXTRACTION_SEMANTIC_RAG"] = "true"
    os.environ["OLLAMA_CONTEXT_SIZE"] = "4096"
    os.environ["OLLAMA_MAX_OUTPUT_TOKENS"] = "1500"
    os.environ["OLLAMA_CPU_THREADS"] = "6"
    from ocr_extractor import OptimizedOCRExtractor
    from extraction_retrieval import ExtractionRetriever
    from ollama_analyzer import OllamaInvoiceAnalyzer
    text, confidence, _ = OptimizedOCRExtractor().extract_text(args.invoice)
    retriever = ExtractionRetriever()
    evidence = retriever.retrieve(text)
    print(json.dumps({"ocr_characters": len(text), "retrieval": retriever.backend,
                      "fields_with_sources": [key for key, value in evidence.items() if value]}, ensure_ascii=False), flush=True)
    for model in args.models:
        analyzer = OllamaInvoiceAnalyzer()
        analyzer.model_name = model
        start = time.monotonic()
        result = analyzer.analyze_invoice_text(text, field_evidence=evidence)
        print(json.dumps({"model": model, "seconds": round(time.monotonic() - start, 2),
                          "success": result is not None, "invoice": result.model_dump() if result else None},
                         ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
