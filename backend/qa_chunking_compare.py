"""Compare only invoice chunking on frozen OCR/layout, without LLM or database."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from extraction_chunks import image_layout_frame
from extraction_retrieval import ExtractionRetriever, selected_context, source_passages


ROOT = Path(__file__).resolve().parent
FIELDS = ("vendor_name", "vendor_tax_id", "subtotal", "tax_amount", "total_amount", "payment_method", "items")


def run(output, names):
    os.environ["EXTRACTION_SEMANTIC_RAG"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    baseline = ROOT / "qa-results/groq-multisource-20260912"
    inputs = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))["selected"]
    retriever = ExtractionRetriever()
    started = time.monotonic()
    with retriever.lock:
        retriever._load()
    if retriever.model is None:
        raise RuntimeError("Cached MiniLM unavailable: refusing a different benchmark configuration")
    retriever.model.encode(["facture"], normalize_embeddings=True)
    result = {"database_writes": False, "ocr_rerun": False, "llm_calls": False,
              "backend": retriever.backend, "model": "sentence-transformers/all-MiniLM-L6-v2",
              "rrf_k": 60, "bm25_k1": 1.2, "bm25_b": .75,
              "model_warmup_seconds": round(time.monotonic() - started, 3), "documents": []}
    report = ["# Comparaison Phase 2 : legacy / layout", "",
              "OCR et layout figes, memes images SHA256, meme BM25 + MiniLM + RRF.",
              "Aucun appel OCR/LLM, aucune ecriture dans la base de factures.",
              "Les candidats header/table forces par le code existant restent forces : score n/a si absent.",
              "Top 3 ci-dessous = trois premiers passages retournes par la politique existante, pas un nouveau ranking.",
              "Ces exemples ne constituent pas une mesure Recall@k annotee.", ""]
    for item in inputs:
        stem = Path(item["entry"]).stem
        if stem not in names:
            continue
        frozen = json.loads((ROOT / f"qa-results/paddle-small-20260913/{stem}.ocr.json").read_text(encoding="utf-8"))
        layout_result = json.loads((baseline / f"{stem}.result.json").read_text(encoding="utf-8"))
        path = Path(item["local_path"])
        if len({frozen["sha256"], layout_result["sha256"], item["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()}) != 1:
            raise ValueError(f"Image hash mismatch: {stem}")
        text = frozen["ocr_text"]
        layout = {**layout_result["classification"][2], **image_layout_frame(path)}
        doc = {"filename": path.name, "sha256": item["sha256"],
               "ocr_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "ocr_file": str(ROOT / f"qa-results/paddle-small-20260913/{stem}.ocr.json"),
               "layout_file": str(baseline / f"{stem}.result.json"),
               "layout_boxes_count": len(layout.get("layout_boxes", [])), "modes": {}}
        report.extend([f"## {path.name}", "", f"Layout existant : {doc['layout_boxes_count']} boites.", ""])
        for mode in ("legacy", "layout"):
            started = time.monotonic()
            evidence = retriever.retrieve(text, ocr_metadata=frozen["ocr_metadata"], layout_metadata=layout,
                                          document_id=item["sha256"], chunking_mode=mode)
            elapsed = round(time.monotonic() - started, 3)
            passages = getattr(evidence, "passages", source_passages(text))
            context = selected_context(text, evidence)
            value = {"seconds": elapsed, "chunks_count": len(passages),
                     "chunks_with_bbox": sum(p.get("bbox") is not None for p in passages),
                     "chunks_with_layout": sum(bool(p.get("layout_labels")) for p in passages),
                     "selected_lines_all_fields": len(context["allowed_lines"]),
                     "top3": {field: evidence[field][:3] for field in FIELDS}, "chunks": passages}
            doc["modes"][mode] = value
            report.extend([f"### {mode}", "", f"{len(passages)} chunks ; {elapsed} s (modele precharge) ; "
                           f"{value['chunks_with_bbox']} avec bbox ; {value['chunks_with_layout']} relies au layout.", ""])
            for field in FIELDS:
                report.extend([f"#### {field}", ""])
                for rank, passage in enumerate(evidence[field][:3], 1):
                    report.extend([f"{rank}. {passage['source_id']} | page {passage['page_number']} | "
                        f"section {passage.get('section', passage.get('kind'))} | "
                        f"BM25 {passage.get('lexical_score', 'n/a')} / semantique {passage.get('semantic_score', 'n/a')} / RRF {passage.get('score', 'n/a')}",
                        "", "```text", passage["text"], "```", ""])
                if not evidence[field]:
                    report.extend(["Aucun passage.", ""])
            print(json.dumps({"filename": path.name, "mode": mode, "seconds": elapsed,
                              "chunks": len(passages), "bbox": value["chunks_with_bbox"],
                              "layout": value["chunks_with_layout"]}), flush=True)
        result["documents"].append(doc)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "TOP3.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "qa-results/chunking-phase2")
    parser.add_argument("--names", nargs="+", default=["facture_00008", "facture_00158", "facture_00079"])
    args = parser.parse_args()
    run(args.output, args.names)
