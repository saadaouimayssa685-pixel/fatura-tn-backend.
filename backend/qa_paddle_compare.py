"""Compare OCR on the frozen multisource images without writing application data."""
import argparse
import hashlib
import json
import time
import os
import faulthandler
from pathlib import Path

from paddle_text import PaddleTextExtractor
from qa_archive_invoices import ROOT, save_json


def run(source, output, names):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    engine = PaddleTextExtractor()
    for item in manifest["selected"]:
        name = Path(item["entry"]).name
        if names and name not in names:
            continue
        path = Path(item["local_path"])
        target = output / (path.stem + ".ocr.json")
        if target.exists():
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Benchmark image changed")
        print(f"OCR {name}", flush=True)
        started = time.monotonic()
        try:
            text, score, metadata = engine.extract_text(str(path))
            result = {"filename": name, "sha256": item["sha256"], "ocr_text": text,
                      "ocr_score": score, "ocr_metadata": metadata, "completed": True}
        except Exception as error:
            result = {"filename": name, "completed": False, "error": f"{type(error).__name__}: {error}"}
        result["seconds"] = round(time.monotonic() - started, 3)
        save_json(target, result)
        print(json.dumps({k: v for k, v in result.items() if k not in {"ocr_text", "ocr_metadata"}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--size", choices=["tiny", "small", "medium"], default="small")
    args = parser.parse_args()
    os.environ["PADDLE_TEXT_SIZE"] = args.size
    faulthandler.dump_traceback_later(180, repeat=True)
    run(ROOT / "qa-results/groq-multisource-20260912", ROOT / f"qa-results/paddle-{args.size}-20260913", args.names)
    faulthandler.cancel_dump_traceback_later()
