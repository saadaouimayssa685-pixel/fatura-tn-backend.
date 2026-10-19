"""Isolated Surya CLI benchmark; never imports the application or writes its DB."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4


def run(manifest_path, output, names, timeout):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cli = Path(sys.executable).parent / ("surya_ocr.exe" if os.name == "nt" else "surya_ocr")
    version = importlib.metadata.version("surya-ocr")
    output.mkdir(parents=True, exist_ok=True)
    for item in manifest["selected"]:
        path = Path(item["local_path"])
        if names and path.name not in names:
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Benchmark image changed: {path.name}")
        target = output / path.stem / uuid4().hex[:12]
        target.mkdir(parents=True)
        result = {"filename": path.name, "sha256": item["sha256"],
                  "engine": "surya", "version": version, "completed": False,
                  "includes_cold_start": True}
        started = time.monotonic()
        env = dict(os.environ, SURYA_INFERENCE_BACKEND="llamacpp", PYTHONUTF8="1",
                   SURYA_INFERENCE_PARALLEL="1", LLAMA_CPP_NGL="0")
        print(f"Testing {path.name}", flush=True)
        with (target / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([str(cli), str(path), "--output_dir", str(target)],
                                       stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                result["exit_code"] = process.wait(timeout=timeout)
                pages = []
                for file in target.rglob("results.json"):
                    payload = json.loads(file.read_text(encoding="utf-8"))
                    for document_pages in payload.values():
                        pages.extend(document_pages)
                blocks = [block for page in pages for block in page.get("blocks", [])]
                result["pages"] = len(pages)
                result["blocks"] = len(blocks)
                result["error_blocks"] = sum(bool(block.get("error")) for block in blocks)
                result["completed"] = (result["exit_code"] == 0
                                       and any(block.get("html") for block in blocks)
                                       and not result["error_blocks"])
            except subprocess.TimeoutExpired:
                # Stop the benchmark's child server too, not the application's processes.
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=log, stderr=subprocess.STDOUT, check=False)
                else:
                    process.kill()
                process.wait()
                result["error"] = "timeout"
        result["seconds"] = round(time.monotonic() - started, 3)
        (target / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        if not result["completed"]:
            break


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "qa-results/groq-multisource-20260912/manifest.json")
    parser.add_argument("--output", type=Path, default=root / "qa-results/surya-20260914")
    parser.add_argument("--names", nargs="*", default=["facture_00079.png", "facture_00158.png"])
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    run(args.manifest, args.output, args.names, args.timeout)
